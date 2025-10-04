// RBAC Middleware
// Phase 5 Week 40 - API endpoint protection with RBAC

use axum::{
    extract::{Request, State},
    http::{StatusCode, HeaderMap},
    middleware::Next,
    response::{Response, IntoResponse},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;
use chrono::Utc;

use super::database::{RbacRepository, AuditLogEntry, AuditResult};
use super::{Resource, Action};

/// Authentication context extracted from request
#[derive(Debug, Clone)]
pub struct AuthContext {
    pub user_id: Uuid,
    pub organization_id: Uuid,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
}

/// Required permission for an endpoint
#[derive(Debug, Clone)]
pub struct RequiredPermission {
    pub resource: String,
    pub action: String,
}

impl RequiredPermission {
    pub fn new(resource: impl Into<String>, action: impl Into<String>) -> Self {
        Self {
            resource: resource.into(),
            action: action.into(),
        }
    }
}

/// Error response for authorization failures
#[derive(Debug, Serialize)]
pub struct AuthorizationError {
    pub error: String,
    pub message: String,
    pub required_permission: Option<String>,
}

impl IntoResponse for AuthorizationError {
    fn into_response(self) -> Response {
        let status = if self.error == "forbidden" {
            StatusCode::FORBIDDEN
        } else {
            StatusCode::UNAUTHORIZED
        };

        (status, Json(self)).into_response()
    }
}

/// RBAC middleware state
pub struct RbacMiddleware {
    repository: Arc<RbacRepository>,
}

impl RbacMiddleware {
    pub fn new(repository: Arc<RbacRepository>) -> Self {
        Self { repository }
    }

    /// Extract authentication context from request headers
    fn extract_auth_context(headers: &HeaderMap) -> Result<AuthContext, AuthorizationError> {
        // Extract user ID from Authorization header (JWT, session, etc.)
        let user_id = headers
            .get("X-User-ID")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| Uuid::parse_str(s).ok())
            .ok_or_else(|| AuthorizationError {
                error: "unauthorized".to_string(),
                message: "Missing or invalid user authentication".to_string(),
                required_permission: None,
            })?;

        // Extract organization ID from header
        let organization_id = headers
            .get("X-Organization-ID")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| Uuid::parse_str(s).ok())
            .ok_or_else(|| AuthorizationError {
                error: "unauthorized".to_string(),
                message: "Missing or invalid organization context".to_string(),
                required_permission: None,
            })?;

        // Extract optional metadata
        let ip_address = headers
            .get("X-Forwarded-For")
            .or_else(|| headers.get("X-Real-IP"))
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string());

        let user_agent = headers
            .get("User-Agent")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string());

        Ok(AuthContext {
            user_id,
            organization_id,
            ip_address,
            user_agent,
        })
    }

    /// Extract required permission from request path and method
    fn extract_permission(request: &Request) -> RequiredPermission {
        let path = request.uri().path();
        let method = request.method();

        // Map REST endpoints to permissions
        // This is a simplified example - production would use route metadata
        match (method.as_str(), path) {
            // Projects
            ("POST", path) if path.starts_with("/api/projects") => {
                RequiredPermission::new("project", "create")
            }
            ("GET", path) if path.starts_with("/api/projects") => {
                RequiredPermission::new("project", "read")
            }
            ("PUT", path) if path.starts_with("/api/projects") => {
                RequiredPermission::new("project", "update")
            }
            ("DELETE", path) if path.starts_with("/api/projects") => {
                RequiredPermission::new("project", "delete")
            }

            // Translations
            ("POST", path) if path.contains("/translate") => {
                RequiredPermission::new("translation", "execute")
            }
            ("GET", path) if path.starts_with("/api/translations") => {
                RequiredPermission::new("translation", "read")
            }

            // Assessments
            ("POST", path) if path.starts_with("/api/assessments") => {
                RequiredPermission::new("assessment", "create")
            }
            ("GET", path) if path.starts_with("/api/assessments") => {
                RequiredPermission::new("assessment", "read")
            }

            // Users
            ("POST", path) if path.starts_with("/api/users") => {
                RequiredPermission::new("user", "create")
            }
            ("GET", path) if path.starts_with("/api/users") => {
                RequiredPermission::new("user", "read")
            }
            ("PUT", path) if path.starts_with("/api/users") => {
                RequiredPermission::new("user", "update")
            }

            // Audit logs
            ("GET", path) if path.starts_with("/api/audit") => {
                RequiredPermission::new("audit", "read")
            }

            // Metrics
            ("GET", path) if path.starts_with("/api/metrics") => {
                RequiredPermission::new("metrics", "read")
            }

            // System
            ("GET", path) if path.starts_with("/api/system") => {
                RequiredPermission::new("system", "read")
            }

            // Default: deny
            _ => RequiredPermission::new("unknown", "access"),
        }
    }

    /// Check if user has required permission
    pub async fn check_permission(
        State(middleware): State<Arc<RbacMiddleware>>,
        mut request: Request,
        next: Next,
    ) -> Result<Response, AuthorizationError> {
        // 1. Extract authentication context
        let auth_context = Self::extract_auth_context(request.headers())?;

        // 2. Verify user is member of organization
        let is_member = middleware
            .repository
            .is_organization_member(auth_context.user_id, auth_context.organization_id)
            .await
            .map_err(|e| AuthorizationError {
                error: "internal_error".to_string(),
                message: format!("Failed to verify organization membership: {}", e),
                required_permission: None,
            })?;

        if !is_member {
            // Record denial in audit log
            let _ = middleware
                .repository
                .record_audit(AuditLogEntry {
                    id: Uuid::new_v4(),
                    user_id: Some(auth_context.user_id),
                    organization_id: Some(auth_context.organization_id),
                    action: "organization.access".to_string(),
                    resource_type: None,
                    resource_id: None,
                    result: AuditResult::Denied,
                    details: serde_json::json!({
                        "reason": "not_organization_member"
                    }),
                    ip_address: auth_context.ip_address.clone(),
                    user_agent: auth_context.user_agent.clone(),
                    created_at: Utc::now(),
                })
                .await;

            return Err(AuthorizationError {
                error: "forbidden".to_string(),
                message: "User is not a member of this organization".to_string(),
                required_permission: None,
            });
        }

        // 3. Extract required permission from request
        let permission = Self::extract_permission(&request);

        // 4. Check if user has permission
        let has_permission = middleware
            .repository
            .check_permission(
                auth_context.user_id,
                auth_context.organization_id,
                &permission.resource,
                &permission.action,
            )
            .await
            .map_err(|e| AuthorizationError {
                error: "internal_error".to_string(),
                message: format!("Failed to check permission: {}", e),
                required_permission: None,
            })?;

        // 5. Record audit log entry
        let result = if has_permission {
            AuditResult::Allowed
        } else {
            AuditResult::Denied
        };

        let _ = middleware
            .repository
            .record_audit(AuditLogEntry {
                id: Uuid::new_v4(),
                user_id: Some(auth_context.user_id),
                organization_id: Some(auth_context.organization_id),
                action: format!("permission.check.{}:{}", permission.resource, permission.action),
                resource_type: Some(permission.resource.clone()),
                resource_id: None,
                result: result.clone(),
                details: serde_json::json!({
                    "path": request.uri().path(),
                    "method": request.method().as_str(),
                }),
                ip_address: auth_context.ip_address.clone(),
                user_agent: auth_context.user_agent.clone(),
                created_at: Utc::now(),
            })
            .await;

        // 6. Deny if permission not granted
        if !has_permission {
            return Err(AuthorizationError {
                error: "forbidden".to_string(),
                message: "Insufficient permissions".to_string(),
                required_permission: Some(format!("{}:{}", permission.resource, permission.action)),
            });
        }

        // 7. Add auth context to request extensions for downstream handlers
        request.extensions_mut().insert(auth_context);

        // 8. Continue to handler
        Ok(next.run(request).await)
    }
}

/// Helper macro to define required permissions for route handlers
#[macro_export]
macro_rules! require_permission {
    ($resource:expr, $action:expr) => {
        RequiredPermission::new($resource, $action)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderMap, Method};

    #[test]
    fn test_extract_permission_projects() {
        let mut request = Request::new(());
        *request.method_mut() = Method::POST;
        *request.uri_mut() = "/api/projects".parse().unwrap();

        let perm = RbacMiddleware::extract_permission(&request);
        assert_eq!(perm.resource, "project");
        assert_eq!(perm.action, "create");
    }

    #[test]
    fn test_extract_permission_translations() {
        let mut request = Request::new(());
        *request.method_mut() = Method::POST;
        *request.uri_mut() = "/api/projects/123/translate".parse().unwrap();

        let perm = RbacMiddleware::extract_permission(&request);
        assert_eq!(perm.resource, "translation");
        assert_eq!(perm.action, "execute");
    }

    #[test]
    fn test_extract_auth_context_missing_user() {
        let headers = HeaderMap::new();
        let result = RbacMiddleware::extract_auth_context(&headers);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().error, "unauthorized");
    }

    #[test]
    fn test_extract_auth_context_success() {
        let mut headers = HeaderMap::new();
        let user_id = Uuid::new_v4();
        let org_id = Uuid::new_v4();

        headers.insert("X-User-ID", user_id.to_string().parse().unwrap());
        headers.insert("X-Organization-ID", org_id.to_string().parse().unwrap());
        headers.insert("User-Agent", "test-agent".parse().unwrap());

        let result = RbacMiddleware::extract_auth_context(&headers);
        assert!(result.is_ok());

        let context = result.unwrap();
        assert_eq!(context.user_id, user_id);
        assert_eq!(context.organization_id, org_id);
        assert_eq!(context.user_agent, Some("test-agent".to_string()));
    }

    #[test]
    fn test_required_permission_new() {
        let perm = RequiredPermission::new("project", "create");
        assert_eq!(perm.resource, "project");
        assert_eq!(perm.action, "create");
    }
}
