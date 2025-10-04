// RBAC Database Layer
// Phase 5 Week 39 - Database integration for RBAC

use sqlx::{PgPool, Row};
use uuid::Uuid;
use std::collections::HashSet;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{Role, Resource, Action};

/// Organization (tenant) entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Organization {
    pub id: Uuid,
    pub name: String,
    pub slug: String,
    pub settings: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// User entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: Uuid,
    pub email: String,
    pub name: Option<String>,
    pub email_verified: bool,
    pub is_active: bool,
    pub last_login: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

/// Role assignment
#[derive(Debug, Clone)]
pub struct UserRoleAssignment {
    pub id: Uuid,
    pub user_id: Uuid,
    pub role_id: Uuid,
    pub organization_id: Uuid,
    pub granted_by: Option<Uuid>,
    pub expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

/// Audit log entry
#[derive(Debug, Clone, Serialize)]
pub struct AuditLogEntry {
    pub id: Uuid,
    pub user_id: Option<Uuid>,
    pub organization_id: Option<Uuid>,
    pub action: String,
    pub resource_type: Option<String>,
    pub resource_id: Option<Uuid>,
    pub result: AuditResult,
    pub details: serde_json::Value,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AuditResult {
    Allowed,
    Denied,
    Error,
}

impl std::fmt::Display for AuditResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditResult::Allowed => write!(f, "allowed"),
            AuditResult::Denied => write!(f, "denied"),
            AuditResult::Error => write!(f, "error"),
        }
    }
}

/// Database-backed RBAC repository
pub struct RbacRepository {
    pool: PgPool,
}

impl RbacRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Get user by ID
    pub async fn get_user(&self, user_id: Uuid) -> Result<Option<User>, sqlx::Error> {
        let user = sqlx::query_as!(
            User,
            r#"
            SELECT id, email, name, email_verified, is_active, last_login, created_at, updated_at as "updated_at!"
            FROM users
            WHERE id = $1 AND deleted_at IS NULL
            "#,
            user_id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(user)
    }

    /// Get user by email
    pub async fn get_user_by_email(&self, email: &str) -> Result<Option<User>, sqlx::Error> {
        let user = sqlx::query_as!(
            User,
            r#"
            SELECT id, email, name, email_verified, is_active, last_login, created_at, updated_at as "updated_at!"
            FROM users
            WHERE email = $1 AND deleted_at IS NULL
            "#,
            email
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(user)
    }

    /// Get organization by slug
    pub async fn get_organization(&self, slug: &str) -> Result<Option<Organization>, sqlx::Error> {
        let org = sqlx::query_as!(
            Organization,
            r#"
            SELECT id, name, slug, settings, created_at, updated_at
            FROM organizations
            WHERE slug = $1 AND deleted_at IS NULL
            "#,
            slug
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(org)
    }

    /// Check if user is member of organization
    pub async fn is_organization_member(
        &self,
        user_id: Uuid,
        organization_id: Uuid,
    ) -> Result<bool, sqlx::Error> {
        let exists = sqlx::query_scalar!(
            r#"
            SELECT EXISTS(
                SELECT 1 FROM organization_members
                WHERE user_id = $1 AND organization_id = $2
            ) as "exists!"
            "#,
            user_id,
            organization_id
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(exists)
    }

    /// Get user's roles in an organization
    pub async fn get_user_roles(
        &self,
        user_id: Uuid,
        organization_id: Uuid,
    ) -> Result<Vec<String>, sqlx::Error> {
        let roles = sqlx::query!(
            r#"
            SELECT r.name
            FROM user_roles ur
            JOIN roles r ON ur.role_id = r.id
            WHERE ur.user_id = $1
              AND ur.organization_id = $2
              AND (ur.expires_at IS NULL OR ur.expires_at > NOW())
            "#,
            user_id,
            organization_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| row.name)
        .collect();

        Ok(roles)
    }

    /// Check if user has a specific permission
    pub async fn check_permission(
        &self,
        user_id: Uuid,
        organization_id: Uuid,
        resource: &str,
        action: &str,
    ) -> Result<bool, sqlx::Error> {
        let has_permission = sqlx::query_scalar!(
            r#"
            SELECT user_has_permission($1, $2, $3, $4) as "has_perm!"
            "#,
            user_id,
            organization_id,
            resource,
            action
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(has_permission)
    }

    /// Assign role to user in organization
    pub async fn assign_role(
        &self,
        user_id: Uuid,
        role_name: &str,
        organization_id: Uuid,
        granted_by: Option<Uuid>,
    ) -> Result<Uuid, sqlx::Error> {
        let assignment_id = sqlx::query_scalar!(
            r#"
            INSERT INTO user_roles (user_id, role_id, organization_id, granted_by)
            SELECT $1, r.id, $2, $3
            FROM roles r
            WHERE r.name = $4
              AND (r.is_system = true OR r.organization_id = $2)
            ON CONFLICT (user_id, role_id, organization_id) DO NOTHING
            RETURNING id
            "#,
            user_id,
            organization_id,
            granted_by,
            role_name
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(assignment_id)
    }

    /// Remove role from user in organization
    pub async fn remove_role(
        &self,
        user_id: Uuid,
        role_name: &str,
        organization_id: Uuid,
    ) -> Result<bool, sqlx::Error> {
        let result = sqlx::query!(
            r#"
            DELETE FROM user_roles
            WHERE user_id = $1
              AND organization_id = $2
              AND role_id IN (
                  SELECT id FROM roles WHERE name = $3
              )
            "#,
            user_id,
            organization_id,
            role_name
        )
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    /// Record audit log entry
    pub async fn record_audit(
        &self,
        entry: AuditLogEntry,
    ) -> Result<Uuid, sqlx::Error> {
        let id = sqlx::query_scalar!(
            r#"
            INSERT INTO audit_log (
                user_id, organization_id, action, resource_type, resource_id,
                result, details, ip_address, user_agent
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id
            "#,
            entry.user_id,
            entry.organization_id,
            entry.action,
            entry.resource_type,
            entry.resource_id,
            entry.result.to_string(),
            entry.details,
            entry.ip_address,
            entry.user_agent
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(id)
    }

    /// Get audit logs for organization with filters
    pub async fn get_audit_logs(
        &self,
        organization_id: Uuid,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<AuditLogEntry>, sqlx::Error> {
        let rows = sqlx::query!(
            r#"
            SELECT
                id, user_id, organization_id, action, resource_type, resource_id,
                result, details, ip_address, user_agent, created_at
            FROM audit_log
            WHERE organization_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            "#,
            organization_id,
            limit,
            offset
        )
        .fetch_all(&self.pool)
        .await?;

        let entries = rows
            .into_iter()
            .map(|row| AuditLogEntry {
                id: row.id,
                user_id: row.user_id,
                organization_id: row.organization_id,
                action: row.action,
                resource_type: row.resource_type,
                resource_id: row.resource_id,
                result: match row.result.as_str() {
                    "allowed" => AuditResult::Allowed,
                    "denied" => AuditResult::Denied,
                    _ => AuditResult::Error,
                },
                details: row.details.unwrap_or(serde_json::json!({})),
                ip_address: row.ip_address,
                user_agent: row.user_agent,
                created_at: row.created_at,
            })
            .collect();

        Ok(entries)
    }

    /// Create new organization
    pub async fn create_organization(
        &self,
        name: &str,
        slug: &str,
    ) -> Result<Organization, sqlx::Error> {
        let org = sqlx::query_as!(
            Organization,
            r#"
            INSERT INTO organizations (name, slug, settings)
            VALUES ($1, $2, '{}')
            RETURNING id, name, slug, settings, created_at, updated_at
            "#,
            name,
            slug
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(org)
    }

    /// Create new user
    pub async fn create_user(
        &self,
        email: &str,
        name: Option<&str>,
        password_hash: Option<&str>,
    ) -> Result<User, sqlx::Error> {
        let user = sqlx::query_as!(
            User,
            r#"
            INSERT INTO users (email, name, password_hash, email_verified, is_active)
            VALUES ($1, $2, $3, false, true)
            RETURNING id, email, name, email_verified, is_active, last_login, created_at, updated_at as "updated_at!"
            "#,
            email,
            name,
            password_hash
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(user)
    }

    /// Add user to organization
    pub async fn add_organization_member(
        &self,
        organization_id: Uuid,
        user_id: Uuid,
        is_owner: bool,
    ) -> Result<Uuid, sqlx::Error> {
        let member_id = sqlx::query_scalar!(
            r#"
            INSERT INTO organization_members (organization_id, user_id, is_owner)
            VALUES ($1, $2, $3)
            ON CONFLICT (organization_id, user_id) DO NOTHING
            RETURNING id
            "#,
            organization_id,
            user_id,
            is_owner
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(member_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a PostgreSQL database connection
    // In production, use sqlx test fixtures or a test database

    #[test]
    fn test_audit_result_display() {
        assert_eq!(AuditResult::Allowed.to_string(), "allowed");
        assert_eq!(AuditResult::Denied.to_string(), "denied");
        assert_eq!(AuditResult::Error.to_string(), "error");
    }

    #[test]
    fn test_audit_result_serialization() {
        let result = AuditResult::Allowed;
        let json = serde_json::to_string(&result).unwrap();
        assert_eq!(json, "\"allowed\"");
    }
}
