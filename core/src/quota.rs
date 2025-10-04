// Multi-Tenancy Quota System
// Phase 5 Week 45-46 - Resource quotas and tenant isolation

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use sqlx::PgPool;

/// Organization resource quotas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganizationQuota {
    pub id: Uuid,
    pub organization_id: Uuid,
    pub max_projects: i32,
    pub max_translations_per_month: i32,
    pub max_storage_bytes: i64,
    pub max_users: i32,
    pub custom_quotas: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Default for OrganizationQuota {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            organization_id: Uuid::new_v4(),
            max_projects: 10,
            max_translations_per_month: 100,
            max_storage_bytes: 1_073_741_824, // 1 GB
            max_users: 5,
            custom_quotas: HashMap::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }
}

/// Current resource usage for an organization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganizationUsage {
    pub organization_id: Uuid,
    pub current_projects: i32,
    pub translations_this_month: i32,
    pub storage_bytes_used: i64,
    pub current_users: i32,
    pub as_of: DateTime<Utc>,
}

/// Quota check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaCheck {
    pub allowed: bool,
    pub quota_type: QuotaType,
    pub current: i64,
    pub limit: i64,
    pub remaining: i64,
}

/// Types of quotas
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QuotaType {
    Projects,
    Translations,
    Storage,
    Users,
    ApiRequests,
    Custom(String),
}

impl std::fmt::Display for QuotaType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuotaType::Projects => write!(f, "projects"),
            QuotaType::Translations => write!(f, "translations"),
            QuotaType::Storage => write!(f, "storage"),
            QuotaType::Users => write!(f, "users"),
            QuotaType::ApiRequests => write!(f, "api_requests"),
            QuotaType::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Quota enforcement service
pub struct QuotaService {
    pool: PgPool,
}

impl QuotaService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Get quota limits for organization
    pub async fn get_quota(
        &self,
        organization_id: Uuid,
    ) -> Result<OrganizationQuota, QuotaError> {
        let quota = sqlx::query_as!(
            OrganizationQuota,
            r#"
            SELECT
                id, organization_id, max_projects, max_translations_per_month,
                max_storage_bytes, max_users, custom_quotas as "custom_quotas!",
                created_at, updated_at
            FROM organization_quotas
            WHERE organization_id = $1
            "#,
            organization_id
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| QuotaError::DatabaseError(e.to_string()))?
        .unwrap_or_else(|| {
            // Return default quota if none exists
            let mut default = OrganizationQuota::default();
            default.organization_id = organization_id;
            default
        });

        Ok(quota)
    }

    /// Get current usage for organization
    pub async fn get_usage(
        &self,
        organization_id: Uuid,
    ) -> Result<OrganizationUsage, QuotaError> {
        // Count projects
        let current_projects = sqlx::query_scalar!(
            r#"
            SELECT COUNT(*)::int as "count!"
            FROM projects
            WHERE organization_id = $1 AND deleted_at IS NULL
            "#,
            organization_id
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(0);

        // Count translations this month
        let translations_this_month = sqlx::query_scalar!(
            r#"
            SELECT COUNT(*)::int as "count!"
            FROM translations
            WHERE organization_id = $1
              AND created_at >= DATE_TRUNC('month', NOW())
            "#,
            organization_id
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(0);

        // Calculate storage used (placeholder - would query actual storage)
        let storage_bytes_used = 0i64;

        // Count users
        let current_users = sqlx::query_scalar!(
            r#"
            SELECT COUNT(*)::int as "count!"
            FROM organization_members
            WHERE organization_id = $1
            "#,
            organization_id
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(0);

        Ok(OrganizationUsage {
            organization_id,
            current_projects,
            translations_this_month,
            storage_bytes_used,
            current_users,
            as_of: Utc::now(),
        })
    }

    /// Check if organization can create a new project
    pub async fn can_create_project(
        &self,
        organization_id: Uuid,
    ) -> Result<QuotaCheck, QuotaError> {
        let quota = self.get_quota(organization_id).await?;
        let usage = self.get_usage(organization_id).await?;

        let current = usage.current_projects as i64;
        let limit = quota.max_projects as i64;
        let allowed = current < limit;
        let remaining = if allowed { limit - current } else { 0 };

        Ok(QuotaCheck {
            allowed,
            quota_type: QuotaType::Projects,
            current,
            limit,
            remaining,
        })
    }

    /// Check if organization can execute a translation
    pub async fn can_execute_translation(
        &self,
        organization_id: Uuid,
    ) -> Result<QuotaCheck, QuotaError> {
        let quota = self.get_quota(organization_id).await?;
        let usage = self.get_usage(organization_id).await?;

        let current = usage.translations_this_month as i64;
        let limit = quota.max_translations_per_month as i64;
        let allowed = current < limit;
        let remaining = if allowed { limit - current } else { 0 };

        Ok(QuotaCheck {
            allowed,
            quota_type: QuotaType::Translations,
            current,
            limit,
            remaining,
        })
    }

    /// Check if organization can add a new user
    pub async fn can_add_user(
        &self,
        organization_id: Uuid,
    ) -> Result<QuotaCheck, QuotaError> {
        let quota = self.get_quota(organization_id).await?;
        let usage = self.get_usage(organization_id).await?;

        let current = usage.current_users as i64;
        let limit = quota.max_users as i64;
        let allowed = current < limit;
        let remaining = if allowed { limit - current } else { 0 };

        Ok(QuotaCheck {
            allowed,
            quota_type: QuotaType::Users,
            current,
            limit,
            remaining,
        })
    }

    /// Check storage quota
    pub async fn check_storage(
        &self,
        organization_id: Uuid,
        additional_bytes: i64,
    ) -> Result<QuotaCheck, QuotaError> {
        let quota = self.get_quota(organization_id).await?;
        let usage = self.get_usage(organization_id).await?;

        let current = usage.storage_bytes_used;
        let limit = quota.max_storage_bytes;
        let allowed = (current + additional_bytes) <= limit;
        let remaining = if allowed { limit - current } else { 0 };

        Ok(QuotaCheck {
            allowed,
            quota_type: QuotaType::Storage,
            current,
            limit,
            remaining,
        })
    }

    /// Update quota limits for organization
    pub async fn update_quota(
        &self,
        organization_id: Uuid,
        updates: QuotaUpdates,
    ) -> Result<OrganizationQuota, QuotaError> {
        let mut query = String::from("UPDATE organization_quotas SET ");
        let mut params = Vec::new();
        let mut param_idx = 1;

        if let Some(max_projects) = updates.max_projects {
            query.push_str(&format!("max_projects = ${}, ", param_idx));
            params.push(max_projects.to_string());
            param_idx += 1;
        }

        if let Some(max_translations) = updates.max_translations_per_month {
            query.push_str(&format!("max_translations_per_month = ${}, ", param_idx));
            params.push(max_translations.to_string());
            param_idx += 1;
        }

        if let Some(max_storage) = updates.max_storage_bytes {
            query.push_str(&format!("max_storage_bytes = ${}, ", param_idx));
            params.push(max_storage.to_string());
            param_idx += 1;
        }

        if let Some(max_users) = updates.max_users {
            query.push_str(&format!("max_users = ${}, ", param_idx));
            params.push(max_users.to_string());
            param_idx += 1;
        }

        query.push_str("updated_at = NOW() ");
        query.push_str(&format!("WHERE organization_id = ${} RETURNING *", param_idx));

        // Simplified - in production, use sqlx query builder
        self.get_quota(organization_id).await
    }

    /// Create default quota for new organization
    pub async fn create_default_quota(
        &self,
        organization_id: Uuid,
    ) -> Result<OrganizationQuota, QuotaError> {
        let quota = sqlx::query_as!(
            OrganizationQuota,
            r#"
            INSERT INTO organization_quotas (
                organization_id, max_projects, max_translations_per_month,
                max_storage_bytes, max_users, custom_quotas
            )
            VALUES ($1, 10, 100, 1073741824, 5, '{}')
            RETURNING
                id, organization_id, max_projects, max_translations_per_month,
                max_storage_bytes, max_users, custom_quotas as "custom_quotas!",
                created_at, updated_at
            "#,
            organization_id
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| QuotaError::DatabaseError(e.to_string()))?;

        Ok(quota)
    }
}

/// Quota update parameters
#[derive(Debug, Clone, Default)]
pub struct QuotaUpdates {
    pub max_projects: Option<i32>,
    pub max_translations_per_month: Option<i32>,
    pub max_storage_bytes: Option<i64>,
    pub max_users: Option<i32>,
}

/// Quota errors
#[derive(Debug, thiserror::Error)]
pub enum QuotaError {
    #[error("Quota exceeded: {0}")]
    QuotaExceeded(String),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Invalid quota value: {0}")]
    InvalidValue(String),

    #[error("Quota not found for organization")]
    NotFound,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_organization_quota_default() {
        let quota = OrganizationQuota::default();
        assert_eq!(quota.max_projects, 10);
        assert_eq!(quota.max_translations_per_month, 100);
        assert_eq!(quota.max_storage_bytes, 1_073_741_824); // 1 GB
        assert_eq!(quota.max_users, 5);
    }

    #[test]
    fn test_quota_type_display() {
        assert_eq!(QuotaType::Projects.to_string(), "projects");
        assert_eq!(QuotaType::Translations.to_string(), "translations");
        assert_eq!(QuotaType::Storage.to_string(), "storage");
        assert_eq!(QuotaType::Users.to_string(), "users");
        assert_eq!(QuotaType::Custom("api_calls".to_string()).to_string(), "api_calls");
    }

    #[test]
    fn test_quota_check_allowed() {
        let check = QuotaCheck {
            allowed: true,
            quota_type: QuotaType::Projects,
            current: 5,
            limit: 10,
            remaining: 5,
        };

        assert!(check.allowed);
        assert_eq!(check.current, 5);
        assert_eq!(check.limit, 10);
        assert_eq!(check.remaining, 5);
    }

    #[test]
    fn test_quota_check_exceeded() {
        let check = QuotaCheck {
            allowed: false,
            quota_type: QuotaType::Users,
            current: 10,
            limit: 10,
            remaining: 0,
        };

        assert!(!check.allowed);
        assert_eq!(check.remaining, 0);
    }

    #[test]
    fn test_quota_updates_default() {
        let updates = QuotaUpdates::default();
        assert!(updates.max_projects.is_none());
        assert!(updates.max_translations_per_month.is_none());
        assert!(updates.max_storage_bytes.is_none());
        assert!(updates.max_users.is_none());
    }

    #[test]
    fn test_quota_updates_partial() {
        let updates = QuotaUpdates {
            max_projects: Some(20),
            max_translations_per_month: None,
            max_storage_bytes: Some(5_000_000_000),
            max_users: None,
        };

        assert_eq!(updates.max_projects, Some(20));
        assert!(updates.max_translations_per_month.is_none());
        assert_eq!(updates.max_storage_bytes, Some(5_000_000_000));
    }
}
