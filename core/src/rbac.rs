// RBAC (Role-Based Access Control) Module
// Phase 5 Week 37-40 - Foundation for enterprise access control
//
// This module provides the foundation for RBAC functionality.
// Full implementation planned for Weeks 39-40.

pub mod database;
pub mod middleware;

use std::collections::{HashMap, HashSet};
use uuid::Uuid;

pub use database::{RbacRepository, Organization, User, AuditLogEntry, AuditResult};
pub use middleware::{RbacMiddleware, AuthContext, RequiredPermission};

/// Standard roles in the Portalis RBAC system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// Full system access, user management, policy configuration
    Admin,
    /// Read + write access to projects and translations
    Developer,
    /// Deploy and operations access
    Operator,
    /// Read-only access with audit capabilities
    Auditor,
    /// Limited read-only access
    Viewer,
}

impl Role {
    /// Get all available roles
    pub fn all() -> Vec<Role> {
        vec![
            Role::Admin,
            Role::Developer,
            Role::Operator,
            Role::Auditor,
            Role::Viewer,
        ]
    }

    /// Get role name
    pub fn name(&self) -> &'static str {
        match self {
            Role::Admin => "admin",
            Role::Developer => "developer",
            Role::Operator => "operator",
            Role::Auditor => "auditor",
            Role::Viewer => "viewer",
        }
    }

    /// Get role description
    pub fn description(&self) -> &'static str {
        match self {
            Role::Admin => "Full system access, user management, policy configuration",
            Role::Developer => "Create/read/update/delete translation projects",
            Role::Operator => "Deploy and operations access",
            Role::Auditor => "Read-only access with audit capabilities",
            Role::Viewer => "Limited read-only access",
        }
    }
}

/// Resource types in the system
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Resource {
    Project,
    Translation,
    Assessment,
    User,
    RoleManagement,
    System,
    Audit,
    Metrics,
}

impl Resource {
    pub fn name(&self) -> &'static str {
        match self {
            Resource::Project => "project",
            Resource::Translation => "translation",
            Resource::Assessment => "assessment",
            Resource::User => "user",
            Resource::RoleManagement => "role",
            Resource::System => "system",
            Resource::Audit => "audit",
            Resource::Metrics => "metrics",
        }
    }
}

/// Actions that can be performed on resources
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Action {
    Create,
    Read,
    Update,
    Delete,
    Execute,
    Admin,
}

impl Action {
    pub fn name(&self) -> &'static str {
        match self {
            Action::Create => "create",
            Action::Read => "read",
            Action::Update => "update",
            Action::Delete => "delete",
            Action::Execute => "execute",
            Action::Admin => "admin",
        }
    }
}

/// Permission represents a resource + action combination
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Permission {
    pub resource: Resource,
    pub action: Action,
}

impl Permission {
    pub fn new(resource: Resource, action: Action) -> Self {
        Self { resource, action }
    }

    /// Format as "resource:action"
    pub fn to_string(&self) -> String {
        format!("{}:{}", self.resource.name(), self.action.name())
    }
}

/// Simple policy engine for RBAC
///
/// This is a placeholder implementation. In production (Week 39-40),
/// this will be replaced with Casbin or a more sophisticated engine.
pub struct PolicyEngine {
    /// Maps roles to their permissions
    role_permissions: HashMap<Role, HashSet<Permission>>,
    /// Maps users to their roles in organizations
    user_roles: HashMap<(Uuid, String), HashSet<Role>>,
}

impl PolicyEngine {
    /// Create a new policy engine with default role permissions
    pub fn new() -> Self {
        let mut engine = Self {
            role_permissions: HashMap::new(),
            user_roles: HashMap::new(),
        };
        engine.initialize_default_permissions();
        engine
    }

    /// Initialize default permissions for standard roles
    fn initialize_default_permissions(&mut self) {
        use Action::*;
        use Resource::*;

        // Admin: Full access to everything
        let admin_perms = vec![
            Permission::new(Project, Create),
            Permission::new(Project, Read),
            Permission::new(Project, Update),
            Permission::new(Project, Delete),
            Permission::new(Project, Execute),
            Permission::new(Translation, Create),
            Permission::new(Translation, Read),
            Permission::new(Translation, Update),
            Permission::new(Translation, Delete),
            Permission::new(Translation, Execute),
            Permission::new(Assessment, Create),
            Permission::new(Assessment, Read),
            Permission::new(Assessment, Update),
            Permission::new(Assessment, Delete),
            Permission::new(Assessment, Execute),
            Permission::new(User, Create),
            Permission::new(User, Read),
            Permission::new(User, Update),
            Permission::new(User, Delete),
            Permission::new(RoleManagement, Create),
            Permission::new(RoleManagement, Read),
            Permission::new(RoleManagement, Update),
            Permission::new(RoleManagement, Delete),
            Permission::new(System, Read),
            Permission::new(System, Update),
            Permission::new(System, Admin),
            Permission::new(Audit, Read),
            Permission::new(Metrics, Read),
        ];
        self.role_permissions
            .insert(Role::Admin, admin_perms.into_iter().collect());

        // Developer: CRUD on projects, translations, assessments
        let dev_perms = vec![
            Permission::new(Project, Create),
            Permission::new(Project, Read),
            Permission::new(Project, Update),
            Permission::new(Project, Delete),
            Permission::new(Project, Execute),
            Permission::new(Translation, Create),
            Permission::new(Translation, Read),
            Permission::new(Translation, Update),
            Permission::new(Translation, Delete),
            Permission::new(Translation, Execute),
            Permission::new(Assessment, Create),
            Permission::new(Assessment, Read),
            Permission::new(Assessment, Update),
            Permission::new(Assessment, Delete),
            Permission::new(Assessment, Execute),
            Permission::new(User, Read),
            Permission::new(System, Read),
            Permission::new(Metrics, Read),
        ];
        self.role_permissions
            .insert(Role::Developer, dev_perms.into_iter().collect());

        // Operator: Read-only on projects, operations on system
        let operator_perms = vec![
            Permission::new(Project, Read),
            Permission::new(Translation, Read),
            Permission::new(Assessment, Read),
            Permission::new(System, Read),
            Permission::new(System, Update),
            Permission::new(Metrics, Read),
        ];
        self.role_permissions
            .insert(Role::Operator, operator_perms.into_iter().collect());

        // Auditor: Read-only on everything
        let auditor_perms = vec![
            Permission::new(Project, Read),
            Permission::new(Translation, Read),
            Permission::new(Assessment, Read),
            Permission::new(User, Read),
            Permission::new(RoleManagement, Read),
            Permission::new(System, Read),
            Permission::new(Audit, Read),
            Permission::new(Metrics, Read),
        ];
        self.role_permissions
            .insert(Role::Auditor, auditor_perms.into_iter().collect());

        // Viewer: Minimal read-only access
        let viewer_perms = vec![
            Permission::new(Project, Read),
            Permission::new(Assessment, Read),
        ];
        self.role_permissions
            .insert(Role::Viewer, viewer_perms.into_iter().collect());
    }

    /// Assign a role to a user in an organization
    pub fn assign_role(&mut self, user_id: Uuid, organization: String, role: Role) {
        self.user_roles
            .entry((user_id, organization))
            .or_insert_with(HashSet::new)
            .insert(role);
    }

    /// Remove a role from a user in an organization
    pub fn remove_role(&mut self, user_id: Uuid, organization: String, role: Role) {
        if let Some(roles) = self.user_roles.get_mut(&(user_id, organization)) {
            roles.remove(&role);
        }
    }

    /// Get all roles for a user in an organization
    pub fn get_user_roles(&self, user_id: Uuid, organization: &str) -> Vec<Role> {
        self.user_roles
            .get(&(user_id, organization.to_string()))
            .map(|roles| roles.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Check if a user has a specific permission in an organization
    pub fn check_permission(
        &self,
        user_id: Uuid,
        organization: &str,
        resource: &Resource,
        action: &Action,
    ) -> bool {
        let roles = self.get_user_roles(user_id, organization);
        let permission = Permission::new(resource.clone(), action.clone());

        for role in roles {
            if let Some(permissions) = self.role_permissions.get(&role) {
                if permissions.contains(&permission) {
                    return true;
                }
            }
        }

        false
    }

    /// Get all permissions for a role
    pub fn get_role_permissions(&self, role: &Role) -> Vec<Permission> {
        self.role_permissions
            .get(role)
            .map(|perms| perms.iter().cloned().collect())
            .unwrap_or_default()
    }
}

impl Default for PolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_names() {
        assert_eq!(Role::Admin.name(), "admin");
        assert_eq!(Role::Developer.name(), "developer");
        assert_eq!(Role::Operator.name(), "operator");
        assert_eq!(Role::Auditor.name(), "auditor");
        assert_eq!(Role::Viewer.name(), "viewer");
    }

    #[test]
    fn test_role_descriptions() {
        assert!(Role::Admin.description().contains("Full system access"));
        assert!(Role::Developer.description().contains("translation projects"));
    }

    #[test]
    fn test_all_roles() {
        let roles = Role::all();
        assert_eq!(roles.len(), 5);
        assert!(roles.contains(&Role::Admin));
        assert!(roles.contains(&Role::Developer));
    }

    #[test]
    fn test_permission_to_string() {
        let perm = Permission::new(Resource::Project, Action::Create);
        assert_eq!(perm.to_string(), "project:create");
    }

    #[test]
    fn test_policy_engine_new() {
        let engine = PolicyEngine::new();
        assert!(!engine.role_permissions.is_empty());
        assert_eq!(engine.role_permissions.len(), 5); // 5 roles
    }

    #[test]
    fn test_admin_has_all_permissions() {
        let engine = PolicyEngine::new();
        let admin_perms = engine.get_role_permissions(&Role::Admin);

        // Admin should have permissions on all resources
        assert!(admin_perms
            .iter()
            .any(|p| p.resource == Resource::Project && p.action == Action::Create));
        assert!(admin_perms
            .iter()
            .any(|p| p.resource == Resource::User && p.action == Action::Delete));
        assert!(admin_perms
            .iter()
            .any(|p| p.resource == Resource::System && p.action == Action::Admin));
    }

    #[test]
    fn test_developer_permissions() {
        let engine = PolicyEngine::new();
        let dev_perms = engine.get_role_permissions(&Role::Developer);

        // Developer can create projects
        assert!(dev_perms
            .iter()
            .any(|p| p.resource == Resource::Project && p.action == Action::Create));

        // Developer cannot delete users
        assert!(!dev_perms
            .iter()
            .any(|p| p.resource == Resource::User && p.action == Action::Delete));

        // Developer cannot access audit logs
        assert!(!dev_perms
            .iter()
            .any(|p| p.resource == Resource::Audit));
    }

    #[test]
    fn test_viewer_has_minimal_permissions() {
        let engine = PolicyEngine::new();
        let viewer_perms = engine.get_role_permissions(&Role::Viewer);

        // Viewer can only read projects and assessments
        assert_eq!(viewer_perms.len(), 2);
        assert!(viewer_perms
            .iter()
            .any(|p| p.resource == Resource::Project && p.action == Action::Read));
        assert!(viewer_perms
            .iter()
            .any(|p| p.resource == Resource::Assessment && p.action == Action::Read));
    }

    #[test]
    fn test_assign_role() {
        let mut engine = PolicyEngine::new();
        let user_id = Uuid::new_v4();
        let org = "acme-corp".to_string();

        engine.assign_role(user_id, org.clone(), Role::Developer);

        let roles = engine.get_user_roles(user_id, &org);
        assert_eq!(roles.len(), 1);
        assert!(roles.contains(&Role::Developer));
    }

    #[test]
    fn test_assign_multiple_roles() {
        let mut engine = PolicyEngine::new();
        let user_id = Uuid::new_v4();
        let org = "acme-corp".to_string();

        engine.assign_role(user_id, org.clone(), Role::Developer);
        engine.assign_role(user_id, org.clone(), Role::Auditor);

        let roles = engine.get_user_roles(user_id, &org);
        assert_eq!(roles.len(), 2);
        assert!(roles.contains(&Role::Developer));
        assert!(roles.contains(&Role::Auditor));
    }

    #[test]
    fn test_remove_role() {
        let mut engine = PolicyEngine::new();
        let user_id = Uuid::new_v4();
        let org = "acme-corp".to_string();

        engine.assign_role(user_id, org.clone(), Role::Developer);
        engine.assign_role(user_id, org.clone(), Role::Auditor);

        engine.remove_role(user_id, org.clone(), Role::Developer);

        let roles = engine.get_user_roles(user_id, &org);
        assert_eq!(roles.len(), 1);
        assert!(!roles.contains(&Role::Developer));
        assert!(roles.contains(&Role::Auditor));
    }

    #[test]
    fn test_check_permission_allowed() {
        let mut engine = PolicyEngine::new();
        let user_id = Uuid::new_v4();
        let org = "acme-corp";

        engine.assign_role(user_id, org.to_string(), Role::Developer);

        // Developer can create projects
        assert!(engine.check_permission(
            user_id,
            org,
            &Resource::Project,
            &Action::Create
        ));
    }

    #[test]
    fn test_check_permission_denied() {
        let mut engine = PolicyEngine::new();
        let user_id = Uuid::new_v4();
        let org = "acme-corp";

        engine.assign_role(user_id, org.to_string(), Role::Viewer);

        // Viewer cannot create projects
        assert!(!engine.check_permission(
            user_id,
            org,
            &Resource::Project,
            &Action::Create
        ));
    }

    #[test]
    fn test_multi_tenant_isolation() {
        let mut engine = PolicyEngine::new();
        let user_id = Uuid::new_v4();
        let org1 = "acme-corp";
        let org2 = "globex";

        engine.assign_role(user_id, org1.to_string(), Role::Admin);

        // User is admin in org1
        assert!(engine.check_permission(
            user_id,
            org1,
            &Resource::User,
            &Action::Delete
        ));

        // User has no roles in org2
        assert!(!engine.check_permission(
            user_id,
            org2,
            &Resource::User,
            &Action::Delete
        ));
    }

    #[test]
    fn test_permission_inheritance() {
        let mut engine = PolicyEngine::new();
        let user_id = Uuid::new_v4();
        let org = "acme-corp";

        // User with multiple roles gets combined permissions
        engine.assign_role(user_id, org.to_string(), Role::Developer);
        engine.assign_role(user_id, org.to_string(), Role::Auditor);

        // From Developer role
        assert!(engine.check_permission(
            user_id,
            org,
            &Resource::Project,
            &Action::Create
        ));

        // From Auditor role
        assert!(engine.check_permission(user_id, org, &Resource::Audit, &Action::Read));
    }
}
