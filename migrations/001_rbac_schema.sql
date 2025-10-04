-- RBAC Schema Migration
-- Phase 5 Week 39 - Role-Based Access Control Database Schema
--
-- This migration creates the complete RBAC infrastructure including:
-- - Organizations (multi-tenancy)
-- - Users and authentication
-- - Roles and permissions
-- - Audit logging
--
-- Migration Version: 001
-- Created: October 4, 2025

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================================================
-- ORGANIZATIONS (Multi-Tenancy)
-- =============================================================================

CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) NOT NULL UNIQUE,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,

    CONSTRAINT slug_format CHECK (slug ~ '^[a-z0-9-]+$')
);

CREATE INDEX idx_organizations_slug ON organizations(slug) WHERE deleted_at IS NULL;
CREATE INDEX idx_organizations_created ON organizations(created_at);

COMMENT ON TABLE organizations IS 'Tenant organizations for multi-tenancy isolation';
COMMENT ON COLUMN organizations.slug IS 'URL-friendly identifier (lowercase, alphanumeric, hyphens)';
COMMENT ON COLUMN organizations.settings IS 'Organization-specific configuration (quotas, branding, etc)';

-- =============================================================================
-- USERS
-- =============================================================================

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255),
    password_hash VARCHAR(255), -- NULL if SSO-only user
    email_verified BOOLEAN NOT NULL DEFAULT false,
    is_active BOOLEAN NOT NULL DEFAULT true,
    last_login TIMESTAMP WITH TIME ZONE,
    login_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,

    CONSTRAINT email_format CHECK (email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

CREATE INDEX idx_users_email ON users(email) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_active ON users(is_active) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_created ON users(created_at);

COMMENT ON TABLE users IS 'User accounts with authentication credentials';
COMMENT ON COLUMN users.password_hash IS 'Bcrypt password hash, NULL for SSO-only users';
COMMENT ON COLUMN users.email_verified IS 'Email verification status for security';

-- =============================================================================
-- ORGANIZATION MEMBERSHIPS
-- =============================================================================

CREATE TABLE organization_members (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    is_owner BOOLEAN NOT NULL DEFAULT false,
    joined_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    UNIQUE(organization_id, user_id)
);

CREATE INDEX idx_org_members_org ON organization_members(organization_id);
CREATE INDEX idx_org_members_user ON organization_members(user_id);
CREATE INDEX idx_org_members_owner ON organization_members(organization_id, is_owner);

COMMENT ON TABLE organization_members IS 'Many-to-many relationship between users and organizations';
COMMENT ON COLUMN organization_members.is_owner IS 'Organization owner has special privileges';

-- =============================================================================
-- ROLES
-- =============================================================================

CREATE TABLE roles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    is_system BOOLEAN NOT NULL DEFAULT false,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    UNIQUE(name, organization_id),
    CONSTRAINT system_role_no_org CHECK (
        (is_system = true AND organization_id IS NULL) OR
        (is_system = false AND organization_id IS NOT NULL)
    )
);

CREATE INDEX idx_roles_org ON roles(organization_id);
CREATE INDEX idx_roles_system ON roles(is_system);
CREATE INDEX idx_roles_name ON roles(name);

COMMENT ON TABLE roles IS 'Roles define sets of permissions';
COMMENT ON COLUMN roles.is_system IS 'System roles (admin, developer, etc) cannot be deleted';
COMMENT ON COLUMN roles.organization_id IS 'NULL for system roles, set for custom org roles';

-- Insert default system roles
INSERT INTO roles (name, description, is_system, organization_id) VALUES
    ('admin', 'Full system access, user management, policy configuration', true, NULL),
    ('developer', 'Create/read/update/delete translation projects', true, NULL),
    ('operator', 'Deploy and operations access', true, NULL),
    ('auditor', 'Read-only access with audit capabilities', true, NULL),
    ('viewer', 'Limited read-only access', true, NULL);

-- =============================================================================
-- PERMISSIONS
-- =============================================================================

CREATE TABLE permissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource VARCHAR(100) NOT NULL,
    action VARCHAR(50) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    UNIQUE(resource, action)
);

CREATE INDEX idx_permissions_resource ON permissions(resource);
CREATE INDEX idx_permissions_action ON permissions(action);

COMMENT ON TABLE permissions IS 'Available permissions in the system (resource:action)';
COMMENT ON COLUMN permissions.resource IS 'Resource type (project, translation, user, etc)';
COMMENT ON COLUMN permissions.action IS 'Action type (create, read, update, delete, execute, admin)';

-- Insert default permissions
INSERT INTO permissions (resource, action, description) VALUES
    -- Project permissions
    ('project', 'create', 'Create new translation projects'),
    ('project', 'read', 'View translation projects'),
    ('project', 'update', 'Modify translation projects'),
    ('project', 'delete', 'Delete translation projects'),
    ('project', 'execute', 'Execute translation operations'),

    -- Translation permissions
    ('translation', 'create', 'Create new translations'),
    ('translation', 'read', 'View translations'),
    ('translation', 'update', 'Modify translations'),
    ('translation', 'delete', 'Delete translations'),
    ('translation', 'execute', 'Execute translation operations'),

    -- Assessment permissions
    ('assessment', 'create', 'Create code assessments'),
    ('assessment', 'read', 'View code assessments'),
    ('assessment', 'update', 'Modify assessments'),
    ('assessment', 'delete', 'Delete assessments'),
    ('assessment', 'execute', 'Run assessment operations'),

    -- User permissions
    ('user', 'create', 'Create new users'),
    ('user', 'read', 'View user information'),
    ('user', 'update', 'Modify user accounts'),
    ('user', 'delete', 'Delete user accounts'),

    -- Role permissions
    ('role', 'create', 'Create custom roles'),
    ('role', 'read', 'View roles and permissions'),
    ('role', 'update', 'Modify roles'),
    ('role', 'delete', 'Delete custom roles'),

    -- System permissions
    ('system', 'read', 'View system configuration'),
    ('system', 'update', 'Modify system settings'),
    ('system', 'admin', 'Administrative system operations'),

    -- Audit permissions
    ('audit', 'read', 'View audit logs'),

    -- Metrics permissions
    ('metrics', 'read', 'View usage metrics and analytics');

-- =============================================================================
-- ROLE-PERMISSION MAPPING
-- =============================================================================

CREATE TABLE role_permissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    permission_id UUID NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    UNIQUE(role_id, permission_id)
);

CREATE INDEX idx_role_permissions_role ON role_permissions(role_id);
CREATE INDEX idx_role_permissions_perm ON role_permissions(permission_id);

COMMENT ON TABLE role_permissions IS 'Many-to-many mapping between roles and permissions';

-- Assign permissions to default roles
-- Admin: Full access to everything
INSERT INTO role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM roles r
CROSS JOIN permissions p
WHERE r.name = 'admin' AND r.is_system = true;

-- Developer: CRUD on projects, translations, assessments + read on users/system/metrics
INSERT INTO role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM roles r
CROSS JOIN permissions p
WHERE r.name = 'developer' AND r.is_system = true
  AND (
      (p.resource IN ('project', 'translation', 'assessment') AND p.action IN ('create', 'read', 'update', 'delete', 'execute'))
      OR (p.resource = 'user' AND p.action = 'read')
      OR (p.resource = 'system' AND p.action = 'read')
      OR (p.resource = 'metrics' AND p.action = 'read')
  );

-- Operator: Read on projects/translations/assessments, read+update on system, read on metrics
INSERT INTO role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM roles r
CROSS JOIN permissions p
WHERE r.name = 'operator' AND r.is_system = true
  AND (
      (p.resource IN ('project', 'translation', 'assessment') AND p.action = 'read')
      OR (p.resource = 'system' AND p.action IN ('read', 'update'))
      OR (p.resource = 'metrics' AND p.action = 'read')
  );

-- Auditor: Read-only on everything
INSERT INTO role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM roles r
CROSS JOIN permissions p
WHERE r.name = 'auditor' AND r.is_system = true
  AND p.action = 'read';

-- Viewer: Read on projects and assessments only
INSERT INTO role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM roles r
CROSS JOIN permissions p
WHERE r.name = 'viewer' AND r.is_system = true
  AND p.resource IN ('project', 'assessment')
  AND p.action = 'read';

-- =============================================================================
-- USER-ROLE ASSIGNMENTS
-- =============================================================================

CREATE TABLE user_roles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    granted_by UUID REFERENCES users(id) ON DELETE SET NULL,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    UNIQUE(user_id, role_id, organization_id)
);

CREATE INDEX idx_user_roles_user ON user_roles(user_id);
CREATE INDEX idx_user_roles_role ON user_roles(role_id);
CREATE INDEX idx_user_roles_org ON user_roles(organization_id);
CREATE INDEX idx_user_roles_expires ON user_roles(expires_at) WHERE expires_at IS NOT NULL;

COMMENT ON TABLE user_roles IS 'User role assignments within organizations';
COMMENT ON COLUMN user_roles.granted_by IS 'User who granted this role (for audit trail)';
COMMENT ON COLUMN user_roles.expires_at IS 'Optional expiration for temporary role assignments';

-- =============================================================================
-- AUDIT LOG
-- =============================================================================

CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL,
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    result VARCHAR(50) NOT NULL,
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT result_values CHECK (result IN ('allowed', 'denied', 'error'))
);

-- Partitioning setup for audit log (by month)
-- This will be implemented separately for production deployments

CREATE INDEX idx_audit_log_user ON audit_log(user_id);
CREATE INDEX idx_audit_log_org ON audit_log(organization_id);
CREATE INDEX idx_audit_log_created ON audit_log(created_at DESC);
CREATE INDEX idx_audit_log_action ON audit_log(action);
CREATE INDEX idx_audit_log_result ON audit_log(result);
CREATE INDEX idx_audit_log_resource ON audit_log(resource_type, resource_id);

COMMENT ON TABLE audit_log IS 'Immutable audit trail of all user actions and authorization decisions';
COMMENT ON COLUMN audit_log.action IS 'Action performed (e.g., "user.login", "project.create", "permission.check")';
COMMENT ON COLUMN audit_log.result IS 'Outcome of the action (allowed, denied, error)';
COMMENT ON COLUMN audit_log.details IS 'Additional context (request params, error messages, etc)';

-- =============================================================================
-- API KEYS (for machine-to-machine authentication)
-- =============================================================================

CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    key_prefix VARCHAR(20) NOT NULL,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    scopes JSONB DEFAULT '[]',
    last_used_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT expires_future CHECK (expires_at IS NULL OR expires_at > created_at)
);

CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_user ON api_keys(user_id);
CREATE INDEX idx_api_keys_org ON api_keys(organization_id);
CREATE INDEX idx_api_keys_active ON api_keys(is_active) WHERE is_active = true;

COMMENT ON TABLE api_keys IS 'API keys for programmatic access';
COMMENT ON COLUMN api_keys.key_hash IS 'SHA-256 hash of the API key';
COMMENT ON COLUMN api_keys.key_prefix IS 'First 8 characters of key for identification (pk_live_xxxxx)';
COMMENT ON COLUMN api_keys.scopes IS 'Limited permissions for this API key (subset of user permissions)';

-- =============================================================================
-- RESOURCE QUOTAS (for multi-tenancy)
-- =============================================================================

CREATE TABLE organization_quotas (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL UNIQUE REFERENCES organizations(id) ON DELETE CASCADE,
    max_projects INTEGER NOT NULL DEFAULT 10,
    max_translations_per_month INTEGER NOT NULL DEFAULT 100,
    max_storage_bytes BIGINT NOT NULL DEFAULT 1073741824, -- 1 GB
    max_users INTEGER NOT NULL DEFAULT 5,
    custom_quotas JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_org_quotas_org ON organization_quotas(organization_id);

COMMENT ON TABLE organization_quotas IS 'Resource quotas per organization for multi-tenancy';
COMMENT ON COLUMN organization_quotas.max_storage_bytes IS 'Maximum storage in bytes (default 1GB)';
COMMENT ON COLUMN organization_quotas.custom_quotas IS 'Additional quota types (API rate limits, etc)';

-- =============================================================================
-- SSO CONNECTIONS (for enterprise SSO)
-- =============================================================================

CREATE TABLE sso_connections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    config JSONB NOT NULL,
    is_enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT provider_values CHECK (provider IN ('saml', 'oauth2', 'oidc', 'ldap'))
);

CREATE INDEX idx_sso_connections_org ON sso_connections(organization_id);
CREATE INDEX idx_sso_connections_provider ON sso_connections(provider);
CREATE INDEX idx_sso_connections_enabled ON sso_connections(is_enabled) WHERE is_enabled = true;

COMMENT ON TABLE sso_connections IS 'SSO provider configurations per organization';
COMMENT ON COLUMN sso_connections.provider IS 'SSO protocol (saml, oauth2, oidc, ldap)';
COMMENT ON COLUMN sso_connections.config IS 'Provider-specific configuration (metadata URL, client ID, etc)';

-- =============================================================================
-- TRIGGERS FOR UPDATED_AT
-- =============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_roles_updated_at BEFORE UPDATE ON roles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_org_quotas_updated_at BEFORE UPDATE ON organization_quotas
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sso_connections_updated_at BEFORE UPDATE ON sso_connections
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- FUNCTIONS FOR PERMISSION CHECKING
-- =============================================================================

CREATE OR REPLACE FUNCTION user_has_permission(
    p_user_id UUID,
    p_organization_id UUID,
    p_resource VARCHAR,
    p_action VARCHAR
)
RETURNS BOOLEAN AS $$
DECLARE
    has_perm BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1
        FROM user_roles ur
        JOIN role_permissions rp ON ur.role_id = rp.role_id
        JOIN permissions perm ON rp.permission_id = perm.id
        WHERE ur.user_id = p_user_id
          AND ur.organization_id = p_organization_id
          AND (ur.expires_at IS NULL OR ur.expires_at > NOW())
          AND perm.resource = p_resource
          AND perm.action = p_action
    ) INTO has_perm;

    RETURN has_perm;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION user_has_permission IS 'Check if user has specific permission in organization';

-- =============================================================================
-- ROW-LEVEL SECURITY (RLS) SETUP
-- =============================================================================

-- Enable RLS on sensitive tables (to be configured per application needs)
-- Example for projects table (when implemented):
-- ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY projects_isolation ON projects
--     USING (organization_id = current_setting('app.current_organization')::uuid);

-- =============================================================================
-- INITIAL DATA
-- =============================================================================

-- Create default organization for existing users
INSERT INTO organizations (name, slug, settings)
VALUES ('Default Organization', 'default', '{"created_during_migration": true}');

-- =============================================================================
-- MIGRATION COMPLETE
-- =============================================================================

COMMENT ON SCHEMA public IS 'RBAC schema migration 001 applied - ' || NOW()::TEXT;
