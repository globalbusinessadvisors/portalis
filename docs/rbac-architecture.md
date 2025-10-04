# RBAC Architecture Design
**Phase 5 Week 37 - Role-Based Access Control**

## Executive Summary

This document defines the Role-Based Access Control (RBAC) architecture for Portalis, enabling fine-grained access control for enterprise deployments. The design supports multi-tenancy, audit logging, and policy-based permissions.

**Implementation Target**: Phase 5 Weeks 39-40
**Status**: Design Phase
**Owner**: Backend Team

---

## 1. Design Goals

### Primary Objectives
1. **Fine-Grained Access Control**: Control access to resources at the API endpoint level
2. **Multi-Tenant Support**: Isolate resources between organizations
3. **Audit Compliance**: Track all access decisions and policy changes
4. **Performance**: <10ms authorization overhead per request
5. **Extensibility**: Support custom roles and permissions

### Non-Goals
- **Row-Level Security**: (Deferred to Phase 6)
- **Attribute-Based Access Control (ABAC)**: (Deferred to Phase 6)
- **Dynamic Role Assignment**: Roles assigned explicitly by admins only

---

## 2. Role Hierarchy

### Standard Roles

```
┌─────────────────────────────────────────────────────────────┐
│                         ADMIN                               │
│  Full system access, user management, policy configuration  │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────────────┐   ┌───────────────┐   ┌──────────────┐
│   DEVELOPER   │   │   OPERATOR    │   │   AUDITOR    │
│ Read + Write  │   │ Deploy + Ops  │   │  Read-Only   │
└───────────────┘   └───────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                    ┌───────────────┐
                    │    VIEWER     │
                    │  Read-Only    │
                    └───────────────┘
```

### Role Definitions

#### 1. **Admin** (Superuser)
- **Permissions**:
  - Full CRUD on all resources
  - User and role management
  - Policy configuration
  - System settings
  - Audit log access
- **Use Cases**: System administrators, CTO, DevOps leads

#### 2. **Developer** (Standard User)
- **Permissions**:
  - Create/read/update/delete translation projects
  - Execute translations
  - View assessment reports
  - Access API endpoints
  - View own usage metrics
- **Restrictions**:
  - Cannot manage users
  - Cannot modify system settings
  - Cannot access other tenants' data
- **Use Cases**: Software engineers, data scientists

#### 3. **Operator** (Operations)
- **Permissions**:
  - Deploy translated code
  - Manage infrastructure
  - View system metrics
  - Trigger health checks
  - Restart services
- **Restrictions**:
  - Cannot create/modify translation projects
  - Cannot manage users
  - Read-only access to translations
- **Use Cases**: DevOps engineers, SRE teams

#### 4. **Auditor** (Compliance)
- **Permissions**:
  - Read-only access to all resources
  - Full audit log access
  - View all usage metrics
  - Export compliance reports
- **Restrictions**:
  - Cannot modify any resources
  - Cannot execute translations
- **Use Cases**: Security teams, compliance officers

#### 5. **Viewer** (Guest)
- **Permissions**:
  - Read-only access to assigned projects
  - View public documentation
  - View own profile
- **Restrictions**:
  - Cannot execute translations
  - Cannot view metrics
  - Cannot access audit logs
- **Use Cases**: Stakeholders, external reviewers

---

## 3. Permission Model

### Permission Structure

Permissions follow the format: `resource:action`

**Resources**:
- `project` - Translation projects
- `translation` - Translation executions
- `assessment` - Code assessments
- `user` - User accounts
- `role` - Roles and permissions
- `system` - System configuration
- `audit` - Audit logs
- `metrics` - Usage metrics

**Actions**:
- `create` - Create new resources
- `read` - View resources
- `update` - Modify existing resources
- `delete` - Remove resources
- `execute` - Run operations (translations, assessments)
- `admin` - Administrative operations

### Permission Matrix

| Role      | Projects | Translations | Assessments | Users | Roles | System | Audit | Metrics |
|-----------|----------|--------------|-------------|-------|-------|--------|-------|---------|
| Admin     | CRUD+X   | CRUD+X       | CRUD+X      | CRUD  | CRUD  | CRUD   | R     | R       |
| Developer | CRUD+X   | CRUD+X       | CRUD+X      | R     | -     | R      | -     | R (own) |
| Operator  | R        | R            | R           | -     | -     | R      | -     | R       |
| Auditor   | R        | R            | R           | R     | R     | R      | R     | R       |
| Viewer    | R        | -            | R           | -     | -     | -      | -     | -       |

**Legend**: C = Create, R = Read, U = Update, D = Delete, X = Execute, - = No Access

---

## 4. Policy Engine

### Option 1: Casbin (Recommended)

**Pros**:
- ✅ Mature, battle-tested library
- ✅ Supports RBAC, ABAC, ACL models
- ✅ High performance (<5ms per check)
- ✅ Adapters for PostgreSQL, Redis
- ✅ Policy management UI available

**Cons**:
- ❌ Additional dependency (Go library, needs Rust bindings)
- ❌ Learning curve for policy syntax

**Implementation**:
```rust
// Casbin model definition (RBAC with domains for multi-tenancy)
[request_definition]
r = sub, dom, obj, act

[policy_definition]
p = sub, dom, obj, act

[role_definition]
g = _, _, _

[policy_effect]
e = some(where (p.eft == allow))

[matchers]
m = g(r.sub, p.sub, r.dom) && r.dom == p.dom && r.obj == p.obj && r.act == p.act
```

**Example Policies**:
```csv
# Format: role, domain, resource, action
p, admin, *, *, *
p, developer, *, project, create
p, developer, *, project, read
p, developer, *, translation, execute
p, viewer, *, project, read

# Role assignments (user, role, domain)
g, user_123, developer, org_acme
g, user_456, admin, org_acme
```

### Option 2: Custom Policy Engine

**Pros**:
- ✅ Full control over implementation
- ✅ No external dependencies
- ✅ Optimized for Portalis use cases

**Cons**:
- ❌ Implementation effort (~2 weeks)
- ❌ Maintenance burden
- ❌ Potential bugs in policy evaluation

**Implementation**:
```rust
pub struct PolicyEngine {
    policies: HashMap<String, Vec<Permission>>,
    role_hierarchy: RoleGraph,
}

impl PolicyEngine {
    pub fn check_permission(
        &self,
        user: &User,
        resource: &str,
        action: &str,
        domain: &str,
    ) -> bool {
        // 1. Get user's roles in domain
        let roles = self.get_user_roles(user, domain);

        // 2. Check each role's permissions
        for role in roles {
            if self.has_permission(&role, resource, action) {
                return true;
            }
        }

        false
    }
}
```

**Recommendation**: **Use Casbin** for faster implementation and proven reliability.

---

## 5. Database Schema

### Tables

```sql
-- Organizations (tenants)
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) NOT NULL UNIQUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Users (extended from existing users table)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255),
    password_hash VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_login TIMESTAMP
);

-- Organization memberships
CREATE TABLE organization_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(organization_id, user_id)
);

-- Roles
CREATE TABLE roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    is_system BOOLEAN NOT NULL DEFAULT false, -- true for built-in roles
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(name, organization_id)
);

-- Permissions
CREATE TABLE permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource VARCHAR(100) NOT NULL, -- e.g., "project", "translation"
    action VARCHAR(50) NOT NULL,    -- e.g., "create", "read", "execute"
    description TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(resource, action)
);

-- Role-Permission mapping
CREATE TABLE role_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    permission_id UUID NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(role_id, permission_id)
);

-- User-Role assignments
CREATE TABLE user_roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP, -- optional expiration
    UNIQUE(user_id, role_id, organization_id)
);

-- Audit log
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL,
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    result VARCHAR(50) NOT NULL, -- "allowed", "denied"
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_organization_members_org ON organization_members(organization_id);
CREATE INDEX idx_organization_members_user ON organization_members(user_id);
CREATE INDEX idx_user_roles_user ON user_roles(user_id);
CREATE INDEX idx_user_roles_org ON user_roles(organization_id);
CREATE INDEX idx_audit_log_user ON audit_log(user_id);
CREATE INDEX idx_audit_log_org ON audit_log(organization_id);
CREATE INDEX idx_audit_log_created ON audit_log(created_at);
```

---

## 6. API Integration

### Middleware Architecture

```rust
// RBAC middleware for Axum
pub struct RBACMiddleware {
    policy_engine: Arc<PolicyEngine>,
}

impl RBACMiddleware {
    pub async fn check_permission<B>(
        State(engine): State<Arc<PolicyEngine>>,
        Extension(user): Extension<User>,
        request: Request<B>,
        next: Next<B>,
    ) -> Result<Response, StatusCode> {
        // 1. Extract resource and action from request
        let (resource, action) = Self::extract_permission(&request);

        // 2. Get user's organization from request context
        let org_id = request.headers()
            .get("X-Organization-ID")
            .and_then(|h| h.to_str().ok())
            .ok_or(StatusCode::BAD_REQUEST)?;

        // 3. Check permission
        if !engine.check_permission(&user, &resource, &action, org_id).await {
            // 4. Log denial
            audit_log::record_denial(&user, &resource, &action, org_id).await;
            return Err(StatusCode::FORBIDDEN);
        }

        // 5. Log access
        audit_log::record_access(&user, &resource, &action, org_id).await;

        // 6. Continue to handler
        Ok(next.run(request).await)
    }
}
```

### Protected Endpoints

```rust
// Apply RBAC middleware to protected routes
let app = Router::new()
    .route("/api/projects", post(create_project))
    .route("/api/projects/:id", get(get_project))
    .route("/api/translations/execute", post(execute_translation))
    .layer(RBACMiddleware::new(policy_engine))
    .route("/api/public/docs", get(get_docs)); // No RBAC
```

---

## 7. Multi-Tenancy

### Tenant Isolation

**Organization Context**:
- Every request includes `X-Organization-ID` header
- Middleware validates user belongs to organization
- Database queries automatically filtered by `organization_id`

**Data Isolation**:
```rust
// Row-Level Security (RLS) in PostgreSQL
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;

CREATE POLICY projects_isolation ON projects
    USING (organization_id = current_setting('app.current_organization')::uuid);
```

**Resource Quotas** (Phase 5 Week 45):
```rust
pub struct OrganizationQuota {
    pub max_projects: u32,
    pub max_translations_per_month: u32,
    pub max_storage_bytes: u64,
}
```

---

## 8. Audit Logging

### Audit Events

**Recorded Events**:
- Authentication (login, logout, failed attempts)
- Authorization (permission checks, denials)
- Resource operations (create, update, delete)
- Administrative actions (user management, role changes)

**Audit Log Entry**:
```rust
pub struct AuditEntry {
    pub id: Uuid,
    pub user_id: Option<Uuid>,
    pub organization_id: Option<Uuid>,
    pub action: String,
    pub resource_type: Option<String>,
    pub resource_id: Option<Uuid>,
    pub result: AuditResult, // Allowed, Denied
    pub ip_address: Option<IpAddr>,
    pub user_agent: Option<String>,
    pub created_at: DateTime<Utc>,
}
```

**Compliance Reporting**:
- Export audit logs in CSV/JSON format
- Filter by user, organization, date range
- Compliance dashboards (SOC2, ISO 27001)

---

## 9. Performance Considerations

### Caching Strategy

**Permission Cache**:
- Cache user permissions in Redis (TTL: 5 minutes)
- Invalidate on role/permission changes
- Cache key: `rbac:user:{user_id}:org:{org_id}:permissions`

```rust
pub async fn check_permission_cached(
    &self,
    user: &User,
    resource: &str,
    action: &str,
    org_id: &str,
) -> bool {
    let cache_key = format!("rbac:user:{}:org:{}:permissions", user.id, org_id);

    // Try cache first
    if let Some(permissions) = self.cache.get(&cache_key).await {
        return permissions.contains(&format!("{}:{}", resource, action));
    }

    // Cache miss - check database
    let allowed = self.check_permission(user, resource, action, org_id).await;

    // Update cache
    self.cache.set(&cache_key, &permissions, 300).await;

    allowed
}
```

**Performance Targets**:
- Authorization check: <10ms (p95)
- Database query: <5ms (p95)
- Cache hit rate: >90%

---

## 10. Implementation Plan

### Week 39: RBAC Foundation
- **Day 1-2**: Database schema migration
- **Day 3-4**: Policy engine implementation (Casbin integration)
- **Day 5**: RBAC middleware
- **Day 6-7**: Unit tests + integration tests

### Week 40: API Integration
- **Day 1-2**: Protect API endpoints
- **Day 3**: Multi-tenancy support
- **Day 4**: Audit logging
- **Day 5-6**: UI for role management
- **Day 7**: End-to-end testing

### Testing Strategy
- Unit tests: 100+ tests for policy engine
- Integration tests: API endpoint protection
- Load tests: 1000 req/sec with <10ms overhead
- Security tests: Attempt privilege escalation

---

## 11. Security Considerations

### Threat Model

**Threats**:
1. **Privilege Escalation**: User gains unauthorized access
2. **Cross-Tenant Access**: User accesses another org's data
3. **Policy Bypass**: Middleware failure allows unauthorized access
4. **Audit Log Tampering**: Attacker modifies audit trail

**Mitigations**:
1. **Defense in Depth**: Multiple layers of checks (middleware + database RLS)
2. **Least Privilege**: Default deny, explicit allow
3. **Immutable Audit Logs**: Append-only, write to separate database
4. **Regular Audits**: Automated security scans

### Best Practices
- ✅ Never trust client-provided role/permissions
- ✅ Always validate organization membership
- ✅ Log all authorization decisions
- ✅ Use prepared statements (prevent SQL injection)
- ✅ Rate limit authentication attempts

---

## 12. Migration Path

### Existing Users

**Phase 1** (Week 39):
- Create default organization for existing users
- Assign "Developer" role to all existing users
- No user impact (backward compatible)

**Phase 2** (Week 40):
- Introduce organization selection on login
- Allow admins to invite users to organizations
- Provide migration tool for multi-org users

**Phase 3** (Week 41+):
- Enforce RBAC on all endpoints
- Deprecate legacy authentication

---

## 13. Success Metrics

### KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| Authorization latency (p95) | <10ms | Prometheus metrics |
| Cache hit rate | >90% | Redis stats |
| Failed authorization attempts | <1% | Audit logs |
| Security vulnerabilities | 0 critical | Penetration testing |
| Audit log completeness | 100% | Automated checks |

### Acceptance Criteria

- [ ] All API endpoints protected by RBAC
- [ ] 5 standard roles implemented (Admin, Developer, Operator, Auditor, Viewer)
- [ ] Multi-tenant isolation verified
- [ ] Audit logging captures all access decisions
- [ ] Performance targets met (<10ms p95)
- [ ] 100+ tests passing (unit + integration)
- [ ] Security review completed (0 critical findings)
- [ ] Documentation published

---

## 14. Future Enhancements (Phase 6+)

### Attribute-Based Access Control (ABAC)
- Policy based on user attributes (department, location)
- Resource attributes (sensitivity, classification)
- Contextual attributes (time, IP address)

### Dynamic Role Assignment
- Auto-assign roles based on user attributes
- Temporary role elevation (time-boxed)

### Fine-Grained Permissions
- Row-level security (access specific projects)
- Field-level security (hide sensitive fields)

### API Keys and Service Accounts
- Machine-to-machine authentication
- Scoped API keys (limited permissions)

---

## 15. References

- [NIST RBAC Standard](https://csrc.nist.gov/projects/role-based-access-control)
- [Casbin Documentation](https://casbin.org/docs/overview)
- [OWASP Access Control Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Access_Control_Cheat_Sheet.html)
- [PostgreSQL Row-Level Security](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)

---

**Document Status**: DRAFT
**Last Updated**: October 4, 2025
**Next Review**: Week 39 (RBAC Implementation Kickoff)
**Owner**: Backend Team
**Approvers**: CTO, VP Engineering, Security Lead
