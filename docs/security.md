# Security Guide

Comprehensive security documentation for deploying and operating Portalis in enterprise environments.

## Table of Contents

- [Threat Modeling](#threat-modeling)
- [WASM Sandboxing](#wasm-sandboxing)
- [Container Security](#container-security)
- [Network Security](#network-security)
- [Authentication & Authorization](#authentication--authorization)
- [Secret Management](#secret-management)
- [Security Testing](#security-testing)
- [Vulnerability Management](#vulnerability-management)
- [Incident Response](#incident-response)
- [Compliance & Auditing](#compliance--auditing)
- [Security Best Practices](#security-best-practices)
- [Reporting Security Issues](#reporting-security-issues)

## Threat Modeling

### Attack Surface Analysis

Portalis has the following attack surfaces that require security controls:

**External Attack Surfaces**:
- **API Endpoints**: REST API for translation requests
- **Container Images**: Docker images pulled from registries
- **Network Ingress**: HTTP/HTTPS traffic from clients
- **WebAssembly Modules**: User-provided WASM code execution
- **File Upload**: Source code file uploads for translation

**Internal Attack Surfaces**:
- **Inter-service Communication**: Agent-to-agent communication
- **Database Access**: Persistent data storage
- **Logging System**: Structured log collection
- **Monitoring Endpoints**: Prometheus metrics, health checks
- **Admin API**: Privileged operations

**Infrastructure Attack Surfaces**:
- **Kubernetes API**: Cluster management
- **Container Runtime**: Docker/containerd
- **GPU Access**: CUDA workloads via NVIDIA DGX
- **Storage Volumes**: Persistent and ephemeral volumes

### STRIDE Threat Categories

Analysis of threats using the STRIDE methodology:

| Category | Threat | Mitigation |
|----------|--------|------------|
| **Spoofing** | Attacker impersonates legitimate user | API key authentication, mTLS for inter-service communication |
| **Tampering** | Modification of data in transit | TLS 1.3 encryption, integrity checks, signed artifacts |
| **Repudiation** | User denies performing action | Comprehensive audit logging (see [logging.rs](/workspace/portalis/core/src/logging.rs)) |
| **Information Disclosure** | Unauthorized data access | Encryption at rest, WASM memory isolation, network segmentation |
| **Denial of Service** | Resource exhaustion attacks | Rate limiting, resource quotas, timeout enforcement |
| **Elevation of Privilege** | Unauthorized privilege escalation | RBAC, non-root containers, capability dropping, WASM sandboxing |

### Risk Assessment Matrix

**Likelihood x Impact = Risk Level**

| Risk | Likelihood | Impact | Priority | Mitigation Status |
|------|------------|--------|----------|-------------------|
| Malicious WASM code execution | High | Critical | P0 | Mitigated (sandboxing) |
| API credential compromise | Medium | High | P1 | Mitigated (rotation, encryption) |
| Container escape | Low | Critical | P1 | Mitigated (runtime security) |
| DDoS attack | Medium | Medium | P2 | Partial (rate limiting) |
| Dependency vulnerability | High | Medium | P2 | Mitigated (automated scanning) |
| Insider threat | Low | High | P2 | Partial (audit logging, RBAC) |
| Supply chain attack | Medium | High | P1 | Mitigated (SBOM, image scanning) |

**Risk Levels**:
- **P0 (Critical)**: Immediate action required
- **P1 (High)**: Address within 1 week
- **P2 (Medium)**: Address within 1 month
- **P3 (Low)**: Address in next planning cycle

## WASM Sandboxing

WebAssembly provides strong isolation for untrusted code execution, critical for translating user-provided code.

### Capability-Based Security Model

Portalis uses WASI (WebAssembly System Interface) with strict capability controls:

**Default Denied Capabilities**:
- File system access (read/write)
- Network socket creation
- Process spawning
- System calls
- Signal handling

**Explicitly Granted Capabilities** (minimal):
```rust
// WASM module initialization with restricted capabilities
use wasmtime::*;

let engine = Engine::default();
let mut linker = Linker::new(&engine);

// Grant only stdio capabilities
wasmtime_wasi::add_to_linker(&mut linker, |s| s)?;

let wasi = WasiCtxBuilder::new()
    .inherit_stdout()  // Allow stdout for results
    .inherit_stderr()  // Allow stderr for errors
    // No filesystem preopens
    // No network sockets
    // No environment variables
    .build();
```

**Capability Verification**:
- All WASI calls are intercepted and validated
- Attempts to access restricted capabilities fail gracefully
- Violations are logged for security monitoring

### Memory Isolation Details

**Linear Memory Bounds**:
- Each WASM module has isolated linear memory (max 4GB)
- Memory is not shared between modules
- Out-of-bounds access triggers immediate trap

**Memory Limits Enforced**:
```rust
// Memory configuration per WASM module
let memory_type = MemoryType::new(
    1,      // Initial: 1 page (64KB)
    Some(256), // Maximum: 256 pages (16MB)
);

// Enforce limits in wasmtime
let mut config = Config::new();
config.max_wasm_stack(512 * 1024);  // 512KB stack
config.allocation_strategy(InstanceAllocationStrategy::Pooling {
    strategy: PoolingAllocationStrategy::default(),
    module_limits: ModuleLimits {
        memory_pages: 256,
        table_elements: 1000,
        ..Default::default()
    },
});
```

**Memory Safety**:
- No pointer arithmetic outside module memory
- Host memory is completely inaccessible
- Stack overflow protection via guard pages

### System Call Restrictions

**Blocked System Calls**:
- `open`, `read`, `write` (file operations)
- `socket`, `bind`, `connect` (networking)
- `fork`, `exec` (process creation)
- `ioctl`, `fcntl` (device control)
- `ptrace`, `mmap` (debugging/memory manipulation)

**Allowed Operations**:
- Pure computation (arithmetic, logic)
- Memory allocation within limits
- Stdout/stderr writes (rate limited)
- Controlled exits

**Seccomp-BPF Filter** (additional layer):
```yaml
# Seccomp profile for WASM runtime
apiVersion: v1
kind: Pod
spec:
  securityContext:
    seccompProfile:
      type: Localhost
      localhostProfile: wasm-restricted.json
```

### Resource Limits

**CPU Time Limits**:
```rust
// Fuel-based execution limits (wasmtime)
let mut config = Config::new();
config.consume_fuel(true);

let mut store = Store::new(&engine, wasi);
store.add_fuel(10_000_000)?;  // 10M instructions

// Trap if fuel exhausted
instance.get_typed_func::<(), ()>(&mut store, "run")
    .call(&mut store, ())
    .map_err(|e| "Execution timeout")?;
```

**Memory Limits** (enforced at multiple layers):
- WASM module memory: 16MB max
- Container memory: 512MB max
- Kubernetes pod memory: 1GB max

**Execution Timeout**:
- Per-function timeout: 30 seconds
- Total translation timeout: 5 minutes
- Idle timeout: 1 minute

**Rate Limits**:
- System calls: 1000/second
- Memory allocations: 100/second
- Stdout/stderr writes: 10KB/second

## Container Security

### Image Scanning

**Trivy Integration** (automated in CI/CD):

Our [security workflow](/.github/workflows/security.yml) runs comprehensive image scanning:

```bash
# Scan filesystem for vulnerabilities
trivy fs --severity CRITICAL,HIGH,MEDIUM .

# Scan Docker images before deployment
trivy image --severity CRITICAL,HIGH portalis:latest

# Scan for secrets in repository
trivy repo --scanners secret .
```

**Automated Scanning Schedule**:
- Pull request: All CRITICAL and HIGH findings block merge
- Daily: Full security scan of all images
- Release: Complete SBOM generation + vulnerability report

**Scan Results**:
```yaml
# Example Trivy output integration
- name: Upload Trivy results to GitHub Security
  uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: 'trivy-results.sarif'
```

**Vulnerability Remediation SLA**:
- **CRITICAL**: Patch within 24 hours
- **HIGH**: Patch within 7 days
- **MEDIUM**: Patch within 30 days
- **LOW**: Address in next release

### Runtime Security

**Non-Root Containers** (mandatory):
```dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
# Create unprivileged user
RUN groupadd -r portalis --gid=1000 && \
    useradd -r -g portalis --uid=1000 --shell=/bin/false portalis

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/portalis /app/

# Set ownership
RUN chown -R portalis:portalis /app

# Drop to unprivileged user
USER portalis:portalis

ENTRYPOINT ["/app/portalis"]
```

**Security Context (Kubernetes)**:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: portalis-transpiler
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: transpiler
    image: portalis/transpiler:latest
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
          - ALL
      privileged: false
    resources:
      limits:
        memory: "1Gi"
        cpu: "2000m"
      requests:
        memory: "512Mi"
        cpu: "1000m"
    volumeMounts:
    - name: tmp
      mountPath: /tmp
      readOnly: false
  volumes:
  - name: tmp
    emptyDir: {}
```

**AppArmor/SELinux Profiles**:
```yaml
# AppArmor profile annotation
metadata:
  annotations:
    container.apparmor.security.beta.kubernetes.io/transpiler: localhost/portalis-restricted
```

**Runtime Monitoring**:
- Process monitoring (Falco)
- Network connection tracking
- File access auditing
- Anomaly detection

### Network Policies

**Kubernetes Network Policies** (default deny):

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: portalis-network-policy
  namespace: portalis-deployment
spec:
  podSelector:
    matchLabels:
      app: portalis
  policyTypes:
  - Ingress
  - Egress

  # Ingress: Allow only from nginx-ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8080

  # Egress: Allow DNS, Triton, monitoring
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53  # DNS
  - to:
    - podSelector:
        matchLabels:
          app: triton-server
    ports:
    - protocol: TCP
      port: 8000  # Triton HTTP
    - protocol: TCP
      port: 8001  # Triton metrics
  - to:
    - podSelector:
        matchLabels:
          app: prometheus
    ports:
    - protocol: TCP
      port: 9090
```

**Service Mesh (Istio)**:
```yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: portalis-mtls
  namespace: portalis-deployment
spec:
  mtls:
    mode: STRICT  # Require mTLS for all traffic
```

**TLS Encryption**:
- All HTTP traffic encrypted with TLS 1.3
- Certificate management via cert-manager
- Automatic certificate renewal (Let's Encrypt)
- HSTS headers enabled

### Secrets Management

**Kubernetes Secrets** (encrypted at rest):

See [secrets.yaml example](/workspace/portalis/nim-microservices/k8s/rust-transpiler/secrets.yaml) for reference.

```bash
# Create secrets from literal values
kubectl create secret generic portalis-secrets \
  --namespace=portalis-deployment \
  --from-literal=api-key=$(openssl rand -base64 32) \
  --from-literal=db-password=$(openssl rand -base64 32) \
  --from-literal=jwt-secret=$(openssl rand -base64 32)

# Create secrets from files
kubectl create secret generic portalis-tls \
  --namespace=portalis-deployment \
  --from-file=tls.crt=/path/to/cert.pem \
  --from-file=tls.key=/path/to/key.pem
```

**External Secrets Operator** (recommended for production):

```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
  namespace: portalis-deployment
spec:
  provider:
    vault:
      server: "https://vault.portalis.internal:8200"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "portalis-role"
---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: portalis-api-keys
  namespace: portalis-deployment
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: portalis-secrets
    creationPolicy: Owner
  data:
  - secretKey: api-key
    remoteRef:
      key: secret/portalis/api
      property: key
```

**Supported Secret Backends**:
- **HashiCorp Vault**: Enterprise secret management
- **AWS Secrets Manager**: AWS-native secrets
- **Azure Key Vault**: Azure-native secrets
- **Google Secret Manager**: GCP-native secrets
- **Sealed Secrets**: GitOps-friendly encrypted secrets

**Secret Rotation**:
- API keys: Rotate every 90 days
- Database passwords: Rotate every 60 days
- TLS certificates: Automatic renewal 30 days before expiry
- JWT secrets: Rotate every 180 days

**Secret Access Audit**:
```bash
# View secret access logs
kubectl logs -n portalis-deployment -l app=portalis --since=24h | \
  grep "secret_access" | \
  jq '.user_id, .secret_name, .timestamp'
```

## Security Testing

### SAST (Static Application Security Testing)

**Automated SAST Tools**:

1. **Cargo Clippy** (Rust linter with security checks):
```bash
cargo clippy --all-targets --all-features -- \
  -W clippy::all \
  -W clippy::pedantic \
  -W clippy::nursery \
  -W clippy::cargo
```

2. **Cargo Audit** (dependency vulnerability scanning):
```bash
# Automated daily via security.yml workflow
cargo audit --json > audit-report.json
cargo audit --deny warnings
```

3. **Semgrep** (pattern-based code analysis):
```bash
semgrep --config=auto --json --output=semgrep-report.json .
```

**SAST in CI/CD**:
- Run on every pull request
- Block merge on HIGH/CRITICAL findings
- Generate SARIF reports for GitHub Security

### DAST (Dynamic Application Security Testing)

**OWASP ZAP** (automated penetration testing):

```bash
# Baseline scan
docker run -t owasp/zap2docker-stable zap-baseline.py \
  -t https://api.portalis.dev \
  -r zap-report.html

# Full scan with authentication
docker run -t owasp/zap2docker-stable zap-full-scan.py \
  -t https://api.portalis.dev \
  -c zap-config.yaml \
  -r zap-full-report.html
```

**API Security Testing**:
- SQL injection testing
- XSS vulnerability scanning
- Authentication bypass attempts
- Authorization checks
- Rate limiting validation
- Input fuzzing

**Testing Schedule**:
- Baseline DAST: Weekly
- Full DAST: Monthly
- Post-deployment: After major releases

### Dependency Scanning

**Cargo Audit** (automated via [security.yml](/.github/workflows/security.yml)):

```bash
# Check for known vulnerabilities
cargo audit

# Generate detailed report
cargo audit --json > vulnerability-report.json

# Check against RustSec Advisory Database
cargo audit --db ~/.cargo/advisory-db
```

**Cargo Deny** (license and security policy enforcement):

```bash
# Check for security advisories
cargo deny check advisories

# Validate licenses
cargo deny check licenses

# Detect duplicate dependencies
cargo deny check bans

# Verify dependency sources
cargo deny check sources
```

**Dependabot** (automated dependency updates):
```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "cargo"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    labels:
      - "dependencies"
      - "security"
```

**SBOM (Software Bill of Materials)**:
```bash
# Generate SPDX format
cargo sbom > sbom-spdx.json

# Generate CycloneDX format
cargo cyclonedx > sbom-cyclonedx.xml

# Verify SBOM integrity
cosign verify-blob --signature sbom.sig --key cosign.pub sbom-spdx.json
```

### Penetration Testing Guidelines

**Pre-Engagement**:
1. Define scope (IP ranges, domains, services)
2. Establish rules of engagement
3. Set testing window
4. Identify emergency contacts

**Testing Methodology**:
- Reconnaissance and information gathering
- Vulnerability assessment
- Exploitation (with approval)
- Post-exploitation (privilege escalation)
- Reporting and remediation

**Approved Testing Tools**:
- Nmap (network scanning)
- Burp Suite (web application testing)
- Metasploit (exploitation framework)
- sqlmap (SQL injection testing)
- Nuclei (vulnerability scanner)

**Out of Scope** (prohibited):
- Social engineering attacks
- Physical security testing
- DoS/DDoS attacks
- Third-party systems
- Production data modification

**Penetration Testing Schedule**:
- External pentest: Annually
- Internal pentest: Bi-annually
- Web application pentest: Quarterly
- Post-incident: After security incidents

### Security Regression Testing

**Automated Security Test Suite**:

```rust
// Security regression tests
#[cfg(test)]
mod security_tests {
    use super::*;

    #[test]
    fn test_api_authentication_required() {
        // Ensure unauthenticated requests are rejected
        let response = client.get("/api/v1/translate").send().unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn test_sql_injection_prevention() {
        // Test SQL injection attempts are blocked
        let payload = "'; DROP TABLE users; --";
        let response = client.post("/api/v1/search")
            .json(&json!({"query": payload}))
            .send().unwrap();
        assert_ne!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_path_traversal_prevention() {
        // Test path traversal attempts are blocked
        let response = client.get("/api/v1/files/../../etc/passwd").send().unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_xss_sanitization() {
        // Test XSS payloads are sanitized
        let payload = "<script>alert('XSS')</script>";
        let response = client.post("/api/v1/comment")
            .json(&json!({"text": payload}))
            .send().unwrap();
        let body = response.text().unwrap();
        assert!(!body.contains("<script>"));
    }
}
```

**Regression Test Categories**:
- Authentication and authorization
- Input validation and sanitization
- Cryptographic operations
- Session management
- Error handling (no sensitive data leaks)
- Security headers (CSP, HSTS, X-Frame-Options)

## Vulnerability Management

### Reporting Process

**DO NOT** create public GitHub issues for security vulnerabilities.

**Reporting Channels**:

1. **Email** (preferred): security@portalis.dev
   - PGP key: https://portalis.dev/security.asc
   - Expected response: 24 hours

2. **GitHub Security Advisories**:
   - Navigate to: https://github.com/portalis/portalis/security/advisories
   - Click "Report a vulnerability"
   - Fill out vulnerability details

3. **Bug Bounty Program** (coming Q2 2026):
   - Scope and rewards: https://portalis.dev/security/bounty
   - Managed via HackerOne

**Required Information**:
- Vulnerability description
- Steps to reproduce
- Proof of concept (if available)
- Impact assessment
- Affected versions
- Suggested remediation (optional)

### Response SLAs

**Acknowledgment**:
- **Target**: Within 24 hours
- **Action**: Confirm receipt and assign tracking ID

**Initial Assessment**:
- **Target**: Within 72 hours
- **Action**: Severity classification, impact analysis

**Remediation Timeline** (from initial assessment):

| Severity | Description | Response Time | Patch Deployment |
|----------|-------------|---------------|------------------|
| **Critical** | Remote code execution, data breach | 24 hours | 48 hours |
| **High** | Privilege escalation, authentication bypass | 7 days | 14 days |
| **Medium** | DoS, information disclosure | 30 days | 60 days |
| **Low** | Minor security improvements | 90 days | Next release |

**Status Updates**:
- Critical: Daily updates
- High: Weekly updates
- Medium/Low: Bi-weekly updates

### Disclosure Policy

**Coordinated Disclosure**:
1. Researcher reports vulnerability privately
2. Portalis confirms and assesses severity
3. Patch developed and tested
4. Security advisory prepared
5. Patch released to customers
6. Public disclosure after 90 days (or earlier if agreed)

**Public Disclosure Timeline**:
- **Preferred**: 90 days after initial report
- **Minimum**: After patch is available (no earlier than 30 days)
- **Maximum**: 120 days (even if patch not ready)

**Security Advisory Format**:
```markdown
# Security Advisory: PORTALIS-2025-001

**Severity**: HIGH
**CVE**: CVE-2025-XXXXX
**Affected Versions**: 1.0.0 - 1.2.3
**Fixed Version**: 1.2.4

## Summary
Brief description of vulnerability.

## Impact
What an attacker could achieve.

## Affected Components
- Component A (versions X.Y.Z)
- Component B (versions X.Y.Z)

## Remediation
- Upgrade to version 1.2.4 or later
- Apply workaround (if applicable)

## Credit
Security researcher name (if public disclosure approved)
```

### CVE Assignment Process

**CVE Numbering Authority**: Portalis is pursuing CNA status (Q2 2026).

**Current Process** (via GitHub):
1. Report vulnerability to security@portalis.dev
2. Portalis requests CVE via GitHub CNA
3. CVE assigned within 72 hours
4. CVE published with security advisory

**CVE Metadata**:
- CVSS v3.1 score
- CWE classification
- Affected products and versions
- References and credits

## Incident Response

### Incident Classification

**Severity Levels**:

| Level | Definition | Examples | Response Time |
|-------|------------|----------|---------------|
| **P0 - Critical** | Active exploitation, data breach | RCE exploit in production, customer data leaked | Immediate (15 min) |
| **P1 - High** | Confirmed vulnerability, high risk | Authentication bypass discovered, unpatched critical CVE | 1 hour |
| **P2 - Medium** | Potential security issue | Failed login attempts spike, suspicious API usage | 4 hours |
| **P3 - Low** | Security concern, low risk | Policy violation, expired certificate warning | 24 hours |

**Incident Types**:
- Intrusion (unauthorized access)
- Malware infection
- Data breach
- Denial of service
- Insider threat
- Supply chain compromise

### Response Procedures

**Incident Response Team**:
- **Incident Commander**: Security Lead
- **Technical Lead**: Engineering Manager
- **Communications Lead**: Product Manager
- **Legal**: General Counsel
- **On-call**: Rotating SRE

**Response Workflow**:

1. **Detection and Reporting** (0-15 minutes):
   - Automated monitoring alerts
   - Security team notification
   - Incident tracking ticket created

2. **Initial Assessment** (15-30 minutes):
   - Verify incident validity
   - Classify severity (P0-P3)
   - Assemble response team
   - Activate communication channels

3. **Containment** (30 minutes - 4 hours):
   ```bash
   # Example containment actions

   # Isolate affected pods
   kubectl cordon node-xyz
   kubectl drain node-xyz --ignore-daemonsets

   # Revoke compromised credentials
   kubectl delete secret compromised-api-key

   # Enable enhanced logging
   kubectl set env deployment/portalis LOG_LEVEL=DEBUG

   # Block malicious IPs
   kubectl apply -f emergency-network-policy.yaml
   ```

4. **Eradication** (4-24 hours):
   - Remove malware/backdoors
   - Patch vulnerabilities
   - Reset compromised credentials
   - Rebuild affected systems

5. **Recovery** (24-72 hours):
   - Restore services from clean backups
   - Verify system integrity
   - Monitor for reinfection
   - Gradual traffic restoration

6. **Post-Incident Activities** (1 week):
   - Root cause analysis
   - Post-mortem report
   - Process improvements
   - Security control updates

### Communication Plan

**Internal Communication**:
- **Slack**: #security-incidents (immediate)
- **Email**: security-team@portalis.dev
- **PagerDuty**: Automated escalation
- **Status Page**: https://status.portalis.dev

**External Communication**:

| Stakeholder | Notification Timing | Channel |
|-------------|---------------------|---------|
| Affected customers | Within 4 hours | Email, in-app notification |
| All customers | Within 24 hours (if widespread) | Email, status page |
| Public | Within 72 hours (if data breach) | Blog post, security advisory |
| Regulators | Within 72 hours (GDPR) | Official notification |

**Communication Templates**:
- Initial notification (facts only)
- Status updates (every 4 hours for P0)
- Resolution notification
- Post-mortem summary

### Post-Mortem Process

**Post-Mortem Document** (blameless):

```markdown
# Incident Post-Mortem: [Title]

**Date**: 2025-XX-XX
**Duration**: X hours
**Severity**: P0/P1/P2/P3
**Status**: Resolved

## Summary
One-paragraph summary of incident.

## Timeline
- 00:00 - Detection: Alert triggered
- 00:15 - Response: Team assembled
- 00:30 - Containment: Services isolated
- 04:00 - Eradication: Vulnerability patched
- 24:00 - Recovery: Services restored

## Root Cause
Technical explanation of what went wrong.

## Impact
- Affected users: X
- Duration: X hours
- Data exposed: Yes/No
- Revenue impact: $X

## What Went Well
- Positive aspects of response

## What Went Wrong
- Areas needing improvement

## Action Items
- [ ] Action 1 (Owner: @user, Due: 2025-XX-XX)
- [ ] Action 2 (Owner: @user, Due: 2025-XX-XX)

## Lessons Learned
Key takeaways for future incidents.
```

**Post-Mortem Meeting**:
- Scheduled within 1 week of resolution
- All stakeholders attend
- Blameless culture
- Focus on process improvements

## Compliance & Auditing

### Audit Logging

Portalis implements comprehensive audit logging using the structured logging framework in [core/src/logging.rs](/workspace/portalis/core/src/logging.rs).

**Audit Log Events**:

```rust
// AuditLogger usage examples
use portalis_core::logging::AuditLogger;

// Translation request
AuditLogger::log_translation_request(
    "user-123",
    "fibonacci.py",
    "192.168.1.100"
);

// Translation completion
AuditLogger::log_translation_complete(
    "user-123",
    "trans-456",
    true  // success
);

// Error events
AuditLogger::log_error(
    "user-123",
    "AuthenticationFailure",
    "Invalid API key provided"
);
```

**Audit Log Format** (JSON):
```json
{
  "timestamp": "2025-10-03T14:23:45.123Z",
  "event_type": "translation_request",
  "user_id": "user-123",
  "source_file": "fibonacci.py",
  "ip_address": "192.168.1.100",
  "trace_id": "trace-abc-123",
  "session_id": "sess-xyz-789",
  "user_agent": "portalis-cli/1.2.0",
  "result": "success"
}
```

**Audit Log Categories**:
- Authentication events (login, logout, failed attempts)
- Authorization events (permission grants, denials)
- Data access (read, write, delete)
- Configuration changes
- Administrative actions
- Security events (alerts, policy violations)

**Log Retention**:
- **Hot storage** (Elasticsearch): 90 days
- **Warm storage** (S3): 1 year
- **Cold storage** (S3 Glacier): 7 years (compliance)

**Log Integrity**:
- Logs signed with HMAC-SHA256
- Write-once, read-many (WORM) storage
- Tamper detection via checksums

### Access Controls

**Role-Based Access Control (RBAC)**:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: portalis-deployment
  name: portalis-developer
rules:
- apiGroups: [""]
  resources: ["pods", "pods/log"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: portalis-deployment
  name: portalis-admin
rules:
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["*"]
```

**Principle of Least Privilege**:
- Users granted minimum required permissions
- Time-limited elevated access (JIT)
- Regular access reviews (quarterly)

**Multi-Factor Authentication (MFA)**:
- Required for all admin access
- TOTP (Google Authenticator, Authy)
- Hardware tokens (YubiKey) for production

### Data Retention

**Retention Policies**:

| Data Type | Retention Period | Purpose | Storage |
|-----------|------------------|---------|---------|
| Translation requests | 30 days | Service delivery | PostgreSQL |
| Audit logs | 7 years | Compliance (SOC2) | S3 + Glacier |
| System metrics | 90 days | Monitoring | Prometheus |
| Application logs | 90 days | Debugging | Elasticsearch |
| Backups | 30 days | Disaster recovery | S3 |
| User data | Until deletion | Service provision | PostgreSQL |

**Automated Deletion**:
```bash
# Cron job for data retention enforcement
0 2 * * * /usr/local/bin/portalis-cleanup \
  --delete-translations-older-than 30d \
  --archive-logs-older-than 90d \
  --delete-temp-files
```

**Legal Holds**:
- Retention extended for litigation
- Flagged data excluded from automated deletion
- Documented in legal hold register

### Compliance Frameworks

**SOC 2 Type II**:
- **Trust Service Criteria**: Security, Availability, Confidentiality
- **Control Activities**: See [compliance.md](/workspace/portalis/docs/compliance.md)
- **Evidence**: Audit logs, access reviews, change logs
- **Status**: In progress (Q3 2026)

**ISO 27001**:
- **Information Security Management System (ISMS)**
- **Risk Assessment**: Annual review
- **Statement of Applicability (SOA)**: 114 controls
- **Status**: Planned (2026)

**GDPR**:
- **Data Processing Agreement (DPA)**: Available for customers
- **Privacy by Design**: Minimal data collection
- **Data Subject Rights**: Export, deletion, portability
- **Status**: Compliant

**NIST Cybersecurity Framework**:
- **Identify**: Asset inventory, risk assessment
- **Protect**: Access controls, encryption
- **Detect**: Monitoring, logging
- **Respond**: Incident response plan
- **Recover**: Backup and disaster recovery

**Compliance Artifacts**:
- Security policies and procedures
- Risk assessment reports
- Penetration test reports
- Audit logs and access reviews
- Vendor security questionnaires
- Security training records

## Security Best Practices

### Operational Security

1. **Keep Software Updated**:
   - Automated dependency updates (Dependabot)
   - Security patches applied within SLA
   - Kubernetes cluster auto-upgrade enabled

2. **Principle of Least Privilege**:
   - Minimal container capabilities
   - RBAC for all access
   - Time-limited elevated permissions

3. **Defense in Depth**:
   - Multiple security layers (WASM, container, network, cluster)
   - No single point of failure
   - Redundant security controls

4. **Input Validation**:
   - Whitelist allowed inputs
   - Sanitize all user data
   - Reject malformed requests early

5. **Secure Defaults**:
   - TLS enabled by default
   - Authentication required
   - Minimal exposed services

### Development Security

1. **Secure Coding Practices**:
   - Avoid unsafe Rust code
   - Use parameterized queries (no SQL injection)
   - Validate all inputs at API boundary

2. **Code Review**:
   - Security-focused code reviews
   - Automated SAST in CI/CD
   - Peer review required for all changes

3. **Dependency Management**:
   - Pin dependency versions
   - Regular updates for security patches
   - Audit third-party libraries

4. **Secrets Management**:
   - Never commit secrets to version control
   - Use environment variables or secret managers
   - Rotate secrets regularly

5. **Security Testing**:
   - Unit tests for security controls
   - Integration tests for authentication/authorization
   - Regular penetration testing

### Deployment Security

1. **Infrastructure as Code**:
   - Version-controlled infrastructure (Terraform, Kubernetes YAML)
   - Code review for infrastructure changes
   - Automated deployment pipelines

2. **Immutable Infrastructure**:
   - Rebuild containers instead of patching
   - No SSH access to production containers
   - Ephemeral workloads

3. **Monitoring and Alerting**:
   - Security metrics (failed logins, policy violations)
   - Real-time alerting (PagerDuty)
   - Log aggregation and analysis

4. **Backup and Disaster Recovery**:
   - Regular automated backups
   - Tested restore procedures
   - Offsite backup storage

5. **Access Control**:
   - MFA for all production access
   - VPN or bastion hosts for infrastructure access
   - Audit all privileged operations

## Reporting Security Issues

**DO NOT** create public GitHub issues for security vulnerabilities.

### Responsible Disclosure

We appreciate security researchers who report vulnerabilities responsibly. Please follow these guidelines:

1. **Do Not** exploit the vulnerability beyond proof of concept
2. **Do Not** access, modify, or delete customer data
3. **Do Not** perform destructive testing (DoS, data corruption)
4. **Do** provide detailed reproduction steps
5. **Do** allow reasonable time for remediation before public disclosure

### Contact Information

**Primary Contact**:
- **Email**: security@portalis.dev
- **PGP Key**: https://portalis.dev/security.asc (Fingerprint: XXXX XXXX XXXX XXXX)
- **Response Time**: Within 24 hours

**GitHub Security Advisories**:
- https://github.com/portalis/portalis/security/advisories
- Click "Report a vulnerability"
- Use for structured vulnerability reports

**Escalation** (if no response within 48 hours):
- **CTO**: cto@portalis.dev
- **CEO**: ceo@portalis.dev

### Bug Bounty Program

**Status**: Coming Q2 2026

**Planned Scope**:
- *.portalis.dev domains
- API endpoints (api.portalis.dev)
- Mobile applications
- Open-source repositories

**Rewards** (planned):
- **Critical**: $5,000 - $20,000
- **High**: $1,000 - $5,000
- **Medium**: $500 - $1,000
- **Low**: $100 - $500

**Out of Scope**:
- Social engineering
- Physical attacks
- Third-party services
- Denial of service

## Related Documentation

- **Compliance**: [compliance.md](/workspace/portalis/docs/compliance.md) - SOC2, GDPR, ISO 27001 compliance
- **Architecture**: architecture.md - System architecture and security boundaries
- **Deployment**: deployment/kubernetes.md - Secure deployment guidelines
- **Logging**: [core/src/logging.rs](/workspace/portalis/core/src/logging.rs) - Audit logging implementation
- **Workflows**: [.github/workflows/security.yml](/.github/workflows/security.yml) - Automated security scanning

## Acknowledgments

We thank the security research community for helping keep Portalis secure. Special thanks to:

- Security researchers who have responsibly disclosed vulnerabilities
- Open-source security tools (Trivy, cargo-audit, Dependabot)
- OWASP for security best practices guidance

---

**Document Version**: 2.0
**Last Updated**: 2025-10-03
**Next Review**: 2026-01-03
**Owner**: Security Team <security@portalis.dev>
