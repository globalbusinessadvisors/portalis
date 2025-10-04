# Compliance Guide

Compliance considerations for enterprise deployment of Portalis.

## Overview

Portalis is designed with enterprise compliance requirements in mind, supporting SOC2, GDPR, and other regulatory frameworks.

## SOC2 Compliance

### Security Principles

**CC6.1 - Logical Access Controls**:
- API key authentication
- Role-based access control (RBAC)
- Multi-factor authentication (MFA) support

**CC6.6 - Encryption**:
- TLS 1.3 for data in transit
- AES-256 for data at rest
- Key management via HashiCorp Vault

**CC7.2 - System Monitoring**:
- Comprehensive logging (Prometheus + Grafana)
- Real-time alerting
- Audit trail for all operations

**CC7.4 - Change Management**:
- Automated CI/CD with approval gates
- Version control (Git)
- Rollback procedures

### Audit Logging

All security-relevant events are logged:

```json
{
  "timestamp": "2025-10-03T12:34:56Z",
  "event_type": "translation.request",
  "user_id": "user-123",
  "api_key_id": "key-456",
  "source_ip": "192.168.1.1",
  "resource": "fibonacci.py",
  "action": "translate",
  "result": "success",
  "duration_ms": 315
}
```

**Retention**: 90 days minimum, configurable up to 7 years

### Evidence Collection

For SOC2 audits, Portalis provides:

1. **Access Logs**: All authentication attempts
2. **Change Logs**: All code deployments
3. **Security Scans**: Weekly Trivy scans
4. **Incident Reports**: Security incident documentation
5. **Training Records**: Security awareness training

## GDPR Compliance

### Data Protection Principles

**Article 5 - Lawfulness, Fairness, Transparency**:
- Clear privacy policy
- Explicit consent for data collection
- Transparent data processing

**Article 25 - Privacy by Design**:
- Minimal data collection
- Pseudonymization where possible
- Data minimization by default

**Article 32 - Security**:
- Encryption at rest and in transit
- Regular security testing
- Incident response procedures

### Data Subject Rights

**Right to Access** (Article 15):
```bash
# Export user data
portalis admin export-user-data --user-id <id> --format json
```

**Right to Erasure** (Article 17):
```bash
# Delete user data
portalis admin delete-user-data --user-id <id> --confirm
```

**Right to Data Portability** (Article 20):
- Export in JSON format
- Machine-readable
- Standard schemas

### Data Processing Agreement (DPA)

Available upon request for enterprise customers.

**Contact**: legal@portalis.dev

## Data Handling

### Data Classification

| Classification | Examples | Handling |
|----------------|----------|----------|
| Public | Documentation, marketing | No restrictions |
| Internal | Logs, metrics | Access controlled |
| Confidential | API keys, customer code | Encrypted |
| Restricted | Personal data, secrets | Strict access control |

### Data Retention

| Data Type | Retention Period | Purpose |
|-----------|------------------|---------|
| Translation requests | 30 days | Service delivery |
| Audit logs | 90 days (SOC2: 1 year) | Security auditing |
| User accounts | Until deletion | Service provision |
| Backups | 30 days | Disaster recovery |

### Data Deletion

**Automated deletion**:
- Translation data: 30 days
- Cache data: 24 hours
- Temporary files: Immediate

**User-initiated deletion**:
- Account deletion: Immediate
- Data export before deletion: Available

## Compliance Certifications

### Current Status

**In Progress**:
- [ ] SOC2 Type I (Q1 2026)
- [ ] SOC2 Type II (Q3 2026)
- [ ] GDPR compliance audit (Q2 2026)

**Planned**:
- [ ] ISO 27001 (2026)
- [ ] HIPAA compliance (2027, healthcare customers)

### Compliance Roadmap

**2026 Q1**:
- Complete SOC2 Type I certification
- GDPR compliance audit
- Privacy impact assessment

**2026 Q2**:
- SOC2 Type II observation period begins
- ISO 27001 preparation

**2026 Q3**:
- SOC2 Type II certification
- Regular compliance audits

## Regional Compliance

### US Compliance

**Data Residency**:
- US customer data stored in US region (AWS us-east-1)
- No data transfer outside US without consent

**Frameworks**:
- SOC2 Type II
- NIST Cybersecurity Framework
- FedRAMP (roadmap)

### EU Compliance

**Data Residency**:
- EU customer data stored in EU region (AWS eu-west-1)
- GDPR-compliant data processing

**Privacy Shield Replacement**:
- Standard Contractual Clauses (SCCs)
- EU representative appointed

### UK Compliance

**Post-Brexit**:
- UK GDPR compliance
- UK representative appointed
- Data adequacy provisions

## Incident Response

### Security Incident Procedure

1. **Detection**: Automated monitoring + manual reporting
2. **Containment**: Isolate affected systems
3. **Investigation**: Root cause analysis
4. **Remediation**: Fix vulnerabilities
5. **Notification**: Notify affected parties (72 hours for GDPR)
6. **Documentation**: Incident report

### Breach Notification

**GDPR Article 33**:
- Notification to supervisory authority within 72 hours
- Notification to data subjects if high risk

**Contact**: security@portalis.dev

## Vendor Management

### Subprocessors

| Vendor | Purpose | Location | Compliance |
|--------|---------|----------|------------|
| AWS | Cloud hosting | US/EU | SOC2, ISO 27001 |
| NVIDIA | GPU infrastructure | US | SOC2 |
| Prometheus | Monitoring | Self-hosted | N/A |

### Due Diligence

All vendors undergo:
- Security questionnaire
- Compliance verification
- Contract review
- Annual re-assessment

## Privacy Policy

**Key Points**:
- What data we collect
- How we use data
- Data retention periods
- User rights
- Contact information

**Full policy**: https://portalis.dev/privacy

## Terms of Service

**Key Sections**:
- Service description
- User obligations
- Data usage
- Liability limitations
- Dispute resolution

**Full terms**: https://portalis.dev/terms

## Compliance Resources

### Documentation

- Privacy Policy: https://portalis.dev/privacy
- Terms of Service: https://portalis.dev/terms
- DPA Template: Available on request
- Security Whitepaper: https://portalis.dev/security

### Support

- Compliance Questions: compliance@portalis.dev
- Security Issues: security@portalis.dev
- Privacy Concerns: privacy@portalis.dev
- Legal Inquiries: legal@portalis.dev

## See Also

- [Security Guide](security.md)
- [Deployment Guide](deployment/kubernetes.md)
- [Architecture Overview](architecture.md)
