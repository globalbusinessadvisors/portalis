# PORTALIS BETA CUSTOMER ONBOARDING CHECKLIST

**Version**: 1.0
**Last Updated**: October 2025

Welcome to the Portalis Beta Program! This checklist will guide you through the onboarding process to ensure a smooth start with our GPU-accelerated Python-to-Rust-to-WASM translation platform.

---

## Pre-Onboarding (Before Day 1)

### Administrative Setup

- [ ] **Beta Agreement Signed**
  - Review and sign Beta Participation Agreement
  - Review and sign NDA (Non-Disclosure Agreement)
  - Complete Terms of Service acceptance
  - Verify authorized signatories

- [ ] **Team Identified**
  - Designate technical lead (primary contact)
  - Assign 1-2 integration engineers
  - Identify QA/testing resource
  - Assign executive sponsor
  - Collect team contact information

- [ ] **Communication Channels**
  - Provide email addresses for Slack invites
  - Confirm phone numbers for emergency contacts
  - Set up distribution list for announcements
  - Verify time zones for meeting scheduling

- [ ] **Environment Assessment**
  - Document current Python codebase details:
    - Total lines of code
    - Python version(s) in use
    - Key dependencies and frameworks
    - Test coverage percentage
  - Identify translation pilot project (1K-10K LOC)
  - Confirm infrastructure availability (cloud/on-prem)
  - Check GPU availability (optional but recommended)

---

## Day 1: Account Setup & Access

### Account Provisioning

- [ ] **Portalis Account Creation**
  - Receive account creation email
  - Set up account credentials
  - Configure two-factor authentication (2FA)
  - Verify email address
  - Complete profile setup

- [ ] **Access Verification**
  - Log in to Portalis web dashboard: https://app.portalis.ai
  - Access API credentials page
  - Generate API key (save securely)
  - Access documentation portal: https://docs.portalis.ai
  - Bookmark key resources

- [ ] **Team Access**
  - Invite team members to organization
  - Assign appropriate roles:
    - Admin (1-2 people)
    - Developer (engineers)
    - Viewer (stakeholders)
  - Verify all team members can access platform

### Communication Setup

- [ ] **Slack Workspace**
  - Accept Slack invite to portalis-beta.slack.com
  - Join channels:
    - `#beta-general` (announcements, general discussion)
    - `#beta-technical` (technical support)
    - `#beta-features` (feature requests)
    - `#beta-performance` (optimization discussions)
  - Introduce your team in `#beta-general`
  - Test direct message with Customer Success Engineer

- [ ] **Support Channels**
  - Save beta support email: beta-support@portalis.ai
  - Save emergency hotline: +1 (555) 123-BETA
  - Access GitHub Issues: github.com/portalis/beta-issues
  - Bookmark feedback portal: feedback.portalis.ai

- [ ] **Meeting Scheduling**
  - Schedule recurring weekly check-in (30 min)
  - Add to calendar: Engineering Office Hours (Fridays 2-3 PM PT)
  - Schedule initial onboarding session (2 hours)
  - Confirm monthly retrospective time slot

---

## Week 1: Installation & First Translation

### Environment Setup

- [ ] **CLI Installation**
  ```bash
  # Install Portalis CLI
  pip install portalis-cli

  # Verify installation
  portalis --version

  # Configure with API key
  portalis configure --api-key <your-key>

  # Test connection
  portalis auth test
  ```

- [ ] **Docker Setup (Optional)**
  ```bash
  # Pull Portalis Docker image
  docker pull portalis/translator:beta

  # Verify image
  docker run portalis/translator:beta --version

  # Test container
  docker run -v $(pwd):/workspace portalis/translator:beta translate --help
  ```

- [ ] **IDE Integration (Optional)**
  - Install VS Code extension: "Portalis Translator"
  - Or PyCharm plugin: "Portalis Integration"
  - Configure plugin with API key
  - Test translation from IDE

### Installation Verification

- [ ] **System Requirements Check**
  ```bash
  # Check Python version
  python --version  # Should be 3.8+

  # Check available memory
  free -h  # Recommended: 8GB+ RAM

  # Check disk space
  df -h  # Need 10GB+ free

  # Check GPU (if available)
  nvidia-smi  # Optional but beneficial
  ```

- [ ] **Dependencies Verification**
  ```bash
  # Verify Rust toolchain (for local builds)
  rustc --version  # Should be 1.70+
  cargo --version

  # Verify WASM tools
  wasm-pack --version

  # Verify Docker (for containerized deployments)
  docker --version
  docker-compose --version
  ```

- [ ] **Network Configuration**
  - Verify outbound HTTPS access (port 443)
  - Whitelist API endpoint: api.portalis.ai
  - Whitelist model serving: models.portalis.ai
  - Configure proxy settings if needed
  - Test API connectivity:
    ```bash
    curl -H "Authorization: Bearer <api-key>" https://api.portalis.ai/v1/health
    ```

### First Translation Walkthrough

- [ ] **Prepare Sample Code**
  - Select simple Python file (50-100 LOC)
  - Ensure file has comprehensive tests
  - Document expected behavior
  - Create backup of original file

- [ ] **Execute Translation (CLI)**
  ```bash
  # Translate single file
  portalis translate input.py --output output.rs

  # Verify generated Rust code
  cat output.rs

  # Build Rust to WASM
  portalis build output.rs --target wasm32-wasi

  # Validate WASM output
  portalis validate output.wasm
  ```

- [ ] **Execute Translation (API)**
  ```python
  from portalis import PortalisClient

  # Initialize client
  client = PortalisClient(api_key="<your-key>")

  # Translate code
  result = client.translate_file("input.py")

  # Save output
  with open("output.rs", "w") as f:
      f.write(result.rust_code)

  # Build to WASM
  wasm_result = client.build_wasm(result.rust_code)

  # Validate
  validation = client.validate_wasm(wasm_result.wasm_binary)
  print(f"Valid: {validation.is_valid}")
  ```

- [ ] **Review Translation Quality**
  - Compare input Python and output Rust
  - Verify type mappings are correct
  - Check API compatibility
  - Review generated documentation
  - Note any issues or concerns

- [ ] **Test Translated Code**
  ```bash
  # Run Rust tests
  cargo test --manifest-path output/Cargo.toml

  # Run WASM in Node.js
  node test-wasm.js

  # Compare behavior with Python original
  python test_original.py
  diff python_output.txt wasm_output.txt
  ```

- [ ] **Document Results**
  - Translation success: Yes/No
  - Build success: Yes/No
  - Tests passing: X/Y
  - Performance comparison: X ms (Python) vs Y ms (WASM)
  - Issues encountered: [list]
  - Questions raised: [list]

---

## Week 1: Assessment Tool Usage

### Feature Detection

- [ ] **Run Feature Analyzer**
  ```bash
  # Analyze Python codebase
  portalis assess analyze --path /path/to/project

  # Review detected features
  portalis assess report --format json > features.json

  # Check compatibility score
  portalis assess compatibility --threshold 0.8
  ```

- [ ] **Review Feature Report**
  - Total Python features detected: ____
  - Features fully supported: ____
  - Features partially supported: ____
  - Features not supported: ____
  - Compatibility score: ____%

- [ ] **Identify Gaps**
  - List unsupported features: [____]
  - Assess impact on translation: High/Medium/Low
  - Plan workarounds or alternatives
  - Submit feature requests if needed

### Complexity Assessment

- [ ] **Run Complexity Analysis**
  ```bash
  # Analyze code complexity
  portalis assess complexity --path /path/to/project

  # Get detailed metrics
  portalis assess metrics --include-all
  ```

- [ ] **Review Complexity Metrics**
  - Cyclomatic complexity: ____
  - Lines of code: ____
  - Number of functions/classes: ____
  - Dependency count: ____
  - Estimated translation time: ____

- [ ] **Risk Assessment**
  - Translation risk level: Low/Medium/High
  - Identify high-risk modules
  - Plan phased translation approach
  - Define success criteria

### Performance Estimation

- [ ] **Run Performance Predictor**
  ```bash
  # Estimate performance gains
  portalis assess performance --path /path/to/project

  # Compare CPU vs GPU acceleration
  portalis assess performance --mode comparison
  ```

- [ ] **Review Estimates**
  - Estimated speedup (CPU): ____x
  - Estimated speedup (GPU): ____x
  - Memory reduction: ____%
  - Cost per translation: $____
  - Total translation time: ____ minutes

---

## Week 2: Migration Planning

### Migration Strategy

- [ ] **Define Translation Phases**
  - **Phase 1**: Pilot modules (1-2 weeks)
    - Modules: [list]
    - Success criteria: [define]
  - **Phase 2**: Core functionality (3-4 weeks)
    - Modules: [list]
    - Success criteria: [define]
  - **Phase 3**: Full migration (timeline: ____)
    - Modules: [list]
    - Success criteria: [define]

- [ ] **Risk Mitigation Plan**
  - Identify critical dependencies
  - Plan for unsupported features
  - Define rollback procedures
  - Set up parallel testing environment
  - Document fallback strategies

- [ ] **Resource Planning**
  - Engineering time allocated: ____ hours
  - QA time allocated: ____ hours
  - Infrastructure costs: $____/month
  - Timeline: ____ weeks
  - Key milestones: [list]

### Integration Planning

- [ ] **CI/CD Integration**
  - Identify CI/CD platform (Jenkins, GitHub Actions, GitLab CI)
  - Plan Portalis integration points
  - Define automated testing strategy
  - Set up deployment pipeline
  - Configure rollback mechanisms

- [ ] **Testing Strategy**
  - Unit tests: Translate and adapt
  - Integration tests: Define WASM test harness
  - Performance tests: Benchmark suite
  - Regression tests: Automated comparison
  - Acceptance tests: Production validation

- [ ] **Deployment Strategy**
  - Deployment target: Kubernetes/Docker/Serverless
  - Scaling strategy: Auto-scaling policies
  - Monitoring: Metrics and alerting
  - Logging: Centralized log aggregation
  - Backup & recovery: DR plan

### Success Metrics Definition

- [ ] **Technical Metrics**
  - Translation success rate target: ____%
  - Build success rate target: ____%
  - Test pass rate target: ____%
  - Performance improvement target: ____x
  - Code coverage target: ____%

- [ ] **Business Metrics**
  - Cost reduction target: ____%
  - Time to market improvement: ____%
  - Developer productivity gain: ____%
  - Operational cost savings: $____/month
  - ROI timeline: ____ months

- [ ] **Quality Metrics**
  - Bug density target: < ____ bugs/1K LOC
  - Critical bug tolerance: 0
  - Security vulnerability target: 0
  - Performance regression tolerance: < ____%
  - Uptime target: ____%

---

## Week 2: Support Contact Information

### Your Customer Success Team

- [ ] **Primary Contacts Documented**
  - **Customer Success Engineer**: ____________
    - Email: ____________
    - Slack: @____________
    - Phone: ____________

  - **Technical Account Manager**: ____________
    - Email: ____________
    - Slack: @____________
    - Phone: ____________

  - **Product Manager**: ____________
    - Email: ____________
    - Slack: @____________

- [ ] **Escalation Path Confirmed**
  - **Level 1**: Community support (Slack, <4 hours)
  - **Level 2**: Engineering support (Ticket, <24 hours)
  - **Level 3**: Critical escalation (Phone, <2 hours)
  - **Emergency**: Hotline +1 (555) 123-BETA (24/7)

- [ ] **Communication Preferences Set**
  - Preferred contact method: ____________
  - Notification preferences: ____________
  - Meeting schedule confirmed: ____________
  - Time zone: ____________

### Support Resources

- [ ] **Documentation Bookmarked**
  - [ ] Getting Started Guide: docs.portalis.ai/getting-started
  - [ ] API Reference: api.portalis.ai/docs
  - [ ] CLI Documentation: docs.portalis.ai/cli
  - [ ] Best Practices: docs.portalis.ai/best-practices
  - [ ] Troubleshooting Guide: docs.portalis.ai/troubleshooting
  - [ ] FAQ: docs.portalis.ai/faq

- [ ] **Tools & Utilities**
  - [ ] Beta Dashboard: app.portalis.ai/beta
  - [ ] Metrics Dashboard: metrics.portalis.ai
  - [ ] Status Page: status.portalis.ai
  - [ ] Feedback Portal: feedback.portalis.ai
  - [ ] Community Forum: community.portalis.ai

- [ ] **Training Resources**
  - [ ] Video tutorials viewed: ____/10
  - [ ] Sample projects cloned: ____/3
  - [ ] Documentation review: ____%
  - [ ] Best practices guide read: Yes/No

---

## Ongoing Activities (Weekly)

### Week 3+ Checklist

- [ ] **Weekly Check-in Preparation**
  - Document previous week's progress
  - List current blockers and issues
  - Prepare questions for engineering team
  - Review metrics and KPIs
  - Update project status

- [ ] **Continuous Integration**
  - Run daily translations (if applicable)
  - Monitor build success rates
  - Track performance metrics
  - Review error logs
  - Update documentation

- [ ] **Feedback Submission**
  - Submit bug reports as discovered
  - Provide feature requests with context
  - Complete weekly satisfaction survey
  - Participate in monthly retrospectives
  - Share success stories

- [ ] **Knowledge Building**
  - Review new documentation releases
  - Attend office hours (Fridays 2-3 PM PT)
  - Participate in beta community discussions
  - Share learnings with internal team
  - Document internal best practices

---

## Milestone Checklist

### End of Week 2 Milestone

- [ ] **Onboarding Complete**
  - All accounts and access configured
  - First successful translation completed
  - Assessment tools used and understood
  - Migration plan documented
  - Support contacts established
  - Team trained on basic usage

- [ ] **Readiness Validation**
  - Technical environment validated
  - Integration points identified
  - Success metrics defined
  - Risks documented and mitigated
  - Timeline established
  - Stakeholder alignment confirmed

### End of Week 4 Milestone

- [ ] **Pilot Complete**
  - First module successfully translated
  - Tests passing for translated code
  - Performance benchmarks completed
  - Issues documented and resolved
  - Lessons learned captured
  - Plan adjusted based on findings

### End of Week 8 Milestone

- [ ] **Production Integration**
  - Core modules translated
  - CI/CD pipeline operational
  - Monitoring and alerting configured
  - Performance targets met
  - Quality metrics achieved
  - Team fully proficient

### End of Week 12 Milestone

- [ ] **Beta Graduation**
  - All planned modules translated
  - Production deployment successful
  - Success metrics exceeded
  - Case study completed
  - Feedback provided
  - Ready for GA transition

---

## Emergency Procedures

### Critical Issue Response

If you encounter a **production-blocking issue**:

1. **Immediate Actions**:
   - [ ] Call emergency hotline: +1 (555) 123-BETA
   - [ ] Post in #beta-technical with @here mention
   - [ ] Email: beta-support@portalis.ai with [CRITICAL] tag
   - [ ] Document issue details: symptoms, impact, timeline

2. **Information to Provide**:
   - [ ] Description of the issue
   - [ ] Impact on your operations
   - [ ] Steps to reproduce
   - [ ] Error messages and logs
   - [ ] Environment details
   - [ ] Temporary workarounds attempted

3. **Escalation Timeline**:
   - [ ] 0-30 min: Initial response from on-call engineer
   - [ ] 30-60 min: Senior engineer assigned
   - [ ] 60-120 min: Root cause analysis begins
   - [ ] 2-4 hours: Temporary mitigation provided
   - [ ] 4-24 hours: Permanent fix deployed

### Rollback Procedures

If you need to **rollback a translation**:

1. **Immediate Rollback**:
   ```bash
   # Revert to Python version
   portalis rollback --deployment <deployment-id>

   # Or manual rollback
   kubectl rollout undo deployment/your-service
   docker-compose down && docker-compose -f docker-compose.old.yml up
   ```

2. **Document Rollback**:
   - [ ] Reason for rollback
   - [ ] Issue encountered
   - [ ] Impact assessment
   - [ ] Notify support team
   - [ ] Plan remediation

---

## Completion Sign-off

### Onboarding Certification

**Beta Customer**: _________________________ (Company)

**Technical Lead**: _________________________ (Name)

**Date Completed**: _________________________

**Portalis Customer Success Engineer**: _________________________

**Signature**: _________________________ **Date**: _____________

---

### Onboarding Status

- [ ] **Week 1 Complete**: Account setup, first translation successful
- [ ] **Week 2 Complete**: Assessment done, migration plan created
- [ ] **Ready for Production Testing**: All prerequisites met

**Next Steps**:
1. Begin pilot module translation (Week 3)
2. Schedule first retrospective (Week 4)
3. Plan production integration (Weeks 5-8)

---

## Appendix: Quick Reference

### Essential Commands

```bash
# Authentication
portalis auth login
portalis auth test

# Translation
portalis translate <file.py> --output <file.rs>
portalis build <file.rs> --target wasm32-wasi
portalis validate <file.wasm>

# Assessment
portalis assess analyze --path <project>
portalis assess compatibility --threshold 0.8
portalis assess performance --mode gpu

# Deployment
portalis deploy --target kubernetes
portalis deploy --target docker
portalis status

# Monitoring
portalis metrics --live
portalis logs --follow
portalis health
```

### Common Issues & Solutions

**Issue**: API authentication fails
**Solution**: Regenerate API key in dashboard, update configuration

**Issue**: Translation fails with type errors
**Solution**: Run `portalis assess analyze` to check compatibility

**Issue**: WASM validation fails
**Solution**: Ensure Rust toolchain is up to date, check build logs

**Issue**: Performance below expectations
**Solution**: Enable GPU acceleration, adjust batch size, contact support

---

## Support Contacts

**Beta Support Email**: beta-support@portalis.ai
**Emergency Hotline**: +1 (555) 123-BETA
**Slack**: portalis-beta.slack.com
**Documentation**: docs.portalis.ai
**Status**: status.portalis.ai

---

**Welcome to Portalis Beta!** 🚀

We're excited to have you on board. If you have any questions or need assistance at any point, don't hesitate to reach out. Our team is here to ensure your success!

---

**Version**: 1.0
**Last Updated**: October 2025
**Next Review**: Monthly
