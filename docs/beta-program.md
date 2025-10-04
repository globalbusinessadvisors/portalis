# PORTALIS BETA PROGRAM GUIDE

**Version**: 1.0
**Date**: October 2025
**Status**: Active Enrollment

---

## Program Overview

The Portalis Beta Program provides early access to our GPU-accelerated Python-to-Rust-to-WASM translation platform. Beta participants will help validate real-world performance, identify edge cases, and shape the product roadmap before general availability.

### What is Portalis?

Portalis is a production-ready AI-powered code translation platform that converts Python applications to high-performance Rust code, compiled to WebAssembly (WASM). The platform leverages NVIDIA's complete GPU stack for unprecedented performance:

- **2-3x faster** translation than CPU-based approaches
- **NVIDIA NeMo** for intelligent code translation
- **CUDA acceleration** for parallel AST parsing
- **Triton Inference Server** for scalable model serving
- **DGX Cloud** for distributed processing
- **Omniverse integration** for WASM runtime validation

---

## Program Goals

### For Participants

1. **Early Access**: Get exclusive access to cutting-edge translation technology
2. **Performance Benefits**: Experience 2-3x faster Python→Rust→WASM conversions
3. **Cost Savings**: Reduce translation costs by up to 92% ($0.008 per translation)
4. **Direct Influence**: Shape product features and roadmap
5. **Technical Support**: Dedicated engineering support during beta period
6. **Case Study Opportunity**: Potential to be featured as a success story

### For Portalis

1. **Real-World Validation**: Test platform with diverse Python codebases
2. **Performance Tuning**: Optimize for actual user workloads
3. **Feature Prioritization**: Understand critical customer needs
4. **Quality Assurance**: Identify edge cases and bugs before GA
5. **Documentation Improvement**: Refine guides based on user feedback
6. **Success Stories**: Build case studies for marketing

---

## Participation Requirements

### Technical Requirements

**Minimum Requirements**:
- Python codebase: 1,000-100,000 lines of code
- Python version: 3.8+ (3.9, 3.10, 3.11 supported)
- Willingness to test translations with production workloads
- Ability to provide performance benchmarks

**Recommended Requirements**:
- CI/CD pipeline for integration
- Test suite for validation (pytest, unittest)
- Access to NVIDIA GPU for local testing (optional)
- Docker/Kubernetes environment (for containerized deployments)

### Organizational Requirements

**Commitment**:
- Dedicated technical contact (min 5 hours/week)
- Executive sponsor for decision-making
- Willingness to provide detailed feedback
- Participation in weekly check-ins (30 min)
- Minimum 3-month engagement

**Resources**:
- 1-2 engineers for integration and testing
- QA resource for validation
- Infrastructure for deployment (cloud or on-premise)

### Legal Requirements

- Signed Beta Agreement (NDA + Terms of Service)
- Acceptance of early-access limitations
- Permission to use anonymized metrics
- Optional: Permission for public case study

---

## Program Benefits

### What You Get

**Technology Access**:
- ✅ Full Portalis platform access (all features)
- ✅ NVIDIA GPU-accelerated translation
- ✅ Unlimited translations during beta period
- ✅ API access + CLI tools + Web UI
- ✅ Docker/Kubernetes deployment packages
- ✅ Complete documentation and guides

**Support & Services**:
- ✅ Dedicated Slack/Discord channel
- ✅ Weekly office hours with engineering team
- ✅ Priority bug fixes (24-48 hour SLA)
- ✅ Custom integration assistance
- ✅ Performance optimization consultation
- ✅ Migration planning support

**Early Adopter Advantages**:
- ✅ 50% discount on first 12 months post-GA
- ✅ Feature request prioritization
- ✅ Beta participant badge/recognition
- ✅ Early access to new capabilities
- ✅ Case study co-marketing opportunity

### What We Ask

**Feedback & Data**:
- Regular feedback on functionality and performance
- Bug reports with reproducible test cases
- Feature requests with business justification
- Anonymous usage metrics and telemetry
- Benchmark data for performance validation

**Participation**:
- Weekly 30-minute check-in calls
- Monthly retrospective meetings
- Completion of feedback surveys
- Testing of new features/releases
- Documentation review and suggestions

**Success Metrics**:
- Minimum 10 successful translations
- At least 1 production integration
- Complete feedback survey
- Participate in case study (optional)

---

## Support Channels

### Primary Support

**Beta Support Slack Channel**:
- **URL**: portalis-beta.slack.com
- **Response Time**: <4 hours (business hours)
- **Available**: Mon-Fri, 9 AM - 6 PM PT
- **Channels**:
  - `#beta-general` - General discussion
  - `#beta-technical` - Technical issues
  - `#beta-features` - Feature requests
  - `#beta-performance` - Performance optimization

**Email Support**:
- **Address**: beta-support@portalis.ai
- **Response Time**: <24 hours
- **For**: Non-urgent issues, documentation questions

### Escalation Path

**Level 1: Community Support** (Slack)
- Beta participants + community managers
- Response: <4 hours
- For: General questions, how-to, best practices

**Level 2: Engineering Support** (Ticket)
- Senior engineers assigned to beta program
- Response: <24 hours
- For: Technical issues, integration problems, bugs

**Level 3: Critical Issues** (Phone/Video)
- Engineering leadership + product team
- Response: <2 hours
- For: Production blockers, critical bugs, security issues

**Emergency Hotline**: +1 (555) 123-BETA
- 24/7 availability for critical production issues

---

## Feedback Process

### How to Provide Feedback

**1. Weekly Check-Ins** (Structured)
- 30-minute video call every Monday
- Review previous week's progress
- Discuss current issues and blockers
- Plan upcoming tests and integrations

**2. Feedback Surveys** (Monthly)
- Comprehensive product feedback survey
- NPS (Net Promoter Score) measurement
- Feature prioritization voting
- Documentation quality assessment

**3. Bug Reports** (As Needed)
- GitHub Issues: github.com/portalis/beta-issues
- Include: reproduction steps, logs, environment
- Expected vs actual behavior
- Screenshots/recordings if applicable

**4. Feature Requests** (As Needed)
- Product feedback board: feedback.portalis.ai
- Business justification required
- Use case description
- Priority ranking (P0-P3)

### Feedback Categories

**Critical** (P0):
- Production blockers
- Data loss or corruption
- Security vulnerabilities
- Incorrect translations causing runtime errors

**High** (P1):
- Significant performance degradation
- Missing critical features
- Poor documentation
- Integration issues

**Medium** (P2):
- Minor bugs or issues
- Enhancement requests
- Documentation improvements
- UX/UI suggestions

**Low** (P3):
- Nice-to-have features
- Cosmetic issues
- Minor optimizations

---

## Success Metrics

### Translation Quality Metrics

**Target: >90% Success Rate**

**Measured By**:
- Successful compilation rate (Rust builds)
- WASM validation pass rate
- Runtime correctness (test suite pass rate)
- Type inference accuracy
- API compatibility preservation

**Tracking**:
- Automated metrics in dashboard
- Weekly quality reports
- Anomaly detection and alerts

### Customer Satisfaction Metrics

**Target: >4.0/5.0 Average**

**Measured By**:
- Overall satisfaction rating (1-5 scale)
- Net Promoter Score (NPS): >30
- Feature satisfaction (per-feature ratings)
- Documentation clarity (1-5 scale)
- Support responsiveness (1-5 scale)

**Collection**:
- Monthly satisfaction surveys
- Post-translation feedback
- Quarterly NPS surveys

### Support Metrics

**Target: <24 Hour Response Time**

**Measured By**:
- First response time (P95 <4 hours)
- Resolution time by severity:
  - P0 (Critical): <2 hours
  - P1 (High): <24 hours
  - P2 (Medium): <72 hours
  - P3 (Low): <1 week
- Support ticket volume trends

**Tracking**:
- Zendesk/Freshdesk metrics
- Weekly support reports

### Quality Metrics

**Target: 0 Critical Bugs**

**Measured By**:
- Critical bug count (must be 0)
- Major bug count (target <5)
- Minor bug count (target <20)
- Time to resolution by severity
- Regression rate (<5%)

**Tracking**:
- GitHub Issues dashboard
- Weekly bug triage meetings
- Automated test suite results

### Documentation Metrics

**Target: 100% Completeness**

**Measured By**:
- API coverage: 100% of endpoints documented
- Feature coverage: All features have guides
- Tutorial completeness: End-to-end workflows
- FAQ coverage: All common questions answered
- Code examples: All major use cases covered

**Validation**:
- Documentation review checklist
- User feedback on clarity
- Time to first successful translation

### Platform Performance Metrics

**Target: Meet/Exceed SLAs**

**Measured By**:
- Translation latency (P95 <500ms for 100 LOC)
- Throughput (>200 req/s batch processing)
- System uptime (>99.5%)
- GPU utilization (>70%)
- Cost per translation (<$0.01)

**Tracking**:
- Prometheus metrics
- Grafana dashboards
- Weekly performance reports

---

## Timeline and Milestones

### Program Duration: 12 Weeks (3 Months)

#### Phase 1: Onboarding (Weeks 1-2)

**Week 1: Setup**
- Beta agreement signed
- Account provisioning
- Initial training session
- Environment setup
- First test translation

**Week 2: Integration**
- API integration complete
- CI/CD pipeline configured
- First production-like translation
- Performance baseline established
- Feedback process confirmed

**Milestone**: First successful translation ✅

---

#### Phase 2: Active Testing (Weeks 3-8)

**Week 3-4: Core Features**
- Test basic translation workflows
- Validate type inference
- Test build pipeline
- Evaluate documentation
- Report initial findings

**Week 5-6: Advanced Features**
- GPU acceleration testing
- Batch processing validation
- NIM microservice deployment
- Performance optimization
- Integration edge cases

**Week 7-8: Production Simulation**
- Production workload testing
- Stress testing
- Failure scenario validation
- Monitoring and alerting
- Cost analysis

**Milestone**: Production-ready integration ✅

---

#### Phase 3: Optimization (Weeks 9-10)

**Week 9: Performance Tuning**
- Benchmark analysis
- Optimization implementation
- A/B testing
- Cost optimization
- SLA validation

**Week 10: Final Validation**
- Full regression testing
- Documentation validation
- Case study preparation
- Success metrics review
- Final feedback collection

**Milestone**: Performance targets met ✅

---

#### Phase 4: Graduation (Weeks 11-12)

**Week 11: Transition Planning**
- GA readiness assessment
- Production deployment plan
- Pricing discussion
- Contract negotiation
- Case study finalization

**Week 12: Program Completion**
- Final retrospective
- Lessons learned documentation
- Beta graduation ceremony
- Early adopter benefits activation
- Ongoing support transition

**Milestone**: Beta graduation to production ✅

---

## Program Structure

### Governance

**Beta Program Lead**:
- Overall program management
- Stakeholder communication
- Success metrics tracking
- Escalation management

**Technical Lead**:
- Platform stability and performance
- Bug triage and prioritization
- Technical support escalation
- Architecture decisions

**Product Manager**:
- Feature prioritization
- Roadmap alignment
- Requirements gathering
- User experience optimization

### Communication Cadence

**Daily**:
- Slack channel monitoring
- Ticket response
- Emergency escalations

**Weekly**:
- 30-min check-in calls (per participant)
- Engineering office hours (1 hour)
- Bug triage meeting (internal)
- Metrics review (internal)

**Monthly**:
- Retrospective meeting (1 hour)
- Satisfaction survey
- Progress report to participants
- Executive stakeholder update

**Quarterly**:
- In-person/virtual summit
- Roadmap review
- Strategic planning
- Networking event

---

## Enrollment Process

### How to Apply

**Step 1: Application Submission**
- Complete application form: apply.portalis.ai/beta
- Provide company and technical details
- Describe use case and codebase
- Submit Python codebase sample (optional)

**Step 2: Review & Selection**
- Application review (3-5 business days)
- Technical fit assessment
- Use case evaluation
- Capacity check

**Step 3: Acceptance & Onboarding**
- Acceptance notification
- Beta agreement signing
- Account provisioning
- Welcome kit delivery
- Onboarding session scheduled

**Step 4: Kickoff**
- Initial training session
- Environment setup
- First translation walkthrough
- Support channel access
- Success metrics baseline

### Selection Criteria

**We Look For**:
- Diverse Python codebases (size, complexity, domain)
- Active development teams (committed participation)
- Production use cases (real-world validation)
- Technical sophistication (can provide detailed feedback)
- Strategic fit (industries we're targeting)

**Ideal Participants**:
- FinTech, Data Science, ML/AI, SaaS companies
- 10-500 employee companies
- Active Python development (5+ engineers)
- Cloud-native or containerized deployments
- Performance-sensitive applications

---

## Frequently Asked Questions

### General Questions

**Q: Is the beta program free?**
A: Yes, beta access is completely free during the program period. You'll receive a 50% discount for 12 months after GA.

**Q: How long does the beta program last?**
A: The program runs for 12 weeks (3 months) per participant. You can extend if needed.

**Q: What happens after beta?**
A: You transition to a production subscription with early adopter pricing (50% off for 12 months).

**Q: Can we use Portalis in production during beta?**
A: Yes, with the understanding that it's pre-GA software. We recommend thorough testing and gradual rollout.

### Technical Questions

**Q: Do we need NVIDIA GPUs?**
A: No, the platform works on CPU-only environments. GPU acceleration is optional for better performance.

**Q: What Python versions are supported?**
A: Python 3.8, 3.9, 3.10, and 3.11 are fully supported. Python 3.7 and 3.12 are in development.

**Q: Can we translate existing production code?**
A: Yes, that's encouraged. We recommend starting with non-critical modules and gradually expanding.

**Q: What if translations fail?**
A: We provide detailed error messages and fallback options. Our support team will help resolve any issues.

### Support Questions

**Q: What's the SLA for bug fixes?**
A: Critical bugs: <2 hours response, <24 hours resolution. High priority: <24 hours response, <72 hours resolution.

**Q: Can we get custom features developed?**
A: Feature requests are prioritized based on beta participant needs. We'll work with you on critical requirements.

**Q: Is there a dedicated support engineer?**
A: Yes, each beta participant gets a designated customer success engineer and access to the engineering team.

---

## Contact Information

### Program Management

**Beta Program Director**:
Sarah Chen
Email: sarah.chen@portalis.ai
Phone: +1 (555) 123-4567

**Technical Program Manager**:
Alex Rodriguez
Email: alex.rodriguez@portalis.ai
Phone: +1 (555) 234-5678

### Support Contacts

**Beta Support Email**: beta-support@portalis.ai
**Emergency Hotline**: +1 (555) 123-BETA
**Slack Workspace**: portalis-beta.slack.com
**Application Portal**: apply.portalis.ai/beta

### Online Resources

**Documentation**: docs.portalis.ai/beta
**API Reference**: api.portalis.ai/docs
**Feedback Portal**: feedback.portalis.ai
**Status Page**: status.portalis.ai
**Community Forum**: community.portalis.ai

---

## Appendix: Beta Agreement Summary

### Key Terms

**Duration**: 12 weeks, renewable by mutual agreement

**License**: Non-exclusive, non-transferable beta license

**Support**: Business hours support with escalation for critical issues

**Data**: Anonymous usage metrics collected, no code stored without permission

**IP**: You retain all rights to your code and translations

**Confidentiality**: NDA covers platform internals, not translated output

**Termination**: Either party can terminate with 7 days notice

**Liability**: Limited liability for beta software, production use at your discretion

### Rights & Obligations

**Your Rights**:
- Use Portalis for internal development and testing
- Provide feedback and feature requests
- Access all platform features and updates
- Receive dedicated support
- Opt-out of case study participation

**Your Obligations**:
- Provide regular feedback
- Report bugs and issues
- Participate in scheduled check-ins
- Protect confidential information
- Use platform in good faith

**Our Rights**:
- Modify platform and features
- Use anonymous feedback and metrics
- Discontinue beta access if terms violated
- Feature your success story (with permission)

**Our Obligations**:
- Provide stable, functional platform
- Deliver committed support SLAs
- Protect your data and IP
- Act on critical feedback
- Provide advance notice of changes

---

**Last Updated**: October 2025
**Version**: 1.0
**Status**: Active

For the latest version of this guide, visit: docs.portalis.ai/beta-program

---

*Join the Portalis Beta Program and be part of the future of code translation!*
