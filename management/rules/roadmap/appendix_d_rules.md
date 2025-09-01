*Appendix D* 

**Version 2.1 – 2025-09-01**

> **Canonical** – This file supersedes all earlier Appendix D roadmaps. 


## Appendix D - Risks & Mitigations
<!-- CURSOR RULE: ALWAYS run risk_register_checks before editing this section -->

**Overall Status:** 📋 Planned 



### Top Technical Risks
- **Scaling bottlenecks:** Distributed training may hit network/IO ceilings → *Mitigation:* sharded parameter servers + gradient compression.
- **Model collapse during continual learning:** Catastrophic forgetting → *Mitigation:* rehearsal buffers, EWC, orthogonal gradients.
- **Safety guard bypass:** Adversarial prompt/tool abuse → *Mitigation:* layered RBAC, real-time anomaly detection, human-in-the-loop override.

### Programmatic Risks
- **Timeline slip:** Multi-team dependencies → *Mitigation:* critical-path tracking, weekly burndown audits.
- **Cost overruns:** Cloud spend spikes → *Mitigation:* spot instance quotas, autoscale caps, cost dashboards.
- **Data governance:** License or PII violations → *Mitigation:* dataset audit pipeline, automated PII scrubbers.

---
End of roadmap suite. Return to [Master Index](MASTER_ROADMAP.md)