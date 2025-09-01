*Appendix B* 

**Version 2.1 – 2025-09-01**

> **Canonical** – This file supersedes all earlier Appendix B roadmaps. 

## Appendix B - Safety, Alignment & Auditing
<!-- CURSOR RULE: ALWAYS run safety_alignment_suite before editing this section -->

**Overall Status:** 📋 Planned 


### <runtime-guards>
- **Scope:** Enforce RBAC + affordance checks on every tool call  
- **Mechanism:** Dynamic policy lookup → capability graph pruning  
- **Verification:** Unit tests + fuzzing on tool affordance boundaries

### <policy-compiler>
- **Input:** Declarative constraint set (YAML / REGO)  
- **Output:** eBPF / runtime monitors injected at orchestration layer  
- **Audit Trail:** Signed provenance hash → stored in `state/audit_log/`

### <provenance>
- **Captured Artifacts:** Inputs, prompts, code revisions, data snapshots, model weights  
- **Hashing:** SHA-256 chained digests; Merkle DAG for traceability  
- **Interfaces:** GraphQL + CLI (`quark prov query <artifact>`)

### <human-review>
- **Trigger Conditions:** High-impact decisions, self-mods, safety threshold breaches  
- **Process:** Sandbox execution → diff report → reviewer sign-off  
- **Escalation:** 24-hr SLA to Safety Officer

### <incident-response>
- **Pipeline:** Anomaly detection → triage board → automated rollback → root-cause notebook auto-generated (`reports/incidents/YYYYMMDD.ipynb`)  
- **Post-Mortem:** 5-Whys + action items stored in `docs/post_mortems/`

---
→ Continue to: [Appendix C – Benchmarks & Probes](appendix_c_rules.md)


