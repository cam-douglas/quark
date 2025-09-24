# 🚀 Quark Validation Quick Reference Card

## 🎯 Most Common Commands

```bash
# What do I need to validate for my changes?
make validate-quick

# Start interactive validation guide
make validate

# Check specific domain (smart matching)
make validate foundation layer
make validate integration
make validate stage 2

# View previous validation results
make validate-metrics

# Generate dashboard from results
make validate-dashboard
```

## 📝 Command Cheat Sheet

| What You Want | Command |
|--------------|---------|
| **"What needs validation?"** | `make validate-quick` |
| **"Guide me through validation"** | `make validate` |
| **"Check foundation layer"** | `make validate foundation layer` |
| **"Check stage 2"** | `python quark_validate.py verify --stage 2` |
| **"Show my metrics"** | `make validate-metrics` |
| **"Generate dashboard"** | `make validate-dashboard` |
| **"Show all commands"** | `make help` or `python quark_validate.py help` |
| **"CI checklist"** | `make validate-ci` |

## 🔄 Typical Workflow

1. **Check what needs validation:**
   ```bash
   make validate-quick
   ```

2. **Run interactive guide:**
   ```bash
   make validate
   # Follow prompts to:
   # - Select domain/stage
   # - Review requirements
   # - Perform manual measurements
   # - Record results
   ```

3. **View your results:**
   ```bash
   make validate-metrics
   make validate-dashboard
   ```

## 💡 Smart Domain Matching

The system understands natural language! All these work:
- `make validate foundation` → STAGE1_EMBRYONIC
- `make validate integration` → MAIN_INTEGRATIONS  
- `make validate adult brain` → STAGE6_ADULT
- `make validate fetal` → STAGE2_FETAL

## 📁 Key Locations

- **Checklists:** `state/tasks/validation/checklists/`
- **Your Evidence:** `state/tasks/validation/evidence/`
- **Dashboard:** `state/tasks/validation/dashboards/`
- **Documentation:** `state/tasks/validation/VALIDATION_GUIDE.md`

## ⚠️ Remember

**ALL validation is MANUAL** - The system shows you what needs validation but never automatically validates. You must:
1. Run your code/experiments
2. Measure KPIs yourself
3. Compare against targets
4. Record results manually

## 🆘 Need Help?

```bash
# Comprehensive help
python quark_validate.py help

# Quick command list
make help

# Validation-specific help
make validate-help
```

---
*Save this file for quick reference!*
