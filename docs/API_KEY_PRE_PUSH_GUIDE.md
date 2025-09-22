# 🔑 API Key Pre-Push Hook Guide

## ✅ OVERVIEW

The pre-push hook now automatically checks all API keys for expiration and validity before allowing pushes to GitHub. This prevents broken validation systems from being deployed.

## 🚀 HOW IT WORKS

### **Automatic Checking**
Every `git push` now triggers:
1. **API Key Validation** - Tests all 24+ API services
2. **Expiration Detection** - Warns 7 days before expiry
3. **Direct Renewal Links** - Provides immediate access to renewal pages
4. **Flexible Ignore System** - Allows temporary bypassing

### **Hook Integration**
```bash
# Pre-push hook now includes:
🔍 Checking API key expiration status...
✅ All API keys are valid and not expired
✅ Pre-push checks completed successfully
```

## 🚨 WHEN PUSH IS BLOCKED

### **Example Blocked Push**:
```
================================================================================
🚨 API KEY ISSUES DETECTED - PUSH BLOCKED
================================================================================

❌ INVALID KEYS (2):
   • huggingface: API key for huggingface is INVALID
     🔗 Hugging Face Access Tokens: https://huggingface.co/settings/tokens
   • materials_project: API key for materials_project is INVALID
     🔗 Materials Project API Dashboard: https://next-gen.materialsproject.org/api

💡 RESOLUTION OPTIONS:
   1. Renew expired/invalid keys using the links above
   2. Temporarily ignore keys: echo 'service_name' >> .git/api_key_ignore
   3. Force push (not recommended): git push --no-verify
```

## 🔧 RESOLUTION OPTIONS

### **Option 1: Renew API Keys (Recommended)**
Click the provided links to renew keys:

| Service | Direct Renewal Link |
|---------|-------------------|
| **Materials Project** | https://next-gen.materialsproject.org/api |
| **OpenAI** | https://platform.openai.com/api-keys |
| **Claude** | https://console.anthropic.com/ |
| **GitHub** | https://github.com/settings/tokens |
| **Hugging Face** | https://huggingface.co/settings/tokens |
| **Context7** | https://context7.ai/dashboard/api-keys |
| **Wolfram Alpha** | https://developer.wolframalpha.com/portal/myapps/ |
| **Kaggle** | https://www.kaggle.com/settings/account |

### **Option 2: Temporarily Ignore Keys**
```bash
# Ignore specific services temporarily
echo 'huggingface' >> .git/api_key_ignore
echo 'materials_project' >> .git/api_key_ignore

# Now push will succeed
git push origin main
```

### **Option 3: Force Push (Not Recommended)**
```bash
# Bypass all checks (use with caution)
git push --no-verify origin main
```

## 📄 IGNORE FILE MANAGEMENT

### **Location**: `.git/api_key_ignore`

### **Format**:
```bash
# Temporarily ignored API keys for pre-push hook
# Format: service_name
# Remove lines to re-enable checking

huggingface
materials_project
context7
```

### **Managing Ignores**:
```bash
# Add service to ignore list
echo 'service_name' >> .git/api_key_ignore

# Remove service from ignore list
sed -i '' '/service_name/d' .git/api_key_ignore

# View current ignores
cat .git/api_key_ignore

# Clear all ignores
rm .git/api_key_ignore
```

## 🔍 KEY STATUS TYPES

### **✅ ACTIVE**
- Key is valid and working
- No action needed

### **⏰ EXPIRING SOON**
- Key expires within 7 days
- Warning shown but push allowed
- Renewal recommended

### **❌ EXPIRED**
- Key has passed expiration date
- Push blocked until renewed or ignored

### **❌ INVALID**
- Key authentication failed
- Push blocked until fixed or ignored

### **❓ UNKNOWN**
- Could not test key (network issues)
- Push allowed with warning

## 🛠️ TROUBLESHOOTING

### **Hook Not Running**
```bash
# Check hook exists and is executable
ls -la .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

### **Python Module Errors**
```bash
# Ensure dependencies are installed
cd /Users/camdouglas/quark
pip install aiohttp psutil
```

### **Network Issues**
```bash
# Test API connectivity manually
python3 tools_utilities/api_key_expiration_checker.py
```

### **Bypass Hook Temporarily**
```bash
# For emergency pushes
git push --no-verify origin main
```

## 📊 MONITORING

### **Check All Keys Manually**
```bash
python3 tools_utilities/api_key_expiration_checker.py
```

### **Generate Key Report**
```bash
python3 tools_utilities/api_key_manager.py
```

### **View Expiring Keys**
The system automatically warns about keys expiring within 7 days.

## 🔄 AUTOMATED WORKFLOWS

### **Weekly Key Check**
Consider setting up a weekly cron job:
```bash
# Add to crontab
0 9 * * 1 cd /Users/camdouglas/quark && python3 tools_utilities/api_key_expiration_checker.py
```

### **CI/CD Integration**
The pre-push hook ensures broken keys never reach production:
- ✅ Validates keys before deployment
- 🔗 Provides immediate renewal links
- ⚠️ Warns about upcoming expirations
- 🚫 Blocks pushes with expired keys

## 💡 BEST PRACTICES

1. **Renew keys promptly** when warnings appear
2. **Use ignore file sparingly** - only for temporary issues
3. **Keep backup keys** for critical services
4. **Monitor expiration dates** regularly
5. **Update credentials file** after renewals

---

**The pre-push hook now ensures your validation system always has working API keys! 🚀**
