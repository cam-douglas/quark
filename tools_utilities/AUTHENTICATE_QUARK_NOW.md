# 🔐 Authenticate Your Quark Google Cloud Project

## ✅ Project Configured: `quark-469604`

Your project is now set! You just need to authenticate your Google account.

## 🚀 Quick Authentication (2 Steps)

### Step 1: Open Authentication URL

**Open this link in your browser:**

```
https://accounts.google.com/o/oauth2/auth?response_type=code&client_id=32555940559.apps.googleusercontent.com&redirect_uri=https%3A%2F%2Fsdk.cloud.google.com%2Fauthcode.html&scope=openid+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fuserinfo.email+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcloud-platform+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fappengine.admin+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fsqlservice.login+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcompute+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Faccounts.reauth&state=o7rTev28pnEfdwc8uBf2gU03ZRF5Lo&prompt=consent&token_usage=remote&access_type=offline&code_challenge=NqrMeNe2E3f9FwQp3Oyl4Rw1tk9Kptuq4hrmnLqiRRQ&code_challenge_method=S256
```

1. Sign in with your Google account
2. Allow access when prompted
3. **Copy the verification code** shown on the page

### Step 2: Enter Verification Code

Run this command and paste your verification code:

```bash
gcloud auth login --no-launch-browser
```

When it asks for the verification code, paste the code you copied from the browser.

## 🎯 After Authentication

### Set up Application Default Credentials (for Python libraries):
```bash
gcloud auth application-default login --no-launch-browser
```

### Verify Everything Works:
```bash
python tools_utilities/google_cloud_api_manager.py
```

### Enable Essential APIs:
```bash
python tools_utilities/google_cloud_api_manager.py --enable-essential
```

## 📋 Current Status

| Component | Status | Details |
|-----------|--------|---------|
| Project ID | ✅ Set | `quark-469604` |
| Authentication | ❌ Needed | Complete Step 1 & 2 above |
| Application Default Credentials | ❌ Needed | Set after authentication |
| APIs | ⏳ Ready to Enable | Can enable after auth |

## ⚡ Alternative: Use Python Helper

If you prefer, after getting the verification code from the browser, you can use:

```bash
python tools_utilities/google_cloud_api_manager.py --auth
```

This will prompt you for the verification code directly.
