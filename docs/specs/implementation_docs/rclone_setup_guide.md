# 🔧 Google Drive Setup Guide for rclone

## 🚨 **Current Issue**
Your rclone configuration was incomplete and set to "filelu" instead of Google Drive.

## ✅ **Solution: Complete Setup Properly**

### **Step 1: Open a NEW terminal window**
Don't use the current one - open a completely new terminal.

### **Step 2: Run rclone config**
```bash
rclone config
```

### **Step 3: Follow these EXACT steps**
1. **Type `n`** and press Enter (for new remote)
2. **Type `gdrive`** and press Enter (name the remote)
3. **Type `18`** and press Enter (choose Google Drive)
4. **Type `1`** and press Enter (Application Default Credentials)
5. **Press Enter** (leave client_id blank)
6. **Press Enter** (leave client_secret blank)
7. **Type `y`** and press Enter (use auto config)
8. **Your browser will open** - sign in with Google account
9. **Grant permissions** to rclone
10. **Copy the verification code** back to terminal
11. **Type `y`** and press Enter (configure as team drive)
12. **Type `n`** and press Enter (no advanced config)
13. **Type `y`** and press Enter (save)
14. **Type `q`** and press Enter (quit)

### **Step 4: Test the connection**
```bash
rclone lsd gdrive:
```

## 🔍 **What Went Wrong**
- You may have selected the wrong option during setup
- The configuration got corrupted
- The setup process was interrupted

## 🎯 **Expected Result**
After proper setup, `rclone lsd gdrive:` should show your Google Drive folders.

## 📋 **Come Back When Ready**
Once you complete the setup above, come back here and I'll proceed with the next 4 steps automatically.

---

**Important:** Make sure to follow the steps EXACTLY as shown. Google Drive should be option 18, not filelu.
