# Cursor Rules Update System - SUPREME AUTHORITY

## 🏛️ Overview

The Cursor Rules Update System is a **SUPREME AUTHORITY** system that automatically keeps all cursor rules synchronized with new knowledge and developments in the project. It runs continuously and ensures all rules maintain compliance with the cognitive_brain_roadmap.md specifications.

**Priority Level**: 0 (Supreme - Above All Others)  
**Status**: Always Active - Maximum Priority  
**Authority**: Can override, veto, or modify any rule set or component

---

## 🚀 Quick Start

### Start the Rule Update System
```bash
# Start the system
.cursor/rules/start_rule_updater.sh start

# Check status
.cursor/rules/start_rule_updater.sh status

# View logs
.cursor/rules/start_rule_updater.sh logs
```

### Enable Auto-Start (System Boot)
```bash
# Copy service file to systemd directory
sudo cp .cursor/rules/cursor-rules-updater.service /Library/LaunchDaemons/

# Enable and start the service
sudo launchctl load /Library/LaunchDaemons/cursor-rules-updater.service
```

---

## 🏗️ System Architecture

### **Supreme Authority Hierarchy**
```
Priority 0: compliance_review.md (SUPREME AUDITOR)
├── Can override any rule set or component
├── Always active at maximum priority
├── Enforces compliance across entire system
└── Maintains system-wide knowledge sharing

Priority 1-14: All other rule sets
├── Subject to compliance_review.md authority
├── Normal operation within hierarchy
└── Can be overridden by supreme authority
```

### **Core Components**

#### **1. RuleUpdateSystem Class**
- **Purpose**: Supreme authority system for updating and synchronizing all cursor rules
- **Authority**: Can override, veto, or modify any rule set or component
- **Status**: Always active at maximum priority

#### **2. File System Monitoring**
- **Purpose**: Continuously monitors project files for changes
- **Triggers**: Automatic updates when new knowledge is detected
- **Scope**: All Python, Markdown, YAML, and JSON files

#### **3. Knowledge Base Management**
- **Purpose**: Tracks and distributes new knowledge across all rule sets
- **Types**: New components, parameters, connections, architectural changes
- **Distribution**: Automatic broadcasting to all rule files

#### **4. Compliance Enforcement**
- **Purpose**: Ensures all rules maintain compliance with roadmap specifications
- **Auto-Fix**: Automatically fixes compliance issues when detected
- **Audit**: Regular compliance audits across all rule sets

---

## 🔧 Features

### **Automatic Knowledge Distribution**
- **New Components**: Automatically adds new brain components to appropriate rule files
- **Updated Parameters**: Distributes new configuration parameters across all rules
- **New Connections**: Updates wiring and connection information
- **Architectural Changes**: Propagates architectural updates to all rule sets

### **Compliance Enforcement**
- **Supreme Authority**: Ensures all rules acknowledge compliance_review.md authority
- **Hierarchy Validation**: Validates proper hierarchy levels across all rules
- **Activation Triggers**: Ensures all rules have proper activation triggers
- **Auto-Fix**: Automatically fixes compliance issues when detected

### **Continuous Monitoring**
- **File System Events**: Real-time monitoring of file changes
- **Change Detection**: SHA-256 hash-based change detection
- **Automatic Updates**: Immediate updates when changes are detected
- **Logging**: Comprehensive logging of all activities

### **System Integration**
- **Systemd Service**: Automatic startup on system boot
- **Process Management**: PID-based process tracking
- **Auto-Restart**: Automatic restart if system stops unexpectedly
- **Resource Management**: Proper resource limits and security settings

---

## 📋 Usage Commands

### **Basic Commands**
```bash
# Start the system
.cursor/rules/start_rule_updater.sh start

# Stop the system
.cursor/rules/start_rule_updater.sh stop

# Restart the system
.cursor/rules/start_rule_updater.sh restart

# Check status
.cursor/rules/start_rule_updater.sh status

# View live logs
.cursor/rules/start_rule_updater.sh logs

# Show help
.cursor/rules/start_rule_updater.sh help
```

### **Advanced Commands**
```bash
# Start monitoring mode (auto-restart)
.cursor/rules/start_rule_updater.sh monitor

# Run Python script directly
python3 .cursor/rules/rule_update_script.py

# Check dependencies
python3 -c "import watchdog, yaml; print('Dependencies OK')"
```

---

## 🔍 Monitoring & Logs

### **Log Files**
- **Main Log**: `.cursor/rules/rule_updater.log`
- **Rule Updates**: `.cursor/rules/rule_updates.log`
- **System Logs**: `journalctl -u cursor-rules-updater`

### **Log Levels**
- **INFO**: Normal operations and updates
- **WARNING**: Compliance issues and auto-fixes
- **ERROR**: System errors and failures
- **DEBUG**: Detailed debugging information

### **Monitoring Dashboard**
```bash
# View real-time status
watch -n 5 '.cursor/rules/start_rule_updater.sh status'

# Monitor logs in real-time
tail -f .cursor/rules/rule_updater.log

# Check system resources
ps aux | grep rule_update_script
```

---

## 🛠️ Configuration

### **Environment Variables**
```bash
# Python path
export PYTHONPATH=/Users/camdouglas/quark

# Python unbuffered output
export PYTHONUNBUFFERED=1

# Log level
export LOG_LEVEL=INFO
```

### **Systemd Service Configuration**
- **User**: camdouglas
- **Group**: staff
- **Working Directory**: /Users/camdouglas/quark
- **Restart Policy**: Always restart
- **Security**: Strict protection settings

### **File Permissions**
```bash
# Make scripts executable
chmod +x .cursor/rules/start_rule_updater.sh
chmod +x .cursor/rules/rule_update_script.py

# Set proper ownership
chown camdouglas:staff .cursor/rules/*
```

---

## 🔒 Security & Permissions

### **Security Features**
- **No New Privileges**: Service cannot gain new privileges
- **Private Temporary Directory**: Isolated temp directory
- **Strict System Protection**: Limited system access
- **Home Directory Protection**: Restricted home directory access
- **Read-Write Paths**: Only specific paths are writable

### **Resource Limits**
- **File Descriptors**: 65,536 limit
- **Processes**: 4,096 limit
- **Memory**: Automatic management
- **CPU**: Fair scheduling

---

## 🚨 Troubleshooting

### **Common Issues**

#### **1. Service Won't Start**
```bash
# Check dependencies
python3 -c "import watchdog, yaml"

# Check permissions
ls -la .cursor/rules/start_rule_updater.sh

# Check logs
tail -f .cursor/rules/rule_updater.log
```

#### **2. Rules Not Updating**
```bash
# Check if system is running
.cursor/rules/start_rule_updater.sh status

# Check file permissions
ls -la .cursor/rules/*.md

# Check Python script
python3 .cursor/rules/rule_update_script.py
```

#### **3. Compliance Issues**
```bash
# Run compliance audit manually
python3 -c "
from .cursor.rules.rule_update_script import RuleUpdateSystem
system = RuleUpdateSystem()
system.run_compliance_audit()
"
```

### **Debug Mode**
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
.cursor/rules/start_rule_updater.sh restart

# View debug logs
tail -f .cursor/rules/rule_updater.log | grep DEBUG
```

---

## 📚 API Reference

### **RuleUpdateSystem Class**

#### **Methods**
- `detect_changes()`: Detect changes in rule files and project files
- `update_rule_with_knowledge()`: Update a rule file with new knowledge
- `enforce_compliance()`: Enforce compliance across all rule sets
- `broadcast_knowledge_update()`: Broadcast knowledge updates to all components
- `run_compliance_audit()`: Run a comprehensive compliance audit
- `start_monitoring()`: Start continuous monitoring of the system

#### **Properties**
- `file_hashes`: Track file hashes for change detection
- `knowledge_base`: Knowledge base for tracking new information
- `supreme_authority`: Supreme authority configuration
- `rule_files`: Dictionary of all rule files in the system

---

## 🔄 Update Process

### **1. Change Detection**
- File system monitoring detects changes
- SHA-256 hash comparison identifies modifications
- New files trigger knowledge updates

### **2. Knowledge Analysis**
- New brain-related code → New components
- New configuration files → Updated parameters
- New documentation → Architectural changes

### **3. Rule Updates**
- Knowledge is distributed to all rule files
- Appropriate sections are updated
- Timestamps are added to track changes

### **4. Compliance Validation**
- All rules are checked for compliance
- Missing elements are auto-fixed
- Supreme authority is enforced

---

## 🎯 Best Practices

### **1. Regular Monitoring**
- Check system status daily
- Review logs weekly
- Run compliance audits monthly

### **2. Backup Strategy**
- Backup rule files before major changes
- Version control all rule modifications
- Document manual changes

### **3. Testing**
- Test rule updates in development environment
- Validate compliance after updates
- Monitor system performance

### **4. Documentation**
- Document new components and parameters
- Update architectural documentation
- Maintain change logs

---

## 📞 Support

### **Getting Help**
- **Logs**: Check `.cursor/rules/rule_updater.log`
- **Status**: Run `.cursor/rules/start_rule_updater.sh status`
- **Documentation**: Review this README and rule files

### **Emergency Procedures**
```bash
# Stop the system immediately
.cursor/rules/start_rule_updater.sh stop

# Check for issues
.cursor/rules/start_rule_updater.sh status

# Restart with monitoring
.cursor/rules/start_rule_updater.sh monitor
```

---

## 🏛️ Supreme Authority Statement

This Rule Update System operates with **SUPREME AUTHORITY** (Priority 0) and can:

1. **Override any rule set** or component in the system
2. **Veto any decision** made by other components
3. **Modify any parameter** in the system
4. **Enforce compliance** across all rule sets
5. **Audit any component** at any time

The system maintains the highest standards of compliance and ensures all cursor rules work together to implement the cognitive brain roadmap effectively.

---

*This system ensures all cursor rules remain synchronized, compliant, and up-to-date with the latest knowledge and developments in the project.*
