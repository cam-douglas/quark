#!/bin/bash
"""
Automated Maintenance Setup Script for Brain Simulation ML Framework

Purpose: Install cron job for daily workspace organization and maintenance
Inputs: None (configures system cron)
Outputs: Installed cron job, backup of existing crontab
Dependencies: cron, python3
Seeds: N/A (system configuration script)
"""

set -e  # Exit on any error

# Configuration
PROJECT_ROOT="/Users/camdouglas/quark"
ORGANIZER_SCRIPT="$PROJECT_ROOT/tools_utilities/scripts/automated_workspace_organizer.py"
CLEANUP_SCRIPT="$PROJECT_ROOT/tools_utilities/scripts/workspace_cleanup.py"
LOG_DIR="$PROJECT_ROOT/logs"
CRON_LOG="$LOG_DIR/automated_maintenance.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Setting up automated workspace maintenance for Brain Simulation ML Framework${NC}"
echo "=================================================="

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Check if scripts exist
if [[ ! -f "$ORGANIZER_SCRIPT" ]]; then
    echo -e "${RED}❌ Error: Organizer script not found at $ORGANIZER_SCRIPT${NC}"
    exit 1
fi

if [[ ! -f "$CLEANUP_SCRIPT" ]]; then
    echo -e "${RED}❌ Error: Cleanup script not found at $CLEANUP_SCRIPT${NC}"
    exit 1
fi

# Make scripts executable
chmod +x "$ORGANIZER_SCRIPT"
chmod +x "$CLEANUP_SCRIPT"

echo -e "${GREEN}✅ Scripts found and made executable${NC}"

# Create wrapper script for cron execution
WRAPPER_SCRIPT="$PROJECT_ROOT/tools_utilities/scripts/cron_maintenance_wrapper.sh"
cat > "$WRAPPER_SCRIPT" << 'EOF'
#!/bin/bash
# Cron wrapper for automated workspace maintenance
# Provides proper environment and error handling for cron execution

# Set PATH to ensure python3 is available
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

# Project configuration
PROJECT_ROOT="/Users/camdouglas/quark"
LOG_FILE="$PROJECT_ROOT/logs/automated_maintenance.log"
LOCK_FILE="$PROJECT_ROOT/logs/maintenance.lock"

# Ensure log directory exists
mkdir -p "$PROJECT_ROOT/logs"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Check for existing lock file (prevent overlapping runs)
if [[ -f "$LOCK_FILE" ]]; then
    LOCK_PID=$(cat "$LOCK_FILE")
    if ps -p "$LOCK_PID" > /dev/null 2>&1; then
        log_message "INFO: Maintenance already running (PID: $LOCK_PID), skipping"
        exit 0
    else
        log_message "INFO: Removing stale lock file"
        rm -f "$LOCK_FILE"
    fi
fi

# Create lock file
echo $$ > "$LOCK_FILE"

# Trap to ensure lock file is removed on exit
trap 'rm -f "$LOCK_FILE"; exit' INT TERM EXIT

log_message "INFO: Starting automated workspace maintenance"

# Change to project directory
cd "$PROJECT_ROOT" || {
    log_message "ERROR: Failed to change to project directory"
    exit 1
}

# Run workspace organization
log_message "INFO: Running workspace organizer"
if python3 tools_utilities/scripts/automated_workspace_organizer.py >> "$LOG_FILE" 2>&1; then
    log_message "INFO: Workspace organization completed successfully"
else
    log_message "ERROR: Workspace organization failed with exit code $?"
fi

# Run cleanup
log_message "INFO: Running workspace cleanup"
if python3 tools_utilities/scripts/workspace_cleanup.py >> "$LOG_FILE" 2>&1; then
    log_message "INFO: Workspace cleanup completed successfully"
else
    log_message "ERROR: Workspace cleanup failed with exit code $?"
fi

log_message "INFO: Automated maintenance cycle completed"

# Clean up lock file (trap will also handle this)
rm -f "$LOCK_FILE"
EOF

chmod +x "$WRAPPER_SCRIPT"
echo -e "${GREEN}✅ Created cron wrapper script${NC}"

# Backup existing crontab
BACKUP_DIR="$PROJECT_ROOT/backups"
mkdir -p "$BACKUP_DIR"
BACKUP_FILE="$BACKUP_DIR/crontab_backup_$(date +%Y%m%d_%H%M%S).txt"

if crontab -l > "$BACKUP_FILE" 2>/dev/null; then
    echo -e "${GREEN}✅ Backed up existing crontab to $BACKUP_FILE${NC}"
else
    echo -e "${YELLOW}⚠️  No existing crontab found (this is normal for first-time setup)${NC}"
    touch "$BACKUP_FILE"
fi

# Create new cron entry
CRON_ENTRY="# Automated Brain Simulation ML Framework Maintenance (runs daily at 2:00 AM)
0 2 * * * $WRAPPER_SCRIPT"

# Check if our cron job already exists
if crontab -l 2>/dev/null | grep -q "automated_workspace_organizer\|cron_maintenance_wrapper"; then
    echo -e "${YELLOW}⚠️  Automated maintenance cron job already exists${NC}"
    echo "Current crontab entries related to maintenance:"
    crontab -l 2>/dev/null | grep -E "automated_workspace_organizer|cron_maintenance_wrapper|Brain Simulation"
    
    echo -e "\n${BLUE}Do you want to replace the existing entry? (y/N)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        # Remove existing entries and add new one
        (crontab -l 2>/dev/null | grep -v -E "automated_workspace_organizer|cron_maintenance_wrapper|Brain Simulation"; echo "$CRON_ENTRY") | crontab -
        echo -e "${GREEN}✅ Updated existing cron job${NC}"
    else
        echo -e "${YELLOW}⏭️  Keeping existing cron job${NC}"
    fi
else
    # Add new cron entry
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
    echo -e "${GREEN}✅ Added new automated maintenance cron job${NC}"
fi

# Verify cron installation
echo -e "\n${BLUE}📋 Current crontab:${NC}"
crontab -l

# Create manual trigger script
MANUAL_TRIGGER="$PROJECT_ROOT/tools_utilities/scripts/run_maintenance_now.sh"
cat > "$MANUAL_TRIGGER" << EOF
#!/bin/bash
# Manual trigger for workspace maintenance
echo "🔧 Running manual workspace maintenance..."
cd "$PROJECT_ROOT"
bash "$WRAPPER_SCRIPT"
echo "✅ Manual maintenance completed. Check logs at: $LOG_FILE"
EOF

chmod +x "$MANUAL_TRIGGER"

# Test scripts
echo -e "\n${BLUE}🧪 Testing scripts...${NC}"
if python3 "$ORGANIZER_SCRIPT" --help > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Organizer script syntax OK${NC}"
else
    echo -e "${YELLOW}⚠️  Organizer script test returned non-zero (may be normal)${NC}"
fi

if python3 "$CLEANUP_SCRIPT" --help > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Cleanup script syntax OK${NC}"
else
    echo -e "${YELLOW}⚠️  Cleanup script test returned non-zero (may be normal)${NC}"
fi

echo -e "\n${GREEN}🎉 Automated maintenance setup complete!${NC}"
echo "=================================================="
echo "📅 Daily maintenance scheduled for 2:00 AM"
echo "📁 Logs will be written to: $LOG_FILE"
echo "🔧 Manual trigger: $MANUAL_TRIGGER"
echo "📋 View current crontab: crontab -l"
echo "🗑️  Remove cron job: crontab -e (then delete the Brain Simulation line)"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Check logs tomorrow after 2:00 AM"
echo "2. Run manual test: bash $MANUAL_TRIGGER"
echo "3. Monitor workspace organization quality"
echo ""
echo -e "${YELLOW}Note: Cron jobs run with limited environment. If issues occur,"
echo "check the log file for detailed error messages.${NC}"
