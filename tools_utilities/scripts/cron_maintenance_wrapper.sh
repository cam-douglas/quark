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

# Run rules compliance check
log_message "INFO: Running rules compliance check"
if python3 tools_utilities/scripts/rules_compliance_checker.py >> "$LOG_FILE" 2>&1; then
    log_message "INFO: Rules compliance check completed successfully"
else
    log_message "ERROR: Rules compliance check failed with exit code $?"
fi

# Run cleanup
log_message "INFO: Running workspace cleanup"
if python3 tools_utilities/scripts/workspace_cleanup.py >> "$LOG_FILE" 2>&1; then
    log_message "INFO: Workspace cleanup completed successfully"
else
    log_message "ERROR: Workspace cleanup failed with exit code $?"
fi

# Run Google Drive auto-sync
log_message "INFO: Running Google Drive auto-sync"
if python3 gdrive_auto_sync.py >> "$LOG_FILE" 2>&1; then
    log_message "INFO: Google Drive auto-sync completed successfully"
else
    log_message "ERROR: Google Drive auto-sync failed with exit code $?"
fi

log_message "INFO: Automated maintenance cycle completed"

# Clean up lock file (trap will also handle this)
rm -f "$LOCK_FILE"
