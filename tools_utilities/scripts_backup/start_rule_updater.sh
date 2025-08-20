#!/bin/bash
# Cursor Rules Update System Startup Script
# Ensures the rule update system always runs with SUPREME AUTHORITY

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RULE_UPDATE_SCRIPT="$SCRIPT_DIR/rule_update_script.py"
LOG_FILE="$SCRIPT_DIR/rule_updater.log"
PID_FILE="$SCRIPT_DIR/rule_updater.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log messages
log_message() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1" | tee -a "$LOG_FILE"
}

# Function to check if Python dependencies are installed
check_dependencies() {
    log_info "Checking Python dependencies..."
    
    # Check if watchdog is installed
    if ! python3 -c "import watchdog" 2>/dev/null; then
        log_warning "watchdog not found. Installing..."
        pip3 install watchdog
    fi
    
    # Check if PyYAML is installed
    if ! python3 -c "import yaml" 2>/dev/null; then
        log_warning "PyYAML not found. Installing..."
        pip3 install PyYAML
    fi
    
    log_info "Dependencies check complete"
}

# Function to check if the rule update system is already running
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0  # Running
        else
            # PID file exists but process is dead
            rm -f "$PID_FILE"
        fi
    fi
    return 1  # Not running
}

# Function to start the rule update system
start_rule_updater() {
    log_info "Starting Cursor Rules Update System with SUPREME AUTHORITY..."
    
    # Check dependencies
    check_dependencies
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    # Start the rule update script in background
    nohup python3 "$RULE_UPDATE_SCRIPT" >> "$LOG_FILE" 2>&1 &
    local pid=$!
    
    # Save PID
    echo "$pid" > "$PID_FILE"
    
    log_info "Rule Update System started with PID: $pid"
    log_info "Priority Level: 0 (Supreme - Above All Others)"
    log_info "Status: Always Active - Maximum Priority"
    log_info "Authority: Can override, veto, or modify any rule set or component"
    
    return $pid
}

# Function to stop the rule update system
stop_rule_updater() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        log_info "Stopping Rule Update System (PID: $pid)..."
        kill "$pid"
        rm -f "$PID_FILE"
        log_info "Rule Update System stopped"
    else
        log_warning "Rule Update System is not running"
    fi
}

# Function to restart the rule update system
restart_rule_updater() {
    log_info "Restarting Rule Update System..."
    stop_rule_updater
    sleep 2
    start_rule_updater
}

# Function to check status
status_rule_updater() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        log_info "Rule Update System is RUNNING (PID: $pid)"
        log_info "Priority Level: 0 (Supreme Authority)"
        log_info "Status: Always Active - Maximum Priority"
        return 0
    else
        log_warning "Rule Update System is NOT RUNNING"
        return 1
    fi
}

# Function to monitor and auto-restart
monitor_rule_updater() {
    log_info "Starting Rule Update System monitor..."
    
    while true; do
        if ! is_running; then
            log_warning "Rule Update System stopped unexpectedly. Restarting..."
            start_rule_updater
        fi
        
        # Check every 30 seconds
        sleep 30
    done
}

# Function to show logs
show_logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        log_warning "No log file found"
    fi
}

# Function to show help
show_help() {
    echo "Cursor Rules Update System - SUPREME AUTHORITY"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     - Start the Rule Update System"
    echo "  stop      - Stop the Rule Update System"
    echo "  restart   - Restart the Rule Update System"
    echo "  status    - Check if the Rule Update System is running"
    echo "  monitor   - Start monitoring and auto-restart if needed"
    echo "  logs      - Show live logs"
    echo "  help      - Show this help message"
    echo ""
    echo "Priority Level: 0 (Supreme - Above All Others)"
    echo "Status: Always Active - Maximum Priority"
    echo "Authority: Can override, veto, or modify any rule set or component"
}

# Main script logic
case "${1:-}" in
    start)
        if is_running; then
            log_warning "Rule Update System is already running"
        else
            start_rule_updater
        fi
        ;;
    stop)
        stop_rule_updater
        ;;
    restart)
        restart_rule_updater
        ;;
    status)
        status_rule_updater
        ;;
    monitor)
        monitor_rule_updater
        ;;
    logs)
        show_logs
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        show_help
        exit 1
        ;;
esac
