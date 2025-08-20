set -euo pipefail

#!/bin/bash
# Startup script for Continuous Learning System
# This script launches the cloud-powered continuous learning system

SMALLMIND_PATH="/Users/camdouglas/quark"
SCRIPT_DIR="$SMALLMIND_PATH/scripts"
LOG_DIR="$SMALLMIND_PATH/logs"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Function to start continuous learning
start_continuous_learning() {
    echo "🚀 Starting Cloud-Powered Continuous Learning System..."
    echo "📅 Started at: $(date)"
    echo "📍 Path: $SMALLMIND_PATH"
    
    # Change to small-mind directory
    cd "$SMALLMIND_PATH"
    
    # Start the continuous learning system
    nohup python3 "$SCRIPT_DIR/cloud_continuous_learner.py" start > "$LOG_DIR/continuous_learning.log" 2>&1 &
    
    # Get the process ID
    CONTINUOUS_LEARNING_PID=$!
    echo $CONTINUOUS_LEARNING_PID > "$LOG_DIR/continuous_learning.pid"
    
    echo "✅ Continuous Learning System started with PID: $CONTINUOUS_LEARNING_PID"
    echo "📋 Log file: $LOG_DIR/continuous_learning.log"
    echo "🔄 System will continuously learn from trusted sources and user interactions"
    echo "🌐 Leveraging cloud resources for accelerated training"
    echo "📚 Crawling Wikipedia, ArXiv, PubMed for synthetic data"
    echo "🧠 Training across all categories: neuroscience, physics, ML, visualization, NLP"
}

# Function to check if already running
check_if_running() {
    if [ -f "$LOG_DIR/continuous_learning.pid" ]; then
        PID=$(cat "$LOG_DIR/continuous_learning.pid")
        if ps -p $PID > /dev/null 2>&1; then
            echo "🟢 Continuous Learning System is already running (PID: $PID)"
            return 0
        else
            echo "🟡 PID file exists but process not running, cleaning up..."
            rm -f "$LOG_DIR/continuous_learning.pid"
            return 1
        fi
    fi
    return 1
}

# Function to stop continuous learning
stop_continuous_learning() {
    if [ -f "$LOG_DIR/continuous_learning.pid" ]; then
        PID=$(cat "$LOG_DIR/continuous_learning.pid")
        if ps -p $PID > /dev/null 2>&1; then
            echo "🛑 Stopping Continuous Learning System (PID: $PID)..."
            kill $PID
            rm -f "$LOG_DIR/continuous_learning.pid"
            echo "✅ Continuous Learning System stopped"
        else
            echo "⚠️  Process not running, cleaning up PID file"
            rm -f "$LOG_DIR/continuous_learning.pid"
        fi
    else
        echo "⚠️  No PID file found, system may not be running"
    fi
}

# Function to show status
show_status() {
    if check_if_running; then
        echo "📊 Continuous Learning System Status:"
        echo "   Status: 🟢 Running"
        echo "   PID: $(cat "$LOG_DIR/continuous_learning.pid")"
        echo "   Log: $LOG_DIR/continuous_learning.log"
        echo "   Started: $(stat -f "%Sm" "$LOG_DIR/continuous_learning.log" 2>/dev/null || echo "Unknown")"
        
        # Show recent log entries
        echo ""
        echo "📋 Recent Log Entries:"
        tail -n 10 "$LOG_DIR/continuous_learning.log" 2>/dev/null || echo "No log entries found"
    else
        echo "📊 Continuous Learning System Status:"
        echo "   Status: 🔴 Stopped"
        echo "   Use: $0 start to launch the system"
    fi
}

# Function to show help
show_help() {
    echo "Cloud-Powered Continuous Learning System - Startup Script"
    echo "========================================================"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start     - Start the continuous learning system"
    echo "  stop      - Stop the continuous learning system"
    echo "  restart   - Restart the continuous learning system"
    echo "  status    - Show current status and recent logs"
    echo "  help      - Show this help message"
    echo ""
    echo "Features:"
    echo "  🚀 Automatic startup on system boot"
    echo "  🌐 Cloud-powered training acceleration"
    echo "  📚 Continuous data crawling from trusted sources"
    echo "  🧠 Training across all AI categories"
    echo "  📊 Real-time learning from user interactions"
    echo "  🔄 Background operation with logging"
    echo ""
    echo "Trusted Sources:"
    echo "  • Wikipedia - General knowledge and concepts"
    echo "  • ArXiv - Scientific papers and research"
    echo "  • PubMed - Medical and neuroscience papers"
    echo ""
    echo "Learning Categories:"
    echo "  • Computational Neuroscience"
    echo "  • Physics Simulation"
    echo "  • ML Optimization"
    echo "  • Data Visualization"
    echo "  • Natural Language Processing"
}

# Main script logic
case "${1:-start}" in
    start)
        if check_if_running; then
            echo "⚠️  System is already running"
        else
            start_continuous_learning
        fi
        ;;
    stop)
        stop_continuous_learning
        ;;
    restart)
        echo "🔄 Restarting Continuous Learning System..."
        stop_continuous_learning
        sleep 2
        start_continuous_learning
        ;;
    status)
        show_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo "Use: $0 help for usage information"
        exit 1
        ;;
esac

echo ""
echo "💡 For more information, run: $0 help"
echo "📋 Check logs at: $LOG_DIR/continuous_learning.log"
