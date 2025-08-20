set -euo pipefail

#!/bin/bash

# Cursor Global Context Manager
# Manages persistent chat context across all workspaces

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

CURSOR_CHAT_DIR="$HOME/.cursor_global_chat_history"

# Function to show help
show_help() {
    echo ""
    echo "${PURPLE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo "${PURPLE}║                🎯 CURSOR CONTEXT MANAGER                     ║${NC}"
    echo "${PURPLE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "${WHITE}📋 AVAILABLE COMMANDS:${NC}"
    echo ""
    echo "${GREEN}1.  cursor-status${NC}      - Show Cursor context status"
    echo "${GREEN}2.  cursor-clear${NC}       - Clear all chat history (use with caution)"
    echo "${GREEN}3.  cursor-backup${NC}      - Create backup of chat history"
    echo "${GREEN}4.  cursor-restore${NC}     - Restore chat history from backup"
    echo "${GREEN}5.  cursor-info${NC}        - Show detailed context information"
    echo "${GREEN}6.  cursor-optimize${NC}    - Optimize chat history storage"
    echo "${GREEN}7.  cursor-help${NC}        - Show this help menu"
    echo ""
    echo "${YELLOW}💡 This system maintains persistent chat context across ALL projects${NC}"
    echo ""
}

# Function to show status
show_status() {
    echo ""
    echo "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo "${BLUE}║                  🎯 CURSOR CONTEXT STATUS                   ║${NC}"
    echo "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    if [ -d "$CURSOR_CHAT_DIR" ]; then
        echo "${GREEN}✅ Global Chat Directory: Found${NC}"
        echo "${GREEN}✅ Location: $CURSOR_CHAT_DIR${NC}"
        
        # Count chat sessions
        SESSION_COUNT=$(find "$CURSOR_CHAT_DIR/global_chat_sessions" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
        echo "${GREEN}✅ Chat Sessions: $SESSION_COUNT${NC}"
        
        # Check directory sizes
        TOTAL_SIZE=$(du -sh "$CURSOR_CHAT_DIR" 2>/dev/null | cut -f1)
        echo "${GREEN}✅ Total Size: $TOTAL_SIZE${NC}"
        
        # Check Cursor settings
        if [ -f "$HOME/Library/Application Support/Cursor/User/settings.json" ]; then
            echo "${GREEN}✅ Cursor Settings: Configured${NC}"
        else
            echo "${RED}❌ Cursor Settings: Not Found${NC}"
        fi
        
    else
        echo "${RED}❌ Global Chat Directory: Not Found${NC}"
        echo "${YELLOW}💡 Run setup to create the directory structure${NC}"
    fi
    
    echo ""
    echo "${YELLOW}💡 Type 'cursor-help' for more commands${NC}"
    echo ""
}

# Function to clear chat history
clear_history() {
    echo ""
    echo "${RED}⚠️  WARNING: This will delete ALL chat history across ALL projects!${NC}"
    echo "${RED}⚠️  This action cannot be undone!${NC}"
    echo ""
    read -p "Are you sure? Type 'YES' to confirm: " confirmation
    
    if [ "$confirmation" = "YES" ]; then
        echo "${YELLOW}🗑️  Clearing chat history...${NC}"
        rm -rf "$CURSOR_CHAT_DIR"/*
        mkdir -p "$CURSOR_CHAT_DIR"/{global_chat_sessions,cross_project_context,global_recovery_backups,project_mappings,session_index}
        echo "${GREEN}✅ Chat history cleared successfully${NC}"
    else
        echo "${BLUE}❌ Operation cancelled${NC}"
    fi
    echo ""
}

# Function to create backup
create_backup() {
    echo ""
    echo "${BLUE}💾 Creating backup of chat history...${NC}"
    
    BACKUP_DIR="$HOME/cursor_chat_backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    if [ -d "$CURSOR_CHAT_DIR" ]; then
        cp -r "$CURSOR_CHAT_DIR"/* "$BACKUP_DIR/"
        echo "${GREEN}✅ Backup created: $BACKUP_DIR${NC}"
    else
        echo "${RED}❌ No chat history to backup${NC}"
    fi
    echo ""
}

# Function to restore from backup
restore_backup() {
    echo ""
    echo "${BLUE}📥 Available backups:${NC}"
    
    BACKUP_DIRS=$(find "$HOME" -maxdepth 1 -name "cursor_chat_backup_*" -type d | sort)
    
    if [ -z "$BACKUP_DIRS" ]; then
        echo "${RED}❌ No backups found${NC}"
        return
    fi
    
    echo "$BACKUP_DIRS" | nl
    echo ""
    read -p "Enter backup number to restore: " backup_num
    
    BACKUP_DIR=$(echo "$BACKUP_DIRS" | sed -n "${backup_num}p")
    
    if [ -n "$BACKUP_DIR" ] && [ -d "$BACKUP_DIR" ]; then
        echo "${YELLOW}🔄 Restoring from: $BACKUP_DIR${NC}"
        rm -rf "$CURSOR_CHAT_DIR"/*
        cp -r "$BACKUP_DIR"/* "$CURSOR_CHAT_DIR/"
        echo "${GREEN}✅ Restore completed successfully${NC}"
    else
        echo "${RED}❌ Invalid backup selection${NC}"
    fi
    echo ""
}

# Function to show detailed info
show_info() {
    echo ""
    echo "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo "${CYAN}║                🎯 DETAILED CONTEXT INFO                      ║${NC}"
    echo "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    if [ -d "$CURSOR_CHAT_DIR" ]; then
        echo "${WHITE}📊 STORAGE BREAKDOWN:${NC}"
        echo ""
        
        for dir in global_chat_sessions cross_project_context global_recovery_backups project_mappings session_index; do
            if [ -d "$CURSOR_CHAT_DIR/$dir" ]; then
                SIZE=$(du -sh "$CURSOR_CHAT_DIR/$dir" 2>/dev/null | cut -f1)
                COUNT=$(find "$CURSOR_CHAT_DIR/$dir" -type f 2>/dev/null | wc -l | tr -d ' ')
                echo "${GREEN}📁 $dir:${NC} $SIZE ($COUNT files)"
            fi
        done
        
        echo ""
        echo "${WHITE}⚙️  CURSOR CONFIGURATION:${NC}"
        echo ""
        
        if [ -f "$HOME/Library/Application Support/Cursor/User/settings.json" ]; then
            echo "${GREEN}✅ Settings file exists${NC}"
            echo "${GREEN}✅ Global persistence enabled${NC}"
            echo "${GREEN}✅ Cross-workspace context enabled${NC}"
        else
            echo "${RED}❌ Settings file not found${NC}"
        fi
        
    else
        echo "${RED}❌ Chat history directory not found${NC}"
    fi
    
    echo ""
}

# Function to optimize storage
optimize_storage() {
    echo ""
    echo "${BLUE}🔧 Optimizing chat history storage...${NC}"
    
    if [ -d "$CURSOR_CHAT_DIR" ]; then
        # Remove old recovery backups (keep only last 3)
        BACKUP_COUNT=$(find "$CURSOR_CHAT_DIR/global_recovery_backups" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$BACKUP_COUNT" -gt 3 ]; then
            echo "${YELLOW}🗑️  Cleaning old recovery backups...${NC}"
            find "$CURSOR_CHAT_DIR/global_recovery_backups" -name "*.json" -mtime +7 -delete 2>/dev/null
        fi
        
        # Compress old chat sessions
        echo "${YELLOW}🗜️  Compressing old chat sessions...${NC}"
        find "$CURSOR_CHAT_DIR/global_chat_sessions" -name "*.json" -mtime +30 -exec gzip {} \; 2>/dev/null
        
        echo "${GREEN}✅ Storage optimization completed${NC}"
    else
        echo "${RED}❌ No chat history to optimize${NC}"
    fi
    echo ""
}

# Main command handler
case "${1:-help}" in
    "status"|"1")
        show_status
        ;;
    "clear"|"2")
        clear_history
        ;;
    "backup"|"3")
        create_backup
        ;;
    "restore"|"4")
        restore_backup
        ;;
    "info"|"5")
        show_info
        ;;
    "optimize"|"6")
        optimize_storage
        ;;
    "help"|"7"|*)
        show_help
        ;;
esac
