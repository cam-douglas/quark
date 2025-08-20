set -euo pipefail

#!/bin/bash

# Small-Mind Terminal Setup Script
# This script helps configure the terminal profile and launch daemon

echo "🧠 Setting up Small-Mind Terminal Profile..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Backup original .zshrc
if [ -f "/Users/camdouglas/.zshrc" ]; then
    cp "/Users/camdouglas/.zshrc" "/Users/camdouglas/.zshrc.backup.$(date +%Y%m%d_%H%M%S)"
    echo "${GREEN}✅ Backup created: .zshrc.backup.$(date +%Y%m%d_%H%M%S)${NC}"
fi

# Create LaunchAgents directory if it doesn't exist
mkdir -p "/Users/camdouglas/Library/LaunchAgents"

# Load the launch daemon
echo "${BLUE}📱 Loading launch daemon...${NC}"
launchctl load "/Users/camdouglas/Library/LaunchAgents/com.smallmind.terminal.plist"

# Test the terminal profile
echo "${BLUE}🧪 Testing terminal profile...${NC}"
source "/Users/camdouglas/.zshrc"

echo ""
echo "${GREEN}🎉 Small-Mind Terminal Profile Setup Complete!${NC}"
echo ""
echo "${YELLOW}📋 Next Steps:${NC}"
echo "1. Close and reopen your terminal"
echo "2. You should see the Small-Mind welcome message"
echo "3. Type 'smallmind-help' to see available commands"
echo "4. The system will automatically load on startup"
echo ""
echo "${BLUE}💡 If you need to restore your original .zshrc, check the backup file${NC}"
echo ""
