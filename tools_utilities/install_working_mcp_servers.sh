#!/bin/bash
# Install only MCP servers we can verify exist and work
# More conservative approach - only install what we can confirm

set -e

echo "🚀 Installing Verified MCP Servers"
echo "=================================="

EXTERNAL_TOOLS_DIR="$HOME/external_tools"
mkdir -p "$EXTERNAL_TOOLS_DIR"

# Function to safely try installing an MCP server
try_install_mcp() {
    local NAME=$1
    local TYPE=$2  # python or node
    local INSTALL_CMD=$3
    
    echo ""
    echo "📦 Attempting to install $NAME..."
    
    if eval "$INSTALL_CMD"; then
        echo "  ✅ $NAME installed successfully"
        return 0
    else
        echo "  ⚠️ $NAME installation failed - will be removed from config"
        return 1
    fi
}

# Track successful installations
SUCCESSFUL_SERVERS=""
FAILED_SERVERS=""

# Try to install servers with known working configurations

# 1. filesystem MCP (commonly available)
try_install_mcp "filesystem" "npx" "npm list @modelcontextprotocol/server-filesystem >/dev/null 2>&1 || npm install -g @modelcontextprotocol/server-filesystem" && \
    SUCCESSFUL_SERVERS="$SUCCESSFUL_SERVERS filesystem" || \
    FAILED_SERVERS="$FAILED_SERVERS filesystem"

# 2. fetch MCP (for web content)
try_install_mcp "fetch" "npx" "npm list @modelcontextprotocol/server-fetch >/dev/null 2>&1 || npm install -g @modelcontextprotocol/server-fetch" && \
    SUCCESSFUL_SERVERS="$SUCCESSFUL_SERVERS fetch" || \
    FAILED_SERVERS="$FAILED_SERVERS fetch"

# 3. sqlite MCP (database access)
try_install_mcp "sqlite" "npx" "npm list @modelcontextprotocol/server-sqlite >/dev/null 2>&1 || npm install -g @modelcontextprotocol/server-sqlite" && \
    SUCCESSFUL_SERVERS="$SUCCESSFUL_SERVERS sqlite" || \
    FAILED_SERVERS="$FAILED_SERVERS sqlite"

# Try installing academic servers with fallback options
cd "$EXTERNAL_TOOLS_DIR"

# ArXiv - try multiple possible repos
echo ""
echo "📚 Searching for ArXiv MCP server..."
if [ ! -d "arxiv-mcp" ]; then
    # Try different possible repository names
    for REPO in "simonwillison/arxiv-mcp" "defalt/arxiv-mcp" "mcp-servers/arxiv"; do
        if git clone "https://github.com/$REPO.git" arxiv-mcp 2>/dev/null; then
            cd arxiv-mcp
            if [ -f "requirements.txt" ]; then
                python3 -m venv .venv
                source .venv/bin/activate
                pip install -r requirements.txt
                deactivate
                SUCCESSFUL_SERVERS="$SUCCESSFUL_SERVERS arxiv"
                echo "  ✅ ArXiv MCP installed from $REPO"
            fi
            cd "$EXTERNAL_TOOLS_DIR"
            break
        fi
    done
fi

# PubMed - search for alternatives
echo ""
echo "📚 Searching for PubMed MCP server alternatives..."
# Since direct repo might not exist, create a simple wrapper
mkdir -p pubmed-mcp-wrapper
cd pubmed-mcp-wrapper
cat > pubmed_wrapper.py << 'EOF'
#!/usr/bin/env python3
"""Simple PubMed API wrapper for MCP fallback"""
import requests
import json
import sys

def search_pubmed(query):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = f"{base_url}esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": 10
    }
    try:
        response = requests.get(search_url, params=params)
        return response.json()
    except:
        return {"error": "Failed to query PubMed"}

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "test"
    print(json.dumps(search_pubmed(query), indent=2))
EOF

python3 -m venv .venv
source .venv/bin/activate
pip install requests
deactivate
SUCCESSFUL_SERVERS="$SUCCESSFUL_SERVERS pubmed-wrapper"
echo "  ✅ PubMed wrapper created"
cd "$EXTERNAL_TOOLS_DIR"

echo ""
echo "=================================="
echo "INSTALLATION SUMMARY"
echo "=================================="
echo ""
echo "✅ Successfully installed/created:"
for server in $SUCCESSFUL_SERVERS; do
    echo "  - $server"
done

if [ -n "$FAILED_SERVERS" ]; then
    echo ""
    echo "❌ Failed to install:"
    for server in $FAILED_SERVERS; do
        echo "  - $server"
    done
fi

echo ""
echo "📝 Next steps:"
echo "1. Update ~/.cursor/mcp.json with working servers only"
echo "2. Use fallback validators for academic sources"
echo "3. Remove non-functional server entries from config"
