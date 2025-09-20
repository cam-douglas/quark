#!/bin/bash
# Academic MCP Servers Installation Script
# This script installs and configures all academic MCP servers for Cursor

set -e  # Exit on error

echo "🚀 Installing Academic MCP Servers for Enhanced Validation"
echo "========================================================="

# Create base directory for external tools if it doesn't exist
EXTERNAL_TOOLS_DIR="$HOME/external_tools"
ACADEMIC_DIR="$EXTERNAL_TOOLS_DIR/academic_mcp_servers"

mkdir -p "$ACADEMIC_DIR"
cd "$ACADEMIC_DIR"

# Function to install Python-based MCP server
install_python_mcp() {
    local NAME=$1
    local REPO_URL=$2
    local DIR_NAME=$3
    
    echo ""
    echo "📦 Installing $NAME..."
    
    if [ -d "$DIR_NAME" ]; then
        echo "  Directory exists, updating..."
        cd "$DIR_NAME"
        git pull || true
    else
        echo "  Cloning repository..."
        git clone "$REPO_URL" "$DIR_NAME" || {
            echo "  ⚠️ Could not clone $REPO_URL"
            echo "  You may need to search for the correct repository"
            return 1
        }
        cd "$DIR_NAME"
    fi
    
    # Create virtual environment
    echo "  Creating virtual environment..."
    python3 -m venv .venv
    
    # Activate and install dependencies
    echo "  Installing dependencies..."
    source .venv/bin/activate
    
    # Try multiple ways to install dependencies
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    elif [ -f "setup.py" ]; then
        pip install -e .
    elif [ -f "pyproject.toml" ]; then
        pip install -e .
    else
        echo "  ⚠️ No requirements file found, trying common packages..."
        pip install mcp aiohttp asyncio
    fi
    
    deactivate
    cd "$ACADEMIC_DIR"
    echo "  ✅ $NAME installed"
}

# Function to install Node.js-based MCP server
install_node_mcp() {
    local NAME=$1
    local REPO_URL=$2
    local DIR_NAME=$3
    
    echo ""
    echo "📦 Installing $NAME..."
    
    if [ -d "$DIR_NAME" ]; then
        echo "  Directory exists, updating..."
        cd "$DIR_NAME"
        git pull || true
    else
        echo "  Cloning repository..."
        git clone "$REPO_URL" "$DIR_NAME" || {
            echo "  ⚠️ Could not clone $REPO_URL"
            echo "  You may need to search for the correct repository"
            return 1
        }
        cd "$DIR_NAME"
    fi
    
    # Install Node dependencies
    echo "  Installing Node dependencies..."
    npm install
    
    # Build if necessary
    if [ -f "tsconfig.json" ]; then
        echo "  Building TypeScript..."
        npm run build || npx tsc
    fi
    
    cd "$ACADEMIC_DIR"
    echo "  ✅ $NAME installed"
}

# Install Python-based academic MCP servers
echo ""
echo "🐍 Installing Python-based MCP servers..."
echo "-----------------------------------------"

# ArXiv MCP Server
install_python_mcp \
    "ArXiv MCP Server" \
    "https://github.com/QuantumArjun/arxiv-mcp-server.git" \
    "arxiv-mcp-server"

# OpenAlex MCP Server  
install_python_mcp \
    "OpenAlex MCP Server" \
    "https://github.com/alexandercerutti/alex-mcp.git" \
    "alex-mcp"

# PubMed MCP Server
install_python_mcp \
    "PubMed MCP Server" \
    "https://github.com/QuantumArjun/PubMed-MCP-Server.git" \
    "PubMed-MCP-Server"

# Google Scholar MCP Server
install_python_mcp \
    "Google Scholar MCP Server" \
    "https://github.com/QuantumArjun/Google-Scholar-MCP-Server.git" \
    "Google-Scholar-MCP-Server"

# Academic Search MCP Server
install_python_mcp \
    "Academic Search MCP Server" \
    "https://github.com/chanyeol525/mcp-server-academic-search.git" \
    "academic-search-mcp-server"

# Install Node.js-based academic MCP servers
echo ""
echo "📘 Installing Node.js-based MCP servers..."
echo "------------------------------------------"

# Unpaywall MCP
install_node_mcp \
    "Unpaywall MCP" \
    "https://github.com/danthegoodman1/unpaywall-mcp.git" \
    "unpaywall-mcp"

# Scientific Papers MCP
install_node_mcp \
    "Scientific Papers MCP" \
    "https://github.com/felores/scientific-papers-mcp.git" \
    "Scientific-Papers-MCP"

# Install other MCP servers
echo ""
echo "🔧 Installing other MCP servers..."
echo "-----------------------------------"

cd "$EXTERNAL_TOOLS_DIR"

# Crawl4AI MCP Server
if [ ! -d "crawl4ai-mcp-server" ]; then
    echo "📦 Installing Crawl4AI MCP Server..."
    git clone https://github.com/crawl4ai/mcp-server-crawl4ai.git crawl4ai-mcp-server || \
    git clone https://github.com/felores/crawl4ai-mcp-server.git crawl4ai-mcp-server || \
    echo "⚠️ Could not find Crawl4AI MCP repository"
fi

if [ -d "crawl4ai-mcp-server" ]; then
    cd crawl4ai-mcp-server
    python3 -m venv .venv
    source .venv/bin/activate
    pip install crawl4ai mcp || pip install -r requirements.txt || true
    deactivate
    cd "$EXTERNAL_TOOLS_DIR"
fi

# Blockscout MCP Server
if [ ! -d "blockscout-mcp-server" ]; then
    echo "📦 Installing Blockscout MCP Server..."
    git clone https://github.com/ethereum/blockscout-mcp-server.git blockscout-mcp-server || \
    echo "⚠️ Could not find Blockscout MCP repository"
fi

if [ -d "blockscout-mcp-server" ]; then
    cd blockscout-mcp-server
    python3 -m venv .venv
    source .venv/bin/activate
    pip install mcp requests || pip install -r requirements.txt || true
    deactivate
    cd "$EXTERNAL_TOOLS_DIR"
fi

# XCode MCP Server
if [ ! -d "xcode-mcp-server" ]; then
    echo "📦 Installing XCode MCP Server..."
    git clone https://github.com/kvn47/xcode-mcp-server.git xcode-mcp-server || \
    echo "⚠️ Could not find XCode MCP repository"
fi

if [ -d "xcode-mcp-server" ]; then
    cd xcode-mcp-server
    npm install || true
    cd "$EXTERNAL_TOOLS_DIR"
fi

# Create credential files if needed
echo ""
echo "🔑 Setting up credential placeholders..."
echo "-----------------------------------------"

# Figma API token
if [ ! -f "$HOME/.figma_api_token" ]; then
    echo "# Add your Figma API token here" > "$HOME/.figma_api_token"
    echo "Created: ~/.figma_api_token (add your token)"
fi

# Crawl4AI config
if [ ! -f "$HOME/.crawl4ai_config.json" ]; then
    echo '{"api_key": "your_api_key_here"}' > "$HOME/.crawl4ai_config.json"
    echo "Created: ~/.crawl4ai_config.json (add your config)"
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "📝 Next steps:"
echo "1. Add API keys to credential files if needed"
echo "2. Restart Cursor to load the MCP servers"
echo "3. Test with: python tools_utilities/test_mcp_servers.py"
echo ""
echo "🔍 Validation sources now available:"
echo "  - ArXiv (research papers)"
echo "  - PubMed (medical/biological papers)"
echo "  - OpenAlex (academic citations)"
echo "  - Google Scholar (academic search)"
echo "  - Unpaywall (open access papers)"
echo "  - Scientific Papers (paper aggregator)"
echo ""
echo "These will help enforce the anti-overconfidence rules!"
