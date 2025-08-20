#!/bin/bash
# Smart Environment Manager for Small-Mind
# Automatically detects and fixes common environment issues

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo -e "${BLUE}🧠 Small-Mind Smart Environment Manager${NC}"
echo "=================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
        PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            echo -e "${GREEN}✅ Python $PYTHON_VERSION detected${NC}"
            return 0
        else
            echo -e "${RED}❌ Python $PYTHON_VERSION is too old. Need Python 3.8+${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ Python3 not found${NC}"
        return 1
    fi
}

# Function to find and activate virtual environment
find_and_activate_venv() {
    local venv_found=false
    
    # Common virtual environment locations
    local venv_locations=("aws_env" "env" "venv" ".venv")
    
    for venv_dir in "${venv_locations[@]}"; do
        if [ -d "$venv_dir" ] && [ -f "$venv_dir/bin/activate" ]; then
            echo -e "${BLUE}🔍 Found virtual environment: $venv_dir${NC}"
            
            # Check if already activated
            if [ -n "$VIRTUAL_ENV" ]; then
                echo -e "${YELLOW}⚠️  Virtual environment already activated: $VIRTUAL_ENV${NC}"
                if [ "$VIRTUAL_ENV" != "$(pwd)/$venv_dir" ]; then
                    echo -e "${YELLOW}⚠️  Different venv active, switching...${NC}"
                    deactivate 2>/dev/null || true
                fi
            fi
            
            # Activate virtual environment
            source "$venv_dir/bin/activate"
            echo -e "${GREEN}✅ Activated virtual environment: $venv_dir${NC}"
            venv_found=true
            break
        fi
    done
    
    if [ "$venv_found" = false ]; then
        echo -e "${YELLOW}⚠️  No virtual environment found${NC}"
        return 1
    fi
    
    return 0
}

# Function to check and fix Python path
fix_python_path() {
    local current_pythonpath="$PYTHONPATH"
    local project_root_abs="$(pwd)"
    
    # Check if project root is in PYTHONPATH
    if [[ ":$current_pythonpath:" != *":$project_root_abs:"* ]]; then
        echo -e "${BLUE}🔧 Adding project root to PYTHONPATH${NC}"
        export PYTHONPATH="$project_root_abs:$current_pythonpath"
        echo -e "${GREEN}✅ PYTHONPATH updated${NC}"
    else
        echo -e "${GREEN}✅ PYTHONPATH already configured${NC}"
    fi
}

# Function to check and install dependencies
check_dependencies() {
    echo -e "${BLUE}🔍 Checking dependencies...${NC}"
    
    # Check if we're in a virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        echo -e "${YELLOW}⚠️  Not in virtual environment, skipping dependency check${NC}"
        return 0
    fi
    
    # Check for requirements.txt
    if [ -f "requirements.txt" ]; then
        echo -e "${BLUE}📦 Found requirements.txt, checking packages...${NC}"
        
        # Check if pip is working
        if ! python -m pip --version >/dev/null 2>&1; then
            echo -e "${RED}❌ Pip not working properly${NC}"
            return 1
        fi
        
        # Check for missing packages
        local missing_packages=()
        while IFS= read -r package; do
            # Skip empty lines and comments
            [[ -z "$package" || "$package" =~ ^[[:space:]]*# ]] && continue
            
            # Extract package name (remove version specifiers)
            local package_name=$(echo "$package" | cut -d'=' -f1 | cut -d'>' -f1 | cut -d'<' -f1 | cut -d'~' -f1 | cut -d'!' -f1 | xargs)
            
            if ! python -c "import $package_name" >/dev/null 2>&1; then
                missing_packages+=("$package")
            fi
        done < requirements.txt
        
        if [ ${#missing_packages[@]} -gt 0 ]; then
            echo -e "${YELLOW}⚠️  Missing packages detected:${NC}"
            printf '%s\n' "${missing_packages[@]}"
            
            echo -e "${BLUE}🔧 Installing missing packages...${NC}"
            if python -m pip install -r requirements.txt; then
                echo -e "${GREEN}✅ Dependencies installed successfully${NC}"
            else
                echo -e "${RED}❌ Failed to install dependencies${NC}"
                return 1
            fi
        else
            echo -e "${GREEN}✅ All dependencies satisfied${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  No requirements.txt found${NC}"
    fi
}

# Function to create environment script
create_env_script() {
    local script_path="activate_env.sh"
    
    cat > "$script_path" << 'EOF'
#!/bin/bash
# Auto-generated environment activation script
# Generated by smart_env.sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set environment variables
export PROJECT_ROOT="$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Activate virtual environment
if [ -f "aws_env/bin/activate" ]; then
    source "aws_env/bin/activate"
    echo "✅ Virtual environment activated: aws_env"
elif [ -f "env/bin/activate" ]; then
    source "env/bin/activate"
    echo "✅ Virtual environment activated: env"
elif [ -f "venv/bin/activate" ]; then
    source "venv/bin/activate"
    echo "✅ Virtual environment activated: venv"
else
    echo "⚠️  No virtual environment found"
fi

# Add scripts to PATH
export PATH="$SCRIPT_DIR/scripts:$PATH"

echo "🚀 Environment ready for Small-Mind development"
echo "Project root: $SCRIPT_DIR"
echo "Python: $(which python 2>/dev/null || echo 'Not found')"
echo "Python version: $(python --version 2>/dev/null || echo 'Not found')"
EOF
    
    chmod +x "$script_path"
    echo -e "${GREEN}✅ Created environment script: $script_path${NC}"
}

# Function to run environment diagnostics
run_diagnostics() {
    echo -e "${BLUE}🔍 Running environment diagnostics...${NC}"
    
    # Check Python
    if ! check_python_version; then
        echo -e "${RED}❌ Python version check failed${NC}"
        return 1
    fi
    
    # Find and activate virtual environment
    if ! find_and_activate_venv; then
        echo -e "${YELLOW}⚠️  Virtual environment setup incomplete${NC}"
    fi
    
    # Fix Python path
    fix_python_path
    
    # Check dependencies
    check_dependencies
    
    # Create environment script
    create_env_script
    
    echo -e "${GREEN}✅ Environment diagnostics complete${NC}"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  -d, --diagnose    Run full environment diagnostics (default)"
    echo "  -f, --fix         Auto-fix common issues"
    echo "  -c, --check       Quick environment check"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                 # Run diagnostics"
    echo "  $0 --fix          # Auto-fix issues"
    echo "  $0 --check        # Quick check"
}

# Main execution
main() {
    case "${1:-}" in
        -h|--help)
            show_usage
            exit 0
            ;;
        -f|--fix)
            echo -e "${BLUE}🔧 Auto-fix mode enabled${NC}"
            run_diagnostics
            ;;
        -c|--check)
            echo -e "${BLUE}🔍 Quick check mode${NC}"
            check_python_version
            find_and_activate_venv
            ;;
        -d|--diagnose|"")
            run_diagnostics
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
    
    echo ""
    echo -e "${GREEN}🎉 Environment setup complete!${NC}"
    echo -e "${BLUE}💡 To activate environment in new terminals, run:${NC}"
    echo -e "   source activate_env.sh"
    echo -e "${BLUE}💡 Or use the smart environment manager:${NC}"
    echo -e "   python scripts/env_manager.py"
}

# Run main function with all arguments
main "$@"
