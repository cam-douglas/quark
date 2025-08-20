set -euo pipefail

#!/bin/bash
# Terminal Integration for Small-Mind Neuro System
# Source this file to enable enhanced terminal features

# Colors and symbols for different environment types
declare -A ENV_COLORS=(
    ["ml"]="🧠 \033[1;35m"      # Bright magenta
    ["web"]="🌐 \033[1;36m"     # Bright cyan
    ["data"]="📊 \033[1;33m"    # Bright yellow  
    ["neuro"]="🧙 \033[1;95m"   # Bright purple
    ["blockchain"]="⛓️ \033[1;32m" # Bright green
    ["cloud"]="☁️ \033[1;34m"   # Bright blue
    ["general"]="🐍 \033[1;37m" # Bright white
)

# Reset color
RESET_COLOR="\033[0m"

# Function to detect project type (lightweight version)
detect_project_type() {
    local current_dir="${1:-$(pwd)}"
    
    # Quick detection based on file existence (much faster)
    if [[ -f "$current_dir/requirements.txt" ]]; then
        # Check for ML/AI patterns in requirements
        if grep -qE "(torch|tensorflow|sklearn|pandas|numpy)" "$current_dir/requirements.txt" 2>/dev/null; then
            echo "ml"
            return
        fi
        # Check for web patterns
        if grep -qE "(flask|django|fastapi)" "$current_dir/requirements.txt" 2>/dev/null; then
            echo "web"
            return
        fi
    fi
    
    # Check directory structure for quick hints
    if [[ -d "$current_dir/src/smallmind" || -d "$current_dir/neuro" ]]; then
        echo "neuro"
    elif [[ -d "$current_dir/models" ]]; then
        echo "ml"
    elif [[ -f "$current_dir/package.json" ]]; then
        echo "web"
    else
        echo "general"
    fi
}

# Function to check for virtual environment
check_virtual_env() {
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo "$(basename "$VIRTUAL_ENV")"
    elif [[ -f "env/bin/activate" ]]; then
        echo "env"
    elif [[ -f ".venv/bin/activate" ]]; then
        echo ".venv"
    elif [[ -f "venv/bin/activate" ]]; then
        echo "venv"
    else
        echo ""
    fi
}

# Function to get environment styling
get_env_style() {
    local project_type="${1:-general}"
    local env_name="$2"
    
    local style="${ENV_COLORS[$project_type]:-${ENV_COLORS[general]}}"
    local symbol="${style%% *}"
    local color="${style#* }"
    
    if [[ -n "$env_name" ]]; then
        echo "${symbol} (${color}${env_name}${RESET_COLOR})"
    else
        echo ""
    fi
}

# Enhanced prompt function
setup_enhanced_prompt() {
    local current_dir="$(pwd)"
    local project_type="$(detect_project_type "$current_dir")"
    local env_name="$(check_virtual_env)"
    local env_style="$(get_env_style "$project_type" "$env_name")"
    
    # Store original PS1 if not already stored
    if [[ -z "$ORIGINAL_PS1" ]]; then
        export ORIGINAL_PS1="$PS1"
    fi
    
    # Set enhanced prompt
    if [[ -n "$env_style" ]]; then
        export PS1="${env_style} $ORIGINAL_PS1"
    else
        export PS1="$ORIGINAL_PS1"
    fi
    
    # Set environment variables for terminal agent
    export NEURO_PROJECT_TYPE="$project_type"
    export NEURO_ENV_ACTIVE="$([[ -n "$env_name" ]] && echo "true" || echo "false")"
}

# Function to auto-suggest environment activation
suggest_env_activation() {
    local current_dir="$(pwd)"
    
    # Check if we're not in a virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        # Look for environment directories
        local env_dirs=("env" ".venv" "venv")
        for env_dir in "${env_dirs[@]}"; do
            if [[ -d "$current_dir/$env_dir" && -f "$current_dir/$env_dir/bin/activate" ]]; then
                echo "💡 Virtual environment detected: $env_dir"
                echo "   Activate with: source $env_dir/bin/activate"
                return
            fi
        done
        
        # Check if we should suggest creating an environment
        if [[ -f "$current_dir/requirements.txt" || -f "$current_dir/setup.py" || -f "$current_dir/pyproject.toml" ]]; then
            echo "💡 Python project detected. Consider creating a virtual environment:"
            echo "   python3 -m venv env && source env/bin/activate"
        fi
    fi
}

# Function to check dependencies automatically (lightweight)
auto_check_deps() {
    local current_dir="$(pwd)"
    
    # Only do a very quick check without heavy Python imports
    if [[ -f "$current_dir/requirements.txt" ]]; then
        # Simple check - just count lines in requirements.txt
        local req_count=$(wc -l < "$current_dir/requirements.txt" 2>/dev/null || echo 0)
        if [[ $req_count -gt 0 ]]; then
            echo "💡 Found requirements.txt with $req_count packages"
            echo "   Run: pip-deps for dependency analysis"
        fi
    fi
}

# Smart aliases for enhanced productivity
setup_smart_aliases() {
    # Dependency management
    alias pip-deps='python3 -m neuro.terminal_agent check-deps'
    alias env-info='python3 -m neuro.terminal_agent env-status'
    alias smart-install='python3 -m neuro.terminal_agent smart-install'
    alias deps-scan='python3 -m neuro.terminal_agent scan'
    
    # Neuro system commands
    alias neuro-scan='python3 -m neuro.cli scan'
    alias neuro-analyze='python3 -m neuro.cli analyze'
    alias neuro-organize='python3 -m neuro.cli organize'
    alias neuro-connectome='python3 -m neuro.cli connectome'
    
    # Environment management
    alias create-env='python3 -m venv env && source env/bin/activate && pip install --upgrade pip'
    alias activate-env='source env/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null'
    
    # Quick navigation and info
    alias project-info='echo "Project: $(basename $(pwd))"; echo "Type: $NEURO_PROJECT_TYPE"; echo "Env: $([[ "$NEURO_ENV_ACTIVE" == "true" ]] && echo "Active" || echo "Inactive")"'
}

# Directory change hook to update prompt and check environment
cd_hook() {
    # Update prompt based on new directory
    setup_enhanced_prompt
    
    # Auto-suggest environment activation (but don't spam)
    if [[ -z "$NEURO_LAST_SUGGESTION_DIR" || "$NEURO_LAST_SUGGESTION_DIR" != "$(pwd)" ]]; then
        suggest_env_activation
        export NEURO_LAST_SUGGESTION_DIR="$(pwd)"
    fi
    
    # Quick dependency check in background
    auto_check_deps
}

# Override cd command to include our hook
cd() {
    builtin cd "$@" && cd_hook
}

# Python execution wrapper to catch import errors
python3() {
    local output
    local exit_code
    
    # Run python and capture output
    output=$(command python3 "$@" 2>&1)
    exit_code=$?
    
    # Check for import errors and suggest installations
    if [[ $exit_code -ne 0 && "$output" =~ "ModuleNotFoundError: No module named" ]]; then
        echo "$output"
        
        # Extract module name
        local module_name=$(echo "$output" | grep -o "No module named '[^']*'" | sed "s/No module named '\\([^']*\\)'/\\1/")
        
        if [[ -n "$module_name" ]]; then
            echo ""
            echo "💡 Missing module detected: $module_name"
            
            # Try to suggest installation
            if command -v python3 >/dev/null 2>&1 && [[ -f "neuro/terminal_agent.py" ]]; then
                local suggestion=$(python3 -c "
from neuro.terminal_agent import TerminalAgent
agent = TerminalAgent()
print(agent._suggest_install_command('$module_name'))
" 2>/dev/null)
                
                if [[ -n "$suggestion" ]]; then
                    echo "   Suggested fix: $suggestion"
                fi
            else
                echo "   Try: pip install $module_name"
            fi
        fi
    else
        echo "$output"
    fi
    
    return $exit_code
}

# Initialize the enhanced terminal system (lightweight)
init_enhanced_terminal() {
    echo "🧙 Initializing Small-Mind Enhanced Terminal..."
    
    # Set up aliases (fast)
    setup_smart_aliases
    
    # Quick project detection (no heavy Python)
    local project_type="$(detect_project_type)"
    export NEURO_PROJECT_TYPE="$project_type"
    
    # Set up basic prompt styling
    if [[ -z "$ORIGINAL_PS1" ]]; then
        export ORIGINAL_PS1="$PS1"
    fi
    
    # Show welcome message
    echo "✅ Enhanced terminal features enabled!"
    echo "   🔧 Type 'pip-deps' to check dependencies"
    echo "   📊 Type 'project-info' for current project status"
    echo "   🧙 Type 'neuro-organize' to organize your files"
    echo "   📁 Project type detected: $project_type"
    echo ""
}

# Auto-initialize if this script is sourced
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    init_enhanced_terminal
fi
