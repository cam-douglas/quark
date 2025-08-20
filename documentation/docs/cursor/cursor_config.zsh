#!/bin/zsh
# =============================================================================
# Cursor Configuration - Consolidated
# =============================================================================
# Purpose: Unified configuration for Cursor IDE environment
# Inputs: None (self-contained)
# Outputs: Configured shell environment with Python management
# Seeds: Deterministic PATH and environment setup
# Dependencies: zsh, Python 3.x, pyenv/asdf (optional)

# =============================================================================
# CORE CONFIGURATION
# =============================================================================

# Prevent Python venv from injecting "(venv)" into the prompt
export VIRTUAL_ENV_DISABLE_PROMPT=1

# =============================================================================
# PATH CONFIGURATION (deduplicated)
# =============================================================================
typeset -Ua _P; _P=(
  /opt/homebrew/bin
  /usr/local/bin
  /usr/bin
  /bin
  /usr/sbin
  /sbin
  $HOME/.pyenv/shims
  $HOME/.local/bin
)
export PATH="${(j.:.)_P}"

# =============================================================================
# SHELL OPTIONS
# =============================================================================
setopt AUTO_CD AUTO_PARAM_KEYS
setopt NO_BEEP
setopt HIST_IGNORE_DUPS SHARE_HISTORY
setopt NO_BG_NICE
unsetopt NOMATCH

# =============================================================================
# COLOR CONFIGURATION
# =============================================================================
autoload -Uz colors && colors

# Portable color setup
if command -v tput >/dev/null 2>&1; then
  export C_RST="$(tput sgr0 || true)"
  export C_RED="$(tput setaf 1 || true)"
  export C_GRN="$(tput setaf 2 || true)"
  export C_YLW="$(tput setaf 3 || true)"
  export C_BLU="$(tput setaf 4 || true)"
  export C_MAG="$(tput setaf 5 || true)"
  export C_CYN="$(tput setaf 6 || true)"
else
  C_RST=$'%{\e[0m%}'; C_RED=$'%{\e[31m%}'; C_GRN=$'%{\e[32m%}'
  C_YLW=$'%{\e[33m%}'; C_BLU=$'%{\e[34m%}'; C_MAG=$'%{\e[35m%}'; C_CYN=$'%{\e[36m%}'
fi

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================
: ${QUIET_LOGIN:=1}
log_info() { [ "${QUIET_LOGIN}" = "0" ] && print -r -- "${C_BLU}ℹ${C_RST} $*"; }
log_ok()   { print -r -- "${C_GRN}✅${C_RST} $*"; }
log_warn() { print -r -- "${C_YLW}⚠${C_RST} $*"; }
log_err()  { print -r -- "${C_RED}❌${C_RST} $*"; }
log_venv() { print -r -- "${C_MAG}🐍${C_RST} $*"; }

# =============================================================================
# PROMPT CONFIGURATION
# =============================================================================
setopt PROMPT_SUBST

# Virtual environment segment
venv_seg() {
  [[ -n "$VIRTUAL_ENV" ]] && print -r -- "${C_CYN}(${VIRTUAL_ENV:t})${C_RST} "
}

# Python version segment
python_seg() {
  local pv; pv="$(python3 -V 2>/dev/null | awk '{print $2}')" || pv="N/A"
  print -r -- "[py:${pv}]"
}

# Main prompt
PROMPT="%n@%m:%/ \$(venv_seg)\$(python_seg) %# "

# =============================================================================
# SAFETY EXECUTOR
# =============================================================================
cx_safe_exec() {
  local cmd="$*"
  local cwd="${PWD:A}"
  local venv="${VIRTUAL_ENV:-none}"
  local deny_regexes=(
    '(^|[[:space:]])rm[[:space:]]+(-[^-]*f[^-]*r|-rf)[[:space:]]'
    '(^|[[:space:]])(chown|chmod)[[:space:]]+-R[[:space:]]'
    '(^|[[:space:]])diskutil[[:space:]].*(erase|repartition)'
    '(^|[[:space:]])curl[[:space:]].*\|[[:space:]]*(sh|bash|zsh)'
    '(^|[[:space:]])ssh[[:space:]].*StrictHostKeyChecking=no'
    '(^|[[:space:]])sudo[[:space:]].*(passwd|shadow|pam|auth|csrutil)'
  )
  for rx in "${deny_regexes[@]}"; do
    if [[ "$cmd" =~ ${~rx} ]]; then
      log_warn "Auto-exec blocked by safety policy."
      print -r -- "CWD: ${cwd}"
      print -r -- "ENV: ${venv}"
      print -r -- "COMMAND (review & paste to run manually):"
      print -r -- "```"
      print -r -- "$cmd"
      print -r -- "```"
      return 77
    fi
  done
  log_info "Exec: $cmd"
  eval "$cmd"; local rc=$?
  if [[ $rc -eq 0 ]]; then log_ok "Done (rc=$rc)"; else log_err "Failed (rc=$rc)"; fi
  return $rc
}

# =============================================================================
# DEPENDENCY MANAGEMENT
# =============================================================================

# Detect dependency manager
dep#detect_manager() {
  if [[ -f "pyproject.toml" ]]; then
    if command -v poetry >/dev/null 2>&1; then print -r -- "poetry"; return 0; fi
    print -r -- "pyproject"; return 0
  fi
  [[ -f "Pipfile" ]] && { print -r -- "pipenv"; return 0; }
  [[ -f "requirements.txt" ]] && { print -r -- "requirements"; return 0; }
  [[ -f "environment.yml" ]] && { print -r -- "conda"; return 0; }
  return 1
}

# Parse Python version requirement
dep#parse_python_requirement() {
  if [[ -f "pyproject.toml" ]]; then
    awk -F '[= ]+' '/^\s*python\s*=/{gsub(/["'\'']/, "", $3); print $3; exit}' pyproject.toml 2>/dev/null && return 0
  fi
  if [[ -f "environment.yml" ]]; then
    awk '/- python(=|==|>=)/{gsub("- ",""); print $2; exit}' environment.yml 2>/dev/null && return 0
  fi
  if [[ -f "Pipfile" ]]; then
    awk -F'=' '/python_version/{gsub(/["'\'']/, "", $2); gsub(/ /,"",$2); print $2; exit}' Pipfile 2>/dev/null && return 0
  fi
  [[ -f ".python-version" ]] && { head -n1 .python-version; return 0; }
  return 1
}

# Ensure Python version
dep#ensure_python_version() {
  local want; want="$1"
  local default="3.11.9"
  local target="${want:-$default}"

  if command -v pyenv >/dev/null 2>&1; then
    pyenv install -s "$target"
    pyenv local "$target" 2>/dev/null || pyenv global "$target" 2>/dev/null || true
    export PATH="$HOME/.pyenv/shims:$PATH"
    return 0
  elif command -v asdf >/dev/null 2>&1; then
    asdf plugin add python >/dev/null 2>&1 || true
    asdf list python | grep -qE "^${target//./\\.}" || asdf install python "$target"
    asdf local python "$target" 2>/dev/null || asdf global python "$target" 2>/dev/null || true
    return 0
  else
    python3 -V >/dev/null 2>&1 || log_warn "python3 not found. Install Xcode CLT or Homebrew python."
  fi
}

# Sync dependencies
dep#sync_deps() {
  local mgr; mgr="$(dep#detect_manager)" || return 0
  case "$mgr" in
    poetry)       cx_safe_exec poetry install --no-interaction ;;
    pyproject)    [[ -f "requirements.txt" ]] && cx_safe_exec pip install -r requirements.txt ;;
    pipenv)       cx_safe_exec pipenv install --dev ;;
    requirements) cx_safe_exec pip install -r requirements.txt ;;
    conda)        [[ -f "environment.yml" ]] && cx_safe_exec conda env update --file environment.yml --prune ;;
  esac
}

# =============================================================================
# VIRTUAL ENVIRONMENT MANAGEMENT
# =============================================================================

# Check if current directory is a Python project
_autoenv#is_python_project() {
  [[ -f "pyproject.toml" || -f "requirements.txt" || -d ".venv" || -d "venv" || -f "Pipfile" || -f "environment.yml" ]]
}

# Discover virtual environment path
_autoenv#discover_path() {
  if [[ -d ".venv/bin" ]]; then print -r -- "$PWD/.venv"; return 0; fi
  if [[ -d "venv/bin"  ]]; then print -r -- "$PWD/venv";  return 0; fi
  if command -v poetry >/dev/null 2>&1; then
    local p; p="$(poetry env info --path 2>/dev/null)"
    [[ -n "$p" && -d "$p" ]] && print -r -- "$p" && return 0
  fi
  if command -v pipenv >/dev/null 2>&1; then
    local p; p="$(pipenv --venv 2>/dev/null)"
    [[ -n "$p" && -d "$p" ]] && print -r -- "$p" && return 0
  fi
  if [[ -f "environment.yml" ]] && command -v conda >/dev/null 2>&1; then
    local name; name="$(awk '/name:/{print $2; exit}' environment.yml 2>/dev/null)"
    if [[ -n "$name" ]]; then
      local base; base="$(conda info --base 2>/dev/null)"
      [[ -n "$base" && -d "$base/envs/$name" ]] && { print -r -- "$base/envs/$name"; return 0; }
    fi
  fi
  return 1
}

# Activate virtual environment
autoenv#activate_path() {
  local path="$1"
  [[ -n "$path" && -d "$path" ]] || return 1
  [[ "$VIRTUAL_ENV" == "$path" ]] && return 0
  if [[ -f "$path/bin/activate" ]]; then
    source "$path/bin/activate"
  elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "$path" 2>/dev/null || return 1
  else
    return 1
  fi
  log_venv "Activated: ${path}"
  return 0
}

# =============================================================================
# HOOKS AND AUTOMATION
# =============================================================================

# Preflight check on every prompt
dep#preflight() {
  # Keep Python consistent
  local want; want="$(dep#parse_python_requirement)" || true
  dep#ensure_python_version "$want" || true

  # Auto-activate env if in a Python project
  if _autoenv#is_python_project; then
    local vpath; vpath="$(_autoenv#discover_path)" || true
    [[ -n "$vpath" ]] && autoenv#activate_path "$vpath" || true
  fi

  # Sync deps (best-effort)
  dep#sync_deps

  # PATH dedupe
  typeset -Ua _dedup; _dedup=(${(s.:.)PATH}); export PATH="${(j.:.)_dedup}"
}

# On directory change
dep#on_dir_change() {
  local want; want="$(dep#parse_python_requirement)" || true
  dep#ensure_python_version "$want" || true
  if _autoenv#is_python_project; then
    local vpath; vpath="$(_autoenv#discover_path)" || true
    [[ -n "$vpath" ]] && autoenv#activate_path "$vpath" || true
  fi
}

# Register hooks
autoload -Uz add-zsh-hook
add-zsh-hook chpwd  dep#on_dir_change
add-zsh-hook precmd dep#preflight

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Quick project setup
cursor#setup_project() {
  local project_type="${1:-python}"
  case "$project_type" in
    python)
      [[ -f "pyproject.toml" ]] || {
        log_info "Creating pyproject.toml..."
        cat > pyproject.toml <<EOF
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "$(basename $PWD)"
version = "0.1.0"
description = "Project description"
requires-python = ">=3.11"
dependencies = []

[tool.poetry]
name = "$(basename $PWD)"
version = "0.1.0"
description = "Project description"
authors = ["Your Name <your.email@example.com>"]
python = "^3.11"
EOF
      }
      [[ -d ".venv" ]] || {
        log_info "Creating virtual environment..."
        python3 -m venv .venv
      }
      log_ok "Python project setup complete"
      ;;
    *)
      log_warn "Unknown project type: $project_type"
      ;;
  esac
}

# =============================================================================
# INITIALIZATION
# =============================================================================

# Kill login noise
unset MAIL MAILPATH MAILCHECK

# Initial preflight
dep#preflight

log_ok "Cursor configuration loaded"
