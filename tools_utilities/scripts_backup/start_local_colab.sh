#!/usr/bin/env bash
set -euo pipefail

# Config
PORT="${PORT:-8888}"
ENV_DIR="${ENV_DIR:-./env}"
REQS="${REQS:-requirements-colab.txt}"  # or requirements-colab-metal.txt on Apple

# Create venv if missing
if [ ! -d "$ENV_DIR" ]; then
  python3 -m venv "$ENV_DIR"
fi

# Activate
source "$ENV_DIR/bin/activate"

# Upgrade pip
python -m pip install --upgrade pip wheel

# Install deps
pip install -r "$REQS"

# Enable Colab-over-WS
python -m jupyter serverextension enable --py jupyter_http_over_ws

# Launch Notebook server that accepts connections from Colab
echo "Starting Jupyter at http://localhost:${PORT}"
exec python -m notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port="${PORT}" \
  --no-browser \
  --NotebookApp.disable_check_xsrf=True
