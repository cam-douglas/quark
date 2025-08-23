#!/usr/bin/env bash
# smctl — Small-Mind Control: Cursor-like CLI router
set -euo pipefail
ROOT="/Users/camdouglas/quark"
PY="$ROOT/env/bin/python"
[[ -x "$PY" ]] || PY="/opt/homebrew/bin/python3"

CMD="${1:-help}"; shift || true

case "$CMD" in
  ask|plan|run|parallel|list|describe|shell)
    exec "$PY" -m agent_hub.cli "$CMD" "$@"
    ;;
  *)
    echo "smctl commands:
  ask \"prompt\" [--tools shell,fs,python] [--allow-shell] [--sudo-ok]
  plan \"goal\" [--parallel N]
  run --model <id> -- task args...
  parallel \"cmd1\" \"cmd2\" ...
  list
  describe <model_id>
  shell            # open agent shell (OI if installed)
"; exit 1
    ;;
esac

chmod +x smctl