#!/usr/bin/env bash
# Strip Windows CRLF from shell scripts (run on Linux after git pull from Windows).
#   bash scripts/stage1/fix_shell_crlf.sh
set -eu
cd "$(dirname "$0")/../.."
find scripts/stage1 scripts/server -name '*.sh' -print0 2>/dev/null \
  | xargs -0 sed -i 's/\r$//' 2>/dev/null || true
[[ -f scripts/server/config/eval_config.env ]] && sed -i 's/\r$//' scripts/server/config/eval_config.env
echo "CRLF stripped from scripts/stage1/*.sh and eval_config.env"
