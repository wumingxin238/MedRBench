#!/bin/bash
# Fix Windows CRLF on uploaded shell scripts (run once after Xftp upload)
set -eu

SCRIPTS_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
find "$SCRIPTS_ROOT" -name '*.sh' -type f | while read -r f; do
  sed -i 's/\r$//' "$f" 2>/dev/null || sed -i '' 's/\r$//' "$f"
  chmod +x "$f"
  echo "fixed: $f"
done
