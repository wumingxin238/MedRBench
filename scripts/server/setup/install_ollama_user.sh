#!/bin/bash
# Install Ollama into ~/bin (no root). Run: bash scripts/server/setup/install_ollama_user.sh
set -eu

INSTALL_DIR="${OLLAMA_INSTALL_DIR:-$HOME/lib/ollama}"
BIN_DIR="$HOME/bin"
mkdir -p "$INSTALL_DIR" "$BIN_DIR"

TMP="$(mktemp -d)"
cd "$TMP"

echo "==> Downloading Ollama linux amd64..."
if command -v curl >/dev/null 2>&1; then
  curl -fsSL -o ollama.tgz "https://ollama.com/download/ollama-linux-amd64.tgz"
elif command -v wget >/dev/null 2>&1; then
  wget -q -O ollama.tgz "https://ollama.com/download/ollama-linux-amd64.tgz"
else
  echo "Need curl or wget"
  exit 1
fi

tar -xzf ollama.tgz
if [ -f bin/ollama ]; then
  cp -f bin/ollama "$BIN_DIR/ollama"
elif [ -f ollama ]; then
  cp -f ollama "$BIN_DIR/ollama"
else
  echo "Unexpected tarball layout:"
  ls -laR
  exit 1
fi

chmod +x "$BIN_DIR/ollama"

if ! grep -q 'export PATH="$HOME/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null; then
  echo 'export PATH="$HOME/bin:$PATH"' >> "$HOME/.bashrc"
fi
export PATH="$HOME/bin:$PATH"

echo "==> Installed: $(which ollama)"
"$BIN_DIR/ollama" --version || true
echo ""
echo "Next:"
echo "  export PATH=\"\$HOME/bin:\$PATH\""
echo "  nohup ollama serve > ~/ollama.log 2>&1 &"
echo "  ollama pull qwen2.5:14b-instruct"

rm -rf "$TMP"
