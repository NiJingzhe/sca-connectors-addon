#!/bin/sh
# Fetch the FEMaster release binary for this platform into vendor/.
# Usage: ./tools/fetch_femaster.sh [version]
set -e

VERSION="${1:-v2.8.0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/vendor"

OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS-$ARCH" in
  Darwin-arm64) ASSET="FEMaster-macos-arm64-omp-accelerate.tar.gz" ;;
  Darwin-x86_64) echo "no official macOS x86_64 release asset; build from source" >&2; exit 1 ;;
  Linux-x86_64) ASSET="FEMaster-linux-mkl-omp.tar.gz" ;;
  Linux-aarch64) echo "no official linux-aarch64 release asset; build from source" >&2; exit 1 ;;
  *) echo "unsupported platform $OS-$ARCH" >&2; exit 1 ;;
esac

URL="https://github.com/Luecx/FEMaster/releases/download/$VERSION/$ASSET"
echo "fetching $URL"
curl -sL -o "$ROOT/vendor/$ASSET" "$URL"
tar -xzf "$ROOT/vendor/$ASSET" -C "$ROOT/vendor"
xattr -d com.apple.quarantine "$ROOT/vendor/FEMaster" 2>/dev/null || true
chmod +x "$ROOT/vendor/FEMaster"
"$ROOT/vendor/FEMaster" --version | tail -1
echo "OK: $ROOT/vendor/FEMaster"
