#!/usr/bin/env bash
# source 
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ! -f "$HOME/.cargo/env" ]]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
source "$HOME/.cargo/env"
cargo --version

uv pip install -e "python"
