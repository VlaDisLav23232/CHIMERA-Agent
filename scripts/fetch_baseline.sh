#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/external/chimera-agent-baseline"
COMMIT=1afa788efc21c0e0144706d75170b0a4cd267367
if [ -e "$DIR" ]; then
  [ "$(git -C "$DIR" remote get-url origin)" = https://github.com/DIAGNijmegen/chimera-agent-baseline ] || { echo 'Unexpected baseline origin' >&2; exit 1; }
  # Upstream tracks bytecode; imports can change it without changing source.
  [ -z "$(git -C "$DIR" -c filter.lfs.process= -c filter.lfs.required=false status --porcelain -- . ':(exclude)**/__pycache__/**' ':(exclude)model/embedding_model/**')" ] || { echo 'Baseline checkout has local changes' >&2; exit 1; }
  [ "$(git -C "$DIR" rev-parse HEAD)" = "$COMMIT" ] || { echo 'Unexpected baseline commit' >&2; exit 1; }
else
  git clone -q https://github.com/DIAGNijmegen/chimera-agent-baseline "$DIR"
  git -C "$DIR" checkout -q "$COMMIT"
fi
git -C "$DIR" rev-parse HEAD
