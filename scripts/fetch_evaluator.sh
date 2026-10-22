#!/usr/bin/env bash
set -euo pipefail
COMMIT=92365b94b0eca90daec9021ffdb18c581dcc4619
DIR="$(cd "$(dirname "$0")/.." && pwd)/external/CHIMERA-agent"
[ -d "$DIR/.git" ] || git clone -q https://github.com/DIAGNijmegen/CHIMERA-agent "$DIR"
git -C "$DIR" fetch -q origin "$COMMIT"
git -C "$DIR" checkout -q "$COMMIT"
echo "evaluator at $(git -C "$DIR" rev-parse --short HEAD)"
