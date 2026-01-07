#!/usr/bin/env bash
set -e

MODE="${1:-train}"

echo "[docker] mode = $MODE"

python -m src.run mode=$MODE
