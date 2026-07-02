#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

if ! command -v cmake >/dev/null 2>&1; then
  echo "[ERROR] cmake not found. Install: sudo apt update && sudo apt install -y cmake"
  exit 1
fi

if ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
  echo "[ERROR] No C++ compiler found. Install: sudo apt update && sudo apt install -y build-essential"
  exit 1
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --config Release

BIN="${BUILD_DIR}/ttl_schedule_search_demo"
if [[ ! -x "${BIN}" ]]; then
  BIN="${BUILD_DIR}/Release/ttl_schedule_search_demo"
fi

if [[ $# -eq 0 ]]; then
  echo "[INFO] Running default schedule search demo (workload=small)."
  "${BIN}" \
    --workload small \
    --warmup 1 \
    --iters 2 \
    --random-budget 40 \
    --hill-steps 6 \
    --max-threads 8 \
    --grid-max-evals 150
else
  "${BIN}" "$@"
fi
