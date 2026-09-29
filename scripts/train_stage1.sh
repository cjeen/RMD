#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "${PYTHON_BIN:-python}" -m torch.distributed.run --standalone \
  --nproc_per_node="${NPROC_PER_NODE:-8}" train.py \
  --config_path configs/stage1.yaml --logdir "${LOGDIR:-logs/stage1}" "$@"
