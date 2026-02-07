#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
export MODEL_TYPE=s6
export S6_SCAN_STATE_MODES=${S6_SCAN_STATE_MODES:-elementwise,elementwise,elementwise}
export BATCH_SIZE=${BATCH_SIZE:-2}
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "experiments/tier4/train_fineweb_vanilla.py" "$@"
