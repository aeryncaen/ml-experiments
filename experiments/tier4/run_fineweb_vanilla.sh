#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "experiments/tier4/train_fineweb_vanilla.py" "$@"
