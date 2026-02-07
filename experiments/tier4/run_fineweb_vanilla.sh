#!/usr/bin/env bash
set -euo pipefail

exec torchrun --standalone --nproc_per_node=8 "experiments/tier4/train_fineweb_vanilla.py" "$@"
