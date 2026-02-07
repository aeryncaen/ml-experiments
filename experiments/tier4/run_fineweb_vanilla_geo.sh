#!/usr/bin/env bash
set -euo pipefail

export GEO_PREBIAS_ENABLE=1
export GEO_PREBIAS_METHOD=kl_bucket_mtp
export GEO_PREBIAS_MTP_WEIGHTS=1.0,0.5,0.25
export GEO_PREBIAS_BLEND=0.75
export GEO_PREBIAS_MAX_TOKENS=50000000

exec torchrun --standalone --nproc_per_node=8 "experiments/tier4/train_fineweb_vanilla.py" "$@"
