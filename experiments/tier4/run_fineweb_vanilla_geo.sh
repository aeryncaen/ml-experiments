#!/usr/bin/env bash
set -euo pipefail

export GEO_PREBIAS_ENABLE=1
export GEO_PREBIAS_METHOD=kl_bucket_mtp
export GEO_PREBIAS_MTP_WEIGHTS=1.0,0.5,0.25
export GEO_PREBIAS_BLEND=0.75
export GEO_PREBIAS_RANK=1
export GEO_PREBIAS_MAX_TOKENS=50000000

export GEO_ATTN_BIAS_ENABLE=1
export GEO_ATTN_BIAS_BLEND=0.3125
export GEO_ATTN_BIAS_RANK=1

export GEO_ATTN_CORR_BIAS_ENABLE=1
export GEO_ATTN_CORR_BLEND=1.0
export GEO_ATTN_CORR_RANK=1
# 0 means auto: round(0.75 * N_LAYER)
export GEO_ATTN_CORR_LAYERS=0
export GEO_ATTN_CORR_HORIZONS=1,2,3
export GEO_ATTN_CORR_HORIZON_WEIGHTS=1.0,0.5,0.25

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "experiments/tier4/train_fineweb_vanilla.py" "$@"
