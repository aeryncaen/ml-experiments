import glob
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.profiler
from einops import rearrange
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from kernels import flash_attn_func, feature_attention, HAS_CUTE_FLASH as HAS_FLASH_ATTN

from s6 import USBBlock, USBConfig


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None else default


@dataclass
class HParams:
    data_path: str = os.environ.get("DATA_PATH", str(Path(__file__).resolve().parent))
    data_format: str = os.environ.get("DATA_FORMAT", "gpt2")  # gpt2 (uint16+header) | yamit (flat uint32)
    train_files: str = os.environ.get(
        "TRAIN_FILES",
        os.path.join(data_path, "train/**/*.bin") if data_format == "yamit"
        else os.path.join(data_path, "data/fineweb10B/fineweb_train_*.bin"),
    )
    val_files: str = os.environ.get(
        "VAL_FILES",
        os.path.join(data_path, "val/**/*.bin") if data_format == "yamit"
        else os.path.join(data_path, "data/fineweb10B/fineweb_val_*.bin"),
    )
    token_bytes_path: str = os.environ.get(
        "TOKEN_BYTES_PATH",
        str(Path(__file__).resolve().parent.parent / "yamit/tokenizer/artifacts/qwen3/token_bytes.npy")
        if data_format == "yamit" else "",
    )  # path to pre-built token_bytes (.npy/.pt)
    yamit_use_idx: bool = _env_bool("YAMIT_USE_IDX", True)  # insert EOS between docs using .idx offsets
    yamit_eos_token_id: int = _env_int("YAMIT_EOS_TOKEN_ID", 149727)

    model_type: str = os.environ.get("MODEL_TYPE", "transformer")  # transformer | feat_attn | fused_seq_feat | fused_qkv | three_stage | three_stage_fsa | qvo | dual_q | transformer_shift | transformer_gate | fused_gate | transformer_s4d | s6 | ulb | ulb_fa | ulb_2d | byte_attn | moe | ngpt_moe
    vocab_size: int = _env_int("VOCAB_SIZE", 50304)
    n_layer: int = _env_int("N_LAYER", 12)
    n_head: int = _env_int("N_HEAD", 12)
    n_kv_head: int = _env_int("N_KV_HEAD", 0)  # 0 = same as n_head (MHA); < n_head = GQA; 1 = MQA
    d_model: int = _env_int("D_MODEL", 768)
    seq_len: int = _env_int("SEQ_LEN", 2048)

    # 2D factorization (ulb_2d): 0 = auto-factor from d_model
    n_features: int = _env_int("N_FEATURES", 0)
    desc_dim: int = _env_int("DESC_DIM", 0)

    # Feature-attention MLP knobs (feat_attn model type)
    fa_n_features: int = _env_int("FA_N_FEATURES", 64)       # features to attend over (must divide hidden=2048)
    fa_activation: str = os.environ.get("FA_ACTIVATION", "softmax")  # softmax | silu | silu2
    fa_post_act: str = os.environ.get("FA_POST_ACT", "none")  # none | silu | gelu — element-wise activation after feature attention
    fa_pre_act: str = os.environ.get("FA_PRE_ACT", "none")   # none | silu | gelu — element-wise activation on gate before Q=K scores
    fa_qk_norm: bool = _env_bool("FA_QK_NORM", False)        # RMSNorm on Q=K before scoring
    feat_attn_mlp: bool = _env_bool("FEAT_ATTN_MLP", False)  # swap SwiGLU MLP for FeatureAttentionMLP in transformer/nGPT blocks

    # Mamba-3-inspired attention enhancements
    dd_rope: bool = _env_bool("DD_ROPE", False)    # data-dependent RoPE on half of head dims (Mamba-3 §3.2)
    trap_mix: bool = _env_bool("TRAP_MIX", False)  # trapezoidal K,V mixing: size-2 conv before attention (Mamba-3 §3.1)

    # Fused gate knobs (0 = use d_model, no expansion)
    fused_inner_dim: int = _env_int("FUSED_INNER_DIM", 0)

    # S6 knobs
    s6_headdim: int = _env_int("S6_HEADDIM", 64)
    s6_expansion_factor: int = _env_int("S6_EXPANSION_FACTOR", 2)
    s6_post_scan_attention: bool = _env_bool("S6_POST_SCAN_ATTENTION", True)
    s6_scan_state_modes: str = os.environ.get("S6_SCAN_STATE_MODES", "elementwise,elementwise,elementwise")

    train_steps: int = _env_int("TRAIN_STEPS", 2000)
    seed: int = _env_int("SEED", 1337)
    batch_size: int = _env_int("BATCH_SIZE", 8)
    grad_accum: int = _env_int("GRAD_ACCUM", 1)
    val_steps: int = _env_int("VAL_STEPS", 32)
    val_every: int = _env_int("VAL_EVERY", 100)
    lr: float = _env_float("LR", 3e-4)
    warmup_frac: float = _env_float("WARMUP_FRAC", 0.02)  # fraction of total tokens for warmup (no grad_accum during warmup)
    weight_decay: float = _env_float("WEIGHT_DECAY", 0.1)
    grad_clip: float = _env_float("GRAD_CLIP", 1.0)
    lr_schedule: str = os.environ.get("LR_SCHEDULE", "cosine")  # cosine | wsd | pressure | gnorm
    schedule_power: float = _env_float("SCHEDULE_POWER", 0.5)  # exponent on gnorm ratio for lr_schedule=gnorm (0.5=sqrt, 1.0=linear)
    wsd_decay_frac: float = _env_float("WSD_DECAY_FRAC", 0.1)  # fraction of total tokens for decay phase
    wsd_decay_end_frac: float = _env_float("WSD_DECAY_END_FRAC", 0.0)  # S phase end LR as fraction of start (0.1 = S decays to 10%, D then decays to 10% of that)
    grad_ckpt: bool = _env_bool("GRAD_CKPT", False)
    dtype: str = os.environ.get("DTYPE", "bf16")  # bf16 | fp32
    ch_loss: bool = _env_bool("CH_LOSS", False)  # contraharmonic mean loss across data sources
    ckpt_dir: str = os.environ.get("CKPT_DIR", "")  # directory for checkpoints; empty = no saving
    compile: bool = _env_bool("TORCH_COMPILE", True)
    torch_profile: bool = _env_bool("TORCH_PROFILE", False)
    torch_profile_steps: int = _env_int("TORCH_PROFILE_STEPS", 50)
    metrics_enabled: bool = _env_bool("METRICS_ENABLED", True)
    metrics_dir: str = os.environ.get("METRICS_DIR", str(Path(__file__).resolve().parent / "runs"))
    metrics_run_name: str = os.environ.get("METRICS_RUN_NAME", "")
    run_group: str = os.environ.get("RUN_GROUP", "")
    run_config_hash: str = os.environ.get("RUN_CONFIG_HASH", "")
    metrics_every: int = _env_int("METRICS_EVERY", 1)
    metrics_flush_every: int = _env_int("METRICS_FLUSH_EVERY", 20)

    # Optimizer selection
    optimizer: str = os.environ.get("OPTIMIZER", "adamw")  # adamw | muon | normuon | autonormuon | orion | muon_sf | normuon_sf | dion | dion2 | geoadam | geomuon | geonormuon
    muon_lr: float = _env_float("MUON_LR", 0.0)            # Muon/NorMuon LR for hidden 2D weights; 0 = auto (0.1/sqrt(2*n_layer))
    muon_momentum: float = _env_float("MUON_MOMENTUM", 0.95)
    normuon_beta2: float = _env_float("NORMUON_BETA2", 0.95)  # NorMuon per-row second moment EMA
    autonormuon_beta: float = _env_float("AUTONORMUON_BETA", 0.55)  # AutoNorMuon grad-norm EMA decay
    autonormuon_adaptation_scope: str = os.environ.get("AUTONORMUON_ADAPTATION_SCOPE", "neuron")  # neuron | matrix
    autonormuon_grad_schedule: str = os.environ.get("AUTONORMUON_GRAD_SCHEDULE", "off")  # always|off|hard:<step>|ramp:<s1>-<s2>|ppl[:<power>]
    autonormuon_weight_schedule: str = os.environ.get("AUTONORMUON_WEIGHT_SCHEDULE", "always")  # same format
    autonormuon_weight_mode: str = os.environ.get("AUTONORMUON_WEIGHT_MODE", "sphere")  # sphere | ns5
    autonormuon_gnorm_mode: str = os.environ.get("AUTONORMUON_GNORM_MODE", "ema")  # ema | accumulate
    autonormuon_ratio_pow: float = _env_float("AUTONORMUON_RATIO_POW", 1.0)
    autonormuon_min_ratio: float = _env_float("AUTONORMUON_MIN_RATIO", 0.0)
    spicydion_selection_sigma: float = _env_float("SPICYDION_SELECTION_SIGMA", 0.5)
    spicydion_ef_decay: float = _env_float("SPICYDION_EF_DECAY", 0.95)
    spicydion_norm_direction: str = os.environ.get("SPICYDION_NORM_DIRECTION", "col_row")  # col_row | row_col | row | col
    spicydion_turbo_prescale: bool = _env_bool("SPICYDION_TURBO_PRESCALE", True)
    spicydion_adaptive_lr_mode: str = os.environ.get("SPICYDION_ADAPTIVE_LR_MODE", "geomean")  # geomean | adam | ratio
    geomuon_ns_steps: int = _env_int("GEOMUON_NS_STEPS", 5)  # Newton-Schulz iterations for GeodesicMuon
    init_mode: str = os.environ.get("INIT_MODE", "default")  # default | sphere | ns5

    # Retract weights to the product of spheres after each optimizer step.
    # Each row of every 2D weight matrix is normalized to unit norm.
    # This is the useful part of nGPT without the full LERP/alpha machinery.
    sphere_retract: bool = _env_bool("SPHERE_RETRACT", False)

    # nGPT: normalized transformer on the hypersphere (Loshchilov et al. 2025)
    ngpt: bool = _env_bool("NGPT", False)
    ngpt_alpha_init: float = _env_float("NGPT_ALPHA_INIT", 0.0)    # 0 = auto -> 1/n_layers
    ngpt_alpha_scale: float = _env_float("NGPT_ALPHA_SCALE", 0.0)  # 0 = auto -> 1/sqrt(d_model)
    ngpt_qk_bias: bool = _env_bool("NGPT_QK_BIAS", False)          # add bias to Q and K projections
    ngpt_sqk_init: float = _env_float("NGPT_SQK_INIT", 1.0)       # QK scaling init
    ngpt_su_init: float = _env_float("NGPT_SU_INIT", 1.0)         # MLP u scaling init
    ngpt_sv_init: float = _env_float("NGPT_SV_INIT", 1.0)         # MLP v scaling init
    ngpt_sz_init: float = _env_float("NGPT_SZ_INIT", 1.0)         # logit scaling init

    # Hybrid architecture features (per-layer specialization for ngpt_moe)
    ngpt_diff_attn_n: int = _env_int("NGPT_DIFF_ATTN_N", 0)       # first N layers use differential attention
    ngpt_paired_odd: bool = _env_bool("NGPT_PAIRED_ODD", False)    # odd layers use paired head attention
    ngpt_skip_trap_even: bool = _env_bool("NGPT_SKIP_TRAP_EVEN", False)  # even layers skip TrapMix
    ngpt_window_layers: str = os.environ.get("NGPT_WINDOW_LAYERS", "")   # comma-sep layer indices for sliding window, e.g. "5,7"
    ngpt_window_size: int = _env_int("NGPT_WINDOW_SIZE", 512)     # sliding window size for window layers
    ngpt_embed_gate_layer: int = _env_int("NGPT_EMBED_GATE_LAYER", -1)  # layer index for content-dependent embed gate (-1=off)

    # Retokenize >16-byte tokens on load (no need to preprocess .bin files)
    retokenize: bool = _env_bool("RETOKENIZE", False)

    # Composite embedding (byte-factored)
    composite_embed: bool = _env_bool("COMPOSITE_EMBED", False)
    composite_token_dims: int = _env_int("COMPOSITE_TOKEN_DIMS", 8)  # per-token dims per byte slot (rest is shared)
    composite_lora: bool = _env_bool("COMPOSITE_LORA", False)
    composite_lora_rank: int = _env_int("COMPOSITE_LORA_RANK", 16)
    composite_conv: bool = _env_bool("COMPOSITE_CONV", False)

    # Decoder head (ML-Decoder style cross-attention over byte slots)
    decoder_head: bool = _env_bool("DECODER_HEAD", False)
    lm_head_type: str = os.environ.get("LM_HEAD_TYPE", "")  # "" = backward-compatible (DECODER_HEAD), else linear|decoder|linear48|pit|bucketed_pit
    decoder_head_vocab_chunk: int = _env_int("DECODER_HEAD_VOCAB_CHUNK", 0)  # 0 = full vocab per token chunk (matches linear-head chunking style)
    decoder_head_token_chunk: int = _env_int("DECODER_HEAD_TOKEN_CHUNK", 1024)
    pit_orth_init: bool = _env_bool("PIT_ORTH_INIT", True)
    pit_eps: float = _env_float("PIT_EPS", 1e-6)
    pit_min_diag: float = _env_float("PIT_MIN_DIAG", 1e-3)
    pit_n_buckets: int = _env_int("PIT_N_BUCKETS", 64)
    pit_top_k: int = _env_int("PIT_TOP_K", 8)
    pit_router_aux_weight: float = _env_float("PIT_ROUTER_AUX_WEIGHT", 0.01)
    pit_bucket_mode: str = os.environ.get("PIT_BUCKET_MODE", "hash")  # hash | semantic
    pit_bucket_labels: str = os.environ.get("PIT_BUCKET_LABELS", "")  # path to .npy with per-token cluster labels
    pit_bucket_centers: str = os.environ.get("PIT_BUCKET_CENTERS", "")  # path to .npy with cluster centers (router init)


    # Byte-attention MLP
    ba_n_byte_heads: int = _env_int("BA_N_BYTE_HEADS", 4)

    # MoE (Mixture of Experts)
    n_experts: int = _env_int("N_EXPERTS", 8)
    top_k: int = _env_int("TOP_K", 2)
    moe_bias_lr: float = _env_float("MOE_BIAS_LR", 0.001)  # loss-free balancing bias update rate

    moe_bypass: bool = _env_bool("MOE_BYPASS", False)  # add a null expert that skips MLP
    moe_shared: float = _env_float("MOE_SHARED", 0.5)  # fraction of hidden dim shared across experts (0=none)
    moe_n_group: int = _env_int("MOE_N_GROUP", 1)            # expert groups for group top-k (1=flat)
    moe_topk_group: int = _env_int("MOE_TOPK_GROUP", 1)      # groups to select from
    moe_scaling_factor: float = _env_float("MOE_SCALING_FACTOR", 1.0)  # routed weight scaling
    moe_dense_layers: int = _env_int("MOE_DENSE_LAYERS", 2)  # first N and last N layers use dense MLP instead of MoE

    # LLaDA (masked diffusion LM)
    llada: bool = _env_bool("LLADA", False)
    llada_subs: bool = _env_bool("LLADA_SUBS", True)
    llada_antithetic: bool = _env_bool("LLADA_ANTITHETIC", True)

    @property
    def is_causal(self) -> bool:
        return not self.llada


HP = HParams()


def _coerce_like(current: Any, value: Any) -> Any:
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, str):
        return str(value)
    return value


def resolve_hparams(overrides: dict[str, Any] | None = None) -> HParams:
    hp = HParams()
    if overrides:
        fields = hp.__dataclass_fields__
        for k, v in overrides.items():
            if k not in fields:
                raise KeyError(f"Unknown hparam override: {k}")
            cur = getattr(hp, k)
            setattr(hp, k, _coerce_like(cur, v))
    return hp


def set_hparams(overrides: dict[str, Any] | None = None) -> HParams:
    global HP
    HP = resolve_hparams(overrides)
    return HP


def run_training(overrides: dict[str, Any] | None = None) -> None:
    set_hparams(overrides)
    main()


def setup_dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return rank, world_size, torch.device("cuda", local_rank)
    return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print0(rank: int, s: str):
    if rank == 0:
        print(s, flush=True)


def _to_json_scalar(x):
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.detach().float().item()
        return x.detach().float().cpu().tolist()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_to_json_scalar(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_json_scalar(v) for k, v in x.items()}
    return str(x)


def _optimizer_group_metrics(optimizer) -> list[dict]:
    out = []
    for gi, pg in enumerate(optimizer.param_groups):
        rec = {
            "group_idx": gi,
            "use_muon": bool(pg.get("use_muon", False)),
        }
        for k in (
            "k",
            "lr",
            "scheduled_lr",
            "momentum",
            "weight_decay",
            "gnorm_ratio",
            "gnorm_ratio_raw",
            "gnorm_median",
            "gnorm_ema",
            "gnorm_max",
            "signal_ratio",
            "lr_mult",
            "gnorm_mean",
            "gnorm_std",
            "ratio_gnorm",
            "grad_gate",
            "weight_gate",
            "ppl_ratio",
            "loss_ema",
            "random_loss_ref",
            "raw_grad_row_norm_mean",
        ):
            if k in pg:
                rec[k] = _to_json_scalar(pg[k])
        if "betas" in pg:
            rec["betas"] = [float(pg["betas"][0]), float(pg["betas"][1])]
        out.append(rec)
    return out


def _open_metrics_run(rank: int, run_base: str, run_name: str, config: dict):
    if rank != 0:
        return None
    base = Path(run_base)
    base.mkdir(parents=True, exist_ok=True)
    final_dir = base / run_name
    # Write to a temp dir during training; atomic rename on success.
    # If we crash or get killed, the _tmp dir won't match skip_existing checks.
    tmp_dir = base / f".{run_name}_tmp"
    if tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    train_path = tmp_dir / "train_events.jsonl"
    eval_path = tmp_dir / "eval_events.jsonl"
    config_path = tmp_dir / "run_config.json"
    summary_path = tmp_dir / "run_summary.json"

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(_to_json_scalar(config), f, indent=2)

    return {
        "run_dir": tmp_dir,
        "final_dir": final_dir,
        "train_file": train_path.open("w", encoding="utf-8"),
        "eval_file": eval_path.open("w", encoding="utf-8"),
        "config_path": config_path,
        "summary_path": summary_path,
        "train_path": train_path,
        "eval_path": eval_path,
        "pending": 0,
        "best_val_loss": float("inf"),
        "best_step": None,
        "best_tokens_seen": None,
        "eval_count": 0,
        "train_count": 0,
        "last_train": None,
        "last_eval": None,
    }


def _finalize_metrics_run(metrics_state):
    """Atomic rename from temp dir to final dir. Called only on successful completion."""
    if metrics_state is None:
        return
    metrics_state["train_file"].close()
    metrics_state["eval_file"].close()
    tmp_dir = metrics_state["run_dir"]
    final_dir = metrics_state["final_dir"]
    if final_dir.exists():
        # Collision — append suffix
        for i in range(1, 10000):
            candidate = final_dir.parent / f"{final_dir.name}_{i:03d}"
            if not candidate.exists():
                final_dir = candidate
                break
    tmp_dir.rename(final_dir)
    metrics_state["run_dir"] = final_dir


def _write_metrics_event(metrics_state, which: str, event: dict, flush_every: int):
    if metrics_state is None:
        return
    fh = metrics_state[f"{which}_file"]
    fh.write(json.dumps(_to_json_scalar(event), sort_keys=True) + "\n")
    metrics_state["pending"] += 1
    if which == "train":
        metrics_state["train_count"] += 1
        metrics_state["last_train"] = event
    else:
        metrics_state["eval_count"] += 1
        metrics_state["last_eval"] = event
    if metrics_state["pending"] >= max(1, flush_every):
        metrics_state["train_file"].flush()
        metrics_state["eval_file"].flush()
        metrics_state["pending"] = 0


# Replacement map: token IDs whose byte sequences exceed 16 bytes.
# Each maps to its BPE decomposition into tokens that are all ≤16 bytes.
# Ported from modded-nanogpt/data/retokenize.go.
_RETOK_MAP: dict[int, list[int]] = {
    3880: [1783, 1783], 8864: [4181, 4181], 10052: [4770, 4770],
    10097: [1783, 1783, 1783, 1783], 10221: [4841, 4841],
    14827: [9364, 9364], 14950: [8184, 8184], 15171: [2424, 7992],
    16529: [220, 1783, 1783, 1783, 1783], 17174: [8412, 8412],
    19351: [1783, 650], 20368: [220, 1783, 1783], 20727: [555, 18789],
    22369: [1783, 982], 23090: [9364, 9364, 9364, 9364],
    23193: [4181, 4181, 4181, 4181], 23926: [4770, 4770, 4770, 4770],
    27006: [15243, 15243], 27193: [4841, 4841, 4841, 4841],
    27473: [5735, 20860], 27754: [4181, 2109], 28542: [16068, 16068],
    28719: [18717, 1286], 29113: [14468, 14468], 29146: [15864, 15864],
    29760: [15149, 28018], 29789: [16782, 5646], 30210: [29372, 18143],
    30213: [30212, 10049], 30542: [8184, 8184, 8184, 8184],
    30899: [21018, 30898], 30906: [30905, 21018, 30898],
    30982: [18717, 378], 31576: [22615, 31573], 32799: [2095, 1634],
    32941: [4841, 2602], 34400: [220, 1783], 35496: [9364] * 8,
    36174: [36173, 35992], 36573: [3753, 19541], 36658: [796, 4770],
    37389: [26825, 12100], 38093: [796, 4770, 4770, 4770, 4770],
    39172: [17811, 17811], 39177: [7449, 39142], 39753: [39752, 10493],
    39755: [39714, 39655], 39756: [24807, 31208], 39757: [17620, 29841],
    40242: [39693, 40241], 40586: [6142, 1023], 40800: [11784, 453],
    40887: [33131, 36387], 41380: [21353, 41215], 41436: [220, 1783, 650],
    41906: [220, 8412, 8412], 42045: [11273, 16607], 43453: [15831, 42202],
    43649: [37665, 41726], 43801: [1783, 1783, 1783, 982],
    44436: [17038, 1056], 44713: [220, 4181], 45545: [45544, 42983],
    45706: [22686, 22686], 46111: [796, 4770, 4770], 46674: [3753, 27781],
    47232: [1783, 1783, 1783], 47757: [13352, 34718], 48667: [14318, 20860],
    49129: [4181, 492], 49527: [20503, 20503], 49704: [27246, 27246],
}


def _retokenize(tokens: torch.Tensor) -> torch.Tensor:
    """Replace >16-byte token IDs with their shorter decompositions."""
    arr = tokens.numpy()
    # Fast path: check if any replacements are needed
    needs = set(_RETOK_MAP) & set(arr.tolist())
    if not needs:
        return tokens
    # Build expanded sequence
    out = []
    for t in arr:
        rep = _RETOK_MAP.get(int(t))
        if rep is not None:
            out.extend(rep)
        else:
            out.append(int(t))
    return torch.tensor(out, dtype=torch.uint16).pin_memory()


def _load_data_shard(file: Path) -> torch.Tensor:
    if HP.data_format == "yamit":
        # YAMIT format: flat uint32 tokens; optional .idx doc boundaries.
        import numpy as np
        tokens_np = np.fromfile(str(file), dtype=np.uint32)
        if HP.yamit_use_idx:
            idx_path = file.with_suffix(".idx")
            if idx_path.exists():
                offsets = np.fromfile(str(idx_path), dtype=np.uint64)
                if offsets.size >= 2:
                    tok_off = offsets // 4  # byte offsets -> uint32 token offsets
                    docs = []
                    eos = np.array([HP.yamit_eos_token_id], dtype=np.uint32)
                    for i in range(len(tok_off) - 1):
                        s = int(tok_off[i])
                        e = int(tok_off[i + 1])
                        if e > s:
                            docs.append(tokens_np[s:e])
                            docs.append(eos)
                    if docs:
                        tokens_np = np.concatenate(docs)
        return torch.from_numpy(tokens_np.astype(np.int64)).pin_memory()
    # GPT-2 format: 1024-byte header + uint16 tokens
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert int(header[0]) == 20240520, "magic number mismatch in .bin"
    assert int(header[1]) == 1, "unsupported .bin version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "token count mismatch"
    if HP.retokenize:
        tokens = _retokenize(tokens)
    return tokens


class ShardStream:
    def __init__(self, pattern: str, rank: int, world_size: int, seq_len: int, batch_size: int):
        self.files = [Path(f) for f in sorted(glob.glob(pattern, recursive=True))]
        if not self.files:
            raise FileNotFoundError(f"No files matched pattern: {pattern}")
        self.rank = rank
        self.world_size = world_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.tokens_per_rank = seq_len * batch_size
        self.tokens_per_global_step = self.tokens_per_rank * world_size
        self.file_idx = 0
        self.pos = 0
        self.tokens = _load_data_shard(self.files[self.file_idx])

    def _advance_shard(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = _load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def reset(self):
        """Reset to start of first shard (for deterministic val eval)."""
        self.file_idx = 0
        self.pos = 0
        self.tokens = _load_data_shard(self.files[0])

    def next_batch(self, device: torch.device):
        needed = self.tokens_per_global_step + 1
        if self.pos + needed >= self.tokens.numel():
            self._advance_shard()
        start = self.pos + self.rank * self.tokens_per_rank
        end = start + self.tokens_per_rank + 1
        buf = self.tokens[start:end]
        self.pos += self.tokens_per_global_step
        x = buf[:-1].to(dtype=torch.int64).view(self.batch_size, self.seq_len)
        y = buf[1:].to(dtype=torch.int64).view(self.batch_size, self.seq_len)
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


class _SourceBuffer:
    """Single data-source token stream. Returns one (seq_len+1) chunk at a time."""

    def __init__(self, files: list[Path], seq_len: int):
        self.files = files
        self.seq_len = seq_len
        self.file_idx = 0
        self.pos = 0
        self.tokens = _load_data_shard(self.files[0])
        self.n_tokens = sum(
            Path(f).stat().st_size // (4 if HP.data_format == "yamit" else 2)
            for f in self.files
        )

    def reset(self):
        self.file_idx = 0
        self.pos = 0
        self.tokens = _load_data_shard(self.files[0])

    def next_seq(self) -> tuple[torch.Tensor, torch.Tensor]:
        needed = self.seq_len + 1
        if self.pos + needed > self.tokens.numel():
            self.file_idx = (self.file_idx + 1) % len(self.files)
            self.tokens = _load_data_shard(self.files[self.file_idx])
            self.pos = 0
        buf = self.tokens[self.pos : self.pos + needed]
        self.pos += self.seq_len
        return buf[:-1].to(dtype=torch.int64), buf[1:].to(dtype=torch.int64)


class MixedShardStream:
    """Multi-source data stream with proportional mixing.

    Discovers source directories, creates per-source buffers, and samples
    sources proportionally (by token volume) for each sequence in a batch.
    Every batch contains a representative mix of all sources.
    """

    def __init__(self, pattern: str, rank: int, world_size: int,
                 seq_len: int, batch_size: int, seed: int = 42):
        import random as _random
        all_files = sorted(glob.glob(pattern, recursive=True))
        if not all_files:
            raise FileNotFoundError(f"No files matched pattern: {pattern}")

        # Group files by parent directory = source name
        sources: dict[str, list[Path]] = {}
        for f in all_files:
            p = Path(f)
            source = p.parent.name
            sources.setdefault(source, []).append(p)

        # If all files are in the same directory (flat layout), fall back to
        # treating the whole thing as one source
        if len(sources) == 1 and list(sources.keys())[0] in ("train", "val"):
            sources = {"all": [Path(f) for f in all_files]}

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rank = rank

        # Build per-source buffers
        self.source_names: list[str] = sorted(sources.keys())
        self.buffers: list[_SourceBuffer] = []
        self.weights: list[float] = []
        for name in self.source_names:
            buf = _SourceBuffer(sorted(sources[name]), seq_len)
            self.buffers.append(buf)
            self.weights.append(float(buf.n_tokens))

        total = sum(self.weights)
        self.probs = [w / total for w in self.weights]
        # Clamp: no source below 1% — prevents stale CH loss for rare sources
        min_prob = 0.01
        if any(p < min_prob and p > 0 for p in self.probs):
            self.probs = [max(p, min_prob) for p in self.probs]
            ptotal = sum(self.probs)
            self.probs = [p / ptotal for p in self.probs]
        self.n_sources = len(self.source_names)
        self._seed = seed + rank
        self.rng = _random.Random(self._seed)

        # For compatibility with code that checks .files
        self.files = [Path(f) for f in all_files]

    def reset(self):
        """Reset all source buffers and RNG for deterministic eval."""
        import random as _random
        for buf in self.buffers:
            buf.reset()
        self.rng = _random.Random(self._seed)

    def next_batch(self, device: torch.device):
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        src_ids: list[int] = []
        for _ in range(self.batch_size):
            src_idx = self.rng.choices(range(self.n_sources), weights=self.probs, k=1)[0]
            x, y = self.buffers[src_idx].next_seq()
            xs.append(x)
            ys.append(y)
            src_ids.append(src_idx)
        x = torch.stack(xs).to(device, non_blocking=True)
        y = torch.stack(ys).to(device, non_blocking=True)
        return x, y, torch.tensor(src_ids, dtype=torch.long, device=device)

    def source_summary(self) -> str:
        parts = []
        for name, prob in zip(self.source_names, self.probs):
            parts.append(f"{name}={prob:.3f}")
        return f"{self.n_sources} sources: {', '.join(parts)}"


class Mamba3Mixer(nn.Module):
    """Mamba-3-style mixer: chunked SSD with trapezoidal discretization,
    data-dependent RoPE on B/C, and BC bias.

    Quadratic form: Y = (decay_mask ⊙ C B^T) X  per chunk.
    C B^T is (chunk, chunk) via N-contraction — never materializes (P, N) per position.
    Inter-chunk state passing via sequential scan over chunk boundaries.
    """

    def __init__(self, d_model: int, n_head: int, d_state: int = 64):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head  # p
        self.d_state = d_state             # n

        # Input projections: x -> (X, B, C, dt, lambda, theta)
        Nr = d_state // 2
        proj_dim = (
            d_model             # X  (H * p)
            + n_head * d_state  # B  (H * n)
            + n_head * d_state  # C  (H * n)
            + n_head            # log_dt (H)
            + n_head            # lambda_logit (H)
            + n_head * Nr       # theta (H * n/2)
        )
        self.in_proj = nn.Linear(d_model, proj_dim, bias=False)

        # BC bias: learnable, head-specific, channel-wise, init=1
        self.b_bias = nn.Parameter(torch.ones(n_head, d_state))
        self.c_bias = nn.Parameter(torch.ones(n_head, d_state))
        self.b_bias._no_weight_decay = True   # type: ignore[attr-defined]
        self.c_bias._no_weight_decay = True   # type: ignore[attr-defined]

        # QK-norm on B,C
        self.b_norm = nn.RMSNorm(d_state)
        self.c_norm = nn.RMSNorm(d_state)

        # Scalar decay per head
        self.log_A = nn.Parameter(torch.log(0.5 * torch.ones(n_head)))
        self.log_A._no_weight_decay = True    # type: ignore[attr-defined]

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
        B, T, D = x.shape
        H, P, N = self.n_head, self.head_dim, self.d_state
        Nr = N // 2
        L = chunk_size
        assert T % L == 0
        K = T // L  # number of chunks

        # ---- project ----
        proj = self.in_proj(x)                                    # (B, T, proj_dim)
        i = 0
        X   = proj[..., i:i+H*P];  i += H*P
        Bv  = proj[..., i:i+H*N];  i += H*N
        Cv  = proj[..., i:i+H*N];  i += H*N
        ldt = proj[..., i:i+H];    i += H
        llam= proj[..., i:i+H];    i += H
        th  = proj[..., i:i+H*Nr]; i += H*Nr

        X   = rearrange(X,   'b t (h p) -> b t h p', h=H)
        Bv  = rearrange(Bv,  'b t (h n) -> b t h n', h=H)
        Cv  = rearrange(Cv,  'b t (h n) -> b t h n', h=H)
        ldt = rearrange(ldt, 'b t h -> b t h')
        llam= rearrange(llam,'b t h -> b t h')
        th  = rearrange(th,  'b t (h r) -> b t h r', h=H)

        # ---- discretization ----
        dt    = F.softplus(ldt).clamp(max=2.0)                    # (B, T, H)
        A_neg = self.log_A.exp().neg()                            # (H,) guaranteed negative
        alpha = torch.exp(dt * A_neg).clamp(max=0.999)            # (B, T, H) strict <1
        lam   = torch.sigmoid(llam)                               # (B, T, H)
        gamma = lam * dt                                          # (B, T, H)
        beta  = (1.0 - lam) * dt * alpha                         # (B, T, H)

        # ---- QK-norm + BC bias ----
        Bv = self.b_norm(Bv) * self.b_bias
        Cv = self.c_norm(Cv) * self.c_bias

        # ---- data-dependent RoPE on B, C ----
        cum_th = th.cumsum(dim=1)
        cos_th, sin_th = cum_th.cos(), cum_th.sin()
        def dd_rope(v: torch.Tensor) -> torch.Tensor:
            v1, v2 = v[..., :Nr], v[..., Nr:]
            return torch.cat([v1*cos_th - v2*sin_th, v1*sin_th + v2*cos_th], dim=-1)
        Bv = dd_rope(Bv)
        Cv = dd_rope(Cv)

        # ---- trapezoidal: apply size-2 conv to B before chunking ----
        # B_trap_t = γ_t * B_t + β_t * B_{t-1}   (and same scaling to X)
        # But we need to keep B and X separate for the SSD form.
        # The trap conv applies to the "KV" = B * x product in the recurrence,
        # but in SSD quadratic form Y = (mask ⊙ C B^T) X, the trap modifies B.
        # We scale: B_trap_t = γ_t * B_t,  and add β_t * B_{t-1} contribution
        # For the quadratic form we need to apply trap to both B and X consistently.
        # Simplest correct way: fold γ into B, β into a shifted-B term.
        Bv_g = Bv * gamma.unsqueeze(-1)                           # γ_t * B_t
        Bv_b = Bv * beta.unsqueeze(-1)                            # β_t * B_t (will shift)
        # Shift Bv_b by 1 position: Bv_b_shifted[t] = β_t * B_{t-1}
        # Bv_b is (B, T, H, N) — pad dim=1 (time): F.pad pads from last dim, so (0,0, 0,0, 1,0)
        Bv_b_shifted = F.pad(Bv_b[:, :-1], (0, 0, 0, 0, 1, 0))  # zero at t=0
        Bv_trap = Bv_g + Bv_b_shifted                             # (B, T, H, N)
        # Similarly for X: (B, T, H, P)
        X_g = X * gamma.unsqueeze(-1)
        X_b = X * beta.unsqueeze(-1)
        X_b_shifted = F.pad(X_b[:, :-1], (0, 0, 0, 0, 1, 0))
        X_trap = X_g + X_b_shifted                                # (B, T, H, P)

        # ---- chunk ----
        Bv_c  = rearrange(Bv_trap, 'b (k l) h n -> b k h l n', l=L)   # (B, K, H, L, N)
        Cv_c  = rearrange(Cv,      'b (k l) h n -> b k h l n', l=L)
        X_c   = rearrange(X_trap,  'b (k l) h p -> b k h l p', l=L)
        al_c  = rearrange(alpha,   'b (k l) h   -> b k h l',   l=L)

        # ---- intra-chunk decay mask (L, L) ----
        log_al = al_c.clamp(min=1e-6).log()                      # (B, K, H, L)
        log_cum = log_al.cumsum(dim=-1)                           # (B, K, H, L)
        # decay[i,j] = exp(cum[i] - cum[j]) for i>=j
        decay = (log_cum.unsqueeze(-1) - log_cum.unsqueeze(-2)).exp()
        decay = decay * torch.tril(torch.ones(L, L, device=x.device))  # (B, K, H, L, L)

        # ---- intra-chunk SSD: Y_intra = (decay ⊙ C B^T) X ----
        # C B^T: (B, K, H, L, L) via N-contraction  — this is the key: no (P,N) per position
        CB = torch.einsum('bkhin,bkhjn->bkhij', Cv_c, Bv_c)      # (B, K, H, L, L)
        attn = decay * CB                                         # (B, K, H, L, L)
        Y_intra = torch.einsum('bkhij,bkhjp->bkhip', attn, X_c)  # (B, K, H, L, P)

        # ---- inter-chunk: accumulate (N, P) states across chunks ----
        # Each chunk's contribution to the state: sum_l decay_to_end[l] * B[l] ⊗ X[l]
        # = B_c^T @ diag(decay_to_end) @ X_c   — (N, L) @ (L, L) @ (L, P) = (N, P)
        # decay_to_end[l] = exp(log_cum[L-1] - log_cum[l])
        decay_to_end = (log_cum[..., -1:] - log_cum).exp()        # (B, K, H, L)
        # chunk_state[k] = B_c^T @ diag(d2e) @ X_c = einsum('bkhln,bkhl,bkhlp->bkhnp')
        chunk_state = torch.einsum(
            'bkhln,bkhl,bkhlp->bkhnp', Bv_c, decay_to_end, X_c
        )                                                         # (B, K, H, N, P)
        chunk_decay = log_al.sum(dim=-1).exp()                    # (B, K, H) total decay per chunk

        # Sequential scan across K chunks
        prev_states = torch.zeros(B, K, H, N, P, device=x.device, dtype=x.dtype)
        h = torch.zeros(B, H, N, P, device=x.device, dtype=x.dtype)
        for k in range(K):
            prev_states[:, k] = h
            h = chunk_decay[:, k, :, None, None] * h + chunk_state[:, k]

        # ---- inter-chunk output contribution ----
        # For position l in chunk k: Y_inter = C[k,l] @ (decay_from_start[l] * prev_state[k]) @ ... hmm
        # Actually: Y_inter[k,l] = C[k,l]^T @ (decay_from_start[l] * h_prev[k]) ... but h is (N,P)
        # So Y_inter[k,l] in R^P = h_prev[k]^T @ (decay[l] * C[k,l])   with h (N,P), C (N,)
        # = (decay[l] * C[k,l])^T @ h_prev[k]  -> einsum over N -> (P,)
        decay_from_start = log_cum.exp()                          # (B, K, H, L)
        # C_scaled = decay_from_start * C
        C_scaled = Cv_c * decay_from_start.unsqueeze(-1)          # (B, K, H, L, N)
        Y_inter = torch.einsum(
            'bkhln,bkhnp->bkhlp', C_scaled, prev_states
        )                                                         # (B, K, H, L, P)

        # ---- combine and project out ----
        Y = Y_intra + Y_inter                                     # (B, K, H, L, P)
        Y = rearrange(Y, 'b k h l p -> b (k l) (h p)')           # (B, T, D)
        return self.out_proj(Y)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.register_buffer(
            "inv_freq",
            1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)),
            persistent=False,
        )
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor):
        t = x.shape[1]
        if self.cos_cached is None or t != self.seq_len_cached:
            self.seq_len_cached = t
            inv_freq = self.inv_freq.to(device=x.device)
            tt = torch.arange(t, device=x.device, dtype=inv_freq.dtype)
            freqs = torch.outer(tt, inv_freq)
            self.cos_cached = freqs.cos()[None, :, None, :]
            self.sin_cached = freqs.sin()[None, :, None, :]
        return self.cos_cached, self.sin_cached


class PairedRotary(nn.Module):
    """Rotary for paired heads: even/odd position encodings for 2*dim width.

    When applied to a tensor of last dim = 2*head_dim, the first head_dim
    channels get even-position encodings and the second head_dim channels
    get odd-position encodings. After reshape to 2T length, this gives
    adjacent heads interleaved positional identity.
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim  # head_dim (half of the paired width)
        self.register_buffer(
            "inv_freq",
            1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)),
            persistent=False,
        )
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor):
        t = x.shape[1]
        if self.cos_cached is None or t != self.seq_len_cached:
            self.seq_len_cached = t
            inv_freq = self.inv_freq.to(device=x.device)
            tt = torch.arange(t, device=x.device, dtype=inv_freq.dtype)
            t_even = 2 * tt
            t_odd = 2 * tt + 1
            freqs_even = torch.outer(t_even, inv_freq)
            freqs_odd = torch.outer(t_odd, inv_freq)
            # Concat along last dim: [even_cos | odd_cos], [even_sin | odd_sin]
            self.cos_cached = torch.cat([freqs_even.cos(), freqs_odd.cos()], dim=-1)[None, :, None, :]
            self.sin_cached = torch.cat([freqs_even.sin(), freqs_odd.sin()], dim=-1)[None, :, None, :]
        return self.cos_cached, self.sin_cached


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        # DD-RoPE: fully data-dependent RoPE on all head dims
        # (Mamba-3 §3.2: complex SSM ↔ dd-RoPE on B,C ↔ K,Q)
        self.dd_rope = HP.dd_rope
        hd = self.head_dim
        if self.dd_rope:
            assert hd % 2 == 0, f"head_dim ({hd}) must be even for dd-rope"
            # Project input → per-head rotation angles (hd/2 angles per head)
            self.theta_proj = nn.Linear(d_model, n_head * (hd // 2), bias=False)
        else:
            self.rotary = Rotary(hd)

        # Trapezoidal K,V mixing: size-2 data-dependent mixing before attention
        # (Mamba-3 §3.1: K_t' = γ_t K_t + β_t K_{t-1}, same for V)
        self.trap_mix = HP.trap_mix
        if self.trap_mix:
            # Project input → per-head (log_dt, lambda_logit, log_A) — all data-dependent
            self.trap_proj = nn.Linear(d_model, n_head * 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.q(x).view(b, t, self.n_head, self.head_dim)
        k = self.k(x).view(b, t, self.n_head, self.head_dim)
        v = self.v(x).view(b, t, self.n_head, self.head_dim)

        if self.dd_rope:
            # Data-dependent RoPE on all dims
            theta = self.theta_proj(x).view(b, t, self.n_head, self.head_dim // 2)
            cum_theta = theta.cumsum(dim=1)
            cos_dd = cum_theta.cos()
            sin_dd = cum_theta.sin()
            q = apply_rotary(q, cos_dd, sin_dd)
            k = apply_rotary(k, cos_dd, sin_dd)
        else:
            cos, sin = self.rotary(q)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

        if self.trap_mix:
            trap = self.trap_proj(x).view(b, t, self.n_head, 3)
            log_dt, lam_logit, log_A = trap[..., 0], trap[..., 1], trap[..., 2]
            dt = F.softplus(log_dt).clamp(max=2.0)               # (b, t, H)
            A_neg = -F.softplus(log_A)                            # (b, t, H) guaranteed negative
            alpha = torch.exp(dt * A_neg).clamp(max=0.999)       # (b, t, H)
            lam = torch.sigmoid(lam_logit)
            gamma = (lam * dt).unsqueeze(-1)                      # (b, t, H, 1)
            beta = ((1.0 - lam) * dt * alpha).unsqueeze(-1)       # (b, t, H, 1)

            k_shifted = F.pad(k[:, :-1], (0, 0, 0, 0, 1, 0))    # (b, t, H, D)
            k = gamma * k + beta * k_shifted
            v_shifted = F.pad(v[:, :-1], (0, 0, 0, 0, 1, 0))
            v = gamma * v + beta * v_shifted

        if HAS_FLASH_ATTN:
            y = flash_attn_func(q, k, v, causal=HP.is_causal)  # (b, t, h, d)
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=HP.is_causal)
            y = y.transpose(1, 2)
        y = y.contiguous().view(b, t, c)
        return self.proj(y)


class ShiftAttention(nn.Module):
    """Attention with temporal channel-group shifts on K and V.

    Within each head, d_head is split into groups. Each group's channels
    at position t come from a different past position (t, t-1, t-2, t-4).
    Each KV cache entry becomes a chimera encoding a temporal neighborhood.
    No new parameters. FlashAttention unchanged. Just a reindex.
    """

    def __init__(self, d_model: int, n_head: int, shifts: tuple[int, ...] = (0, 1, 2, 4)):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        assert self.head_dim % len(shifts) == 0, \
            f"head_dim {self.head_dim} must be divisible by {len(shifts)} shift groups"
        self.cpg = self.head_dim // len(shifts)  # channels per group
        self.shifts_list = shifts
        self.max_shift = max(shifts)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.head_dim)

    def _shift_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Shift channel groups within each head along the time axis.
        x: (B, T, H, D).  Group i gets shifted by shifts[i] positions causally.
        """
        if self.max_shift == 0:
            return x
        B, T, H, D = x.shape
        out = torch.zeros_like(x)
        for i, s in enumerate(self.shifts_list):
            sl = slice(i * self.cpg, (i + 1) * self.cpg)
            if s == 0:
                out[:, :, :, sl] = x[:, :, :, sl]
            else:
                out[:, s:, :, sl] = x[:, :T-s, :, sl]
                # out[:, :s, :, sl] stays zero — no history yet
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.q(x).view(b, t, self.n_head, self.head_dim)
        k = self.k(x).view(b, t, self.n_head, self.head_dim)
        v = self.v(x).view(b, t, self.n_head, self.head_dim)
        k = self._shift_kv(k)
        v = self._shift_kv(v)
        cos, sin = self.rotary(q)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if HAS_FLASH_ATTN:
            y = flash_attn_func(q, k, v, causal=HP.is_causal)
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=HP.is_causal)
            y = y.transpose(1, 2)
        y = y.contiguous().view(b, t, c)
        return self.proj(y)


class GatedNeighborAttention(nn.Module):
    """Attention with gated neighbor mixing on K.

    Half the channels in each head's K get a single lerp with t-1:
        k_out[t] = (1 - g[t]) * k[t] + g[t] * k[t-1]
    Gate is per-channel per-head, content-dependent.
    First half of channels stays untouched.
    """

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.half_dim = self.head_dim // 2

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.head_dim)
        self.gate_proj = nn.Linear(d_model, n_head * self.half_dim, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -2.0)

    def _gated_neighbor(self, k: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        half = self.half_dim
        k_static = k[:, :, :, :half]
        k_cur = k[:, :, :, half:]
        k_prev = F.pad(k_cur[:, :-1], (0, 0, 0, 0, 1, 0))
        k_mixed = (1 - gate) * k_cur + gate * k_prev
        return torch.cat([k_static, k_mixed], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.q(x).view(b, t, self.n_head, self.head_dim)
        k = self.k(x).view(b, t, self.n_head, self.head_dim)
        v = self.v(x).view(b, t, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        gate = torch.sigmoid(self.gate_proj(x)).view(b, t, self.n_head, self.half_dim)
        k = self._gated_neighbor(k, gate)
        if HAS_FLASH_ATTN:
            y = flash_attn_func(q, k, v, causal=HP.is_causal)
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=HP.is_causal)
            y = y.transpose(1, 2)
        y = y.contiguous().view(b, t, c)
        return self.proj(y)


class GatedNeighborBlock(nn.Module):
    """Separate attn + MLP block using GatedNeighborAttention."""
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = GatedNeighborAttention(d_model, n_head)
        self.ln2 = RMSNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class FusedGatedNeighborBlock(nn.Module):
    """Fused attention+MLP block. No separate MLP — one expand/contract cycle.

    x -> norm -> up_proj(d_model -> inner) -> SiLU -> h_up
             -> Q(inner -> inner)                      |
             -> K(inner -> inner)                      |
             -> V(inner -> inner)                      |
             -> gated neighbor on K                    |
             -> RoPE on Q, K                           |
             -> attention(Q, K, V) -> attn_out         |
             -> norm(attn_out) + h_up  <---------------+
             -> down_proj(inner -> d_model) -> residual

    SiLU after up_proj feeds rich features into QKV. Skip-add from h_up
    to normed attention output preserves pre-attn features for down_proj.
    """

    def __init__(self, d_model: int, n_head: int, inner_dim: int | None = None, expand: int = 1, paired: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.paired = paired
        self.inner_dim = inner_dim if inner_dim is not None else d_model * expand
        assert self.inner_dim % n_head == 0
        self.head_dim = self.inner_dim // n_head
        self.half_dim = self.head_dim // 2

        self.norm = RMSNorm(d_model)

        # Shared expansion: d_model -> inner_dim
        self.up_proj = nn.Linear(d_model, self.inner_dim, bias=False)

        # QKV projections from expanded space
        self.q_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)
        self.v_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)

        # Post-attention norm (before skip add)
        self.attn_norm = RMSNorm(self.inner_dim)

        # Down-project: inner_dim -> d_model
        self.down_proj = nn.Linear(self.inner_dim, d_model, bias=False)

        # Learnable Swish beta (per-channel)
        self.swish_beta_up = nn.Parameter(torch.ones(self.inner_dim))
        self.swish_beta_down = nn.Parameter(torch.ones(self.inner_dim))

        # QK norm (per-head RMSNorm before RoPE)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # RoPE — paired mode uses PairedRotary with 2*head_dim width
        if paired:
            assert n_head % 2 == 0, "paired heads requires even n_head"
            self.rotary = PairedRotary(self.head_dim)
        else:
            self.rotary = Rotary(self.head_dim)

        # Neighbor gate: single lerp on K, per-channel per-head
        self.neighbor_gate_proj = nn.Linear(d_model, n_head * self.half_dim, bias=True)
        nn.init.zeros_(self.neighbor_gate_proj.weight)
        nn.init.constant_(self.neighbor_gate_proj.bias, -2.0)

    def _gated_neighbor(self, k: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        half = self.half_dim
        k_static = k[:, :, :, :half]
        k_cur = k[:, :, :, half:]
        k_prev = F.pad(k_cur[:, :-1], (0, 0, 0, 0, 1, 0))
        k_mixed = (1 - gate) * k_cur + gate * k_prev
        return torch.cat([k_static, k_mixed], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        h = self.norm(x)

        # Expand and activate (Swish with learnable beta)
        h_up = self.up_proj(h)
        h_up = h_up * torch.sigmoid(self.swish_beta_up * h_up)

        # QKV from activated expanded space
        q = self.q_proj(h_up).view(b, t, self.n_head, self.head_dim)
        k = self.k_proj(h_up).view(b, t, self.n_head, self.head_dim)
        v = self.v_proj(h_up).view(b, t, self.n_head, self.head_dim)

        # QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Gated neighbor mixing on K (before pairing)
        gate = torch.sigmoid(self.neighbor_gate_proj(h)).view(b, t, self.n_head, self.half_dim)
        k = self._gated_neighbor(k, gate)

        if self.paired:
            # Merge adjacent head pairs: (b, t, n_head, hd) -> (b, t, n_head//2, hd*2)
            n2 = self.n_head // 2
            q = q.view(b, t, n2, self.head_dim * 2)
            k = k.view(b, t, n2, self.head_dim * 2)

            # Paired RoPE (even/odd positions for each half)
            cos, sin = self.rotary(q)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

            # Unmerge to doubled sequence: (b, t, n2, hd*2) -> (b, t*2, n2, hd)
            q = q.view(b, t * 2, n2, self.head_dim)
            k = k.view(b, t * 2, n2, self.head_dim)
            v = v.reshape(b, t * 2, n2, self.head_dim)

            # Attention on interleaved 2T sequence
            if HAS_FLASH_ATTN:
                y = flash_attn_func(q, k, v, causal=HP.is_causal)
            else:
                q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                y = F.scaled_dot_product_attention(q, k, v, is_causal=HP.is_causal)
                y = y.transpose(1, 2)

            # Reshape back: (b, t*2, n2, hd) -> (b, t, n_head, hd)
            y = y.contiguous().view(b, t, self.n_head, self.head_dim)
        else:
            # Standard RoPE
            cos, sin = self.rotary(q)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

            # Attention
            if HAS_FLASH_ATTN:
                y = flash_attn_func(q, k, v, causal=HP.is_causal)
            else:
                q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                y = F.scaled_dot_product_attention(q, k, v, is_causal=HP.is_causal)
                y = y.transpose(1, 2)

        y = y.contiguous().view(b, t, self.inner_dim)

        # Skip-multiply
        y = self.attn_norm(y) * h_up

        # Down-project (with Swish)
        y = self.down_proj(y * torch.sigmoid(self.swish_beta_down * y))

        return y


class TransformerShiftBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = ShiftAttention(d_model, n_head)
        self.ln2 = RMSNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function("ts/block_attn"):
            x = x + self.attn(self.ln1(x))
        with torch.autograd.profiler.record_function("ts/block_mlp"):
            x = x + self.mlp(self.ln2(x))
        return x


class MLP(nn.Module):
    """SwiGLU MLP: gate_proj + up_proj → SiLU gating → down_proj."""

    def __init__(self, d_model: int):
        super().__init__()
        hidden = int(d_model * 8 / 3)
        # Snap to multiple of 256 for hardware efficiency
        hidden = ((hidden + 255) // 256) * 256
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function("tf/mlp"):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEMLP(nn.Module):
    """Top-k gated Mixture of Experts replacing a single MLP.

    Fixed-capacity scatter-gather: assign each token-slot to its expert's
    buffer (capacity = ceil(N*K/E)), run all experts as a single bmm with
    batch=E, scatter results back. No GPU→CPU syncs, no dynamic shapes,
    only processes ~N*K/E tokens per expert.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int,
                 bypass: bool = False, shared_frac: float = 0.0,
                 bias_lr: float = 0.001):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_model = d_model
        self.bypass = bypass
        self.bias_lr = bias_lr
        hidden = int(d_model * 8 / 3)
        hidden = ((hidden + 255) // 256) * 256
        self.hidden = hidden

        # Split hidden dim into shared (same for all experts) + per-expert
        self.shared_hidden = int(hidden * shared_frac)
        self.expert_hidden = hidden - self.shared_hidden

        # Shared weights (broadcast to all experts in forward)
        if self.shared_hidden > 0:
            self.gate_shared = nn.Parameter(torch.empty(self.shared_hidden, d_model))
            self.up_shared = nn.Parameter(torch.empty(self.shared_hidden, d_model))
            self.down_shared = nn.Parameter(torch.empty(d_model, self.shared_hidden))
            for p in [self.gate_shared, self.up_shared, self.down_shared]:
                nn.init.normal_(p, mean=0.0, std=0.02)

        # Per-expert weights
        self.gate_expert = nn.Parameter(torch.empty(n_experts, self.expert_hidden, d_model))
        self.up_expert = nn.Parameter(torch.empty(n_experts, self.expert_hidden, d_model))
        self.down_expert = nn.Parameter(torch.empty(n_experts, d_model, self.expert_hidden))

        # Router has E+K outputs when bypass is on (K null experts so top-k can go all-bypass)
        n_gate = n_experts + top_k if bypass else n_experts
        self.router = nn.Linear(d_model, n_gate, bias=False)

        # Loss-free balancing: gradient-isolated per-expert bias
        self.register_buffer('expert_bias', torch.zeros(n_gate))

        for p in [self.gate_expert, self.up_expert, self.down_expert]:
            nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        """Returns (output, aux_loss)."""
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # (N, D)
        N = x_flat.shape[0]
        E = self.n_experts
        K = self.top_k

        # Routing: biased logits for selection, unbiased for weights
        gate_logits = self.router(x_flat)  # (N, n_gate)
        biased_logits = gate_logits + self.expert_bias  # gradient-isolated bias
        _, top_idx = torch.topk(biased_logits, K, dim=-1)  # (N, K) — selection from biased
        top_vals = torch.gather(gate_logits, 1, top_idx)   # (N, K) — weights from unbiased
        top_weights = torch.softmax(top_vals, dim=-1)       # (N, K)

        # Update bias (loss-free balancing, no grad)
        if self.training:
            with torch.no_grad():
                n_gate = gate_logits.shape[1]
                counts = torch.zeros(n_gate, device=x.device)
                counts.scatter_add_(0, top_idx.view(-1),
                                    torch.ones(N * K, device=x.device))
                target = N * K / n_gate
                self.expert_bias += self.bias_lr * torch.sign(target - counts)

        # Flatten to (N*K,) token-slot pairs
        flat_expert = top_idx.view(-1)      # (N*K,) values in [0, E) or [0, E] with bypass
        flat_weight = top_weights.view(-1)  # (N*K,)
        flat_token = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, K).reshape(-1)

        # Bypass: slots routed to expert E are null — mask them out
        if self.bypass:
            is_real = (flat_expert < E).unsqueeze(-1)  # (N*K, 1)
            flat_expert_real = flat_expert.clamp(max=E - 1)  # safe index for buffer math
        else:
            is_real = None
            flat_expert_real = flat_expert

        # Compute position of each slot within its expert's group (vectorized)
        one_hot = F.one_hot(flat_expert_real, E)  # (N*K, E)
        cumpos = one_hot.cumsum(0)                # (N*K, E)
        positions = torch.gather(cumpos, 1, flat_expert_real.unsqueeze(1)).squeeze(1) - 1

        # Fixed capacity per expert — drop overflow
        capacity = (N * K + E - 1) // E
        valid = (positions < capacity).unsqueeze(-1)  # (N*K, 1)
        if is_real is not None:
            valid = valid & is_real
        positions = positions.clamp(max=capacity - 1)

        # Scatter inputs into (E, capacity, D) expert buffer
        buffer_idx = (flat_expert_real * capacity + positions).unsqueeze(-1)  # (N*K, 1)
        expert_input = torch.zeros(E * capacity, D, device=x.device, dtype=x.dtype)
        expert_input.scatter_(0, buffer_idx.expand(-1, D), x_flat[flat_token])
        expert_input = expert_input.view(E, capacity, D)

        # Build full weight tensors: cat shared (broadcast) + per-expert
        if self.shared_hidden > 0:
            sh = self.gate_shared.unsqueeze(0).expand(E, -1, -1)
            gate_w = torch.cat([sh, self.gate_expert], dim=1)                        # (E, hidden, D)
            sh = self.up_shared.unsqueeze(0).expand(E, -1, -1)
            up_w = torch.cat([sh, self.up_expert], dim=1)                            # (E, hidden, D)
            sh = self.down_shared.unsqueeze(0).expand(E, -1, -1)
            down_w = torch.cat([sh, self.down_expert], dim=2)                        # (E, D, hidden)
        else:
            gate_w, up_w, down_w = self.gate_expert, self.up_expert, self.down_expert

        # Batched SwiGLU: bmm with batch=E
        g = torch.bmm(expert_input, gate_w.transpose(1, 2))   # (E, cap, hidden)
        u = torch.bmm(expert_input, up_w.transpose(1, 2))     # (E, cap, hidden)
        expert_output = torch.bmm(F.silu(g) * u, down_w.transpose(1, 2))  # (E, cap, D)

        # Gather outputs and scatter back to tokens
        slot_output = expert_output.view(-1, D)[buffer_idx.squeeze(-1)]  # (N*K, D)
        weighted = (slot_output * flat_weight.unsqueeze(-1) * valid.to(slot_output.dtype)).to(x_flat.dtype)
        out = torch.zeros_like(x_flat)
        out.scatter_add_(0, flat_token.unsqueeze(-1).expand(-1, D), weighted)

        return out.view(B, T, D), torch.tensor(0.0, device=x.device)


class DSMoEMLP(nn.Module):
    """DeepSeek/GLM-4-style MoE: sigmoid routing, group top-k, no token dropping,
    separate shared expert as dense MLP, routed scaling factor.

    Used by NGPTMoEBlock (MODEL_TYPE=ngpt_moe). Old MoEMLP kept for MODEL_TYPE=moe.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int,
                 n_group: int = 1, topk_group: int = 1,
                 scaling_factor: float = 1.0, bias_lr: float = 0.001):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_model = d_model
        self.n_group = n_group
        self.topk_group = topk_group
        self.scaling_factor = scaling_factor
        self.bias_lr = bias_lr

        hidden = int(d_model * 8 / 3)
        hidden = ((hidden + 255) // 256) * 256
        self.hidden = hidden

        # 3D expert weights: fused gate+up as (E, 2*hidden, d_model), down as (E, d_model, hidden)
        self.gate_up_proj = nn.Parameter(torch.empty(n_experts, 2 * hidden, d_model))
        self.down_proj = nn.Parameter(torch.empty(n_experts, d_model, hidden))

        # Router: sigmoid-based (no softmax)
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Loss-free balancing: gradient-isolated per-expert bias
        self.register_buffer('expert_bias', torch.zeros(n_experts))

        # Shared expert: separate dense SwiGLU MLP added to routed output
        self.shared_gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.shared_up_proj = nn.Linear(d_model, hidden, bias=False)
        self.shared_down_proj = nn.Linear(hidden, d_model, bias=False)

        # Init
        for p in [self.gate_up_proj, self.down_proj]:
            nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        N = x_flat.shape[0]
        E = self.n_experts
        K = self.top_k

        # Sigmoid routing (fp32 for stability, cast weight explicitly for bf16 compat)
        router_logits = F.linear(x_flat.float(), self.router.weight.float())
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.expert_bias

        # Group-based top-k
        if self.n_group > 1 and self.topk_group < self.n_group:
            group_scores = (
                scores_for_choice.view(N, self.n_group, E // self.n_group)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(-1, self.n_group, E // self.n_group)
                .reshape(N, E)
            )
            scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

        topk_indices = torch.topk(scores_for_choice, k=K, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.scaling_factor

        # Loss-free balancing bias update
        if self.training:
            with torch.no_grad():
                counts = torch.zeros(E, device=x.device)
                counts.scatter_add_(0, topk_indices.view(-1),
                                    torch.ones(N * K, device=x.device))
                target = N * K / E
                self.expert_bias += self.bias_lr * torch.sign(target - counts)

        # No-drop expert loop with index_add_
        with torch.no_grad():
            expert_mask = F.one_hot(topk_indices, num_classes=E).permute(2, 1, 0)
            expert_hit = expert_mask.sum(dim=(-1, -2)).nonzero()

        final_out = torch.zeros(N, D, device=x.device, dtype=x_flat.dtype)

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = x_flat[token_idx]
            gate_up = F.linear(current_state, self.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            current_out = F.linear(F.silu(gate) * up, self.down_proj[expert_idx])
            current_out = current_out * topk_weights[token_idx, top_k_pos, None]
            final_out.index_add_(0, token_idx, current_out.to(final_out.dtype))

        # Shared expert (dense SwiGLU, added to routed output)
        shared_out = self.shared_down_proj(
            F.silu(self.shared_gate_proj(x_flat)) * self.shared_up_proj(x_flat)
        )
        final_out = final_out + shared_out

        return final_out.view(B, T, D)


class TransformerMoEBlock(nn.Module):
    """Pre-norm block: SelfAttention + MoEMLP. Returns (x, aux_loss)."""

    def __init__(self, d_model: int, n_head: int, n_experts: int, top_k: int,
                 bypass: bool = False, shared_frac: float = 0.0, bias_lr: float = 0.001):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = SelfAttention(d_model, n_head)
        self.ln2 = RMSNorm(d_model)
        self.mlp = MoEMLP(d_model, n_experts, top_k, bypass=bypass, shared_frac=shared_frac, bias_lr=bias_lr)

    def forward(self, x: torch.Tensor):
        with torch.autograd.profiler.record_function("moe/block_attn"):
            x = x + self.attn(self.ln1(x))
        with torch.autograd.profiler.record_function("moe/block_mlp"):
            mlp_out, aux_loss = self.mlp(self.ln2(x))
            x = x + mlp_out
        return x, aux_loss


class FeatureAttentionMLP(nn.Module):
    """SwiGLU MLP with feature attention replacing element-wise gating.

    SwiGLU:         out = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    Feature Attn:   out = down_proj(feat_attn(gate_proj(x), up_proj(x)))

    gate_proj provides shared Q=K (Reformer-style), up_proj provides V.
    Attention is over n_features (non-causal, within each token).
    Same three projections, same param count as SwiGLU.

    Args:
        d_model:    Input/output dimension.
        n_features: Number of features to attend over.
        activation: Attention activation — 'softmax', 'silu', or 'silu2'.
    """

    def __init__(self, d_model: int, n_features: int, activation: str = 'softmax',
                 post_act: str = 'none', pre_act: str = 'none', qk_norm: bool = False):
        super().__init__()
        hidden = int(d_model * 8 / 3)
        hidden = ((hidden + 255) // 256) * 256
        assert hidden % n_features == 0, (
            f"MLP hidden dim ({hidden}) must be divisible by n_features ({n_features})")
        self.n_features = n_features
        self.desc_dim = hidden // n_features
        self.hidden = hidden
        self.activation = activation
        self.post_act = post_act
        self.pre_act = pre_act
        self.qk_norm = qk_norm
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)  # → shared Q=K
        self.up_proj = nn.Linear(d_model, hidden, bias=False)    # → V
        self.down_proj = nn.Linear(hidden, d_model, bias=False)
        if qk_norm:
            self.qk_rms_norm = RMSNorm(self.desc_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function("fa/mlp"):
            B, T, D = x.shape
            NF = self.n_features
            DD = self.desc_dim

            # gate_proj → Q=K, up_proj → V
            qk = self.gate_proj(x).view(B * T, NF, DD)  # shared Q and K
            v = self.up_proj(x).view(B * T, NF, DD)

            # Pre-attention activation on gate (restores SiLU from SwiGLU gate path)
            if self.pre_act == 'silu':
                qk = F.silu(qk)
            elif self.pre_act == 'gelu':
                qk = F.gelu(qk)
            elif self.pre_act != 'none':
                raise ValueError(f"Unknown FA_PRE_ACT: {self.pre_act}")

            # QK normalization (cosine similarity instead of raw dot product)
            if self.qk_norm:
                qk = self.qk_rms_norm(qk)

            # Attention scores from gate (Q=K, Reformer-style)
            out = feature_attention(qk, qk, v, self.activation)
            out = out.view(B, T, self.hidden)

            # Post-attention element-wise activation
            if self.post_act == 'silu':
                out = F.silu(out)
            elif self.post_act == 'gelu':
                out = F.gelu(out)
            elif self.post_act != 'none':
                raise ValueError(f"Unknown FA_POST_ACT: {self.post_act}")

            return self.down_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = SelfAttention(d_model, n_head)
        self.ln2 = RMSNorm(d_model)
        if HP.feat_attn_mlp:
            self.mlp = FeatureAttentionMLP(d_model, HP.fa_n_features, HP.fa_activation,
                                           post_act=HP.fa_post_act, pre_act=HP.fa_pre_act,
                                           qk_norm=HP.fa_qk_norm)
        else:
            self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function("tf/block_attn"):
            x = x + self.attn(self.ln1(x))
        with torch.autograd.profiler.record_function("tf/block_mlp"):
            x = x + self.mlp(self.ln2(x))
        return x


# ── nGPT: Normalized Transformer on the Hypersphere ─────────────────────────

def _unit_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize to unit norm along dim. No learnable scale (unlike RMSNorm)."""
    return F.normalize(x, p=2, dim=dim, eps=1e-8)


def _ngpt_scale_param(shape, init_val: float, scale_val: float) -> nn.Parameter:
    """Create a scaling parameter with the nGPT init/scale trick.

    The parameter is stored as `scale_val` but during forward we multiply by
    `init_val / scale_val` so the actual value starts at init_val while Adam
    sees a parameter of magnitude scale_val (controlling effective LR).
    """
    p = nn.Parameter(torch.full(shape, scale_val))
    p._ngpt_init = init_val  # type: ignore[attr-defined]
    p._ngpt_scale = scale_val  # type: ignore[attr-defined]
    p._no_weight_decay = True  # type: ignore[attr-defined]
    return p


def _ngpt_actual(p: nn.Parameter) -> torch.Tensor:
    """Recover actual value: param * (init / scale)."""
    return p * (p._ngpt_init / p._ngpt_scale)  # type: ignore[attr-defined]


class NGPTSelfAttention(nn.Module):
    """Self-attention for nGPT: QK normalization + s_qk scaling, sqrt(dk) softmax scale.

    Supports GQA (n_kv_head < n_head), optional QK bias, and per-layer features:
    - differential: Differential attention (Ye et al. 2025) — Q,K split into halves,
      two attention passes, subtracted: out1 - λ*out2
    - paired: Paired Head Attention — adjacent heads interleaved to 2T sequence
    - window_size: Sliding window attention (0 = full context)
    """

    def __init__(self, d_model: int, n_head: int, layer_idx: int = 0,
                 differential: bool = False, paired: bool = False,
                 trap_mix: bool = False, window_size: int = 0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.differential = differential
        self.paired = paired
        self.window_size = window_size
        n_kv = HP.n_kv_head if HP.n_kv_head > 0 else n_head
        assert n_head % n_kv == 0, f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv})"
        self.n_kv_head = n_kv
        self.n_kv_groups = n_head // n_kv  # how many Q heads per KV head

        qk_bias = HP.ngpt_qk_bias
        self.q = nn.Linear(d_model, n_head * self.head_dim, bias=qk_bias)
        self.k = nn.Linear(d_model, n_kv * self.head_dim, bias=qk_bias)
        self.v = nn.Linear(d_model, n_kv * self.head_dim, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        # Differential attention: λ reparameterization vectors
        if self.differential:
            assert self.head_dim % 2 == 0, f"head_dim ({self.head_dim}) must be even for diff attn"
            self.half_dim = self.head_dim // 2
            self.lambda_q1 = nn.Parameter(torch.randn(self.half_dim) * 0.1)
            self.lambda_k1 = nn.Parameter(torch.randn(self.half_dim) * 0.1)
            self.lambda_q2 = nn.Parameter(torch.randn(self.half_dim) * 0.1)
            self.lambda_k2 = nn.Parameter(torch.randn(self.half_dim) * 0.1)
            self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
            # Mark lambda params so _ngpt_normalize_weights skips them
            for p in (self.lambda_q1, self.lambda_k1, self.lambda_q2, self.lambda_k2):
                p._ngpt_skip_normalize = True  # type: ignore[attr-defined]

        # RoPE: standard or paired, dimension depends on diff attn
        # For diff attn: RoPE applied to sub-heads of half_dim, not full head_dim
        self.dd_rope = HP.dd_rope
        hd = self.head_dim
        rope_dim = self.half_dim if self.differential else self.head_dim
        if self.dd_rope:
            assert hd % 2 == 0, f"head_dim ({hd}) must be even for dd-rope"
            self.theta_proj = nn.Linear(d_model, n_head * (hd // 2), bias=False)
        elif self.paired:
            # PairedRotary: applied to merged pair width (2*rope_dim), produces cos/sin of rope_dim width
            self.rotary = PairedRotary(rope_dim)
        else:
            self.rotary = Rotary(rope_dim)

        # Trapezoidal K,V mixing
        self.trap_mix = trap_mix
        if self.trap_mix:
            self.trap_proj = nn.Linear(d_model, n_head * 3, bias=False)

        # QK scaling: shared per head, one scalar per head_dim element
        s_scale = 1.0 / math.sqrt(d_model)
        self.s_qk = _ngpt_scale_param((self.head_dim,), HP.ngpt_sqk_init, s_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.q(x).view(b, t, self.n_head, self.head_dim)
        k = self.k(x).view(b, t, self.n_kv_head, self.head_dim)
        v = self.v(x).view(b, t, self.n_kv_head, self.head_dim)

        # TrapMix: data-dependent K,V mixing with temporal neighbor (before RoPE, before head splitting)
        if self.trap_mix:
            trap = self.trap_proj(x).view(b, t, self.n_head, 3)
            log_dt, lam_logit, log_A = trap[..., 0], trap[..., 1], trap[..., 2]
            dt = F.softplus(log_dt).clamp(max=2.0)
            A_neg = -F.softplus(log_A)
            alpha_t = torch.exp(dt * A_neg).clamp(max=0.999)
            lam = torch.sigmoid(lam_logit)
            gamma = (lam * dt).unsqueeze(-1)
            beta = ((1.0 - lam) * dt * alpha_t).unsqueeze(-1)
            k_shifted = F.pad(k[:, :-1], (0, 0, 0, 0, 1, 0))
            k = gamma[:, :, :self.n_kv_head] * k + beta[:, :, :self.n_kv_head] * k_shifted
            v_shifted = F.pad(v[:, :-1], (0, 0, 0, 0, 1, 0))
            v = gamma[:, :, :self.n_kv_head] * v + beta[:, :, :self.n_kv_head] * v_shifted

        # GQA: expand KV heads to match Q heads (before RoPE so all heads get position encoding)
        if self.n_kv_groups > 1:
            k = k[:, :, :, None, :].expand(b, t, self.n_kv_head, self.n_kv_groups, self.head_dim).reshape(b, t, self.n_head, self.head_dim)
            v = v[:, :, :, None, :].expand(b, t, self.n_kv_head, self.n_kv_groups, self.head_dim).reshape(b, t, self.n_head, self.head_dim)

        # nGPT: normalize q,k then apply learned scaling
        s_qk = _ngpt_actual(self.s_qk)
        q = _unit_norm(q, dim=-1) * s_qk
        k = _unit_norm(k, dim=-1) * s_qk

        if self.differential:
            return self._forward_diff(x, q, k, v, b, t, c)
        else:
            return self._forward_standard(x, q, k, v, b, t, c)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, x: torch.Tensor,
                    b: int, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE (standard or dd-rope). x needed for dd-rope projection."""
        if self.dd_rope:
            theta = self.theta_proj(x).view(b, t, self.n_head, self.head_dim // 2)
            cum_theta = theta.cumsum(dim=1)
            cos_dd, sin_dd = cum_theta.cos(), cum_theta.sin()
            q = apply_rotary(q, cos_dd, sin_dd)
            k = apply_rotary(k, cos_dd, sin_dd)
        elif self.paired:
            cos, sin = self.rotary(q)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)
        else:
            cos, sin = self.rotary(q)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)
        return q, k

    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              head_dim: int, causal: bool = True) -> torch.Tensor:
        """Run attention (flash or SDPA) with optional sliding window."""
        ws = (self.window_size, -1) if self.window_size > 0 else (-1, -1)
        if HAS_FLASH_ATTN:
            q_s = q * head_dim  # flash_attn divides by sqrt(dk), we want sqrt(dk) scale
            return flash_attn_func(q_s, k, v, causal=causal, window_size=ws)
        else:
            q_t, k_t, v_t = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            attn_mask = None
            if self.window_size > 0:
                # Build sliding-window + causal mask for SDPA fallback
                seq_len = q_t.size(-2)
                row = torch.arange(seq_len, device=q.device).unsqueeze(1)
                col = torch.arange(seq_len, device=q.device).unsqueeze(0)
                attn_mask = (row >= col) & (row - col < self.window_size)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
                causal = False  # mask handles causality
            y = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal,
                                               attn_mask=attn_mask,
                                               scale=math.sqrt(head_dim))
            return y.transpose(1, 2)

    def _forward_standard(self, x: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                          v: torch.Tensor, b: int, t: int, c: int) -> torch.Tensor:
        """Standard (non-differential) attention, optionally with PHA."""
        if self.paired:
            assert self.n_head % 2 == 0, "PHA requires even number of heads"
            n2 = self.n_head // 2
            # Merge adjacent head pairs: (b,t,H,hd) -> (b,t,H/2,2*hd)
            q = q.view(b, t, n2, self.head_dim * 2)
            k = k.view(b, t, n2, self.head_dim * 2)
            # Apply paired RoPE on merged width
            q, k = self._apply_rope(q, k, x, b, t)
            # Split to doubled sequence: (b,t,H/2,2*hd) -> (b,2t,H/2,hd)
            q = q.view(b, t * 2, n2, self.head_dim)
            k = k.view(b, t * 2, n2, self.head_dim)
            v = v.reshape(b, t * 2, n2, self.head_dim)
            # Attention on interleaved 2T sequence
            y = self._attn(q, k, v, self.head_dim)
            # Reshape back: (b,2t,H/2,hd) -> (b,t,H,hd)
            y = y.contiguous().view(b, t, self.n_head, self.head_dim)
        else:
            # Standard RoPE
            q, k = self._apply_rope(q, k, x, b, t)
            y = self._attn(q, k, v, self.head_dim)

        y = y.contiguous().view(b, t, c)
        return self.proj(y)

    def _forward_diff(self, x: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                      v: torch.Tensor, b: int, t: int, c: int) -> torch.Tensor:
        """Differential attention: out1 - λ*out2, optionally with PHA."""
        hd2 = self.half_dim  # half_dim = head_dim // 2
        # Split Q,K into two sub-heads: (b,t,H,hd) -> q1,q2 each (b,t,H,hd/2)
        q1, q2 = q[..., :hd2], q[..., hd2:]
        k1, k2 = k[..., :hd2], k[..., hd2:]

        # Compute λ
        lam = (torch.exp(torch.dot(self.lambda_q1, self.lambda_k1))
               - torch.exp(torch.dot(self.lambda_q2, self.lambda_k2))
               + self.lambda_init)

        if self.paired:
            assert self.n_head % 2 == 0, "PHA requires even number of heads"
            n2 = self.n_head // 2
            # Merge adjacent head pairs for sub-heads: (b,t,H,hd/2) -> (b,t,H/2,hd)
            # .contiguous() needed because slicing q[..., :hd2] produces non-contiguous view
            q1 = q1.contiguous().view(b, t, n2, hd2 * 2)
            k1 = k1.contiguous().view(b, t, n2, hd2 * 2)
            q2 = q2.contiguous().view(b, t, n2, hd2 * 2)
            k2 = k2.contiguous().view(b, t, n2, hd2 * 2)
            # Apply paired RoPE (operates on hd = 2*hd2 width)
            q1, k1 = self._apply_rope(q1, k1, x, b, t)
            q2, k2 = self._apply_rope(q2, k2, x, b, t)
            # Split to doubled sequence: (b,t,H/2,hd) -> (b,2t,H/2,hd/2)
            q1 = q1.view(b, t * 2, n2, hd2)
            k1 = k1.view(b, t * 2, n2, hd2)
            q2 = q2.view(b, t * 2, n2, hd2)
            k2 = k2.view(b, t * 2, n2, hd2)
            # V interleaved: (b,t,H,hd) -> (b,2t,H/2,hd)
            v_pha = v.reshape(b, t * 2, n2, self.head_dim)
            # Two attention passes
            out1 = self._attn(q1, k1, v_pha, hd2)  # (b,2t,H/2,hd)
            out2 = self._attn(q2, k2, v_pha, hd2)  # (b,2t,H/2,hd)
            # Reshape back: (b,2t,H/2,hd) -> (b,t,H,hd)
            out1 = out1.contiguous().view(b, t, self.n_head, self.head_dim)
            out2 = out2.contiguous().view(b, t, self.n_head, self.head_dim)
        else:
            # Standard RoPE on sub-heads
            q1, k1 = self._apply_rope(q1, k1, x, b, t)
            q2, k2 = self._apply_rope(q2, k2, x, b, t)
            # Two attention passes with same V
            out1 = self._attn(q1, k1, v, hd2)  # (b,t,H,hd)
            out2 = self._attn(q2, k2, v, hd2)  # (b,t,H,hd)

        # Differential: out1 - λ*out2, then per-head unit norm for direction stability
        diff = out1 - lam * out2
        diff = _unit_norm(diff, dim=-1)  # per-head normalization (replaces head_norm + 1-λ_init scalar)

        y = diff.contiguous().view(b, t, c)
        return self.proj(y)


class NGPTMLP(nn.Module):
    """SwiGLU MLP for nGPT: s_u, s_v scaling on intermediates."""

    def __init__(self, d_model: int):
        super().__init__()
        hidden = int(d_model * 8 / 3)
        hidden = ((hidden + 255) // 256) * 256
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

        # Scaling factors for intermediate states
        self.s_u = _ngpt_scale_param((hidden,), HP.ngpt_su_init, 1.0)
        self.s_v = _ngpt_scale_param((hidden,), HP.ngpt_sv_init, 1.0)
        self._sqrt_d = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function("ngpt/mlp"):
            u = self.up_proj(x) * _ngpt_actual(self.s_u)
            v = self.gate_proj(x) * _ngpt_actual(self.s_v) * self._sqrt_d
            return self.down_proj(F.silu(v) * u)


class NGPTBlock(nn.Module):
    """nGPT transformer block: LERP updates with eigen learning rates on the hypersphere.

    h = Norm(h + alpha_A * (Norm(attn(h)) - h))
    h = Norm(h + alpha_M * (Norm(mlp(h)) - h))
    [optional] h = Norm(h + alpha_G * (Norm(gate_proj(h) * x0) - h))  # embed gate
    No RMSNorm layers. Hidden state stays unit-norm throughout.
    """

    def __init__(self, d_model: int, n_head: int, layer_idx: int = 0,
                 differential: bool = False, paired: bool = False,
                 trap_mix: bool = False, window_size: int = 0,
                 embed_gate: bool = False):
        super().__init__()
        self.attn = NGPTSelfAttention(d_model, n_head, layer_idx=layer_idx,
                                      differential=differential, paired=paired,
                                      trap_mix=trap_mix, window_size=window_size)
        if HP.feat_attn_mlp:
            self.mlp = FeatureAttentionMLP(d_model, HP.fa_n_features, HP.fa_activation,
                                           post_act=HP.fa_post_act, pre_act=HP.fa_pre_act,
                                           qk_norm=HP.fa_qk_norm)
        else:
            self.mlp = NGPTMLP(d_model)

        # Eigen learning rates (per embedding dimension)
        alpha_init = HP.ngpt_alpha_init if HP.ngpt_alpha_init > 0 else 1.0 / HP.n_layer
        alpha_scale = HP.ngpt_alpha_scale if HP.ngpt_alpha_scale > 0 else 1.0 / math.sqrt(d_model)
        self.alpha_attn = _ngpt_scale_param((d_model,), alpha_init, alpha_scale)
        self.alpha_mlp = _ngpt_scale_param((d_model,), alpha_init, alpha_scale)

        # Embed gate: content-dependent residual from original embedding
        self.embed_gate = embed_gate
        if self.embed_gate:
            self.gate_proj = nn.Linear(d_model, d_model, bias=False)
            self.alpha_gate = _ngpt_scale_param((d_model,), alpha_init, alpha_scale)
        self._x0 = None  # set by model forward before running blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function("ngpt/block_attn"):
            h_a = _unit_norm(self.attn(x))
            alpha_a = _ngpt_actual(self.alpha_attn).abs()
            x = _unit_norm(x + alpha_a * (h_a - x))
        with torch.autograd.profiler.record_function("ngpt/block_mlp"):
            h_m = _unit_norm(self.mlp(x))
            alpha_m = _ngpt_actual(self.alpha_mlp).abs()
            x = _unit_norm(x + alpha_m * (h_m - x))
        if self.embed_gate and self._x0 is not None:
            with torch.autograd.profiler.record_function("ngpt/embed_gate"):
                h_g = _unit_norm(self.gate_proj(x) * self._x0)
                alpha_g = _ngpt_actual(self.alpha_gate).abs()
                x = _unit_norm(x + alpha_g * (h_g - x))
        return x


class NGPTMoEBlock(nn.Module):
    """nGPT transformer block with DeepSeek-style MoE MLP.

    Same LERP-on-hypersphere update as NGPTBlock, but replaces the dense MLP
    with DSMoEMLP (sigmoid routing, group top-k, no token dropping, shared expert).
    No RMSNorm — hidden state stays unit-norm throughout.
    """

    def __init__(self, d_model: int, n_head: int, n_experts: int, top_k: int,
                 n_group: int = 1, topk_group: int = 1,
                 scaling_factor: float = 1.0, bias_lr: float = 0.001,
                 layer_idx: int = 0,
                 differential: bool = False, paired: bool = False,
                 trap_mix: bool = False, window_size: int = 0,
                 embed_gate: bool = False):
        super().__init__()
        self.attn = NGPTSelfAttention(d_model, n_head, layer_idx=layer_idx,
                                      differential=differential, paired=paired,
                                      trap_mix=trap_mix, window_size=window_size)
        self.mlp = DSMoEMLP(d_model, n_experts, top_k, n_group=n_group,
                            topk_group=topk_group, scaling_factor=scaling_factor,
                            bias_lr=bias_lr)

        # Eigen learning rates (per embedding dimension)
        alpha_init = HP.ngpt_alpha_init if HP.ngpt_alpha_init > 0 else 1.0 / HP.n_layer
        alpha_scale = HP.ngpt_alpha_scale if HP.ngpt_alpha_scale > 0 else 1.0 / math.sqrt(d_model)
        self.alpha_attn = _ngpt_scale_param((d_model,), alpha_init, alpha_scale)
        self.alpha_mlp = _ngpt_scale_param((d_model,), alpha_init, alpha_scale)

        # Embed gate: content-dependent residual from original embedding
        self.embed_gate = embed_gate
        if self.embed_gate:
            self.gate_proj = nn.Linear(d_model, d_model, bias=False)
            self.alpha_gate = _ngpt_scale_param((d_model,), alpha_init, alpha_scale)
        self._x0 = None  # set by model forward before running blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function("ngpt_moe/block_attn"):
            h_a = _unit_norm(self.attn(x))
            alpha_a = _ngpt_actual(self.alpha_attn).abs()
            x = _unit_norm(x + alpha_a * (h_a - x))
        with torch.autograd.profiler.record_function("ngpt_moe/block_mlp"):
            h_m = _unit_norm(self.mlp(x))  # DSMoEMLP returns tensor directly, no drops
            alpha_m = _ngpt_actual(self.alpha_mlp).abs()
            x = _unit_norm(x + alpha_m * (h_m - x))
        if self.embed_gate and self._x0 is not None:
            with torch.autograd.profiler.record_function("ngpt_moe/embed_gate"):
                h_g = _unit_norm(self.gate_proj(x) * self._x0)
                alpha_g = _ngpt_actual(self.alpha_gate).abs()
                x = _unit_norm(x + alpha_g * (h_g - x))
        return x


def _ngpt_normalize_weights(model: nn.Module):
    """No-op: weight normalization removed. Embedding unit-norm is handled
    inside CompositeEmbedding / the model forward pass instead."""
    pass


def _ngpt_normalize_pit_memory(model: nn.Module):
    """No-op: PIT memory normalization removed."""
    pass


@torch.no_grad()
def _sphere_retract_weights(model: nn.Module):
    """Retract all 2D weight matrices to the product of spheres.

    After the optimizer step, each row of every 2D parameter is normalized
    to unit norm. Skips embeddings, biases, and 1D params.
    """
    _skip_names = {"wte", "lm_head", "embed", "query_bank", "queries", "memory",
                   "byte_memory", "token_embed", "token_params", "byte_embed"}
    for name, p in model.named_parameters():
        if p.ndim < 2:
            continue
        if getattr(p, "_no_weight_decay", False):
            continue
        if getattr(p, "_ngpt_skip_normalize", False):
            continue
        if any(k in name for k in _skip_names):
            continue
        p.data.div_(p.data.norm(dim=-1, keepdim=True).clamp(min=1e-8))


class TransformerFeatureAttnBlock(nn.Module):
    """Transformer block with feature attention MLP (replacing GELU with feature self-attn)."""

    def __init__(self, d_model: int, n_head: int, n_features: int, activation: str,
                 post_act: str = 'none', pre_act: str = 'none', qk_norm: bool = False):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = SelfAttention(d_model, n_head)
        self.ln2 = RMSNorm(d_model)
        self.mlp = FeatureAttentionMLP(d_model, n_features, activation, post_act=post_act, pre_act=pre_act, qk_norm=qk_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function("fa/block_attn"):
            x = x + self.attn(self.ln1(x))
        with torch.autograd.profiler.record_function("fa/block_mlp"):
            x = x + self.mlp(self.ln2(x))
        return x


class ThreeStageBlock(nn.Module):
    """Three-stage block: seq attention → feature attention → MLP.

    Stage 1 — Sequence attention (tokens talk to each other):
      q, k, v = Q/K/V_proj(x)       # standard projections
      seq_out = attn(q, k, v) + x    # causal, RoPE, residual

    Stage 2 — Feature attention (token talks to itself):
      q_feat = q_pre_rope.view(B*T, N_f, D_f)   # reuse Q before RoPE
      k_feat = k_pre_rope.view(B*T, N_f, D_f)   # reuse K before RoPE
      v_feat = seq_out.view(B*T, N_f, D_f)       # attend over seq_out
      feat_out = feat_attn(q_feat, k_feat, v_feat)
      feat_out = SiLU(feat_out) + seq_out         # activate + residual

    Stage 3 — MLP (enrichment):
      out = SwiGLU(feat_out) + feat_out           # standard MLP, residual

    Same params as standard transformer — feature attention adds zero projections.
    """

    def __init__(self, d_model: int, n_head: int, n_features: int,
                 feat_activation: str = 'softmax'):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.n_features = n_features
        self.desc_dim = d_model // n_features
        self.feat_activation = feat_activation

        assert d_model % n_features == 0, (
            f"d_model ({d_model}) must be divisible by n_features ({n_features})")

        # Stage 1: sequence attention (standard)
        self.ln1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.head_dim)

        # Stage 2: feature attention (no new projections)
        self.ln2 = RMSNorm(d_model)

        # Stage 3: MLP (standard SwiGLU)
        self.ln3 = RMSNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        NH = self.n_head
        HD = self.head_dim
        NF = self.n_features
        DD = self.desc_dim

        # ── Stage 1: Sequence attention ──
        h = self.ln1(x)
        q = self.q_proj(h).view(B, T, NH, HD)
        k = self.k_proj(h).view(B, T, NH, HD)
        v = self.v_proj(h).view(B, T, NH, HD)

        q_pre_rope = q  # save for feature attention
        k_pre_rope = k  # save for feature attention

        cos, sin = self.rotary(q)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if HAS_FLASH_ATTN:
            seq_out = flash_attn_func(q, k, v, causal=HP.is_causal)
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            seq_out = F.scaled_dot_product_attention(q, k, v, is_causal=HP.is_causal)
            seq_out = seq_out.transpose(1, 2)

        seq_out = self.out_proj(seq_out.contiguous().view(B, T, D))
        x = x + seq_out  # residual

        # ── Stage 2: Feature attention ──
        h2 = self.ln2(x)
        q_feat = q_pre_rope.contiguous().view(B * T, NF, DD)
        k_feat = k_pre_rope.contiguous().view(B * T, NF, DD)
        v_feat = h2.view(B * T, NF, DD)

        scale = DD ** -0.5
        feat_out = feature_attention(q_feat, k_feat, v_feat, self.feat_activation)
        feat_out = F.silu(feat_out.view(B, T, D))
        x = x + feat_out  # residual

        # ── Stage 3: MLP ──
        x = x + self.mlp(self.ln3(x))

        return x


class ThreeStageFSABlock(nn.Module):
    """Three-stage block: feature attention → seq attention → MLP.

    Compute Q/K/V projections once from ln1(x), then:

    Stage 1 — Feature attention (reorganize own features):
      q_feat = Q_pre_rope.view(B*T, N_f, D_f)   # reuse Q before RoPE
      v_feat = ln1(x).view(B*T, N_f, D_f)
      feat_out = SiLU(feat_attn(q_feat, q_feat, v_feat)) + x

    Stage 2 — Sequence attention (tokens talk with reorganized repr):
      Apply RoPE to Q, K from the SAME projections
      seq_out = attn(q, k, v) + feat_out

    Stage 3 — MLP (enrichment):
      out = SwiGLU(seq_out) + seq_out

    Zero extra params vs baseline — same as ThreeStageBlock, just stages 1 and 2 swapped.
    """

    def __init__(self, d_model: int, n_head: int, n_features: int,
                 feat_activation: str = 'softmax'):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.n_features = n_features
        self.desc_dim = d_model // n_features
        self.feat_activation = feat_activation

        assert d_model % n_features == 0, (
            f"d_model ({d_model}) must be divisible by n_features ({n_features})")

        # Shared projections for both feature and sequence attention
        self.ln1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.head_dim)

        # MLP
        self.ln3 = RMSNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        NH = self.n_head
        HD = self.head_dim
        NF = self.n_features
        DD = self.desc_dim

        # Project Q and V; defer K to after feature attention
        h = self.ln1(x)
        q = self.q_proj(h)  # (B, T, D)
        v = self.v_proj(h).view(B, T, NH, HD)

        # ── Stage 1: Feature attention (using pre-RoPE Q) ──
        q_feat = q.view(B * T, NF, DD)
        v_feat = h.view(B * T, NF, DD)

        feat_out = feature_attention(q_feat, q_feat, v_feat, self.feat_activation)
        feat_out = F.silu(feat_out.view(B, T, D))
        x = x + feat_out  # residual

        # ── Stage 2: Sequence attention — K computed here since feat attn doesn't need it ──
        q_seq = q.view(B, T, NH, HD)
        k = self.k_proj(h).view(B, T, NH, HD)

        cos, sin = self.rotary(q_seq)
        q_seq = apply_rotary(q_seq, cos, sin)
        k = apply_rotary(k, cos, sin)

        if HAS_FLASH_ATTN:
            seq_out = flash_attn_func(q_seq, k, v, causal=HP.is_causal)
        else:
            q_seq, k, v = q_seq.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            seq_out = F.scaled_dot_product_attention(q_seq, k, v, is_causal=HP.is_causal)
            seq_out = seq_out.transpose(1, 2)

        seq_out = self.out_proj(seq_out.contiguous().view(B, T, D))
        x = x + seq_out  # residual

        # ── Stage 3: MLP ──
        x = x + self.mlp(self.ln3(x))

        return x


class QVOBlock(nn.Module):
    """Three-stage block: feature attention → seq attention → MLP.

    Only 3 projections: Q, V, O. No K projection at all.
    Q is used as both Q and K for both feature and sequence attention.

    Stage 1 — Feature attention:
      q_feat = Q_proj(ln1(x)).view(B*T, N_f, D_f)   # Q=K
      v_feat = V_proj(ln1(x)).view(B*T, N_f, D_f)
      feat_out = SiLU(feat_attn(q, q, v)) + x

    Stage 2 — Sequence attention:
      q_seq = Q_proj(ln1(x)).view(B, T, NH, HD) + RoPE
      v_seq = V_proj(ln1(x)).view(B, T, NH, HD)
      seq_out = attn(q, q, v) + x'

    Stage 3 — MLP:
      out = SwiGLU(seq_out) + seq_out

    Saves 768*768 = ~590K params per layer vs QKVO. Same effective arch as
    ThreeStageFSABlock with Q=K (where K was dead weight).
    """

    def __init__(self, d_model: int, n_head: int, n_features: int,
                 feat_activation: str = 'softmax'):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.n_features = n_features
        self.desc_dim = d_model // n_features
        self.feat_activation = feat_activation

        assert d_model % n_features == 0, (
            f"d_model ({d_model}) must be divisible by n_features ({n_features})")

        # Only 3 projections: Q, V, O — no K
        self.ln1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.head_dim)

        # MLP
        self.ln3 = RMSNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        NH = self.n_head
        HD = self.head_dim
        NF = self.n_features
        DD = self.desc_dim

        # Project Q and V once — no K
        h = self.ln1(x)
        q = self.q_proj(h)  # (B, T, D) — used as both Q and K
        v = self.v_proj(h)  # (B, T, D)

        # ── Stage 1: Feature attention — Q=K, reshape as (N_f, D_f) ──
        q_feat = q.view(B * T, NF, DD)
        v_feat = v.view(B * T, NF, DD)

        feat_out = feature_attention(q_feat, q_feat, v_feat, self.feat_activation)
        feat_out = F.silu(feat_out.view(B, T, D))
        x = x + feat_out  # residual

        # ── Stage 2: Sequence attention — Q=K, reshape as (N_heads, H_dim), apply RoPE ──
        q_seq = q.view(B, T, NH, HD)
        v_seq = v.view(B, T, NH, HD)

        cos, sin = self.rotary(q_seq)
        q_seq = apply_rotary(q_seq, cos, sin)
        k_seq = apply_rotary(q.view(B, T, NH, HD), cos, sin)  # same Q as K, with RoPE

        if HAS_FLASH_ATTN:
            seq_out = flash_attn_func(q_seq, k_seq, v_seq, causal=HP.is_causal)
        else:
            q_seq, k_seq, v_seq = q_seq.transpose(1, 2), k_seq.transpose(1, 2), v_seq.transpose(1, 2)
            seq_out = F.scaled_dot_product_attention(q_seq, k_seq, v_seq, is_causal=HP.is_causal)
            seq_out = seq_out.transpose(1, 2)

        seq_out = self.out_proj(seq_out.contiguous().view(B, T, D))
        x = x + seq_out  # residual

        # ── Stage 3: MLP ──
        x = x + self.mlp(self.ln3(x))

        return x


class DualQBlock(nn.Module):
    """Three-stage block: feature attention → seq attention → MLP.

    K projection is replaced by a second Q projection. Both attentions use Q=K.

    Stage 1 — Feature attention:
      q_feat = Q_feat_proj(ln1(x)).view(B*T, N_f, D_f)   # Q=K
      v_feat = V_proj(ln1(x)).view(B*T, N_f, D_f)
      feat_out = SiLU(feat_attn(q, q, v)) + x

    Stage 2 — Sequence attention (from post-feat-attn space):
      q_seq = Q_seq_proj(ln2(x')).view(B, T, NH, HD)      # Q=K + RoPE
      v_seq = V_proj(ln1(x)).view(B, T, NH, HD)            # V from original
      seq_out = attn(q, q, v) + x'

    Stage 3 — MLP

    Same param count as QKVO: Q_feat + Q_seq + V + O = 4 projections.
    K is gone. Both attentions learn their own Q for symmetric scoring.
    Seq Q comes from post-feat-attn space — asks questions with organized features.
    """

    def __init__(self, d_model: int, n_head: int, n_features: int,
                 feat_activation: str = 'softmax'):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.n_features = n_features
        self.desc_dim = d_model // n_features
        self.feat_activation = feat_activation

        assert d_model % n_features == 0, (
            f"d_model ({d_model}) must be divisible by n_features ({n_features})")

        # Stage 1: feature attention
        self.ln1 = RMSNorm(d_model)
        self.q_feat_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Stage 2: sequence attention (Q from post-feat-attn, V shared)
        self.ln2 = RMSNorm(d_model)
        self.q_seq_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.head_dim)

        # Stage 3: MLP
        self.ln3 = RMSNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        NH = self.n_head
        HD = self.head_dim
        NF = self.n_features
        DD = self.desc_dim

        # ── Stage 1: Feature attention ──
        h = self.ln1(x)
        q_feat = self.q_feat_proj(h).view(B * T, NF, DD)
        v = self.v_proj(h)  # (B, T, D) — shared with seq attn
        v_feat = v.view(B * T, NF, DD)

        feat_out = feature_attention(q_feat, q_feat, v_feat, self.feat_activation)
        feat_out = F.silu(feat_out.view(B, T, D))
        x = x + feat_out  # residual

        # ── Stage 2: Sequence attention (Q from post-feat-attn space) ──
        h2 = self.ln2(x)
        q_seq = self.q_seq_proj(h2).view(B, T, NH, HD)
        v_seq = v.view(B, T, NH, HD)  # V from original projection

        cos, sin = self.rotary(q_seq)
        q_seq = apply_rotary(q_seq, cos, sin)  # Q=K, RoPE applied once

        if HAS_FLASH_ATTN:
            seq_out = flash_attn_func(q_seq, q_seq, v_seq, causal=HP.is_causal)
        else:
            q_seq, v_seq = q_seq.transpose(1, 2), v_seq.transpose(1, 2)
            seq_out = F.scaled_dot_product_attention(q_seq, q_seq, v_seq, is_causal=HP.is_causal)
            seq_out = seq_out.transpose(1, 2)

        seq_out = self.out_proj(seq_out.contiguous().view(B, T, D))
        x = x + seq_out  # residual

        # ── Stage 3: MLP ──
        x = x + self.mlp(self.ln3(x))

        return x


class FusedSeqFeatureAttnBlock(nn.Module):
    """Fused block: shared QK/V projections → seq attention → feature attention → down.

    Single set of projections serves both sequence-level and feature-level attention:
      qk = SiLU(QK_proj(x))     # (B, T, hidden) — shared Q=K for both attentions
      v  = V_proj(x)             # (B, T, hidden)

      # 1. Sequence attention over T (causal, with RoPE)
      qk_seq = qk.view(B, T, n_heads, head_dim)
      v_seq  = v.view(B, T, n_heads, head_dim)
      out = seq_attn(qk_seq, qk_seq, v_seq)  # Q=K, attend over T

      # 2. Feature attention over N_f (non-causal, within each token)
      out_feat = out.view(B*T, N_f, D_f)
      qk_feat  = qk.view(B*T, N_f, D_f)
      out = feat_attn(qk_feat, qk_feat, out_feat)  # Q=K, attend over N_f

      out = SiLU(out)
      out = down_proj(out)

    Params: QK_proj(d→H) + V_proj(d→H) + down(H→d) = 3*d*H
    vs standard block: Q+K+V+O(4*d²) + gate+up+down(3*d*H) ≈ 7.1M → 4.7M at d=768,H=2048
    """

    def __init__(self, d_model: int, hidden: int, n_seq_heads: int,
                 n_features: int, feat_activation: str = 'softmax'):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.n_seq_heads = n_seq_heads
        self.seq_head_dim = hidden // n_seq_heads
        self.n_features = n_features
        self.desc_dim = hidden // n_features
        self.feat_activation = feat_activation

        assert hidden % n_seq_heads == 0, f"hidden ({hidden}) must be divisible by n_seq_heads ({n_seq_heads})"
        assert hidden % n_features == 0, f"hidden ({hidden}) must be divisible by n_features ({n_features})"

        self.ln = RMSNorm(d_model)
        self.qk_proj = nn.Linear(d_model, hidden, bias=False)
        self.v_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)
        self.rotary = Rotary(self.seq_head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        NH = self.n_seq_heads
        HD = self.seq_head_dim
        NF = self.n_features
        DD = self.desc_dim
        H = self.hidden

        h = self.ln(x)

        # Shared projections
        qk = F.silu(self.qk_proj(h))  # (B, T, H) — pre-activated gate
        v = self.v_proj(h)              # (B, T, H)

        # ── Sequence attention (causal, RoPE, Q=K) ──
        qk_seq = qk.view(B, T, NH, HD)
        v_seq = v.view(B, T, NH, HD)
        cos, sin = self.rotary(qk_seq)
        q_seq = apply_rotary(qk_seq, cos, sin)
        k_seq = apply_rotary(qk_seq, cos, sin)  # Q=K, same RoPE

        if HAS_FLASH_ATTN:
            out = flash_attn_func(q_seq, k_seq, v_seq, causal=HP.is_causal)
        else:
            q_seq = q_seq.transpose(1, 2)
            k_seq = k_seq.transpose(1, 2)
            v_seq = v_seq.transpose(1, 2)
            out = F.scaled_dot_product_attention(q_seq, k_seq, v_seq, is_causal=HP.is_causal)
            out = out.transpose(1, 2)

        out = out.contiguous().view(B, T, H)

        # ── Feature attention (non-causal, within each token) ──
        qk_feat = qk.view(B * T, NF, DD)
        out_feat = out.view(B * T, NF, DD)

        out = feature_attention(qk_feat, qk_feat, out_feat, self.feat_activation)
        out = F.silu(out.view(B, T, H))     # post-activation
        out = self.down_proj(out)

        return x + out


class FusedQKVBlock(nn.Module):
    """Fused block with full Q/K/V projections shared across seq + feature attention.

    Flow:
      q = Q_proj(x)                          # (B, T, H)
      k = K_proj(x)                          # (B, T, H)
      v = V_proj(x)                          # (B, T, H)

      seq_out = seq_attn(q, k, v)            # causal, RoPE, over T
      seq_out = seq_out + v                   # skip from V around seq attn

      feat_out = feat_attn(q, q, seq_out)    # Q=K from q, over N_f
      out = SiLU(feat_out)
      out = down_proj(out)

    Params: Q(d,H) + K(d,H) + V(d,H) + down(H,d) = 4*d*H
    At d=768, H=2304: 124M params with 12 layers.
    """

    def __init__(self, d_model: int, hidden: int, n_seq_heads: int,
                 n_features: int, feat_activation: str = 'softmax'):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.n_seq_heads = n_seq_heads
        self.seq_head_dim = hidden // n_seq_heads
        self.n_features = n_features
        self.desc_dim = hidden // n_features
        self.feat_activation = feat_activation

        assert hidden % n_seq_heads == 0, f"hidden ({hidden}) must be divisible by n_seq_heads ({n_seq_heads})"
        assert hidden % n_features == 0, f"hidden ({hidden}) must be divisible by n_features ({n_features})"

        self.ln = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, hidden, bias=False)
        self.k_proj = nn.Linear(d_model, hidden, bias=False)
        self.v_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)
        self.rotary = Rotary(self.seq_head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        NH = self.n_seq_heads
        HD = self.seq_head_dim
        NF = self.n_features
        DD = self.desc_dim
        H = self.hidden

        h = self.ln(x)

        # Full Q, K, V projections
        q = self.q_proj(h)   # (B, T, H)
        k = self.k_proj(h)   # (B, T, H)
        v = self.v_proj(h)   # (B, T, H)

        # ── Sequence attention (causal, RoPE, separate Q/K) ──
        q_seq = q.view(B, T, NH, HD)
        k_seq = k.view(B, T, NH, HD)
        v_seq = v.view(B, T, NH, HD)
        cos, sin = self.rotary(q_seq)
        q_seq = apply_rotary(q_seq, cos, sin)
        k_seq = apply_rotary(k_seq, cos, sin)

        if HAS_FLASH_ATTN:
            seq_out = flash_attn_func(q_seq, k_seq, v_seq, causal=HP.is_causal)
        else:
            q_seq = q_seq.transpose(1, 2)
            k_seq = k_seq.transpose(1, 2)
            v_seq = v_seq.transpose(1, 2)
            seq_out = F.scaled_dot_product_attention(q_seq, k_seq, v_seq, is_causal=HP.is_causal)
            seq_out = seq_out.transpose(1, 2)

        seq_out = seq_out.contiguous().view(B, T, H)

        # Skip connection: V around seq attention
        seq_out = seq_out + v

        # ── Feature attention (non-causal, Q=K from q) ──
        q_feat = q.view(B * T, NF, DD)
        seq_feat = seq_out.view(B * T, NF, DD)

        out = feature_attention(q_feat, q_feat, seq_feat, self.feat_activation)
        out = F.silu(out.view(B, T, H))     # post-activation
        out = self.down_proj(out)

        return x + out


class TransformerMamba3Block(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.ln0 = RMSNorm(d_model)
        self.mamba3 = Mamba3Mixer(d_model, n_head)
        self.ln1 = RMSNorm(d_model)
        self.attn = SelfAttention(d_model, n_head)
        self.ln2 = RMSNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function("tm3/block_mamba3"):
            x = x + self.mamba3(self.ln0(x))
        with torch.autograd.profiler.record_function("tm3/block_attn"):
            x = x + self.attn(self.ln1(x))
        with torch.autograd.profiler.record_function("tm3/block_mlp"):
            x = x + self.mlp(self.ln2(x))
        return x


def chunked_cross_entropy(
    hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute cross-entropy without materializing full (B*T, vocab) logits.

    Processes chunk_size tokens at a time through the LM head and loss,
    avoiding the ~6 GB logits tensor at batch 32 / seqlen 2048 / vocab 50k.
    """
    B, T, D = hidden.shape
    hidden_flat = hidden.reshape(-1, D)       # (B*T, D)
    targets_flat = targets.reshape(-1)        # (B*T,)
    total_tokens = B * T
    loss_sum = hidden.new_zeros(())
    for start in range(0, total_tokens, chunk_size):
        end = min(start + chunk_size, total_tokens)
        logits_chunk = F.linear(hidden_flat[start:end], weight)  # (chunk, vocab)
        loss_sum = loss_sum + F.cross_entropy(logits_chunk, targets_flat[start:end], reduction="sum")
    return loss_sum / total_tokens


def _init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)


def _run_blocks(blocks: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
    """Run a sequence of blocks with optional gradient checkpointing."""
    for block in blocks:
        if HP.grad_ckpt and x.requires_grad:
            x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        else:
            x = block(x)
    return x


# ── Composite (byte-factored) embedding ─────────────────────────────────────

def _build_token_byte_table(vocab_size: int, max_bytes: int = 16, pad_idx: int = 256):
    """Build a lookup table mapping each token ID to its byte sequence, padded to max_bytes."""
    if HP.data_format == "yamit" and not (HP.token_bytes_path and os.path.isfile(HP.token_bytes_path)):
        raise FileNotFoundError(
            f"YAMIT mode requires TOKEN_BYTES_PATH to an existing .npy/.pt file, got: {HP.token_bytes_path!r}"
        )
    if HP.token_bytes_path and os.path.isfile(HP.token_bytes_path):
        if HP.token_bytes_path.endswith(".pt"):
            table = torch.load(HP.token_bytes_path, map_location="cpu", weights_only=True).long()
        else:
            import numpy as np
            table = torch.from_numpy(np.load(HP.token_bytes_path)).long()
        # Trim or pad byte width to match max_bytes
        if table.shape[1] < max_bytes:
            extra_cols = torch.full((table.shape[0], max_bytes - table.shape[1]), pad_idx, dtype=torch.long)
            table = torch.cat([table, extra_cols], dim=1)
        elif table.shape[1] > max_bytes:
            table = table[:, :max_bytes]
        # Trim or pad to match vocab_size
        if table.shape[0] < vocab_size:
            extra = torch.full((vocab_size - table.shape[0], max_bytes), pad_idx, dtype=torch.long)
            table = torch.cat([table, extra])
        elif table.shape[0] > vocab_size:
            table = table[:vocab_size]
        return table
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    table = torch.full((vocab_size, max_bytes), pad_idx, dtype=torch.long)
    for i in range(min(vocab_size, enc.n_vocab)):
        try:
            b = enc.decode_single_token_bytes(i)
            if len(b) <= max_bytes:
                for j, byte_val in enumerate(b):
                    table[i, j] = byte_val
        except Exception:
            pass  # special tokens / gaps get all-pad
    return table


class CompositeEmbedding(nn.Module):
    """Matryoshka byte-factored embedding: no zero-padding, variable byte utilization.

    Each token's d_model dims are packed from its actual bytes:
      - byte_embed: 257 x byte_budget
      - token_params: V x (max_tok_params + max_soak)

    max_tok_params = 16 * tpb  — spread across N slots (tps = mtp // N per slot)
    max_soak = max(byte_budget % N for N in 1..16)  — fills remainder dims

    Layout: [slot_0 | slot_1 | ... | slot_{N-1} | tok_remainder | soak]
    Every dim is learned. No zero-padding.
    """
    def __init__(self, vocab_size: int, model_dim: int, token_per_byte: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.token_per_byte = token_per_byte
        self.max_bytes = 16
        self.max_tok_params = self.max_bytes * token_per_byte  # 128
        self.byte_budget = model_dim - self.max_tok_params      # 640
        self.max_soak = max(self.byte_budget % N for N in range(1, self.max_bytes + 1))
        self.pad_idx = 256  # sentinel in token_bytes table (never looked up — matryoshka only reads :N)

        self.byte_embed = nn.Embedding(256, self.byte_budget)

        self.token_params = nn.Embedding(vocab_size, self.max_tok_params + self.max_soak)

        # Precompute byte counts
        token_bytes_table = _build_token_byte_table(vocab_size, self.max_bytes, self.pad_idx)
        n_bytes = (token_bytes_table != self.pad_idx).sum(dim=1).clamp(min=1)
        # Clamp pad entries to valid byte range — matryoshka only reads :N real bytes,
        # but tokens with 0 real bytes (clamped to 1) would read the pad value.
        token_bytes_table = token_bytes_table.clamp(max=255)

        self.register_buffer('token_bytes', token_bytes_table, persistent=False)
        self.register_buffer('n_bytes', n_bytes, persistent=False)

        self._unique_n = sorted(set(n_bytes.tolist()))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Assemble embeddings only for tokens in the batch. No full-vocab rebuild."""
        shape = token_ids.shape
        flat = token_ids.reshape(-1)                                # (M,)
        D = self.model_dim
        bb = self.byte_budget
        mtp = self.max_tok_params

        n = self.n_bytes[flat]                                      # (M,)
        byte_seqs = self.token_bytes[flat]                          # (M, 16)
        tp = self.token_params(flat)                                # (M, mtp + max_soak)

        parts = []
        order = []
        for N in self._unique_n:
            idx = (n == N).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue

            bps = bb // N                                           # byte dims per slot
            tps = mtp // N                                          # token dims per slot
            slot_size = bps + tps
            remainder = D - N * slot_size                           # leftover dims at end

            bs = byte_seqs[idx, :N]
            be = self.byte_embed(bs)[:, :, :bps]                    # (count, N, bps)

            tp_sub = tp[idx]
            tp_per_slot = tp_sub[:, :N * tps].view(-1, N, tps)

            slots = torch.cat([be, tp_per_slot], dim=-1)            # (count, N, slot_size)
            assembled = slots.reshape(-1, N * slot_size)            # (count, N * slot_size)

            if remainder > 0:
                tok_rem = mtp % N                                   # leftover from slot division
                soak = bb % N                                       # dims that were zero-padded
                # tok_rem from regular params, soak from soak region
                rem_parts = []
                if tok_rem > 0:
                    rem_parts.append(tp_sub[:, N * tps : N * tps + tok_rem])
                if soak > 0:
                    rem_parts.append(tp_sub[:, mtp : mtp + soak])
                rem = torch.cat(rem_parts, dim=-1) if len(rem_parts) > 1 else rem_parts[0]
                assembled = torch.cat([assembled, rem], dim=-1)     # (count, D)

            parts.append(assembled)
            order.append(idx)

        all_parts = torch.cat(parts, dim=0)                         # (M, D)
        all_order = torch.cat(order, dim=0)
        _, restore = all_order.sort()
        out = all_parts[restore].view(*shape, D)
        return _unit_norm(out)


class DecoderHead(nn.Module):
    """ML-Decoder style cross-attention head over byte-slot features.

    Reshapes hidden (B, T, d_model) → (B, T, 16, dps) byte slots, then uses
    per-vocab-token queries to cross-attend over the 16 positions.
    Logits produced via dot-product readout: (context * query).sum(-1).
    For training, use streamed_cross_entropy() to avoid materializing (B, T, V, 16).
    """
    def __init__(self, vocab_size: int, d_model: int, max_bytes: int = 16):
        super().__init__()
        assert d_model % max_bytes == 0
        self.dps = d_model // max_bytes
        self.max_bytes = max_bytes
        self.vocab_size = vocab_size

        self.queries = nn.Embedding(vocab_size, self.dps)
        self.k_proj = nn.Linear(self.dps, self.dps, bias=False)
        self.v_proj = nn.Linear(self.dps, self.dps, bias=False)
        self.scale = self.dps ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        slots = x.view(B, T, self.max_bytes, self.dps)              # (B, T, 16, dps)
        k = self.k_proj(slots)                                       # (B, T, 16, dps)
        v = self.v_proj(slots)                                       # (B, T, 16, dps)
        q = self.queries.weight                                      # (V, dps)

        # Cross-attention: each vocab query attends over 16 byte positions
        scores = torch.einsum('btsd,vd->btvs', k, q) * self.scale   # (B, T, V, 16)
        attn = scores.softmax(dim=-1)                                # (B, T, V, 16)

        # Fused readout: dot(attn-weighted v, q) without materializing (B,T,V,dps)
        v_dot_q = torch.einsum('btsd,vd->btvs', v, q)               # (B, T, V, 16)
        return (attn * v_dot_q).sum(dim=-1)                          # (B, T, V)

    @torch._dynamo.disable()
    def streamed_cross_entropy(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        vocab_chunk_size: int = 512,
        token_chunk_size: int = 1024,
    ) -> torch.Tensor:
        """Compute CE loss without building full (B, T, V) or (B, T, V, 16) tensors.

        Uses two-level chunking over flattened tokens and vocabulary classes, and
        accumulates log-sum-exp in a numerically stable streaming form.
        """
        B, T, D = x.shape
        if targets.shape != (B, T):
            raise ValueError(f"targets shape {targets.shape} must match {(B, T)}")

        slots = x.view(B, T, self.max_bytes, self.dps)              # (B, T, 16, dps)
        k = self.k_proj(slots).reshape(B * T, self.max_bytes, self.dps)
        v = self.v_proj(slots).reshape(B * T, self.max_bytes, self.dps)
        t_flat = targets.reshape(B * T)
        q_all = self.queries.weight                                  # (V, dps)
        q_all_t = q_all.t().contiguous()                             # (dps, V)

        if token_chunk_size <= 0:
            token_chunk_size = 1024
        if vocab_chunk_size <= 0:
            vocab_chunk_size = self.vocab_size

        total_tokens = B * T
        loss_sum = x.new_zeros((), dtype=torch.float32)
        use_ckpt = self.training and x.requires_grad

        # Fast path: one vocab block per token chunk (matches linear-head style)
        if vocab_chunk_size >= self.vocab_size:
            def _fast_chunk_loss(k_tok: torch.Tensor, v_tok: torch.Tensor, t_tok: torch.Tensor) -> torch.Tensor:
                # Keep layout (N, 16, V): softmax over byte-slot axis (16)
                scores = torch.matmul(k_tok, q_all_t) * self.scale       # (N, 16, V)
                attn = scores.softmax(dim=1)
                v_dot_q = torch.matmul(v_tok, q_all_t)                   # (N, 16, V)
                logits = (attn * v_dot_q).sum(dim=1)                     # (N, V)
                return F.cross_entropy(logits, t_tok, reduction="sum")

            for tok_start in range(0, total_tokens, token_chunk_size):
                tok_end = min(tok_start + token_chunk_size, total_tokens)
                k_tok = k[tok_start:tok_end]                             # (N, 16, dps)
                v_tok = v[tok_start:tok_end]                             # (N, 16, dps)
                t_tok = t_flat[tok_start:tok_end]                        # (N,)
                if use_ckpt:
                    chunk_loss = torch.utils.checkpoint.checkpoint(
                        _fast_chunk_loss, k_tok, v_tok, t_tok, use_reentrant=False
                    )
                else:
                    chunk_loss = _fast_chunk_loss(k_tok, v_tok, t_tok)
                loss_sum = loss_sum + chunk_loss.float()

            return loss_sum / total_tokens

        for tok_start in range(0, total_tokens, token_chunk_size):
            tok_end = min(tok_start + token_chunk_size, total_tokens)
            k_tok = k[tok_start:tok_end]                             # (N, 16, dps)
            v_tok = v[tok_start:tok_end]                             # (N, 16, dps)
            t_tok = t_flat[tok_start:tok_end]                        # (N,)
            N = k_tok.shape[0]

            max_logit = torch.full((N,), -float("inf"), device=x.device, dtype=torch.float32)
            sum_exp = torch.zeros((N,), device=x.device, dtype=torch.float32)
            target_logit = torch.zeros((N,), device=x.device, dtype=torch.float32)

            for v_start in range(0, self.vocab_size, vocab_chunk_size):
                v_end = min(v_start + vocab_chunk_size, self.vocab_size)
                q_chunk_t = q_all_t[:, v_start:v_end]                # (dps, C)

                # Compute in (N, 16, C) layout; softmax over slot axis.
                scores = torch.matmul(k_tok, q_chunk_t) * self.scale
                attn = scores.softmax(dim=1)
                v_dot_q = torch.matmul(v_tok, q_chunk_t)
                logits_chunk = (attn * v_dot_q).sum(dim=1).float()   # (N, C)

                chunk_max = logits_chunk.max(dim=1).values
                new_max = torch.maximum(max_logit, chunk_max)
                sum_exp = sum_exp * torch.exp(max_logit - new_max)
                sum_exp = sum_exp + torch.exp(logits_chunk - new_max.unsqueeze(1)).sum(dim=1)
                max_logit = new_max

                local_idx = t_tok - v_start
                in_chunk = (local_idx >= 0) & (local_idx < (v_end - v_start))
                safe_idx = local_idx.clamp(0, (v_end - v_start) - 1)
                gathered = logits_chunk.gather(1, safe_idx.unsqueeze(1)).squeeze(1)
                target_logit = torch.where(in_chunk, gathered, target_logit)

            log_denom = max_logit + torch.log(sum_exp)
            loss_sum = loss_sum + (log_denom - target_logit).sum()

        return loss_sum / total_tokens


class StructuredLinearHead(nn.Module):
    """Structured LM head with per-token query over byte slots: (B,T,16,48) -> (B,T,V)."""

    def __init__(self, vocab_size: int, d_model: int, max_bytes: int = 16):
        super().__init__()
        assert d_model % max_bytes == 0
        self.max_bytes = max_bytes
        self.d_model = d_model
        self.dps = d_model // max_bytes
        self.q_proj = nn.Linear(d_model, self.dps, bias=False)
        self.k_proj = nn.Linear(self.dps, self.dps, bias=False)
        self.v_proj = nn.Linear(self.dps, self.dps, bias=False)
        self.scale = self.dps ** -0.5
        # Base readout from pooled slot state.
        self.proj = nn.Linear(self.dps, vocab_size, bias=False)
        # Explicit vocab query bank (one learned 48-d query per token id).
        self.query_bank = nn.Embedding(vocab_size, self.dps)
        self.out_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Expected hidden dim {self.d_model}, got {D}")
        slots = x.view(B, T, self.max_bytes, self.dps)
        # Per-token query from full hidden state (not per-output-token query bank).
        q = self.q_proj(x).unsqueeze(2)                                   # (B, T, 1, dps)
        k = self.k_proj(slots)                                            # (B, T, 16, dps)
        v = self.v_proj(slots)                                            # (B, T, 16, dps)
        scores = torch.einsum('btqd,btsd->btqs', q, k) * self.scale      # (B, T, 1, 16)
        attn = scores.softmax(dim=-1)
        pooled = torch.einsum('btqs,btsd->btqd', attn, v).squeeze(2)      # (B, T, dps)
        base_logits = self.proj(pooled)
        bank_logits = F.linear(q.squeeze(2), self.query_bank.weight, self.out_bias)
        return base_logits + bank_logits


class PITTokenInterface(nn.Module):
    """Pseudo-Inverse Tying interface: E=Z T^{-1}, W_out=T Z^T."""

    def __init__(self, vocab_size: int, d_model: int,
                 eps: float = 1e-6, min_diag: float = 1e-3,
                 orth_init: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.eps = eps
        self.min_diag = max(min_diag, eps)

        self.memory = nn.Parameter(torch.empty(vocab_size, d_model))         # Z
        self.chol_raw = nn.Parameter(torch.zeros(d_model, d_model))          # unconstrained L
        self.embed_norm = RMSNorm(d_model)

        self.reset_parameters(orth_init=orth_init)

    def reset_parameters(self, orth_init: bool = True):
        with torch.no_grad():
            if orth_init:
                q, _ = torch.linalg.qr(torch.randn(self.vocab_size, self.d_model), mode="reduced")
                self.memory.copy_(q)
            else:
                nn.init.normal_(self.memory, mean=0.0, std=0.02)

            self.chol_raw.zero_()
            # softplus^{-1}(1.0): initialize T near identity.
            diag_init = math.log(math.expm1(1.0))
            self.chol_raw.diagonal().fill_(diag_init)

    def _chol_factor(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        raw = torch.tril(self.chol_raw.to(device=device, dtype=dtype))
        raw_diag = torch.diagonal(raw)
        pos_diag = (F.softplus(raw_diag) + self.eps).clamp_min(self.min_diag)
        return raw - torch.diag_embed(raw_diag) + torch.diag_embed(pos_diag)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        # E = Z T^{-1}; do right-multiply by T^{-1} via stable Cholesky solves.
        z = F.embedding(token_ids, self.memory)                                   # (..., d)
        flat = z.reshape(-1, self.d_model)
        L32 = self._chol_factor(torch.float32, z.device)
        x_t = torch.cholesky_solve(flat.to(torch.float32).T, L32)                 # (d, N)
        x = x_t.T.reshape_as(z)
        return self.embed_norm(x.to(dtype=z.dtype))

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        # W_out = T Z^T, with T=LL^T.
        B, T, D = hidden.shape
        if D != self.d_model:
            raise ValueError(f"Expected hidden dim {self.d_model}, got {D}")

        L = self._chol_factor(hidden.dtype, hidden.device)
        g = hidden.reshape(-1, D) @ L
        g = g @ L.transpose(0, 1)
        logits = F.linear(g, self.memory.to(dtype=g.dtype))
        return logits.view(B, T, self.vocab_size)


class PITEmbedding(nn.Module):
    def __init__(self, interface: PITTokenInterface):
        super().__init__()
        self.interface = interface

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.interface.embed(token_ids)


class PITHead(nn.Module):
    def __init__(self, interface: PITTokenInterface):
        super().__init__()
        self.interface = interface

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.interface.project(hidden)


class CompositePITTokenInterface(nn.Module):
    """Matryoshka composite PIT: full d_model Cholesky transform, matryoshka byte assembly."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        token_per_byte: int = 8,
        eps: float = 1e-6,
        min_diag: float = 1e-3,
        orth_init: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_per_byte = token_per_byte
        self.max_bytes = 16
        self.max_tok_params = self.max_bytes * token_per_byte  # 128
        self.byte_budget = d_model - self.max_tok_params        # 640
        self.max_soak = max(self.byte_budget % N for N in range(1, self.max_bytes + 1))
        self.eps = eps
        self.min_diag = max(min_diag, eps)
        self.pad_idx = 256  # sentinel in token_bytes table (never looked up)

        self.byte_memory = nn.Parameter(torch.empty(256, self.byte_budget))
        self.token_embed = nn.Embedding(vocab_size, self.max_tok_params + self.max_soak)

        # Full d_model Cholesky factor
        self.chol_raw = nn.Parameter(torch.zeros(d_model, d_model))
        self.token_out_bias = nn.Parameter(torch.zeros(vocab_size))

        token_bytes_table = _build_token_byte_table(vocab_size, self.max_bytes, self.pad_idx)
        n_bytes = (token_bytes_table != self.pad_idx).sum(dim=1).clamp(min=1)
        token_bytes_table = token_bytes_table.clamp(max=255)
        self.register_buffer('token_bytes', token_bytes_table, persistent=False)
        self.register_buffer('n_bytes', n_bytes, persistent=False)
        self._unique_n = sorted(set(n_bytes.tolist()))

        self.reset_parameters(orth_init=orth_init)

    def reset_parameters(self, orth_init: bool = True):
        with torch.no_grad():
            if orth_init:
                # 256 orthonormal rows in R^byte_budget
                q, _ = torch.linalg.qr(torch.randn(self.byte_budget, 256), mode="reduced")
                self.byte_memory.copy_(q.T)  # (256, byte_budget)
            else:
                nn.init.normal_(self.byte_memory, mean=0.0, std=0.02)

            nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
            self.chol_raw.zero_()
            diag_init = math.log(math.expm1(1.0))
            self.chol_raw.diagonal().fill_(diag_init)

    def _chol_factor(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        raw = torch.tril(self.chol_raw.to(device=device, dtype=dtype))
        raw_diag = torch.diagonal(raw)
        pos_diag = (F.softplus(raw_diag) + self.eps).clamp_min(self.min_diag)
        return raw - torch.diag_embed(raw_diag) + torch.diag_embed(pos_diag)

    def _assemble_patterns(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Assemble matryoshka patterns for given tokens. Returns (..., d_model)."""
        shape = token_ids.shape
        flat = token_ids.reshape(-1)
        D = self.d_model
        bb = self.byte_budget
        mtp = self.max_tok_params

        n = self.n_bytes[flat]
        byte_seqs = self.token_bytes[flat]
        tp = self.token_embed(flat)

        parts = []
        order = []
        for N in self._unique_n:
            idx = (n == N).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue

            bps = bb // N                                           # byte dims per slot
            tps = mtp // N                                          # token dims per slot
            slot_size = bps + tps
            remainder = D - N * slot_size                           # leftover dims at end

            bs = byte_seqs[idx, :N]
            be = F.embedding(bs, self.byte_memory)[:, :, :bps]      # (count, N, bps)
            tp_sub = tp[idx]
            tp_per_slot = tp_sub[:, :N * tps].view(-1, N, tps)     # (count, N, tps)

            slots = torch.cat([be, tp_per_slot], dim=-1)            # (count, N, slot_size)
            assembled = slots.reshape(-1, N * slot_size)            # (count, N * slot_size)

            if remainder > 0:
                tok_rem = mtp % N                                   # leftover from slot division
                soak = bb % N                                       # dims that were zero-padded
                # tok_rem from regular params, soak from soak region
                rem_parts = []
                if tok_rem > 0:
                    rem_parts.append(tp_sub[:, N * tps : N * tps + tok_rem])
                if soak > 0:
                    rem_parts.append(tp_sub[:, mtp : mtp + soak])
                rem = torch.cat(rem_parts, dim=-1) if len(rem_parts) > 1 else rem_parts[0]
                assembled = torch.cat([assembled, rem], dim=-1)     # (count, D)

            parts.append(assembled)
            order.append(idx)

        all_parts = torch.cat(parts, dim=0)
        all_order = torch.cat(order, dim=0)
        _, restore = all_order.sort()
        return all_parts[restore].view(*shape, D)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        z = self._assemble_patterns(token_ids)                  # (..., d_model)
        flat = z.reshape(-1, self.d_model)
        L = self._chol_factor(torch.float32, z.device)
        x_t = torch.cholesky_solve(flat.to(torch.float32).T, L)
        x = x_t.T.reshape_as(z).to(dtype=z.dtype)
        return _unit_norm(x)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        B, T, D = hidden.shape
        L = self._chol_factor(hidden.dtype, hidden.device)
        T_mat = L @ L.transpose(0, 1)                          # (D, D)
        g = hidden @ T_mat                                      # (B, T, D)

        # Assemble patterns for full vocab
        all_ids = torch.arange(self.vocab_size, device=hidden.device)
        Z = self._assemble_patterns(all_ids).to(dtype=g.dtype)  # (V, D)
        logits = g @ Z.T + self.token_out_bias
        return logits


class CompositePITEmbedding(nn.Module):
    def __init__(self, interface: CompositePITTokenInterface):
        super().__init__()
        self.interface = interface

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.interface.embed(token_ids)


class CompositePITHead(nn.Module):
    def __init__(self, interface: CompositePITTokenInterface):
        super().__init__()
        self.interface = interface

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.interface.project(hidden)


class BucketedCompositePITHead(nn.Module):
    """Composite-PIT head with hash-routed sub-heads.

    Splits vocab into n_buckets via hash. A SwiGLU router selects top_k
    buckets per position. 2-level hierarchical softmax:
    P(token) = P(bucket|h) * P(token|bucket,h). Inference scores top_k buckets.
    """

    def __init__(self, interface: CompositePITTokenInterface,
                 n_buckets: int = 64, top_k: int = 8,
                 router_aux_weight: float = 0.01,
                 bucket_labels: torch.Tensor | None = None,
                 bucket_centers: torch.Tensor | None = None):
        super().__init__()
        self.interface = interface
        self.top_k = top_k
        self.router_aux_weight = router_aux_weight

        V = interface.vocab_size
        d_model = interface.d_model

        # ── Bucket assignment ──
        if bucket_labels is not None:
            n_labeled = len(bucket_labels)
            n_buckets = int(bucket_labels.max().item()) + 1
            if n_labeled < V:
                extra = torch.arange(V - n_labeled, dtype=torch.long) % n_buckets
                bucket_ids = torch.cat([bucket_labels.long(), extra])
            else:
                bucket_ids = bucket_labels[:V].long()
        else:
            ids = torch.arange(V, dtype=torch.long)
            bucket_ids = ((ids * 2654435761) % (2**32)) % n_buckets

        self.n_buckets = n_buckets
        self.register_buffer('token_to_bucket', bucket_ids)

        # Build padded bucket membership table
        bucket_lists: list[torch.Tensor] = []
        for b in range(n_buckets):
            bucket_lists.append((bucket_ids == b).nonzero(as_tuple=True)[0])

        max_bs = max(len(bl) for bl in bucket_lists)
        members = torch.zeros(n_buckets, max_bs, dtype=torch.long)
        sizes = torch.zeros(n_buckets, dtype=torch.long)
        tok_in_bucket = torch.zeros(V, dtype=torch.long)

        for b in range(n_buckets):
            sz = len(bucket_lists[b])
            sizes[b] = sz
            members[b, :sz] = bucket_lists[b]
            tok_in_bucket[bucket_lists[b]] = torch.arange(sz)

        self.register_buffer('bucket_members', members)       # (K, max_bs)
        self.register_buffer('bucket_sizes', sizes)            # (K,)
        self.register_buffer('token_in_bucket_idx', tok_in_bucket)  # (V,)
        self.max_bucket_size = max_bs
        self._bucket_sizes_list = sizes.tolist()  # plain Python list, no .item() graph breaks

        # ── Router: single SwiGLU MLP ──
        router_hidden = ((int(d_model * 8 / 3) + 255) // 256) * 256
        self.router_gate = nn.Linear(d_model, router_hidden, bias=False)
        self.router_up = nn.Linear(d_model, router_hidden, bias=False)
        self.router_down = nn.Linear(router_hidden, n_buckets, bias=False)

        # Init router from cluster centers if available
        if bucket_centers is not None:
            with torch.no_grad():
                c = bucket_centers.float()  # (K, d_model)
                c = c / (c.norm(dim=-1, keepdim=True) + 1e-8)
                if d_model >= router_hidden:
                    self.router_down.weight.copy_(c[:, :router_hidden].to(self.router_down.weight.dtype))

        # ── SiLU² correction gates (learned, init 0 = pure softmax at start) ──
        self.silu2_gate_bucket = nn.Parameter(torch.zeros(1))
        self.silu2_gate_token = nn.Parameter(torch.zeros(1))

    def _route(self, hidden: torch.Tensor) -> torch.Tensor:
        """SwiGLU router: returns (B, T, n_buckets) bucket logits."""
        return self.router_down(F.silu(self.router_gate(hidden)) * self.router_up(hidden))

    def _prepare_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply full d_model Cholesky transform: g = h @ L @ L^T."""
        iface = self.interface
        L = iface._chol_factor(hidden.dtype, hidden.device)
        T_mat = L @ L.transpose(0, 1)                          # (D, D)
        return hidden @ T_mat                                   # (B, T, D)

    def _bucket_logits(self, g: torch.Tensor, bucket_idx: int) -> torch.Tensor:
        """Compute PIT logits for one bucket using matryoshka patterns. Returns (*, bucket_size)."""
        iface = self.interface
        bs = self._bucket_sizes_list[bucket_idx]
        members = self.bucket_members[bucket_idx, :bs]

        Z = iface._assemble_patterns(members).to(dtype=g.dtype)  # (bs, D)
        return g @ Z.T + iface.token_out_bias[members]            # (*, bs)

    @staticmethod
    def _silu2(x: torch.Tensor) -> torch.Tensor:
        """SiLU-squared: (x * sigmoid(x))^2."""
        return F.silu(x).square()

    def _corrected_bucket_scores(self, logits: torch.Tensor) -> torch.Tensor:
        """softmax(logits) + α * SiLU²(logits) for bucket selection."""
        return F.softmax(logits, dim=-1) + self.silu2_gate_bucket * self._silu2(logits)

    def _corrected_token_scores(self, logits: torch.Tensor) -> torch.Tensor:
        """softmax(logits) + α * SiLU²(logits) for token selection."""
        return F.softmax(logits, dim=-1) + self.silu2_gate_token * self._silu2(logits)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Inference: top-k bucket logits, scaled by router confidence."""
        B, T, D = hidden.shape
        g = self._prepare_hidden(hidden)
        router_logits = self._route(hidden)
        bucket_scores = self._corrected_bucket_scores(router_logits)  # (B, T, K)
        _, top_k_idx = bucket_scores.topk(self.top_k, dim=-1)
        active = top_k_idx.unique()

        logits = torch.full((B, T, self.interface.vocab_size), float('-inf'),
                            device=hidden.device, dtype=hidden.dtype)
        for b_idx in active:
            b = b_idx.item()
            bs = self._bucket_sizes_list[b]
            members = self.bucket_members[b, :bs]
            confidence = bucket_scores[:, :, b].unsqueeze(-1)  # (B, T, 1)
            bl = self._corrected_token_scores(self._bucket_logits(g, b))
            logits[:, :, members] = bl * confidence
        return logits

    def routed_cross_entropy(self, hidden: torch.Tensor, targets: torch.Tensor):
        """2-level hierarchical softmax: P(token) = P(bucket|h) * P(token|bucket,h).

        Level 1: router softmax over K buckets → CE against target bucket.
        Level 2: within-bucket softmax → CE against target token in its bucket.
        Total loss = bucket_loss + token_loss. Exact, no approximation.

        Returns (total_loss, bucket_loss, token_loss, token_acc).
        """
        B, T, D = hidden.shape
        N = B * T
        g = self._prepare_hidden(hidden)  # (B, T, D)

        # ── Level 1: bucket prediction ──
        router_logits = self._route(hidden)  # (B, T, K)
        target_flat = targets.reshape(N)
        target_buckets = self.token_to_bucket[target_flat]
        target_local = self.token_in_bucket_idx[target_flat]

        bucket_loss = F.cross_entropy(router_logits.reshape(N, -1), target_buckets)

        # ── Level 2: within-bucket token prediction ──
        # Only need to compute logits for the target bucket of each position.
        # Group positions by their target bucket to vectorize.
        g_flat = g.reshape(N, D)
        token_loss_sum = torch.zeros(1, device=hidden.device, dtype=torch.float32)

        for b in range(self.n_buckets):
            in_bucket = (target_buckets == b)
            if not in_bucket.any():
                continue

            local_idx = target_local[in_bucket]  # (n_b,)
            bl = self._bucket_logits(g_flat[in_bucket], b).float()  # (n_b, bs)
            token_loss_sum += F.cross_entropy(bl, local_idx, reduction='sum')

        token_loss = token_loss_sum / N
        total_loss = bucket_loss + token_loss

        # ── Accuracy: argmax across router's top-k buckets (matches inference) ──
        with torch.no_grad():
            bucket_scores = self._corrected_bucket_scores(router_logits).reshape(N, -1)
            _, top_k_idx = bucket_scores.topk(self.top_k, dim=-1)  # (N, top_k)
            active_buckets = top_k_idx.unique().tolist()
            best_token = torch.zeros(N, device=hidden.device, dtype=torch.long)
            best_score = torch.full((N,), float('-inf'), device=hidden.device)
            for b in active_buckets:
                bs = self._bucket_sizes_list[b]
                members = self.bucket_members[b, :bs]
                in_topk = (top_k_idx == b).any(dim=-1)  # (N,)
                if not in_topk.any():
                    continue
                confidence = bucket_scores[in_topk, b].unsqueeze(-1)  # (n, 1)
                bl = self._corrected_token_scores(self._bucket_logits(g_flat[in_topk], b).float())
                bl = bl * confidence
                top_val, top_idx = bl.max(dim=-1)
                improved = top_val > best_score[in_topk]
                idx_into_n = in_topk.nonzero(as_tuple=True)[0]
                update_mask = improved
                best_score[idx_into_n[update_mask]] = top_val[update_mask]
                best_token[idx_into_n[update_mask]] = members[top_idx[update_mask]]
            token_acc = (best_token == target_flat).float().mean()

        return total_loss, bucket_loss, token_loss, token_acc


def _resolved_head_type() -> str:
    head_type = HP.lm_head_type.strip().lower()
    if not head_type:
        return "decoder" if HP.decoder_head else "linear"
    return head_type


def _make_embed():
    """Create token embedding — standard or composite (byte-factored)."""
    if HP.composite_embed:
        return CompositeEmbedding(
            HP.vocab_size, HP.d_model,
            token_per_byte=HP.composite_token_dims,
        )
    return nn.Embedding(HP.vocab_size, HP.d_model)


def _make_head():
    """Create LM head — standard linear or DecoderHead (cross-attention over byte slots)."""
    head_type = _resolved_head_type()
    if head_type == "decoder":
        return DecoderHead(HP.vocab_size, HP.d_model)
    if head_type == "linear48":
        return StructuredLinearHead(HP.vocab_size, HP.d_model)
    if head_type == "pit":
        raise ValueError("LM_HEAD_TYPE=pit requires paired embed/head construction")
    if head_type == "linear":
        return nn.Linear(HP.d_model, HP.vocab_size, bias=False)
    raise ValueError(f"Unknown LM_HEAD_TYPE={head_type}")


def _make_embed_head_pair():
    """Create embedding + head pair, including coupled PIT interface."""
    head_type = _resolved_head_type()
    if head_type in ("pit", "bucketed_pit"):
        if HP.composite_embed or head_type == "bucketed_pit":
            interface = CompositePITTokenInterface(
                HP.vocab_size,
                HP.d_model,
                token_per_byte=HP.composite_token_dims,
                eps=HP.pit_eps,
                min_diag=HP.pit_min_diag,
                orth_init=HP.pit_orth_init,
            )
            if head_type == "bucketed_pit":
                bucket_labels = None
                bucket_centers = None
                if HP.pit_bucket_mode == "semantic":
                    if HP.pit_bucket_labels and os.path.isfile(HP.pit_bucket_labels):
                        import numpy as np
                        bucket_labels = torch.from_numpy(np.load(HP.pit_bucket_labels))
                        if HP.pit_bucket_centers and os.path.isfile(HP.pit_bucket_centers):
                            bucket_centers = torch.from_numpy(np.load(HP.pit_bucket_centers))
                return (CompositePITEmbedding(interface),
                        BucketedCompositePITHead(interface,
                                                  n_buckets=HP.pit_n_buckets,
                                                  top_k=HP.pit_top_k,
                                                  router_aux_weight=HP.pit_router_aux_weight,
                                                  bucket_labels=bucket_labels,
                                                  bucket_centers=bucket_centers))
            return CompositePITEmbedding(interface), CompositePITHead(interface)

        interface = PITTokenInterface(
            HP.vocab_size,
            HP.d_model,
            eps=HP.pit_eps,
            min_diag=HP.pit_min_diag,
            orth_init=HP.pit_orth_init,
        )
        return PITEmbedding(interface), PITHead(interface)
    return _make_embed(), _make_head()


def _tie_weights(model: nn.Module):
    """Post-init fixups (placeholder for future use)."""
    pass


def _get_embed_weight(model: nn.Module) -> torch.Tensor | None:
    """Extract the raw embedding weight matrix for standard CE computation.

    Returns (vocab_size, d_model) tensor, or None if not extractable.
    """
    wte = getattr(model, 'wte', None)
    if wte is None:
        return None
    # nn.Embedding
    if isinstance(wte, nn.Embedding):
        return wte.weight
    # CompositeEmbedding — no single weight matrix
    # PITEmbedding — interface.memory is the Z matrix, not a standard embed
    # CompositePITEmbedding — same
    return None


@torch.no_grad()
def _std_cross_entropy(hidden: torch.Tensor, targets: torch.Tensor,
                       embed_weight: torch.Tensor) -> float:
    """Compute standard cross-entropy from hidden states and embedding weight.

    hidden: (B, T, D), targets: (B, T), embed_weight: (V, D).
    Returns scalar loss value.
    """
    logits = F.linear(hidden.float(), embed_weight.float())
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           targets.reshape(-1)).item()


# ── Byte-attention MLP ───────────────────────────────────────────────────────

class ByteAttentionMLP(nn.Module):
    """SwiGLU MLP with byte attention replacing element-wise gating.

    SwiGLU:       out = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    ByteAttn:     out = down_proj(SiLU(byte_attn(SiLU(gate_proj(x)), up_proj(x))))

    gate_proj provides Q=K (Reformer-style), up_proj provides V.
    Multi-head attention with RoPE over 16 byte slots, acausal.
    Same three projections as SwiGLU, same param count.
    """

    def __init__(self, d_model: int, n_byte_heads: int):
        super().__init__()
        hidden = int(d_model * 8 / 3)
        hidden = ((hidden + 255) // 256) * 256
        self.hidden = hidden
        self.max_bytes = 16
        assert hidden % self.max_bytes == 0
        self.desc_dim = hidden // self.max_bytes
        self.n_byte_heads = n_byte_heads
        assert self.desc_dim % n_byte_heads == 0
        self.byte_head_dim = self.desc_dim // n_byte_heads

        self.gate_proj = nn.Linear(d_model, hidden, bias=False)  # → Q=K
        self.up_proj = nn.Linear(d_model, hidden, bias=False)    # → V
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

        # Precompute RoPE for byte positions (0-15)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.byte_head_dim, 2).float() / self.byte_head_dim))
        positions = torch.arange(self.max_bytes).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer('byte_cos', freqs.cos()[None, :, None, :], persistent=False)
        self.register_buffer('byte_sin', freqs.sin()[None, :, None, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # gate_proj → Q=K (with SiLU pre-act), up_proj → V
        qk = F.silu(self.gate_proj(x))                                              # (B, T, hidden)
        v = self.up_proj(x)                                                          # (B, T, hidden)

        qk = qk.view(B * T, self.max_bytes, self.n_byte_heads, self.byte_head_dim)  # (B*T, 16, H, D)
        v = v.view(B * T, self.max_bytes, self.n_byte_heads, self.byte_head_dim)

        # RoPE over byte positions
        qk = apply_rotary(qk, self.byte_cos, self.byte_sin)

        # SDPA: (B*T, n_heads, 16, head_dim), Q=K
        qk, v = qk.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(qk, qk, v, is_causal=False)
        out = out.transpose(1, 2).contiguous()                                       # (B*T, 16, H, D)

        out = out.reshape(B, T, self.hidden)
        out = F.silu(out)                                                            # post-attention activation
        return self.down_proj(out)


class TransformerByteAttnBlock(nn.Module):
    """Pre-norm block: sequence attention + byte-attention MLP."""

    def __init__(self, d_model: int, n_head: int, n_byte_heads: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = SelfAttention(d_model, n_head)
        self.ln2 = RMSNorm(d_model)
        self.mlp = ByteAttentionMLP(d_model, n_byte_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte, self.lm_head = _make_embed_head_pair()
        if HP.ngpt:
            self.blocks = nn.ModuleList([NGPTBlock(HP.d_model, HP.n_head) for _ in range(HP.n_layer)])
            self.ln_f = nn.Identity()  # nGPT: hidden state is already unit-norm
            # Logit scaling: per-vocab learnable temperature
            s_z_scale = 1.0 / math.sqrt(HP.d_model)
            self.s_z = _ngpt_scale_param((HP.vocab_size,), HP.ngpt_sz_init, s_z_scale)
        else:
            self.blocks = nn.ModuleList([TransformerBlock(HP.d_model, HP.n_head) for _ in range(HP.n_layer)])
            self.ln_f = RMSNorm(HP.d_model)
            self.s_z = None
        self.apply(_init_weights)
        _tie_weights(self)
        # nGPT: normalize all weights after init
        if HP.ngpt:
            _ngpt_normalize_weights(self)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.wte(idx)
        x = _run_blocks(self.blocks, x)
        x = self.ln_f(x)
        self._last_hidden = x.detach()
        if targets is not None and isinstance(self.lm_head, DecoderHead):
            vocab_chunk = HP.vocab_size if HP.decoder_head_vocab_chunk <= 0 else HP.decoder_head_vocab_chunk
            loss = self.lm_head.streamed_cross_entropy(
                x,
                targets,
                vocab_chunk_size=vocab_chunk,
                token_chunk_size=HP.decoder_head_token_chunk,
            )
            self._last_acc = None
            return None, loss
        if targets is not None and isinstance(self.lm_head, BucketedCompositePITHead):
            total_loss, _, _, acc = self.lm_head.routed_cross_entropy(x, targets)
            self._last_acc = acc
            return None, total_loss
        logits = self.lm_head(x)
        if self.s_z is not None:
            logits = logits * _ngpt_actual(self.s_z)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            self._last_acc = (logits.detach().argmax(-1) == targets).float().mean()
        return logits, loss


class GPTTransformerConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = _make_embed()
        self.blocks = nn.ModuleList([TransformerMamba3Block(HP.d_model, HP.n_head) for _ in range(HP.n_layer)])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.apply(_init_weights)
        _tie_weights(self)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = _run_blocks(self.blocks, self.wte(idx))
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTTransformerShift(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = _make_embed()
        self.blocks = nn.ModuleList([TransformerShiftBlock(HP.d_model, HP.n_head) for _ in range(HP.n_layer)])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.apply(_init_weights)
        _tie_weights(self)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = _run_blocks(self.blocks, self.wte(idx))
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTGatedNeighbor(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = _make_embed()
        self.blocks = nn.ModuleList([GatedNeighborBlock(HP.d_model, HP.n_head) for _ in range(HP.n_layer)])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.apply(_init_weights)
        _tie_weights(self)
        # Re-apply gate init after _init_weights (which overwrites it)
        for block in self.blocks:
            nn.init.zeros_(block.attn.gate_proj.weight)
            nn.init.constant_(block.attn.gate_proj.bias, -2.0)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = _run_blocks(self.blocks, self.wte(idx))
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTFusedGatedNeighbor(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = _make_embed()
        inner = HP.fused_inner_dim if HP.fused_inner_dim > 0 else None
        paired_str = os.environ.get("PAIRED_HEAD_LAYERS", "0,2,5,9")
        paired_layers = set(int(x) for x in paired_str.split(",") if x.strip()) if paired_str.strip() else set()
        self.blocks = nn.ModuleList([
            FusedGatedNeighborBlock(HP.d_model, HP.n_head, inner_dim=inner, paired=(i in paired_layers))
            for i in range(HP.n_layer)
        ])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.apply(_init_weights)
        _tie_weights(self)
        # Re-apply gate init after _init_weights (which overwrites it)
        for block in self.blocks:
            nn.init.zeros_(block.neighbor_gate_proj.weight)
            nn.init.constant_(block.neighbor_gate_proj.bias, -2.0)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.wte(idx)
        for block in self.blocks:
            if HP.grad_ckpt and x.requires_grad:
                x = x + torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = x + block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTS6(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = _make_embed()
        modes = tuple(x.strip() for x in HP.s6_scan_state_modes.split(",") if x.strip())
        if len(modes) != 3:
            raise ValueError("S6_SCAN_STATE_MODES must have 3 comma-separated values")
        cfg = USBConfig(
            d_model=HP.d_model,
            headdim=HP.s6_headdim,
            expansion_factor=HP.s6_expansion_factor,
            post_scan_attention=HP.s6_post_scan_attention,
            scan_state_modes=modes,
        )
        self.blocks = nn.ModuleList([USBBlock(cfg) for _ in range(HP.n_layer)])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.apply(_init_weights)
        _tie_weights(self)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.wte(idx)
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        x = self.ln_f(x)
        if targets is not None:
            loss = chunked_cross_entropy(x, self.lm_head.weight, targets)
            return None, loss
        logits = self.lm_head(x)
        return logits, None


class GPTULB(nn.Module):
    """ULBBlendP wrapper for fineweb training (sigmoid gate, no feat-attn)."""
    def __init__(self, feat_attn: bool = False):
        super().__init__()
        from ulb.transformer import CausalULB
        self.inner = CausalULB(
            vocab_size=HP.vocab_size,
            dim=HP.d_model,
            n_heads=HP.n_head,
            n_layers=HP.n_layer,
            max_seq_len=HP.seq_len,
            paired=True,
            attn_mode='blend',
            feat_attn=feat_attn,
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.inner(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTULB1D(nn.Module):
    """Clean 1D ULB with silu² feat-attn MLP (for ablating 2D vs 1D)."""
    def __init__(self):
        super().__init__()
        from ulb.transformer import CausalULB1D
        self.inner = CausalULB1D(
            vocab_size=HP.vocab_size,
            d_model=HP.d_model,
            n_heads=HP.n_head,
            n_layers=HP.n_layer,
            max_seq_len=HP.seq_len,
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.inner(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTULB2D(nn.Module):
    """True 2D ULB wrapper for fineweb training."""
    def __init__(self):
        super().__init__()
        from ulb.transformer import CausalULB2D
        nf, dd = HP.n_features, HP.desc_dim
        if nf == 0 or dd == 0:
            # Auto-factor from d_model, targeting 1:3 ratio (n_features:desc_dim)
            dim = HP.d_model
            target_nf = int((dim / 3) ** 0.5)
            best = 1
            for s in range(1, int(dim ** 0.5) + 1):
                if dim % s == 0 and (dim // s) % 4 == 0:
                    if abs(s - target_nf) <= abs(best - target_nf):
                        best = s
            nf, dd = best, dim // best
        self.inner = CausalULB2D(
            vocab_size=HP.vocab_size,
            n_features=nf,
            desc_dim=dd,
            n_layers=HP.n_layer,
            max_seq_len=HP.seq_len,
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.inner(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTFeatureAttn(nn.Module):
    """GPT with feature attention MLP — SwiGLU gating replaced by feature self-attention.

    Same three projections (gate/up/down), same param count as GPTTransformer.
    Configured via FA_N_FEATURES and FA_ACTIVATION.
    """

    def __init__(self):
        super().__init__()
        nf = HP.fa_n_features
        act = HP.fa_activation
        hidden = int(HP.d_model * 8 / 3)
        hidden = ((hidden + 255) // 256) * 256
        assert hidden % nf == 0, (
            f"MLP hidden dim ({hidden}) not divisible by FA_N_FEATURES ({nf}). "
            f"Valid: {[f for f in [8, 16, 32, 64, 128, 256] if hidden % f == 0]}")
        post_act = HP.fa_post_act
        pre_act = HP.fa_pre_act
        qk_norm = HP.fa_qk_norm
        self.wte = _make_embed()
        self.blocks = nn.ModuleList([
            TransformerFeatureAttnBlock(HP.d_model, HP.n_head, nf, act, post_act=post_act, pre_act=pre_act, qk_norm=qk_norm)
            for _ in range(HP.n_layer)
        ])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.apply(_init_weights)
        _tie_weights(self)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = _run_blocks(self.blocks, self.wte(idx))
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTFusedSeqFeature(nn.Module):
    """GPT with fused seq+feature attention blocks — shared QK/V projections."""

    def __init__(self):
        super().__init__()
        if HP.fused_inner_dim > 0:
            hidden = HP.fused_inner_dim
        else:
            hidden = int(HP.d_model * 8 / 3)
        hidden = ((hidden + 255) // 256) * 256
        nf = HP.fa_n_features
        act = HP.fa_activation
        # n_seq_heads: use head_dim=64 by default
        n_seq_heads = hidden // 64
        assert hidden % nf == 0, (
            f"hidden ({hidden}) not divisible by FA_N_FEATURES ({nf})")
        self.wte = _make_embed()
        self.blocks = nn.ModuleList([
            FusedSeqFeatureAttnBlock(HP.d_model, hidden, n_seq_heads, nf, act)
            for _ in range(HP.n_layer)
        ])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.apply(_init_weights)
        _tie_weights(self)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = _run_blocks(self.blocks, self.wte(idx))
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTThreeStage(nn.Module):
    """GPT with three-stage blocks: seq attn → feat attn → MLP. Same params as baseline."""

    def __init__(self):
        super().__init__()
        nf = HP.fa_n_features
        act = HP.fa_activation
        assert HP.d_model % nf == 0, (
            f"d_model ({HP.d_model}) not divisible by FA_N_FEATURES ({nf})")
        self.wte = _make_embed()
        self.blocks = nn.ModuleList([
            ThreeStageBlock(HP.d_model, HP.n_head, nf, act)
            for _ in range(HP.n_layer)
        ])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.apply(_init_weights)
        _tie_weights(self)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = _run_blocks(self.blocks, self.wte(idx))
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTQVO(nn.Module):
    """GPT with QVO blocks: Q + V + O only, no K projection.

    Same architecture as ThreeStageFSA with Q=K, but K weight is removed.
    Saves ~590K params per layer (768*768).
    """

    def __init__(self):
        super().__init__()
        nf = HP.fa_n_features
        act = HP.fa_activation
        assert HP.d_model % nf == 0, (
            f"d_model ({HP.d_model}) not divisible by FA_N_FEATURES ({nf})")
        self.wte = _make_embed()
        self.blocks = nn.ModuleList([
            QVOBlock(HP.d_model, HP.n_head, nf, act)
            for _ in range(HP.n_layer)
        ])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.apply(_init_weights)
        _tie_weights(self)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = _run_blocks(self.blocks, self.wte(idx))
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTThreeStageFSA(nn.Module):
    """GPT with three-stage blocks: feat attn → seq attn → MLP.

    Reversed order from GPTThreeStage to test whether reorganizing features
    before inter-token communication helps. Zero extra params — reuses Q
    projection for both feature attention (pre-RoPE Q as feat Q=K) and
    sequence attention (re-projected from post-feat-attn state + RoPE).
    """

    def __init__(self):
        super().__init__()
        nf = HP.fa_n_features
        act = HP.fa_activation
        assert HP.d_model % nf == 0, (
            f"d_model ({HP.d_model}) not divisible by FA_N_FEATURES ({nf})")
        self.wte = _make_embed()
        self.blocks = nn.ModuleList([
            ThreeStageFSABlock(HP.d_model, HP.n_head, nf, act)
            for _ in range(HP.n_layer)
        ])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.apply(_init_weights)
        _tie_weights(self)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = _run_blocks(self.blocks, self.wte(idx))
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTDualQ(nn.Module):
    """GPT with DualQ blocks: Q_feat + Q_seq + V + O, no K projection.

    Both attentions use Q=K. K is replaced by Q_feat.
    Seq Q comes from post-feat-attn space.
    """

    def __init__(self):
        super().__init__()
        nf = HP.fa_n_features
        act = HP.fa_activation
        assert HP.d_model % nf == 0, (
            f"d_model ({HP.d_model}) not divisible by FA_N_FEATURES ({nf})")
        self.wte = _make_embed()
        self.blocks = nn.ModuleList([
            DualQBlock(HP.d_model, HP.n_head, nf, act)
            for _ in range(HP.n_layer)
        ])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.apply(_init_weights)
        _tie_weights(self)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = _run_blocks(self.blocks, self.wte(idx))
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTFusedQKV(nn.Module):
    """GPT with fused QKV blocks — full Q/K/V shared across seq + feature attention."""

    def __init__(self):
        super().__init__()
        if HP.fused_inner_dim > 0:
            hidden = HP.fused_inner_dim
        else:
            hidden = int(HP.d_model * 8 / 3)
        hidden = ((hidden + 255) // 256) * 256
        nf = HP.fa_n_features
        act = HP.fa_activation
        n_seq_heads = hidden // 64
        assert hidden % nf == 0, (
            f"hidden ({hidden}) not divisible by FA_N_FEATURES ({nf})")
        self.wte = _make_embed()
        self.blocks = nn.ModuleList([
            FusedQKVBlock(HP.d_model, hidden, n_seq_heads, nf, act)
            for _ in range(HP.n_layer)
        ])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.apply(_init_weights)
        _tie_weights(self)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = _run_blocks(self.blocks, self.wte(idx))
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTByteAttn(nn.Module):
    """GPT with byte-attention MLP — MLP replaced by multi-head attention over 16 byte slots."""

    def __init__(self):
        super().__init__()
        self.wte = _make_embed()
        self.blocks = nn.ModuleList([
            TransformerByteAttnBlock(HP.d_model, HP.n_head, HP.ba_n_byte_heads)
            for _ in range(HP.n_layer)
        ])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.apply(_init_weights)
        _tie_weights(self)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = _run_blocks(self.blocks, self.wte(idx))
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTMoE(nn.Module):
    """GPT with Mixture of Experts MLP — each layer uses top-k gated MoE."""

    def __init__(self):
        super().__init__()
        self.wte, self.lm_head = _make_embed_head_pair()
        self.blocks = nn.ModuleList([
            TransformerMoEBlock(HP.d_model, HP.n_head, HP.n_experts, HP.top_k, bypass=HP.moe_bypass, shared_frac=HP.moe_shared, bias_lr=HP.moe_bias_lr)
            for _ in range(HP.n_layer)
        ])
        self.ln_f = RMSNorm(HP.d_model)
        self.apply(_init_weights)
        _tie_weights(self)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.wte(idx)
        for block in self.blocks:
            if HP.grad_ckpt and x.requires_grad:
                x, _ = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x, _ = block(x)
        x = self.ln_f(x)
        self._last_hidden = x.detach()
        if targets is not None and isinstance(self.lm_head, DecoderHead):
            vocab_chunk = HP.vocab_size if HP.decoder_head_vocab_chunk <= 0 else HP.decoder_head_vocab_chunk
            loss = self.lm_head.streamed_cross_entropy(
                x,
                targets,
                vocab_chunk_size=vocab_chunk,
                token_chunk_size=HP.decoder_head_token_chunk,
            )
            self._last_acc = None
            return None, loss
        if targets is not None and isinstance(self.lm_head, BucketedCompositePITHead):
            total_loss, _, _, acc = self.lm_head.routed_cross_entropy(x, targets)
            self._last_acc = acc
            return None, total_loss
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            self._last_acc = (logits.detach().argmax(-1) == targets).float().mean()
        return logits, loss


class GPTNGPTMoE(nn.Module):
    """nGPT with Mixture of Experts MLP — LERP updates on the hypersphere + MoE.

    Supports per-layer specialization:
    - Differential attention on first N layers (NGPT_DIFF_ATTN_N)
    - Paired Head Attention on odd layers (NGPT_PAIRED_ODD)
    - TrapMix skipped on even layers (NGPT_SKIP_TRAP_EVEN)
    - Sliding window on specified layers (NGPT_WINDOW_LAYERS / NGPT_WINDOW_SIZE)
    - Content-dependent embed gate at middle layer (NGPT_EMBED_GATE_LAYER)
    """

    def __init__(self):
        super().__init__()
        self.wte, self.lm_head = _make_embed_head_pair()
        D = HP.moe_dense_layers
        n_layer = HP.n_layer

        # Parse per-layer config
        window_set = set()
        if HP.ngpt_window_layers:
            window_set = {int(x.strip()) for x in HP.ngpt_window_layers.split(",") if x.strip()}

        blocks = []
        for i in range(n_layer):
            is_odd = (i % 2 == 1)
            is_dense = (i < D or i >= n_layer - D)
            attn_cfg = dict(
                layer_idx=i,
                differential=(i < HP.ngpt_diff_attn_n),
                paired=(HP.ngpt_paired_odd and is_odd),
                trap_mix=(HP.trap_mix and not (HP.ngpt_skip_trap_even and not is_odd)),
                window_size=(HP.ngpt_window_size if i in window_set else 0),
                embed_gate=(i == HP.ngpt_embed_gate_layer),
            )
            if is_dense:
                blocks.append(NGPTBlock(HP.d_model, HP.n_head, **attn_cfg))
            else:
                blocks.append(NGPTMoEBlock(
                    HP.d_model, HP.n_head, HP.n_experts, HP.top_k,
                    n_group=HP.moe_n_group, topk_group=HP.moe_topk_group,
                    scaling_factor=HP.moe_scaling_factor, bias_lr=HP.moe_bias_lr,
                    **attn_cfg))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.Identity()  # nGPT: hidden state is already unit-norm
        # Logit scaling: per-vocab learnable temperature
        s_z_scale = 1.0 / math.sqrt(HP.d_model)
        self.s_z = _ngpt_scale_param((HP.vocab_size,), HP.ngpt_sz_init, s_z_scale)
        self.apply(_init_weights)
        _tie_weights(self)
        _ngpt_normalize_weights(self)

        # Log layer config
        _cfg_parts = []
        for i, blk in enumerate(self.blocks):
            parts = []
            if blk.attn.differential:
                parts.append("diff")
            if blk.attn.paired:
                parts.append("pha")
            if blk.attn.trap_mix:
                parts.append("trap")
            if blk.attn.window_size > 0:
                parts.append(f"win{blk.attn.window_size}")
            if blk.embed_gate:
                parts.append("egate")
            mlp_type = "dense" if isinstance(blk, NGPTBlock) else "moe"
            _cfg_parts.append(f"  L{i}: {mlp_type} {' '.join(parts) if parts else 'std'}")
        if any(p for p in _cfg_parts if "std" not in p.split()[-1]):
            print("\n".join(["[GPTNGPTMoE] Per-layer config:"] + _cfg_parts))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.wte(idx)
        # Store x0 for embed-gate blocks
        for blk in self.blocks:
            if blk.embed_gate:
                blk._x0 = x
        for blk in self.blocks:
            x = blk(x)
        # x is already unit-norm from the blocks
        self._last_hidden = x.detach()
        if targets is not None and isinstance(self.lm_head, DecoderHead):
            vocab_chunk = HP.vocab_size if HP.decoder_head_vocab_chunk <= 0 else HP.decoder_head_vocab_chunk
            loss = self.lm_head.streamed_cross_entropy(
                x,
                targets,
                vocab_chunk_size=vocab_chunk,
                token_chunk_size=HP.decoder_head_token_chunk,
            )
            self._last_acc = None
            return None, loss
        if targets is not None and isinstance(self.lm_head, BucketedCompositePITHead):
            total_loss, _, _, acc = self.lm_head.routed_cross_entropy(x, targets)
            self._last_acc = acc
            return None, total_loss
        logits = self.lm_head(x)
        logits = logits * _ngpt_actual(self.s_z)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            self._last_acc = (logits.detach().argmax(-1) == targets).float().mean()
        return logits, loss


# ── LLaDA wrapper ───────────────────────────────────────────────────────────

class LLaDAWrapper(nn.Module):
    """Wraps any GPT model for masked diffusion (LLaDA) training.

    Adds a learned mask embedding that replaces masked token positions at the
    embedding level. The backbone runs bidirectionally (HP.is_causal=False).
    Loss is computed only on masked positions, weighted by 1/p_mask.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.mask_embed = nn.Parameter(torch.randn(HP.d_model) * 0.02)

    def forward(self, idx: torch.Tensor, mask: torch.Tensor,
                p_mask: torch.Tensor | None = None):
        """Forward pass for LLaDA.

        Args:
            idx: (B, T) ground-truth token IDs (clean x0)
            mask: (B, T) bool, True = masked
            p_mask: (B,) mask probability per sample (for loss weighting)

        Returns:
            (logits, loss) where loss is None if p_mask is None
        """
        # Embed tokens, then replace masked positions with mask_embed
        x = self.backbone.wte(idx)
        x = torch.where(mask.unsqueeze(-1), self.mask_embed, x)

        # Run through backbone blocks + final norm + head
        x = _run_blocks(self.backbone.blocks, x)
        x = self.backbone.ln_f(x)
        logits = self.backbone.lm_head(x)

        # SUBS parameterization: at unmasked positions, force logits to one-hot
        if HP.llada_subs:
            one_hot = torch.full_like(logits, float('-inf'))
            one_hot.scatter_(-1, idx.unsqueeze(-1), 0.0)
            logits = torch.where(mask.unsqueeze(-1), logits, one_hot)

        loss = None
        if p_mask is not None:
            per_token_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), idx.view(-1), reduction='none'
            ).view_as(idx)
            # Weight masked tokens by 1/p_mask, ignore unmasked
            p_mask_expanded = p_mask[:, None].expand_as(idx)
            weighted = per_token_loss[mask] / p_mask_expanded[mask]
            B, T = idx.shape
            loss = weighted.sum() / (B * T)

        return logits, loss


def llada_make_mask(batch: torch.Tensor, device: torch.device):
    """Create LLaDA training masks with optional antithetic sampling.

    Returns:
        mask: (B, T) bool — True = masked
        p_mask: (B,) — mask probability per sample
    """
    B, T = batch.shape
    eps = 1e-3

    if HP.llada_antithetic:
        offset = torch.arange(B, device=device, dtype=torch.float32) / B
        t = (torch.rand(1, device=device) / B + offset) % 1.0
    else:
        t = torch.rand(B, device=device)

    p_mask = (1.0 - eps) * t + eps  # (B,) in [eps, 1.0]
    mask = torch.rand(B, T, device=device) < p_mask[:, None]
    return mask, p_mask


def build_model() -> nn.Module:
    if HP.model_type == "transformer":
        return GPTTransformer()
    if HP.model_type == "feat_attn":
        return GPTFeatureAttn()
    if HP.model_type == "fused_seq_feat":
        return GPTFusedSeqFeature()
    if HP.model_type == "fused_qkv":
        return GPTFusedQKV()
    if HP.model_type == "three_stage":
        return GPTThreeStage()
    if HP.model_type == "three_stage_fsa":
        return GPTThreeStageFSA()
    if HP.model_type == "qvo":
        return GPTQVO()
    if HP.model_type == "dual_q":
        return GPTDualQ()
    if HP.model_type == "transformer_shift":
        return GPTTransformerShift()
    if HP.model_type == "transformer_s4d":
        return GPTTransformerConv()
    if HP.model_type == "transformer_gate":
        return GPTGatedNeighbor()
    if HP.model_type == "fused_gate":
        return GPTFusedGatedNeighbor()
    if HP.model_type == "s6":
        return GPTS6()
    if HP.model_type == "ulb":
        return GPTULB(feat_attn=False)
    if HP.model_type == "ulb_fa":
        return GPTULB1D()
    if HP.model_type == "ulb_2d":
        return GPTULB2D()
    if HP.model_type == "byte_attn":
        return GPTByteAttn()
    if HP.model_type == "moe":
        return GPTMoE()
    if HP.model_type == "ngpt_moe":
        return GPTNGPTMoE()
    raise ValueError(f"Unknown MODEL_TYPE={HP.model_type}")


def _llada_apply_head_override(backbone: nn.Module) -> nn.Module:
    """For LLaDA, allow any backbone to use the configured LM head type."""
    if HP.model_type in ("transformer", "moe", "ngpt_moe"):
        # These already route through _make_embed_head_pair().
        return backbone

    wte, lm_head = _make_embed_head_pair()
    wte.apply(_init_weights)
    lm_head.apply(_init_weights)
    backbone.wte = wte
    backbone.lm_head = lm_head
    _tie_weights(backbone)
    return backbone


def build_model_maybe_llada() -> nn.Module:
    backbone = build_model()
    if HP.llada:
        backbone = _llada_apply_head_override(backbone)
        return LLaDAWrapper(backbone)
    return backbone


def lr_for_tokens(tokens_seen: int, total_tokens: int) -> float:
    """LR schedule based on fraction of total tokens consumed. No LR warmup —
    grad_accum ramp handles the warmup phase instead."""
    warmup_tokens = int(total_tokens * HP.warmup_frac)

    if HP.lr_schedule == "wsd":
        # Warmup-Stable-Decay with linear ramps in both S and D phases.
        # S phase: linear from peak_lr → peak_lr * wsd_decay_end_frac.
        # D phase: linear from S_end → S_end * wsd_decay_end_frac.
        # With wsd_decay_end_frac=0.1: S decays to 10% of peak, D decays to 1% of peak.
        decay_tokens = int(total_tokens * HP.wsd_decay_frac)
        stable_end = total_tokens - decay_tokens
        s_end_lr = HP.lr * HP.wsd_decay_end_frac
        if tokens_seen <= stable_end:
            # S phase: linear decay from peak to s_end_lr
            t = tokens_seen / max(1, stable_end)
            return HP.lr + (s_end_lr - HP.lr) * t
        else:
            # D phase: linear decay from s_end_lr to s_end_lr * wsd_decay_end_frac
            t = min(1.0, (tokens_seen - stable_end) / max(1, decay_tokens))
            d_end_lr = s_end_lr * HP.wsd_decay_end_frac
            return s_end_lr + (d_end_lr - s_end_lr) * t
    elif HP.lr_schedule == "pressure":
        # Hyperbolic pressure decay: lr = base / (1 + k * t/T)
        # Decays faster early but has a longer tail than cosine.
        k = 4.0
        t = tokens_seen / max(1, total_tokens)
        return HP.lr / (1.0 + k * t)
    else:
        # Default: cosine decay from warmup end to total_tokens
        t = (tokens_seen - warmup_tokens) / max(1, total_tokens - warmup_tokens)
        return HP.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))


@torch.no_grad()
def evaluate(model: nn.Module, val_stream: ShardStream, device: torch.device, world_size: int,
             ch_weights: torch.Tensor | None = None):
    """Returns (val_loss, val_acc, raw_val_loss, per_source_losses).
    val_loss: CH-weighted if ch_weights provided, else standard mean.
    raw_val_loss: always unweighted mean (None if no CH).
    per_source_losses: dict {source_name: float} if MixedShardStream, else None."""
    val_stream.reset()  # always evaluate the same data for deterministic val loss
    raw = model.module if hasattr(model, 'module') else model
    model.eval()
    loss_sum = torch.zeros(1, device=device)
    acc_sum = torch.zeros(1, device=device)
    has_acc = False
    # Per-source val loss tracking
    _mixed = isinstance(val_stream, MixedShardStream)
    if _mixed:
        _n_src = val_stream.n_sources
        _src_loss_sum = torch.zeros(_n_src, device=device)
        _src_count = torch.zeros(_n_src, device=device)
    for _ in range(HP.val_steps):
        _vbatch = val_stream.next_batch(device)
        x, y = _vbatch[0], _vbatch[1]
        src_ids = _vbatch[2] if len(_vbatch) > 2 else None
        if HP.llada:
            B, T = x.shape
            mask = torch.rand(B, T, device=device) < 0.5
            p_mask = torch.full((B,), 0.5, device=device)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = model(x, mask, p_mask)
            else:
                logits, loss = model(x, mask, p_mask)
        else:
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = model(x, y)
            else:
                logits, loss = model(x, y)
        loss_sum += loss
        if _mixed and logits is not None and src_ids is not None:
            B, T, V = logits.shape
            per_tok = F.cross_entropy(logits.view(-1, V), y.view(-1), reduction='none').view(B, T)
            per_seq = per_tok.mean(dim=1)
            for _si in range(_n_src):
                _mask = src_ids == _si
                if _mask.any():
                    _src_loss_sum[_si] += per_seq[_mask].sum()
                    _src_count[_si] += _mask.sum()
        if hasattr(raw, '_last_acc') and raw._last_acc is not None:
            acc_sum += raw._last_acc
            has_acc = True
    loss_sum /= HP.val_steps
    if has_acc:
        acc_sum /= HP.val_steps
    per_source = None
    raw_val_loss = None
    if _mixed:
        per_source = {}
        _per_src_losses = torch.zeros(_n_src, device=device)
        for i, name in enumerate(val_stream.source_names):
            _per_src_losses[i] = _src_loss_sum[i] / _src_count[i].clamp(min=1)
            per_source[name] = float(_per_src_losses[i])
        if ch_weights is not None:
            # CH-weighted val loss as main, unweighted as raw
            raw_val_loss = float(loss_sum.item())
            loss_sum = (_per_src_losses * ch_weights).sum() / ch_weights.sum()
            loss_sum = loss_sum.unsqueeze(0)
    if world_size > 1:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
        if has_acc:
            dist.all_reduce(acc_sum, op=dist.ReduceOp.AVG)
    model.train()
    return (float(loss_sum.item()),
            float(acc_sum.item()) if has_acc else None,
            raw_val_loss,
            per_source)


def main():
    rank, world_size, device = setup_dist()
    _rank_seed = int(HP.seed) + rank
    random.seed(_rank_seed)
    torch.manual_seed(_rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_rank_seed)
    try:
        import numpy as np
        np.random.seed(_rank_seed)
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # If a pre-built token_bytes table is provided, derive vocab size from it.
    if HP.token_bytes_path and os.path.isfile(HP.token_bytes_path):
        if HP.token_bytes_path.endswith(".pt"):
            _tb = torch.load(HP.token_bytes_path, map_location="cpu", weights_only=True)
            HP.vocab_size = int(_tb.shape[0])
        else:
            import numpy as np
            HP.vocab_size = int(np.load(HP.token_bytes_path, mmap_mode="r").shape[0])

    print0(rank, f"rank={rank} world_size={world_size} device={device} seed={_rank_seed}")
    _extra = f" n_features={HP.n_features} desc_dim={HP.desc_dim}" if HP.n_features > 0 and HP.desc_dim > 0 else ""
    if HP.model_type == "feat_attn" or HP.feat_attn_mlp:
        hidden = ((int(HP.d_model * 8 / 3) + 255) // 256) * 256
        _extra += f" fa_n_features={HP.fa_n_features} fa_desc_dim={hidden // HP.fa_n_features} fa_activation={HP.fa_activation} fa_pre_act={HP.fa_pre_act} fa_post_act={HP.fa_post_act} fa_qk_norm={HP.fa_qk_norm}"
    if HP.model_type in ("moe", "ngpt_moe"):
        _extra += f" n_experts={HP.n_experts} top_k={HP.top_k} bypass={HP.moe_bypass} shared={HP.moe_shared} bias_lr={HP.moe_bias_lr}"
    _head = _resolved_head_type() if HP.model_type in ("transformer", "moe", "ngpt_moe") else "fixed"
    if _head == "bucketed_pit":
        _extra += f" pit_eps={HP.pit_eps} pit_min_diag={HP.pit_min_diag} pit_n_buckets={HP.pit_n_buckets} pit_top_k={HP.pit_top_k} pit_router_aux={HP.pit_router_aux_weight} routing={HP.pit_bucket_mode}"
    elif _head == "pit":
        _extra += f" pit_eps={HP.pit_eps} pit_min_diag={HP.pit_min_diag}"
    if HP.ngpt:
        _extra += f" ngpt=True alpha_init={HP.ngpt_alpha_init}"
    if HP.dd_rope:
        _extra += " dd_rope=True"
    if HP.trap_mix:
        _extra += " trap_mix=True"
    if HP.ch_loss:
        _extra += " ch_loss=True"
    if HP.composite_embed:
        _extra += " composite_embed=True"
    if HP.sphere_retract:
        _extra += " sphere_retract=True"
    _sched = f"lr_schedule={HP.lr_schedule}"
    if HP.lr_schedule == "wsd":
        _sched += f" decay_frac={HP.wsd_decay_frac}"
    _eff_bs = HP.batch_size * HP.grad_accum
    _tok_per_step = _eff_bs * HP.seq_len
    _warmup_tok_per_step = HP.batch_size * HP.seq_len
    print0(rank, f"model_type={HP.model_type} lm_head={_head} layers={HP.n_layer} heads={HP.n_head} d_model={HP.d_model} seq_len={HP.seq_len}{_extra} {_sched}")
    print0(rank, f"batch_size={HP.batch_size} grad_accum={HP.grad_accum} effective_batch={_eff_bs} tok/step(stable)={_tok_per_step:,} tok/step(warmup)={_warmup_tok_per_step:,} vocab={HP.vocab_size} data_format={HP.data_format}")
    if HP.llada:
        _llada_feats = f"llada=True subs={HP.llada_subs} antithetic={HP.llada_antithetic} bidirectional=True"
        print0(rank, _llada_feats)

    if HP.data_format == "yamit":
        train_stream = MixedShardStream(HP.train_files, rank, world_size, HP.seq_len, HP.batch_size, seed=HP.seed + 1)
        val_stream = MixedShardStream(HP.val_files, rank, world_size, HP.seq_len, HP.batch_size, seed=HP.seed + 2)
        print0(rank, f"train: {train_stream.source_summary()}")
        print0(rank, f"val:   {val_stream.source_summary()}")
    else:
        train_stream = ShardStream(HP.train_files, rank, world_size, HP.seq_len, HP.batch_size)
        val_stream = ShardStream(HP.val_files, rank, world_size, HP.seq_len, HP.batch_size)
        print0(rank, f"train_shards={len(train_stream.files)} val_shards={len(val_stream.files)}")

    _pt_dtype = torch.bfloat16 if HP.dtype == "bf16" else torch.float32
    model = build_model_maybe_llada().to(device=device, dtype=_pt_dtype)

    # Optional weight initialization normalization.
    if HP.init_mode == "sphere":
        _init_count = 0
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.ndim >= 2:
                    p.div_(p.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                    _init_count += 1
        print0(rank, f"init_mode=sphere: normalized {_init_count} weight matrices to unit row norm")
    elif HP.init_mode == "ns5":
        from autonormuon import zeropower_via_newtonschulz5
        _init_count = 0
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.ndim >= 2:
                    p.copy_(zeropower_via_newtonschulz5(p.float(), steps=5).to(p.dtype))
                    _init_count += 1
        print0(rank, f"init_mode=ns5: orthogonalized {_init_count} weight matrices via Newton-Schulz")

    n_params = sum(p.numel() for p in model.parameters())
    _mem_per_param = 2 if HP.dtype == "bf16" else 4
    _weight_mb = n_params * _mem_per_param / 1024 / 1024
    print0(rank, f"parameters={n_params:,} dtype={HP.dtype} weight_mem={_weight_mb:.0f}MB")

    # Build optimizer param groups before compile/DDP wrap
    _is_ngpt = HP.ngpt or HP.model_type == "ngpt_moe"
    _wd = 0.0 if _is_ngpt else HP.weight_decay
    _muon_like_opts = ("muon", "normuon", "autonormuon", "orion", "muon_sf", "normuon_sf", "spicydion")
    _n_residuals = HP.n_layer * 2
    _muon_formula_lr = 0.1 / math.sqrt(_n_residuals)
    # Names that identify embedding/head params (should not get Muon-style whitening)
    _embed_head_names = {"wte", "lm_head", "embed", "query_bank", "queries", "memory", "byte_memory", "token_embed", "token_params", "byte_embed"}

    def _is_embed_head(name: str) -> bool:
        return any(k in name for k in _embed_head_names)

    if HP.optimizer in ("dion", "dion2"):
        # Dion/Dion2: param groups with 'algorithm' key; lazy import (CUDA-only)
        _dion_params = []
        _adamw_embed = []
        _adamw_head = []
        _adamw_other = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or getattr(p, "_no_weight_decay", False):
                _adamw_other.append(p)
            elif "lm_head" in name:
                _adamw_head.append(p)
            elif _is_embed_head(name):
                _adamw_embed.append(p)
            else:
                _dion_params.append(p)
        _head_lr = HP.muon_lr / math.sqrt(HP.d_model)
        print0(rank, f"optimizer: {HP.optimizer} ({len(_dion_params)} ortho, {len(_adamw_embed)} embed, {len(_adamw_head)} head, {len(_adamw_other)} other, wd={_wd})")
        param_groups = [
            dict(params=_dion_params),
            dict(params=_adamw_embed, algorithm="adamw"),
            dict(params=_adamw_head, algorithm="adamw", lr=_head_lr),
            dict(params=_adamw_other, algorithm="adamw", weight_decay=0.0),
        ]
        param_groups = [g for g in param_groups if g["params"]]
        # Base LRs: ortho groups use muon_lr (dion scales internally), adamw use lr, head uses scaled lr
        _base_lrs = []
        for g in param_groups:
            if g.get("algorithm") == "adamw":
                _base_lrs.append(g["lr"] if "lr" in g else HP.lr)
            else:
                _base_lrs.append(HP.muon_lr)
        decay_params = _dion_params + _adamw_embed + _adamw_head
        no_decay_params = _adamw_other

    elif HP.optimizer in _muon_like_opts:
        # Shared param split for Muon and NorMuon (both use Muon+AuxAdam pattern)
        _muon_params = []
        adam_decay_params = []
        adam_no_decay_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or getattr(p, "_no_weight_decay", False):
                adam_no_decay_params.append(p)
            elif _is_embed_head(name):
                adam_decay_params.append(p)
            else:
                _muon_params.append(p)
        _opt_name = HP.optimizer
        _base_lr_muon_like = HP.muon_lr if HP.muon_lr > 0 else _muon_formula_lr
        _lr_source = "override" if HP.muon_lr > 0 else f"auto(0.1/sqrt({_n_residuals}))"
        print0(rank, f"optimizer: {_opt_name} ({len(_muon_params)} muon, {len(adam_decay_params)} adam-decay, {len(adam_no_decay_params)} adam-nodecay, wd={_wd}, muon_lr={_base_lr_muon_like:.6f} [{_lr_source}])")
        _muon_group = dict(params=_muon_params, lr=_base_lr_muon_like, momentum=HP.muon_momentum, weight_decay=_wd, use_muon=True)
        if HP.optimizer in ("normuon", "autonormuon", "normuon_sf"):
            _muon_group["beta2"] = HP.normuon_beta2
        param_groups = [
            _muon_group,
            dict(params=adam_decay_params, lr=_base_lr_muon_like, betas=(0.9, 0.95), eps=1e-10, weight_decay=_wd, use_muon=False),
            dict(params=adam_no_decay_params, lr=_base_lr_muon_like, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0, use_muon=False),
        ]
        param_groups = [g for g in param_groups if g["params"]]
        _base_lrs = [g["lr"] for g in param_groups]
        decay_params = _muon_params + adam_decay_params
        no_decay_params = adam_no_decay_params

    elif HP.optimizer in ("geoadam", "geomuon", "geonormuon"):
        # Geodesic optimizers: 2D weights are spherical when running nGPT
        all_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        spherical_ids: set[int] = set()
        if _is_ngpt:
            for name, p in all_params:
                if p.ndim >= 2:
                    spherical_ids.add(id(p))
        _n_sph = len(spherical_ids)
        _n_euc = len(all_params) - _n_sph
        decay_params = [p for _, p in all_params if p.ndim > 1 and not getattr(p, "_no_weight_decay", False)]
        no_decay_params = [p for _, p in all_params if p.ndim <= 1 or getattr(p, "_no_weight_decay", False)]

        if HP.optimizer == "geoadam":
            from geoadam import GeodesicAdam
            print0(rank, f"optimizer: geoadam ({_n_sph} spherical, {_n_euc} euclidean, wd={_wd})")
            param_groups = [{"params": [p for _, p in all_params]}]
            _base_lrs = [HP.lr]
        elif HP.optimizer == "geomuon":
            from geomuon import GeodesicMuon
            print0(rank, f"optimizer: geomuon ({_n_sph} spherical, {_n_euc} euclidean, wd={_wd}, ns_steps={HP.geomuon_ns_steps})")
            param_groups = [{"params": [p for _, p in all_params]}]
            _base_lrs = [HP.muon_lr if _is_ngpt else HP.lr]
        else:  # geonormuon
            from geonormuon import GeodesicNorMuon
            print0(rank, f"optimizer: geonormuon ({_n_sph} spherical, {_n_euc} euclidean, wd={_wd}, ns_steps={HP.geomuon_ns_steps}, beta2={HP.normuon_beta2})")
            param_groups = [{"params": [p for _, p in all_params]}]
            _base_lrs = [HP.muon_lr if _is_ngpt else HP.lr]

    else:  # adamw (default)
        decay_params = []
        no_decay_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or getattr(p, "_no_weight_decay", False):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        print0(rank, f"optimizer: adamw ({len(decay_params)} decay, {len(no_decay_params)} nodecay, wd={_wd})")
        param_groups = [
            {"params": decay_params, "weight_decay": _wd},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        _base_lrs = [HP.lr] * len(param_groups)

    raw_model = model  # keep ref before compile/DDP wrapping
    if HP.sphere_retract:
        _sphere_retract_weights(raw_model)
        print0(rank, "sphere_retract: initial weight retraction applied")
    if HP.compile:
        model = torch.compile(model, dynamic=False)
    if world_size > 1:
        model = DDP(model, device_ids=[device.index])

    if HP.optimizer == "dion":
        from dion import Dion
        _dion_pg = model.process_group if hasattr(model, 'process_group') else None
        optimizer = Dion(param_groups, lr=HP.muon_lr, weight_decay=_wd,
                         **(dict(distributed_mesh=_dion_pg) if _dion_pg else {}))
    elif HP.optimizer == "dion2":
        from dion import Dion2
        _dion_pg = model.process_group if hasattr(model, 'process_group') else None
        optimizer = Dion2(param_groups, lr=HP.muon_lr, weight_decay=_wd,
                          **(dict(distributed_mesh=_dion_pg) if _dion_pg else {}))
    elif HP.optimizer == "muon":
        from muon import SingleDeviceMuonWithAuxAdam
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    elif HP.optimizer == "orion":
        from orion import SingleDeviceOrionWithAuxAdam
        optimizer = SingleDeviceOrionWithAuxAdam(param_groups)
    elif HP.optimizer == "normuon":
        from normuon import SingleDeviceNorMuonWithAuxAdam
        optimizer = SingleDeviceNorMuonWithAuxAdam(param_groups)
    elif HP.optimizer == "autonormuon":
        from autonormuon import AutoNorMuon
        # LR already set by formula for all muon-like optimizers: 0.1 / sqrt(R), R = 2 * n_layer
        # nGPT does its own retraction, so force off; otherwise use config
        _grad_sched = "off" if _is_ngpt else HP.autonormuon_grad_schedule
        _weight_sched = "off" if _is_ngpt else HP.autonormuon_weight_schedule
        _weight_mode = "sphere" if _is_ngpt else HP.autonormuon_weight_mode
        optimizer = AutoNorMuon(
            param_groups,
            total_steps=HP.train_steps,
            beta=HP.autonormuon_beta,
            adaptation_scope=HP.autonormuon_adaptation_scope,
            grad_schedule=_grad_sched,
            weight_schedule=_weight_sched,
            weight_mode=_weight_mode,
            gnorm_mode=HP.autonormuon_gnorm_mode,
            ratio_pow=HP.autonormuon_ratio_pow,
            min_ratio=HP.autonormuon_min_ratio,
        )
        print0(rank, (
            f"  autonormuon: R={_n_residuals} auto_lr={_muon_formula_lr:.6f} "
            f"grad_schedule={_grad_sched} weight_schedule={_weight_sched} "
            f"weight_mode={_weight_mode} gnorm_mode={HP.autonormuon_gnorm_mode} "
            f"beta={HP.autonormuon_beta} adaptation_scope={HP.autonormuon_adaptation_scope} "
            f"ratio_pow={HP.autonormuon_ratio_pow} min_ratio={HP.autonormuon_min_ratio}"
        ))
    elif HP.optimizer == "spicydion":
        from spicydion import SpicyDion
        # SpicyDion: PolarExpress + AOL + NorMuon-EMA + pressure-based LR + sphere retract.
        # No external LR scheduling, no weight decay. Muon-like + Adam-like params via algorithm tags.
        spicy_groups = []
        for g in param_groups:
            if g.get("use_muon", False):
                spicy_groups.append({
                    "params": g["params"],
                    "lr": g["lr"],
                    "ef_decay": HP.spicydion_ef_decay,
                    "fraction": 1.0,
                    "weight_decay": 0.0,
                    "epsilon": 1e-8,
                    "adjust_lr": None,
                    "flatten": False,
                    "algorithm": "spicydion",
                    "step": 0,
                })
            else:
                spicy_groups.append({
                    "params": g["params"],
                    "lr": g["lr"],
                    "beta1": 0.9,
                    "beta2": 0.95,
                    "weight_decay": 0.0,
                    "epsilon": 1e-10,
                    "algorithm": "adamw",
                    "step": 0,
                })

        optimizer = SpicyDion(
            spicy_groups,
            distributed_mesh=None,
            lr=_base_lrs[0],
            fraction=1.0,
            ef_decay=HP.spicydion_ef_decay,
            weight_decay=0.0,
            adjust_lr=None,
            gnorm_beta=HP.autonormuon_beta,
            total_steps=HP.train_steps,
            adaptive_lr_mode=HP.spicydion_adaptive_lr_mode,
        )
        print0(rank, (
            f"  spicydion: R={_n_residuals} base_lr={_base_lrs[0]:.6f} "
            f"ef_decay={HP.spicydion_ef_decay} gnorm_beta={HP.autonormuon_beta} "
            f"total_steps={HP.train_steps} adaptive_lr={HP.spicydion_adaptive_lr_mode} wd=0 "
            f"[PolarExpress+AOL, NorMuon-EMA, sphere-retract]"
        ))
    elif HP.optimizer == "muon_sf":
        from muon_sf import ScheduleFreeMuon
        _sf_warmup = 0
        optimizer = ScheduleFreeMuon(param_groups, warmup_steps=_sf_warmup)
        print0(rank, f"  schedule-free: warmup_steps={_sf_warmup} sf_momentum=0.9")
    elif HP.optimizer == "normuon_sf":
        from normuon_sf import ScheduleFreeNorMuon
        _sf_warmup = 0
        optimizer = ScheduleFreeNorMuon(param_groups, warmup_steps=_sf_warmup)
        print0(rank, f"  schedule-free: warmup_steps={_sf_warmup} sf_momentum=0.9")
    elif HP.optimizer == "geoadam":
        optimizer = GeodesicAdam(
            param_groups[0]["params"], lr=HP.lr, betas=(0.9, 0.95),
            spherical_params=spherical_ids, normalize_dim=1,
            weight_decay=_wd, adam_lr=HP.lr,
        )
        param_groups = optimizer.param_groups  # re-bind for LR scheduling
    elif HP.optimizer == "geomuon":
        optimizer = GeodesicMuon(
            param_groups[0]["params"], lr=_base_lrs[0],
            momentum=HP.muon_momentum, ns_steps=HP.geomuon_ns_steps,
            betas=(0.9, 0.95), weight_decay=_wd,
            spherical_params=spherical_ids, normalize_dim=1,
            adam_lr=HP.lr,
        )
        param_groups = optimizer.param_groups
    elif HP.optimizer == "geonormuon":
        optimizer = GeodesicNorMuon(
            param_groups[0]["params"], lr=_base_lrs[0],
            momentum=HP.muon_momentum, beta2=HP.normuon_beta2,
            ns_steps=HP.geomuon_ns_steps,
            betas=(0.9, 0.95), weight_decay=_wd,
            spherical_params=spherical_ids, normalize_dim=1,
            adam_lr=HP.lr,
        )
        param_groups = optimizer.param_groups
    else:
        _fused = device.type == "cuda"
        optimizer = torch.optim.AdamW(param_groups, lr=HP.lr, betas=(0.9, 0.95), fused=_fused)

    assert len(_base_lrs) == len(optimizer.param_groups), \
        f"_base_lrs length {len(_base_lrs)} != optimizer.param_groups length {len(optimizer.param_groups)} — " \
        f"optimizer may have restructured param groups internally"

    _is_sf = HP.optimizer.endswith("_sf")  # schedule-free: optimizer handles its own LR warmup/schedule
    _is_auto = HP.optimizer in ("autonormuon",)  # handle own LR schedule internally
    _is_spicydion = HP.optimizer == "spicydion"  # self-scheduling: handles own LR warmup/cruise/cooldown

    _gnorm_scheduler = None
    if HP.lr_schedule == "gnorm":
        from gnorm_scheduler import GnormScheduler
        # Use the actual main-group LR (muon/dion LR), not the default HP.lr (Adam LR)
        _main_lr = next(
            (pg["lr"] for pg in optimizer.param_groups
             if pg.get("use_muon", False) or pg.get("algorithm") == "spicydion"),
            HP.lr,  # fallback
        )
        _gnorm_scheduler = GnormScheduler(base_lr=_main_lr, n_layers=HP.n_layer, schedule_power=HP.schedule_power)
        _gs = _gnorm_scheduler
        print0(rank, f"  gnorm_scheduler: base_lr={_main_lr:.4e} cold_lr={_main_lr*0.1:.4e} n_layers={HP.n_layer} ma_win={_gs._ma_buf.maxlen} var_win={_gs._var_buf.maxlen} schedule_power={HP.schedule_power} step_down=5%")

    profiler = None
    if HP.torch_profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        )
        profiler.start()
        print0(rank, f"torch profiler enabled for {HP.torch_profile_steps} train steps")

    # Standard CE reference: extract raw embed weight for cross-model comparison


    # Contraharmonic loss state
    _ch_on = HP.ch_loss and isinstance(train_stream, MixedShardStream)
    if _ch_on:
        _ch_n_src = train_stream.n_sources
        _ch_source_losses = torch.ones(_ch_n_src, device=device)  # init uniform
        _ch_weights = _ch_source_losses / _ch_source_losses.sum()  # init equal weights
        print0(rank, f"ch_loss=True sources={_ch_n_src} ({', '.join(train_stream.source_names)})")

    # Token-based schedule: total tokens budget is fixed regardless of grad_accum changes
    _tok_per_micro = HP.batch_size * HP.seq_len
    _total_tokens = HP.train_steps * HP.batch_size * HP.grad_accum * HP.seq_len
    _warmup_tokens = int(_total_tokens * HP.warmup_frac)
    _warmup_ramp_start = _warmup_tokens // 4
    print0(rank, f"total_tokens={_total_tokens:,} warmup_tokens={_warmup_tokens:,} ({HP.warmup_frac*100:.1f}%)")

    _metrics = None
    if HP.metrics_enabled:
        _hparams = {f.name: getattr(HP, f.name) for f in HP.__dataclass_fields__.values()}
        _run_name = HP.metrics_run_name.strip()
        if not _run_name:
            _stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            _run_name = f"{_stamp}_{HP.model_type}_{HP.optimizer}_L{HP.n_layer}_D{HP.d_model}_T{HP.seq_len}"
        _metrics_cfg = {
            "hparams": _hparams,
            "derived": {
                "rank": rank,
                "world_size": world_size,
                "device": str(device),
                "hostname": os.uname().nodename,
                "total_tokens": _total_tokens,
                "warmup_tokens": _warmup_tokens,
                "warmup_ramp_start_tokens": _warmup_ramp_start,
                "tokens_per_microbatch": _tok_per_micro,
                "muon_formula_lr": _muon_formula_lr,
                "n_residuals": _n_residuals,
                "n_params": n_params,
                "weight_mem_mb": _weight_mb,
                "base_lrs": _base_lrs,
                "ckpt_enabled": bool(HP.ckpt_dir),
                "val_every": HP.val_every,
                "val_steps": HP.val_steps,
            },
        }
        _metrics = _open_metrics_run(rank, HP.metrics_dir, _run_name, _metrics_cfg)
        if _metrics is not None:
            print0(rank, f"metrics: writing JSON artifacts to {_metrics['run_dir']}")

    # Checkpoint saving
    _ckpt_on = bool(HP.ckpt_dir) and rank == 0
    _best_val_loss = float("inf")
    if _ckpt_on:
        os.makedirs(HP.ckpt_dir, exist_ok=True)
        print0(rank, f"checkpoints → {HP.ckpt_dir}")

    t0 = time.time()
    _run_start_time = t0
    _last_step_wall = t0
    tokens_seen = 0
    step = 0
    while tokens_seen <= _total_tokens:
        if step % HP.val_every == 0 or tokens_seen >= _total_tokens:
            if _is_sf:
                optimizer.eval()  # switch params from y to x for evaluation
            _eval_ch_w = _ch_weights if _ch_on else None
            val_loss, val_acc, raw_val_loss, val_per_src = evaluate(model, val_stream, device, world_size, ch_weights=_eval_ch_w)
            _acc_str = f" | val_acc {val_acc:.4f}" if val_acc is not None else ""
            _raw_val_str = f" | raw_loss {raw_val_loss:.5f}" if raw_val_loss is not None else ""
            print0(rank, f"step {step:5d} | val_loss {val_loss:.5f}{_raw_val_str}{_acc_str} | tokens {tokens_seen:,}/{_total_tokens:,}")
            if val_per_src is not None:
                _vps_parts = [f"{n}={v:.4f}" for n, v in val_per_src.items()]
                print0(rank, f"  val_per_source: {' '.join(_vps_parts)}")

            if _metrics is not None:
                _eval_event = {
                    "event": "eval",
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "total_tokens": _total_tokens,
                    "timestamp": time.time(),
                    "elapsed_s": time.time() - _run_start_time,
                    "val_loss": val_loss,
                    "raw_val_loss": raw_val_loss,
                    "val_acc": val_acc,
                    "val_per_source": val_per_src,
                    "optimizer_groups": _optimizer_group_metrics(optimizer),
                }
                _write_metrics_event(_metrics, "eval", _eval_event, HP.metrics_flush_every)
                if val_loss < _metrics["best_val_loss"]:
                    _metrics["best_val_loss"] = float(val_loss)
                    _metrics["best_step"] = int(step)
                    _metrics["best_tokens_seen"] = int(tokens_seen)

            # Save checkpoints (schedule-free already in eval mode — params hold x)
            if _ckpt_on:
                _ckpt = {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "val_loss": val_loss,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "hparams": {f.name: getattr(HP, f.name) for f in HP.__dataclass_fields__.values()},
                }
                torch.save(_ckpt, os.path.join(HP.ckpt_dir, "last.pt"))
                if val_loss < _best_val_loss:
                    _best_val_loss = val_loss
                    torch.save(_ckpt, os.path.join(HP.ckpt_dir, "best.pt"))
                    print0(rank, f"  saved best checkpoint (val_loss={val_loss:.5f})")
            if tokens_seen >= _total_tokens:
                break  # stay in eval mode (params = x, the averaged solution)
            if _is_sf:
                optimizer.train()  # switch params back from x to y for training

        # During warmup: ramp grad_accum from 1 -> HP.grad_accum.
        # When GnormScheduler is active, tie accum ramp to scheduler phase:
        #   cold  → accum=1, ramp → linear 1→full, cruise → full.
        # Otherwise use the token-based ramp (first 25% of warmup at accum=1,
        # then linear ramp over remaining 75%).
        if _gnorm_scheduler is not None and HP.grad_accum > 1:
            _ramp_t = _gnorm_scheduler._ramp_progress
            _in_warmup = _gnorm_scheduler.phase in ("cold", "ramp")
            _in_accum_ramp = _gnorm_scheduler.phase == "ramp"
            _eff_accum = 1 + int(round(_ramp_t * (HP.grad_accum - 1)))
            _eff_accum = max(1, min(HP.grad_accum, _eff_accum))
        else:
            _in_warmup = tokens_seen < _warmup_tokens
            _in_accum_ramp = _in_warmup and HP.grad_accum > 1 and tokens_seen >= _warmup_ramp_start
            if _in_warmup:
                if _in_accum_ramp:
                    _ramp_span = max(1, _warmup_tokens - _warmup_ramp_start)
                    _ramp_t = (tokens_seen - _warmup_ramp_start) / _ramp_span
                    _eff_accum = 1 + int(round(_ramp_t * (HP.grad_accum - 1)))
                    _eff_accum = max(1, min(HP.grad_accum, _eff_accum))
                else:
                    _eff_accum = 1
            else:
                _eff_accum = HP.grad_accum

        if _is_sf or _is_auto or _is_spicydion or _gnorm_scheduler is not None:
            # Schedule-free / self-scheduling optimizers / GnormScheduler: skip external LR schedule
            lr = optimizer.param_groups[0].get("scheduled_lr", optimizer.param_groups[0]["lr"])
        else:
            _lr_factor = lr_for_tokens(tokens_seen, _total_tokens) / max(HP.lr, 1e-12)
            for pg, base_lr in zip(optimizer.param_groups, _base_lrs):
                pg["lr"] = base_lr * _lr_factor
                if "adam_lr" in pg:
                    pg["adam_lr"] = HP.lr * _lr_factor
            lr = _base_lrs[0] * _lr_factor  # for logging

        optimizer.zero_grad(set_to_none=True)
        # Per-source loss accumulators for contraharmonic weighting
        # CH_LOSS only active after warmup
        _ch_active = _ch_on and not _in_warmup
        if _ch_active:
            _ch_src_loss_sum = torch.zeros(_ch_n_src, device=device)
            _ch_src_count = torch.zeros(_ch_n_src, device=device)
            _ch_std_loss_sum = 0.0  # unweighted loss accumulator
        for _micro in range(_eff_accum):
            _tbatch = train_stream.next_batch(device)
            x, y = _tbatch[0], _tbatch[1]
            src_ids = _tbatch[2] if len(_tbatch) > 2 else None
            if HP.llada:
                mask, p_mask = llada_make_mask(x, device)
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _, loss = model(x, mask, p_mask)
                else:
                    _, loss = model(x, mask, p_mask)
            else:
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                else:
                    logits, loss = model(x, y)

                # Contraharmonic loss: reweight per-sequence losses by source
                if _ch_active and logits is not None and src_ids is not None:
                    B, T, V = logits.shape
                    per_tok = F.cross_entropy(logits.view(-1, V), y.view(-1), reduction='none').view(B, T)
                    per_seq = per_tok.mean(dim=1)  # (B,)
                    # Accumulate per-source losses (detached) for next step's weights
                    for _si in range(_ch_n_src):
                        _mask = src_ids == _si
                        if _mask.any():
                            _ch_src_loss_sum[_si] += per_seq[_mask].detach().sum()
                            _ch_src_count[_si] += _mask.sum()
                    # Track unweighted (standard) loss for logging
                    _ch_std_loss_sum += per_seq.detach().mean().item()
                    # Apply contraharmonic weights from previous step
                    seq_w = _ch_weights[src_ids]  # (B,)
                    loss = (per_seq * seq_w).sum() / seq_w.sum()
            loss = loss / _eff_accum
            loss.backward()
            tokens_seen += _tok_per_micro
        # Update contraharmonic weights for next step
        if _ch_active:
            for _si in range(_ch_n_src):
                if _ch_src_count[_si] > 0:
                    _ch_source_losses[_si] = _ch_src_loss_sum[_si] / _ch_src_count[_si]
            # Contraharmonic weight: w_i = L_i^2 / sum(L_j^2), floored at 1/n_sources
            _ch_sq = _ch_source_losses ** 2
            _ch_weights = _ch_sq / _ch_sq.sum().clamp(min=1e-8)
            _ch_floor = 1.0 / _ch_n_src
            _ch_weights = _ch_weights.clamp(min=_ch_floor)
            _ch_weights = _ch_weights / _ch_weights.sum()
        _all_params = decay_params + no_decay_params
        if HP.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(_all_params, HP.grad_clip)
        else:
            grad_norm = torch.cat([p.grad.flatten() for p in _all_params if p.grad is not None]).norm()
        # Feed loss to optimizer for anneal gate (no-op if optimizer doesn't support it)
        if hasattr(optimizer, "set_train_loss"):
            optimizer.set_train_loss(float(loss.detach().item()) * _eff_accum)
        # Gnorm scheduler: set LR and adaptive_active on param groups before step.
        if _gnorm_scheduler is not None:
            _sched = _gnorm_scheduler.step(float(grad_norm), float(loss.detach().item()) * _eff_accum, step)
            _sched_lr = _sched["lr"]
            _sched_aa = 1.0 if _sched["adaptive_active"] else 0.0
            for pg in optimizer.param_groups:
                # Muon/Dion/SpicyDion groups get full scheduled LR;
                # Adam/auxiliary groups get 1/3.
                _is_main = (pg.get("algorithm") == "spicydion"
                            or pg.get("use_muon", False)
                            or pg.get("algorithm") in ("muon", "normuon", "autonormuon"))
                _pg_lr = _sched_lr if _is_main else _sched_lr / 3.0
                pg["lr"] = _pg_lr
                pg["scheduled_lr"] = _pg_lr
                if _is_spicydion:
                    pg["_adaptive_active"] = _sched_aa
            if _sched["early_exit"]:
                print0(rank, f"  GnormScheduler: early exit at step {step} — LR below floor ({_sched_lr:.2e}), tap_count={_sched['tap_count']}")
                break
        optimizer.step()
        if _is_sf or _is_auto:
            lr = optimizer.param_groups[0].get("scheduled_lr", lr)  # read actual LR after step
        if _is_ngpt:
            _ngpt_normalize_weights(raw_model)
            _ngpt_normalize_pit_memory(raw_model)
        if HP.sphere_retract:
            _sphere_retract_weights(raw_model)

        _train_loss_local = float((loss.detach() * _eff_accum).item())
        _train_acc_local = None
        if hasattr(raw_model, '_last_acc') and raw_model._last_acc is not None:
            _train_acc_local = _to_json_scalar(raw_model._last_acc)
        if _gnorm_scheduler is not None:
            _tc = _gnorm_scheduler.tap_count
            _phase = f"{_gnorm_scheduler.phase}(t={_tc})" if _tc > 0 else _gnorm_scheduler.phase
        else:
            _phase = "warmup-ramp" if _in_warmup and _in_accum_ramp else ("warmup" if _in_warmup else "stable")

        if _metrics is not None and step % max(1, HP.metrics_every) == 0:
            _now = time.time()
            _elapsed = _now - _run_start_time
            _step_wall = _now - _last_step_wall
            _tokens_per_sec = tokens_seen / max(_elapsed, 1e-12)
            _train_event = {
                "event": "train",
                "step": step,
                "tokens_seen": tokens_seen,
                "total_tokens": _total_tokens,
                "timestamp": _now,
                "elapsed_s": _elapsed,
                "step_wall_s": _step_wall,
                "throughput_tok_s": _tokens_per_sec,
                "phase": _phase,
                "in_warmup": _in_warmup,
                "in_accum_ramp": _in_accum_ramp,
                "eff_accum": _eff_accum,
                "lr": _to_json_scalar(lr),
                "grad_norm": _to_json_scalar(grad_norm),
                "train_loss": _train_loss_local,
                "train_acc": _train_acc_local,
                "raw_ch_loss": (_ch_std_loss_sum / max(1, _eff_accum)) if _ch_active else None,
                "ch_source_losses": _ch_source_losses.detach().float().cpu().tolist() if _ch_active else None,
                "ch_weights": _ch_weights.detach().float().cpu().tolist() if _ch_active else None,
                "optimizer_groups": _optimizer_group_metrics(optimizer),
            }
            if _gnorm_scheduler is not None:
                _train_event["loss_ratio"] = _gnorm_scheduler.loss_ratio
                _train_event["deviation"] = _gnorm_scheduler.deviation
                _train_event["ma_variance"] = _gnorm_scheduler.current_variance
                _train_event["tap_count"] = _gnorm_scheduler.tap_count
            _write_metrics_event(_metrics, "train", _train_event, HP.metrics_flush_every)

        _last_step_wall = time.time()

        if step % 20 == 0:
            loss_t = loss.detach() * _eff_accum
            if world_size > 1:
                dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
            dt = (time.time() - t0) / max(1, step + 1)
            _train_acc_str = ""
            if hasattr(raw_model, '_last_acc') and raw_model._last_acc is not None:
                _train_acc_str = f" | train_acc {float(raw_model._last_acc):.4f}"
            _ch_raw_str = ""
            if _ch_active:
                _ch_raw_str = f" | raw_loss {_ch_std_loss_sum / max(1, _eff_accum):.5f}"
            _adapt_lr_str = ""
            if _is_spicydion:
                _adapt_lr = optimizer.param_groups[0].get("_diag_median_adaptive_lr")
                if _adapt_lr is not None:
                    _adapt_lr_str = f" | adapt_lr {_adapt_lr:.3e}"
            _lr_str = ""
            if _gnorm_scheduler is not None:
                _lr_str = f" | dev={_gnorm_scheduler.deviation:.3f}"
            print0(rank, f"step {step:5d} | train_loss {loss_t.item():.5f}{_ch_raw_str}{_train_acc_str} | lr {lr:.3e}{_adapt_lr_str} | gnorm {grad_norm:.3f} | sec/step {dt:.3f} | {_phase}{_lr_str} acc={_eff_accum}")
            if _ch_active and step % 100 == 0:
                _ch_parts = [f"{n}={_ch_source_losses[i]:.2f}({_ch_weights[i]:.3f})" for i, n in enumerate(train_stream.source_names)]
                print0(rank, f"  ch_weights: {' '.join(_ch_parts)}")

        step += 1

        if profiler is not None:
            profiler.step()
            if step + 1 >= HP.torch_profile_steps:
                profiler.stop()
                if rank == 0:
                    print("\n=== Torch Profile: CUDA Time ===", flush=True)
                    print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=40), flush=True)
                    print("\n=== Torch Profile: CUDA Memory ===", flush=True)
                    print(profiler.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=40), flush=True)
                profiler = None

    if _metrics is not None:
        _metrics["train_file"].flush()
        _metrics["eval_file"].flush()
        _summary = {
            "run_dir": str(_metrics["run_dir"]),
            "config_path": str(_metrics["config_path"]),
            "train_events_path": str(_metrics["train_path"]),
            "eval_events_path": str(_metrics["eval_path"]),
            "started_at": datetime.fromtimestamp(_run_start_time).isoformat(),
            "finished_at": datetime.now().isoformat(),
            "elapsed_s": time.time() - _run_start_time,
            "tokens_seen": int(tokens_seen),
            "total_tokens": int(_total_tokens),
            "steps_completed": int(step),
            "train_event_count": int(_metrics["train_count"]),
            "eval_event_count": int(_metrics["eval_count"]),
            "best_val_loss": _metrics["best_val_loss"],
            "best_step": _metrics["best_step"],
            "best_tokens_seen": _metrics["best_tokens_seen"],
            "last_train": _metrics["last_train"],
            "last_eval": _metrics["last_eval"],
        }
        with _metrics["summary_path"].open("w", encoding="utf-8") as f:
            json.dump(_to_json_scalar(_summary), f, indent=2)
        _metrics["train_file"].close()
        _metrics["eval_file"].close()
        print0(rank, f"metrics summary: {_metrics['summary_path']}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
