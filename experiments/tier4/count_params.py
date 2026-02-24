#!/usr/bin/env python3
"""Count total and active-per-token parameters for the hybrid nGPT MoE architecture.

Usage:
  Same env vars as train_fineweb_vanilla.py, e.g.:
  MODEL_TYPE=ngpt_moe NGPT=1 N_LAYER=13 D_MODEL=768 N_HEAD=12 N_EXPERTS=5 TOP_K=2 \
  MOE_DENSE_LAYERS=2 TRAP_MIX=1 NGPT_DIFF_ATTN_N=2 NGPT_PAIRED_ODD=1 \
  NGPT_SKIP_TRAP_EVEN=1 NGPT_WINDOW_LAYERS=5,7 NGPT_WINDOW_SIZE=512 \
  NGPT_EMBED_GATE_LAYER=6 COMPOSITE_EMBED=1 LM_HEAD_TYPE=pit \
  COMPOSITE_TOKEN_DIMS=24 VOCAB_SIZE=149760 DATA_FORMAT=yamit \
  TOKEN_BYTES_PATH=experiments/yamit/tokenizer/artifacts/yamit/token_bytes.npy \
  YAMIT_EOS_TOKEN_ID=149727 \
  python experiments/tier4/count_params.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import train_fineweb_vanilla as T

HP = T.HP

def main():
    model = T.GPTNGPTMoE()

    # Unique total params (deduplicated by id for weight tying)
    seen = set()
    total_unique = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total_unique += p.numel()

    print(f"\n{'='*70}")
    print(f"PARAMETER COUNT — {HP.model_type} d={HP.d_model} L={HP.n_layer} H={HP.n_head}")
    print(f"{'='*70}\n")

    # Embed + head
    embed_seen = set()
    embed_p = 0
    for p in model.wte.parameters():
        if id(p) not in embed_seen:
            embed_seen.add(id(p))
            embed_p += p.numel()
    head_seen = set()
    head_p = 0
    for p in model.lm_head.parameters():
        if id(p) not in head_seen:
            head_seen.add(id(p))
            head_p += p.numel()
    # Deduplicate across embed/head (weight tying)
    all_embed_head_ids = embed_seen | head_seen
    embed_head_unique = 0
    eh_seen = set()
    for p in list(model.wte.parameters()) + list(model.lm_head.parameters()):
        if id(p) not in eh_seen:
            eh_seen.add(id(p))
            embed_head_unique += p.numel()
    s_z_p = model.s_z.numel()

    print(f"  Embedding:     {embed_p:>12,}")
    print(f"  LM Head:       {head_p:>12,}")
    print(f"  Embed+Head:    {embed_head_unique:>12,}  (deduplicated)")
    print(f"  s_z:           {s_z_p:>12,}")
    print()

    # Per-block breakdown
    total_block_params = 0
    total_block_active = 0

    for i, blk in enumerate(model.blocks):
        blk_seen = set()
        blk_total = 0
        for p in blk.parameters():
            if id(p) not in blk_seen:
                blk_seen.add(id(p))
                blk_total += p.numel()

        # Attention: always fully active
        attn_total = sum(p.numel() for p in blk.attn.parameters())

        # Alphas: always active
        alpha_total = blk.alpha_attn.numel() + blk.alpha_mlp.numel()
        if blk.embed_gate:
            alpha_total += blk.alpha_gate.numel()

        # Embed gate proj: always active
        egate_total = sum(p.numel() for p in blk.gate_proj.parameters()) if blk.embed_gate else 0

        if isinstance(blk, T.NGPTMoEBlock):
            moe = blk.mlp
            # Shared expert (always active)
            shared_p = 0
            for n, p in moe.named_parameters():
                if 'shared' in n:
                    shared_p += p.numel()
            # Router (always active)
            router_p = moe.router.weight.numel()
            if hasattr(moe, 'router_bias'):
                router_p += moe.router_bias.numel()
            # Routed experts (total vs active)
            routed_total = moe.gate_up_proj.numel() + moe.down_proj.numel()
            routed_active = routed_total * HP.top_k // HP.n_experts

            blk_active = attn_total + alpha_total + egate_total + shared_p + router_p + routed_active

            feats = []
            if blk.attn.differential: feats.append("diff")
            if blk.attn.paired: feats.append("pha")
            if blk.attn.trap_mix: feats.append("trap")
            if blk.attn.window_size: feats.append(f"win{blk.attn.window_size}")
            if blk.embed_gate: feats.append("egate")
            feat_str = "+".join(feats) if feats else "std"

            print(f"  L{i:2d} MoE  {feat_str:20s}  total={blk_total:>10,}  active={blk_active:>10,}  "
                  f"(attn={attn_total:,} shared={shared_p:,} routed={routed_active:,}/{routed_total:,})")
        else:
            mlp_total = sum(p.numel() for p in blk.mlp.parameters())
            blk_active = attn_total + alpha_total + egate_total + mlp_total  # dense = 100% active

            feats = []
            if blk.attn.differential: feats.append("diff")
            if blk.attn.paired: feats.append("pha")
            if blk.attn.trap_mix: feats.append("trap")
            if blk.embed_gate: feats.append("egate")
            feat_str = "+".join(feats) if feats else "std"

            print(f"  L{i:2d} Dense {feat_str:20s}  total={blk_total:>10,}  active={blk_active:>10,}  "
                  f"(attn={attn_total:,} mlp={mlp_total:,})")

        total_block_params += blk_total
        total_block_active += blk_active

    print()
    print(f"  {'─'*66}")
    print(f"  Blocks total:  {total_block_params:>12,}")
    print(f"  Blocks active: {total_block_active:>12,}")
    print()

    grand_total = embed_head_unique + s_z_p + total_block_params
    grand_active = embed_head_unique + s_z_p + total_block_active

    print(f"  TOTAL PARAMS:  {total_unique:>12,}  (unique, deduplicated)")
    print(f"  ACTIVE/TOKEN:  {grand_active:>12,}")
    print(f"  ACTIVE RATIO:  {grand_active/total_unique:>11.1%}")
    print()


if __name__ == "__main__":
    main()
