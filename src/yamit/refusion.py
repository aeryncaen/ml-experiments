"""
ReFusion forward process and dual loss for YAMIT training.

Implements the AR+MDM hybrid training objective from:
  ReFusion: slot-level masked diffusion + autoregressive verification.

Usage (pretraining from scratch):
    from refusion import forward_process, refusion_loss

    # Prepare batch (prompt_len=0 for pretraining).
    fp = forward_process(input_ids, attention_mask, mask_token_id=151670)
    logits = model(fp.input_ids, position_ids=fp.position_ids)
    loss, ar_loss, mdm_loss = refusion_loss(
        logits, fp.labels, fp.masked_indices, fp.p_masks, fp.answer_lengths,
    )
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Forward process output
# ---------------------------------------------------------------------------

@dataclass
class ForwardProcessOutput:
    """Packed output of the ReFusion forward process."""
    input_ids: torch.Tensor         # (B, T) — mask tokens inserted
    labels: torch.Tensor            # (B, T) — target token IDs, IGNORE_INDEX where no loss
    masked_indices: torch.Tensor    # (B, T) — bool, True = MDM position
    p_masks: torch.Tensor           # (B, T) — per-sample p_mask broadcast to seq len
    answer_lengths: torch.Tensor    # (B, T) — per-sample answer length broadcast
    position_ids: torch.Tensor      # (B, T) — remapped positions (original positions)


# ---------------------------------------------------------------------------
# Forward process
# ---------------------------------------------------------------------------

def forward_process(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token_id: int,
    prompt_lengths: Optional[torch.Tensor] = None,
    boundary_token_id: Optional[int] = None,
    slot_size_set: tuple[int, ...] = (4, 8, 16, 32),
    eps: float = 1e-3,
) -> ForwardProcessOutput:
    """ReFusion forward process: slot partition → mask → shuffle → relabel.

    For pretraining from scratch, set prompt_lengths=None (defaults to 0 for
    all samples, treating the entire sequence as the response region).

    Args:
        input_ids:       (B, T) raw token IDs.
        attention_mask:  (B, T) 1 = real token, 0 = padding.
        mask_token_id:   ID of the <|mask|> token.
        prompt_lengths:  (B,) or (B,1) number of prompt tokens per sample.
                         None → all zeros (pretraining mode).
        boundary_token_id:
                         Optional token ID used as document boundary. If set,
                         slot partitioning never crosses this boundary token.
        slot_size_set:   candidate slot sizes (uniform random per sample).
        eps:             minimum mask probability floor.

    Returns:
        ForwardProcessOutput with all tensors on the same device as input_ids.
    """
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    # Default: pretraining → prompt_len = 0 for all samples.
    if prompt_lengths is None:
        prompt_lengths_list = [0] * batch_size
    else:
        if prompt_lengths.ndim == 2:
            prompt_lengths = prompt_lengths.squeeze(1)
        prompt_lengths_list = prompt_lengths.tolist()

    total_lengths = attention_mask.sum(dim=1).tolist()
    input_ids_list = input_ids.tolist()

    out_input_ids = []
    out_labels = []
    out_masked_indices = []
    out_p_masks = []
    out_answer_lengths = []
    out_position_ids = []

    for i in range(batch_size):
        slot_size = random.choice(slot_size_set)
        prompt_len = int(prompt_lengths_list[i])
        total_len = int(total_lengths[i])
        pad_len = seq_length - total_len

        # ── prompt region (no loss) ──
        prompt_ids = input_ids_list[i][:prompt_len]
        prompt_labels = [IGNORE_INDEX] * prompt_len
        prompt_positions = list(range(prompt_len))

        # ── answer region ──
        answer_ids = input_ids_list[i][prompt_len:total_len]
        answer_length = len(answer_ids)

        if answer_length == 0:
            # Nothing to process — fill with padding.
            out_input_ids.append(torch.tensor(input_ids_list[i], dtype=torch.long))
            out_labels.append(torch.full((seq_length,), IGNORE_INDEX, dtype=torch.long))
            out_masked_indices.append(torch.zeros(seq_length, dtype=torch.bool))
            out_p_masks.append(torch.tensor(0.0))
            out_answer_lengths.append(torch.tensor(0.0))
            out_position_ids.append(torch.arange(seq_length, dtype=torch.long))
            continue

        # ── slot partitioning (respect doc boundaries when provided) ──
        answer_start_pos = prompt_len
        answer_positions = list(range(answer_start_pos, answer_start_pos + answer_length))

        answer_slots: list[list[int]] = []
        position_slots: list[list[int]] = []

        if boundary_token_id is None:
            answer_slots = [
                answer_ids[j : j + slot_size]
                for j in range(0, answer_length, slot_size)
            ]
            position_slots = [
                answer_positions[j : j + slot_size]
                for j in range(0, answer_length, slot_size)
            ]
        else:
            # Split answer into document segments, then slot each segment.
            seg_tokens: list[int] = []
            seg_pos: list[int] = []
            for tok, pos in zip(answer_ids, answer_positions):
                seg_tokens.append(tok)
                seg_pos.append(pos)
                if tok == boundary_token_id:
                    for j in range(0, len(seg_tokens), slot_size):
                        answer_slots.append(seg_tokens[j : j + slot_size])
                        position_slots.append(seg_pos[j : j + slot_size])
                    seg_tokens = []
                    seg_pos = []

            if seg_tokens:
                for j in range(0, len(seg_tokens), slot_size):
                    answer_slots.append(seg_tokens[j : j + slot_size])
                    position_slots.append(seg_pos[j : j + slot_size])

        num_slots = len(answer_slots)

        # ── mask schedule ──
        t = random.random()
        p_mask = (1.0 - eps) * t + eps         # p_mask ∈ [eps, 1.0)
        slot_mask = [random.random() < p_mask for _ in range(num_slots)]

        unmasked_indices = [s for s, m in enumerate(slot_mask) if not m]
        masked_indices_list = [s for s, m in enumerate(slot_mask) if m]

        # ── shuffle unmasked (AR) slots ──
        random.shuffle(unmasked_indices)

        # ── build rearranged answer ──
        final_ids: list[int] = []
        final_labels: list[int] = []
        final_masked: list[bool] = []
        final_positions: list[int] = []

        # AR slots (shuffled, next-token labels within slot)
        for slot_idx in unmasked_indices:
            slot = answer_slots[slot_idx]
            final_ids.extend(slot)
            # Next-token prediction: label[i] = slot[i+1], last gets IGNORE.
            ar_labels = list(slot[1:]) + [IGNORE_INDEX]
            final_labels.extend(ar_labels)
            final_masked.extend([False] * len(slot))
            final_positions.extend(position_slots[slot_idx])

        # MDM slots (original order, original-token labels)
        for slot_idx in masked_indices_list:
            slot = answer_slots[slot_idx]
            final_ids.extend([mask_token_id] * len(slot))
            final_labels.extend(slot)
            final_masked.extend([True] * len(slot))
            final_positions.extend(position_slots[slot_idx])

        # ── assemble full sequence ──
        # [prompt] [rearranged answer] [padding]
        pad_ids = input_ids_list[i][total_len:]  # original padding tokens
        full_ids = prompt_ids + final_ids + pad_ids
        full_labels = prompt_labels + final_labels + [IGNORE_INDEX] * pad_len
        full_masked = [False] * prompt_len + final_masked + [False] * pad_len
        full_positions = (
            prompt_positions
            + final_positions
            + list(range(total_len, seq_length))
        )

        assert len(full_ids) == seq_length, f"ids len {len(full_ids)} != {seq_length}"
        assert len(full_labels) == seq_length
        assert len(full_masked) == seq_length
        assert len(full_positions) == seq_length

        out_input_ids.append(torch.tensor(full_ids, dtype=torch.long))
        out_labels.append(torch.tensor(full_labels, dtype=torch.long))
        out_masked_indices.append(torch.tensor(full_masked, dtype=torch.bool))
        out_p_masks.append(torch.tensor(p_mask, dtype=torch.float32))
        out_answer_lengths.append(torch.tensor(float(answer_length), dtype=torch.float32))
        out_position_ids.append(torch.tensor(full_positions, dtype=torch.long))

    # ── stack and broadcast scalars ──
    input_ids_out = torch.stack(out_input_ids).to(device)
    labels_out = torch.stack(out_labels).to(device)
    masked_out = torch.stack(out_masked_indices).to(device)
    position_ids_out = torch.stack(out_position_ids).to(device)

    # p_masks and answer_lengths: broadcast scalar per sample → (B, T)
    p_masks_out = torch.stack(out_p_masks).view(-1, 1).expand(-1, seq_length).to(device)
    answer_lengths_out = (
        torch.stack(out_answer_lengths).view(-1, 1).expand(-1, seq_length).to(device)
    )

    return ForwardProcessOutput(
        input_ids=input_ids_out,
        labels=labels_out,
        masked_indices=masked_out,
        p_masks=p_masks_out,
        answer_lengths=answer_lengths_out,
        position_ids=position_ids_out,
    )


# ---------------------------------------------------------------------------
# Dual loss (AR + MDM)
# ---------------------------------------------------------------------------

def refusion_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    masked_indices: torch.Tensor,
    p_masks: torch.Tensor,
    answer_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ReFusion dual loss: L_AR + L_MDM.

    Args:
        logits:          (B, T, V) model output logits.
        labels:          (B, T)    target token IDs (IGNORE_INDEX = no loss).
        masked_indices:  (B, T)    bool, True = MDM position.
        p_masks:         (B, T)    per-sample mask probability (broadcast).
        answer_lengths:  (B, T)    per-sample answer length (broadcast).

    Returns:
        (loss_total, loss_ar, loss_mdm) — all scalar tensors.

    Loss formulas (from ReFusion paper / spec):
        L_AR  = mean CE over AR tokens (ignore_index=-100)
        L_MDM = (1/B) Σ_i CE(logit_i, label_i) / (p_mask_i × answer_length_i)
        L     = L_AR + L_MDM
    """
    B = logits.shape[0]
    V = logits.shape[-1]

    # Flatten everything.
    flat_logits = logits.float().reshape(-1, V)     # (B*T, V)
    flat_labels = labels.reshape(-1)                 # (B*T,)
    flat_mask = masked_indices.reshape(-1)            # (B*T,) bool
    flat_pmask = p_masks.reshape(-1)                  # (B*T,)
    flat_alen = answer_lengths.reshape(-1)            # (B*T,)

    ar_idx = ~flat_mask
    mdm_idx = flat_mask

    # ── AR loss ──
    # Only compute over positions where labels != IGNORE_INDEX.
    ar_logits = flat_logits[ar_idx]
    ar_labels = flat_labels[ar_idx]
    if ar_logits.numel() > 0 and (ar_labels != IGNORE_INDEX).any():
        loss_ar = torch.nn.functional.cross_entropy(
            ar_logits, ar_labels, ignore_index=IGNORE_INDEX, reduction="mean"
        )
    else:
        loss_ar = logits.new_tensor(0.0)

    # ── MDM loss ──
    mdm_logits = flat_logits[mdm_idx]
    mdm_labels = flat_labels[mdm_idx]
    mdm_pmask = flat_pmask[mdm_idx]
    mdm_alen = flat_alen[mdm_idx]
    if mdm_logits.numel() > 0 and (mdm_labels != IGNORE_INDEX).any():
        per_token = torch.nn.functional.cross_entropy(
            mdm_logits, mdm_labels, ignore_index=IGNORE_INDEX, reduction="none"
        )
        # Importance weighting: 1/p_mask, normalised by answer_length and batch.
        weighted = per_token / mdm_pmask.clamp(min=1e-6)
        loss_mdm = torch.sum(weighted / mdm_alen.clamp(min=1.0)) / B
    else:
        loss_mdm = logits.new_tensor(0.0)

    loss_total = loss_ar + loss_mdm
    return loss_total, loss_ar, loss_mdm


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    B, T, V = 4, 128, 1000
    mask_token_id = 999

    # Simulate a batch of sequences (no padding for simplicity).
    input_ids = torch.randint(0, V - 1, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)

    print("=== Forward Process ===")
    fp = forward_process(input_ids, attention_mask, mask_token_id=mask_token_id)
    print(f"  input_ids shape:    {fp.input_ids.shape}")
    print(f"  labels shape:       {fp.labels.shape}")
    print(f"  masked_indices:     {fp.masked_indices.sum().item()} MDM tokens / {B*T} total")
    print(f"  position_ids range: [{fp.position_ids.min().item()}, {fp.position_ids.max().item()}]")
    print(f"  p_masks (sample 0): {fp.p_masks[0, 0].item():.4f}")
    print(f"  answer_len (all):   {fp.answer_lengths[0, 0].item():.0f}")

    # Verify mask tokens are in the right places.
    n_mask_tokens = (fp.input_ids == mask_token_id).sum().item()
    n_masked_idx = fp.masked_indices.sum().item()
    assert n_mask_tokens == n_masked_idx, (
        f"Mask token count {n_mask_tokens} != masked_indices count {n_masked_idx}"
    )
    print(f"  Mask token count matches masked_indices: {n_mask_tokens}")

    # Verify labels are set for masked positions.
    mdm_labels = fp.labels[fp.masked_indices]
    assert (mdm_labels != IGNORE_INDEX).all(), "MDM labels should all be valid tokens"
    print(f"  MDM labels valid: all {mdm_labels.numel()} labels are real tokens")

    print("\n=== Loss Computation ===")
    logits = torch.randn(B, T, V)
    loss, ar, mdm = refusion_loss(
        logits, fp.labels, fp.masked_indices, fp.p_masks, fp.answer_lengths
    )
    print(f"  L_total = {loss.item():.4f}")
    print(f"  L_AR    = {ar.item():.4f}")
    print(f"  L_MDM   = {mdm.item():.4f}")
    print("\nAll checks passed.")
