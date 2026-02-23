"""
ReFusion diffusion sampler for YAMIT.

Implements iterative block/slot decode with AR verification and MLA latent cache.
This is Phase-1 correctness code (batch size 1 for now).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from yamit.model import YAMIT


@dataclass
class ReFusionSamplerConfig:
    slot_size: int = 8
    serial_num_blocks: int = 2
    slot_threshold: float = 0.9
    token_threshold: float = 0.9
    max_refinement_iters: Optional[int] = None
    temperature: float = 0.0
    force_accept_fallback: bool = True


@dataclass
class ReFusionSamplerStats:
    slot_accept_rate: float
    verification_pass_rate: float
    iterations_per_generated_token: float
    eos_hit: bool


def _sample_tokens(logits: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample token IDs and return their probabilities.

    Args:
        logits: (N, V)
        temperature: 0.0 means greedy; >0 uses Gumbel-max style sampling in float64.
    """
    if temperature <= 0.0:
        token_ids = torch.argmax(logits, dim=-1)
    else:
        x = logits.to(torch.float64)
        noise = torch.rand_like(x, dtype=torch.float64)
        gumbel = (-torch.log(noise)).pow(temperature)
        token_ids = torch.argmax(x.exp() / gumbel, dim=-1)

    probs = torch.softmax(logits.float(), dim=-1)
    token_probs = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    return token_ids, token_probs


@torch.no_grad()
def generate_refusion(
    model: YAMIT,
    prompt_ids: torch.Tensor,
    gen_length: int,
    mask_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    config: Optional[ReFusionSamplerConfig] = None,
) -> tuple[torch.Tensor, ReFusionSamplerStats]:
    """Generate with ReFusion iterative diffusion sampling.

    Args:
        model: YAMIT model.
        prompt_ids: (B, P) prompt token IDs. Current implementation supports B=1.
        gen_length: number of tokens to generate.
        mask_token_id: diffusion mask token. Defaults to model.cfg.mask_token_id.
        eos_token_id: EOS token. Defaults to model.cfg.eos_token_id.
        config: sampler config.

    Returns:
        (full_sequence_ids, stats), where full sequence shape is (B, P+generated).
    """
    if config is None:
        config = ReFusionSamplerConfig()
    if config.max_refinement_iters is None:
        config.max_refinement_iters = config.slot_size

    if prompt_ids.ndim != 2:
        raise ValueError("prompt_ids must be rank-2 [B, P]")
    if prompt_ids.shape[0] != 1:
        raise ValueError("Phase-1 sampler currently supports batch size 1")

    device = prompt_ids.device
    B, prompt_len = prompt_ids.shape
    mask_token_id = model.cfg.mask_token_id if mask_token_id is None else mask_token_id
    eos_token_id = model.cfg.eos_token_id if eos_token_id is None else eos_token_id

    # Prefill prompt cache.
    cache = model.init_cache()
    prompt_pos = torch.arange(prompt_len, device=device).unsqueeze(0)
    _, cache = model.forward_with_cache(
        prompt_ids,
        position_ids=prompt_pos,
        cache=cache,
        use_cache=True,
    )

    target_positions = torch.arange(prompt_len, prompt_len + gen_length, device=device)
    block_len = math.ceil(gen_length / config.serial_num_blocks)

    generated_by_pos: dict[int, int] = {}
    total_slots_seen = 0
    total_slots_selected = 0
    total_tokens_verified = 0
    total_tokens_accepted = 0
    total_refinement_iters = 0
    eos_hit = False

    for block_idx in range(config.serial_num_blocks):
        b0 = block_idx * block_len
        b1 = min((block_idx + 1) * block_len, gen_length)
        if b0 >= b1:
            break

        block_positions = target_positions[b0:b1]
        slots: list[dict[str, torch.Tensor]] = []
        for i in range(0, block_positions.numel(), config.slot_size):
            slots.append({"positions": block_positions[i : i + config.slot_size]})

        refinement_iter = 0
        while slots:
            if refinement_iter >= config.max_refinement_iters * max(1, len(slots)):
                break
            refinement_iter += 1
            total_refinement_iters += 1

            total_slots_seen += len(slots)

            flat_pos = torch.cat([s["positions"] for s in slots], dim=0)
            flat_in = torch.full_like(flat_pos, fill_value=mask_token_id)

            # MDM prediction over all unresolved slots.
            work_cache = cache.clone()
            mdm_logits, _ = model.forward_with_cache(
                flat_in.unsqueeze(0),
                position_ids=flat_pos.unsqueeze(0),
                cache=work_cache,
                use_cache=True,
            )
            mdm_logits = mdm_logits[0]  # (N, V)

            proposal_ids, proposal_probs = _sample_tokens(mdm_logits, config.temperature)

            # Split proposals back per-slot.
            offset = 0
            slot_props = []
            slot_conf = []
            for s in slots:
                n = int(s["positions"].numel())
                ids = proposal_ids[offset : offset + n]
                probs = proposal_probs[offset : offset + n]
                slot_props.append((ids, probs))
                slot_conf.append(probs[0])
                offset += n

            conf = torch.stack(slot_conf)
            selected = torch.nonzero(conf > config.slot_threshold, as_tuple=False).flatten()
            if selected.numel() == 0 and config.force_accept_fallback:
                selected = torch.topk(conf, k=1).indices
            selected = torch.sort(selected).values
            selected_set = set(int(i.item()) for i in selected)
            total_slots_selected += len(selected_set)

            next_slots: list[dict[str, torch.Tensor]] = []

            # Unselected slots remain unresolved.
            for i, s in enumerate(slots):
                if i not in selected_set:
                    next_slots.append(s)

            # Verify selected slots autoregressively.
            for i in selected.tolist():
                slot = slots[i]
                positions = slot["positions"]
                proposed_ids, proposed_probs = slot_props[i]

                # AR verification pass on selected slot.
                verify_cache = cache.clone()
                ar_logits, _ = model.forward_with_cache(
                    proposed_ids.unsqueeze(0),
                    position_ids=positions.unsqueeze(0),
                    cache=verify_cache,
                    use_cache=True,
                )
                ar_logits = ar_logits[0]  # (L, V)
                ar_logits = torch.cat([ar_logits[:1], ar_logits[:-1]], dim=0)
                ar_probs = (
                    torch.softmax(ar_logits.float(), dim=-1)
                    .gather(-1, proposed_ids.unsqueeze(-1))
                    .squeeze(-1)
                )

                total_tokens_verified += int(proposed_ids.numel())

                final_probs = proposed_probs.clone()
                if final_probs.numel() > 1:
                    final_probs[1:] = ar_probs[1:]

                accept = final_probs > config.token_threshold
                accept[0] = True
                prefix_len = int(torch.cumprod(accept.long(), dim=0).sum().item())

                accepted_ids = proposed_ids[:prefix_len]
                accepted_pos = positions[:prefix_len]
                total_tokens_accepted += int(prefix_len)

                # Commit accepted prefix to real cache.
                _, cache = model.forward_with_cache(
                    accepted_ids.unsqueeze(0),
                    position_ids=accepted_pos.unsqueeze(0),
                    cache=cache,
                    use_cache=True,
                )

                for p, tok in zip(accepted_pos.tolist(), accepted_ids.tolist()):
                    generated_by_pos[int(p)] = int(tok)
                    if tok == eos_token_id:
                        eos_hit = True

                if eos_hit:
                    break

                remain_pos = positions[prefix_len:]
                if remain_pos.numel() > 0:
                    next_slots.append({"positions": remain_pos})

            slots = next_slots

            if eos_hit:
                break

        if eos_hit:
            break

    generated_tokens = []
    for p in target_positions.tolist():
        if p in generated_by_pos:
            generated_tokens.append(generated_by_pos[p])
            if generated_by_pos[p] == eos_token_id:
                break
        else:
            generated_tokens.append(mask_token_id)

    generated_tensor = torch.tensor(generated_tokens, device=device, dtype=prompt_ids.dtype).unsqueeze(0)
    out = torch.cat([prompt_ids, generated_tensor], dim=1)

    slot_accept_rate = (
        float(total_slots_selected) / float(total_slots_seen)
        if total_slots_seen > 0
        else 0.0
    )
    verification_pass_rate = (
        float(total_tokens_accepted) / float(total_tokens_verified)
        if total_tokens_verified > 0
        else 0.0
    )
    iterations_per_generated_token = (
        float(total_refinement_iters) / float(max(1, len(generated_tokens)))
    )

    stats = ReFusionSamplerStats(
        slot_accept_rate=slot_accept_rate,
        verification_pass_rate=verification_pass_rate,
        iterations_per_generated_token=iterations_per_generated_token,
        eos_hit=eos_hit,
    )
    return out, stats
