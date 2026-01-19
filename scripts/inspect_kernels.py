#!/usr/bin/env python3
"""Inspect which conv branches contribute most and show adaptation histograms."""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from heuristic_secrets.models import Model, ModelConfig, create_binary_batches
from heuristic_secrets.models.backbone import (
    MultiKernelSSMBlock,
    AdaptiveDeformConv1d,
    ContextualAttentionBlock,
)


def load_checkpoint(path: Path, device: torch.device) -> tuple[Model, ModelConfig]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = Model(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, config


def ascii_histogram(
    values: list[float], bins: int = 20, width: int = 40, label: str = ""
) -> str:
    if not values:
        return f"  {label}: (no data)"

    min_v, max_v = min(values), max(values)
    if min_v == max_v:
        return f"  {label}: all values = {min_v:.4f}"

    bin_width = (max_v - min_v) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - min_v) / bin_width), bins - 1)
        counts[idx] += 1

    max_count = max(counts) if counts else 1
    lines = [f"  {label} (n={len(values)}, range=[{min_v:.3f}, {max_v:.3f}])"]

    for i, count in enumerate(counts):
        bar_len = int(count / max_count * width) if max_count > 0 else 0
        bin_start = min_v + i * bin_width
        lines.append(
            f"  {bin_start:>7.3f} |{'█' * bar_len}{' ' * (width - bar_len)}| {count}"
        )

    return "\n".join(lines)


class BranchTracker:
    def __init__(self, model: Model, config: ModelConfig):
        self.config = config
        self.is_adaptive = config.adaptive_conv
        self.n_branches = 0
        self.branch_labels = []
        self.branch_norms: dict[str, list[float]] = {}
        self.branch_outputs: dict[str, list[float]] = {}
        self.hook_handles = []
        self.block = None

        self.effective_sigmas: dict[str, list[float]] = {}
        self.attn_sigma_biases: list[torch.Tensor] = []
        self.attn_offset_biases: list[torch.Tensor] = []
        self.attn_omega_biases: list[torch.Tensor] = []

        self.correct_sigma_biases: dict[str, list[float]] = {}
        self.incorrect_sigma_biases: dict[str, list[float]] = {}
        self.correct_offset_biases: dict[str, list[float]] = {}
        self.incorrect_offset_biases: dict[str, list[float]] = {}
        self.correct_norms: dict[str, list[float]] = {}
        self.incorrect_norms: dict[str, list[float]] = {}

        self.pos_sigma_biases: dict[str, list[float]] = {}
        self.neg_sigma_biases: dict[str, list[float]] = {}
        self.pos_offset_biases: dict[str, list[float]] = {}
        self.neg_offset_biases: dict[str, list[float]] = {}
        self.pos_norms: dict[str, list[float]] = {}
        self.neg_norms: dict[str, list[float]] = {}

        self._find_block(model)

    def _find_block(self, model: Model):
        for name, module in model.named_modules():
            if isinstance(module, MultiKernelSSMBlock):
                self.block = module
                self.n_branches = module.n_branches

                for i, branch in enumerate(module.branches):
                    if hasattr(branch, "conv") and isinstance(
                        branch.conv, AdaptiveDeformConv1d
                    ):
                        sigma = branch.conv.log_sigma.exp().item()
                        label = f"b{i}(σ={sigma:.2f})"
                    else:
                        label = f"branch_{i}"
                    self.branch_labels.append(label)
                    self.branch_norms[label] = []
                    self.effective_sigmas[label] = []
                    self.correct_sigma_biases[label] = []
                    self.incorrect_sigma_biases[label] = []
                    self.correct_offset_biases[label] = []
                    self.incorrect_offset_biases[label] = []
                    self.correct_norms[label] = []
                    self.incorrect_norms[label] = []
                    self.pos_sigma_biases[label] = []
                    self.neg_sigma_biases[label] = []
                    self.pos_offset_biases[label] = []
                    self.neg_offset_biases[label] = []
                    self.pos_norms[label] = []
                    self.neg_norms[label] = []
                break

            if self.is_adaptive and isinstance(module, ContextualAttentionBlock):
                handle = module.register_forward_hook(self._attn_hook_fn)
                self.hook_handles.append(handle)

    def _attn_hook_fn(self, module, inputs, output):
        if isinstance(output, tuple) and len(output) >= 2:
            _, biases, _ = output
            if biases is not None:
                if biases.sigma is not None:
                    self.attn_sigma_biases.append(biases.sigma.detach().cpu())
                if biases.offset_scale is not None:
                    self.attn_offset_biases.append(biases.offset_scale.detach().cpu())
                if biases.omega is not None:
                    self.attn_omega_biases.append(biases.omega.detach().cpu())

    def track_batch(self, x: torch.Tensor, mask: torch.Tensor | None, biases):
        if self.block is None:
            return

        with torch.no_grad():
            h = self.block.in_proj(self.block.in_norm(x))
            chunks = h.chunk(self.n_branches, dim=-1)

            for i, (label, branch, chunk) in enumerate(
                zip(self.branch_labels, self.block.branches, chunks)
            ):
                if biases is not None:
                    from heuristic_secrets.models.backbone import AdaptiveConvBiases

                    branch_bias = AdaptiveConvBiases(
                        sigma=biases.sigma[:, i : i + 1].squeeze(-1)
                        if biases.sigma is not None
                        else None,
                        offset_scale=biases.offset_scale[:, i : i + 1].squeeze(-1)
                        if biases.offset_scale is not None
                        else None,
                        omega=biases.omega[:, i : i + 1].squeeze(-1)
                        if biases.omega is not None
                        else None,
                    )
                    out, aux = branch(chunk, mask, branch_bias)

                    if hasattr(branch, "conv") and isinstance(
                        branch.conv, AdaptiveDeformConv1d
                    ):
                        base_sigma = branch.conv.log_sigma.exp()
                        sigma_bias = (
                            branch_bias.sigma if branch_bias.sigma is not None else 0
                        )
                        eff_sigma = torch.clamp(
                            base_sigma + sigma_bias,
                            branch.conv.min_sigma,
                            branch.conv.max_sigma,
                        )
                        for s in eff_sigma.tolist():
                            self.effective_sigmas[label].append(s)
                else:
                    out, aux = branch(chunk, mask)
                    if hasattr(branch, "conv") and isinstance(
                        branch.conv, AdaptiveDeformConv1d
                    ):
                        sigma = branch.conv.log_sigma.exp().item()
                        for _ in range(chunk.size(0)):
                            self.effective_sigmas[label].append(sigma)

                norm = out.norm(dim=-1).mean(dim=-1).mean().item()
                self.branch_norms[label].append(norm)

    def record_correctness(self, correct_mask: torch.Tensor):
        if not self.attn_sigma_biases:
            return

        last_sigma = self.attn_sigma_biases[-1]
        last_offset = self.attn_offset_biases[-1] if self.attn_offset_biases else None

        for i, label in enumerate(self.branch_labels):
            sigma_vals = last_sigma[:, i].tolist()
            offset_vals = last_offset[:, i].tolist() if last_offset is not None else []

            for j, is_correct in enumerate(correct_mask.tolist()):
                if is_correct:
                    self.correct_sigma_biases[label].append(sigma_vals[j])
                    if offset_vals:
                        self.correct_offset_biases[label].append(offset_vals[j])
                else:
                    self.incorrect_sigma_biases[label].append(sigma_vals[j])
                    if offset_vals:
                        self.incorrect_offset_biases[label].append(offset_vals[j])

    def record_labels(self, labels: torch.Tensor):
        if not self.attn_sigma_biases:
            return

        last_sigma = self.attn_sigma_biases[-1]
        last_offset = self.attn_offset_biases[-1] if self.attn_offset_biases else None

        for i, branch_label in enumerate(self.branch_labels):
            sigma_vals = last_sigma[:, i].tolist()
            offset_vals = last_offset[:, i].tolist() if last_offset is not None else []

            for j, y in enumerate(labels.tolist()):
                is_pos = y > 0.5
                if is_pos:
                    self.pos_sigma_biases[branch_label].append(sigma_vals[j])
                    if offset_vals:
                        self.pos_offset_biases[branch_label].append(offset_vals[j])
                else:
                    self.neg_sigma_biases[branch_label].append(sigma_vals[j])
                    if offset_vals:
                        self.neg_offset_biases[branch_label].append(offset_vals[j])

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def report(self):
        print("\n" + "=" * 70)
        print("BRANCH CONTRIBUTIONS (by output norm)")
        print("=" * 70)

        total_norm = sum(sum(v) for v in self.branch_norms.values())

        results = []
        for label in self.branch_labels:
            norms = self.branch_norms[label]
            if norms:
                mean_norm = sum(norms) / len(norms)
                contribution = sum(norms) / total_norm * 100 if total_norm > 0 else 0
                results.append((label, mean_norm, contribution))

        results.sort(key=lambda x: -x[2])

        max_contrib = max(r[2] for r in results) if results else 1
        print()
        for label, mean_norm, contribution in results:
            bar_len = int(contribution / max_contrib * 30)
            print(
                f"  {label:<18} {'█' * bar_len:<30} {contribution:>5.1f}% (norm={mean_norm:.3f})"
            )

        if self.is_adaptive:
            self._report_sigma_histograms()
            self._report_attention_bias_histograms()
            self._report_correct_vs_incorrect()
            self._report_pos_vs_neg()

    def _report_sigma_histograms(self):
        print("\n" + "=" * 70)
        print("EFFECTIVE SIGMA DISTRIBUTIONS (learned + attention bias)")
        print("=" * 70)

        for label in self.branch_labels:
            sigmas = self.effective_sigmas[label]
            if sigmas:
                print()
                print(ascii_histogram(sigmas, bins=15, width=35, label=label))

    def _report_attention_bias_histograms(self):
        if not self.attn_sigma_biases:
            return

        print("\n" + "=" * 70)
        print("ATTENTION-GENERATED SIGMA BIASES (per branch)")
        print("=" * 70)

        sigma_all = torch.cat(self.attn_sigma_biases, dim=0)

        for i, label in enumerate(self.branch_labels):
            biases = sigma_all[:, i].tolist()
            print()
            print(ascii_histogram(biases, bins=15, width=35, label=f"{label} Δσ"))

        if self.attn_offset_biases:
            print("\n" + "=" * 70)
            print("ATTENTION-GENERATED OFFSET SCALE BIASES")
            print("=" * 70)
            offset_all = torch.cat(self.attn_offset_biases, dim=0)
            for i, label in enumerate(self.branch_labels):
                biases = offset_all[:, i].tolist()
                print()
                print(
                    ascii_histogram(biases, bins=15, width=35, label=f"{label} Δoffset")
                )

    def _report_correct_vs_incorrect(self):
        has_data = any(self.correct_sigma_biases.get(l) for l in self.branch_labels)
        if not has_data:
            return

        print("\n" + "=" * 70)
        print("CORRECT vs INCORRECT PREDICTIONS - SIGMA BIAS COMPARISON")
        print("=" * 70)

        for label in self.branch_labels:
            correct = self.correct_sigma_biases.get(label, [])
            incorrect = self.incorrect_sigma_biases.get(label, [])

            if not correct and not incorrect:
                continue

            print(f"\n  {label}:")

            c_mean = sum(correct) / len(correct) if correct else 0
            i_mean = sum(incorrect) / len(incorrect) if incorrect else 0
            c_std = (
                (sum((x - c_mean) ** 2 for x in correct) / len(correct)) ** 0.5
                if len(correct) > 1
                else 0
            )
            i_std = (
                (sum((x - i_mean) ** 2 for x in incorrect) / len(incorrect)) ** 0.5
                if len(incorrect) > 1
                else 0
            )

            print(
                f"    CORRECT   (n={len(correct):>5}): mean={c_mean:+.4f}, std={c_std:.4f}"
            )
            print(
                f"    INCORRECT (n={len(incorrect):>5}): mean={i_mean:+.4f}, std={i_std:.4f}"
            )
            print(f"    DIFFERENCE: {c_mean - i_mean:+.4f}")

            if correct:
                print()
                print(
                    ascii_histogram(correct, bins=12, width=30, label="    Correct Δσ")
                )
            if incorrect:
                print()
                print(
                    ascii_histogram(
                        incorrect, bins=12, width=30, label="    Incorrect Δσ"
                    )
                )

        has_offset = any(self.correct_offset_biases.get(l) for l in self.branch_labels)
        if not has_offset:
            return

        print("\n" + "=" * 70)
        print("CORRECT vs INCORRECT PREDICTIONS - OFFSET BIAS COMPARISON")
        print("=" * 70)

        for label in self.branch_labels:
            correct = self.correct_offset_biases.get(label, [])
            incorrect = self.incorrect_offset_biases.get(label, [])

            if not correct and not incorrect:
                continue

            print(f"\n  {label}:")

            c_mean = sum(correct) / len(correct) if correct else 0
            i_mean = sum(incorrect) / len(incorrect) if incorrect else 0

            print(f"    CORRECT   (n={len(correct):>5}): mean={c_mean:+.4f}")
            print(f"    INCORRECT (n={len(incorrect):>5}): mean={i_mean:+.4f}")
            print(f"    DIFFERENCE: {c_mean - i_mean:+.4f}")

    def _report_pos_vs_neg(self):
        has_data = any(self.pos_sigma_biases.get(l) for l in self.branch_labels)
        if not has_data:
            return

        print("\n" + "=" * 70)
        print("POSITIVE (secrets) vs NEGATIVE (clean) - SIGMA BIAS")
        print("=" * 70)

        for label in self.branch_labels:
            pos = self.pos_sigma_biases.get(label, [])
            neg = self.neg_sigma_biases.get(label, [])

            if not pos and not neg:
                continue

            print(f"\n  {label}:")

            p_mean = sum(pos) / len(pos) if pos else 0
            n_mean = sum(neg) / len(neg) if neg else 0
            p_std = (
                (sum((x - p_mean) ** 2 for x in pos) / len(pos)) ** 0.5
                if len(pos) > 1
                else 0
            )
            n_std = (
                (sum((x - n_mean) ** 2 for x in neg) / len(neg)) ** 0.5
                if len(neg) > 1
                else 0
            )

            print(
                f"    POSITIVE (n={len(pos):>5}): mean={p_mean:+.4f}, std={p_std:.4f}"
            )
            print(
                f"    NEGATIVE (n={len(neg):>5}): mean={n_mean:+.4f}, std={n_std:.4f}"
            )
            print(f"    DIFFERENCE (pos-neg): {p_mean - n_mean:+.4f}")

            if pos:
                print()
                print(ascii_histogram(pos, bins=12, width=30, label="    Positive Δσ"))
            if neg:
                print()
                print(ascii_histogram(neg, bins=12, width=30, label="    Negative Δσ"))

        has_offset = any(self.pos_offset_biases.get(l) for l in self.branch_labels)
        if not has_offset:
            return

        print("\n" + "=" * 70)
        print("POSITIVE vs NEGATIVE - OFFSET BIAS")
        print("=" * 70)

        for label in self.branch_labels:
            pos = self.pos_offset_biases.get(label, [])
            neg = self.neg_offset_biases.get(label, [])

            if not pos and not neg:
                continue

            print(f"\n  {label}:")

            p_mean = sum(pos) / len(pos) if pos else 0
            n_mean = sum(neg) / len(neg) if neg else 0

            print(f"    POSITIVE (n={len(pos):>5}): mean={p_mean:+.4f}")
            print(f"    NEGATIVE (n={len(neg):>5}): mean={n_mean:+.4f}")
            print(f"    DIFFERENCE (pos-neg): {p_mean - n_mean:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Inspect conv branch contributions")
    parser.add_argument("checkpoint", type=Path, help="Path to model checkpoint")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="Data directory"
    )
    parser.add_argument(
        "--max-samples", type=int, default=2000, help="Max samples to evaluate"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--task",
        type=str,
        default="line_filter",
        choices=["line_filter", "pathx"],
        help="Task type for data loading",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint, device)

    if config.arch_type != "unified":
        print("ERROR: Model must be unified architecture")
        print(f"  arch_type={config.arch_type}")
        return

    is_adaptive = config.adaptive_conv
    if is_adaptive:
        n_branches = config.n_adaptive_branches
        print(f"Mode: ADAPTIVE ({n_branches} branches)")

        print("\nLearned base sigmas:")
        for name, module in model.named_modules():
            if isinstance(module, MultiKernelSSMBlock):
                for i, branch in enumerate(module.branches):
                    if hasattr(branch, "conv") and isinstance(
                        branch.conv, AdaptiveDeformConv1d
                    ):
                        sigma = branch.conv.log_sigma.exp().item()
                        print(f"  Branch {i}: σ = {sigma:.4f}")
    else:
        print(f"Mode: STANDARD (kernel_sizes={config.ssm_kernel_sizes})")

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    if args.task == "line_filter":
        from train_line_filter import (
            load_dataset_with_features,
            load_or_build_text_freq_table,
        )

        print("\nLoading line_filter validation data...")
        text_freq = load_or_build_text_freq_table(args.data_dir)
        val_data = load_dataset_with_features(args.data_dir, "val", text_freq)
    else:
        from train_pathx import load_pathx_split

        print("\nLoading pathx validation data...")
        val_data = load_pathx_split("val", args.data_dir)

    n_samples = min(args.max_samples, len(val_data["labels"]))
    indices = list(range(n_samples))

    print(f"Creating batches for {n_samples} samples...")
    val_batches = create_binary_batches(
        val_data["bytes"],
        val_data["labels"],
        val_data["lengths"],
        args.batch_size,
        feature_tensors=val_data.get("features"),
        indices=indices,
        shuffle=False,
        show_progress=True,
    )

    tracker = BranchTracker(model, config)

    correct = 0
    total = 0
    last_biases = None

    print("\nRunning evaluation...")
    with torch.no_grad():
        for batch in tqdm(val_batches, desc="Evaluating"):
            x = batch.bytes.to(device)
            y = batch.labels.to(device)
            lengths = batch.lengths
            if lengths is not None:
                lengths = lengths.to(device)
                L = x.shape[1]
                mask = torch.arange(L, device=device).unsqueeze(0) >= lengths.unsqueeze(
                    1
                )
            else:
                mask = None

            features = batch.features
            if features is not None:
                features = features.to(device)

            result = model(x, mask=mask, precomputed=features)
            logits = result[0] if isinstance(result, tuple) else result

            preds = (torch.sigmoid(logits) > 0.5).float().squeeze()
            y_flat = y.squeeze()

            correct_mask = preds == y_flat
            correct += correct_mask.sum().item()
            total += y_flat.numel()

            tracker.record_correctness(correct_mask)
            tracker.record_labels(y_flat)

            if is_adaptive and tracker.attn_sigma_biases:
                biases_tensor = tracker.attn_sigma_biases[-1].to(device)
                from heuristic_secrets.models.backbone import AdaptiveConvBiases

                last_biases = AdaptiveConvBiases(
                    sigma=biases_tensor,
                    offset_scale=tracker.attn_offset_biases[-1].to(device)
                    if tracker.attn_offset_biases
                    else None,
                    omega=tracker.attn_omega_biases[-1].to(device)
                    if tracker.attn_omega_biases
                    else None,
                )

            h = model._impl.embedding(x)
            tracker.track_batch(h, mask, last_biases)

    print(f"\nAccuracy: {correct / total * 100:.2f}% ({correct}/{total})")

    tracker.report()
    tracker.remove_hooks()


if __name__ == "__main__":
    main()
