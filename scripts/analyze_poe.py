"""Analyze Pool-of-Experts routing behavior from a saved model.

Loads a saved PoE model, runs it on generated data with tracing enabled,
and reports:
  - Per-task depth distribution (mean, median, std, histogram)
  - Hot paths: most common expert sequences
  - Expert utilization: how often each expert is selected
  - Depth vs correctness: are deeper samples harder?

Usage:
    python scripts/analyze_poe.py --checkpoint path/to/ULBBlendP_mixed.pt [--n-batches 50]
"""

import sys, os, argparse, random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from ulb import ULBBlock, ULBConfig, PoolOfExperts


# ---------------------------------------------------------------------------
# Data generation (reuse from bench_ssm)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_ssm import pregen_task_data, ALL_TASKS, VOCAB_SIZE


def rebuild_model(ckpt, device='cpu'):
    """Reconstruct a PoolOfExperts model + embed + head from checkpoint."""
    args = ckpt['args']
    dim = ckpt['dim']
    n_layers = ckpt['n_layers']
    n_experts = args['n_experts']
    top_k = args['top_k']
    pool_size = n_experts * n_layers
    poe_max_hops = args.get('poe_max_hops')

    # Infer config from model name
    name = ckpt['model_name']
    if 'BlendP' in name:
        cfg = ULBConfig(d_model=dim, n_heads=4, paired=True, attn_mode='blend',
                        q_mix='lerp', k_lerp=True, swish_mode='learnable')
    elif 'Blend' in name:
        cfg = ULBConfig(d_model=dim, n_heads=4, paired=False, attn_mode='blend',
                        q_mix='lerp', k_lerp=True, swish_mode='learnable')
    else:
        cfg = ULBConfig(d_model=dim, n_heads=4)

    make = lambda: ULBBlock(cfg)
    model = PoolOfExperts(make, pool_size=pool_size, dim=dim, top_k=top_k, max_hops=poe_max_hops)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()
    model.trace = True

    # Rebuild embed/head
    embed = torch.nn.Embedding(VOCAB_SIZE, dim).to(device)
    head = torch.nn.Linear(dim, VOCAB_SIZE).to(device)

    if 'embed_state_dict' in ckpt:
        embed.load_state_dict(ckpt['embed_state_dict'])
        head.load_state_dict(ckpt['head_state_dict'])
    else:
        print("WARNING: checkpoint has no embed/head weights — accuracy will be random")

    embed.eval()
    head.eval()

    return model, embed, head


def analyze(model, embed, head, task_data, task_name, dim, device, n_batches=50):
    """Run model on data with tracing, collect routing stats and accuracy."""
    is_mixed = task_name == 'mixed'

    all_traces = []
    val_batches = task_data['val'][:n_batches]

    with torch.no_grad():
        for batch in tqdm(val_batches, desc="Tracing"):
            if is_mixed:
                inp, tgt, task_ids = batch
            else:
                inp, tgt = batch
                task_ids = None

            inp = inp.to(device)
            tgt = tgt.to(device)

            y = head(model(embed(inp)))
            preds = y.argmax(dim=-1)  # (B, L)
            valid = tgt != -100
            correct = (preds == tgt) & valid  # (B, L)
            per_sample_acc = correct.float().sum(dim=-1) / valid.float().sum(dim=-1).clamp(min=1)  # (B,)

            trace = model.last_trace
            if trace is None:
                continue

            batch_size = inp.shape[0]
            n_hops = len(trace)
            for b in range(batch_size):
                sample_depth = n_hops  # max if never exited
                sample_path = []
                for h, hop_data in enumerate(trace):
                    experts_selected = hop_data['topk_idx'][b].tolist()
                    exited = hop_data['has_exit'][b].item()

                    real_experts = [e for e in experts_selected if e < model.pool_size]
                    sample_path.extend(real_experts)

                    if exited:
                        sample_depth = h + 1
                        break

                record = {
                    'depth': sample_depth,
                    'path': tuple(sample_path),
                    'acc': per_sample_acc[b].item(),
                    'task': None,
                }
                if is_mixed and task_ids is not None:
                    tid = task_ids[b].item()
                    record['task'] = ALL_TASKS[tid] if tid < len(ALL_TASKS) else f'task_{tid}'

                all_traces.append(record)

    return all_traces


def print_report(traces, pool_size):
    """Print analysis report."""
    depths = [t['depth'] for t in traces]
    paths = [t['path'] for t in traces]

    print("=" * 80)
    print("POOL OF EXPERTS — ROUTING ANALYSIS")
    print("=" * 80)

    # Overall depth stats
    print(f"\n## Depth Statistics (n={len(traces)} samples)")
    print(f"  Mean:   {np.mean(depths):.2f}")
    print(f"  Median: {np.median(depths):.1f}")
    print(f"  Std:    {np.std(depths):.2f}")
    print(f"  Min:    {min(depths)}")
    print(f"  Max:    {max(depths)}")

    # Depth histogram
    depth_counts = Counter(depths)
    print(f"\n## Depth Histogram")
    max_depth = max(depths)
    for d in range(1, max_depth + 1):
        count = depth_counts.get(d, 0)
        pct = count / len(traces) * 100
        bar = "█" * int(pct / 2)
        print(f"  {d:>3} hops: {count:>6} ({pct:>5.1f}%) {bar}")

    # Per-task depth breakdown
    tasks = set(t['task'] for t in traces if t['task'] is not None)
    if tasks:
        print(f"\n## Per-Task Depth")
        print(f"  {'Task':<20} {'Mean':>6} {'Median':>7} {'Std':>6} {'Min':>4} {'Max':>4}")
        print(f"  {'-'*50}")
        for task in sorted(tasks):
            task_depths = [t['depth'] for t in traces if t['task'] == task]
            print(f"  {task:<20} {np.mean(task_depths):>6.2f} {np.median(task_depths):>7.1f} "
                  f"{np.std(task_depths):>6.2f} {min(task_depths):>4} {max(task_depths):>4}")

    # Expert utilization
    expert_counts = Counter()
    for path in paths:
        for e in path:
            expert_counts[e] += 1
    total_expert_uses = sum(expert_counts.values())

    print(f"\n## Expert Utilization (total selections: {total_expert_uses})")
    for eid in range(pool_size):
        count = expert_counts.get(eid, 0)
        pct = count / total_expert_uses * 100 if total_expert_uses > 0 else 0
        bar = "█" * int(pct)
        print(f"  Expert {eid:>2}: {count:>7} ({pct:>5.1f}%) {bar}")

    # Hot paths (top 20 most common expert sequences)
    path_counts = Counter(paths)
    print(f"\n## Hot Paths (top 20 of {len(path_counts)} unique)")
    print(f"  {'Rank':>4} {'Count':>7} {'Pct':>6} {'Path'}")
    print(f"  {'-'*60}")
    for rank, (path, count) in enumerate(path_counts.most_common(20), 1):
        pct = count / len(traces) * 100
        path_str = " → ".join(str(e) for e in path) if path else "(empty/immediate exit)"
        print(f"  {rank:>4} {count:>7} {pct:>5.1f}% {path_str}")

    # Per-task hot paths
    if tasks:
        for task in sorted(tasks):
            task_paths = [t['path'] for t in traces if t['task'] == task]
            task_path_counts = Counter(task_paths)
            print(f"\n## Hot Paths — {task} (top 10 of {len(task_path_counts)} unique)")
            print(f"  {'Rank':>4} {'Count':>7} {'Pct':>6} {'Path'}")
            for rank, (path, count) in enumerate(task_path_counts.most_common(10), 1):
                pct = count / len(task_paths) * 100
                path_str = " → ".join(str(e) for e in path) if path else "(empty/immediate exit)"
                print(f"  {rank:>4} {count:>7} {pct:>5.1f}% {path_str}")

    # Depth vs accuracy
    print(f"\n## Depth vs Accuracy")
    depth_accs = defaultdict(list)
    for t in traces:
        depth_accs[t['depth']].append(t['acc'])
    print(f"  {'Depth':>5} {'Count':>7} {'Mean Acc':>9} {'Std':>6}")
    print(f"  {'-'*30}")
    for d in sorted(depth_accs.keys()):
        accs = depth_accs[d]
        print(f"  {d:>5} {len(accs):>7} {np.mean(accs):>8.1%} {np.std(accs):>6.3f}")

    # Per-task depth and accuracy
    if tasks:
        print(f"\n## Per-Task Depth & Accuracy")
        print(f"  {'Task':<20} {'Mean Depth':>10} {'Mean Acc':>9}")
        print(f"  {'-'*42}")
        task_stats = {}
        for task in sorted(tasks):
            task_traces = [t for t in traces if t['task'] == task]
            mean_d = np.mean([t['depth'] for t in task_traces])
            mean_a = np.mean([t['acc'] for t in task_traces])
            task_stats[task] = (mean_d, mean_a)
            print(f"  {task:<20} {mean_d:>10.2f} {mean_a:>8.1%}")

        print(f"\n  Ranked by mean depth (deepest first):")
        for task, (mean_d, mean_a) in sorted(task_stats.items(), key=lambda x: x[1][0], reverse=True):
            print(f"    {task:<20} {mean_d:.2f} hops, {mean_a:.1%} acc")


def main():
    parser = argparse.ArgumentParser(description='Analyze PoE routing behavior')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to saved .pt checkpoint')
    parser.add_argument('--n-batches', type=int, default=50, help='Number of val batches to analyze')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data generation')
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    print(f"Model: {ckpt['model_name']}, Task: {ckpt['task']}, "
          f"Params: {ckpt['params']:,}, dim={ckpt['dim']}, layers={ckpt['n_layers']}")
    print(f"Result: acc={ckpt['result']['acc']:.1%}, stop={ckpt['result']['stop_reason']}")

    model, embed, head = rebuild_model(ckpt, device=args.device)
    print(f"Pool size: {model.pool_size}, top_k: {model.top_k}, max_hops: {model.max_hops}")

    # Generate val data
    task = ckpt['task']
    bench_args = ckpt['args']
    B = bench_args.get('batch_size', 256)
    L = bench_args.get('seq_len', 64)
    n_val = bench_args.get('val_batches', 20)
    if task == 'mixed':
        n_val *= len(ALL_TASKS)

    print(f"\nGenerating {n_val} val batches (B={B}, L={L})...")
    task_data = pregen_task_data(task, 0, n_val, B, L, args.seed, device=args.device)

    print(f"Running {min(args.n_batches, len(task_data['val']))} batches with tracing...\n")
    traces = analyze(model, embed, head, task_data, task, ckpt['dim'], args.device, n_batches=args.n_batches)

    print_report(traces, model.pool_size)


if __name__ == '__main__':
    main()
