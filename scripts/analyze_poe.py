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

import sys, argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from ulb import ULBBlock, ULBConfig, PoolOfExperts


sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_ssm import ALL_TASKS, VOCAB_SIZE, CACHE_DIR, _cache_key


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


def analyze(model, embed, head, task_data, task_name, dim, device):
    """Run model on all data (train + val) with tracing, collect routing stats and accuracy."""
    is_mixed = task_name == 'mixed'

    all_traces = []
    all_batches = task_data['train'] + task_data['val']

    with torch.no_grad():
        for batch in tqdm(all_batches, desc="Tracing"):
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
                hit_max = True
                hop_experts = []  # list of lists, one per hop
                for h, hop_data in enumerate(trace):
                    experts_selected = hop_data['topk_idx'][b].tolist()
                    exited = hop_data['has_exit'][b].item()

                    real_experts = [e for e in experts_selected if e < model.pool_size]
                    if real_experts:
                        hop_experts.append(real_experts)

                    if exited:
                        sample_depth = h + 1
                        hit_max = False
                        break

                # Cartesian product across hops -> all paths through this sample
                if hop_experts:
                    paths = [[]]
                    for hop in hop_experts:
                        paths = [p + [e] for p in paths for e in hop]
                    sample_paths = [tuple(p) for p in paths]
                else:
                    sample_paths = [()]  # immediate exit

                acc = per_sample_acc[b].item()
                record = {
                    'depth': sample_depth,
                    'paths': sample_paths,
                    'acc': acc,
                    'correct': acc >= 0.98,
                    'hit_max': hit_max,
                    'task': None,
                }
                if is_mixed and task_ids is not None:
                    tid = task_ids[b].item()
                    record['task'] = ALL_TASKS[tid] if tid < len(ALL_TASKS) else f'task_{tid}'

                all_traces.append(record)

    return all_traces


def _fmt_path(path):
    """Format a path tuple as 'e1 → e2 → e3'"""
    if not path:
        return "(exit)"
    return " → ".join(str(e) for e in path)


def print_report(traces, pool_size):
    """Print analysis report."""
    n_hit_max = sum(1 for t in traces if t['hit_max'])
    clean = [t for t in traces if not t['hit_max']]
    depths = [t['depth'] for t in clean]

    # Collect paths only from clean (non-max_hops) samples
    all_paths = []
    for t in clean:
        all_paths.extend(t['paths'])

    print("=" * 80)
    print("POOL OF EXPERTS — ROUTING ANALYSIS")
    print("=" * 80)

    print(f"\n  Total samples: {len(traces)}")
    print(f"  Hit max_hops:  {n_hit_max} ({n_hit_max/len(traces)*100:.1f}%) — excluded from analysis below")
    print(f"  Clean samples: {len(clean)}")

    # Overall depth stats
    print(f"\n## Depth Statistics (n={len(clean)} samples)")
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
        pct = count / len(clean) * 100
        bar = "█" * int(pct / 2)
        print(f"  {d:>3} hops: {count:>6} ({pct:>5.1f}%) {bar}")

    # Per-task depth breakdown
    tasks = set(t['task'] for t in clean if t['task'] is not None)
    if tasks:
        print(f"\n## Per-Task Depth")
        print(f"  {'Task':<20} {'Mean':>6} {'Median':>7} {'Std':>6} {'Min':>4} {'Max':>4}")
        print(f"  {'-'*50}")
        for task in sorted(tasks):
            task_depths = [t['depth'] for t in clean if t['task'] == task]
            print(f"  {task:<20} {np.mean(task_depths):>6.2f} {np.median(task_depths):>7.1f} "
                  f"{np.std(task_depths):>6.2f} {min(task_depths):>4} {max(task_depths):>4}")

    # Expert utilization
    expert_counts = Counter()
    for path in all_paths:
        for e in path:
            expert_counts[e] += 1
    total_expert_uses = sum(expert_counts.values())

    print(f"\n## Expert Utilization (total selections: {total_expert_uses})")
    for eid in range(pool_size):
        count = expert_counts.get(eid, 0)
        pct = count / total_expert_uses * 100 if total_expert_uses > 0 else 0
        bar = "█" * int(pct)
        print(f"  Expert {eid:>2}: {count:>7} ({pct:>5.1f}%) {bar}")

    # Hot paths
    path_counts = Counter(all_paths)
    n_total_paths = len(all_paths)
    print(f"\n## Hot Paths (top 20 of {len(path_counts)} unique, {n_total_paths} total)")
    print(f"  {'Rank':>4} {'Count':>7} {'Pct':>6} {'Path'}")
    print(f"  {'-'*60}")
    for rank, (path, count) in enumerate(path_counts.most_common(20), 1):
        pct = count / n_total_paths * 100
        print(f"  {rank:>4} {count:>7} {pct:>5.1f}% {_fmt_path(path)}")

    # Per-task hot paths
    if tasks:
        for task in sorted(tasks):
            task_paths = []
            for t in clean:
                if t['task'] == task:
                    task_paths.extend(t['paths'])
            task_path_counts = Counter(task_paths)
            print(f"\n## Hot Paths — {task} (top 10 of {len(task_path_counts)} unique)")
            print(f"  {'Rank':>4} {'Count':>7} {'Pct':>6} {'Path'}")
            for rank, (path, count) in enumerate(task_path_counts.most_common(10), 1):
                pct = count / len(task_paths) * 100
                print(f"  {rank:>4} {count:>7} {pct:>5.1f}% {_fmt_path(path)}")

    # Depth vs accuracy
    print(f"\n## Depth vs Accuracy")
    depth_accs = defaultdict(list)
    for t in clean:
        depth_accs[t['depth']].append(t['acc'])
    print(f"  {'Depth':>5} {'Count':>7} {'Mean Acc':>9} {'Std':>6}")
    print(f"  {'-'*30}")
    for d in sorted(depth_accs.keys()):
        accs = depth_accs[d]
        print(f"  {d:>5} {len(accs):>7} {np.mean(accs):>8.1%} {np.std(accs):>6.3f}")

    # Correct vs incorrect path comparison
    correct_paths = Counter()
    incorrect_paths = Counter()
    n_correct = 0
    n_incorrect = 0
    for t in clean:
        if t['correct']:
            n_correct += 1
            for p in t['paths']:
                correct_paths[p] += 1
        else:
            n_incorrect += 1
            for p in t['paths']:
                incorrect_paths[p] += 1

    print(f"\n## Correct vs Incorrect Samples")
    print(f"  Correct:   {n_correct} ({n_correct/len(clean)*100:.1f}%)")
    print(f"  Incorrect: {n_incorrect} ({n_incorrect/len(clean)*100:.1f}%)")

    if n_incorrect > 0:
        print(f"\n  Top 10 paths for INCORRECT samples:")
        print(f"  {'Rank':>4} {'Count':>7} {'Pct':>6} {'Path'}")
        total_inc = sum(incorrect_paths.values())
        for rank, (path, count) in enumerate(incorrect_paths.most_common(10), 1):
            pct = count / total_inc * 100
            print(f"  {rank:>4} {count:>7} {pct:>5.1f}% {_fmt_path(path)}")

    # Per-task depth and accuracy
    if tasks:
        print(f"\n## Per-Task Depth & Accuracy")
        print(f"  {'Task':<20} {'Mean Depth':>10} {'Mean Acc':>9} {'Correct%':>9}")
        print(f"  {'-'*52}")
        task_stats = {}
        for task in sorted(tasks):
            task_traces = [t for t in clean if t['task'] == task]
            mean_d = np.mean([t['depth'] for t in task_traces])
            mean_a = np.mean([t['acc'] for t in task_traces])
            pct_correct = sum(1 for t in task_traces if t['correct']) / len(task_traces) * 100
            task_stats[task] = (mean_d, mean_a, pct_correct)
            print(f"  {task:<20} {mean_d:>10.2f} {mean_a:>8.1%} {pct_correct:>8.1f}%")

        print(f"\n  Ranked by mean depth (deepest first):")
        for task, (mean_d, mean_a, pct_c) in sorted(task_stats.items(), key=lambda x: x[1][0], reverse=True):
            print(f"    {task:<20} {mean_d:.2f} hops, {mean_a:.1%} acc, {pct_c:.1f}% correct")


def main():
    parser = argparse.ArgumentParser(description='Analyze PoE routing behavior')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to saved .pt checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on')
    parser.add_argument('--rerun', action='store_true', help='Force rerun even if cached traces exist')
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    print(f"Model: {ckpt['model_name']}, Task: {ckpt['task']}, "
          f"Params: {ckpt['params']:,}, dim={ckpt['dim']}, layers={ckpt['n_layers']}")
    print(f"Result: acc={ckpt['result']['acc']:.1%}, stop={ckpt['result']['stop_reason']}")

    model, embed, head = rebuild_model(ckpt, device=args.device)
    print(f"Pool size: {model.pool_size}, top_k: {model.top_k}, max_hops: {model.max_hops}")

    task = ckpt['task']

    # Cache traces next to checkpoint
    trace_cache = Path(args.checkpoint).with_suffix('.traces.pt')
    if trace_cache.exists() and not args.rerun:
        print(f"Loading cached traces: {trace_cache}")
        traces = torch.load(trace_cache, weights_only=False)
    else:
        # Find the bench data cache from checkpoint args
        bench_args = ckpt['args']
        B = bench_args['batch_size']
        L = bench_args['seq_len']
        seed = bench_args['seed']
        n_train = bench_args['train_batches']
        n_val = bench_args['val_batches']
        if task == 'mixed':
            n_train *= len(ALL_TASKS)
            n_val *= len(ALL_TASKS)
        key = _cache_key(task, n_train + n_val, B, L, seed)
        data_path = CACHE_DIR / f"{task}_{key}.pt"

        if not data_path.exists():
            print(f"ERROR: bench data cache not found: {data_path}")
            print(f"Run bench_ssm.py first to generate the data.")
            sys.exit(1)

        print(f"Loading data: {data_path}")
        cached = torch.load(data_path, weights_only=True)
        def _to(t):
            return t.to(args.device, non_blocking=True) if isinstance(t, torch.Tensor) else t
        task_data = {
            'train': [tuple(_to(t) for t in batch) for batch in cached['train']],
            'val': [tuple(_to(t) for t in batch) for batch in cached['val']],
        }

        n_total = len(task_data['train']) + len(task_data['val'])
        print(f"Running {n_total} batches (train + val) with tracing...\n")
        traces = analyze(model, embed, head, task_data, task, ckpt['dim'], args.device)

        print(f"Caching traces → {trace_cache}")
        torch.save(traces, trace_cache)

    print_report(traces, model.pool_size)


if __name__ == '__main__':
    main()
