#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import subprocess
from pathlib import Path


def run_cmd(cmd, cwd):
    printable = " ".join(shlex.quote(str(c)) for c in cmd)
    print(f"\n[ruler] {printable}\n")
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_taskset(config_tasks_path: Path, taskset: str) -> list[str]:
    if not config_tasks_path.exists():
        raise FileNotFoundError(f"Missing {config_tasks_path}")

    lines = config_tasks_path.read_text().splitlines()
    pattern = re.compile(rf"^{re.escape(taskset)}=\(")
    tasks = []
    inside = False

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not inside:
            if pattern.match(line):
                inside = True
                line = line.split("(", 1)[1]
            else:
                continue

        if inside:
            if ")" in line:
                chunk = line.split(")", 1)[0]
                tasks.extend(chunk.split())
                break
            tasks.extend(line.split())

    if not tasks:
        raise ValueError(f"Taskset '{taskset}' not found in {config_tasks_path}")
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Run NVIDIA RULER evaluation.")
    parser.add_argument("--ruler-path", type=Path, default=Path("external/RULER"))
    parser.add_argument("--benchmark", type=str, default="synthetic")
    parser.add_argument("--taskset", type=str, default="synthetic")
    parser.add_argument("--tasks", type=str, default="", help="Comma-separated task list")
    parser.add_argument("--seq-lengths", type=str, default="4096", help="Comma-separated lengths")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--subset", type=str, default="validation")
    parser.add_argument("--output-dir", type=Path, default=Path("ruler_runs"))

    parser.add_argument("--model-name", type=str, default="local_model")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--tokenizer-type", type=str, default="hf")
    parser.add_argument("--template", type=str, default="base")

    parser.add_argument("--server-type", type=str, default="hf", choices=[
        "hf", "mamba", "vllm", "trtllm", "sglang", "openai", "gemini",
    ])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)

    parser.add_argument("--remove-newline-tab", action="store_true")
    parser.add_argument("--prepare-for-ns", action="store_true")

    args = parser.parse_args()

    ruler_path = args.ruler_path
    if not ruler_path.exists():
        raise FileNotFoundError(
            f"RULER repo not found at {ruler_path}. Clone it first: \n"
            f"  git clone https://github.com/NVIDIA/RULER {ruler_path}"
        )

    scripts_dir = ruler_path / "scripts"
    config_tasks_path = scripts_dir / "config_tasks.sh"

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        tasks = parse_taskset(config_tasks_path, args.taskset)

    seq_lengths = [int(x) for x in args.seq_lengths.split(",") if x.strip()]

    for max_seq_length in seq_lengths:
        results_dir = args.output_dir / args.model_name / args.benchmark / str(max_seq_length)
        data_dir = results_dir / "data"
        pred_dir = results_dir / "pred"
        data_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        for task in tasks:
            prepare_cmd = [
                "python", "data/prepare.py",
                "--save_dir", str(data_dir),
                "--benchmark", args.benchmark,
                "--task", task,
                "--subset", args.subset,
                "--tokenizer_path", args.tokenizer_path,
                "--tokenizer_type", args.tokenizer_type,
                "--max_seq_length", str(max_seq_length),
                "--model_template_type", args.template,
                "--num_samples", str(args.num_samples),
            ]
            if args.remove_newline_tab:
                prepare_cmd.append("--remove_newline_tab")
            if args.prepare_for_ns:
                prepare_cmd.append("--prepare_for_ns")

            run_cmd(prepare_cmd, scripts_dir)

            pred_cmd = [
                "python", "pred/call_api.py",
                "--data_dir", str(data_dir),
                "--save_dir", str(pred_dir),
                "--benchmark", args.benchmark,
                "--task", task,
                "--subset", args.subset,
                "--server_type", args.server_type,
                "--model_name_or_path", args.model_path,
                "--temperature", str(args.temperature),
                "--top_k", str(args.top_k),
                "--top_p", str(args.top_p),
                "--batch_size", str(args.batch_size),
                "--threads", str(args.threads),
            ]
            run_cmd(pred_cmd, scripts_dir)

        eval_cmd = [
            "python", "eval/evaluate.py",
            "--data_dir", str(pred_dir),
            "--benchmark", args.benchmark,
        ]
        run_cmd(eval_cmd, scripts_dir)


if __name__ == "__main__":
    main()
