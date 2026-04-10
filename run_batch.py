"""
Batch experiment runner.
Run all 18 experiment configurations with progress tracking.
"""

import subprocess
import sys
import time
import os
from datetime import datetime

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Experiment configuration
EXPERIMENTS = [
    # Qwen2.5-Math-1.5B
    {"model": "qwen2.5-math-1.5b", "dataset": "gsm8k", "method": "cot", "id": 1},
    {"model": "qwen2.5-math-1.5b", "dataset": "gsm8k", "method": "self_refine", "id": 2},
    {"model": "qwen2.5-math-1.5b", "dataset": "gsm8k", "method": "self_consistency", "id": 3},
    {"model": "qwen2.5-math-1.5b", "dataset": "math500", "method": "cot", "id": 4},
    {"model": "qwen2.5-math-1.5b", "dataset": "math500", "method": "self_refine", "id": 5},
    {"model": "qwen2.5-math-1.5b", "dataset": "math500", "method": "self_consistency", "id": 6},
    {"model": "qwen2.5-math-1.5b", "dataset": "aime2024", "method": "cot", "id": 7},
    {"model": "qwen2.5-math-1.5b", "dataset": "aime2024", "method": "self_refine", "id": 8},
    {"model": "qwen2.5-math-1.5b", "dataset": "aime2024", "method": "self_consistency", "id": 9},

    # DeepSeek-R1-Qwen-1.5B
    {"model": "deepseek-r1-qwen-1.5b", "dataset": "gsm8k", "method": "cot", "id": 10},
    {"model": "deepseek-r1-qwen-1.5b", "dataset": "gsm8k", "method": "self_refine", "id": 11},
    {"model": "deepseek-r1-qwen-1.5b", "dataset": "gsm8k", "method": "self_consistency", "id": 12},
    {"model": "deepseek-r1-qwen-1.5b", "dataset": "math500", "method": "cot", "id": 13},
    {"model": "deepseek-r1-qwen-1.5b", "dataset": "math500", "method": "self_refine", "id": 14},
    {"model": "deepseek-r1-qwen-1.5b", "dataset": "math500", "method": "self_consistency", "id": 15},
    {"model": "deepseek-r1-qwen-1.5b", "dataset": "aime2024", "method": "cot", "id": 16},
    {"model": "deepseek-r1-qwen-1.5b", "dataset": "aime2024", "method": "self_refine", "id": 17},
    {"model": "deepseek-r1-qwen-1.5b", "dataset": "aime2024", "method": "self_consistency", "id": 18},
]


def run_experiment(model, dataset, method, exp_id):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment {exp_id}/18: {model} + {dataset} + {method}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable,
        "experiments/runner.py",
        "--model", model,
        "--dataset", dataset,
        "--method", method,
    ]

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        elapsed = time.time() - start_time
        print(f"\nExperiment {exp_id} completed in {elapsed/60:.1f} minutes")
        if result.stdout:
            # Print last 1000 chars of output
            print(result.stdout[-1000:])
        return True, None
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\nExperiment {exp_id} FAILED after {elapsed/60:.1f} minutes")
        if e.stdout:
            print("STDOUT:", e.stdout[-500:])
        if e.stderr:
            print("STDERR:", e.stderr[-500:])
        return False, {"model": model, "dataset": dataset, "method": method, "id": exp_id}


def run_all():
    """Run all 18 experiments."""
    print("="*70)
    print("Starting all 18 experiments")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.executable}")
    print("="*70)

    success_count = 0
    failed_experiments = []

    for exp in EXPERIMENTS:
        ok, failed = run_experiment(exp["model"], exp["dataset"], exp["method"], exp["id"])
        if ok:
            success_count += 1
        else:
            failed_experiments.append(failed)

    print(f"\n{'='*60}")
    print(f"All experiments finished!")
    print(f"Success: {success_count}/18")
    print(f"Failed: {len(failed_experiments)}")
    if failed_experiments:
        print(f"Failed experiment IDs: {[e['id'] for e in failed_experiments]}")
        print("Rerun failed experiments with:")
        for e in failed_experiments:
            print(f"  python run_batch.py --id {e['id']}")
    print(f"{'='*60}\n")


def run_single_model(model_name):
    """Run only the 9 experiments for a specific model."""
    model_exps = [e for e in EXPERIMENTS if e["model"] == model_name]
    print(f"Running {len(model_exps)} experiments for {model_name}...")
    for exp in model_exps:
        run_experiment(exp["model"], exp["dataset"], exp["method"], exp["id"])


def run_single_dataset(dataset_name):
    """Run only the 6 experiments for a specific dataset."""
    dataset_exps = [e for e in EXPERIMENTS if e["dataset"] == dataset_name]
    print(f"Running {len(dataset_exps)} experiments for {dataset_name}...")
    for exp in dataset_exps:
        run_experiment(exp["model"], exp["dataset"], exp["method"], exp["id"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch experiment runner")
    parser.add_argument("--all", action="store_true", help="Run all 18 experiments")
    parser.add_argument(
        "--model",
        choices=["qwen2.5-math-1.5b", "deepseek-r1-qwen-1.5b"],
        help="Run only the 9 experiments for the specified model"
    )
    parser.add_argument(
        "--dataset",
        choices=["gsm8k", "math500", "aime2024"],
        help="Run only the 6 experiments for the specified dataset"
    )
    parser.add_argument(
        "--id",
        type=int,
        choices=range(1, 19),
        help="Run a single experiment by ID"
    )

    args = parser.parse_args()

    if args.all:
        run_all()
    elif args.model:
        run_single_model(args.model)
    elif args.dataset:
        run_single_dataset(args.dataset)
    elif args.id:
        exp = EXPERIMENTS[args.id - 1]
        run_experiment(exp["model"], exp["dataset"], exp["method"], exp["id"])
    else:
        print("Please specify a run mode: --all, --model, --dataset, or --id")
        print("\nExamples:")
        print("  python run_batch.py --all                    # Run all 18")
        print("  python run_batch.py --model qwen2.5-math-1.5b  # Run Qwen 9")
        print("  python run_batch.py --dataset gsm8k             # Run GSM8K 6")
        print("  python run_batch.py --id 1                      # Run experiment 1")
