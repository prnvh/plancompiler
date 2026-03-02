"""
benchmark/run_baseline.py

Runs GPT-4o baseline separately from compiler benchmark.
Run AFTER compiler benchmark completes:
    python benchmark/run_baseline.py --results benchmark/results.json --tasks benchmark/tasks.json

Updates results.json in-place with baseline fields.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.baseline import run_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to existing results.json from compiler run")
    parser.add_argument("--tasks",   required=True, help="Path to tasks.json")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    with open(args.tasks) as f:
        tasks = json.load(f)

    task_map = {t["task_id"]: t for t in tasks}
    results  = data["results"]

    n = len(results)
    print(f"\nRunning baseline for {n} tasks (GPT-4o direct generation)")
    print("="*60)

    for i, result in enumerate(results):
        task_id = result["task_id"]
        task    = task_map.get(task_id)
        if not task:
            print(f"[{i+1}/{n}] SKIP: no task definition found for '{task_id}'")
            continue

        print(f"\n[{i+1}/{n}] Baseline: {task_id}")

        if task.get("skip_baseline"):
            print(f"  → SKIPPED (skip_baseline flag set)")
            result["baseline_success"] = None
            result["baseline_error"] = "skipped"
            result["baseline_duration_seconds"] = 0
            continue

        b_start = time.time()
        try:
            b_success, b_error = run_baseline(task)
            result["baseline_success"] = b_success
            result["baseline_error"]   = b_error
        except Exception as e:
            result["baseline_success"] = False
            result["baseline_error"]   = str(e)

        result["baseline_duration_seconds"] = round(time.time() - b_start, 2)

        status = "✓ PASS" if result["baseline_success"] else "✗ FAIL"
        print(f"  → {status} ({result['baseline_duration_seconds']}s)")
        if result.get("baseline_error"):
            print(f"  ! {result['baseline_error'][:200]}")

        # Save after every task so a hang doesn't lose prior results
        data["baseline_first_pass_rate"] = (
            sum(1 for r in results if r.get("baseline_success")) / n
        )
        with open(args.results, "w") as f:
            json.dump(data, f, indent=2)

        if i < n - 1:
            print(f"  [waiting 20s before next baseline call...]")
            time.sleep(20)

    # Final summary
    baseline_passes  = sum(1 for r in results if r.get("baseline_success"))
    compiler_passes  = sum(1 for r in results if r.get("first_pass_success"))
    print(f"\n{'='*60}")
    print(f"Compiler first-pass:  {compiler_passes}/{n} ({100*compiler_passes//n}%)")
    print(f"Baseline first-pass:  {baseline_passes}/{n} ({100*baseline_passes//n}%)")
    print(f"{'='*60}")
    print(f"\nResults updated in {args.results}")


if __name__ == "__main__":
    main()