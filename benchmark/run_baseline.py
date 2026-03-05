"""
benchmark/run_baseline.py

Runs GPT-4o baseline separately from the compiler benchmark.
Run AFTER the compiler benchmark completes:

    python benchmark/run_baseline.py --results benchmark/results.json \\
                                     --tasks benchmark/tasks.json

Updates results.json in-place with baseline fields.
Each task is run N_RUNS times; baseline_success is True only if ALL runs pass.
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.baseline import run_baseline

# Must match harness.py
N_RUNS = 5


# ─────────────────────────────────────────────────────────────────────
# Hard per-task timeout wrapper
# ─────────────────────────────────────────────────────────────────────

def run_baseline_with_hard_timeout(
    task: dict,
    hard_timeout: int = 120,
) -> tuple[bool, str | None]:
    """
    Runs run_baseline(task) on a daemon thread.

    If it doesn't complete within hard_timeout seconds the thread is
    abandoned (daemon=True so it won't block process exit) and we return
    a timeout failure. This is the final safety net — baseline.py already
    kills the subprocess process group, but this catches any edge case
    where something in Python itself blocks (e.g. a stuck tempfile cleanup).
    """
    result: dict = {"success": False, "error": "Hard timeout — task never completed", "done": False}

    def _target():
        try:
            success, error = run_baseline(task)
            result["success"] = success
            result["error"] = error
            result["done"] = True
        except Exception as e:
            result["error"] = f"Unhandled exception in run_baseline: {e}"
            result["done"] = True

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=hard_timeout)

    if not result["done"]:
        print(f"  [!] Hard timeout ({hard_timeout}s) hit — abandoning task")

    return result["success"], result["error"]


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run GPT-4o baseline and merge results into existing results.json"
    )
    parser.add_argument("--results", required=True, help="Path to existing results.json from compiler run")
    parser.add_argument("--tasks",   required=True, help="Path to tasks.json")
    parser.add_argument(
        "--hard-timeout", type=int, default=120,
        help="Max seconds to wait for a single baseline run before abandoning (default: 120)"
    )
    parser.add_argument(
        "--task-id", help="Run baseline for a single task only (for debugging)"
    )
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    from glob import glob

    tasks = []
    task_path = Path(args.tasks)

    if task_path.is_dir():
        for file in glob(str(task_path / "*.json")):
            with open(file) as f:
                tasks.extend(json.load(f))
    else:
        with open(task_path) as f:
            tasks = json.load(f)

    task_map = {t["task_id"]: t for t in tasks}
    results  = data["results"]

    # Filter to a single task if requested
    if args.task_id:
        results = [r for r in results if r["task_id"] == args.task_id]
        if not results:
            print(f"No result found with task_id '{args.task_id}'")
            sys.exit(1)

    n = len(results)
    print(f"\nRunning baseline for {n} task(s) — GPT-4o direct generation")
    print(f"Runs per task:         {N_RUNS}")
    print(f"Hard timeout per run:  {args.hard_timeout}s")
    print("=" * 60)

    for i, result in enumerate(results):
        task_id = result["task_id"]
        task    = task_map.get(task_id)

        if not task:
            print(f"\n[{i+1}/{n}] SKIP: no task definition found for '{task_id}'")
            continue

        print(f"\n[{i+1}/{n}] Baseline: {task_id} ({N_RUNS} runs)")
        print(f"  {task['description'][:100]}")

        if task.get("skip_baseline"):
            print("  → SKIPPED (skip_baseline flag set)")
            result["baseline_success"] = None
            result["baseline_error"]   = "skipped"
            result["baseline_duration_seconds"] = 0
            result["baseline_pass_count"] = None
            result["baseline_run_count"]  = N_RUNS
            result["baseline_runs"]       = []
            continue

        # ── Run N_RUNS times ──────────────────────────────────────────
        baseline_runs = []
        total_duration = 0.0

        for run_idx in range(N_RUNS):
            print(f"    run {run_idx + 1}/{N_RUNS} ...", end=" ", flush=True)
            b_start = time.time()
            b_success, b_error = run_baseline_with_hard_timeout(task, args.hard_timeout)
            b_duration = round(time.time() - b_start, 2)
            total_duration += b_duration

            status = "✓" if b_success else "✗"
            print(f"{status} ({b_duration}s)")
            if b_error and not b_success:
                print(f"      ! {str(b_error)[:120]}")

            baseline_runs.append({
                "run": run_idx + 1,
                "success": b_success,
                "error": b_error,
                "duration_seconds": b_duration,
            })

            # Small delay between LLM calls
            if run_idx < N_RUNS - 1:
                time.sleep(2)

        baseline_pass_count = sum(1 for r in baseline_runs if r["success"])
        baseline_success    = (baseline_pass_count == N_RUNS)

        # Surface the last run's error for the top-level field (best effort)
        last_error = next(
            (r["error"] for r in reversed(baseline_runs) if r["error"] and not r["success"]),
            None,
        )

        result["baseline_success"]          = baseline_success
        result["baseline_error"]            = last_error
        result["baseline_duration_seconds"] = round(total_duration, 2)
        result["baseline_pass_count"]       = baseline_pass_count
        result["baseline_run_count"]        = N_RUNS
        result["baseline_runs"]             = baseline_runs

        overall = "✓ PASS" if baseline_success else "✗ FAIL"
        print(f"  → {overall} ({baseline_pass_count}/{N_RUNS} runs passed)")

        # Write after every task — a future hang won't lose completed results
        all_results = data["results"]
        data["baseline_first_pass_rate"] = (
            sum(1 for r in all_results if r.get("baseline_success")) / len(all_results)
        )
        with open(args.results, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  [saved]")

        if i < n - 1:
            time.sleep(2)

    # ── Final summary ──────────────────────────────────────────────────
    all_results     = data["results"]
    total           = len(all_results)
    baseline_passes = sum(1 for r in all_results if r.get("baseline_success"))
    compiler_passes = sum(1 for r in all_results if r.get("first_pass_success"))

    print(f"\n{'=' * 60}")
    print(f"Runs per task:        {N_RUNS} (task passes only if all {N_RUNS} pass)")
    print(f"Compiler first-pass:  {compiler_passes}/{total} ({100 * compiler_passes // total}%)")
    print(f"Baseline first-pass:  {baseline_passes}/{total} ({100 * baseline_passes // total}%)")
    print(f"{'=' * 60}")
    print(f"\nResults updated in {args.results}")


if __name__ == "__main__":
    main()