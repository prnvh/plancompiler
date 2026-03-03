"""
benchmark/consistency_harness.py

Deterministic consistency measurement for the LLM Code Graph Compiler benchmark.

Runs each task N times (default 5) and measures:
    - outcome_consistency   — did the same task always pass or always fail?
    - output_hash_match     — did output files produce identical content across runs?
    - compiler_stability    — did the compiler produce identical code across runs?

Usage:
    # Run all tasks 5 times
    python benchmark/consistency_harness.py --tasks benchmark/tasks.json --runs 5

    # Single task
    python benchmark/consistency_harness.py --tasks benchmark/tasks.json --task-id csv_passthrough_export --runs 5

    # Write detailed results
    python benchmark/consistency_harness.py --tasks benchmark/tasks.json --output benchmark/consistency_results.json

Core claim for the research paper:
    If the compiler produces identical outcomes across all N runs while the baseline
    fluctuates (varies between pass/fail or produces different outputs), this is strong
    evidence that typed DAG constraints eliminate non-determinism from the synthesis step.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.planner import get_plan
from core.validator import validate_plan
from core.compiler import compile_output
from benchmark.criteria import check_criteria


def _task_deadline_exceeded(start: float, task: dict) -> bool:
    """Hard wall-clock guard to avoid apparent indefinite hangs."""
    hard_limit = task.get("max_task_duration_seconds")
    if hard_limit is None:
        hard_limit = task.get("timeout_seconds", 30) + 120
    return (time.time() - start) > hard_limit


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _hash_file(path: str) -> str | None:
    """SHA-256 of a file's content. Returns None if file does not exist."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _hash_string(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _run_single(task: dict) -> dict:
    """
    Runs the full compiler pipeline once for a task.

    Returns a result dict with:
        success         bool
        stage_failed    str | None
        compiled_code   str | None     (for compiler stability check)
        output_hashes   dict           {filename: sha256} for all criteria files
        error           str | None
    """
    result = {
        "success": False,
        "stage_failed": None,
        "compiled_code": None,
        "output_hashes": {},
        "error": None,
    }

    with tempfile.TemporaryDirectory() as run_dir:
        # ── Plan ──────────────────────────────────────────────────
        try:
            plan = get_plan(task["description"])
        except Exception as e:
            result["stage_failed"] = "planner"
            result["error"] = str(e)
            return result

        # ── Validate ──────────────────────────────────────────────
        try:
            ok, errors = validate_plan(plan)
            if not ok:
                result["stage_failed"] = "validator"
                result["error"] = "; ".join(errors)
                return result
        except Exception as e:
            result["stage_failed"] = "validator"
            result["error"] = str(e)
            return result

        # ── Compile ───────────────────────────────────────────────
        try:
            code = compile_output(plan)
            result["compiled_code"] = code
            app_path = os.path.join(run_dir, "app.py")
            with open(app_path, "w") as f:
                f.write(code)
        except Exception as e:
            result["stage_failed"] = "compiler"
            result["error"] = str(e)
            return result

        # ── Copy fixtures ─────────────────────────────────────────
        for dest_name, src_path in task.get("fixtures", {}).items():
            try:
                shutil.copy(src_path, os.path.join(run_dir, dest_name))
            except Exception as e:
                result["stage_failed"] = "fixture"
                result["error"] = f"Could not copy fixture '{src_path}': {e}"
                return result

        # ── Execute ───────────────────────────────────────────────
        try:
            proc = subprocess.Popen(
                [sys.executable, app_path],
                cwd=run_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=task.get("timeout_seconds", 30))
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                result["stage_failed"] = "timeout"
                result["error"] = "Execution timed out"
                return result

            if proc.returncode != 0:
                result["stage_failed"] = "runtime"
                result["error"] = f"Exit {proc.returncode}: {stderr[:300]}"
                return result

        except Exception as e:
            result["stage_failed"] = "runtime"
            result["error"] = str(e)
            return result

        # ── Criteria ──────────────────────────────────────────────
        criteria = task.get("success_criteria", [])
        if criteria:
            passed, failures = check_criteria(criteria, stdout, run_dir)
            if not passed:
                result["stage_failed"] = "criteria"
                result["error"] = "; ".join(failures)
                return result

        # ── Hash output files for consistency check ────────────────
        for c in criteria:
            if "path" in c:
                fpath = os.path.join(run_dir, c["path"])
                result["output_hashes"][c["path"]] = _hash_file(fpath)

        result["success"] = True

    return result


# ─────────────────────────────────────────────────────────────────────
# Task consistency runner
# ─────────────────────────────────────────────────────────────────────

def run_consistency(task: dict, n_runs: int, delay_seconds: float = 3.0) -> dict:
    """
    Runs one task N times and computes consistency metrics.

    Returns:
        {
          task_id, description,
          runs: [...],             # per-run results
          outcome_consistency,     # fraction of runs with same success/fail
          output_hash_consistent,  # bool: all successful runs produced identical outputs
          compiler_code_consistent,# bool: all runs produced identical compiled code
          pass_rate,               # fraction of runs that passed
          notes: [...]             # human-readable observations
        }
    """
    task_id = task["task_id"]
    print(f"\n  Running {n_runs}x: {task_id}")

    runs = []
    for i in range(n_runs):
        if i > 0:
            time.sleep(delay_seconds)
        r = _run_single(task)
        status = "✓" if r["success"] else f"✗ ({r['stage_failed']})"
        print(f"    run {i+1}/{n_runs}: {status}")
        runs.append(r)

    # ── Metrics ───────────────────────────────────────────────────

    outcomes = [r["success"] for r in runs]
    pass_rate = sum(outcomes) / n_runs

    # Outcome consistency: all same OR none of them differ
    outcome_consistency = all(o == outcomes[0] for o in outcomes)

    # Compiler code consistency: all compiled code (when available) identical
    codes = [r["compiled_code"] for r in runs if r["compiled_code"] is not None]
    compiler_code_consistent = len(set(_hash_string(c) for c in codes)) <= 1 if codes else True

    # Output hash consistency: across successful runs, output files must be identical
    successful_runs = [r for r in runs if r["success"]]
    if len(successful_runs) >= 2:
        # For each output file, check all hashes match
        all_files = set()
        for r in successful_runs:
            all_files.update(r["output_hashes"].keys())

        hash_consistent = True
        inconsistent_files = []
        for fname in all_files:
            file_hashes = [r["output_hashes"].get(fname) for r in successful_runs]
            if len(set(h for h in file_hashes if h is not None)) > 1:
                hash_consistent = False
                inconsistent_files.append(fname)
        output_hash_consistent = hash_consistent
    else:
        output_hash_consistent = True  # can't measure with < 2 successes
        inconsistent_files = []

    # ── Notes ─────────────────────────────────────────────────────
    notes = []
    if not outcome_consistency:
        n_pass = sum(outcomes)
        n_fail = n_runs - n_pass
        notes.append(f"INCONSISTENT OUTCOME: {n_pass} pass, {n_fail} fail across {n_runs} runs")

    if not compiler_code_consistent:
        notes.append("INCONSISTENT COMPILED CODE: planner produced different graphs across runs")

    if not output_hash_consistent:
        notes.append(f"INCONSISTENT OUTPUT: differing file content in {inconsistent_files}")

    if not notes:
        if pass_rate == 1.0:
            notes.append("STABLE PASS: identical outcome and output across all runs")
        elif pass_rate == 0.0:
            notes.append("STABLE FAIL: consistent failure across all runs")

    # Stage distribution for failing runs
    failing_stages = [r["stage_failed"] for r in runs if not r["success"]]
    if failing_stages:
        stage_dist = dict(Counter(failing_stages))
        notes.append(f"Failure stages: {stage_dist}")

    return {
        "task_id":                   task_id,
        "description":               task["description"],
        "n_runs":                    n_runs,
        "pass_rate":                 pass_rate,
        "outcome_consistency":       outcome_consistency,
        "output_hash_consistent":    output_hash_consistent,
        "compiler_code_consistent":  compiler_code_consistent,
        "inconsistent_files":        inconsistent_files,
        "notes":                     notes,
        "runs":                      runs,
    }


# ─────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────

def print_consistency_summary(results: list[dict], n_runs: int) -> None:
    n = len(results)
    consistent = sum(1 for r in results if r["outcome_consistency"])
    stable_pass = sum(1 for r in results if r["pass_rate"] == 1.0)
    stable_fail = sum(1 for r in results if r["pass_rate"] == 0.0)
    flaky       = sum(1 for r in results if 0 < r["pass_rate"] < 1.0)

    print("\n" + "=" * 70)
    print(f"CONSISTENCY REPORT  ({n} tasks, {n_runs} runs each)")
    print("=" * 70)
    print(f"  Outcome consistent tasks:  {consistent}/{n}  ({100*consistent//n}%)")
    print(f"    └─ Stable PASS:          {stable_pass}")
    print(f"    └─ Stable FAIL:          {stable_fail}")
    print(f"  Flaky tasks (mixed):       {flaky}")
    print()
    print(f"  {'Task ID':<35} {'Pass Rate':<12} {'Consistent':<12} Notes")
    print("  " + "-" * 68)
    for r in results:
        rate_str    = f"{r['pass_rate']*100:.0f}%"
        consist_str = "✓" if r["outcome_consistency"] else "✗ FLAKY"
        note        = r["notes"][0][:40] if r["notes"] else ""
        print(f"  {r['task_id']:<35} {rate_str:<12} {consist_str:<12} {note}")
    print("=" * 70)
    print()
    print("RESEARCH INTERPRETATION:")
    print(f"  If compiler shows {consistent}/{n} consistent outcomes")
    print(f"  while baseline fluctuates → confirms typed constraints eliminate non-determinism.")
    print(f"  Flaky compiler tasks ({flaky}) indicate planner variance, NOT compiler instability.")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Consistency harness: run each task N times to measure determinism"
    )
    parser.add_argument("--tasks",   required=True, help="Path to tasks.json")
    parser.add_argument("--runs",    type=int, default=1, help="Number of runs per task (default: 1)")
    parser.add_argument("--output",  default="benchmark/consistency_results.json")
    parser.add_argument("--task-id", help="Run a single task by ID")
    parser.add_argument("--delay",   type=float, default=3.0,
                        help="Seconds between runs to avoid rate limiting (default: 3)")
    args = parser.parse_args()

    with open(args.tasks) as f:
        tasks = json.load(f)

    if args.task_id:
        tasks = [t for t in tasks if t["task_id"] == args.task_id]
        if not tasks:
            print(f"No task found: '{args.task_id}'")
            sys.exit(1)

    print(f"\nConsistency Harness — {len(tasks)} tasks × {args.runs} runs")
    print(f"Output: {args.output}")
    print("=" * 70)

    all_results = []
    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task['task_id']}")
        result = run_consistency(task, n_runs=args.runs, delay_seconds=args.delay)
        all_results.append(result)

        # Save incrementally (crash-safe)
        output = {
            "run_at":    datetime.utcnow().isoformat(),
            "n_runs":    args.runs,
            "task_count": len(all_results),
            "results":   all_results,
        }
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

    print_consistency_summary(all_results, args.runs)
    print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()