"""
benchmark/harness.py

Benchmark harness for LLM Code Graph Compiler.

Runs each task through the full compiler pipeline, executes the emitted code,
checks success criteria, and records results. Also runs a direct GPT-4o baseline
for comparison.

Usage:
    python benchmark/harness.py --tasks benchmark/tasks.json --output benchmark/results.json
    python benchmark/harness.py --tasks benchmark/tasks.json --output benchmark/results.json --skip-baseline
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

# --- Allow imports from project root ---
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.planner import get_plan
from core.validator import validate_plan
from core.compiler import compile_output
from benchmark.criteria import check_criteria
from benchmark.baseline import run_baseline


# ─────────────────────────────────────────────────────────────────────
# Result structure
# ─────────────────────────────────────────────────────────────────────

def empty_result(task_id: str, description: str) -> dict:
    return {
        "task_id": task_id,
        "description": description,

        # Compiler pipeline stages
        "plan_success": False,
        "validation_success": False,
        "compile_success": False,
        "run_success": False,
        "criteria_passed": False,

        # Final verdict
        "first_pass_success": False,

        # Detail
        "plan": None,
        "validation_errors": [],
        "compile_error": None,
        "run_stdout": "",
        "run_stderr": "",
        "run_returncode": None,
        "criteria_failures": [],
        "duration_seconds": None,
        "error": None,

        # Baseline (filled separately)
        "baseline_success": None,
        "baseline_error": None,
        "baseline_duration_seconds": None,
    }


# ─────────────────────────────────────────────────────────────────────
# Single task runner
# ─────────────────────────────────────────────────────────────────────

def run_task(task: dict, skip_baseline: bool = False) -> dict:
    result = empty_result(task["task_id"], task["description"])
    start = time.time()

    with tempfile.TemporaryDirectory() as run_dir:
        try:
            # ── Stage 1: Plan ──────────────────────────────────────────
            plan = get_plan(task["description"])
            result["plan"] = plan
            result["plan_success"] = True

        except Exception as e:
            result["error"] = f"Planner error: {e}"
            result["duration_seconds"] = round(time.time() - start, 2)
            return result

        try:
            # ── Stage 2: Validate ──────────────────────────────────────
            ok, errors = validate_plan(plan)
            result["validation_errors"] = errors
            result["validation_success"] = ok

            if not ok:
                result["duration_seconds"] = round(time.time() - start, 2)
                return result

        except Exception as e:
            result["error"] = f"Validator error: {traceback.format_exc()}"
            result["duration_seconds"] = round(time.time() - start, 2)
            return result

        try:
            # ── Stage 3: Compile ───────────────────────────────────────
            code = compile_output(plan)
            app_path = os.path.join(run_dir, "app.py")
            with open(app_path, "w") as f:
                f.write(code)
            result["compile_success"] = True

        except Exception as e:
            result["compile_error"] = str(e)
            result["duration_seconds"] = round(time.time() - start, 2)
            return result

        try:
            # ── Stage 4: Copy fixture data into run dir ────────────────
            fixtures = task.get("fixtures", {})
            for dest_name, src_path in fixtures.items():
                import shutil
                shutil.copy(src_path, os.path.join(run_dir, dest_name))

            # ── Stage 5: Execute ───────────────────────────────────────
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
                result["run_stdout"] = stdout
                result["run_stderr"] = stderr
                if task.get("timeout_is_expected"):
                    # e.g. Flask server — times out by design, check criteria on files written before block
                    result["run_success"] = True
                    result["run_returncode"] = None
                else:
                    result["error"] = "Execution timed out"
                    result["duration_seconds"] = round(time.time() - start, 2)
                    return result

            result["run_stdout"] = stdout
            result["run_stderr"] = stderr
            result["run_returncode"] = proc.returncode
            result["run_success"] = proc.returncode == 0

            if not result["run_success"]:
                result["duration_seconds"] = round(time.time() - start, 2)
                return result

        except subprocess.TimeoutExpired:
            result["error"] = "Execution timed out"
            result["duration_seconds"] = round(time.time() - start, 2)
            return result

        except Exception as e:
            result["error"] = f"Execution error: {e}"
            result["duration_seconds"] = round(time.time() - start, 2)
            return result

        # ── Stage 6: Criteria check ────────────────────────────────────
        criteria = task.get("success_criteria", [])
        if criteria:
            passed, failures = check_criteria(criteria, result["run_stdout"], run_dir)
            result["criteria_passed"] = passed
            result["criteria_failures"] = failures
        else:
            # No criteria defined — run success is enough
            result["criteria_passed"] = True

    result["first_pass_success"] = (
        result["plan_success"]
        and result["validation_success"]
        and result["compile_success"]
        and result["run_success"]
        and result["criteria_passed"]
    )

    result["duration_seconds"] = round(time.time() - start, 2)

    # ── Baseline ───────────────────────────────────────────────────────
    if not skip_baseline:
        b_start = time.time()
        try:
            b_success, b_error = run_baseline(task)
            result["baseline_success"] = b_success
            result["baseline_error"] = b_error
        except Exception as e:
            result["baseline_success"] = False
            result["baseline_error"] = str(e)
        result["baseline_duration_seconds"] = round(time.time() - b_start, 2)

    return result


# ─────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict], skip_baseline: bool):
    n = len(results)
    compiler_passes = sum(1 for r in results if r["first_pass_success"])

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Tasks run:              {n}")
    print(f"Compiler first-pass:    {compiler_passes}/{n} ({100*compiler_passes//n}%)")

    if not skip_baseline:
        baseline_passes = sum(1 for r in results if r.get("baseline_success"))
        print(f"Baseline first-pass:    {baseline_passes}/{n} ({100*baseline_passes//n}%)")

    print()
    print(f"{'ID':<20} {'Compiler':<12} {'Baseline':<12} Stage Failed")
    print("-" * 60)
    for r in results:
        compiler = "✓ PASS" if r["first_pass_success"] else "✗ FAIL"
        baseline = ("✓ PASS" if r.get("baseline_success") else "✗ FAIL") if not skip_baseline else "—"

        # Identify first failed stage
        if not r["plan_success"]:
            stage = "planner"
        elif not r["validation_success"]:
            stage = f"validator ({len(r['validation_errors'])} errors)"
        elif not r["compile_success"]:
            stage = "compiler"
        elif not r["run_success"]:
            stage = f"runtime (exit {r['run_returncode']})"
        elif not r["criteria_passed"]:
            stage = f"criteria ({len(r['criteria_failures'])} failures)"
        else:
            stage = "—"

        print(f"{r['task_id']:<20} {compiler:<12} {baseline:<12} {stage}")

    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Code Graph Compiler Benchmark Harness")
    parser.add_argument("--tasks", required=True, help="Path to tasks.json")
    parser.add_argument("--output", default="benchmark/results.json", help="Path to write results JSON")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip GPT-4o baseline runs")
    parser.add_argument("--task-id", help="Run a single task by ID (for debugging)")
    args = parser.parse_args()

    with open(args.tasks) as f:
        tasks = json.load(f)

    if args.task_id:
        tasks = [t for t in tasks if t["task_id"] == args.task_id]
        if not tasks:
            print(f"No task found with id '{args.task_id}'")
            sys.exit(1)

    results = []
    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] Running: {task['task_id']}")
        print(f"  {task['description']}")
        result = run_task(task, skip_baseline=True)

        status = "✓ PASS" if result["first_pass_success"] else "✗ FAIL"
        print(f"  → {status} ({result['duration_seconds']}s)")
        if result.get("error"):
            print(f"  ! {result['error']}")
        if result.get("criteria_failures"):
            for f in result["criteria_failures"]:
                print(f"  ! {f}")

        results.append(result)

        if i < len(tasks) - 1:
            time.sleep(5)

    print_summary(results, args.skip_baseline)

    # Write results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "run_at": datetime.utcnow().isoformat(),
        "task_count": len(results),
        "compiler_first_pass_rate": sum(1 for r in results if r["first_pass_success"]) / len(results),
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()