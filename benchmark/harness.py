"""
benchmark/harness.py

Benchmark harness for LLM Code Graph Compiler.

Runs each task through the full compiler pipeline, executes the emitted
artifact, checks success criteria, and records results.

Token + latency tracking:
    Planner token counts are captured if get_plan() returns a (plan, usage)
    tuple. If your planner returns only a plan dict (legacy), token fields
    will be None — update core/planner.py to return usage (see note below).

Usage:
    python benchmark/harness.py \\
        --tasks  benchmark/tasks/tasks_set_d.json \\
        --output benchmark/results/results_set_d.json \\
        --skip-baseline

NOTE — planner.py change required for token tracking:
    The harness expects get_plan() to optionally return:
        (plan: dict, usage: dict)   where usage = {
            "input_tokens": int,
            "output_tokens": int,
            "total_tokens": int,
            "cost_usd": float,       # gpt-4o-mini pricing
        }
    If get_plan() returns only a dict, token fields are set to None.
    To enable tracking, update core/planner.py to return the usage dict
    from the OpenAI response alongside the plan. Example:

        usage = response.usage
        plan_usage = {
            "input_tokens":  usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens":  usage.total_tokens,
            "cost_usd": round(
                (usage.prompt_tokens / 1_000_000) * 0.15 +
                (usage.completion_tokens / 1_000_000) * 0.60, 6
            ),  # gpt-4o-mini rates
        }
        return plan, plan_usage
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.planner import get_plan
from core.validator import validate_plan
from core.compiler import compile_output
from benchmark.criteria import check_criteria

N_RUNS = 1

# gpt-4o-mini pricing (planner model)
PLANNER_PRICING = {"input": 0.15, "output": 0.60}  # $/1M tokens


def _planner_cost(input_tokens: int, output_tokens: int) -> float:
    return round(
        (input_tokens / 1_000_000) * PLANNER_PRICING["input"]
        + (output_tokens / 1_000_000) * PLANNER_PRICING["output"],
        6,
    )


# ─────────────────────────────────────────────────────────────────────
# Result structure
# ─────────────────────────────────────────────────────────────────────

def empty_result(task_id: str, description: str) -> dict:
    return {
        "task_id":     task_id,
        "description": description,

        # Compiler pipeline stages (from last run)
        "plan_success":       False,
        "validation_success": False,
        "compile_success":    False,
        "run_success":        False,
        "criteria_passed":    False,

        # Primary metric — True only if ALL N_RUNS passed
        "first_pass_success": False,

        # Multi-run aggregates
        "pass_count": 0,
        "run_count":  N_RUNS,

        # Latency
        "duration_seconds":     None,  # last run wall time (plan→criteria)
        "avg_duration_seconds": None,  # average across N_RUNS

        # Planner token usage (per-run and aggregated)
        # Set to None if planner.py does not return usage dict.
        "planner_input_tokens":  None,
        "planner_output_tokens": None,
        "planner_total_tokens":  None,
        "planner_cost_usd":      None,

        # Detail from last run
        "plan":              None,
        "validation_errors": [],
        "compile_error":     None,
        "run_stdout":        "",
        "run_stderr":        "",
        "run_returncode":    None,
        "criteria_failures": [],
        "error":             None,

        # Per-run breakdown
        "runs": [],

        # Baseline fields (filled by run_baseline.py)
        "baseline_success":          None,
        "baseline_error":            None,
        "baseline_duration_seconds": None,
    }


# ─────────────────────────────────────────────────────────────────────
# Single task runner
# ─────────────────────────────────────────────────────────────────────

def run_task(task: dict) -> dict:
    result = empty_result(task["task_id"], task["description"])
    start  = time.time()

    with tempfile.TemporaryDirectory() as run_dir:

        # ── Stage 1: Plan ──────────────────────────────────────────────
        try:
            plan_result = get_plan(task["description"])

            # Support both legacy (plan only) and new (plan, usage) returns
            if isinstance(plan_result, tuple) and len(plan_result) == 2:
                plan, plan_usage = plan_result
                result["planner_input_tokens"]  = plan_usage.get("input_tokens")
                result["planner_output_tokens"] = plan_usage.get("output_tokens")
                result["planner_total_tokens"]  = plan_usage.get("total_tokens")
                result["planner_cost_usd"]      = plan_usage.get("cost_usd")
            else:
                plan = plan_result
                # Token fields remain None

            result["plan"]         = plan
            result["plan_success"] = True

        except Exception as e:
            result["error"]            = f"Planner error: {e}"
            result["duration_seconds"] = round(time.time() - start, 2)
            return result

        # ── Stage 2: Validate ──────────────────────────────────────────
        try:
            ok, errors = validate_plan(plan)
            result["validation_errors"]  = errors
            result["validation_success"] = ok
            if not ok:
                result["duration_seconds"] = round(time.time() - start, 2)
                return result
        except Exception as e:
            result["error"]            = f"Validator error: {traceback.format_exc()}"
            result["duration_seconds"] = round(time.time() - start, 2)
            return result

        # ── Stage 3: Compile ───────────────────────────────────────────
        try:
            code     = compile_output(plan)
            app_path = os.path.join(run_dir, "app.py")
            with open(app_path, "w", encoding="utf-8") as f:
                f.write(code)
            result["compile_success"] = True
        except Exception as e:
            result["compile_error"]    = str(e)
            result["duration_seconds"] = round(time.time() - start, 2)
            return result

        # ── Stage 4: Fixtures + Execute ────────────────────────────────
        try:
            import shutil
            for dest_name, src_path in task.get("fixtures", {}).items():
                shutil.copy(src_path, os.path.join(run_dir, dest_name))

            proc = subprocess.Popen(
                [sys.executable, app_path],
                cwd=run_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            try:
                stdout, stderr = proc.communicate(timeout=task.get("timeout_seconds", 30))
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                result["run_stdout"] = stdout
                result["run_stderr"] = stderr
                if task.get("timeout_is_expected"):
                    result["run_success"]    = True
                    result["run_returncode"] = None
                else:
                    result["error"]            = "Execution timed out"
                    result["duration_seconds"] = round(time.time() - start, 2)
                    return result

            result["run_stdout"]     = stdout
            result["run_stderr"]     = stderr
            result["run_returncode"] = proc.returncode
            result["run_success"]    = proc.returncode == 0

            if not result["run_success"]:
                result["duration_seconds"] = round(time.time() - start, 2)
                return result

        except Exception as e:
            result["error"]            = f"Execution error: {e}"
            result["duration_seconds"] = round(time.time() - start, 2)
            return result

        # ── Stage 5: Criteria ──────────────────────────────────────────
        criteria = task.get("success_criteria", [])
        if criteria:
            passed, failures = check_criteria(criteria, result["run_stdout"], run_dir)
            result["criteria_passed"]    = passed
            result["criteria_failures"]  = failures
        else:
            result["criteria_passed"] = True

    result["first_pass_success"] = (
        result["plan_success"]
        and result["validation_success"]
        and result["compile_success"]
        and result["run_success"]
        and result["criteria_passed"]
    )
    result["duration_seconds"] = round(time.time() - start, 2)
    return result


# ─────────────────────────────────────────────────────────────────────
# Multi-run aggregator
# ─────────────────────────────────────────────────────────────────────

def run_task_repeated(task: dict) -> dict:
    """
    Calls run_task N_RUNS times and merges into a single result record.

    Aggregates:
        pass_count          : runs where first_pass_success was True
        first_pass_success  : True only if ALL runs passed
        avg_duration_seconds: mean wall time across runs
        planner_*_tokens    : summed across runs (total API usage)
        planner_cost_usd    : summed across runs
        runs[]              : compact per-run snapshots
    All other fields reflect the LAST run (representative detail).
    """
    runs          = []
    last_full     = None
    total_dur     = 0.0
    total_in_tok  = 0
    total_out_tok = 0
    total_cost    = 0.0
    has_usage     = False

    for run_idx in range(N_RUNS):
        print(f"    run {run_idx+1}/{N_RUNS} ...", end=" ", flush=True)
        single   = run_task(task)
        dur      = single["duration_seconds"]
        total_dur += dur

        # Accumulate planner token usage if available
        if single.get("planner_input_tokens") is not None:
            has_usage      = True
            total_in_tok  += single["planner_input_tokens"]
            total_out_tok += single["planner_output_tokens"]
            total_cost    += single["planner_cost_usd"] or 0.0

        status = "PASS" if single["first_pass_success"] else "FAIL"
        token_str = (
            f" | in={single['planner_input_tokens']} out={single['planner_output_tokens']}"
            if single.get("planner_input_tokens") is not None else ""
        )
        print(f"{status} ({dur}s{token_str})")

        runs.append({
            "run":                run_idx + 1,
            "first_pass_success": single["first_pass_success"],
            "plan_success":       single["plan_success"],
            "validation_success": single["validation_success"],
            "compile_success":    single["compile_success"],
            "run_success":        single["run_success"],
            "criteria_passed":    single["criteria_passed"],
            "duration_seconds":   dur,
            "planner_input_tokens":  single.get("planner_input_tokens"),
            "planner_output_tokens": single.get("planner_output_tokens"),
            "planner_cost_usd":      single.get("planner_cost_usd"),
            "error":              single.get("error"),
            "criteria_failures":  single.get("criteria_failures", []),
            "validation_errors":  single.get("validation_errors", []),
        })

        last_full = single

        if run_idx < N_RUNS - 1:
            time.sleep(2)

    pass_count = sum(1 for r in runs if r["first_pass_success"])

    last_full["pass_count"]           = pass_count
    last_full["run_count"]            = N_RUNS
    last_full["first_pass_success"]   = (pass_count == N_RUNS)
    last_full["avg_duration_seconds"] = round(total_dur / N_RUNS, 2)
    last_full["runs"]                 = runs

    if has_usage:
        last_full["planner_input_tokens"]  = total_in_tok
        last_full["planner_output_tokens"] = total_out_tok
        last_full["planner_total_tokens"]  = total_in_tok + total_out_tok
        last_full["planner_cost_usd"]      = round(total_cost, 6)

    return last_full


# ─────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    n               = len(results)
    compiler_passes = sum(1 for r in results if r["first_pass_success"])
    total_cost      = sum(r.get("planner_cost_usd") or 0.0 for r in results)
    avg_dur         = sum(r.get("avg_duration_seconds") or 0.0 for r in results) / n
    has_cost        = any(r.get("planner_cost_usd") is not None for r in results)

    print("\n" + "=" * 72)
    print("BENCHMARK RESULTS")
    print("=" * 72)
    print(f"Tasks:              {n}")
    print(f"Runs per task:      {N_RUNS}")
    print(f"Compiler passes:    {compiler_passes}/{n} ({100*compiler_passes//n}%)")
    print(f"Avg latency/run:    {avg_dur:.1f}s")
    if has_cost:
        print(f"Total planner cost: ${total_cost:.4f}")
        print(f"Avg cost/task:      ${total_cost/n:.4f}")
    print()
    print(f"{'ID':<22} {'Passes':<8} {'Result':<10} Stage Failed")
    print("-" * 72)
    for r in results:
        pass_str = f"{r.get('pass_count','?')}/{r.get('run_count', N_RUNS)}"
        status   = "PASS" if r["first_pass_success"] else "FAIL"
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
            stage = "-"
        print(f"{r['task_id']:<22} {pass_str:<8} {status:<10} {stage}")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM Code Graph Compiler Benchmark Harness"
    )
    parser.add_argument("--tasks",  required=True,
                        help="Path to tasks.json")
    parser.add_argument("--output", default="benchmark/results.json",
                        help="Path to write results JSON")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline runs (run_baseline.py separately)")
    parser.add_argument("--task-id",
                        help="Run a single task by ID (debugging)")
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
        print(f"\n[{i+1}/{len(tasks)}] {task['task_id']}  ({N_RUNS} runs)")
        print(f"  {task['description']}")
        result = run_task_repeated(task)

        status = "PASS" if result["first_pass_success"] else "FAIL"
        cost_str = (
            f" | planner cost=${result['planner_cost_usd']:.4f}"
            if result.get("planner_cost_usd") is not None else ""
        )
        print(
            f"  -> {status} ({result['pass_count']}/{result['run_count']} runs | "
            f"avg {result['avg_duration_seconds']}s/run{cost_str})"
        )
        if result.get("error"):
            print(f"  ! {result['error']}")
        if result.get("criteria_failures"):
            for cf in result["criteria_failures"]:
                print(f"  ! {cf}")

        results.append(result)

        if i < len(tasks) - 1:
            time.sleep(5)

    print_summary(results)

    # ── Aggregate set-level stats ──────────────────────────────────────
    total_planner_cost = sum(r.get("planner_cost_usd") or 0.0 for r in results)
    avg_latency        = sum(r.get("avg_duration_seconds") or 0.0 for r in results) / len(results)
    avg_in_tokens      = (
        sum(r.get("planner_input_tokens") or 0 for r in results) / len(results)
        if any(r.get("planner_input_tokens") for r in results) else None
    )
    avg_out_tokens     = (
        sum(r.get("planner_output_tokens") or 0 for r in results) / len(results)
        if any(r.get("planner_output_tokens") for r in results) else None
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    output = {
        "run_at":                    datetime.utcnow().isoformat(),
        "task_count":                len(results),
        "runs_per_task":             N_RUNS,
        "compiler_first_pass_rate":  sum(1 for r in results if r["first_pass_success"]) / len(results),
        "avg_latency_seconds":       round(avg_latency, 2),
        "planner_total_cost_usd":    round(total_planner_cost, 6),
        "planner_avg_cost_per_task": round(total_planner_cost / len(results), 6),
        "planner_avg_input_tokens":  round(avg_in_tokens, 1) if avg_in_tokens else None,
        "planner_avg_output_tokens": round(avg_out_tokens, 1) if avg_out_tokens else None,
        "results":                   results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
