"""
benchmark/ablations/ablation_harness.py

Ablation-only harness for compiler workflow evaluation.

Important:
- This file intentionally does NOT support the normal compiler mode.
- It only runs ablated planner variants.
- Downstream evaluation remains the same:
    planner -> validate_plan -> compile_output -> execute -> check_criteria
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, UTC
from pathlib import Path

import requests

# repo root resolution (file lives in benchmark/ablations/)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.criteria import check_criteria
from core.compiler import compile_output
from core.plan_utils import normalize_plan_shape
from core.validator import validate_plan
from nodes.registry import NODE_REGISTRY

N_RUNS = 3
DEFAULT_TIMEOUT_SECONDS = 30
ABLATION_MODES = ["no-registry", "blind-types", "no-linearize"]

# gpt-4o-mini pricing
PLANNER_INPUT_PRICE_PER_1M = 0.15
PLANNER_OUTPUT_PRICE_PER_1M = 0.60


def empty_result(task_id, description, mode):
    return {
        "task_id": task_id,
        "description": description,
        "mode": mode,

        "plan_success": False,
        "validation_success": False,
        "compile_success": False,
        "run_success": False,
        "criteria_passed": False,

        "first_pass_success": False,

        "duration_seconds": None,
        "avg_duration_seconds": None,

        "planner_input_tokens": None,
        "planner_output_tokens": None,
        "planner_total_tokens": None,
        "planner_cost_usd": None,

        "llm_raw_output": None,
        "llm_raw_output_preview": None,
        "llm_output_captured": False,

        "plan": None,
        "validation_errors": [],
        "compile_error": None,
        "run_stdout": "",
        "run_stderr": "",
        "run_returncode": None,
        "criteria_failures": [],
        "error": None,

        "pass_count": 0,
        "run_count": N_RUNS,
        "runs": [],
    }


def _safe_preview(text, limit=500):
    if text is None:
        return None
    text = str(text)
    return text[:limit]


def _planner_cost(input_tokens, output_tokens):
    if input_tokens is None or output_tokens is None:
        return None

    return round(
        (input_tokens / 1_000_000) * PLANNER_INPUT_PRICE_PER_1M
        + (output_tokens / 1_000_000) * PLANNER_OUTPUT_PRICE_PER_1M,
        6,
    )


def _planner_system_prompt():
    try:
        from core.planner import SYSTEM_PROMPT
        return SYSTEM_PROMPT
    except Exception:
        return """You are a code graph planner.

You select nodes from a fixed library and connect them to solve the user's task.

Output STRICTLY raw JSON.

Response format must be:
{
  "nodes": [],
  "edges": [],
  "parameters": {},
  "flags": [],
  "glue_code": ""
}

Rules:
- Only use nodes from the provided library.
- Never invent nodes.
- Every edge must be type-compatible.
- If a required node is missing, add flag MISSING_NODE.
- If credentials are needed, add flag REQUIRED_CREDENTIAL.
- Glue code must follow topological execution order.
- Return raw JSON only.
"""


def _build_node_summary(include_types=True):
    lines = []
    for name, node in NODE_REGISTRY.items():
        parts = [f"- {name}: {node.description}"]
        if include_types:
            parts.append(f"input: {node.input_type}")
            parts.append(f"output: {node.output_type}")
        parts.append(f"required params: {node.required_params}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _raw_chat_completion(system_prompt, user_prompt):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }

    last_error = None

    for attempt in range(4):
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=45,
            )

            if resp.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue

            if 500 <= resp.status_code < 600:
                time.sleep(5 * (attempt + 1))
                continue

            resp.raise_for_status()
            data = resp.json()

            raw = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")

            usage_raw = data.get("usage") or {}
            usage = None
            if usage_raw:
                input_tokens = usage_raw.get("prompt_tokens")
                output_tokens = usage_raw.get("completion_tokens")
                total_tokens = usage_raw.get("total_tokens")
                usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "cost_usd": _planner_cost(input_tokens, output_tokens),
                }

            return raw, usage

        except Exception as e:
            last_error = e
            if attempt < 3:
                time.sleep(5)
            else:
                break

    raise RuntimeError(f"Planner failed after retries: {last_error}")


def _normalize_no_linearize(plan):
    return normalize_plan_shape(plan)


def _get_plan_custom(task_description, include_registry=True, include_types=True, linearize=True):
    user_parts = []

    if include_registry:
        user_parts.append(
            f"Available Nodes:\n{_build_node_summary(include_types=include_types)}"
        )

    user_parts.append(f"Task:\n{task_description}")
    user_prompt = "\n\n".join(user_parts)

    raw_text, usage = _raw_chat_completion(_planner_system_prompt(), user_prompt)

    try:
        plan = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Planner returned non-JSON output: {e}\nRaw output:\n{raw_text}") from e

    if linearize:
        try:
            from core.planner import normalize_plan
            plan = normalize_plan(plan)
        except Exception:
            pass
    else:
        plan = _normalize_no_linearize(plan)

    return plan, usage, raw_text


def get_plan_for_mode(task_description, mode):
    if mode == "no-registry":
        return _get_plan_custom(
            task_description,
            include_registry=False,
            include_types=False,
            linearize=True,
        )

    if mode == "blind-types":
        return _get_plan_custom(
            task_description,
            include_registry=True,
            include_types=False,
            linearize=True,
        )

    if mode == "no-linearize":
        return _get_plan_custom(
            task_description,
            include_registry=True,
            include_types=True,
            linearize=False,
        )

    raise ValueError(
        f"Unsupported mode '{mode}'. Allowed ablation modes: {', '.join(ABLATION_MODES)}"
    )


def run_task(task):
    result = empty_result(task["task_id"], task["description"], task["mode"])
    start = time.time()

    with tempfile.TemporaryDirectory() as run_dir:
        # -----------------------------
        # Stage 1 — Planner
        # -----------------------------
        try:
            plan, usage, raw_output = get_plan_for_mode(task["description"], task["mode"])

            result["plan"] = plan
            result["plan_success"] = True

            if usage:
                result["planner_input_tokens"] = usage.get("input_tokens")
                result["planner_output_tokens"] = usage.get("output_tokens")
                result["planner_total_tokens"] = usage.get("total_tokens")
                result["planner_cost_usd"] = usage.get("cost_usd")

            result["llm_raw_output"] = raw_output
            result["llm_raw_output_preview"] = _safe_preview(raw_output)
            result["llm_output_captured"] = raw_output is not None

        except Exception as e:
            result["error"] = f"Planner error: {e}"
            result["duration_seconds"] = round(time.time() - start, 2)
            return result

        # -----------------------------
        # Stage 2 — Validator
        # -----------------------------
        try:
            ok, errors = validate_plan(result["plan"])

            result["validation_success"] = ok
            result["validation_errors"] = errors

            if not ok:
                result["duration_seconds"] = round(time.time() - start, 2)
                return result

        except Exception:
            result["error"] = f"Validator error: {traceback.format_exc()}"
            result["duration_seconds"] = round(time.time() - start, 2)
            return result

        # -----------------------------
        # Stage 3 — Compiler
        # -----------------------------
        try:
            code = compile_output(result["plan"])

            app_path = os.path.join(run_dir, "app.py")
            with open(app_path, "w", encoding="utf-8") as f:
                f.write(code)

            result["compile_success"] = True

        except Exception as e:
            result["compile_error"] = str(e)
            result["duration_seconds"] = round(time.time() - start, 2)
            return result

        # -----------------------------
        # Stage 4 — Execution
        # -----------------------------
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
                stdout, stderr = proc.communicate(
                    timeout=task.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()

                result["run_stdout"] = stdout
                result["run_stderr"] = stderr

                if task.get("timeout_is_expected"):
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

        except Exception as e:
            result["error"] = f"Execution error: {e}"
            result["duration_seconds"] = round(time.time() - start, 2)
            return result

        # -----------------------------
        # Stage 5 — Criteria
        # -----------------------------
        criteria = task.get("success_criteria", [])
        if criteria:
            passed, failures = check_criteria(criteria, result["run_stdout"], run_dir)
            result["criteria_passed"] = passed
            result["criteria_failures"] = failures
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


def make_run_summary(single, run_index):
    return {
        "run": run_index,
        "first_pass_success": single["first_pass_success"],
        "plan_success": single["plan_success"],
        "validation_success": single["validation_success"],
        "compile_success": single["compile_success"],
        "run_success": single["run_success"],
        "criteria_passed": single["criteria_passed"],
        "duration_seconds": single["duration_seconds"],

        "planner_input_tokens": single.get("planner_input_tokens"),
        "planner_output_tokens": single.get("planner_output_tokens"),
        "planner_total_tokens": single.get("planner_total_tokens"),
        "planner_cost_usd": single.get("planner_cost_usd"),

        "llm_output_captured": single.get("llm_output_captured"),
        "llm_raw_output": single.get("llm_raw_output"),
        "llm_raw_output_preview": single.get("llm_raw_output_preview"),

        "plan": single.get("plan"),
        "validation_errors": single.get("validation_errors", []),
        "compile_error": single.get("compile_error"),
        "run_stdout": single.get("run_stdout", ""),
        "run_stderr": single.get("run_stderr", ""),
        "run_returncode": single.get("run_returncode"),
        "criteria_failures": single.get("criteria_failures", []),
        "error": single.get("error"),
    }


def run_task_repeated(task):
    runs = []
    raw_results = []
    total_duration = 0.0

    planner_input_total = 0
    planner_output_total = 0
    planner_total_total = 0
    planner_cost_total = 0.0

    for i in range(N_RUNS):
        print(f"    run {i+1}/{N_RUNS} ...", end=" ")

        single = run_task(task)
        raw_results.append(single)
        runs.append(make_run_summary(single, i + 1))

        total_duration += single["duration_seconds"] or 0.0
        planner_input_total += single.get("planner_input_tokens") or 0
        planner_output_total += single.get("planner_output_tokens") or 0
        planner_total_total += single.get("planner_total_tokens") or 0
        planner_cost_total += single.get("planner_cost_usd") or 0.0

        status = "✓" if single["first_pass_success"] else "✗"
        print(f"{status} ({single['duration_seconds']}s)")

        if i < N_RUNS - 1:
            time.sleep(2)

    pass_count = sum(1 for r in raw_results if r["first_pass_success"])

    final = copy.deepcopy(raw_results[-1])
    final["pass_count"] = pass_count
    final["run_count"] = N_RUNS
    final["runs"] = runs
    final["avg_duration_seconds"] = round(total_duration / N_RUNS, 2)

    final["planner_input_tokens"] = planner_input_total or None
    final["planner_output_tokens"] = planner_output_total or None
    final["planner_total_tokens"] = planner_total_total or None
    final["planner_cost_usd"] = round(planner_cost_total, 6) if planner_cost_total else None

    # task-level pass means all runs passed
    final["first_pass_success"] = pass_count == N_RUNS

    return final


def main():
    parser = argparse.ArgumentParser(description="Ablation-only harness")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=ABLATION_MODES,
        help="Ablation mode only. Compiler mode is intentionally disabled.",
    )
    args = parser.parse_args()

    with open(args.tasks, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    for task in tasks:
        task["mode"] = args.mode

    results = []

    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task['task_id']} [{args.mode}]")
        print(task["description"])

        result = run_task_repeated(task)

        status = "✓ PASS" if result["first_pass_success"] else "✗ FAIL"
        print(
            f"  → {status} ({result['pass_count']}/{result['run_count']} runs | "
            f"avg {result['avg_duration_seconds']}s)"
        )

        results.append(result)

        if i < len(tasks) - 1:
            time.sleep(5)

    out = {
        "run_at": datetime.now(UTC).isoformat(),
        "task_count": len(results),
        "runs_per_task": N_RUNS,
        "mode": args.mode,
        "ablation_pass_rate": (
            sum(1 for r in results if r["first_pass_success"]) / len(results)
            if results else 0.0
        ),
        "results": results,
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
