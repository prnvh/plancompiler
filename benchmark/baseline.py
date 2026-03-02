"""
benchmark/baseline.py

Direct GPT-4o baseline for benchmark comparison.

Prompts GPT-4o to write a complete Python script for the task,
then executes it and checks the same success criteria.
The prompt deliberately gives GPT-4o no scaffolding — pure generation.
"""

import os
import subprocess
import sys
import tempfile
import shutil
import time
import requests as req
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASELINE_SYSTEM_PROMPT = """You are an expert Python developer.
Write a complete, runnable Python script that accomplishes the task described.
Return only the Python code — no explanation, no markdown fences, no preamble.
The script must run as-is with standard Python libraries plus pandas, sqlalchemy, flask."""


def _call_openai(prompt: str, timeout: int = 45) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    for attempt in range(4):
        try:
            resp = req.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  [baseline] rate limited, waiting {wait}s (attempt {attempt+1}/4)...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except req.exceptions.Timeout:
            if attempt == 3:
                raise RuntimeError("Baseline API call timed out after 45 seconds")
            print(f"  [baseline] timeout, retrying (attempt {attempt+1}/4)...")
            time.sleep(5)
        except Exception as e:
            raise RuntimeError(f"Baseline API call failed: {e}") from e
    raise RuntimeError("Baseline failed after 4 attempts")


def run_baseline(task: dict) -> tuple[bool, str | None]:
    """
    Runs GPT-4o directly on the task description, executes the result,
    and checks success criteria.

    Returns:
        (success: bool, error_message: str | None)
    """
    try:
        code = _call_openai(task["description"])

        # Strip markdown fences if the model ignored instructions
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    except Exception as e:
        return False, f"Baseline generation failed: {e}"

    with tempfile.TemporaryDirectory() as run_dir:
        # Write generated code
        app_path = os.path.join(run_dir, "app.py")
        with open(app_path, "w") as f:
            f.write(code)

        # Copy fixtures
        for dest_name, src_path in task.get("fixtures", {}).items():
            shutil.copy(src_path, os.path.join(run_dir, dest_name))

        # Execute
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
                if not task.get("timeout_is_expected"):
                    return False, "Baseline execution timed out"
                returncode = None
            else:
                returncode = proc.returncode
        except Exception as e:
            return False, f"Baseline execution error: {e}"

        if returncode is not None and returncode != 0:
            return False, f"Baseline runtime error (exit {returncode}):\n{stderr[:500]}"

        # Check criteria
        criteria = task.get("success_criteria", [])
        if not criteria:
            return True, None

        from benchmark.harness import check_criteria
        passed, failures = check_criteria(criteria, stdout, run_dir)
        if not passed:
            return False, "Criteria failures:\n" + "\n".join(failures)

        return True, None