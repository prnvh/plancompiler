"""
benchmark/baseline.py

Direct GPT-4o baseline for benchmark comparison.

Prompts GPT-4o to write a complete Python script for the task,
then executes it and checks the same success criteria.
The prompt deliberately gives GPT-4o no scaffolding — pure generation.
"""

import os
import signal
import subprocess
import sys
import tempfile
import shutil
import threading
import time
import requests as req
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASELINE_SYSTEM_PROMPT = """You are an expert Python developer.
Write a complete, runnable Python script that accomplishes the task described.
Return only the Python code — no explanation, no markdown fences, no preamble.
The script must run as-is with standard Python libraries plus pandas, sqlalchemy, flask."""


# ─────────────────────────────────────────────────────────────────────
# OpenAI call
# ─────────────────────────────────────────────────────────────────────

def _call_openai(prompt: str, timeout: int = 60) -> str:
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
                wait = 15 * (attempt + 1)
                print(f"  [baseline] rate limited, waiting {wait}s (attempt {attempt+1}/4)...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except req.exceptions.Timeout:
            if attempt == 3:
                raise RuntimeError("Baseline API call timed out after 60 seconds")
            print(f"  [baseline] timeout on API call, retrying (attempt {attempt+1}/4)...")
            time.sleep(5)
        except Exception as e:
            raise RuntimeError(f"Baseline API call failed: {e}") from e
    raise RuntimeError("Baseline failed after 4 attempts")


# ─────────────────────────────────────────────────────────────────────
# Process execution with hard kill
# ─────────────────────────────────────────────────────────────────────

def _kill_process_group(proc: subprocess.Popen) -> None:
    """
    Kill the entire process group, not just the parent process.

    This is the critical fix. When a generated script starts a Flask server
    (or any server), it may spawn threads or child processes that inherit
    the stdout/stderr pipe file descriptors. Killing only proc leaves those
    alive, so proc.communicate() blocks forever waiting for EOF on the pipe.

    start_new_session=True in Popen puts the subprocess in its own process
    group. os.killpg then kills every process in that group at once.
    """
    try:
        if sys.platform == "win32":
            subprocess.call(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # Already gone
            except Exception:
                proc.kill()  # Fallback
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _communicate_with_timeout(
    proc: subprocess.Popen,
    timeout_secs: int,
) -> tuple[str, str, bool]:
    """
    Run proc.communicate() on a daemon thread with a hard wall-clock timeout.

    Returns (stdout, stderr, timed_out).

    Why a thread instead of proc.communicate(timeout=...)?
    The stdlib timeout raises TimeoutExpired and then you call proc.kill(),
    followed by another proc.communicate() to drain the pipes. If the child
    spawned server threads that still hold the pipe open, that second
    communicate() blocks forever. The thread approach lets us kill the whole
    process group and then just abandon the thread if draining takes too long.
    """
    result: dict = {"stdout": "", "stderr": "", "exc": None}

    def _target():
        try:
            out, err = proc.communicate()
            result["stdout"] = out or ""
            result["stderr"] = err or ""
        except Exception as e:
            result["exc"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_secs)

    if t.is_alive():
        # Timed out — kill the whole process group, then give the thread
        # a short grace period to drain and exit.
        _kill_process_group(proc)
        t.join(timeout=5)
        return result["stdout"], result["stderr"], True

    return result["stdout"], result["stderr"], False


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────

def run_baseline(task: dict) -> tuple[bool, str | None]:
    """
    Runs GPT-4o directly on the task description, executes the result,
    and checks success criteria.

    Returns:
        (success: bool, error_message: str | None)
    """
    # ── Generate code ──────────────────────────────────────────────────
    try:
        code = _call_openai(task["description"])

        # Strip markdown fences if the model ignored instructions
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    except Exception as e:
        return False, f"Baseline generation failed: {e}"

    # ── Execute in temp dir ────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as run_dir:
        app_path = os.path.join(run_dir, "app.py")
        with open(app_path, "w") as f:
            f.write(code)

        # Copy fixtures
        for dest_name, src_path in task.get("fixtures", {}).items():
            try:
                shutil.copy(src_path, os.path.join(run_dir, dest_name))
            except Exception as e:
                return False, f"Fixture copy failed ({dest_name}): {e}"

        # Launch subprocess in its own process group (key for clean kill)
        try:
            popen_kwargs = dict(
                cwd=run_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if sys.platform != "win32":
                popen_kwargs["start_new_session"] = True  # New process group

            proc = subprocess.Popen(
                [sys.executable, app_path],
                **popen_kwargs,
            )
        except Exception as e:
            return False, f"Baseline launch error: {e}"

        timeout = task.get("timeout_seconds", 30)
        stdout, stderr, timed_out = _communicate_with_timeout(proc, timeout)
        returncode = proc.returncode  # None if we killed it

        if timed_out:
            if not task.get("timeout_is_expected"):
                return False, "Baseline execution timed out"
            # timeout_is_expected (e.g. Flask server) — treat as ok, check
            # criteria on files written before the process was killed
        elif returncode != 0:
            return False, f"Baseline runtime error (exit {returncode}):\n{stderr[:500]}"

        # ── Check criteria ─────────────────────────────────────────────
        criteria = task.get("success_criteria", [])
        if not criteria:
            return True, None

        from benchmark.criteria import check_criteria
        passed, failures = check_criteria(criteria, stdout, run_dir)
        if not passed:
            return False, "Criteria failures:\n" + "\n".join(failures)

        return True, None