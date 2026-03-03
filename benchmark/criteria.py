"""
benchmark/criteria.py

Shared success criteria checker used by both harness.py and baseline.py.
Extracted to break the circular import: harness -> baseline -> harness.
"""

import os
import pandas as pd


def _read_tabular(path: str):
    if path.endswith(".json"):
        return pd.read_json(path)
    return pd.read_csv(path)


def check_criteria(criteria: list, stdout: str, run_dir: str) -> tuple[bool, list[str]]:
    failures = []

    for c in criteria:
        ctype = c["type"]

        if ctype == "stdout_contains":
            if c["expected"] not in (stdout or ""):
                failures.append(f"stdout_contains: expected '{c['expected']}' not found in output")

        elif ctype == "file_exists":
            full_path = os.path.join(run_dir, c["path"])
            if not os.path.exists(full_path):
                failures.append(f"file_exists: '{c['path']}' not found")

        elif ctype == "file_row_count":
            full_path = os.path.join(run_dir, c["path"])
            try:
                df = _read_tabular(full_path)
                if len(df) != c["expected"]:
                    failures.append(
                        f"file_row_count: '{c['path']}' has {len(df)} rows, expected {c['expected']}"
                    )
            except Exception as e:
                failures.append(f"file_row_count: could not read '{c['path']}': {e}")

        elif ctype == "file_has_column":
            full_path = os.path.join(run_dir, c["path"])
            try:
                df = _read_tabular(full_path)
                if c["column"] not in df.columns:
                    failures.append(
                        f"file_has_column: column '{c['column']}' not in '{c['path']}'"
                    )
            except Exception as e:
                failures.append(f"file_has_column: could not read '{c['path']}': {e}")

    return len(failures) == 0, failures