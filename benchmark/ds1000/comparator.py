from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from benchmark.ds1000.loader import infer_result_kind


@dataclass(slots=True)
class DS1000ComparisonResult:
    passed: bool
    expected_kind: str
    actual_kind: str
    failures: list[str] = field(default_factory=list)


def _compare_scalars(actual: Any, expected: Any, *, rtol: float, atol: float) -> list[str]:
    if isinstance(actual, (pd.DataFrame, pd.Series)):
        return [f"scalar mismatch: expected {expected!r}, got {type(actual).__name__}"]
    if isinstance(expected, (pd.DataFrame, pd.Series)):
        return [f"scalar mismatch: expected {type(expected).__name__}, got {actual!r}"]

    if pd.isna(actual) and pd.isna(expected):
        return []

    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol):
            return []
        return [f"scalar mismatch: expected {expected!r}, got {actual!r}"]

    if actual == expected:
        return []

    return [f"scalar mismatch: expected {expected!r}, got {actual!r}"]


def compare_ds1000_result(
    actual: Any,
    expected: Any,
    *,
    expected_kind: str | None = None,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> DS1000ComparisonResult:
    resolved_expected_kind = expected_kind or infer_result_kind(expected)
    actual_kind = infer_result_kind(actual)
    failures: list[str] = []

    if resolved_expected_kind == "dataframe":
        if not isinstance(actual, pd.DataFrame):
            failures.append(f"expected DataFrame, got {type(actual).__name__}")
        else:
            try:
                pd.testing.assert_frame_equal(
                    actual,
                    expected,
                    check_dtype=False,
                    rtol=rtol,
                    atol=atol,
                )
            except AssertionError as error:
                failures.append(str(error))

    elif resolved_expected_kind == "series":
        if not isinstance(actual, pd.Series):
            failures.append(f"expected Series, got {type(actual).__name__}")
        else:
            try:
                pd.testing.assert_series_equal(
                    actual,
                    expected,
                    check_dtype=False,
                    rtol=rtol,
                    atol=atol,
                )
            except AssertionError as error:
                failures.append(str(error))

    elif resolved_expected_kind == "scalar":
        failures.extend(_compare_scalars(actual, expected, rtol=rtol, atol=atol))

    else:
        failures.append(
            f"unsupported expected result kind '{resolved_expected_kind}' for DS-1000 comparator."
        )

    return DS1000ComparisonResult(
        passed=not failures,
        expected_kind=resolved_expected_kind,
        actual_kind=actual_kind,
        failures=failures,
    )
