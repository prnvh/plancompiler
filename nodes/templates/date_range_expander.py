from __future__ import annotations

from typing import Any

import pandas as pd

from nodes.contracts import normalize_node_parameters
from nodes.templates.frame_support import resolve_column_label, resolve_column_labels


def _resolve_fill_value(group_frame: pd.DataFrame, column: str, strategy: str):
    non_null = group_frame[column].dropna()

    if strategy == "group_min":
        return non_null.min() if not non_null.empty else None
    if strategy == "group_max":
        return non_null.max() if not non_null.empty else None
    if strategy == "group_first":
        return non_null.iloc[0] if not non_null.empty else None
    if strategy == "group_last":
        return non_null.iloc[-1] if not non_null.empty else None
    if strategy == "zero":
        return 0
    if strategy == "empty_string":
        return ""

    raise ValueError(f"Unsupported DateRangeExpander fill strategy: {strategy}")


def _apply_fill_rules(
    frame: pd.DataFrame,
    *,
    group_keys: list[Any],
    date_column: Any,
    fill_values: dict[str, Any] | None,
    fill_strategies: dict[str, str] | None,
) -> pd.DataFrame:
    result = frame.copy()
    fill_values = fill_values or {}
    fill_strategies = fill_strategies or {}

    group_columns = [key for key in group_keys if key in result.columns]
    protected_columns = set(group_columns + [date_column])

    for column in result.columns:
        if column in protected_columns:
            continue

        if column in fill_values:
            result[column] = result[column].fillna(fill_values[column])
            continue

        strategy = fill_strategies.get(column)
        if not strategy:
            continue

        if strategy in {"ffill", "bfill"}:
            result[column] = (
                result.groupby(group_columns, dropna=False)[column]
                .transform(lambda series: getattr(series, strategy)())
            )
            continue

        result[column] = result.groupby(group_columns, dropna=False)[column].transform(
            lambda series: series.fillna(_resolve_fill_value(result.loc[series.index], column, strategy))
        )

    return result


def date_range_expander(
    df: pd.DataFrame,
    group_keys=None,
    date_column: str | None = None,
    freq: str = "D",
    range_scope: str = "group",
    fill_values: dict[str, Any] | None = None,
    fill_strategies: dict[str, str] | None = None,
    date_format: str | None = None,
    sort: bool = True,
    group_by=None,
    **kwargs,
) -> pd.DataFrame:
    params = {
        "group_keys": group_keys,
        "date_column": date_column,
        "freq": freq,
        "range_scope": range_scope,
        "fill_values": fill_values,
        "fill_strategies": fill_strategies,
        "date_format": date_format,
        "sort": sort,
        "group_by": group_by,
    }
    params.update(kwargs)
    normalized = normalize_node_parameters("DateRangeExpander", params)

    resolved_group_keys = resolve_column_labels(df, normalized["group_keys"]) or []
    resolved_date_column = resolve_column_label(df, normalized["date_column"])

    working = df.copy()
    working[resolved_date_column] = pd.to_datetime(working[resolved_date_column], errors="raise")

    if normalized.get("range_scope", "group") == "global":
        global_start = working[resolved_date_column].min()
        global_end = working[resolved_date_column].max()
    else:
        global_start = global_end = None

    expanded_frames: list[pd.DataFrame] = []

    for group_key, group_frame in working.groupby(resolved_group_keys, dropna=False, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_values = dict(zip(resolved_group_keys, group_key))

        start = global_start if global_start is not None else group_frame[resolved_date_column].min()
        end = global_end if global_end is not None else group_frame[resolved_date_column].max()
        date_range = pd.date_range(start, end, freq=normalized.get("freq", "D"))

        expanded = pd.DataFrame({resolved_date_column: date_range})
        for key, value in group_values.items():
            expanded[key] = value

        merged = expanded.merge(
            group_frame,
            on=resolved_group_keys + [resolved_date_column],
            how="left",
            sort=False,
        )
        expanded_frames.append(merged)

    result = pd.concat(expanded_frames, ignore_index=True)
    result = _apply_fill_rules(
        result,
        group_keys=resolved_group_keys,
        date_column=resolved_date_column,
        fill_values=normalized.get("fill_values"),
        fill_strategies=normalized.get("fill_strategies"),
    )

    if normalized.get("sort", True):
        result = result.sort_values(by=resolved_group_keys + [resolved_date_column], kind="stable").reset_index(drop=True)

    if normalized.get("date_format"):
        result[resolved_date_column] = result[resolved_date_column].dt.strftime(normalized["date_format"])

    print("[DateRangeExpander] Expanded grouped date ranges and filled missing values")
    return result
