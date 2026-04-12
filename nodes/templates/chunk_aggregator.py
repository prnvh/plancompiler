from __future__ import annotations

import re
from typing import Any

import pandas as pd

from nodes.contracts import normalize_node_parameters
from nodes.templates.frame_support import resolve_column_label, resolve_column_labels


_CHUNK_AGG_FUNCS = {
    "sum": lambda series: series.sum(),
    "mean": lambda series: series.mean(),
    "min": lambda series: series.min(),
    "max": lambda series: series.max(),
    "count": lambda series: series.count(),
    "first": lambda series: series.iloc[0] if len(series) else None,
    "last": lambda series: series.iloc[-1] if len(series) else None,
}


def _resolve_rule_columns(df: pd.DataFrame, rule: dict) -> list[Any]:
    columns = list(df.columns)
    selected: list[Any] = []

    explicit = resolve_column_labels(df, rule.get("columns"))
    if explicit:
        selected.extend(explicit)

    prefix = rule.get("columns_prefix")
    if prefix:
        selected.extend([column for column in columns if str(column).startswith(str(prefix))])

    suffix = rule.get("columns_suffix")
    if suffix:
        selected.extend([column for column in columns if str(column).endswith(str(suffix))])

    regex = rule.get("columns_regex")
    if regex:
        pattern = re.compile(str(regex))
        selected.extend([column for column in columns if pattern.search(str(column))])

    if not selected:
        selected = columns[:]

    exclude = set(resolve_column_labels(df, rule.get("exclude_columns")) or [])
    seen = set()
    resolved: list[Any] = []
    for column in selected:
        resolved_column = resolve_column_label(df, column)
        if resolved_column in exclude or resolved_column in seen or resolved_column not in df.columns:
            continue
        seen.add(resolved_column)
        resolved.append(resolved_column)

    return resolved


def _chunk_groups(length: int, size: int, *, from_end: bool, drop_remainder: bool) -> list[tuple[int, int]]:
    if size <= 0:
        raise ValueError("ChunkAggregator size must be positive.")
    if length == 0:
        return []

    if from_end:
        remainder = length % size
        boundaries: list[tuple[int, int]] = []
        start = 0
        if remainder:
            if not drop_remainder:
                boundaries.append((0, remainder))
            start = remainder
        while start < length:
            boundaries.append((start, min(start + size, length)))
            start += size
        return boundaries

    boundaries = []
    start = 0
    while start < length:
        end = min(start + size, length)
        if end - start < size and drop_remainder:
            break
        boundaries.append((start, end))
        start = end
    return boundaries


def _aggregate_chunks(series: pd.Series, *, size: int, agg: str, from_end: bool, drop_remainder: bool) -> pd.Series:
    aggregator = _CHUNK_AGG_FUNCS[agg]
    boundaries = _chunk_groups(len(series), size, from_end=from_end, drop_remainder=drop_remainder)
    values = [
        aggregator(series.iloc[start:end])
        for start, end in boundaries
    ]
    return pd.Series(values)


def chunk_aggregator(
    df: pd.DataFrame,
    rules=None,
    from_end: bool = False,
    drop_remainder: bool = False,
    windows=None,
    value_columns=None,
    **kwargs,
) -> pd.DataFrame:
    params = {
        "rules": rules,
        "from_end": from_end,
        "drop_remainder": drop_remainder,
        "windows": windows,
        "value_columns": value_columns,
    }
    params.update(kwargs)
    normalized = normalize_node_parameters("ChunkAggregator", params)

    result_columns: dict[Any, pd.Series] = {}
    for rule in normalized["rules"]:
        selected_columns = _resolve_rule_columns(df, rule)
        for column in selected_columns:
            source_series = df[column]
            if rule["agg"] in {"sum", "mean", "min", "max"}:
                source_series = pd.to_numeric(source_series, errors="coerce")
            result_columns[column] = _aggregate_chunks(
                source_series,
                size=int(rule["size"]),
                agg=rule["agg"],
                from_end=bool(normalized.get("from_end", False)),
                drop_remainder=bool(normalized.get("drop_remainder", False)),
            )

    result = pd.concat(result_columns, axis=1) if result_columns else pd.DataFrame()
    print("[ChunkAggregator] Aggregated fixed-size row chunks")
    return result
