from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def resolve_column_label(df: pd.DataFrame, reference: Any) -> Any:
    columns = list(df.columns)

    if reference in columns:
        return reference

    reference_text = str(reference).strip()
    string_matches = [column for column in columns if str(column) == reference_text]
    if len(string_matches) == 1:
        return string_matches[0]

    if reference_text.isdigit():
        numeric_reference = int(reference_text)
        if numeric_reference in columns:
            return numeric_reference

    try:
        float_reference = float(reference_text)
    except Exception:
        float_reference = None
    if float_reference is not None and float_reference in columns:
        return float_reference

    return reference


def resolve_column_labels(df: pd.DataFrame, references: list[Any] | tuple[Any, ...] | Any | None) -> list[Any] | None:
    if references is None:
        return None
    if not isinstance(references, (list, tuple)):
        references = [references]
    return [resolve_column_label(df, reference) for reference in references]


def resolve_index_level_reference(df: pd.DataFrame, reference: Any) -> Any:
    index = df.index

    if not isinstance(index, pd.MultiIndex):
        if reference in {0, index.name, str(index.name)}:
            return 0
        return reference

    level_names = list(index.names)
    if isinstance(reference, int) and 0 <= reference < len(level_names):
        return reference

    reference_text = str(reference).strip()
    for position, level_name in enumerate(level_names):
        if level_name == reference or str(level_name) == reference_text:
            return position

    if reference_text.isdigit():
        position = int(reference_text)
        if 0 <= position < len(level_names):
            return position

    return reference


def set_index_level_values(df: pd.DataFrame, level_reference: Any, values) -> pd.DataFrame:
    level = resolve_index_level_reference(df, level_reference)

    if not isinstance(df.index, pd.MultiIndex):
        updated = df.copy()
        updated.index = pd.Index(values, name=df.index.name)
        return updated

    level_position = level if isinstance(level, int) else df.index.names.index(level)
    arrays = [
        df.index.get_level_values(position)
        for position in range(df.index.nlevels)
    ]
    arrays[level_position] = values
    updated = df.copy()
    updated.index = pd.MultiIndex.from_arrays(arrays, names=df.index.names)
    return updated


def resolve_mapping_keys(series: pd.Series, mapping: Mapping[Any, Any]) -> dict[Any, Any]:
    resolved: dict[Any, Any] = {}
    unique_values = list(pd.Index(series.dropna().unique()))

    for key, value in mapping.items():
        if key in unique_values or key in series.index:
            resolved[key] = value
            continue

        key_text = str(key).strip()
        matches = [candidate for candidate in unique_values if str(candidate) == key_text]
        if len(matches) == 1:
            resolved[matches[0]] = value
        else:
            resolved[key] = value

    return resolved


def first_present_column(df: pd.DataFrame, *candidates: Any) -> Any | None:
    for candidate in candidates:
        resolved = resolve_column_label(df, candidate)
        if resolved in df.columns:
            return resolved
    return None
