from __future__ import annotations

import re

import pandas as pd

from nodes.contracts import normalize_node_parameters
from nodes.templates.expression_support import evaluate_frame_expression
from nodes.templates.frame_support import evaluate_dynamic_value, resolve_column_label, resolve_column_labels


_PANDAS_AGG_OPS = {
    "sum": "sum",
    "mean": "mean",
    "std": "std",
    "min": "min",
    "max": "max",
    "count": "count",
    "first": "first",
    "last": "last",
    "nunique": "nunique",
}


def _resolve_group_keys(df: pd.DataFrame, group_keys: list) -> tuple[pd.DataFrame, list[str]]:
    working = df.copy()
    resolved_keys: list[str] = []

    for index, group_key in enumerate(group_keys):
        if isinstance(group_key, str):
            resolved_keys.append(group_key)
            continue

        key_type = group_key["type"]
        if key_type == "column":
            resolved_keys.append(resolve_column_label(working, group_key["column"]))
            continue

        if key_type == "expression":
            name = group_key["name"]
            working[name] = evaluate_frame_expression(working, group_key["expression"])
            resolved_keys.append(name)
            continue

        raise ValueError(f"Unsupported Aggregator group key type: {key_type}")

    return working, resolved_keys


def _collect_grouped_rows(frame: pd.DataFrame, columns: list[str]) -> list[list]:
    return frame[columns].values.tolist()


def _select_columns(df: pd.DataFrame, aggregation: dict) -> list:
    selected: list = []
    explicit_columns = resolve_column_labels(df, aggregation.get("columns"))
    if explicit_columns:
        selected.extend(explicit_columns)

    explicit_column = evaluate_dynamic_value(df, aggregation.get("column"))
    if explicit_column is not None:
        if isinstance(explicit_column, (list, tuple, pd.Index, pd.Series)):
            selected.extend(resolve_column_labels(df, explicit_column) or [])
        else:
            selected.append(resolve_column_label(df, explicit_column))

    prefix = aggregation.get("columns_prefix")
    if prefix:
        selected.extend([column for column in df.columns if str(column).startswith(str(prefix))])

    suffix = aggregation.get("columns_suffix")
    if suffix:
        selected.extend([column for column in df.columns if str(column).endswith(str(suffix))])

    regex = aggregation.get("columns_regex")
    if regex:
        pattern = re.compile(str(regex))
        selected.extend([column for column in df.columns if pattern.search(str(column))])

    exclude_columns = set(resolve_column_labels(df, aggregation.get("exclude_columns")) or [])
    seen = set()
    resolved: list = []
    for column in selected:
        resolved_column = resolve_column_label(df, column)
        if resolved_column in seen or resolved_column in exclude_columns or resolved_column not in df.columns:
            continue
        seen.add(resolved_column)
        resolved.append(resolved_column)

    return resolved


def _materialize_aggregation_expression(df: pd.DataFrame, aggregation: dict, index: int) -> dict | None:
    if "column" not in aggregation:
        return None

    evaluated = evaluate_dynamic_value(df, aggregation.get("column"))
    if not isinstance(evaluated, pd.Series):
        return None
    if len(evaluated) != len(df):
        return None

    temp_column = aggregation.get("expression_name") or f"__agg_expr_{index}"
    df[temp_column] = evaluated.reindex(df.index)
    return {
        **aggregation,
        "column": temp_column,
        "output": aggregation.get("output") or temp_column,
    }


def _default_output_name(column, aggregation: dict) -> str:
    if aggregation.get("output"):
        return str(aggregation["output"])
    return str(column)


def _expand_aggregation_specs(df: pd.DataFrame, aggregations: list[dict]) -> list[dict]:
    expanded: list[dict] = []

    for index, aggregation in enumerate(aggregations):
        op = aggregation["op"]
        if op == "size":
            expanded.append({**aggregation, "output": aggregation.get("output") or "size"})
            continue

        materialized = _materialize_aggregation_expression(df, aggregation, index)
        if materialized is not None:
            expanded.append(materialized)
            continue

        if op == "collect_rows":
            columns = resolve_column_labels(df, aggregation.get("columns")) or []
            expanded.append({**aggregation, "columns": columns, "output": aggregation.get("output") or "rows"})
            continue

        selected_columns = _select_columns(df, aggregation)
        if not selected_columns:
            expanded.append(dict(aggregation))
            continue

        if len(selected_columns) == 1 and aggregation.get("output"):
            expanded.append({**aggregation, "column": selected_columns[0]})
            continue

        for column in selected_columns:
            expanded.append(
                {
                    **aggregation,
                    "column": column,
                    "output": _default_output_name(column, aggregation),
                }
            )

    return expanded


def _run_aggregation(grouped, aggregation: dict) -> pd.Series:
    op = aggregation["op"]
    output = aggregation["output"]

    if op == "size":
        return grouped.size().rename(output)

    if op in _PANDAS_AGG_OPS:
        return grouped[aggregation["column"]].agg(_PANDAS_AGG_OPS[op]).rename(output)

    if op == "collect_list":
        series = grouped[aggregation["column"]]
        if aggregation.get("sort_by"):
            sort_by = resolve_column_label(grouped.obj, aggregation["sort_by"])
            ascending = aggregation.get("ascending", True)
            return grouped.apply(
                lambda frame: frame.sort_values(sort_by, ascending=ascending)[aggregation["column"]].tolist()
            ).rename(output)
        return series.apply(list).rename(output)

    if op == "collect_set":
        series = grouped[aggregation["column"]]
        return series.apply(
            lambda values: sorted(set(values.tolist())) if aggregation.get("unique", True) else list(dict.fromkeys(values.tolist()))
        ).rename(output)

    if op == "collect_rows":
        columns = list(aggregation["columns"])
        if aggregation.get("sort_by"):
            sort_by = resolve_column_label(grouped.obj, aggregation["sort_by"])
            ascending = aggregation.get("ascending", True)
            return grouped.apply(
                lambda frame: _collect_grouped_rows(frame.sort_values(sort_by, ascending=ascending), columns)
            ).rename(output)
        return grouped.apply(lambda frame: _collect_grouped_rows(frame, columns)).rename(output)

    raise ValueError(f"Unsupported Aggregator operation: {op}")


def aggregator(
    df: pd.DataFrame,
    group_keys=None,
    aggregations=None,
    dropna: bool = True,
    sort_by=None,
    ascending=True,
    reset_index: bool = True,
    group_by=None,
    agg_func=None,
    **kwargs,
) -> pd.DataFrame:
    """
    Canonical Aggregator contract:
    {
      "group_keys": [...],
      "aggregations": [...],
      "dropna": true,
      "sort_by": [...],
      "ascending": true,
      "reset_index": true
    }
    """
    params = {
        "group_keys": group_keys,
        "aggregations": aggregations,
        "dropna": dropna,
        "sort_by": sort_by,
        "ascending": ascending,
        "reset_index": reset_index,
        "group_by": group_by,
        "agg_func": agg_func,
    }
    params.update(kwargs)
    normalized = normalize_node_parameters("Aggregator", params)

    working, resolved_group_keys = _resolve_group_keys(df, normalized["group_keys"])
    grouped = working.groupby(resolved_group_keys, dropna=bool(normalized.get("dropna", True)), sort=False)

    aggregation_specs = _expand_aggregation_specs(working, normalized["aggregations"])
    outputs = [
        _run_aggregation(grouped, aggregation)
        for aggregation in aggregation_specs
    ]
    result = pd.concat(outputs, axis=1)
    duplicate_group_outputs = [key for key in resolved_group_keys if key in result.columns]
    if duplicate_group_outputs:
        result = result.drop(columns=duplicate_group_outputs)

    if bool(normalized.get("reset_index", True)):
        result = result.reset_index()
        if result.columns.duplicated().any():
            result = result.loc[:, ~result.columns.duplicated()]

    sort_columns = normalized.get("sort_by")
    if sort_columns:
        result = result.sort_values(
            by=list(sort_columns),
            ascending=normalized.get("ascending", True),
        ).reset_index(drop=True)

    print(
        "[Aggregator] Aggregated using "
        + ", ".join(aggregation["op"] for aggregation in aggregation_specs)
    )
    return result
