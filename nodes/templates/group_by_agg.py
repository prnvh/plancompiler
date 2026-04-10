import pandas as pd


_COUNT_ALIASES = {"count", "count(*)", "row_count", "rows", "size"}
_AGGREGATION_ALIASES = {"avg": "mean"}


def _normalize_group_keys(group_by: str | list[str]) -> list[str]:
    if isinstance(group_by, str):
        return [group_by]
    if isinstance(group_by, list) and group_by:
        return group_by
    raise ValueError("GroupByAgg requires one or more group_by columns.")


def _canonical_agg_name(agg_name):
    if not isinstance(agg_name, str):
        return agg_name
    lowered = agg_name.strip().lower()
    if lowered in _COUNT_ALIASES:
        return "size"
    return _AGGREGATION_ALIASES.get(lowered, lowered)


def _normalize_aggregation_item(item) -> dict:
    if not isinstance(item, dict):
        raise ValueError("Each GroupByAgg aggregation must be a dict.")

    if "output" in item and ("agg" in item or "func" in item):
        normalized = dict(item)
        normalized["agg"] = _canonical_agg_name(
            normalized.get("agg") or normalized.pop("func", None)
        )
        if normalized.get("agg") == "size":
            normalized.pop("column", None)
        return normalized

    if len(item) != 1:
        raise ValueError("Unsupported GroupByAgg aggregation spec.")

    key, value = next(iter(item.items()))
    output_name = str(key)
    key_lower = output_name.strip().lower()

    if isinstance(value, dict):
        normalized = {
            "output": value.get("output") or output_name,
            "agg": _canonical_agg_name(value.get("agg") or value.get("func")),
        }
        column = value.get("column")
        if normalized["agg"] != "size" and column is not None:
            normalized["column"] = column
        return normalized

    if isinstance(value, (list, tuple)) and len(value) == 2:
        column, agg_name = value
        normalized = {
            "output": output_name,
            "agg": _canonical_agg_name(agg_name),
        }
        if normalized["agg"] != "size":
            normalized["column"] = column
        return normalized

    if isinstance(value, str):
        value_lower = value.strip().lower()

        if key_lower in _COUNT_ALIASES:
            if value_lower not in {"count", "count(*)", "size"}:
                derived_output = value
            elif key_lower not in {"count", "count(*)", "size"}:
                derived_output = output_name
            else:
                derived_output = "count"
            return {"output": derived_output, "agg": "size"}

        if value_lower in _COUNT_ALIASES:
            return {"output": output_name, "agg": "size"}

    raise ValueError("Unsupported GroupByAgg aggregation spec.")


def _normalize_aggregations(aggregations) -> list[dict]:
    if isinstance(aggregations, list):
        normalized = [_normalize_aggregation_item(item) for item in aggregations]
    elif isinstance(aggregations, dict):
        normalized = []
        for output, spec in aggregations.items():
            if isinstance(spec, dict):
                normalized.append(
                    _normalize_aggregation_item(
                        {
                            output: {
                                "column": spec.get("column"),
                                "agg": spec.get("agg") or spec.get("func"),
                                "output": spec.get("output"),
                            }
                        }
                    )
                )
            elif isinstance(spec, (list, tuple)) and len(spec) == 2:
                normalized.append(_normalize_aggregation_item({output: spec}))
            elif isinstance(spec, str):
                normalized.append(_normalize_aggregation_item({output: spec}))
            else:
                raise ValueError("Unsupported GroupByAgg aggregation spec.")
    else:
        raise ValueError("GroupByAgg aggregations must be a list or dict.")

    if not normalized:
        raise ValueError("GroupByAgg requires at least one aggregation.")

    return normalized


def group_by_agg(
    df: pd.DataFrame,
    group_by: str | list[str],
    aggregations,
    sort: bool = False,
    dropna: bool = True,
    observed: bool = False,
    as_index: bool = False,
) -> pd.DataFrame:
    """
    Groups a DataFrame by one or more keys and returns named aggregations.
    Aggregations are typically a list of
    {'output': 'total_sales', 'column': 'sales', 'agg': 'sum'}.
    Node: GroupByAgg
    """
    group_keys = _normalize_group_keys(group_by)
    agg_specs = _normalize_aggregations(aggregations)
    grouped = df.groupby(group_keys, sort=sort, dropna=dropna, observed=observed)

    output_columns = []
    for spec in agg_specs:
        output = spec.get("output")
        column = spec.get("column")
        agg_name = spec.get("agg")

        if not output or not agg_name:
            raise ValueError("Each GroupByAgg aggregation requires 'output' and 'agg'.")

        if agg_name == "size":
            series = grouped.size().rename(output)
        else:
            if not column:
                raise ValueError(f"GroupByAgg aggregation '{output}' requires 'column'.")
            series = grouped[column].agg(agg_name).rename(output)

        output_columns.append(series)

    result = pd.concat(output_columns, axis=1)
    if not as_index:
        result = result.reset_index()

    print(
        f"[GroupByAgg] Aggregated by {group_keys} with {len(agg_specs)} output columns"
    )
    return result
