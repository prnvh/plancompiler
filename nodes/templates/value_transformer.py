import pandas as pd

from nodes.contracts import normalize_node_parameters
from nodes.templates.frame_support import resolve_column_label, resolve_mapping_keys


def _resolve_target_columns(df: pd.DataFrame, reference) -> list:
    if isinstance(reference, str) and reference.strip() == "*":
        return list(df.columns)
    return [resolve_column_label(df, reference)]


def _apply_low_frequency_rule(
    df: pd.DataFrame,
    *,
    column: str,
    threshold: int,
    replacement,
    preserve_values=None,
    preserve_na: bool = False,
) -> pd.DataFrame:
    counts = df[column].value_counts(dropna=False)
    keep_values = set(counts[counts >= threshold].index.tolist())
    if preserve_values:
        keep_values.update(preserve_values)

    keep_mask = df[column].isin(keep_values)
    if preserve_na:
        keep_mask = keep_mask | df[column].isna()

    df[column] = df[column].where(keep_mask, replacement)
    return df


def _apply_value_operation(df: pd.DataFrame, operation: dict) -> pd.DataFrame:
    op_type = operation["type"]

    if op_type == "replace_infrequent":
        if operation.get("column_rules"):
            shared_preserve_values = operation.get("preserve_values")
            for rule in operation["column_rules"]:
                df = _apply_low_frequency_rule(
                    df,
                    column=rule["column"],
                    threshold=int(rule["threshold"]),
                    replacement=rule["replacement"],
                    preserve_values=rule.get("preserve_values", shared_preserve_values),
                    preserve_na=bool(operation.get("preserve_na", False)),
                )
            return df

        preserve_values = operation.get("preserve_values")
        for column in operation.get("columns", []):
            column = resolve_column_label(df, column)
            column_preserve_values = preserve_values.get(column) if isinstance(preserve_values, dict) else preserve_values
            df = _apply_low_frequency_rule(
                df,
                column=column,
                threshold=int(operation["threshold"]),
                replacement=operation["replacement"],
                preserve_values=column_preserve_values,
                preserve_na=bool(operation.get("preserve_na", False)),
            )
        return df

    if op_type == "replace_values":
        column = resolve_column_label(df, operation["column"])
        if operation.get("mapping") is not None:
            df[column] = df[column].replace(resolve_mapping_keys(df[column], operation["mapping"]))
            return df

        if "old_value" in operation and "new_value" in operation:
            df[column] = df[column].replace(operation["old_value"], operation["new_value"])
            return df

        return df

    if op_type == "map_values":
        column = resolve_column_label(df, operation["column"])
        mapped = df[column].map(resolve_mapping_keys(df[column], operation["mapping"]))
        if "default" in operation:
            mapped = mapped.fillna(operation["default"])
        df[column] = mapped
        return df

    if op_type == "factorize_values":
        column = resolve_column_label(df, operation["column"])
        target = operation.get("new_column") or column
        sort = bool(operation.get("sort", False))
        start = int(operation.get("start", 0))
        codes, _uniques = pd.factorize(df[column], sort=sort)
        coded = pd.Series(codes, index=df.index)
        coded = coded.where(coded < 0, coded + start)
        df[target] = coded
        return df

    if op_type == "fill_missing":
        if operation.get("mapping") is not None:
            return df.fillna(operation["mapping"])

        column = resolve_column_label(df, operation.get("column")) if operation.get("column") is not None else None
        if column:
            df[column] = df[column].fillna(operation.get("value"))
        return df

    if op_type == "normalize_text":
        style = operation["style"]
        for column in _resolve_target_columns(df, operation["column"]):
            series = df[column].astype(str)
            if style == "lower":
                df[column] = series.str.lower()
            elif style == "upper":
                df[column] = series.str.upper()
            elif style == "strip":
                df[column] = series.str.strip()
        return df

    if op_type in {"replace_substring", "remove_substring"}:
        replace_kwargs = {
            "pat": operation["old"],
            "repl": operation.get("new", ""),
            "regex": bool(operation.get("regex", False)),
        }
        if "case" in operation:
            replace_kwargs["case"] = operation["case"]
        target_columns = _resolve_target_columns(df, operation["column"])
        if isinstance(operation["column"], str) and operation["column"].strip() == "*":
            target_columns = [
                column
                for column in target_columns
                if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column])
            ]
        for column in target_columns:
            df[column] = df[column].astype(str).str.replace(**replace_kwargs)
        return df

    if op_type == "strip_prefix":
        column = resolve_column_label(df, operation["column"])
        prefix = str(operation["prefix"])
        series = df[column].astype(str)
        df[column] = series.map(
            lambda value: value[len(prefix):] if value.startswith(prefix) else value
        )
        return df

    if op_type == "round_values":
        column = resolve_column_label(df, operation["column"])
        target = operation.get("new_column") or column
        decimals = int(operation.get("decimals", 0))
        df[target] = pd.to_numeric(df[column], errors="coerce").round(decimals)
        return df

    if op_type == "coerce_numeric":
        column = resolve_column_label(df, operation["column"])
        target = operation.get("new_column") or column
        errors = operation.get("errors", "coerce")
        df[target] = pd.to_numeric(df[column], errors=errors)
        return df

    if op_type == "clip_values":
        column = resolve_column_label(df, operation["column"])
        target = operation.get("new_column") or column
        series = pd.to_numeric(df[column], errors=operation.get("errors", "coerce"))
        lower = operation.get("lower")
        upper = operation.get("upper")
        df[target] = series.clip(lower=lower, upper=upper)
        return df

    if op_type == "parse_duration_text":
        column = resolve_column_label(df, operation["column"])
        target = operation.get("new_column") or column
        result_kind = operation.get("result", "number")
        unit_map = {
            "day": 1,
            "days": 1,
            "week": 7,
            "weeks": 7,
            "month": 30,
            "months": 30,
            "year": 365,
            "years": 365,
        }
        unit_map.update(operation.get("unit_map") or {})

        extracted = df[column].astype(str).str.extract(r"^\s*(?P<number>-?\d+(?:\.\d+)?)\s+(?P<unit>[A-Za-z_]+)\s*$")
        numbers = pd.to_numeric(extracted["number"], errors=operation.get("errors", "coerce"))
        units = extracted["unit"].str.lower()

        if result_kind == "number":
            df[target] = numbers
            return df

        day_factors = units.map(resolve_mapping_keys(units, unit_map))
        days = numbers * pd.to_numeric(day_factors, errors="coerce")
        if result_kind == "days":
            df[target] = days
            return df
        if result_kind == "timedelta":
            df[target] = pd.to_timedelta(days, unit="D")
            return df
        raise ValueError(f"Unsupported parse_duration_text result: {result_kind}")

    raise ValueError(f"Unsupported ValueTransformer operation: {op_type}")


def value_transformer(df: pd.DataFrame, operations=None, **kwargs) -> pd.DataFrame:
    """
    Canonical ValueTransformer contract:
    {
      "operations": [
        {"type": "...", ...}
      ]
    }
    """
    params = {"operations": operations}
    params.update(kwargs)
    normalized = normalize_node_parameters("ValueTransformer", params)
    transformed = df.copy()

    for operation in normalized.get("operations", []):
        transformed = _apply_value_operation(transformed, operation)

    print("[ValueTransformer] Applied value-level transformations")
    return transformed
