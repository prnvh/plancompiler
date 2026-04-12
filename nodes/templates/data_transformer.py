import operator
import re

import pandas as pd

from nodes.templates.date_range_expander import date_range_expander
from nodes.templates.expression_support import (
    evaluate_frame_expression,
    evaluate_row_expression,
)
from nodes.templates.frame_support import (
    resolve_column_label,
    resolve_column_labels,
    resolve_mapping_keys,
)
from nodes.templates.value_counts_reporter import value_counts_reporter


_COUNT_COMPARATORS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

def _apply_cast(df: pd.DataFrame, column: str, target_type: str, spec: dict | None = None) -> pd.DataFrame:
    spec = spec or {}
    column = resolve_column_label(df, column)
    if "datetime" in str(target_type).lower():
        df[column] = pd.to_datetime(
            df[column],
            errors=spec.get("errors", "raise"),
            format=spec.get("format"),
            utc=spec.get("utc", False),
        )
        return df

    df[column] = df[column].astype(target_type)
    return df


def _apply_assign(df: pd.DataFrame, column: str, value=None, expression: str | None = None) -> pd.DataFrame:
    resolved_column = resolve_column_label(df, column)
    if resolved_column in df.columns:
        column = resolved_column
    if expression:
        try:
            df[column] = df.eval(expression)
        except Exception:
            try:
                df[column] = evaluate_frame_expression(df, expression)
            except Exception:
                def evaluate_row(row):
                    return evaluate_row_expression(row, expression, df=df)

                df[column] = df.apply(evaluate_row, axis=1)
    else:
        df[column] = value
    return df


def _apply_circular_shift(df: pd.DataFrame, column: str, shift: int) -> pd.DataFrame:
    column = resolve_column_label(df, column)
    values = df[column].tolist()
    if not values:
        return df

    offset = shift % len(values)
    if offset == 0:
        return df

    df[column] = values[-offset:] + values[:-offset]
    return df


def _shift_non_nulls_left(df: pd.DataFrame, columns: list[str] | None = None, fill_value=None) -> pd.DataFrame:
    target_columns = resolve_column_labels(df, columns) if columns is not None else df.columns.tolist()

    def _compress_row(row):
        values = [row[column] for column in target_columns if pd.notna(row[column])]
        values.extend([fill_value] * (len(target_columns) - len(values)))
        for index, column in enumerate(target_columns):
            row[column] = values[index]
        return row

    return df.apply(_compress_row, axis=1)


def _shift_nulls_to_top_per_column(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    target_columns = resolve_column_labels(df, columns) if columns is not None else df.columns.tolist()
    result = df.copy()

    for column in target_columns:
        series = result[column]
        reordered = pd.concat([series[series.isna()], series[series.notna()]], ignore_index=True)
        result[column] = reordered.reindex(range(len(result)))

    return result


def _concatenate_columns(df: pd.DataFrame, source_columns: list[str], *, separator: str = "", strip: bool = False) -> pd.Series:
    combined = df[source_columns].fillna("").astype(str).agg(separator.join, axis=1)
    return combined.str.strip() if strip else combined


def _expand_date_range_per_group(
    df: pd.DataFrame,
    group_by,
    date_column: str,
    freq: str = "D",
    sort: bool = True,
) -> pd.DataFrame:
    group_keys = [group_by] if isinstance(group_by, str) else list(group_by)
    working = df.copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="raise")

    expanded_frames = []
    for group_key, group_frame in working.groupby(group_keys, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_values = dict(zip(group_keys, group_key))
        date_range = pd.date_range(
            group_frame[date_column].min(),
            group_frame[date_column].max(),
            freq=freq,
        )
        expanded = pd.DataFrame({date_column: date_range})
        for key, value in group_values.items():
            expanded[key] = value
        merged = expanded.merge(group_frame, on=group_keys + [date_column], how="left")
        expanded_frames.append(merged)

    result = pd.concat(expanded_frames, ignore_index=True)
    if sort:
        result = result.sort_values(by=group_keys + [date_column]).reset_index(drop=True)
    return result


def _apply_value_count_replace(
    df: pd.DataFrame,
    column: str,
    condition: str,
    new_value,
    *,
    replace_mode: str = "non_matching",
) -> pd.DataFrame:
    match = re.fullmatch(r"value_counts\(\)\s*([<>]=?|==|!=)\s*(-?\d+)", condition.strip())
    if not match:
        raise ValueError(f"Unsupported value-count condition: {condition}")

    comparator = _COUNT_COMPARATORS[match.group(1)]
    threshold = int(match.group(2))
    counts = df[column].value_counts(dropna=False)
    matching_values = counts[comparator(counts, threshold)].index
    keep_mask = df[column].isin(matching_values)

    if replace_mode == "matching":
        replace_mask = keep_mask
    else:
        replace_mask = ~keep_mask

    df[column] = df[column].where(~replace_mask, new_value)
    return df


def _normalize_operations(
    *,
    operations: dict | None = None,
    transformations: list[dict] | None = None,
    rename: dict | None = None,
    filter_expr: str | None = None,
    cast: dict | None = None,
    fillna: dict | None = None,
    assign: dict | None = None,
    replace: dict | list[dict] | None = None,
    to_datetime: dict | list[str] | None = None,
    **kwargs,
) -> tuple[dict, list[dict]]:
    merged_operations: dict = dict(operations) if isinstance(operations, dict) else {}
    op_aliases = {
        "rename": rename,
        "filter": filter_expr or kwargs.get("filter") or kwargs.get("query"),
        "cast": cast or kwargs.get("dtype_mapping"),
        "fillna": fillna,
        "assign": assign,
    }

    for key, value in op_aliases.items():
        if value is not None:
            merged_operations[key] = value

    normalized_transformations: list = []

    if isinstance(operations, list):
        normalized_transformations.extend(operations)
    elif isinstance(operations, str):
        normalized_transformations.append(operations)

    if isinstance(transformations, list):
        normalized_transformations.extend(transformations)
    elif transformations is not None:
        normalized_transformations.append(transformations)

    if replace:
        if isinstance(replace, list):
            normalized_transformations.extend(replace)
        else:
            for column, value in replace.items():
                normalized_transformations.append(
                    {
                        "operation": "replace_values",
                        "column": column,
                        "mapping": value if isinstance(value, dict) else {column: value},
                    }
                )

    if to_datetime:
        if isinstance(to_datetime, list):
            for column in to_datetime:
                normalized_transformations.append(
                    {"operation": "cast", "column": column, "to": "datetime64[ns]"}
                )
        else:
            for column, spec in to_datetime.items():
                transform = {
                    "operation": "cast",
                    "column": column,
                    "to": "datetime64[ns]",
                }
                if isinstance(spec, dict):
                    transform.update(spec)
                normalized_transformations.append(transform)

    for extra_key in (
        "operation",
        "transformation",
        "code",
        "script",
        "example_pandas_code",
        "transform_code",
        "description",
        "method",
    ):
        if kwargs.get(extra_key) is not None:
            normalized_transformations.append(kwargs[extra_key])

    return merged_operations, normalized_transformations


def _apply_simple_operations(df: pd.DataFrame, operations: dict) -> pd.DataFrame:
    if "rename" in operations:
        df = df.rename(columns=operations["rename"])

    if "filter" in operations and operations["filter"]:
        df = df.query(operations["filter"], engine="python")

    if "cast" in operations:
        for column, target_type in operations["cast"].items():
            df = _apply_cast(df, column, target_type)

    if "fillna" in operations:
        fill_spec = operations["fillna"]
        if isinstance(fill_spec, dict):
            df = df.fillna(fill_spec)

    if "assign" in operations:
        for column, value in operations["assign"].items():
            if isinstance(value, dict):
                df = _apply_assign(
                    df,
                    column,
                    value=value.get("value"),
                    expression=value.get("expression"),
                )
            else:
                df = _apply_assign(df, column, value=value)

    return df


def _apply_transformation(df: pd.DataFrame, transform: dict) -> pd.DataFrame:
    if not isinstance(transform, dict):
        return df

    operation = (
        transform.get("operation")
        or transform.get("type")
        or transform.get("action")
    )
    if not operation:
        return df

    operation = str(operation).strip().lower()
    operation_aliases = {
        "rename_columns": "rename",
        "cast_column": "cast",
        "drop_duplicate_rows": "drop_duplicate_rows",
        "melt": "melt",
        "circular_shift": "circular_shift",
        "shift_non_nulls_left": "shift_non_nulls_left",
        "shift_nulls_to_top_per_column": "shift_nulls_to_top_per_column",
        "expand_date_range_per_group": "expand_date_range_per_group",
        "derive_column": "derive",
        "derive_first_matching_label": "derive_first_matching_label",
        "derive_matching_labels": "derive_matching_labels",
        "replace_values": "replace_values",
        "fill_missing": "fillna",
        "map_values": "map",
        "parse_datetime": "cast",
    }
    operation = operation_aliases.get(operation, operation)

    if operation == "no_change":
        return df

    if operation == "rename":
        mapping = transform.get("mapping")
        if not mapping and transform.get("from") and transform.get("to"):
            mapping = {transform["from"]: transform["to"]}
        if mapping:
            df = df.rename(columns=mapping)
        return df

    if operation == "drop_duplicate_rows":
        return df.drop_duplicates(
            subset=resolve_column_labels(df, transform.get("subset")),
            keep=transform.get("keep", "first"),
            ignore_index=bool(transform.get("ignore_index", False)),
        )

    if operation == "melt":
        id_vars = resolve_column_labels(df, transform.get("id_vars"))
        value_vars = resolve_column_labels(df, transform.get("value_vars"))
        value_name = transform.get("value_name", "value")
        preserve_row_order = bool(transform.get("preserve_row_order", False))
        working = df.copy()
        row_order_column = "__row_order__"
        if preserve_row_order:
            working[row_order_column] = range(len(working))
            id_vars = list(id_vars or []) + [row_order_column]

        melted = working.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=transform.get("var_name"),
            value_name=value_name,
            ignore_index=bool(transform.get("ignore_index", True)),
        )
        if transform.get("dropna"):
            melted = melted[melted[value_name].notna()]
        if preserve_row_order:
            variable_column = transform.get("var_name") or "variable"
            order_columns = [row_order_column]
            if value_vars:
                melted[variable_column] = pd.Categorical(
                    melted[variable_column],
                    categories=list(value_vars),
                    ordered=True,
                )
                order_columns.append(variable_column)
            melted = melted.sort_values(order_columns, kind="stable").drop(columns=[row_order_column])
        if bool(transform.get("ignore_index", True)):
            melted = melted.reset_index(drop=True)
        return melted

    if operation == "cast":
        column = transform["column"]
        target_type = (
            transform.get("to")
            or transform.get("dtype")
            or transform.get("target_type")
            or "datetime64[ns]"
        )
        if not target_type:
            raise ValueError(f"Cast transformation for column '{column}' is missing a target type.")
        return _apply_cast(df, column, target_type, transform)

    if operation == "circular_shift":
        return _apply_circular_shift(df, transform["column"], int(transform["shift"]))

    if operation == "shift_non_nulls_left":
        return _shift_non_nulls_left(
            df.copy(),
            columns=transform.get("columns"),
            fill_value=transform.get("fill_value"),
        )

    if operation == "shift_nulls_to_top_per_column":
        return _shift_nulls_to_top_per_column(
            df,
            columns=transform.get("columns"),
        )

    if operation == "expand_date_range_per_group":
        return date_range_expander(
            df,
            group_keys=transform["group_by"],
            date_column=transform["date_column"],
            freq=transform.get("freq", "D"),
            range_scope=transform.get("range_scope", "group"),
            fill_values=transform.get("fill_values"),
            fill_strategies=transform.get("fill_strategies"),
            date_format=transform.get("date_format"),
            sort=bool(transform.get("sort", True)),
        )

    if operation == "value_counts_report":
        return value_counts_reporter(
            df,
            columns=transform.get("columns"),
            dropna=bool(transform.get("dropna", False)),
            sort=bool(transform.get("sort", True)),
            include_header=bool(transform.get("include_header", True)),
        )

    if operation in {"assign", "set", "derive"}:
        column = transform.get("new_column") or transform.get("column")
        if not column:
            raise ValueError("Assign transformation is missing a target column.")
        return _apply_assign(
            df,
            column,
            value=transform.get("value"),
            expression=transform.get("expression"),
        )

    if operation == "derive_first_matching_label":
        column = transform.get("new_column") or transform.get("column")
        if not column:
            raise ValueError("derive_first_matching_label transformation is missing a target column.")
        source_columns = resolve_column_labels(df, transform.get("source_columns") or [])
        labels = {
            resolve_column_label(df, key): value
            for key, value in (transform.get("labels") or {}).items()
        }
        match_value = transform.get("match_value", 1)
        default = transform.get("default")

        df[column] = df[source_columns].apply(
            lambda row: next(
                (
                    labels.get(source_column, source_column)
                    for source_column, value in row.items()
                    if value == match_value
                ),
                default,
            ),
            axis=1,
        )
        return df

    if operation == "derive_matching_labels":
        column = transform.get("new_column") or transform.get("column")
        if not column:
            raise ValueError("derive_matching_labels transformation is missing a target column.")
        source_columns = resolve_column_labels(df, transform.get("source_columns") or [])
        labels = {
            resolve_column_label(df, key): value
            for key, value in (transform.get("labels") or {}).items()
        }
        match_value = transform.get("match_value", 1)

        df[column] = df[source_columns].apply(
            lambda row: [
                labels.get(source_column, source_column)
                for source_column, value in row.items()
                if value == match_value
            ],
            axis=1,
        )
        return df

    if operation in {"replace", "replace_values"}:
        column = resolve_column_label(df, transform["column"])
        if "condition" in transform and "new_value" in transform and "value_counts()" in str(transform["condition"]):
            return _apply_value_count_replace(
                df,
                column,
                str(transform["condition"]),
                transform["new_value"],
                replace_mode=transform.get("replace_mode", "non_matching"),
            )

        mapping = transform.get("mapping")
        if mapping is not None:
            df[column] = df[column].replace(resolve_mapping_keys(df[column], mapping))
            return df

        old_value = transform.get("old_value")
        if old_value is not None and "new_value" in transform:
            df[column] = df[column].replace(old_value, transform["new_value"])
            return df

        raise ValueError(f"Replace transformation for column '{column}' is missing mapping or new_value.")

    if operation == "fillna":
        if transform.get("mapping") is not None:
            return df.fillna(transform["mapping"])

        column = resolve_column_label(df, transform["column"])
        df[column] = df[column].fillna(transform.get("value"))
        return df

    if operation == "map":
        source_column = resolve_column_label(df, transform.get("source_column") or transform.get("column"))
        target_column = transform.get("new_column") or transform.get("column") or source_column
        mapping = transform.get("mapping", {})
        default = transform.get("default")
        mapped = df[source_column].map(resolve_mapping_keys(df[source_column], mapping))
        if default is not None:
            mapped = mapped.fillna(default)
        df[target_column] = mapped
        return df

    if operation == "concatenate_columns":
        source_columns = resolve_column_labels(df, transform["source_columns"])
        target_column = transform["new_column"]
        df[target_column] = _concatenate_columns(
            df,
            source_columns,
            separator=str(transform.get("separator", "")),
            strip=bool(transform.get("strip", False)),
        )
        return df

    if operation == "explode_column":
        source_column = resolve_column_label(df, transform["source_column"])
        result = df.explode(source_column, ignore_index=bool(transform.get("ignore_index", False)))
        target_column = transform.get("new_column")
        if target_column and target_column != source_column:
            result[target_column] = result[source_column]
        return result

    if operation == "lower":
        column = resolve_column_label(df, transform["column"])
        df[column] = df[column].astype(str).str.lower()
        return df

    if operation == "upper":
        column = resolve_column_label(df, transform["column"])
        df[column] = df[column].astype(str).str.upper()
        return df

    if operation == "strip":
        column = resolve_column_label(df, transform["column"])
        df[column] = df[column].astype(str).str.strip()
        return df

    if operation == "coerce_numeric":
        column = resolve_column_label(df, transform["column"])
        target_column = transform.get("new_column") or column
        df[target_column] = pd.to_numeric(df[column], errors=transform.get("errors", "coerce"))
        return df

    if operation == "clip_values":
        column = resolve_column_label(df, transform["column"])
        target_column = transform.get("new_column") or column
        series = pd.to_numeric(df[column], errors=transform.get("errors", "coerce"))
        df[target_column] = series.clip(
            lower=transform.get("lower"),
            upper=transform.get("upper"),
        )
        return df

    if operation == "remove_timezone":
        column = resolve_column_label(df, transform["column"])
        series = pd.to_datetime(df[column], errors=transform.get("errors", "raise"))
        if getattr(series.dt, "tz", None) is not None:
            strategy = transform.get("strategy", "preserve_wall_time")
            if strategy == "convert_utc_then_naive":
                df[column] = series.dt.tz_convert("UTC").dt.tz_localize(None)
            else:
                df[column] = series.dt.tz_localize(None)
        else:
            df[column] = series
        return df

    if operation == "format_datetime":
        column = resolve_column_label(df, transform["column"])
        df[column] = pd.to_datetime(df[column], errors=transform.get("errors", "raise")).dt.strftime(
            transform["format"]
        )
        return df

    if operation == "extract_part":
        column = resolve_column_label(df, transform["column"])
        target_column = transform["new_column"]
        series = pd.to_datetime(df[column], errors=transform.get("errors", "raise"))
        part = transform["part"]
        if part == "year":
            df[target_column] = series.dt.year
        elif part == "quarter":
            df[target_column] = series.dt.quarter
        elif part == "month":
            df[target_column] = series.dt.month
        elif part == "day":
            df[target_column] = series.dt.day
        elif part == "hour":
            df[target_column] = series.dt.hour
        elif part == "minute":
            df[target_column] = series.dt.minute
        elif part == "second":
            df[target_column] = series.dt.second
        elif part in {"weekday", "dayofweek"}:
            df[target_column] = series.dt.dayofweek
        elif part == "date":
            df[target_column] = series.dt.date
        elif part == "time":
            df[target_column] = series.dt.time
        elif part == "timestamp":
            df[target_column] = series.astype("int64")
        else:
            raise ValueError(f"Unsupported extract_part transformation value: {part}")
        return df

    if operation == "date_diff":
        left_column = resolve_column_label(df, transform["column"])
        right_column = resolve_column_label(df, transform["other_column"])
        left = pd.to_datetime(df[left_column], errors=transform.get("errors", "raise"))
        right = pd.to_datetime(df[right_column], errors=transform.get("errors", "raise"))
        delta = left - right
        target_column = transform["new_column"]
        unit = transform.get("unit", "days")
        if unit == "days":
            df[target_column] = delta.dt.days
        elif unit == "hours":
            df[target_column] = delta.dt.total_seconds() / 3600
        elif unit == "minutes":
            df[target_column] = delta.dt.total_seconds() / 60
        elif unit == "seconds":
            df[target_column] = delta.dt.total_seconds()
        else:
            raise ValueError(f"Unsupported date_diff transformation unit: {unit}")
        return df

    raise ValueError(f"Unsupported DataTransformer operation: {operation}")


def data_transformer(
    df: pd.DataFrame,
    operations: dict | None = None,
    transformations: list[dict] | None = None,
    rename: dict | None = None,
    filter_expr: str | None = None,
    cast: dict | None = None,
    fillna: dict | None = None,
    assign: dict | None = None,
    replace: dict | list[dict] | None = None,
    to_datetime: dict | list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    General-purpose DataFrame cleanup and light transformation node.

    Supported styles:
    - operations={"rename": {...}, "filter": "...", "cast": {...}, ...}
    - transformations=[{"operation": "cast", ...}, {"operation": "replace", ...}]
    - direct keyword aliases like rename=..., cast=..., fillna=...
    """
    merged_operations, normalized_transformations = _normalize_operations(
        operations=operations,
        transformations=transformations,
        rename=rename,
        filter_expr=filter_expr,
        cast=cast,
        fillna=fillna,
        assign=assign,
        replace=replace,
        to_datetime=to_datetime,
        **kwargs,
    )

    transformed = df.copy()
    transformed = _apply_simple_operations(transformed, merged_operations)

    for transform in normalized_transformations:
        transformed = _apply_transformation(transformed, transform)

    print("[DataTransformer] Applied general DataFrame cleanup and transformation steps")
    return transformed
