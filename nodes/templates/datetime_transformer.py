import pandas as pd

from nodes.contracts import normalize_node_parameters
from nodes.templates.frame_support import (
    resolve_column_label,
    resolve_index_level_reference,
    set_index_level_values,
)


def _to_datetime_series(series: pd.Series, *, fmt: str | None = None, utc: bool | None = None, errors: str | None = None) -> pd.Series:
    return pd.to_datetime(
        series,
        format=fmt,
        utc=bool(utc) if utc is not None else False,
        errors=errors or "raise",
    )


def _remove_timezone(series: pd.Series, strategy: str = "preserve_wall_time") -> pd.Series:
    parsed = pd.to_datetime(series, errors="raise")

    if getattr(parsed.dt, "tz", None) is None:
        return parsed

    if strategy == "convert_utc_then_naive":
        return parsed.dt.tz_convert("UTC").dt.tz_localize(None)

    # Default: preserve local clock time.
    return parsed.dt.tz_localize(None)


def _apply_datetime_operation(df: pd.DataFrame, operation: dict) -> pd.DataFrame:
    op_type = operation["type"]
    column = resolve_column_label(df, operation["column"])
    index_level = operation.get("index_level")

    if op_type == "parse_datetime":
        if column in df.columns:
            df[column] = _to_datetime_series(
                df[column],
                fmt=operation.get("format"),
                utc=operation.get("utc"),
                errors=operation.get("errors"),
            )
        elif index_level is not None:
            level = resolve_index_level_reference(df, index_level)
            values = _to_datetime_series(
                pd.Series(df.index.get_level_values(level)),
                fmt=operation.get("format"),
                utc=operation.get("utc"),
                errors=operation.get("errors"),
            )
            df = set_index_level_values(df, level, values)
        else:
            raise KeyError(column)
        return df

    if op_type == "remove_timezone":
        if column in df.columns:
            df[column] = _remove_timezone(
                df[column],
                strategy=operation.get("strategy", "preserve_wall_time"),
            )
        elif index_level is not None:
            level = resolve_index_level_reference(df, index_level)
            values = _remove_timezone(
                pd.Series(df.index.get_level_values(level)),
                strategy=operation.get("strategy", "preserve_wall_time"),
            )
            df = set_index_level_values(df, level, values)
        else:
            raise KeyError(column)
        return df

    if op_type == "format_datetime":
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="raise").dt.strftime(operation["format"])
        elif index_level is not None:
            level = resolve_index_level_reference(df, index_level)
            values = pd.to_datetime(pd.Series(df.index.get_level_values(level)), errors="raise").dt.strftime(operation["format"])
            df = set_index_level_values(df, level, values)
        else:
            raise KeyError(column)
        return df

    if op_type == "extract_part":
        part = operation["part"]
        if column in df.columns:
            series = pd.to_datetime(df[column], errors="raise")
        elif index_level is not None:
            level = resolve_index_level_reference(df, index_level)
            series = pd.to_datetime(pd.Series(df.index.get_level_values(level)), errors="raise")
        else:
            raise KeyError(column)
        if part == "year":
            df[operation["new_column"]] = series.dt.year
        elif part == "quarter":
            df[operation["new_column"]] = series.dt.quarter
        elif part == "month":
            df[operation["new_column"]] = series.dt.month
        elif part == "day":
            df[operation["new_column"]] = series.dt.day
        elif part == "hour":
            df[operation["new_column"]] = series.dt.hour
        elif part == "minute":
            df[operation["new_column"]] = series.dt.minute
        elif part == "second":
            df[operation["new_column"]] = series.dt.second
        elif part in {"weekday", "dayofweek"}:
            df[operation["new_column"]] = series.dt.dayofweek
        elif part == "date":
            df[operation["new_column"]] = series.dt.date
        elif part == "time":
            df[operation["new_column"]] = series.dt.time
        elif part == "timestamp":
            df[operation["new_column"]] = series.astype("int64")
        else:
            raise ValueError(f"Unsupported DatetimeTransformer extract_part value: {part}")
        return df

    if op_type == "date_diff":
        other = resolve_column_label(df, operation["other_column"])
        target = operation["new_column"]
        unit = operation.get("unit", "days")
        if column in df.columns:
            left = pd.to_datetime(df[column], errors="raise")
        elif index_level is not None:
            left = pd.to_datetime(
                pd.Series(df.index.get_level_values(resolve_index_level_reference(df, index_level))),
                errors="raise",
            )
        else:
            raise KeyError(column)

        other_index_level = operation.get("other_index_level")
        if other in df.columns:
            right = pd.to_datetime(df[other], errors="raise")
        elif other_index_level is not None:
            right = pd.to_datetime(
                pd.Series(df.index.get_level_values(resolve_index_level_reference(df, other_index_level))),
                errors="raise",
            )
        else:
            raise KeyError(other)
        delta = left - right

        if unit == "days":
            df[target] = delta.dt.days
        elif unit == "hours":
            df[target] = delta.dt.total_seconds() / 3600
        elif unit == "minutes":
            df[target] = delta.dt.total_seconds() / 60
        elif unit == "seconds":
            df[target] = delta.dt.total_seconds()
        else:
            raise ValueError(f"Unsupported DatetimeTransformer date_diff unit: {unit}")
        return df

    raise ValueError(f"Unsupported DatetimeTransformer operation: {op_type}")


def datetime_transformer(df: pd.DataFrame, operations=None, **kwargs) -> pd.DataFrame:
    """
    Canonical DatetimeTransformer contract:
    {
      "operations": [
        {"type": "...", ...}
      ]
    }
    """
    params = {"operations": operations}
    params.update(kwargs)
    normalized = normalize_node_parameters("DatetimeTransformer", params)
    transformed = df.copy()

    for operation in normalized.get("operations", []):
        transformed = _apply_datetime_operation(transformed, operation)

    print("[DatetimeTransformer] Applied datetime parsing, timezone, or formatting transformations")
    return transformed
