import pandas as pd


def date_diff(
    df: pd.DataFrame,
    start: str,
    end: str,
    target: str,
    unit: str = "days",
    absolute: bool = False,
    format: str | None = None,
    errors: str = "raise",
    utc: bool = False,
) -> pd.DataFrame:
    """
    Computes differences between two datetime-like columns.
    Node: DateDiff
    """
    result = df.copy()
    delta = pd.to_datetime(
        result[end],
        format=format,
        errors=errors,
        utc=utc,
    ) - pd.to_datetime(
        result[start],
        format=format,
        errors=errors,
        utc=utc,
    )
    if absolute:
        delta = delta.abs()

    if unit == "timedelta":
        output = delta
    elif unit == "days":
        output = delta.dt.days
    elif unit == "hours":
        output = delta.dt.total_seconds() / 3600
    elif unit == "minutes":
        output = delta.dt.total_seconds() / 60
    elif unit == "seconds":
        output = delta.dt.total_seconds()
    else:
        raise ValueError(f"Unsupported DateDiff unit: {unit}")

    result[target] = output
    print(f"[DateDiff] Computed date difference into '{target}'")
    return result
