import pandas as pd


def to_datetime(
    df: pd.DataFrame,
    column: str,
    target: str | None = None,
    format: str | None = None,
    errors: str = "raise",
    utc: bool = False,
    dayfirst: bool = False,
    yearfirst: bool = False,
) -> pd.DataFrame:
    """
    Parses a column into pandas datetime values.
    Node: ToDatetime
    """
    result = df.copy()
    output_column = target or column
    result[output_column] = pd.to_datetime(
        result[column],
        format=format,
        errors=errors,
        utc=utc,
        dayfirst=dayfirst,
        yearfirst=yearfirst,
    )
    print(f"[ToDatetime] Parsed '{column}' into '{output_column}'")
    return result
