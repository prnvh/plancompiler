import pandas as pd


def extract_day(
    df: pd.DataFrame,
    column: str,
    target: str | None = None,
    format: str | None = None,
    errors: str = "raise",
    utc: bool = False,
) -> pd.DataFrame:
    """
    Extracts the day of month from a datetime-like column.
    Node: ExtractDay
    """
    result = df.copy()
    output_column = target or f"{column}_day"
    result[output_column] = pd.to_datetime(
        result[column],
        format=format,
        errors=errors,
        utc=utc,
    ).dt.day
    print(f"[ExtractDay] Extracted day into '{output_column}'")
    return result
