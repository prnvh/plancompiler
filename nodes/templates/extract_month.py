import pandas as pd


def extract_month(
    df: pd.DataFrame,
    column: str,
    target: str | None = None,
    format: str | None = None,
    errors: str = "raise",
    utc: bool = False,
) -> pd.DataFrame:
    """
    Extracts the month from a datetime-like column.
    Node: ExtractMonth
    """
    result = df.copy()
    output_column = target or f"{column}_month"
    result[output_column] = pd.to_datetime(
        result[column],
        format=format,
        errors=errors,
        utc=utc,
    ).dt.month
    print(f"[ExtractMonth] Extracted month into '{output_column}'")
    return result
