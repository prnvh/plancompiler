import pandas as pd


def extract_year(
    df: pd.DataFrame,
    column: str,
    target: str | None = None,
    format: str | None = None,
    errors: str = "raise",
    utc: bool = False,
) -> pd.DataFrame:
    """
    Extracts the year from a datetime-like column.
    Node: ExtractYear
    """
    result = df.copy()
    output_column = target or f"{column}_year"
    result[output_column] = pd.to_datetime(
        result[column],
        format=format,
        errors=errors,
        utc=utc,
    ).dt.year
    print(f"[ExtractYear] Extracted year into '{output_column}'")
    return result
