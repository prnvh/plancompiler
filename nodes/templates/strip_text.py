import pandas as pd


def strip_text(
    df: pd.DataFrame,
    column: str,
    target: str | None = None,
    chars: str | None = None,
) -> pd.DataFrame:
    """
    Strips whitespace or characters from a text column.
    Node: StripText
    """
    result = df.copy()
    output_column = target or column
    result[output_column] = result[column].astype("string").str.strip(chars)
    print(f"[StripText] Stripped text into '{output_column}'")
    return result
