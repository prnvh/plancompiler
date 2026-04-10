import pandas as pd


def upper_text(df: pd.DataFrame, column: str, target: str | None = None) -> pd.DataFrame:
    """
    Uppercases a text column.
    Node: UpperText
    """
    result = df.copy()
    output_column = target or column
    result[output_column] = result[column].astype("string").str.upper()
    print(f"[UpperText] Uppercased text into '{output_column}'")
    return result
