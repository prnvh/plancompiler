import pandas as pd


def lower_text(df: pd.DataFrame, column: str, target: str | None = None) -> pd.DataFrame:
    """
    Lowercases a text column.
    Node: LowerText
    """
    result = df.copy()
    output_column = target or column
    result[output_column] = result[column].astype("string").str.lower()
    print(f"[LowerText] Lowercased text into '{output_column}'")
    return result
