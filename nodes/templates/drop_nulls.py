import pandas as pd


def drop_nulls(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    how: str = "any",
) -> pd.DataFrame:
    """
    Drops rows with missing values.
    Node: DropNulls
    """
    result = df.dropna(subset=subset, how=how)
    print("[DropNulls] Dropped rows with null values")
    return result
