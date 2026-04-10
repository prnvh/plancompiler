import pandas as pd


def fill_nulls(
    df: pd.DataFrame,
    values,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fills missing values with a scalar or mapping.
    Node: FillNulls
    """
    result = df.copy()
    if columns:
        result[columns] = result[columns].fillna(values)
    else:
        result = result.fillna(values)
    print("[FillNulls] Filled null values")
    return result
