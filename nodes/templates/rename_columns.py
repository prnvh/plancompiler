import pandas as pd


def rename_columns(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """
    Renames DataFrame columns using a mapping.
    Node: RenameColumns
    """
    result = df.rename(columns=columns)
    print(f"[RenameColumns] Renamed columns: {list(columns.keys())}")
    return result
