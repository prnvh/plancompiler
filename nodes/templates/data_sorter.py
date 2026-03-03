import pandas as pd


def data_sorter(df: pd.DataFrame, by: str, ascending: bool) -> pd.DataFrame:
    """
    Sorts DataFrame by column.
    Node: DataSorter

    Required params:
        by        - column name to sort by
        ascending - True for ascending order, False for descending order
                    Must be explicitly specified — no default.
    """
    sorted_df = df.sort_values(by=by, ascending=ascending)
    direction = "ascending" if ascending else "descending"
    print(f"[DataSorter] Sorted by '{by}' ({direction})")
    return sorted_df