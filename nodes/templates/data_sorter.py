import pandas as pd


def data_sorter(
    df: pd.DataFrame,
    by: str | list[str],
    ascending: bool | list[bool],
) -> pd.DataFrame:
    """
    Sorts DataFrame rows by one or more columns.
    Node: DataSorter

    Required params:
        by        - column name or list of column names to sort by
        ascending - bool or list of bools matching the sort columns
    """
    sorted_df = df.sort_values(by=by, ascending=ascending)
    print(f"[DataSorter] Sorted by {by} with ascending={ascending}")
    return sorted_df
