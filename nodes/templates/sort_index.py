import pandas as pd


def sort_index(
    df: pd.DataFrame,
    axis: int = 0,
    level=None,
    ascending: bool = True,
    na_position: str = "last",
    sort_remaining: bool = True,
) -> pd.DataFrame:
    """
    Sorts a DataFrame by its index.
    Node: SortIndex
    """
    result = df.sort_index(
        axis=axis,
        level=level,
        ascending=ascending,
        na_position=na_position,
        sort_remaining=sort_remaining,
    )
    print("[SortIndex] Sorted DataFrame index")
    return result
