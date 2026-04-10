import pandas as pd


def reindex_frame(
    df: pd.DataFrame,
    index=None,
    columns=None,
    fill_value=None,
    fill_method: str | None = None,
    copy: bool | None = None,
) -> pd.DataFrame:
    """
    Reindexes rows or columns to a new label set.
    Node: ReindexFrame
    """
    result = df.reindex(
        index=index,
        columns=columns,
        fill_value=fill_value,
        method=fill_method,
    )
    if copy:
        result = result.copy()
    print("[ReindexFrame] Reindexed DataFrame")
    return result
