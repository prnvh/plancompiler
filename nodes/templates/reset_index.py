import pandas as pd


def reset_index(
    df: pd.DataFrame,
    level=None,
    drop: bool = False,
    names=None,
) -> pd.DataFrame:
    """
    Resets the DataFrame index back to columns.
    Node: ResetIndex
    """
    result = df.reset_index(level=level, drop=drop, names=names)
    print("[ResetIndex] Reset DataFrame index")
    return result
