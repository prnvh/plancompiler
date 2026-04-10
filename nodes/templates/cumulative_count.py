import pandas as pd


def _listify(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def cumulative_count(
    df: pd.DataFrame,
    target: str,
    group_by=None,
    sort_by=None,
    sort_ascending: bool = True,
    ascending: bool = True,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Computes a cumulative row count over the frame or within groups.
    Node: CumulativeCount
    """
    result = df.copy()
    group_keys = _listify(group_by)
    sort_columns = _listify(sort_by)
    if sort_columns:
        result = result.sort_values(by=sort_columns, ascending=sort_ascending)

    if group_keys:
        counts = result.groupby(group_keys, dropna=dropna, sort=False).cumcount(ascending=ascending)
    else:
        values = range(len(result)) if ascending else range(len(result) - 1, -1, -1)
        counts = pd.Series(values, index=result.index)

    result[target] = counts
    print(f"[CumulativeCount] Computed cumulative count into '{target}'")
    return result
