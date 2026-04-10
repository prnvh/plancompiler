import pandas as pd


def _listify(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def cumulative_sum(
    df: pd.DataFrame,
    column: str,
    target: str,
    group_by=None,
    sort_by=None,
    sort_ascending: bool = True,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Computes a cumulative sum over a column.
    Node: CumulativeSum
    """
    result = df.copy()
    group_keys = _listify(group_by)
    sort_columns = _listify(sort_by)
    if sort_columns:
        result = result.sort_values(by=sort_columns, ascending=sort_ascending)

    if group_keys:
        values = result.groupby(group_keys, dropna=dropna, sort=False)[column].cumsum()
    else:
        values = result[column].cumsum()

    result[target] = values
    print(f"[CumulativeSum] Computed cumulative sum into '{target}'")
    return result
