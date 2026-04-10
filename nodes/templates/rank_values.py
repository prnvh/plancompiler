import pandas as pd


def _listify(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def rank_values(
    df: pd.DataFrame,
    column: str,
    target: str,
    group_by=None,
    sort_by=None,
    sort_ascending: bool = True,
    method: str = "average",
    ascending: bool = True,
    pct: bool = False,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Ranks values within the whole frame or within groups.
    Node: RankValues
    """
    result = df.copy()
    group_keys = _listify(group_by)
    sort_columns = _listify(sort_by)
    if sort_columns:
        result = result.sort_values(by=sort_columns, ascending=sort_ascending)

    if group_keys:
        ranked = result.groupby(group_keys, dropna=dropna, sort=False)[column].rank(
            method=method,
            ascending=ascending,
            pct=pct,
        )
    else:
        ranked = result[column].rank(method=method, ascending=ascending, pct=pct)

    result[target] = ranked
    print(f"[RankValues] Ranked '{column}' into '{target}'")
    return result
