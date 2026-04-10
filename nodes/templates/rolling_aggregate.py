import pandas as pd


def _listify(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def rolling_aggregate(
    df: pd.DataFrame,
    column: str,
    target: str,
    window: int,
    group_by=None,
    sort_by=None,
    sort_ascending: bool = True,
    min_periods: int | None = None,
    center: bool = False,
    agg="mean",
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Computes a rolling aggregation over a column.
    Node: RollingAggregate
    """
    result = df.copy()
    group_keys = _listify(group_by)
    sort_columns = _listify(sort_by)
    if sort_columns:
        result = result.sort_values(by=sort_columns, ascending=sort_ascending)

    periods = window if min_periods is None else min_periods
    if group_keys:
        values = (
            result.groupby(group_keys, dropna=dropna, sort=False)[column]
            .rolling(window=window, min_periods=periods, center=center)
            .agg(agg)
            .reset_index(level=list(range(len(group_keys))), drop=True)
        )
    else:
        values = result[column].rolling(
            window=window,
            min_periods=periods,
            center=center,
        ).agg(agg)

    result[target] = values
    print(f"[RollingAggregate] Computed rolling aggregate into '{target}'")
    return result
