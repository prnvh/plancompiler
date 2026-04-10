import pandas as pd


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [
        "_".join(str(level) for level in column if str(level) != "").rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in result.columns
    ]
    return result


def pivot_table_frame(
    df: pd.DataFrame,
    index=None,
    columns=None,
    values=None,
    aggfunc="mean",
    fill_value=None,
    margins: bool = False,
    dropna: bool = True,
    observed: bool = False,
    reset_index: bool = True,
    flatten_columns: bool = False,
) -> pd.DataFrame:
    """
    Builds a pivot table with an aggregation function.
    Node: PivotTableFrame
    """
    result = df.pivot_table(
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        fill_value=fill_value,
        margins=margins,
        dropna=dropna,
        observed=observed,
    )
    if reset_index:
        result = result.reset_index()
    if flatten_columns:
        result = _flatten_columns(result)
    print("[PivotTableFrame] Built pivot table")
    return result
