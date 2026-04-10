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


def pivot_frame(
    df: pd.DataFrame,
    index,
    columns,
    values=None,
    reset_index: bool = True,
    flatten_columns: bool = False,
) -> pd.DataFrame:
    """
    Pivots a DataFrame into wide format.
    Node: PivotFrame
    """
    result = df.pivot(index=index, columns=columns, values=values)
    if reset_index:
        result = result.reset_index()
    if flatten_columns:
        result = _flatten_columns(result)
    print("[PivotFrame] Pivoted DataFrame")
    return result
