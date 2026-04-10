import pandas as pd


def explode_column(
    df: pd.DataFrame,
    column: str,
    ignore_index: bool = False,
) -> pd.DataFrame:
    """
    Explodes list-like values from a column into multiple rows.
    Node: ExplodeColumn
    """
    result = df.explode(column=column, ignore_index=ignore_index)
    print(f"[ExplodeColumn] Exploded column '{column}'")
    return result
