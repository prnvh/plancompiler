import pandas as pd


def where_column(
    df: pd.DataFrame,
    source: str,
    target: str,
    condition: str,
    other,
) -> pd.DataFrame:
    """
    Applies pandas where semantics to a source column.
    Node: WhereColumn
    """
    result = df.copy()
    mask = result.eval(condition).fillna(False).astype(bool)
    result[target] = result[source].where(mask, other)
    print(f"[WhereColumn] Applied where logic from '{source}' into '{target}'")
    return result
