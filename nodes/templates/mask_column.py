import pandas as pd


def mask_column(
    df: pd.DataFrame,
    source: str,
    target: str,
    condition: str,
    value,
) -> pd.DataFrame:
    """
    Applies pandas mask semantics to a source column.
    Node: MaskColumn
    """
    result = df.copy()
    mask = result.eval(condition).fillna(False).astype(bool)
    result[target] = result[source].mask(mask, value)
    print(f"[MaskColumn] Applied mask logic from '{source}' into '{target}'")
    return result
