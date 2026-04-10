import pandas as pd


def coalesce_columns(
    df: pd.DataFrame,
    target: str,
    sources: list[str],
) -> pd.DataFrame:
    """
    Creates a target column from the first non-null value across source columns.
    Node: CoalesceColumns
    """
    result = df.copy()
    result[target] = result[sources].bfill(axis=1).iloc[:, 0]
    print(f"[CoalesceColumns] Coalesced {sources} into '{target}'")
    return result
