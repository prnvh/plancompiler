import pandas as pd

def data_deduplicator(
    df: pd.DataFrame,
    subset=None,
    keep: str = "first",
    ignore_index: bool = False,
) -> pd.DataFrame:
    """
    Removes duplicate rows.
    Node: DataDeduplicator
    """
    deduped = df.drop_duplicates(
        subset=subset,
        keep=keep,
        ignore_index=ignore_index,
    )
    print(f"[DataDeduplicator] Removed duplicate rows using subset={subset}, keep={keep}, ignore_index={ignore_index}")
    return deduped
