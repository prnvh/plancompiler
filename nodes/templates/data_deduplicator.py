import pandas as pd

def data_deduplicator(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: str | bool = "first",
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
    print("[DataDeduplicator] Removed duplicate rows")
    return deduped
