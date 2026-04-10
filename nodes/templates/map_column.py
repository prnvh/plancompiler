import pandas as pd


def map_column(
    df: pd.DataFrame,
    source: str,
    target: str,
    mapping: dict,
    default=None,
) -> pd.DataFrame:
    """
    Maps a source column through a dictionary.
    Node: MapColumn
    """
    result = df.copy()
    mapped = result[source].map(mapping)
    if default is not None:
        mapped = mapped.fillna(default)
    result[target] = mapped
    print(f"[MapColumn] Mapped '{source}' into '{target}'")
    return result
