import pandas as pd

def aggregator(df: pd.DataFrame, group_by: str, agg_func: str) -> pd.DataFrame:
    """
    Aggregates DataFrame.
    Node: Aggregator
    """

    if agg_func == "count":
        grouped = df.groupby(group_by).size().reset_index(name="count")
    else:
        grouped = df.groupby(group_by).agg(agg_func).reset_index()

    print(f"[Aggregator] Aggregated using {agg_func}")
    return grouped