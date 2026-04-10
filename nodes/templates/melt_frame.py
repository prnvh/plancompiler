import pandas as pd


def melt_frame(
    df: pd.DataFrame,
    id_vars=None,
    value_vars=None,
    var_name: str | None = None,
    value_name: str = "value",
    ignore_index: bool = True,
) -> pd.DataFrame:
    """
    Melts a DataFrame from wide to long format.
    Node: MeltFrame
    """
    result = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
        ignore_index=ignore_index,
    )
    print("[MeltFrame] Melted DataFrame")
    return result
