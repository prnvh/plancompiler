import pandas as pd


def set_index(
    df: pd.DataFrame,
    keys,
    drop: bool = True,
    append: bool = False,
    verify_integrity: bool = False,
) -> pd.DataFrame:
    """
    Sets one or more columns as the DataFrame index.
    Node: SetIndex
    """
    result = df.set_index(
        keys,
        drop=drop,
        append=append,
    )
    if verify_integrity and not result.index.is_unique:
        raise ValueError("[SetIndex] Index contains duplicate values.")
    print("[SetIndex] Set DataFrame index")
    return result
