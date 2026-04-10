import pandas as pd


def dataframe_input(
    dataframe: pd.DataFrame | None = None,
    source_name: str = "df",
    copy: bool = True,
) -> pd.DataFrame:
    """
    Loads an in-memory DataFrame from the runtime context.
    Node: DataFrameInput
    """
    candidate = dataframe if dataframe is not None else globals().get(source_name)

    if candidate is None:
        raise ValueError(
            f"DataFrameInput could not find a DataFrame in '{source_name}'."
        )
    if not isinstance(candidate, pd.DataFrame):
        raise TypeError(
            f"DataFrameInput expected '{source_name}' to be a pandas DataFrame, "
            f"got {type(candidate).__name__}."
        )

    result = candidate.copy() if copy else candidate
    print(f"[DataFrameInput] Loaded DataFrame from '{source_name}' with shape {result.shape}")
    return result
