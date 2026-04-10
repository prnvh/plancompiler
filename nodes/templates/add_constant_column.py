import pandas as pd


def add_constant_column(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
    """
    Adds or overwrites a column with a constant value.
    Node: AddConstantColumn
    """
    result = df.copy()
    result[column] = value
    print(f"[AddConstantColumn] Set '{column}' to a constant value")
    return result
