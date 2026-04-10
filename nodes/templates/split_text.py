import pandas as pd


def split_text(
    df: pd.DataFrame,
    column: str,
    pattern: str | None = None,
    target: str | None = None,
    targets: list[str] | None = None,
    n: int = -1,
    regex: bool | None = None,
    expand: bool = False,
) -> pd.DataFrame:
    """
    Splits a text column into lists or expanded columns.
    Node: SplitText
    """
    result = df.copy()
    series = result[column].astype("string")

    if expand:
        expanded = series.str.split(pat=pattern, n=n, regex=regex, expand=True)
        output_columns = targets or [f"{column}_{idx}" for idx in range(expanded.shape[1])]
        if len(output_columns) != expanded.shape[1]:
            raise ValueError("SplitText targets must match expanded column count.")
        expanded.columns = output_columns
        for output_column in output_columns:
            result[output_column] = expanded[output_column]
        print(f"[SplitText] Expanded '{column}' into {output_columns}")
        return result

    output_column = target or column
    result[output_column] = series.str.split(pat=pattern, n=n, regex=regex)
    print(f"[SplitText] Split '{column}' into '{output_column}'")
    return result
