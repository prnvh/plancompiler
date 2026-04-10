import pandas as pd


def value_counts_ops(
    df: pd.DataFrame,
    column: str | list[str],
    normalize: bool = False,
    sort: bool = True,
    ascending: bool = False,
    dropna: bool = True,
    top_k: int | None = None,
    name: str | None = None,
) -> pd.Series:
    """
    Returns value counts for one or more columns.
    Node: ValueCountsOps
    """
    columns = [column] if isinstance(column, str) else list(column)
    if not columns:
        raise ValueError("ValueCountsOps requires at least one column.")

    if len(columns) == 1:
        result = df[columns[0]].value_counts(
            normalize=normalize,
            sort=sort,
            ascending=ascending,
            dropna=dropna,
        )
    else:
        result = df.value_counts(
            subset=columns,
            normalize=normalize,
            sort=sort,
            ascending=ascending,
            dropna=dropna,
        )

    if top_k is not None:
        result = result.head(top_k)
    if name:
        result = result.rename(name)

    print(f"[ValueCountsOps] Computed value counts for {columns}")
    return result
