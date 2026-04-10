import pandas as pd


def contains_text(
    df: pd.DataFrame,
    column: str,
    pattern: str,
    target: str | None = None,
    case: bool = True,
    regex: bool = True,
    na=None,
) -> pd.DataFrame:
    """
    Computes a boolean column from text containment.
    Node: ContainsText
    """
    result = df.copy()
    output_column = target or f"{column}_contains"
    result[output_column] = result[column].astype("string").str.contains(
        pattern,
        case=case,
        regex=regex,
        na=na,
    )
    print(f"[ContainsText] Computed contains flag in '{output_column}'")
    return result
