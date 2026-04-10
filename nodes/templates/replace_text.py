import pandas as pd


def replace_text(
    df: pd.DataFrame,
    column: str,
    pattern: str,
    repl: str,
    target: str | None = None,
    n: int = -1,
    case: bool | None = None,
    regex: bool = False,
) -> pd.DataFrame:
    """
    Replaces text in a column using pandas string replace.
    Node: ReplaceText
    """
    result = df.copy()
    output_column = target or column
    result[output_column] = result[column].astype("string").str.replace(
        pattern,
        repl,
        n=n,
        case=case,
        regex=regex,
    )
    print(f"[ReplaceText] Replaced text into '{output_column}'")
    return result
