import pandas as pd


def extract_text(
    df: pd.DataFrame,
    column: str,
    pattern: str,
    target: str | None = None,
    targets: list[str] | None = None,
    flags: int = 0,
) -> pd.DataFrame:
    """
    Extracts regex capture groups from a text column.
    Node: ExtractText
    """
    result = df.copy()
    extracted = result[column].astype("string").str.extract(pattern, flags=flags, expand=True)

    if extracted.shape[1] == 1:
        output_column = target or f"{column}_extract"
        result[output_column] = extracted.iloc[:, 0]
        print(f"[ExtractText] Extracted text into '{output_column}'")
        return result

    output_columns = targets or [f"{column}_extract_{idx}" for idx in range(extracted.shape[1])]
    if len(output_columns) != extracted.shape[1]:
        raise ValueError("ExtractText targets must match extracted column count.")
    extracted.columns = output_columns
    for output_column in output_columns:
        result[output_column] = extracted[output_column]
    print(f"[ExtractText] Extracted text into {output_columns}")
    return result
