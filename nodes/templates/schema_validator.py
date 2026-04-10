import pandas as pd

def schema_validator(df: pd.DataFrame, schema: dict | None = None) -> pd.DataFrame:
    """
    Validates DataFrame schema.
    Node: SchemaValidator
    """
    if schema:
        for col, dtype in schema.items():
            if col not in df.columns:
                raise ValueError(f"[SchemaValidator] Missing column: {col}")
            if str(df[col].dtype) != str(dtype):
                raise ValueError(
                    f"[SchemaValidator] Column {col} dtype mismatch: "
                    f"expected {dtype}, got {df[col].dtype}"
                )

    print("[SchemaValidator] Schema validated")
    return df
