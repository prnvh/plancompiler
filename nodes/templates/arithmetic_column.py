import pandas as pd


_OPS = {
    "+": lambda left, right: left + right,
    "-": lambda left, right: left - right,
    "*": lambda left, right: left * right,
    "/": lambda left, right: left / right,
    "//": lambda left, right: left // right,
    "%": lambda left, right: left % right,
    "**": lambda left, right: left ** right,
}


def _resolve_operand(df: pd.DataFrame, operand):
    if isinstance(operand, str) and operand in df.columns:
        return df[operand]
    return operand


def arithmetic_column(
    df: pd.DataFrame,
    target: str,
    left,
    op: str,
    right,
) -> pd.DataFrame:
    """
    Creates a column from arithmetic on two operands.
    Node: ArithmeticColumn
    """
    if op not in _OPS:
        raise ValueError(f"Unsupported arithmetic operator: {op}")

    result = df.copy()
    result[target] = _OPS[op](_resolve_operand(result, left), _resolve_operand(result, right))
    print(f"[ArithmeticColumn] Computed '{target}' with operator '{op}'")
    return result
