import re

import pandas as pd

from nodes.templates.expression_support import build_frame_expression_env


def _split_top_level(expression: str, word: str) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    quote: str | None = None
    paren_depth = 0
    bracket_depth = 0
    i = 0

    token = f" {word} "
    token_length = len(token)

    while i < len(expression):
        char = expression[i]

        if quote:
            current.append(char)
            if char == quote:
                quote = None
            i += 1
            continue

        if char in {"'", '"'}:
            quote = char
            current.append(char)
            i += 1
            continue

        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1

        if (
            paren_depth == 0
            and bracket_depth == 0
            and expression[i:i + token_length] == token
        ):
            pieces.append("".join(current).strip())
            current = []
            i += token_length
            continue

        current.append(char)
        i += 1

    pieces.append("".join(current).strip())
    return [piece for piece in pieces if piece]


def _compile_atomic_condition(expression: str) -> str:
    compiled = expression.strip()

    not_in_match = re.fullmatch(r"([A-Za-z_]\w*)\s+not\s+in\s+(.+)", compiled)
    if not_in_match:
        left = not_in_match.group(1)
        right = not_in_match.group(2).strip()
        return f"~{left}.isin({right})"

    in_match = re.fullmatch(r"([A-Za-z_]\w*)\s+in\s+(.+)", compiled)
    if in_match:
        left = in_match.group(1)
        right = in_match.group(2).strip()
        return f"{left}.isin({right})"

    return compiled


def _compile_condition_expression(expression: str) -> str:
    or_parts = _split_top_level(expression, "or")
    if len(or_parts) > 1:
        return " | ".join(
            f"({_compile_condition_expression(part)})" for part in or_parts
        )

    and_parts = _split_top_level(expression, "and")
    if len(and_parts) > 1:
        return " & ".join(
            f"({_compile_condition_expression(part)})" for part in and_parts
        )

    stripped = expression.strip()
    if stripped.startswith("not "):
        return f"~({_compile_condition_expression(stripped[4:])})"

    return _compile_atomic_condition(stripped)


def _evaluate_condition_mask(df: pd.DataFrame, condition: str) -> pd.Series:
    env = build_frame_expression_env(df)

    compiled = _compile_condition_expression(condition)
    if (
        not compiled.startswith("df.")
        and re.match(r"^(filter|duplicated|sort_values|query)\s*\(", compiled)
    ):
        compiled = f"df.{compiled}"
    mask = eval(compiled, env, {})

    if isinstance(mask, pd.Series):
        return mask.reindex(df.index, fill_value=False).astype(bool)

    if hasattr(mask, "__len__") and len(mask) == len(df):
        return pd.Series(mask, index=df.index).astype(bool)

    if isinstance(mask, bool):
        return pd.Series([mask] * len(df), index=df.index)

    raise TypeError(f"Condition did not produce a boolean mask: {condition}")


def data_filter(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """
    Filters rows by a boolean condition.

    Supports:
    - pandas query-style conditions
    - boolean mask expressions
    - membership patterns like "col in values" and "col not in values"
    """
    try:
        filtered = df.query(condition, engine="python")
        mode = "query"
    except Exception:
        mask = _evaluate_condition_mask(df, condition)
        filtered = df.loc[mask].copy()
        mode = "boolean-mask"

    print(f"[DataFilter] Applied condition using {mode}: {condition}")
    return filtered
