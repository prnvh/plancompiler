from __future__ import annotations

import numpy as np
import pandas as pd


SAFE_EXPRESSION_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "getattr": getattr,
    "hasattr": hasattr,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def _where_like(df: pd.DataFrame, condition, when_true, when_false):
    if not isinstance(condition, pd.Series):
        condition = pd.Series([bool(condition)] * len(df), index=df.index)
    else:
        condition = condition.reindex(df.index, fill_value=False)

    def _coerce(value):
        if isinstance(value, pd.Series):
            return value.reindex(df.index)
        return pd.Series([value] * len(df), index=df.index)

    true_values = _coerce(when_true)
    false_values = _coerce(when_false)
    return true_values.where(condition.astype(bool), false_values)


def _column_accessor(df: pd.DataFrame):
    return lambda name: df[name]


def build_frame_expression_env(
    df: pd.DataFrame,
    *,
    extra: dict | None = None,
) -> dict:
    env = {
        "__builtins__": SAFE_EXPRESSION_BUILTINS,
        "pd": pd,
        "np": np,
        "df": df,
        "index": df.index,
        "col": _column_accessor(df),
        "column": _column_accessor(df),
        "where": lambda condition, when_true, when_false: _where_like(
            df,
            condition,
            when_true,
            when_false,
        ),
    }
    env.update({column: df[column] for column in df.columns})
    if extra:
        env.update(extra)
    return env


def build_row_expression_env(
    row: pd.Series,
    *,
    df: pd.DataFrame | None = None,
    extra: dict | None = None,
) -> dict:
    env = {
        "__builtins__": SAFE_EXPRESSION_BUILTINS,
        "pd": pd,
        "np": np,
        "row": row,
        "index": row.name,
    }
    if df is not None:
        env["df"] = df
        env["col"] = _column_accessor(df)
        env["column"] = _column_accessor(df)
    env.update(row.to_dict())
    if extra:
        env.update(extra)
    return env


def evaluate_frame_expression(
    df: pd.DataFrame,
    expression: str,
    *,
    extra: dict | None = None,
):
    return eval(expression, build_frame_expression_env(df, extra=extra), {})


def evaluate_row_expression(
    row: pd.Series,
    expression: str,
    *,
    df: pd.DataFrame | None = None,
    extra: dict | None = None,
):
    return eval(expression, build_row_expression_env(row, df=df, extra=extra), {})
