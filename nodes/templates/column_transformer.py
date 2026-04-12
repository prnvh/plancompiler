import re

import numpy as np
import pandas as pd

from nodes.contracts import normalize_node_parameters
from nodes.templates.expression_support import (
    build_row_expression_env,
    build_frame_expression_env,
    evaluate_frame_expression,
)
from nodes.templates.frame_support import resolve_column_label, resolve_column_labels, resolve_mapping_keys


def _evaluate_expression(df: pd.DataFrame, expression: str):
    return evaluate_frame_expression(df, expression)


def _parse_conditional_expression(expression: str) -> tuple[str, str, str] | None:
    stripped = expression.strip()

    if_then_else = re.fullmatch(
        r"if\s+(?P<condition>.+?)\s+then\s+(?P<when_true>.+?)\s+else\s+(?P<when_false>.+)",
        stripped,
        flags=re.IGNORECASE,
    )
    if if_then_else:
        return (
            if_then_else.group("condition"),
            if_then_else.group("when_true"),
            if_then_else.group("when_false"),
        )

    python_ternary = re.fullmatch(
        r"(?P<when_true>.+?)\s+if\s+(?P<condition>.+?)\s+else\s+(?P<when_false>.+)",
        stripped,
    )
    if python_ternary:
        return (
            python_ternary.group("condition"),
            python_ternary.group("when_true"),
            python_ternary.group("when_false"),
        )

    return None


def _evaluate_rowwise_expression(df: pd.DataFrame, condition: str, when_true: str, when_false: str):
    def evaluate_row(row):
        env = build_row_expression_env(row, df=df)
        branch = when_true if eval(condition, env, {}) else when_false
        return eval(branch, env, {})

    return df.apply(evaluate_row, axis=1)


def _evaluate_rowwise_rule(df: pd.DataFrame, rule: str):
    rule_globals = build_frame_expression_env(df)
    row_func = eval(rule, rule_globals, {})
    return df.apply(row_func, axis=1)


def _concatenate_columns(df: pd.DataFrame, source_columns: list[str], *, separator: str = "", strip: bool = False) -> pd.Series:
    combined = df[source_columns].fillna("").astype(str).agg(separator.join, axis=1)
    return combined.str.strip() if strip else combined


def _apply_column_operation(df: pd.DataFrame, operation: dict) -> pd.DataFrame:
    op_type = operation["type"]

    if op_type == "rename_columns":
        return df.rename(columns=operation["mapping"])

    if op_type == "select_columns":
        return df.loc[:, resolve_column_labels(df, operation["columns"])].copy()

    if op_type == "drop_columns":
        return df.drop(columns=resolve_column_labels(df, operation["columns"]))

    if op_type == "reorder_columns":
        return df.loc[:, resolve_column_labels(df, operation["columns"])].copy()

    if op_type == "melt":
        id_vars = resolve_column_labels(df, operation.get("id_vars"))
        value_vars = resolve_column_labels(df, operation.get("value_vars"))
        value_name = operation.get("value_name", "value")
        preserve_row_order = bool(operation.get("preserve_row_order", False))
        working = df.copy()
        row_order_column = "__row_order__"
        if preserve_row_order:
            working[row_order_column] = range(len(working))
            id_vars = list(id_vars or []) + [row_order_column]

        melted = working.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=operation.get("var_name"),
            value_name=value_name,
            ignore_index=bool(operation.get("ignore_index", True)),
        )
        if operation.get("dropna"):
            melted = melted[melted[value_name].notna()]
        if preserve_row_order:
            variable_column = operation.get("var_name") or "variable"
            order_columns = [row_order_column]
            if value_vars:
                melted[variable_column] = pd.Categorical(
                    melted[variable_column],
                    categories=list(value_vars),
                    ordered=True,
                )
                order_columns.append(variable_column)
            melted = melted.sort_values(order_columns, kind="stable")
            melted = melted.drop(columns=[row_order_column])
        if bool(operation.get("ignore_index", True)):
            melted = melted.reset_index(drop=True)
        return melted

    if op_type == "drop_duplicate_rows":
        return df.drop_duplicates(
            subset=resolve_column_labels(df, operation.get("subset")),
            keep=operation.get("keep", "first"),
            ignore_index=bool(operation.get("ignore_index", False)),
        )

    if op_type == "extract_regex":
        target = operation.get("new_column")
        if not target:
            raise ValueError("extract_regex requires 'new_column'.")

        source_column = resolve_column_label(df, operation["source_column"])
        extracted = df[source_column].astype(str).str.extract(
            operation["pattern"],
            expand=True,
        )
        group = operation.get("expand_group")
        if isinstance(extracted, pd.DataFrame):
            if group is None:
                selected = extracted.iloc[:, 0]
            else:
                selected = extracted.iloc[:, int(group)]
        else:
            selected = extracted

        if "default" in operation:
            selected = selected.fillna(operation["default"])
        df[target] = selected
        return df

    if op_type == "split_column":
        source = resolve_column_label(df, operation["source_column"])
        new_columns = list(operation["new_columns"])
        separator = operation.get("separator", " ")
        regex = bool(operation.get("regex", False))
        max_splits = operation.get("max_splits")
        split_frame = df[source].astype(str).str.split(
            separator,
            n=int(max_splits) if max_splits is not None else -1,
            expand=True,
            regex=regex,
        )
        split_frame = split_frame.reindex(columns=range(len(new_columns)))
        split_frame.columns = new_columns
        if operation.get("strip"):
            split_frame = split_frame.apply(lambda series: series.str.strip() if series.dtype == object else series)
        for column in new_columns:
            df[column] = split_frame[column]
        return df

    if op_type == "map_values":
        source = resolve_column_label(df, operation["source_column"])
        target = operation.get("new_column") or source
        mapped = df[source].map(resolve_mapping_keys(df[source], operation["mapping"]))
        if "default" in operation:
            mapped = mapped.fillna(operation["default"])
        df[target] = mapped
        return df

    if op_type == "concatenate_columns":
        target = operation["new_column"]
        source_columns = resolve_column_labels(df, operation["source_columns"])
        df[target] = _concatenate_columns(
            df,
            source_columns,
            separator=str(operation.get("separator", "")),
            strip=bool(operation.get("strip", False)),
        )
        return df

    if op_type == "explode_column":
        source = resolve_column_label(df, operation["source_column"])
        result = df.explode(source, ignore_index=bool(operation.get("ignore_index", False)))
        target = operation.get("new_column")
        if target and target != source:
            result[target] = result[source]
        return result

    if op_type == "derive_inverse_columns":
        numerator = operation.get("numerator", 1)
        prefix = operation.get("prefix", "inv_")
        suffix = operation.get("suffix", "")
        preserve_zero = bool(operation.get("preserve_zero", False))
        zero_value = operation.get("zero_value")
        for source_column in resolve_column_labels(df, operation["source_columns"]):
            values = pd.to_numeric(df[source_column], errors="coerce")
            target = f"{prefix}{source_column}{suffix}"
            if preserve_zero:
                mask = values == 0
                result = numerator / values.replace(0, np.nan)
                df[target] = result.where(~mask, zero_value)
            else:
                df[target] = numerator / values
        return df

    if op_type == "derive_function_columns":
        function = str(operation["function"]).strip().lower()
        prefix = operation.get("prefix")
        suffix = operation.get("suffix", "")
        preserve_zero = bool(operation.get("preserve_zero", False))
        zero_value = operation.get("zero_value")
        default_prefix = {
            "inverse": "inv_",
            "exp": "exp_",
            "exponential": "exp_",
            "sigmoid": "sigmoid_",
            "log": "log_",
            "log1p": "log1p_",
            "square": "sq_",
            "sqrt": "sqrt_",
            "abs": "abs_",
            "negate": "neg_",
        }.get(function, f"{function}_")
        prefix = default_prefix if prefix is None else prefix

        for source_column in resolve_column_labels(df, operation["source_columns"]):
            values = pd.to_numeric(df[source_column], errors="coerce")
            if function == "inverse":
                mask = values == 0
                transformed = 1 / values.replace(0, np.nan)
                if preserve_zero:
                    transformed = transformed.where(~mask, zero_value)
            elif function in {"exp", "exponential"}:
                transformed = np.exp(values)
            elif function == "sigmoid":
                transformed = 1 / (1 + np.exp(-values))
            elif function == "log":
                transformed = np.log(values)
            elif function == "log1p":
                transformed = np.log1p(values)
            elif function == "square":
                transformed = values ** 2
            elif function == "sqrt":
                transformed = np.sqrt(values)
            elif function == "abs":
                transformed = values.abs()
            elif function == "negate":
                transformed = -values
            else:
                raise ValueError(f"Unsupported derive_function_columns function: {function}")

            df[f"{prefix}{source_column}{suffix}"] = transformed
        return df

    if op_type == "derive_column":
        target = operation["new_column"]
        if operation.get("rowwise_rule"):
            df[target] = _evaluate_rowwise_rule(df, operation["rowwise_rule"])
        elif operation.get("expression"):
            expression = operation["expression"]
            conditional = _parse_conditional_expression(expression)
            if conditional:
                df[target] = _evaluate_rowwise_expression(df, *conditional)
            else:
                df[target] = _evaluate_expression(df, expression)
        else:
            df[target] = operation.get("value")
        return df

    if op_type == "derive_first_matching_label":
        target = operation["new_column"]
        source_columns = resolve_column_labels(df, operation["source_columns"])
        match_value = operation.get("match_value", 1)
        labels = {
            resolve_column_label(df, key): value
            for key, value in (operation.get("labels") or {}).items()
        }
        default = operation.get("default")

        def pick_label(row):
            for column in source_columns:
                if row[column] == match_value:
                    return labels.get(column, column)
            return default

        df[target] = df[source_columns].apply(pick_label, axis=1)
        return df

    if op_type == "derive_matching_labels":
        target = operation["new_column"]
        source_columns = resolve_column_labels(df, operation["source_columns"])
        match_value = operation.get("match_value", 1)
        labels = {
            resolve_column_label(df, key): value
            for key, value in (operation.get("labels") or {}).items()
        }
        df[target] = df[source_columns].apply(
            lambda row: [labels.get(column, column) for column, value in row.items() if value == match_value],
            axis=1,
        )
        return df

    raise ValueError(f"Unsupported ColumnTransformer operation: {op_type}")


def column_transformer(df: pd.DataFrame, operations=None, **kwargs) -> pd.DataFrame:
    """
    Canonical ColumnTransformer contract:
    {
      "operations": [
        {"type": "...", ...}
      ]
    }
    """
    params = {"operations": operations}
    params.update(kwargs)
    normalized = normalize_node_parameters("ColumnTransformer", params)
    transformed = df.copy()

    for operation in normalized.get("operations", []):
        transformed = _apply_column_operation(transformed, operation)

    print("[ColumnTransformer] Applied column structure and derived-column transformations")
    return transformed
