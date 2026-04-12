from __future__ import annotations

import ast
import copy
import re
from typing import Any


NODE_CONTRACTS: dict[str, dict[str, Any]] = {
    "DataFrameInput": {
        "top_level_params": {
            "source_name": "string",
            "copy": "boolean",
        },
        "examples": [
            {},
            {"copy": False},
        ],
    },
    "DataFilter": {
        "top_level_params": {
            "condition": "string",
        },
        "examples": [
            {"condition": "score >= 0.5"},
            {"condition": "(preserve_duplicate == 'Yes') or ~record_id.duplicated()"},
        ],
    },
    "DataDeduplicator": {
        "top_level_params": {
            "subset": "list[string]",
            "keep": "string",
            "ignore_index": "boolean",
        },
        "examples": [
            {"subset": ["customer_id"], "keep": "last"},
        ],
    },
    "Aggregator": {
        "top_level_params": {
            "group_keys": "list[string|GroupKey]",
            "aggregations": "list[AggregationSpec]",
            "dropna": "boolean",
            "sort_by": "list[string]",
            "ascending": "boolean|list[boolean]",
            "reset_index": "boolean",
        },
        "examples": [
            {
                "group_keys": ["region"],
                "aggregations": [
                    {"column": "sales", "op": "sum", "output": "total_sales"},
                    {"column": "sales", "op": "mean", "output": "avg_sales"},
                    {"column": "sales", "op": "std", "output": "std_sales"},
                    {"op": "size", "output": "row_count"},
                ],
            },
            {
                "group_keys": [
                    {"type": "expression", "name": "bucket", "expression": "index // 3"}
                ],
                "aggregations": [
                    {"column": "amount", "op": "collect_list", "output": "amounts"},
                ],
                "sort_by": ["bucket"],
            },
        ],
    },
    "DateRangeExpander": {
        "top_level_params": {
            "group_keys": "list[string]",
            "date_column": "string",
            "freq": "string",
            "range_scope": "string",
            "fill_values": "dict[string, Any]",
            "fill_strategies": "dict[string, string]",
            "date_format": "string",
            "sort": "boolean",
        },
        "examples": [
            {
                "group_keys": ["user"],
                "date_column": "dt",
                "range_scope": "global",
                "fill_values": {"val": 0},
            },
            {
                "group_keys": ["user"],
                "date_column": "dt",
                "range_scope": "global",
                "fill_strategies": {"val": "group_max"},
                "date_format": "%d-%b-%Y",
            },
        ],
    },
    "ChunkAggregator": {
        "top_level_params": {
            "rules": "list[ChunkRule]",
            "from_end": "boolean",
            "drop_remainder": "boolean",
        },
        "examples": [
            {
                "rules": [{"size": 3, "agg": "mean", "columns": ["col1"]}],
            },
            {
                "rules": [
                    {"size": 4, "agg": "sum", "columns_suffix": "_sum"},
                    {"size": 3, "agg": "mean", "exclude_columns": ["id"]},
                ],
                "from_end": True,
            },
        ],
    },
    "ValueCountsReporter": {
        "top_level_params": {
            "columns": "list[string]",
            "dropna": "boolean",
            "sort": "boolean",
            "include_header": "boolean",
        },
        "examples": [
            {"columns": ["id", "status"], "dropna": False},
            {"dropna": False, "include_header": True},
        ],
    },
    "ReduceOutput": {
        "top_level_params": {
            "method": "string",
            "column": "string",
            "row": "int|label",
            "label": "label",
            "name": "string|null",
            "position": "int",
            "n": "int",
            "agg": "string",
            "anchor_extreme": "string",
            "anchor_occurrence": "string",
            "target_extreme": "string",
            "target_occurrence": "string",
            "direction": "string",
            "skipna": "boolean",
        },
        "examples": [
            {},
            {"method": "column", "column": "result"},
            {"method": "column_to_series", "column": "value", "label": "date"},
            {"method": "column_to_series", "column": "value", "label": "date", "name": None},
            {"method": "scalar_agg", "column": "score", "agg": "max"},
            {
                "method": "columnwise_extreme_index",
                "anchor_extreme": "min",
                "anchor_occurrence": "first",
                "target_extreme": "max",
                "target_occurrence": "first",
                "direction": "after",
            },
        ],
    },
    "ColumnTransformer": {
        "top_level_params": {
            "operations": "list[ColumnOperation]",
        },
        "operation_types": {
            "rename_columns": {"required": ["mapping"], "optional": []},
            "select_columns": {"required": ["columns"], "optional": []},
            "drop_columns": {"required": ["columns"], "optional": []},
            "reorder_columns": {"required": ["columns"], "optional": []},
            "melt": {
                "required": [],
                "optional": ["id_vars", "value_vars", "var_name", "value_name", "ignore_index", "preserve_row_order", "dropna"],
            },
            "drop_duplicate_rows": {"required": [], "optional": ["subset", "keep", "ignore_index"]},
            "derive_column": {
                "required": ["new_column"],
                "optional": ["value", "expression", "rowwise_rule", "source_columns"],
            },
            "extract_regex": {
                "required": ["source_column", "pattern"],
                "optional": ["new_column", "default", "expand_group"],
            },
            "split_column": {
                "required": ["source_column", "new_columns"],
                "optional": ["separator", "regex", "max_splits", "strip"],
            },
            "map_values": {
                "required": ["source_column", "mapping"],
                "optional": ["new_column", "default"],
            },
            "concatenate_columns": {
                "required": ["new_column", "source_columns"],
                "optional": ["separator", "strip"],
            },
            "explode_column": {
                "required": ["source_column"],
                "optional": ["new_column", "ignore_index"],
            },
            "derive_inverse_columns": {
                "required": ["source_columns"],
                "optional": ["prefix", "suffix", "numerator", "preserve_zero", "zero_value"],
            },
            "derive_function_columns": {
                "required": ["source_columns", "function"],
                "optional": ["prefix", "suffix", "preserve_zero", "zero_value"],
            },
            "shift_non_nulls_left": {
                "required": [],
                "optional": ["columns", "fill_value"],
            },
            "shift_nulls_to_top_per_column": {
                "required": [],
                "optional": ["columns"],
            },
            "derive_first_matching_label": {
                "required": ["new_column", "source_columns"],
                "optional": ["match_value", "labels", "default"],
            },
            "derive_matching_labels": {
                "required": ["new_column", "source_columns"],
                "optional": ["match_value", "labels"],
            },
        },
        "examples": [
            {
                "operations": [
                    {
                        "type": "derive_first_matching_label",
                        "new_column": "primary_channel",
                        "source_columns": ["email_flag", "sms_flag", "phone_flag"],
                        "match_value": 1,
                        "labels": {
                            "email_flag": "email",
                            "sms_flag": "sms",
                            "phone_flag": "phone",
                        },
                    }
                ]
            },
            {
                "operations": [
                    {
                        "type": "derive_matching_labels",
                        "new_column": "all_channels",
                        "source_columns": ["email_flag", "sms_flag", "phone_flag"],
                        "match_value": 1,
                        "labels": {
                            "email_flag": "email",
                            "sms_flag": "sms",
                            "phone_flag": "phone",
                        },
                    }
                ]
            },
            {
                "operations": [
                    {"type": "rename_columns", "mapping": {"customer_id": "client_id"}},
                    {"type": "reorder_columns", "columns": ["client_id", "region", "status"]},
                ]
            },
            {
                "operations": [
                    {
                        "type": "extract_regex",
                        "source_column": "duration",
                        "pattern": "^(\\d+)",
                        "new_column": "number",
                    },
                    {
                        "type": "map_values",
                        "source_column": "unit",
                        "new_column": "unit_days",
                        "mapping": {"day": 1, "week": 7},
                    },
                ]
            },
            {
                "operations": [
                    {
                        "type": "derive_function_columns",
                        "source_columns": ["A", "B"],
                        "function": "inverse",
                        "prefix": "inv_",
                        "preserve_zero": True,
                    }
                ]
            },
            {
                "operations": [
                    {
                        "type": "shift_non_nulls_left",
                        "columns": ["A", "B", "C"],
                    }
                ]
            },
        ],
    },
    "ValueTransformer": {
        "top_level_params": {
            "operations": "list[ValueOperation]",
        },
        "operation_types": {
            "replace_infrequent": {
                "required": [],
                "optional": ["columns", "threshold", "replacement", "preserve_values", "column_rules", "preserve_na"],
            },
            "replace_values": {
                "required": ["column"],
                "optional": ["mapping", "old_value", "new_value"],
            },
            "map_values": {
                "required": ["column", "mapping"],
                "optional": ["default"],
            },
            "factorize_values": {
                "required": ["column"],
                "optional": ["start", "sort", "new_column"],
            },
            "fill_missing": {
                "required": [],
                "optional": ["column", "value", "mapping"],
            },
            "normalize_text": {
                "required": ["column", "style"],
                "optional": [],
            },
            "replace_substring": {
                "required": ["column", "old", "new"],
                "optional": ["regex", "case"],
            },
            "strip_prefix": {
                "required": ["column", "prefix"],
                "optional": [],
            },
            "coerce_numeric": {
                "required": ["column"],
                "optional": ["new_column", "errors"],
            },
            "clip_values": {
                "required": ["column"],
                "optional": ["lower", "upper", "new_column", "errors"],
            },
            "round_values": {
                "required": ["column"],
                "optional": ["decimals", "new_column"],
            },
            "parse_duration_text": {
                "required": ["column"],
                "optional": ["new_column", "result", "unit_map", "errors"],
            },
        },
        "examples": [
            {
                "operations": [
                    {
                        "type": "replace_infrequent",
                        "columns": ["segment", "channel", "region"],
                        "threshold": 2,
                        "replacement": "Other",
                    }
                ]
            },
            {
                "operations": [
                    {
                        "type": "replace_infrequent",
                        "columns": ["segment", "channel"],
                        "threshold": 5,
                        "replacement": "Other",
                    }
                ]
            },
            {
                "operations": [
                    {
                        "type": "replace_infrequent",
                        "column_rules": [
                            {"column": "priority", "threshold": 3, "replacement": "Other"},
                            {"column": "tier", "threshold": 2, "replacement": "Other"},
                        ],
                        "preserve_values": ["VIP"],
                    }
                ]
            },
            {
                "operations": [
                    {
                        "type": "replace_substring",
                        "column": "sku",
                        "old": "item-",
                        "new": "",
                    }
                ]
            },
            {
                "operations": [
                    {
                        "type": "factorize_values",
                        "column": "category_code",
                        "start": 1,
                    }
                ]
            },
            {
                "operations": [
                    {
                        "type": "parse_duration_text",
                        "column": "duration",
                        "new_column": "duration_days",
                        "result": "days",
                        "unit_map": {"day": 1, "week": 7, "month": 30},
                    }
                ]
            },
        ],
    },
    "DatetimeTransformer": {
        "top_level_params": {
            "operations": "list[DatetimeOperation]",
        },
        "operation_types": {
            "parse_datetime": {
                "required": ["column"],
                "optional": ["format", "utc", "errors", "index_level"],
            },
            "remove_timezone": {
                "required": ["column"],
                "optional": ["strategy", "index_level"],
            },
            "format_datetime": {
                "required": ["column", "format"],
                "optional": ["index_level"],
            },
            "extract_part": {
                "required": ["column", "part", "new_column"],
                "optional": ["index_level"],
            },
            "date_diff": {
                "required": ["column", "other_column", "new_column"],
                "optional": ["unit", "index_level", "other_index_level"],
            },
        },
        "examples": [
            {
                "operations": [
                    {"type": "parse_datetime", "column": "event_time"},
                    {"type": "remove_timezone", "column": "event_time", "strategy": "preserve_wall_time"},
                ]
            },
            {
                "operations": [
                    {"type": "format_datetime", "column": "event_time", "format": "%Y-%m-%d %H:%M:%S"},
                    {"type": "extract_part", "column": "event_time", "part": "hour", "new_column": "event_hour"},
                ]
            },
            {
                "operations": [
                    {"type": "parse_datetime", "column": "timestamp", "index_level": "timestamp"},
                ]
            },
        ],
    },
    "DataTransformer": {
        "top_level_params": {
            "operations": "list[GenericOperation]",
        },
        "operation_types": {
            "rename_columns": {"required": ["mapping"], "optional": []},
            "cast_column": {"required": ["column"], "optional": ["to", "dtype", "target_type", "format", "utc", "errors"]},
            "melt": {"required": [], "optional": ["id_vars", "value_vars", "var_name", "value_name", "ignore_index", "preserve_row_order", "dropna"]},
            "drop_duplicate_rows": {"required": [], "optional": ["subset", "keep", "ignore_index"]},
            "circular_shift": {"required": ["column", "shift"], "optional": []},
            "shift_non_nulls_left": {"required": [], "optional": ["columns", "fill_value"]},
            "shift_nulls_to_top_per_column": {"required": [], "optional": ["columns"]},
            "expand_date_range_per_group": {"required": ["group_by", "date_column"], "optional": ["freq", "sort", "range_scope", "fill_values", "fill_strategies", "date_format"]},
            "value_counts_report": {"required": [], "optional": ["columns", "dropna", "sort", "include_header"]},
            "replace_values": {"required": ["column"], "optional": ["mapping", "old_value", "new_value"]},
            "fill_missing": {"required": [], "optional": ["column", "value", "mapping"]},
            "map_values": {"required": ["column", "mapping"], "optional": ["default", "source_column", "new_column"]},
            "derive_column": {"required": ["new_column"], "optional": ["value", "expression", "rowwise_rule", "source_columns"]},
            "derive_first_matching_label": {"required": ["new_column", "source_columns"], "optional": ["match_value", "labels", "default"]},
            "derive_matching_labels": {"required": ["new_column", "source_columns"], "optional": ["match_value", "labels"]},
            "parse_datetime": {"required": ["column"], "optional": ["format", "utc", "errors"]},
            "remove_timezone": {"required": ["column"], "optional": ["strategy"]},
            "format_datetime": {"required": ["column", "format"], "optional": []},
        },
        "examples": [
            {
                "operations": [
                    {"type": "rename_columns", "mapping": {"old_name": "new_name"}},
                    {"type": "cast_column", "column": "event_time", "to": "datetime64[ns]"},
                ]
            },
            {
                "operations": [
                    {"type": "melt", "id_vars": ["user_id"], "var_name": "metric", "value_name": "value"},
                ]
            },
            {
                "operations": [
                    {"type": "circular_shift", "column": "value", "shift": 1},
                ]
            }
        ],
    },
}


_NON_EXECUTABLE_METADATA_KEYS = {
    "description",
    "note",
    "notes",
    "method",
    "rationale",
    "implementation_note",
}


def get_node_contract(node_type: str) -> dict[str, Any]:
    return copy.deepcopy(NODE_CONTRACTS.get(node_type, {}))


def get_contract_examples(node_type: str) -> list[dict[str, Any]]:
    return copy.deepcopy(NODE_CONTRACTS.get(node_type, {}).get("examples", []))


def get_representative_contract_examples(node_type: str, max_examples: int = 2) -> list[dict[str, Any]]:
    examples = get_contract_examples(node_type)
    if max_examples <= 0 or len(examples) <= max_examples:
        return examples[:max_examples] if max_examples > 0 else []

    def _flatten_example(value: Any, prefix: str = "") -> set[str]:
        keys: set[str] = set()
        if isinstance(value, dict):
            for key, nested in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                keys.add(path)
                keys.update(_flatten_example(nested, path))
        elif isinstance(value, list):
            for item in value:
                keys.update(_flatten_example(item, prefix))
        elif prefix:
            keys.add(prefix)
        return keys

    selected_indices = [0]
    covered = _flatten_example(examples[0])

    while len(selected_indices) < max_examples:
        best_index = None
        best_gain = -1
        best_size = -1

        for index, example in enumerate(examples):
            if index in selected_indices:
                continue

            example_keys = _flatten_example(example)
            gain = len(example_keys - covered)
            size = len(example_keys)

            if gain > best_gain or (gain == best_gain and size > best_size):
                best_index = index
                best_gain = gain
                best_size = size

        if best_index is None:
            break

        selected_indices.append(best_index)
        covered.update(_flatten_example(examples[best_index]))

    return [copy.deepcopy(examples[index]) for index in selected_indices[:max_examples]]


def get_contract_param_schema(node_type: str) -> dict[str, str]:
    return dict(NODE_CONTRACTS.get(node_type, {}).get("top_level_params", {}))


def _clean_operation_dict(operation: dict[str, Any], *, allowed_keys: set[str]) -> dict[str, Any]:
    cleaned = {}
    for key, value in operation.items():
        if key in _NON_EXECUTABLE_METADATA_KEYS:
            continue
        if key in allowed_keys and value is not None:
            cleaned[key] = value
    return cleaned


def _normalize_operation_type(raw_type: Any, synonyms: dict[str, str]) -> str | None:
    if raw_type is None:
        return None

    normalized = str(raw_type).strip().lower()
    return synonyms.get(normalized, normalized)


def _coerce_operation_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _normalize_dataframe_input_params(params: dict[str, Any]) -> dict[str, Any]:
    raw = dict(params or {})
    normalized: dict[str, Any] = {}

    source_name = raw.pop("source_name", None)
    if source_name is None:
        source_name = raw.pop("dataframe_name", None)
    if source_name is not None:
        normalized["source_name"] = source_name

    if "copy" in raw:
        normalized["copy"] = raw.pop("copy")

    normalized.update(raw)
    return normalized


def _normalize_column_operation(operation: Any) -> Any:
    if not isinstance(operation, dict):
        return operation

    raw = dict(operation)
    raw_type = raw.pop("type", None) or raw.pop("operation", None) or raw.pop("action", None)

    if raw_type is None and "rename" in raw:
        raw_type = "rename_columns"
        raw["mapping"] = raw.pop("rename")
    elif raw_type is None and "order" in raw:
        raw_type = "reorder_columns"
        raw["columns"] = raw.pop("order")

    synonyms = {
        "rename": "rename_columns",
        "rename_columns": "rename_columns",
        "select": "select_columns",
        "keep_columns": "select_columns",
        "select_columns": "select_columns",
        "drop": "drop_columns",
        "drop_columns": "drop_columns",
        "reorder": "reorder_columns",
        "reorder_columns": "reorder_columns",
        "melt": "melt",
        "drop_duplicates": "drop_duplicate_rows",
        "drop_duplicate_rows": "drop_duplicate_rows",
        "assign": "derive_column",
        "set": "derive_column",
        "derive": "derive_column",
        "derive_column": "derive_column",
        "extract_regex": "extract_regex",
        "regex_extract": "extract_regex",
        "split_on_single_space": "split_column",
        "split_column": "split_column",
        "split_text": "split_column",
        "map_values": "map_values",
        "concatenate_columns": "concatenate_columns",
        "concat_columns": "concatenate_columns",
        "join_columns": "concatenate_columns",
        "explode": "explode_column",
        "explode_column": "explode_column",
        "derive_inverse_columns": "derive_inverse_columns",
        "derive_category_from_one_hot": "derive_first_matching_label",
        "one_hot_to_category": "derive_first_matching_label",
        "binary_columns_to_category": "derive_first_matching_label",
        "reverse_get_dummies": "derive_first_matching_label",
        "create_category_from_binaries": "derive_first_matching_label",
        "derive_first_matching_label": "derive_first_matching_label",
        "create_column": "create_column",
        "one_hot_to_category_list": "derive_matching_labels",
        "binary_columns_to_category_list": "derive_matching_labels",
        "reverse_get_dummies_list": "derive_matching_labels",
        "create_categories_from_binaries": "derive_matching_labels",
        "rowwise_derive_list": "derive_matching_labels",
        "derive_rowwise_list": "derive_matching_labels",
        "rowwise_list": "derive_matching_labels",
        "derive_matching_labels": "derive_matching_labels",
    }
    op_type = _normalize_operation_type(raw_type, synonyms)
    if not op_type:
        return operation

    if op_type == "create_column":
        if raw.get("as_list"):
            op_type = "derive_matching_labels"
        elif raw.get("method") == "first_match":
            op_type = "derive_first_matching_label"
        else:
            op_type = "derive_column"

    if op_type == "rename_columns":
        return _clean_operation_dict(
            {"type": op_type, "mapping": raw.get("mapping")},
            allowed_keys={"type", "mapping"},
        )

    if op_type in {"select_columns", "drop_columns"}:
        return _clean_operation_dict(
            {"type": op_type, "columns": raw.get("columns") or raw.get("keep_columns") or raw.get("drop_columns")},
            allowed_keys={"type", "columns"},
        )

    if op_type == "reorder_columns":
        return _clean_operation_dict(
            {"type": op_type, "columns": raw.get("columns") or raw.get("order")},
            allowed_keys={"type", "columns"},
        )

    if op_type == "melt":
        return _clean_operation_dict(
            {
                "type": op_type,
                "id_vars": raw.get("id_vars"),
                "value_vars": raw.get("value_vars"),
                "var_name": raw.get("var_name"),
                "value_name": raw.get("value_name"),
                "ignore_index": raw.get("ignore_index"),
                "preserve_row_order": raw.get("preserve_row_order"),
                "dropna": raw.get("dropna"),
            },
            allowed_keys={"type", "id_vars", "value_vars", "var_name", "value_name", "ignore_index", "preserve_row_order", "dropna"},
        )

    if op_type == "drop_duplicate_rows":
        return _clean_operation_dict(
            {
                "type": op_type,
                "subset": raw.get("subset"),
                "keep": raw.get("keep"),
                "ignore_index": raw.get("ignore_index"),
            },
            allowed_keys={"type", "subset", "keep", "ignore_index"},
        )

    if op_type == "extract_regex":
        return _clean_operation_dict(
            {
                "type": op_type,
                "source_column": raw.get("source_column") or raw.get("column"),
                "pattern": raw.get("pattern") or raw.get("regex"),
                "new_column": raw.get("new_column") or raw.get("target_column") or raw.get("output_column"),
                "default": raw.get("default"),
                "expand_group": raw.get("expand_group") or raw.get("group"),
            },
            allowed_keys={"type", "source_column", "pattern", "new_column", "default", "expand_group"},
        )

    if op_type == "split_column":
        separator = raw.get("separator")
        if separator is None and str(raw_type).strip().lower() == "split_on_single_space":
            separator = " "
        return _clean_operation_dict(
            {
                "type": op_type,
                "source_column": raw.get("source_column") or raw.get("column"),
                "new_columns": raw.get("new_columns") or raw.get("target_columns"),
                "separator": separator,
                "regex": raw.get("regex"),
                "max_splits": raw.get("max_splits") or raw.get("n"),
                "strip": raw.get("strip"),
            },
            allowed_keys={"type", "source_column", "new_columns", "separator", "regex", "max_splits", "strip"},
        )

    if op_type == "map_values":
        return _clean_operation_dict(
            {
                "type": op_type,
                "source_column": raw.get("source_column") or raw.get("column"),
                "new_column": raw.get("new_column") or raw.get("target_column") or raw.get("output_column"),
                "mapping": raw.get("mapping"),
                "default": raw.get("default"),
            },
            allowed_keys={"type", "source_column", "new_column", "mapping", "default"},
        )

    if op_type == "concatenate_columns":
        return _clean_operation_dict(
            {
                "type": op_type,
                "new_column": raw.get("new_column") or raw.get("output_column") or raw.get("column"),
                "source_columns": raw.get("source_columns") or raw.get("columns"),
                "separator": raw.get("separator"),
                "strip": raw.get("strip"),
            },
            allowed_keys={"type", "new_column", "source_columns", "separator", "strip"},
        )

    if op_type == "explode_column":
        return _clean_operation_dict(
            {
                "type": op_type,
                "source_column": raw.get("source_column") or raw.get("column"),
                "new_column": raw.get("new_column") or raw.get("output_column"),
                "ignore_index": raw.get("ignore_index"),
            },
            allowed_keys={"type", "source_column", "new_column", "ignore_index"},
        )

    if op_type == "derive_inverse_columns":
        return _clean_operation_dict(
            {
                "type": op_type,
                "source_columns": raw.get("source_columns") or raw.get("columns"),
                "prefix": raw.get("prefix"),
                "suffix": raw.get("suffix"),
                "numerator": raw.get("numerator"),
                "preserve_zero": raw.get("preserve_zero"),
                "zero_value": raw.get("zero_value"),
            },
            allowed_keys={"type", "source_columns", "prefix", "suffix", "numerator", "preserve_zero", "zero_value"},
        )

    if op_type == "derive_function_columns":
        source_columns = raw.get("source_columns") or raw.get("columns")
        collapsed_function = re.sub(r"\s+", "", str(raw.get("function") or "").lower())
        if collapsed_function == "lambdarow:dict(zip(row.index,list(row.dropna().values)+[np.nan]*(len(row)-row.count())))":
            return _clean_operation_dict(
                {
                    "type": "shift_non_nulls_left",
                    "columns": source_columns,
                },
                allowed_keys={"type", "columns"},
            )
        if collapsed_function == "lambdacol:pd.series(list(col[col.isnull()])+list(col[col.notnull()]))":
            return _clean_operation_dict(
                {
                    "type": "shift_nulls_to_top_per_column",
                    "columns": source_columns,
                },
                allowed_keys={"type", "columns"},
            )
        normalized_function = {
            "np.exp": "exp",
            "numpy.exp": "exp",
            "1/(1+np.exp(-col))": "sigmoid",
            "1/(1+numpy.exp(-col))": "sigmoid",
            "lambdacol:col/col.sum()": "normalize_sum",
            "col/col.sum()": "normalize_sum",
        }.get(collapsed_function, raw.get("function"))
        if re.fullmatch(
            r"lambda([A-Za-z_][A-Za-z0-9_]*):1/\(1\+(?:np|numpy)\.exp\(-\1\)\)",
            collapsed_function,
        ):
            normalized_function = "sigmoid"
        return _clean_operation_dict(
            {
                "type": op_type,
                "source_columns": source_columns,
                "function": normalized_function,
                "prefix": raw.get("prefix"),
                "suffix": raw.get("suffix"),
                "preserve_zero": raw.get("preserve_zero"),
                "zero_value": raw.get("zero_value"),
            },
            allowed_keys={"type", "source_columns", "function", "prefix", "suffix", "preserve_zero", "zero_value"},
        )

    if op_type == "derive_column":
        source_columns = raw.get("source_columns") or raw.get("columns")
        match_value = raw.get("match_value", raw.get("present_value", raw.get("active_value", raw.get("value"))))
        if source_columns and match_value is not None and raw.get("expression") is None and raw.get("rowwise_rule") is None:
            return _clean_operation_dict(
                {
                    "type": "derive_matching_labels",
                    "new_column": raw.get("new_column") or raw.get("output_column") or raw.get("column") or raw.get("name"),
                    "source_columns": source_columns,
                    "match_value": match_value,
                },
                allowed_keys={"type", "new_column", "source_columns", "match_value"},
            )
        return _clean_operation_dict(
            {
                "type": op_type,
                "new_column": raw.get("new_column") or raw.get("output_column") or raw.get("column") or raw.get("name"),
                "value": raw.get("value"),
                "expression": raw.get("expression"),
                "rowwise_rule": raw.get("rowwise_rule") or raw.get("rule"),
                "source_columns": raw.get("source_columns") or raw.get("columns"),
            },
            allowed_keys={"type", "new_column", "value", "expression", "rowwise_rule", "source_columns"},
        )

    if op_type == "derive_first_matching_label":
        labels = raw.get("labels")
        source_columns = raw.get("source_columns") or raw.get("columns") or raw.get("from_columns")
        match_value = raw.get("match_value", raw.get("present_value", raw.get("active_value", raw.get("value", 1))))
        if labels is None and raw.get("matches"):
            labels = {
                match["columns"][0]: match.get("label", match["columns"][0])
                for match in raw["matches"]
                if isinstance(match, dict) and match.get("columns")
            }
        if source_columns is None and raw.get("matches"):
            source_columns = [
                match["columns"][0]
                for match in raw["matches"]
                if isinstance(match, dict) and match.get("columns")
            ]
        if raw.get("matches"):
            first_match = next(
                (
                    match
                    for match in raw["matches"]
                    if isinstance(match, dict) and match.get("value") is not None
                ),
                None,
            )
            if first_match is not None:
                match_value = first_match.get("value", match_value)

        return _clean_operation_dict(
            {
                "type": op_type,
                "new_column": raw.get("new_column") or raw.get("output_column") or raw.get("column") or raw.get("name"),
                "source_columns": source_columns,
                "match_value": match_value,
                "labels": labels,
                "default": raw.get("default"),
            },
            allowed_keys={"type", "new_column", "source_columns", "match_value", "labels", "default"},
        )

    if op_type == "derive_matching_labels":
        labels = raw.get("labels")
        source_columns = raw.get("source_columns") or raw.get("columns") or raw.get("from_columns")
        match_value = raw.get("match_value", raw.get("present_value", raw.get("active_value", raw.get("value", 1))))
        if source_columns is None and raw.get("matches"):
            source_columns = [
                match["columns"][0]
                for match in raw["matches"]
                if isinstance(match, dict) and match.get("columns")
            ]
        if labels is None and raw.get("matches"):
            labels = {
                match["columns"][0]: match.get("label", match["columns"][0])
                for match in raw["matches"]
                if isinstance(match, dict) and match.get("columns")
            }
        return _clean_operation_dict(
            {
                "type": op_type,
                "new_column": raw.get("new_column") or raw.get("output_column") or raw.get("column") or raw.get("name"),
                "source_columns": source_columns,
                "match_value": match_value,
                "labels": labels,
            },
            allowed_keys={"type", "new_column", "source_columns", "match_value", "labels"},
        )

    return operation


def _normalize_column_transformer_params(params: dict[str, Any]) -> dict[str, Any]:
    raw = dict(params or {})
    operations: list[Any] = []

    if "rename" in raw:
        operations.append({"type": "rename_columns", "mapping": raw.pop("rename")})
    if "keep_columns" in raw:
        operations.append({"type": "select_columns", "columns": raw.pop("keep_columns")})
    if "drop_columns" in raw:
        operations.append({"type": "drop_columns", "columns": raw.pop("drop_columns")})
    if "reorder_columns" in raw:
        operations.append({"type": "reorder_columns", "columns": raw.pop("reorder_columns")})
    if "derive" in raw:
        operations.append({"type": "derive_column", **(raw.pop("derive") or {})})

    for key in ("operations", "transformations"):
        operations.extend(_coerce_operation_list(raw.pop(key, None)))
    leftover_type = raw.pop("operation", None) or raw.pop("transformation", None)
    if leftover_type is not None:
        candidate = {"type": leftover_type}
        for extra_key in list(raw.keys()):
            if extra_key in {
                "mapping",
                "columns",
                "keep_columns",
                "drop_columns",
                "order",
                "subset",
                "keep",
                "ignore_index",
                "new_column",
                "output_column",
                "column",
                "name",
                "value",
                "expression",
                "rowwise_rule",
                "rule",
                "source_columns",
                "matches",
                "match_value",
                "present_value",
                "active_value",
                "labels",
                "default",
                "source_column",
                "pattern",
                "regex",
                "target_column",
                "group",
                "expand_group",
                "new_columns",
                "target_columns",
                "separator",
                "max_splits",
                "n",
                "strip",
                "ignore_index",
                "prefix",
                "suffix",
                "numerator",
                "preserve_zero",
                "zero_value",
                "function",
                "id_vars",
                "value_vars",
                "var_name",
                "value_name",
                "preserve_row_order",
                "dropna",
            }:
                candidate[extra_key] = raw.pop(extra_key)
        operations.append(candidate)

    for item in _coerce_operation_list(raw.pop("new_columns", None)):
        if isinstance(item, dict):
            if item.get("type") == "rowwise_list":
                operations.append(
                    {
                        "type": "derive_matching_labels",
                        "new_column": item.get("name") or item.get("new_column"),
                        "source_columns": item.get("source_columns"),
                        "match_value": item.get("match_value", 1),
                    }
                )
            else:
                operations.append(
                    {
                        "type": "derive_column",
                        "new_column": item.get("name") or item.get("new_column"),
                        "expression": item.get("expression"),
                        "rowwise_rule": item.get("rowwise_rule") or item.get("rule"),
                        "source_columns": item.get("source_columns"),
                        "value": item.get("value"),
                    }
                )

    operations.extend(_coerce_operation_list(raw.pop("post_ops", None)))

    normalized = {"operations": [_normalize_column_operation(item) for item in operations]}
    normalized.update(raw)
    return normalized


def _normalize_value_rule(column: str, rule: dict[str, Any]) -> dict[str, Any]:
    return _clean_operation_dict(
        {
            "column": column,
            "threshold": rule.get("threshold") or rule.get("min_count") or rule.get("min_count_inclusive"),
            "replacement": rule.get("replacement") or rule.get("replace_with") or rule.get("other_label") or rule.get("others_label") or rule.get("other_value"),
            "preserve_values": rule.get("preserve_values") or rule.get("reserve_values") or rule.get("reserved_values") or rule.get("keep_values"),
        },
        allowed_keys={"column", "threshold", "replacement", "preserve_values"},
    )


def _normalize_value_operation(operation: Any) -> Any:
    if not isinstance(operation, dict):
        return operation

    raw = dict(operation)
    raw_type = raw.pop("type", None) or raw.pop("operation", None) or raw.pop("action", None) or raw.pop("mode", None) or raw.pop("method", None)
    synonyms = {
        "replace_low_frequency": "replace_infrequent",
        "replace_infrequent_with_other": "replace_infrequent",
        "replace_infrequent": "replace_infrequent",
        "rare_value_bucketing_by_count": "replace_infrequent",
        "replace": "replace_values",
        "replace_values": "replace_values",
        "map": "map_values",
        "map_values": "map_values",
        "fillna": "fill_missing",
        "fill_missing": "fill_missing",
        "lower": "normalize_text",
        "upper": "normalize_text",
        "strip": "normalize_text",
        "normalize_text": "normalize_text",
        "replace_substring": "replace_substring",
        "strip_prefix": "strip_prefix",
        "coerce_numeric": "coerce_numeric",
        "clip": "clip_values",
        "clip_values": "clip_values",
        "round": "round_values",
        "round_values": "round_values",
        "factorize": "factorize_values",
        "factorize_values": "factorize_values",
        "encode_categories": "factorize_values",
    }
    op_type = _normalize_operation_type(raw_type, synonyms)

    if op_type == "replace_infrequent":
        column_rules = raw.get("column_rules")
        if column_rules is None and isinstance(raw.get("rules"), dict):
            column_rules = [_normalize_value_rule(column, rule) for column, rule in raw["rules"].items()]
        elif column_rules is None and isinstance(raw.get("rules"), list):
            column_rules = []
            for rule in raw["rules"]:
                if isinstance(rule, dict):
                    column_rules.append(
                        _clean_operation_dict(
                            {
                                "column": rule.get("column"),
                                "threshold": rule.get("threshold") or rule.get("min_count") or rule.get("min_count_inclusive"),
                                "replacement": rule.get("replacement") or rule.get("replace_with") or rule.get("other_label") or rule.get("others_label") or rule.get("other_value"),
                                "preserve_values": rule.get("preserve_values") or rule.get("reserve_values") or rule.get("reserved_values") or rule.get("keep_values"),
                            },
                            allowed_keys={"column", "threshold", "replacement", "preserve_values"},
                        )
                    )

        return _clean_operation_dict(
            {
                "type": "replace_infrequent",
                "columns": raw.get("columns") if isinstance(raw.get("columns"), list) else None,
                "threshold": raw.get("threshold") or raw.get("min_count") or raw.get("min_count_inclusive"),
                "replacement": raw.get("replacement") or raw.get("replace_with") or raw.get("other_label") or raw.get("others_label") or raw.get("other_value"),
                "preserve_values": raw.get("preserve_values") or raw.get("reserve_values") or raw.get("reserved_values") or raw.get("keep_values"),
                "column_rules": column_rules,
                "preserve_na": raw.get("preserve_na"),
            },
            allowed_keys={"type", "columns", "threshold", "replacement", "preserve_values", "column_rules", "preserve_na"},
        )

    if op_type == "replace_values":
        return _clean_operation_dict(
            {
                "type": "replace_values",
                "column": raw.get("column"),
                "mapping": raw.get("mapping"),
                "old_value": raw.get("old_value"),
                "new_value": raw.get("new_value"),
            },
            allowed_keys={"type", "column", "mapping", "old_value", "new_value"},
        )

    if op_type == "map_values":
        return _clean_operation_dict(
            {
                "type": "map_values",
                "column": raw.get("column") or raw.get("target_column"),
                "mapping": raw.get("mapping"),
                "default": raw.get("default"),
            },
            allowed_keys={"type", "column", "mapping", "default"},
        )

    if op_type == "factorize_values":
        return _clean_operation_dict(
            {
                "type": "factorize_values",
                "column": raw.get("column") or raw.get("target_column"),
                "start": raw.get("start"),
                "sort": raw.get("sort"),
                "new_column": raw.get("new_column") or raw.get("target_column") or raw.get("output_column"),
            },
            allowed_keys={"type", "column", "start", "sort", "new_column"},
        )

    if op_type == "fill_missing":
        return _clean_operation_dict(
            {
                "type": "fill_missing",
                "column": raw.get("column"),
                "value": raw.get("value"),
                "mapping": raw.get("mapping") or raw.get("fillna"),
            },
            allowed_keys={"type", "column", "value", "mapping"},
        )

    if op_type == "normalize_text":
        style = raw_type if str(raw_type).strip().lower() in {"lower", "upper", "strip"} else raw.get("style")
        return _clean_operation_dict(
            {
                "type": "normalize_text",
                "column": raw.get("column"),
                "style": style,
            },
            allowed_keys={"type", "column", "style"},
        )

    if op_type == "replace_substring":
        return _clean_operation_dict(
            {
                "type": "replace_substring",
                "column": raw.get("column"),
                "old": raw.get("old") or raw.get("substring") or raw.get("pattern"),
                "new": raw.get("new") or raw.get("replacement") or raw.get("replace_with"),
                "regex": raw.get("regex"),
                "case": raw.get("case"),
            },
            allowed_keys={"type", "column", "old", "new", "regex", "case"},
        )

    if op_type == "strip_prefix":
        return _clean_operation_dict(
            {
                "type": "strip_prefix",
                "column": raw.get("column"),
                "prefix": raw.get("prefix"),
            },
            allowed_keys={"type", "column", "prefix"},
        )

    if op_type == "coerce_numeric":
        return _clean_operation_dict(
            {
                "type": "coerce_numeric",
                "column": raw.get("column"),
                "new_column": raw.get("new_column") or raw.get("target_column") or raw.get("output_column"),
                "errors": raw.get("errors"),
            },
            allowed_keys={"type", "column", "new_column", "errors"},
        )

    if op_type == "clip_values":
        return _clean_operation_dict(
            {
                "type": "clip_values",
                "column": raw.get("column"),
                "lower": raw.get("lower"),
                "upper": raw.get("upper"),
                "new_column": raw.get("new_column") or raw.get("target_column") or raw.get("output_column"),
                "errors": raw.get("errors"),
            },
            allowed_keys={"type", "column", "lower", "upper", "new_column", "errors"},
        )

    if op_type == "round_values":
        return _clean_operation_dict(
            {
                "type": "round_values",
                "column": raw.get("column"),
                "decimals": raw.get("decimals"),
                "new_column": raw.get("new_column") or raw.get("target_column") or raw.get("output_column"),
            },
            allowed_keys={"type", "column", "decimals", "new_column"},
        )

    if op_type == "parse_duration_text":
        return _clean_operation_dict(
            {
                "type": "parse_duration_text",
                "column": raw.get("column"),
                "new_column": raw.get("new_column") or raw.get("target_column") or raw.get("output_column"),
                "result": raw.get("result"),
                "unit_map": raw.get("unit_map") or raw.get("mapping"),
                "errors": raw.get("errors"),
            },
            allowed_keys={"type", "column", "new_column", "result", "unit_map", "errors"},
        )

    return operation


def _normalize_value_transformer_params(params: dict[str, Any]) -> dict[str, Any]:
    raw = dict(params or {})
    operations: list[Any] = []

    if "rules" in raw:
        operations.append({"type": "replace_infrequent", "rules": raw.pop("rules")})
    elif isinstance(raw.get("columns"), dict):
        operations.append(
            {
                "type": "replace_infrequent",
                "column_rules": [
                    _normalize_value_rule(column, rule if isinstance(rule, dict) else {"threshold": rule})
                    for column, rule in raw.pop("columns").items()
                ],
            }
        )
    elif "columns" in raw and any(key in raw for key in ("threshold", "min_count", "min_count_inclusive", "replacement", "replace_with", "other_label", "others_label")):
        operations.append(
            {
                "type": "replace_infrequent",
                "columns": raw.pop("columns"),
                "threshold": raw.pop("threshold", None) or raw.pop("min_count", None) or raw.pop("min_count_inclusive", None),
                "replacement": raw.pop("replacement", None) or raw.pop("replace_with", None) or raw.pop("other_label", None) or raw.pop("others_label", None),
                "preserve_values": raw.pop("preserve_values", None) or raw.pop("reserve_values", None) or raw.pop("reserved_values", None) or raw.pop("keep_values", None),
                "preserve_na": raw.pop("preserve_na", None),
            }
        )

    if "replace" in raw:
        operations.append(raw.pop("replace"))
    if "mapping" in raw and ("column" in raw or "target_column" in raw or raw.get("operation") in {"map", "map_values"}):
        operations.append(
            {
                "type": raw.pop("operation", None) or "map_values",
                "column": raw.pop("column", None) or raw.pop("target_column", None),
                "mapping": raw.pop("mapping"),
                "default": raw.pop("default", None),
            }
        )
    if "fillna" in raw:
        operations.append({"type": "fill_missing", "mapping": raw.pop("fillna")})

    for key in ("operations", "transformations"):
        operations.extend(_coerce_operation_list(raw.pop(key, None)))

    leftover_type = raw.pop("operation", None) or raw.pop("transformation", None)
    if leftover_type is not None and not operations:
        candidate = {"type": leftover_type}
        for extra_key in list(raw.keys()):
            if extra_key in {
                "column",
                "columns",
                "threshold",
                "min_count",
                "min_count_inclusive",
                "replacement",
                "replace_with",
                "other_label",
                "others_label",
                "other_value",
                "preserve_values",
                "reserve_values",
                "reserved_values",
                "keep_values",
                "column_rules",
                "mapping",
                "old_value",
                "new_value",
                "start",
                "sort",
                "old",
                "new",
                "regex",
                "case",
                "prefix",
                "lower",
                "upper",
                "errors",
                "decimals",
                "new_column",
                "output_column",
                "default",
                "fillna",
                "style",
            }:
                candidate[extra_key] = raw.pop(extra_key)
        operations.append(candidate)

    normalized = {"operations": [_normalize_value_operation(item) for item in operations]}
    normalized.update(raw)
    return normalized


def _normalize_datetime_operation(operation: Any) -> Any:
    if not isinstance(operation, dict):
        return operation

    raw = dict(operation)
    raw_type = raw.pop("type", None) or raw.pop("operation", None) or raw.pop("action", None)
    synonyms = {
        "parse": "parse_datetime",
        "to_datetime": "parse_datetime",
        "parse_datetime": "parse_datetime",
        "cast": "parse_datetime",
        "remove_timezone": "remove_timezone",
        "remove_tzinfo": "remove_timezone",
        "remove_tz": "remove_timezone",
        "strip_timezone": "remove_timezone",
        "format": "format_datetime",
        "format_datetime": "format_datetime",
        "extract": "extract_part",
        "extract_part": "extract_part",
        "date_diff": "date_diff",
    }
    op_type = _normalize_operation_type(raw_type, synonyms)
    if not op_type:
        return operation

    if op_type == "parse_datetime":
        return _clean_operation_dict(
            {
                "type": op_type,
                "column": raw.get("column"),
                "format": raw.get("format"),
                "utc": raw.get("utc"),
                "errors": raw.get("errors"),
                "index_level": raw.get("index_level") or raw.get("level"),
            },
            allowed_keys={"type", "column", "format", "utc", "errors", "index_level"},
        )

    if op_type == "remove_timezone":
        return _clean_operation_dict(
            {
                "type": op_type,
                "column": raw.get("column"),
                "strategy": raw.get("strategy", "preserve_wall_time"),
                "index_level": raw.get("index_level") or raw.get("level"),
            },
            allowed_keys={"type", "column", "strategy", "index_level"},
        )

    if op_type == "format_datetime":
        return _clean_operation_dict(
            {
                "type": op_type,
                "column": raw.get("column"),
                "format": raw.get("format"),
                "index_level": raw.get("index_level") or raw.get("level"),
            },
            allowed_keys={"type", "column", "format", "index_level"},
        )

    if op_type == "extract_part":
        return _clean_operation_dict(
            {
                "type": op_type,
                "column": raw.get("column"),
                "part": raw.get("part"),
                "new_column": raw.get("new_column") or raw.get("output_column"),
                "index_level": raw.get("index_level") or raw.get("level"),
            },
            allowed_keys={"type", "column", "part", "new_column", "index_level"},
        )

    if op_type == "date_diff":
        return _clean_operation_dict(
            {
                "type": "date_diff",
                "column": raw.get("column"),
                "other_column": raw.get("other_column") or raw.get("end_column"),
                "new_column": raw.get("new_column") or raw.get("output_column"),
                "unit": raw.get("unit"),
                "index_level": raw.get("index_level") or raw.get("level"),
                "other_index_level": raw.get("other_index_level"),
            },
            allowed_keys={"type", "column", "other_column", "new_column", "unit", "index_level", "other_index_level"},
        )

    return operation


def _normalize_datetime_transformer_params(params: dict[str, Any]) -> dict[str, Any]:
    raw = dict(params or {})
    operations: list[Any] = []

    parse = raw.pop("parse", None)
    if isinstance(parse, dict):
        for column, spec in parse.items():
            op = {"type": "parse_datetime", "column": column}
            if isinstance(spec, dict):
                op.update(spec)
            operations.append(op)
    elif isinstance(parse, list):
        for column in parse:
            operations.append({"type": "parse_datetime", "column": column})

    remove_timezone = raw.pop("remove_timezone", None)
    if remove_timezone is None and "remove_tz" in raw:
        remove_timezone = raw.pop("remove_tz")
    if isinstance(remove_timezone, list):
        for column in remove_timezone:
            operations.append({"type": "remove_timezone", "column": column, "strategy": "preserve_wall_time"})
    elif isinstance(remove_timezone, str):
        operations.append({"type": "remove_timezone", "column": remove_timezone, "strategy": "preserve_wall_time"})
    elif remove_timezone and isinstance(raw.get("columns"), list):
        for column in raw["columns"]:
            operations.append({"type": "remove_timezone", "column": column, "strategy": "preserve_wall_time"})

    fmt = raw.pop("fmt", None) or raw.get("format")
    if fmt and isinstance(raw.get("columns"), list) and not raw.get("operation"):
        for column in raw["columns"]:
            operations.append({"type": "format_datetime", "column": column, "format": fmt})
        raw.pop("columns", None)
        raw.pop("format", None)

    for key in ("operations", "transformations"):
        operations.extend(_coerce_operation_list(raw.pop(key, None)))

    leftover_type = raw.pop("operation", None) or raw.pop("transformation", None) or raw.pop("action", None)
    if leftover_type is not None:
        if isinstance(raw.get("columns"), list) and str(leftover_type).strip().lower() in {"remove_timezone", "remove_tz", "format", "format_datetime"}:
            for column in raw.pop("columns"):
                op = {"type": leftover_type, "column": column}
                if "format" in raw:
                    op["format"] = raw["format"]
                operations.append(op)
        else:
            candidate = {"type": leftover_type}
            for extra_key in list(raw.keys()):
                if extra_key in {"column", "columns", "format", "utc", "errors", "part", "new_column", "output_column", "strategy", "other_column", "end_column", "unit", "index_level", "level", "other_index_level"}:
                    candidate[extra_key] = raw.pop(extra_key)
            operations.append(candidate)

    normalized = {"operations": [_normalize_datetime_operation(item) for item in operations]}
    normalized.update(raw)
    return normalized


def _normalize_data_operation(operation: Any) -> Any:
    if not isinstance(operation, dict):
        return operation

    raw = dict(operation)
    raw_type = raw.pop("type", None) or raw.pop("operation", None) or raw.pop("action", None)
    synonyms = {
        "cast": "cast_column",
        "cast_column": "cast_column",
        "melt": "melt",
        "circular_shift": "circular_shift",
        "shift_non_nulls_left": "shift_non_nulls_left",
        "shift_nulls_to_top_per_column": "shift_nulls_to_top_per_column",
        "expand_date_range_per_group": "expand_date_range_per_group",
        "value_counts_report": "value_counts_report",
    }
    op_type = _normalize_operation_type(raw_type, synonyms)
    if not op_type:
        return operation

    if op_type == "cast_column":
        return _clean_operation_dict(
            {
                "type": op_type,
                "column": raw.get("column"),
                "to": raw.get("to"),
                "dtype": raw.get("dtype"),
                "target_type": raw.get("target_type"),
                "format": raw.get("format"),
                "utc": raw.get("utc"),
                "errors": raw.get("errors"),
            },
            allowed_keys={"type", "column", "to", "dtype", "target_type", "format", "utc", "errors"},
        )

    if op_type == "melt":
        return _clean_operation_dict(
            {
                "type": op_type,
                "id_vars": raw.get("id_vars"),
                "value_vars": raw.get("value_vars"),
                "var_name": raw.get("var_name"),
                "value_name": raw.get("value_name"),
                "ignore_index": raw.get("ignore_index"),
                "preserve_row_order": raw.get("preserve_row_order"),
                "dropna": raw.get("dropna"),
            },
            allowed_keys={"type", "id_vars", "value_vars", "var_name", "value_name", "ignore_index", "preserve_row_order", "dropna"},
        )

    if op_type == "circular_shift":
        return _clean_operation_dict(
            {
                "type": op_type,
                "column": raw.get("column"),
                "shift": raw.get("shift"),
            },
            allowed_keys={"type", "column", "shift"},
        )

    if op_type == "shift_non_nulls_left":
        return _clean_operation_dict(
            {
                "type": op_type,
                "columns": raw.get("columns"),
                "fill_value": raw.get("fill_value"),
            },
            allowed_keys={"type", "columns", "fill_value"},
        )

    if op_type == "shift_nulls_to_top_per_column":
        return _clean_operation_dict(
            {
                "type": op_type,
                "columns": raw.get("columns"),
            },
            allowed_keys={"type", "columns"},
        )

    if op_type == "expand_date_range_per_group":
        return _clean_operation_dict(
            {
                "type": op_type,
                "group_by": raw.get("group_by"),
                "date_column": raw.get("date_column"),
                "freq": raw.get("freq"),
                "sort": raw.get("sort"),
                "range_scope": raw.get("range_scope"),
                "fill_values": raw.get("fill_values"),
                "fill_strategies": raw.get("fill_strategies"),
                "date_format": raw.get("date_format"),
            },
            allowed_keys={"type", "group_by", "date_column", "freq", "sort", "range_scope", "fill_values", "fill_strategies", "date_format"},
        )

    if op_type == "value_counts_report":
        return _clean_operation_dict(
            {
                "type": op_type,
                "columns": raw.get("columns"),
                "dropna": raw.get("dropna"),
                "sort": raw.get("sort"),
                "include_header": raw.get("include_header"),
            },
            allowed_keys={"type", "columns", "dropna", "sort", "include_header"},
        )

    return operation


def _normalize_generic_transformer_operation(operation: Any) -> Any:
    normalized = _normalize_column_operation(operation)
    if (
        isinstance(normalized, dict)
        and normalized.get("type") == "map_values"
        and normalized.get("new_column") is None
    ):
        normalized = operation
    if isinstance(normalized, dict) and normalized.get("type") in {
        "rename_columns",
        "select_columns",
        "drop_columns",
        "reorder_columns",
        "melt",
        "drop_duplicate_rows",
        "extract_regex",
        "split_column",
        "map_values",
        "concatenate_columns",
        "explode_column",
        "derive_inverse_columns",
        "derive_function_columns",
        "derive_column",
        "derive_first_matching_label",
        "derive_matching_labels",
    }:
        return normalized

    normalized = _normalize_value_operation(operation)
    if isinstance(normalized, dict) and normalized.get("type") in {
        "replace_infrequent",
        "replace_values",
        "map_values",
        "factorize_values",
        "fill_missing",
        "normalize_text",
        "replace_substring",
        "strip_prefix",
        "coerce_numeric",
        "clip_values",
        "round_values",
        "parse_duration_text",
    }:
        return normalized

    normalized = _normalize_datetime_operation(operation)
    if isinstance(normalized, dict) and normalized.get("type") in {
        "parse_datetime",
        "remove_timezone",
        "format_datetime",
        "extract_part",
        "date_diff",
    }:
        return normalized

    normalized = _normalize_data_operation(operation)
    if isinstance(normalized, dict) and normalized.get("type") in {
        "cast_column",
        "melt",
        "circular_shift",
        "shift_non_nulls_left",
        "shift_nulls_to_top_per_column",
        "expand_date_range_per_group",
        "value_counts_report",
    }:
        return normalized

    return operation


def _normalize_data_transformer_params(params: dict[str, Any]) -> dict[str, Any]:
    raw = dict(params or {})
    operations: list[Any] = []

    if "rename" in raw:
        operations.append({"type": "rename_columns", "mapping": raw.pop("rename")})
    if "replace" in raw:
        operations.extend(_coerce_operation_list(raw.pop("replace")))
    if "fillna" in raw:
        operations.append({"type": "fill_missing", "mapping": raw.pop("fillna")})
    if "assign" in raw:
        assign_spec = raw.pop("assign")
        if isinstance(assign_spec, dict):
            for column, spec in assign_spec.items():
                if isinstance(spec, dict):
                    operations.append({"type": "derive_column", "new_column": column, "expression": spec.get("expression"), "value": spec.get("value")})
                else:
                    operations.append({"type": "derive_column", "new_column": column, "value": spec})

    for key in ("operations", "transformations"):
        operations.extend(_coerce_operation_list(raw.pop(key, None)))
    for key in ("operation", "transformation"):
        value = raw.pop(key, None)
        if value is not None:
            operations.append(value)

    normalized = {"operations": [_normalize_generic_transformer_operation(item) for item in operations]}
    normalized.update(raw)
    return normalized


def _normalize_group_key(raw_key: Any, index: int) -> Any:
    if isinstance(raw_key, str):
        return raw_key
    if not isinstance(raw_key, dict):
        return raw_key

    key_type = raw_key.get("type") or raw_key.get("kind")
    if key_type is None:
        if raw_key.get("expression") is not None:
            key_type = "expression"
        elif raw_key.get("column") is not None or raw_key.get("name") is not None:
            key_type = "column"

    key_type = str(key_type).strip().lower() if key_type is not None else None
    if key_type == "column":
        return _clean_operation_dict(
            {
                "type": "column",
                "column": raw_key.get("column") or raw_key.get("name"),
            },
            allowed_keys={"type", "column"},
        )

    if key_type == "expression":
        return _clean_operation_dict(
            {
                "type": "expression",
                "name": raw_key.get("name") or raw_key.get("column") or f"group_key_{index + 1}",
                "expression": raw_key.get("expression"),
            },
            allowed_keys={"type", "name", "expression"},
        )

    return raw_key


def _normalize_aggregation_spec(raw_spec: Any) -> Any:
    if not isinstance(raw_spec, dict):
        return raw_spec

    spec = dict(raw_spec)
    op = spec.get("op") or spec.get("agg") or spec.get("agg_func") or spec.get("function")
    normalized = {
        "column": spec.get("column"),
        "columns": spec.get("columns"),
        "columns_regex": spec.get("columns_regex"),
        "columns_prefix": spec.get("columns_prefix"),
        "columns_suffix": spec.get("columns_suffix"),
        "exclude_columns": spec.get("exclude_columns"),
        "op": op,
        "output": spec.get("output") or spec.get("name") or spec.get("as"),
        "sort_by": spec.get("sort_by"),
        "ascending": spec.get("ascending"),
        "unique": spec.get("unique"),
    }
    return _clean_operation_dict(
        normalized,
        allowed_keys={"column", "columns", "columns_regex", "columns_prefix", "columns_suffix", "exclude_columns", "op", "output", "sort_by", "ascending", "unique"},
    )


def _normalize_lambda_aggregations(expression: str) -> list[dict[str, Any]] | None:
    text = expression.strip()
    if not text.startswith("lambda"):
        return None

    fixed_specs: list[dict[str, Any]] = []
    for column, op in re.findall(r"'([^']+)'\s*:\s*'([^']+)'", text):
        fixed_specs.append({"column": column, "op": op, "output": column})

    conditional_match = re.search(
        r"col\.endswith\('([^']+)'\)\s*else\s*'([^']+)'",
        text,
    )
    true_op_match = re.search(
        r"'([^']+)'\s+if\s+col\.endswith\('([^']+)'\)\s+else\s+'([^']+)'",
        text,
    )
    excluded_columns_match = re.search(r"col\s+not\s+in\s+\[([^\]]+)\]", text)

    excluded_columns: list[str] = []
    if excluded_columns_match:
        excluded_columns = re.findall(r"'([^']+)'", excluded_columns_match.group(1))

    if true_op_match:
        true_op, suffix, false_op = true_op_match.groups()
    elif conditional_match:
        suffix, false_op = conditional_match.groups()
        true_op = re.search(r":\s*\('([^']+)'", text).group(1) if re.search(r":\s*\('([^']+)'", text) else None
    else:
        true_op = None
        suffix = None
        false_op = None

    if true_op and suffix and false_op:
        fixed_specs.append(
            {
                "op": true_op,
                "columns_suffix": suffix,
                "exclude_columns": excluded_columns,
            }
        )
        fixed_specs.append(
            {
                "op": false_op,
                "columns_regex": rf"^(?!.*{re.escape(suffix)}$).+",
                "exclude_columns": excluded_columns,
            }
        )

    return fixed_specs or None


def _normalize_aggregator_params(params: dict[str, Any]) -> dict[str, Any]:
    raw = dict(params or {})
    normalized: dict[str, Any] = {}

    group_keys = raw.pop("group_keys", None)
    group_by = raw.pop("group_by", None)
    if group_keys is None and group_by is not None:
        group_keys = group_by

    if group_keys is not None:
        key_list = group_keys if isinstance(group_keys, list) else [group_keys]
        normalized["group_keys"] = [
            _normalize_group_key(item, index)
            for index, item in enumerate(key_list)
        ]

    aggregations = raw.pop("aggregations", None)
    legacy_agg = raw.pop("agg_func", None)
    if aggregations is None and legacy_agg is not None:
        if isinstance(legacy_agg, str):
            parsed = _normalize_lambda_aggregations(legacy_agg)
            if parsed is not None:
                aggregations = parsed
            else:
                aggregations = [{"op": legacy_agg, "output": legacy_agg}]
        elif isinstance(legacy_agg, list):
            aggregations = legacy_agg
        elif isinstance(legacy_agg, dict):
            aggregations = []
            for column, spec in legacy_agg.items():
                if isinstance(spec, dict):
                    normalized_spec = dict(spec)
                    normalized_spec.setdefault("column", column)
                    normalized_spec.setdefault("output", column)
                    aggregations.append(normalized_spec)
                    continue
                if isinstance(spec, list):
                    for op in spec:
                        aggregations.append(
                            {
                                "column": column,
                                "op": op,
                                "output": f"{column}_{op}",
                            }
                        )
                else:
                    aggregations.append(
                        {
                            "column": column,
                            "op": spec,
                            "output": column,
                        }
                    )

    if aggregations is not None:
        aggregation_list = aggregations if isinstance(aggregations, list) else [aggregations]
        normalized["aggregations"] = [
            _normalize_aggregation_spec(item)
            for item in aggregation_list
        ]

    for key in ("dropna", "sort_by", "ascending", "reset_index"):
        if key in raw:
            normalized[key] = raw.pop(key)

    normalized.update(raw)
    return normalized


def _normalize_date_range_expander_params(params: dict[str, Any]) -> dict[str, Any]:
    raw = dict(params or {})
    normalized: dict[str, Any] = {}

    group_keys = raw.pop("group_keys", None) or raw.pop("group_by", None)
    if group_keys is not None:
        normalized["group_keys"] = list(group_keys) if isinstance(group_keys, (list, tuple)) else [group_keys]

    if "date_column" in raw:
        normalized["date_column"] = raw.pop("date_column")
    elif "column" in raw:
        normalized["date_column"] = raw.pop("column")

    for key in ("freq", "range_scope", "fill_values", "fill_strategies", "date_format", "sort"):
        if key in raw:
            normalized[key] = raw.pop(key)

    normalized.update(raw)
    return normalized


def _normalize_chunk_aggregator_params(params: dict[str, Any]) -> dict[str, Any]:
    raw = dict(params or {})
    normalized: dict[str, Any] = {}

    rules = raw.pop("rules", None)
    windows = raw.pop("windows", None) or raw.pop("chunks", None) or raw.pop("window_spec", None)
    value_columns = raw.pop("value_columns", None) or raw.pop("columns", None)

    if rules is None and windows is not None:
        windows_list = windows if isinstance(windows, list) else [windows]
        normalized["rules"] = [
            _clean_operation_dict(
                {
                    "size": window.get("size") if isinstance(window, dict) else None,
                    "agg": window.get("agg") if isinstance(window, dict) else None,
                    "columns": list(value_columns) if isinstance(value_columns, (list, tuple)) else ([value_columns] if value_columns is not None else None),
                },
                allowed_keys={"size", "agg", "columns"},
            )
            for window in windows_list
        ]
    elif rules is not None:
        rules_list = rules if isinstance(rules, list) else [rules]
        normalized["rules"] = [
            _clean_operation_dict(
                {
                    "size": rule.get("size") if isinstance(rule, dict) else None,
                    "agg": rule.get("agg") if isinstance(rule, dict) else None,
                    "columns": (
                        list(rule.get("columns"))
                        if isinstance(rule, dict) and isinstance(rule.get("columns"), (list, tuple))
                        else ([rule.get("columns")] if isinstance(rule, dict) and rule.get("columns") is not None else None)
                    ),
                    "columns_regex": rule.get("columns_regex") if isinstance(rule, dict) else None,
                    "columns_prefix": rule.get("columns_prefix") if isinstance(rule, dict) else None,
                    "columns_suffix": rule.get("columns_suffix") if isinstance(rule, dict) else None,
                    "exclude_columns": (
                        list(rule.get("exclude_columns"))
                        if isinstance(rule, dict) and isinstance(rule.get("exclude_columns"), (list, tuple))
                        else ([rule.get("exclude_columns")] if isinstance(rule, dict) and rule.get("exclude_columns") is not None else None)
                    ),
                },
                allowed_keys={"size", "agg", "columns", "columns_regex", "columns_prefix", "columns_suffix", "exclude_columns"},
            )
            for rule in rules_list
        ]

    if "from_end" in raw:
        normalized["from_end"] = raw.pop("from_end")
    if "drop_remainder" in raw:
        normalized["drop_remainder"] = raw.pop("drop_remainder")

    normalized.update(raw)
    return normalized


def _normalize_value_counts_reporter_params(params: dict[str, Any]) -> dict[str, Any]:
    raw = dict(params or {})
    normalized: dict[str, Any] = {}
    if "columns" in raw:
        columns = raw.pop("columns")
        if columns is not None:
            normalized["columns"] = list(columns) if isinstance(columns, (list, tuple)) else [columns]
    for key in ("dropna", "sort", "include_header"):
        if key in raw:
            normalized[key] = raw.pop(key)
    normalized.update(raw)
    return normalized


def _normalize_reduce_output_params(params: dict[str, Any]) -> dict[str, Any]:
    raw = dict(params or {})
    normalized: dict[str, Any] = {}
    for key in (
        "method",
        "column",
        "row",
        "label",
        "name",
        "position",
        "n",
        "agg",
        "anchor_extreme",
        "anchor_occurrence",
        "target_extreme",
        "target_occurrence",
        "direction",
        "skipna",
    ):
        if key in raw:
            normalized[key] = raw.pop(key)
    method = normalized.get("method")
    if isinstance(method, str) and method.strip().lower() in {"to_series", "as_series", "series"}:
        normalized["method"] = "column_to_series"
        normalized.setdefault("name", None)
    normalized.update(raw)
    return normalized


_NODE_NORMALIZERS = {
    "DataFrameInput": _normalize_dataframe_input_params,
    "Aggregator": _normalize_aggregator_params,
    "DateRangeExpander": _normalize_date_range_expander_params,
    "ChunkAggregator": _normalize_chunk_aggregator_params,
    "ValueCountsReporter": _normalize_value_counts_reporter_params,
    "ReduceOutput": _normalize_reduce_output_params,
    "ColumnTransformer": _normalize_column_transformer_params,
    "ValueTransformer": _normalize_value_transformer_params,
    "DatetimeTransformer": _normalize_datetime_transformer_params,
    "DataTransformer": _normalize_data_transformer_params,
}


def normalize_node_parameters(node_type: str, params: dict[str, Any]) -> dict[str, Any]:
    normalizer = _NODE_NORMALIZERS.get(node_type)
    if normalizer is None:
        return dict(params or {})
    return normalizer(params or {})


def _validate_operation_dict(node_type: str, operation_index: int, operation: Any) -> list[str]:
    contract = NODE_CONTRACTS.get(node_type, {})
    operation_types = contract.get("operation_types", {})
    errors: list[str] = []

    if not isinstance(operation, dict):
        return [
            f"INVALID_PARAM: {node_type}.operations[{operation_index}] must be an object with a 'type' field, not prose or code."
        ]

    op_type = operation.get("type")
    if not isinstance(op_type, str) or not op_type.strip():
        return [f"INVALID_PARAM: {node_type}.operations[{operation_index}] is missing a non-empty 'type' field."]

    if op_type not in operation_types:
        return [
            f"INVALID_PARAM: {node_type}.operations[{operation_index}].type='{op_type}' is not allowed."
        ]

    spec = operation_types[op_type]
    required = set(spec.get("required", []))
    optional = set(spec.get("optional", []))
    allowed = {"type"} | required | optional

    missing = [field for field in required if operation.get(field) in (None, [], {}, "")]
    if missing:
        errors.append(
            f"INVALID_PARAM: {node_type}.operations[{operation_index}] type='{op_type}' is missing required fields: {', '.join(missing)}."
        )

    extras = sorted(set(operation) - allowed)
    if extras:
        errors.append(
            f"INVALID_PARAM: {node_type}.operations[{operation_index}] type='{op_type}' has unsupported fields: {', '.join(extras)}."
        )

    def _validate_expression_field(field_name: str) -> None:
        value = operation.get(field_name)
        if not isinstance(value, str):
            return
        if ";" in value or "\n" in value:
            errors.append(
                f"INVALID_PARAM: {node_type}.operations[{operation_index}] field '{field_name}' must be a single expression, not multi-statement code."
            )
            return
        try:
            expression_tree = ast.parse(value, mode="eval")
        except SyntaxError:
            errors.append(
                f"INVALID_PARAM: {node_type}.operations[{operation_index}] field '{field_name}' must be a valid single Python expression."
            )
            return
        if any(isinstance(node, ast.NamedExpr) for node in ast.walk(expression_tree)):
            errors.append(
                f"INVALID_PARAM: {node_type}.operations[{operation_index}] field '{field_name}' must not contain assignment statements."
            )

    if op_type == "replace_infrequent":
        has_global = isinstance(operation.get("columns"), list) and operation.get("threshold") is not None and operation.get("replacement") is not None
        has_column_rules = isinstance(operation.get("column_rules"), list) and len(operation["column_rules"]) > 0
        if not has_global and not has_column_rules:
            errors.append(
                f"INVALID_PARAM: {node_type}.operations[{operation_index}] type='replace_infrequent' requires either columns+threshold+replacement or non-empty column_rules."
            )
        if has_column_rules:
            for rule_index, rule in enumerate(operation["column_rules"]):
                if not isinstance(rule, dict):
                    errors.append(
                        f"INVALID_PARAM: {node_type}.operations[{operation_index}].column_rules[{rule_index}] must be an object."
                    )
                    continue
                rule_allowed = {"column", "threshold", "replacement", "preserve_values"}
                missing_rule = [field for field in ("column", "threshold", "replacement") if rule.get(field) in (None, [], {}, "")]
                if missing_rule:
                    errors.append(
                        f"INVALID_PARAM: {node_type}.operations[{operation_index}].column_rules[{rule_index}] is missing required fields: {', '.join(missing_rule)}."
                    )
                extras_rule = sorted(set(rule) - rule_allowed)
                if extras_rule:
                    errors.append(
                        f"INVALID_PARAM: {node_type}.operations[{operation_index}].column_rules[{rule_index}] has unsupported fields: {', '.join(extras_rule)}."
                    )

    if op_type == "derive_function_columns":
        if operation.get("function") not in {
            "inverse",
            "exp",
            "exponential",
            "sigmoid",
            "normalize_sum",
            "log",
            "log1p",
            "square",
            "sqrt",
            "abs",
            "negate",
        }:
            errors.append(
                f"INVALID_PARAM: {node_type}.operations[{operation_index}] type='derive_function_columns' has unsupported function '{operation.get('function')}'."
            )

    if op_type == "derive_column":
        _validate_expression_field("expression")
        _validate_expression_field("rowwise_rule")

    if op_type == "parse_duration_text":
        if operation.get("result") not in (None, "number", "timedelta", "days"):
            errors.append(
                f"INVALID_PARAM: {node_type}.operations[{operation_index}] type='parse_duration_text' result must be one of number, timedelta, days."
            )

    return errors


def _validate_aggregator_params(params: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    group_keys = params.get("group_keys")
    aggregations = params.get("aggregations")

    if not isinstance(group_keys, list) or not group_keys:
        errors.append("INVALID_PARAM: Aggregator.group_keys must be a non-empty list.")
    else:
        for index, key in enumerate(group_keys):
            if isinstance(key, str):
                if not key.strip():
                    errors.append(f"INVALID_PARAM: Aggregator.group_keys[{index}] must not be empty.")
                continue
            if not isinstance(key, dict):
                errors.append(f"INVALID_PARAM: Aggregator.group_keys[{index}] must be a string or object.")
                continue

            key_type = key.get("type")
            if key_type not in {"column", "expression"}:
                errors.append(
                    f"INVALID_PARAM: Aggregator.group_keys[{index}].type must be 'column' or 'expression'."
                )
                continue
            if key_type == "column" and not key.get("column"):
                errors.append(f"INVALID_PARAM: Aggregator.group_keys[{index}] type='column' requires 'column'.")
            if key_type == "expression":
                if not key.get("name"):
                    errors.append(
                        f"INVALID_PARAM: Aggregator.group_keys[{index}] type='expression' requires 'name'."
                    )
                if not key.get("expression"):
                    errors.append(
                        f"INVALID_PARAM: Aggregator.group_keys[{index}] type='expression' requires 'expression'."
                    )

    allowed_ops = {
        "sum",
        "mean",
        "std",
        "min",
        "max",
        "count",
        "size",
        "first",
        "last",
        "nunique",
        "collect_list",
        "collect_set",
        "collect_rows",
    }
    if not isinstance(aggregations, list) or not aggregations:
        errors.append("INVALID_PARAM: Aggregator.aggregations must be a non-empty list.")
    else:
        for index, aggregation in enumerate(aggregations):
            if not isinstance(aggregation, dict):
                errors.append(f"INVALID_PARAM: Aggregator.aggregations[{index}] must be an object.")
                continue
            op = aggregation.get("op")
            if op not in allowed_ops:
                errors.append(
                    f"INVALID_PARAM: Aggregator.aggregations[{index}].op must be one of {', '.join(sorted(allowed_ops))}."
                )

            selector_present = any(
                aggregation.get(key) not in (None, [], {}, "")
                for key in ("column", "columns", "columns_regex", "columns_prefix", "columns_suffix")
            )
            if op in {"sum", "mean", "std", "min", "max", "count", "first", "last", "nunique", "collect_list", "collect_set"}:
                if not selector_present:
                    errors.append(
                        f"INVALID_PARAM: Aggregator.aggregations[{index}] op='{op}' requires a column selector."
                    )
            if op == "collect_rows":
                columns = aggregation.get("columns")
                if not isinstance(columns, list) or not columns:
                    errors.append(
                        f"INVALID_PARAM: Aggregator.aggregations[{index}] op='collect_rows' requires non-empty 'columns'."
                    )

    if "dropna" in params and not isinstance(params["dropna"], bool):
        errors.append("INVALID_PARAM: Aggregator.dropna must be a boolean.")
    if "reset_index" in params and not isinstance(params["reset_index"], bool):
        errors.append("INVALID_PARAM: Aggregator.reset_index must be a boolean.")

    return errors


def _validate_date_range_expander_params(params: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    group_keys = params.get("group_keys")
    if not isinstance(group_keys, list) or not group_keys:
        errors.append("INVALID_PARAM: DateRangeExpander.group_keys must be a non-empty list.")
    if not isinstance(params.get("date_column"), str) or not str(params.get("date_column")).strip():
        errors.append("INVALID_PARAM: DateRangeExpander.date_column must be a non-empty string.")
    if "range_scope" in params and params.get("range_scope") not in {"group", "global"}:
        errors.append("INVALID_PARAM: DateRangeExpander.range_scope must be 'group' or 'global'.")
    return errors


def _validate_chunk_aggregator_params(params: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    rules = params.get("rules")
    if not isinstance(rules, list) or not rules:
        errors.append("INVALID_PARAM: ChunkAggregator.rules must be a non-empty list.")
        return errors

    allowed_aggs = {"sum", "mean", "min", "max", "count", "first", "last"}
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            errors.append(f"INVALID_PARAM: ChunkAggregator.rules[{index}] must be an object.")
            continue
        if not isinstance(rule.get("size"), int) or rule["size"] <= 0:
            errors.append(f"INVALID_PARAM: ChunkAggregator.rules[{index}].size must be a positive integer.")
        if rule.get("agg") not in allowed_aggs:
            errors.append(
                f"INVALID_PARAM: ChunkAggregator.rules[{index}].agg must be one of {', '.join(sorted(allowed_aggs))}."
            )
        selector_present = any(
            rule.get(key) not in (None, [], {}, "")
            for key in ("columns", "columns_regex", "columns_prefix", "columns_suffix")
        )
        if not selector_present:
            errors.append(
                f"INVALID_PARAM: ChunkAggregator.rules[{index}] requires at least one column selector."
            )
    return errors


def _validate_reduce_output_params(params: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    method = params.get("method", "identity")
    if method == "columnwise_extreme_index":
        for field in ("anchor_extreme", "anchor_occurrence", "target_extreme", "target_occurrence", "direction"):
            if params.get(field) in (None, ""):
                errors.append(
                    f"INVALID_PARAM: ReduceOutput method='columnwise_extreme_index' requires '{field}'."
                )
        if params.get("anchor_extreme") not in {None, "min", "max"}:
            errors.append("INVALID_PARAM: ReduceOutput.anchor_extreme must be 'min' or 'max'.")
        if params.get("target_extreme") not in {None, "min", "max"}:
            errors.append("INVALID_PARAM: ReduceOutput.target_extreme must be 'min' or 'max'.")
        if params.get("anchor_occurrence") not in {None, "first", "last"}:
            errors.append("INVALID_PARAM: ReduceOutput.anchor_occurrence must be 'first' or 'last'.")
        if params.get("target_occurrence") not in {None, "first", "last"}:
            errors.append("INVALID_PARAM: ReduceOutput.target_occurrence must be 'first' or 'last'.")
        if params.get("direction") not in {None, "after", "before"}:
            errors.append("INVALID_PARAM: ReduceOutput.direction must be 'after' or 'before'.")
    if method == "column_to_series" and params.get("column") in (None, ""):
        errors.append("INVALID_PARAM: ReduceOutput method='column_to_series' requires 'column'.")
    return errors


def validate_canonical_node_parameters(node_type: str, params: dict[str, Any]) -> list[str]:
    contract = NODE_CONTRACTS.get(node_type)
    if not contract:
        return []

    errors: list[str] = []
    params = params or {}

    top_level_allowed = set(contract.get("top_level_params", {}))
    extras = sorted(set(params) - top_level_allowed)
    if extras:
        errors.append(
            f"INVALID_PARAM: {node_type} has unsupported top-level fields: {', '.join(extras)}."
        )

    if "operations" in top_level_allowed:
        operations = params.get("operations")
        if operations is None:
            errors.append(f"INVALID_PARAM: {node_type} requires an 'operations' list.")
        elif not isinstance(operations, list):
            errors.append(f"INVALID_PARAM: {node_type}.operations must be a list.")
        else:
            for index, operation in enumerate(operations):
                errors.extend(_validate_operation_dict(node_type, index, operation))
    else:
        if "source_name" in params and not isinstance(params["source_name"], str):
            errors.append("INVALID_PARAM: DataFrameInput.source_name must be a string.")
        if "copy" in params and not isinstance(params["copy"], bool):
            errors.append("INVALID_PARAM: DataFrameInput.copy must be a boolean.")

    if node_type == "Aggregator":
        errors.extend(_validate_aggregator_params(params))
    if node_type == "DateRangeExpander":
        errors.extend(_validate_date_range_expander_params(params))
    if node_type == "ChunkAggregator":
        errors.extend(_validate_chunk_aggregator_params(params))
    if node_type == "ReduceOutput":
        errors.extend(_validate_reduce_output_params(params))

    return errors
