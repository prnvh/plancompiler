import unittest

from core.plan_utils import normalize_plan_shape
from core.validator import validate_plan


class PlanContractTests(unittest.TestCase):
    def test_validator_rejects_non_object_plan_without_crashing(self):
        is_valid, errors = validate_plan(None)

        self.assertFalse(is_valid)
        self.assertIn("INVALID_PLAN: plan must be a JSON object.", errors)

    def test_normalize_plan_shape_canonicalizes_known_aliases(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "ValueTransformer"},
            ],
            "edges": [["n1", "n2"]],
            "parameters": {
                "n1": {"dataframe_name": "input_df"},
                "n2": {
                    "operation": "replace_low_frequency",
                    "columns": ["product"],
                    "min_count": 2,
                    "other_label": "Other",
                },
            },
        }

        normalized = normalize_plan_shape(plan)

        self.assertEqual(normalized["parameters"]["n1"], {"source_name": "input_df"})
        self.assertEqual(
            normalized["parameters"]["n2"],
            {
                "operations": [
                    {
                        "type": "replace_infrequent",
                        "columns": ["product"],
                        "threshold": 2,
                        "replacement": "Other",
                    }
                ]
            },
        )

    def test_validator_rejects_prose_transformer_operations(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "ColumnTransformer"},
            ],
            "edges": [["n1", "n2"]],
            "parameters": {
                "n1": {},
                "n2": {
                    "operations": [
                        "Create a new category column from A, B, C, D",
                    ]
                },
            },
        }

        is_valid, errors = validate_plan(plan)

        self.assertFalse(is_valid)
        self.assertTrue(
            any("must be an object with a 'type' field" in error for error in errors),
            msg=errors,
        )

    def test_validator_rejects_unknown_top_level_fields_for_canonical_nodes(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "DatetimeTransformer"},
            ],
            "edges": [["n1", "n2"]],
            "parameters": {
                "n2": {
                    "script": "df['datetime'] = pd.to_datetime(df['datetime'])",
                },
            },
        }

        is_valid, errors = validate_plan(plan)

        self.assertFalse(is_valid)
        self.assertTrue(
            any("unsupported top-level fields" in error for error in errors),
            msg=errors,
        )

    def test_validator_rejects_multi_statement_expression_code(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "DataTransformer"},
            ],
            "edges": [["n1", "n2"]],
            "parameters": {
                "n2": {
                    "operations": [
                        {
                            "type": "derive_column",
                            "new_column": "result",
                            "expression": "df = df.sort_values('a'); df",
                        }
                    ]
                }
            },
        }

        is_valid, errors = validate_plan(plan)

        self.assertFalse(is_valid)
        self.assertTrue(
            any("single expression" in error or "assignment statements" in error for error in errors),
            msg=errors,
        )

    def test_validator_allows_keyword_arguments_inside_expression(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "ColumnTransformer"},
            ],
            "edges": [["n1", "n2"]],
            "parameters": {
                "n2": {
                    "operations": [
                        {
                            "type": "derive_column",
                            "new_column": "rank",
                            "expression": "df.groupby('ID')['TIME'].rank(ascending=True)",
                        }
                    ]
                }
            },
        }

        is_valid, errors = validate_plan(plan)

        self.assertTrue(is_valid, msg=errors)

    def test_normalize_plan_shape_maps_generic_category_creation_aliases(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "ColumnTransformer"},
            ],
            "edges": [["n1", "n2"]],
            "parameters": {
                "n2": {
                    "operations": [
                        {
                            "type": "create_column",
                            "column": "category",
                            "method": "first_match",
                            "matches": [
                                {"columns": ["A"], "value": 0, "label": "A"},
                                {"columns": ["B"], "value": 0, "label": "B"},
                            ],
                            "default": None,
                        }
                    ]
                },
            },
        }

        normalized = normalize_plan_shape(plan)

        self.assertEqual(
            normalized["parameters"]["n2"]["operations"][0],
            {
                "type": "derive_first_matching_label",
                "new_column": "category",
                "source_columns": ["A", "B"],
                "match_value": 0,
                "labels": {"A": "A", "B": "B"},
            },
        )

    def test_normalize_plan_shape_maps_supported_function_aliases(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "ColumnTransformer"},
            ],
            "edges": [["n1", "n2"]],
            "parameters": {
                "n2": {
                    "operations": [
                        {
                            "type": "derive_function_columns",
                            "source_columns": ["A"],
                            "function": "np.exp",
                        },
                        {
                            "type": "derive_function_columns",
                            "source_columns": ["B"],
                            "function": "1/(1+np.exp(-col))",
                        },
                        {
                            "type": "derive_function_columns",
                            "source_columns": ["C"],
                            "function": "lambda col: col / col.sum()",
                        },
                        {
                            "type": "derive_function_columns",
                            "source_columns": ["D"],
                            "function": "lambda x: 1/(1+np.exp(-x))",
                        },
                    ]
                }
            },
        }

        normalized = normalize_plan_shape(plan)

        self.assertEqual(
            normalized["parameters"]["n2"]["operations"],
            [
                {"type": "derive_function_columns", "source_columns": ["A"], "function": "exp"},
                {"type": "derive_function_columns", "source_columns": ["B"], "function": "sigmoid"},
                {"type": "derive_function_columns", "source_columns": ["C"], "function": "normalize_sum"},
                {"type": "derive_function_columns", "source_columns": ["D"], "function": "sigmoid"},
            ],
        )

    def test_normalize_plan_shape_maps_reduce_output_series_aliases(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "ReduceOutput"},
            ],
            "edges": [["n1", "n2"]],
            "parameters": {
                "n2": {"method": "to_series", "column": "value", "label": "date"}
            },
        }

        normalized = normalize_plan_shape(plan)

        self.assertEqual(normalized["parameters"]["n2"]["method"], "column_to_series")
        self.assertIsNone(normalized["parameters"]["n2"]["name"])

    def test_normalize_plan_shape_maps_shift_lambda_aliases(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "ColumnTransformer"},
            ],
            "edges": [["n1", "n2"]],
            "parameters": {
                "n2": {
                    "operations": [
                        {
                            "type": "derive_function_columns",
                            "source_columns": ["0", "1", "2"],
                            "function": "lambda row: dict(zip(row.index, list(row.dropna().values) + [np.nan] * (len(row) - row.count())))",
                        },
                        {
                            "type": "derive_function_columns",
                            "source_columns": ["0", "1", "2"],
                            "function": "lambda col: pd.Series(list(col[col.isnull()]) + list(col[col.notnull()]))",
                        },
                    ]
                }
            },
        }

        normalized = normalize_plan_shape(plan)

        self.assertEqual(
            normalized["parameters"]["n2"]["operations"],
            [
                {"type": "shift_non_nulls_left", "columns": ["0", "1", "2"]},
                {"type": "shift_nulls_to_top_per_column", "columns": ["0", "1", "2"]},
            ],
        )

    def test_normalize_plan_shape_maps_reverse_dummy_aliases(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "ColumnTransformer"},
            ],
            "edges": [["n1", "n2"]],
            "parameters": {
                "n2": {
                    "operations": [
                        {
                            "type": "create_category_from_binaries",
                            "columns": ["A", "B", "C"],
                            "new_column": "category",
                            "present_value": 0,
                        },
                        {
                            "type": "one_hot_to_category_list",
                            "columns": ["A", "B", "C"],
                            "new_column": "all_categories",
                            "match_value": 1,
                        },
                    ]
                },
            },
        }

        normalized = normalize_plan_shape(plan)

        self.assertEqual(
            normalized["parameters"]["n2"]["operations"],
            [
                {
                    "type": "derive_first_matching_label",
                    "new_column": "category",
                    "source_columns": ["A", "B", "C"],
                    "match_value": 0,
                },
                {
                    "type": "derive_matching_labels",
                    "new_column": "all_categories",
                    "source_columns": ["A", "B", "C"],
                    "match_value": 1,
                },
            ],
        )

    def test_normalize_plan_shape_maps_legacy_aggregator_params(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "Aggregator"},
            ],
            "edges": [["n1", "n2"]],
            "parameters": {
                "n2": {
                    "group_by": "region",
                    "agg_func": {"sales": ["sum", "mean"]},
                },
            },
        }

        normalized = normalize_plan_shape(plan)

        self.assertEqual(normalized["parameters"]["n2"]["group_keys"], ["region"])
        self.assertEqual(
            normalized["parameters"]["n2"]["aggregations"],
            [
                {"column": "sales", "op": "sum", "output": "sales_sum"},
                {"column": "sales", "op": "mean", "output": "sales_mean"},
            ],
        )

    def test_validator_rejects_invalid_aggregator_specs(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "Aggregator"},
            ],
            "edges": [["n1", "n2"]],
            "parameters": {
                "n2": {
                    "group_keys": [{"type": "expression", "name": "bucket"}],
                    "aggregations": [{"op": "sum"}],
                },
            },
        }

        is_valid, errors = validate_plan(plan)

        self.assertFalse(is_valid)
        self.assertTrue(
            any("Aggregator.group_keys[0]" in error for error in errors),
            msg=errors,
        )
        self.assertTrue(
            any("Aggregator.aggregations[0]" in error for error in errors),
            msg=errors,
        )

    def test_normalize_plan_shape_maps_chunk_aggregator_legacy_fields(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "ChunkAggregator"},
            ],
            "edges": [["n1", "n2"]],
            "parameters": {
                "n2": {
                    "windows": [{"size": 3, "agg": "mean"}],
                    "value_columns": ["col1"],
                    "from_end": True,
                },
            },
        }

        normalized = normalize_plan_shape(plan)

        self.assertEqual(
            normalized["parameters"]["n2"],
            {
                "rules": [{"size": 3, "agg": "mean", "columns": ["col1"]}],
                "from_end": True,
            },
        )

    def test_validator_rejects_invalid_chunk_aggregator_rule(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "ChunkAggregator"},
            ],
            "edges": [["n1", "n2"]],
            "parameters": {
                "n2": {
                    "rules": [{"size": 0, "agg": "median"}],
                },
            },
        }

        is_valid, errors = validate_plan(plan)

        self.assertFalse(is_valid)
        self.assertTrue(any("ChunkAggregator.rules[0].size" in error for error in errors), msg=errors)
        self.assertTrue(any("ChunkAggregator.rules[0].agg" in error for error in errors), msg=errors)


if __name__ == "__main__":
    unittest.main()
