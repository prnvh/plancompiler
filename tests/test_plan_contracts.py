import unittest

from core.plan_utils import normalize_plan_shape
from core.validator import validate_plan


class PlanContractTests(unittest.TestCase):
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
