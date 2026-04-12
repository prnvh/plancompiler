import unittest

import pandas as pd

from core.planner import (
    PlanningContext,
    SYSTEM_PROMPT,
    build_node_summary,
    build_planning_context_summary,
)
from nodes.registry import NODE_REGISTRY
from nodes.templates.aggregator import aggregator
from nodes.templates.chunk_aggregator import chunk_aggregator
from nodes.templates.column_transformer import column_transformer
from nodes.templates.data_filter import data_filter
from nodes.templates.data_transformer import data_transformer
from nodes.templates.date_range_expander import date_range_expander
from nodes.templates.datetime_transformer import datetime_transformer
from nodes.templates.reduce_output import reduce_output
from nodes.templates.value_transformer import value_transformer
from nodes.templates.value_counts_reporter import value_counts_reporter


class SemanticTransformerTests(unittest.TestCase):
    def test_registry_contains_new_transformer_nodes(self):
        for node_name in ("ColumnTransformer", "ValueTransformer", "DatetimeTransformer"):
            self.assertIn(node_name, NODE_REGISTRY)
        for node_name in ("DateRangeExpander", "ChunkAggregator", "ValueCountsReporter"):
            self.assertIn(node_name, NODE_REGISTRY)

    def test_planner_summary_surfaces_contract_examples(self):
        summary = build_node_summary()
        self.assertIn("ColumnTransformer", summary)
        self.assertIn("ValueTransformer", summary)
        self.assertIn("DatetimeTransformer", summary)
        self.assertIn("Respect the planning context block", SYSTEM_PROMPT)
        self.assertIn("canonical params:", summary)
        self.assertIn("examples:", summary)
        self.assertIn("replace_infrequent", summary)
        self.assertTrue(
            "derive_first_matching_label" in summary or "derive_matching_labels" in summary
        )
        self.assertIn("extract_regex", summary)
        self.assertIn("expression environment:", summary)

    def test_build_planning_context_summary_is_generic(self):
        context = PlanningContext(
            allowed_nodes=["DataFrameInput", "ColumnTransformer", "ReduceOutput"],
            source_type="in_memory_dataframe",
            desired_output_kind="dataframe",
            workflow_mode="in_memory",
            source_binding_names=["input_df"],
            notes=["Avoid file export nodes for this workflow."],
        )

        summary = build_planning_context_summary(context)

        self.assertIn("Allowed nodes: DataFrameInput, ColumnTransformer, ReduceOutput", summary)
        self.assertIn("Source type: in_memory_dataframe", summary)
        self.assertIn("Desired output kind: dataframe", summary)
        self.assertIn("Workflow mode: in_memory", summary)
        self.assertIn("Runtime dataframe bindings: input_df", summary)

        filtered_summary = build_node_summary(context)
        self.assertIn("DataFrameInput", filtered_summary)
        self.assertIn("ColumnTransformer", filtered_summary)
        self.assertNotIn("CSVParser", filtered_summary)

    def test_column_transformer_can_derive_matching_labels_column(self):
        frame = pd.DataFrame(
            {
                "A": [1, 0, 0],
                "B": [0, 1, 0],
                "C": [1, 1, 0],
            }
        )

        transformed = column_transformer(
            frame,
            operations=[
                {
                    "type": "derive_matching_labels",
                    "new_column": "category",
                    "source_columns": ["A", "B", "C"],
                    "match_value": 1,
                }
            ],
        )

        self.assertEqual(
            transformed["category"].tolist(),
            [["A", "C"], ["B", "C"], []],
        )

    def test_column_transformer_can_derive_matching_labels_from_value_and_source_columns(self):
        frame = pd.DataFrame(
            {
                "A": [1, 0, 1],
                "B": [0, 1, 1],
            }
        )

        transformed = column_transformer(
            frame,
            operations=[
                {
                    "type": "derive_column",
                    "new_column": "labels",
                    "source_columns": ["A", "B"],
                    "value": 1,
                }
            ],
        )

        self.assertEqual(transformed["labels"].tolist(), [["A"], ["B"], ["A", "B"]])

    def test_column_transformer_can_derive_first_matching_label(self):
        frame = pd.DataFrame(
            {
                "email_flag": [0, 1, 0],
                "sms_flag": [1, 0, 0],
                "phone_flag": [0, 0, 1],
            }
        )

        transformed = column_transformer(
            frame,
            operations=[
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
            ],
        )

        self.assertEqual(
            transformed["primary_channel"].tolist(),
            ["sms", "email", "phone"],
        )

    def test_column_transformer_can_evaluate_if_then_else_expression_rowwise(self):
        frame = pd.DataFrame(
            {
                "score": [90, 72, 55],
            }
        )

        transformed = column_transformer(
            frame,
            operations=[
                {
                    "type": "derive_column",
                    "new_column": "band",
                    "expression": "if score >= 80 then 'high' else 'standard'",
                }
            ],
        )

        self.assertEqual(transformed["band"].tolist(), ["high", "standard", "standard"])

    def test_column_transformer_can_use_direct_column_expression_and_where_helper(self):
        frame = pd.DataFrame(
            {
                "A": [1, -2, 3],
                "B": [10, 20, 30],
            }
        )

        transformed = column_transformer(
            frame,
            operations=[
                {"type": "derive_column", "new_column": "sum_col", "expression": "A + B"},
                {"type": "derive_column", "new_column": "chosen", "expression": "where(A > 0, A, B)"},
            ],
        )

        self.assertEqual(transformed["sum_col"].tolist(), [11, 18, 33])
        self.assertEqual(transformed["chosen"].tolist(), [1, 20, 3])

    def test_column_transformer_can_extract_regex_split_and_map_to_new_columns(self):
        frame = pd.DataFrame(
            {
                "duration": ["12 week", "3 day"],
            }
        )

        transformed = column_transformer(
            frame,
            operations=[
                {
                    "type": "extract_regex",
                    "source_column": "duration",
                    "pattern": "^(\\d+)",
                    "new_column": "number",
                },
                {
                    "type": "split_column",
                    "source_column": "duration",
                    "new_columns": ["amount", "unit"],
                    "separator": " ",
                },
                {
                    "type": "map_values",
                    "source_column": "unit",
                    "new_column": "unit_days",
                    "mapping": {"week": 7, "day": 1},
                },
            ],
        )

        self.assertEqual(transformed["number"].tolist(), ["12", "3"])
        self.assertEqual(transformed["amount"].tolist(), ["12", "3"])
        self.assertEqual(transformed["unit"].tolist(), ["week", "day"])
        self.assertEqual(transformed["unit_days"].tolist(), [7, 1])

    def test_column_transformer_can_melt_and_derive_inverse_columns(self):
        frame = pd.DataFrame(
            {
                "id": [1, 2],
                "A": [2.0, 4.0],
                "B": [5.0, 10.0],
            }
        )

        transformed = column_transformer(
            frame,
            operations=[
                {
                    "type": "derive_inverse_columns",
                    "source_columns": ["A", "B"],
                    "prefix": "inv_",
                },
                {
                    "type": "melt",
                    "id_vars": ["id"],
                    "value_vars": ["inv_A", "inv_B"],
                    "var_name": "feature",
                    "value_name": "value",
                },
            ],
        )

        self.assertEqual(list(transformed.columns), ["id", "feature", "value"])
        self.assertEqual(len(transformed), 4)
        self.assertAlmostEqual(float(transformed.iloc[0]["value"]), 0.5)

    def test_column_transformer_preserves_row_order_when_melting(self):
        frame = pd.DataFrame(
            {
                "id": [1, 2],
                "A": [10, None],
                "B": [100, 200],
            }
        )

        transformed = column_transformer(
            frame,
            operations=[
                {
                    "type": "melt",
                    "id_vars": ["id"],
                    "value_vars": ["A", "B"],
                    "var_name": "feature",
                    "value_name": "value",
                    "preserve_row_order": True,
                    "dropna": True,
                }
            ],
        )

        self.assertEqual(
            transformed.to_dict(orient="records"),
            [
                {"id": 1, "feature": "A", "value": 10.0},
                {"id": 1, "feature": "B", "value": 100.0},
                {"id": 2, "feature": "B", "value": 200.0},
            ],
        )

    def test_column_transformer_can_derive_general_function_columns(self):
        frame = pd.DataFrame({"A": [0.0, 1.0], "B": [2.0, 3.0]})

        transformed = column_transformer(
            frame,
            operations=[
                {
                    "type": "derive_function_columns",
                    "source_columns": ["A", "B"],
                    "function": "sigmoid",
                }
            ],
        )

        self.assertEqual(list(transformed.columns), ["A", "B", "sigmoid_A", "sigmoid_B"])
        self.assertAlmostEqual(float(transformed.loc[0, "sigmoid_A"]), 0.5)

    def test_column_transformer_can_normalize_column_lists_and_mapping_expressions(self):
        frame = pd.DataFrame(
            {
                "A_Name": ["x", "y"],
                "B_Detail": ["u", "v"],
                "Value_left": [1, 2],
                "Value_right": [3, 4],
            }
        )

        transformed = column_transformer(
            frame,
            operations=[
                {
                    "type": "select_columns",
                    "columns": "['A_Name', 'B_Detail'] + df.filter(regex='^Value').columns.tolist()",
                },
                {
                    "type": "rename_columns",
                    "mapping": "{c: c.replace('Value_', '') for c in df.filter(regex='^Value').columns}",
                },
            ],
        )

        self.assertEqual(list(transformed.columns), ["A_Name", "B_Detail", "left", "right"])

    def test_column_transformer_can_pivot_wider_and_split_rows(self):
        long_frame = pd.DataFrame(
            {
                "country": ["US", "US", "IN", "IN"],
                "year": [2024, 2024, 2024, 2024],
                "metric": ["sales", "cost", "sales", "cost"],
                "value": [10, 7, 12, 8],
            }
        )
        list_frame = pd.DataFrame({"id": [1, 2], "tags": ["a,b", "c"]})

        wide = column_transformer(
            long_frame,
            operations=[
                {
                    "type": "pivot_wider",
                    "index": ["country", "year"],
                    "columns": "metric",
                    "values": "value",
                }
            ],
        )
        exploded = column_transformer(
            list_frame,
            operations=[
                {"type": "split_rows", "source_column": "tags", "separator": ",", "ignore_index": True}
            ],
        )

        self.assertEqual(list(wide.columns), ["country", "year", "sales", "cost"])
        self.assertEqual(wide.columns.name, "metric")
        self.assertEqual(exploded["tags"].tolist(), ["a", "b", "c"])

    def test_data_transformer_can_derive_row_lists_from_column_expression(self):
        frame = pd.DataFrame({"time": [1, 2], "amount": [10, 20]})

        transformed = data_transformer(
            frame,
            operations=[
                {"type": "derive_column", "new_column": "pair", "expression": "[time, amount]"}
            ],
        )

        self.assertEqual(transformed["pair"].tolist(), [[1, 10], [2, 20]])

    def test_transformers_can_compute_row_value_counts(self):
        frame = pd.DataFrame(
            {
                "a": ["x", "x", None],
                "b": ["x", "y", None],
                "c": ["z", "y", "q"],
            }
        )

        transformed = column_transformer(
            frame,
            operations=[
                {
                    "type": "row_value_counts",
                    "source_columns": ["a", "b", "c"],
                    "values_column": "frequent",
                    "count_column": "freq_count",
                }
            ],
        )

        self.assertEqual(transformed["frequent"].tolist(), [["x"], ["y"], ["q"]])
        self.assertEqual(transformed["freq_count"].tolist(), [2, 2, 1])

    def test_data_transformer_can_compute_groupwise_extreme_neighbors(self):
        frame = pd.DataFrame(
            {
                "time": [1, 1, 1, 2],
                "car": [10, 20, 30, 40],
                "x": [0.0, 3.0, 10.0, 5.0],
                "y": [0.0, 4.0, 0.0, 5.0],
            }
        )

        nearest = data_transformer(
            frame,
            operations=[
                {
                    "type": "groupwise_extreme_neighbor",
                    "group_by": ["time"],
                    "entity_column": "car",
                    "coordinate_columns": ["x", "y"],
                    "neighbor_column": "nearest_car",
                    "distance_column": "nearest_distance",
                }
            ],
        )
        farthest = data_transformer(
            frame,
            operations=[
                {
                    "type": "groupwise_extreme_neighbor",
                    "group_by": ["time"],
                    "entity_column": "car",
                    "coordinate_columns": ["x", "y"],
                    "neighbor_column": "farthest_car",
                    "distance_column": "farthest_distance",
                    "extreme": "max",
                }
            ],
        )

        self.assertEqual(nearest["nearest_car"].iloc[:3].tolist(), [20, 10, 20])
        self.assertAlmostEqual(float(nearest["nearest_distance"].iloc[0]), 5.0)
        self.assertEqual(farthest["farthest_car"].iloc[:3].tolist(), [30, 30, 10])
        self.assertTrue(pd.isna(nearest["nearest_car"].iloc[3]))

    def test_column_transformer_can_evaluate_rowwise_rule_without_lambda_wrapper(self):
        frame = pd.DataFrame({"text": ["a1", "abc", "z9!"]})

        transformed = column_transformer(
            frame,
            operations=[
                {
                    "type": "derive_column",
                    "new_column": "letters",
                    "rowwise_rule": "sum(1 for ch in row['text'] if ch.isalpha())",
                }
            ],
        )

        self.assertEqual(transformed["letters"].tolist(), [1, 3, 1])

    def test_column_transformer_can_shift_null_patterns(self):
        frame = pd.DataFrame(
            {
                "A": [None, 1.0],
                "B": [2.0, None],
                "C": [3.0, 4.0],
            }
        )

        left_shifted = column_transformer(
            frame,
            operations=[{"type": "shift_non_nulls_left", "columns": ["A", "B", "C"]}],
        )
        nulls_to_top = column_transformer(
            frame,
            operations=[{"type": "shift_nulls_to_top_per_column", "columns": ["A", "B"]}],
        )

        self.assertEqual(left_shifted.iloc[0].iloc[:2].tolist(), [2.0, 3.0])
        self.assertTrue(pd.isna(left_shifted.iloc[0].iloc[2]))
        self.assertTrue(pd.isna(nulls_to_top.iloc[0]["A"]))
        self.assertTrue(pd.isna(nulls_to_top.iloc[0]["B"]))

    def test_aggregator_supports_std_and_expression_selected_columns(self):
        frame = pd.DataFrame(
            {
                "group": ["a", "a", "b", "b"],
                "value": [1.0, 3.0, 2.0, 6.0],
                "keep": [10.0, 20.0, 30.0, 40.0],
            }
        )

        transformed = aggregator(
            frame,
            group_keys=["group"],
            aggregations=[
                {"column": "value", "op": "std", "output": "value_std"},
                {
                    "column": {"type": "expression", "expression": "[c for c in df.columns if c.startswith('ke')]"},
                    "op": "sum",
                    "output": "keep",
                },
            ],
        )

        self.assertEqual(list(transformed.columns), ["group", "value_std", "keep"])
        self.assertAlmostEqual(float(transformed.loc[0, "keep"]), 30.0)

    def test_aggregator_can_materialize_series_expressions(self):
        frame = pd.DataFrame(
            {
                "group": ["a", "a", "b"],
                "text": ["one", "two", "three"],
            }
        )

        transformed = aggregator(
            frame,
            group_keys=["group"],
            aggregations=[
                {"column": "col('text').str.contains('o')", "op": "sum", "output": "contains_o"}
            ],
        )

        self.assertEqual(transformed["contains_o"].tolist(), [2, 0])

    def test_reduce_output_can_convert_dataframe_column_to_series(self):
        frame = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "value": [5, 6],
            }
        )

        result = reduce_output(frame, method="column_to_series", column="value", label="date")

        self.assertEqual(result.index.tolist(), ["2024-01-01", "2024-01-02"])
        self.assertEqual(result.tolist(), [5, 6])

    def test_reduce_output_can_control_series_name(self):
        frame = pd.DataFrame({"date": ["2024-01-01"], "value": [5]})

        result = reduce_output(frame, method="column_to_series", column="value", label="date", name=None)

        self.assertIsNone(result.name)

    def test_reduce_output_can_join_column_values(self):
        frame = pd.DataFrame({"text": ["a", "b", None, "c"]})

        result = reduce_output(frame, method="scalar_agg", column="text", agg="join", separator=",")

        self.assertEqual(result, "a,b,c")

    def test_data_filter_supports_datetime_index_normalize_conditions(self):
        frame = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.to_datetime(
                ["2020-02-17 10:00:00", "2020-02-18 09:00:00", "2020-02-19 08:00:00"]
            ),
        )

        filtered = data_filter(
            frame,
            "~index.normalize().isin([pd.to_datetime('2020-02-17'), pd.to_datetime('2020-02-18')])",
        )

        self.assertEqual(filtered["value"].tolist(), [3])

    def test_column_transformer_can_concatenate_and_explode_columns(self):
        frame = pd.DataFrame(
            {
                "first": ["A", "B"],
                "second": ["1", "2"],
                "items": [["x", "y"], ["z"]],
            }
        )

        transformed = column_transformer(
            frame,
            operations=[
                {
                    "type": "concatenate_columns",
                    "source_columns": ["first", "second"],
                    "new_column": "joined",
                    "separator": "-",
                },
                {
                    "type": "explode_column",
                    "source_column": "items",
                    "ignore_index": True,
                },
            ],
        )

        self.assertEqual(transformed["joined"].tolist(), ["A-1", "A-1", "B-2"])
        self.assertEqual(transformed["items"].tolist(), ["x", "y", "z"])

    def test_value_transformer_can_replace_low_frequency_values(self):
        frame = pd.DataFrame(
            {
                "product": ["apple", "potato", "cheese", "banana", "cheese", "banana", "cheese", "potato", "egg"],
                "channel": ["retail", "web", "store", "store", "store", "retail", "web", "web", "web"],
                "region": ["east", "west", "north", "south", "south", "west", "south", "west", "central"],
            }
        )

        transformed = value_transformer(
            frame,
            operations=[
                {
                    "type": "replace_infrequent",
                    "columns": ["product", "channel", "region"],
                    "threshold": 3,
                    "replacement": "Other",
                }
            ],
        )

        self.assertEqual(
            transformed["product"].tolist(),
            ["Other", "Other", "cheese", "Other", "cheese", "Other", "cheese", "Other", "Other"],
        )

    def test_value_transformer_can_preserve_reserved_values(self):
        frame = pd.DataFrame(
            {
                "priority": ["VIP", "low", "normal", "low", "normal", "normal"],
            }
        )

        transformed = value_transformer(
            frame,
            operations=[
                {
                    "type": "replace_infrequent",
                    "columns": ["priority"],
                    "threshold": 2,
                    "replacement": "Other",
                    "preserve_values": ["VIP"],
                }
            ],
        )

        self.assertEqual(
            transformed["priority"].tolist(),
            ["VIP", "low", "normal", "low", "normal", "normal"],
        )

    def test_value_transformer_can_share_reserved_values_across_column_rules(self):
        frame = pd.DataFrame(
            {
                "priority": ["VIP", "low", "normal", "low"],
                "tier": ["VIP", "silver", "gold", "silver"],
            }
        )

        transformed = value_transformer(
            frame,
            operations=[
                {
                    "type": "replace_infrequent",
                    "column_rules": [
                        {"column": "priority", "threshold": 2, "replacement": "Other"},
                        {"column": "tier", "threshold": 2, "replacement": "Other"},
                    ],
                    "preserve_values": ["VIP"],
                }
            ],
        )

        self.assertEqual(transformed["priority"].tolist(), ["VIP", "low", "Other", "low"])
        self.assertEqual(transformed["tier"].tolist(), ["VIP", "silver", "Other", "silver"])

    def test_value_transformer_accepts_column_rule_mapping(self):
        frame = pd.DataFrame(
            {
                "Qu1": ["A", "A", "B", "C"],
                "Qu2": ["X", "Y", "Y", "Z"],
            }
        )

        transformed = value_transformer(
            frame,
            operations=[
                {
                    "type": "replace_infrequent",
                    "column_rules": {
                        "Qu1": {"threshold": 2, "replacement": "other"},
                        "Qu2": {"threshold": 2, "replacement": "other"},
                    },
                }
            ],
        )

        self.assertEqual(transformed["Qu1"].tolist(), ["A", "A", "other", "other"])
        self.assertEqual(transformed["Qu2"].tolist(), ["other", "Y", "Y", "other"])

    def test_value_transformer_can_replace_substrings_strip_prefix_and_round(self):
        frame = pd.DataFrame(
            {
                "sku": ["item-100", "item-250"],
                "amount": [1.234, 9.876],
            }
        )

        transformed = value_transformer(
            frame,
            operations=[
                {"type": "replace_substring", "column": "sku", "old": "-", "new": "_"},
                {"type": "strip_prefix", "column": "sku", "prefix": "item_"},
                {"type": "round_values", "column": "amount", "decimals": 1, "new_column": "rounded_amount"},
            ],
        )

        self.assertEqual(transformed["sku"].tolist(), ["100", "250"])
        self.assertEqual(transformed["rounded_amount"].tolist(), [1.2, 9.9])

    def test_value_transformer_can_replace_substring_across_all_columns(self):
        frame = pd.DataFrame(
            {
                "left": ["&LT;a", "&LT;b"],
                "right": ["x&LT;", "y&LT;"],
                "count": [1, 2],
            }
        )

        transformed = value_transformer(
            frame,
            operations=[
                {"type": "replace_substring", "column": "*", "old": "&LT;", "new": "<"},
            ],
        )

        self.assertEqual(transformed["left"].tolist(), ["<a", "<b"])
        self.assertEqual(transformed["right"].tolist(), ["x<", "y<"])
        self.assertEqual(transformed["count"].tolist(), [1, 2])

    def test_value_transformer_can_remove_substrings(self):
        frame = pd.DataFrame({"text": ["[a]", "[b]"]})

        transformed = value_transformer(
            frame,
            operations=[
                {"type": "remove_substring", "column": "text", "old": "["},
                {"type": "remove_substring", "column": "text", "old": "]"},
            ],
        )

        self.assertEqual(transformed["text"].tolist(), ["a", "b"])

    def test_value_transformer_can_coerce_numeric_and_clip(self):
        frame = pd.DataFrame({"raw": ["1", "20", "bad"]})

        transformed = value_transformer(
            frame,
            operations=[
                {"type": "coerce_numeric", "column": "raw", "new_column": "value"},
                {"type": "clip_values", "column": "value", "lower": 0, "upper": 10},
            ],
        )

        self.assertEqual(transformed["value"].iloc[0], 1.0)
        self.assertEqual(transformed["value"].iloc[1], 10.0)
        self.assertTrue(pd.isna(transformed["value"].iloc[2]))

    def test_value_transformer_can_factorize_and_parse_duration_text(self):
        frame = pd.DataFrame(
            {
                "category": ["b", "a", "b"],
                "duration": ["2 week", "3 day", "1 month"],
            }
        )

        transformed = value_transformer(
            frame,
            operations=[
                {"type": "factorize_values", "column": "category", "start": 1, "sort": True, "new_column": "category_id"},
                {
                    "type": "parse_duration_text",
                    "column": "duration",
                    "new_column": "duration_days",
                    "result": "days",
                    "unit_map": {"day": 1, "week": 7, "month": 30},
                },
            ],
        )

        self.assertEqual(transformed["category_id"].tolist(), [2, 1, 2])
        self.assertEqual(transformed["duration_days"].tolist(), [14.0, 3.0, 30.0])

    def test_datetime_transformer_can_parse_remove_timezone_and_format(self):
        frame = pd.DataFrame(
            {
                "datetime": [
                    "2015-12-01 00:00:00-06:00",
                    "2015-12-02 00:01:00-06:00",
                ]
            }
        )

        transformed = datetime_transformer(
            frame,
            operations=[
                {"type": "parse_datetime", "column": "datetime"},
                {"type": "remove_timezone", "column": "datetime"},
                {"type": "format_datetime", "column": "datetime", "format": "%d-%b-%Y %H:%M:%S"},
            ],
        )

        self.assertEqual(
            transformed["datetime"].tolist(),
            ["01-Dec-2015 00:00:00", "02-Dec-2015 00:01:00"],
        )

    def test_datetime_transformer_can_extract_timestamp_part(self):
        frame = pd.DataFrame(
            {
                "datetime": [
                    "2015-12-01 00:00:00",
                    "2015-12-02 00:01:00",
                ]
            }
        )

        transformed = datetime_transformer(
            frame,
            operations=[
                {"type": "parse_datetime", "column": "datetime"},
                {"type": "extract_part", "column": "datetime", "part": "timestamp", "new_column": "ts"},
            ],
        )

        self.assertEqual(list(transformed["ts"].astype("int64")), sorted(list(transformed["ts"].astype("int64"))))

    def test_datetime_transformer_can_extract_hour_and_date_diff(self):
        frame = pd.DataFrame(
            {
                "start": ["2024-01-01 10:30:00", "2024-01-02 12:00:00"],
                "end": ["2024-01-01 08:30:00", "2024-01-02 11:00:00"],
            }
        )

        transformed = datetime_transformer(
            frame,
            operations=[
                {"type": "parse_datetime", "column": "start"},
                {"type": "parse_datetime", "column": "end"},
                {"type": "extract_part", "column": "start", "part": "hour", "new_column": "start_hour"},
                {"type": "date_diff", "column": "start", "other_column": "end", "new_column": "hours", "unit": "hours"},
            ],
        )

        self.assertEqual(transformed["start_hour"].tolist(), [10, 12])
        self.assertEqual(transformed["hours"].tolist(), [2.0, 1.0])

    def test_datetime_transformer_can_parse_multiindex_level(self):
        frame = pd.DataFrame(
            {"value": [1, 2]},
            index=pd.MultiIndex.from_tuples(
                [("a", "2024-01-01"), ("b", "2024-01-02")],
                names=["user", "timestamp"],
            ),
        )

        transformed = datetime_transformer(
            frame,
            operations=[
                {"type": "parse_datetime", "column": "timestamp", "index_level": "timestamp"},
            ],
        )

        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(
                pd.Series(transformed.index.get_level_values("timestamp"))
            )
        )

    def test_aggregator_supports_named_aggs_group_expressions_and_collect_list(self):
        frame = pd.DataFrame(
            {
                "region": ["east", "east", "west", "west"],
                "amount": [1, 2, 10, 11],
            }
        )

        grouped = aggregator(
            frame,
            group_keys=[
                "region",
                {"type": "expression", "name": "bucket", "expression": "index // 2"},
            ],
            aggregations=[
                {"column": "amount", "op": "sum", "output": "total_amount"},
                {"column": "amount", "op": "collect_list", "output": "amounts"},
                {"op": "size", "output": "row_count"},
            ],
            sort_by=["bucket", "region"],
        )

        self.assertEqual(list(grouped.columns), ["region", "bucket", "total_amount", "amounts", "row_count"])
        self.assertEqual(grouped["total_amount"].tolist(), [3, 21])
        self.assertEqual(grouped["amounts"].tolist(), [[1, 2], [10, 11]])
        self.assertEqual(grouped["row_count"].tolist(), [2, 2])

    def test_aggregator_supports_suffix_and_regex_column_selectors(self):
        frame = pd.DataFrame(
            {
                "group_color": ["red", "red", "blue"],
                "val_a": [1, 2, 3],
                "val_x": [10, 20, 30],
                "val_y": [5, 7, 9],
            }
        )

        grouped = aggregator(
            frame,
            group_keys=["group_color"],
            aggregations=[
                {"op": "first", "columns": ["group_color"]},
                {"op": "sum", "columns_suffix": "_x"},
                {"op": "mean", "columns_regex": r"^val_(?!x$).+"},
            ],
        )

        self.assertEqual(list(grouped.columns), ["group_color", "val_x", "val_a", "val_y"])
        self.assertEqual(grouped.loc[0, "val_x"], 30)
        self.assertAlmostEqual(float(grouped.loc[0, "val_a"]), 1.5)

    def test_date_range_expander_supports_global_range_and_group_fill_strategies(self):
        frame = pd.DataFrame(
            {
                "user": ["a", "a", "b"],
                "dt": ["2024-01-01", "2024-01-03", "2024-01-02"],
                "value": [1, 5, 2],
            }
        )

        expanded = date_range_expander(
            frame,
            group_keys=["user"],
            date_column="dt",
            range_scope="global",
            fill_strategies={"value": "group_max"},
        )

        self.assertEqual(len(expanded), 6)
        self.assertEqual(
            expanded.loc[(expanded["user"] == "b") & (expanded["dt"] == pd.Timestamp("2024-01-01")), "value"].iloc[0],
            2,
        )

    def test_chunk_aggregator_supports_mixed_rules_from_end(self):
        frame = pd.DataFrame(
            {
                "value_sum": [1, 2, 3, 4, 5],
                "value_mean": [10, 20, 30, 40, 50],
            }
        )

        aggregated = chunk_aggregator(
            frame,
            rules=[
                {"size": 2, "agg": "sum", "columns_suffix": "_sum"},
                {"size": 3, "agg": "mean", "columns_suffix": "_mean"},
            ],
            from_end=True,
        )

        self.assertEqual(aggregated["value_sum"].tolist(), [1, 5, 9])
        self.assertEqual(aggregated["value_mean"].iloc[0], 15.0)
        self.assertEqual(aggregated["value_mean"].iloc[1], 40.0)
        self.assertTrue(pd.isna(aggregated["value_mean"].iloc[2]))

    def test_value_counts_reporter_and_reduce_output_support_text_and_columnwise_reduction(self):
        frame = pd.DataFrame(
            {
                "id": [1, 1, 2],
                "status": ["open", "closed", "open"],
            }
        )
        report = value_counts_reporter(frame, dropna=False)
        self.assertIn("id", report)
        self.assertIn("status", report)

        extrema = reduce_output(
            pd.DataFrame({"A": [1, 3, 3], "B": [5, 2, 5]}, index=[10, 11, 12]),
            method="columnwise_extreme_index",
            anchor_extreme="min",
            anchor_occurrence="first",
            target_extreme="max",
            target_occurrence="first",
            direction="after",
        )

        self.assertEqual(extrema.to_dict(), {"A": 11, "B": 12})


if __name__ == "__main__":
    unittest.main()
