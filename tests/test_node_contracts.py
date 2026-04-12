import unittest

import pandas as pd

from core.planner import build_node_summary
from nodes.registry import NODE_REGISTRY
from nodes.templates.data_filter import data_filter
from nodes.templates.data_deduplicator import data_deduplicator
from nodes.templates.data_sorter import data_sorter
from nodes.templates.data_transformer import data_transformer


class NodeContractsTests(unittest.TestCase):
    def test_all_nodes_have_clear_planner_guidance(self):
        for node_type, node in NODE_REGISTRY.items():
            self.assertTrue(node.name.strip(), msg=f"{node_type} is missing a clear display name")
            self.assertTrue(node.description.strip(), msg=f"{node_type} is missing a description")
            self.assertTrue(node.when_to_use.strip(), msg=f"{node_type} is missing when_to_use guidance")
            self.assertTrue(node.avoid_for.strip(), msg=f"{node_type} is missing avoid_for guidance")

    def test_build_node_summary_surfaces_usage_and_avoidance_notes(self):
        summary = build_node_summary()

        self.assertIn("DataFilter [Keep Matching Rows Only]", summary)
        self.assertIn("use when:", summary)
        self.assertIn("avoid for:", summary)
        self.assertIn("ReduceOutput [Return Final DataFrame Series Or Scalar]", summary)
        self.assertIn("canonical params:", summary)
        self.assertIn("example:", summary)
        self.assertNotIn("DataFrameJoin [Join Two DataFrames]", summary)

    def test_data_transformer_supports_transformations_list_and_casts(self):
        frame = pd.DataFrame(
            {
                "fruit": ["apple", "banana", "banana", "pear"],
                "event_time": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            }
        )

        transformed = data_transformer(
            frame,
            transformations=[
                {
                    "operation": "replace",
                    "column": "fruit",
                    "condition": "value_counts() >= 2",
                    "new_value": "other",
                },
                {
                    "type": "cast",
                    "column": "event_time",
                    "to": "datetime64[ns]",
                },
            ],
        )

        self.assertEqual(
            transformed["fruit"].tolist(),
            ["other", "banana", "banana", "other"],
        )
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(transformed["event_time"]))

    def test_data_filter_supports_membership_logic_outside_query(self):
        frame = pd.DataFrame(
            {
                "record_id": ["a", "a", "b", "c"],
                "preserve_duplicate": ["Yes", "No", "No", "No"],
            }
        )

        filtered = data_filter(
            frame,
            "preserve_duplicate == 'Yes' or record_id not in record_id[preserve_duplicate == 'No']",
        )

        self.assertEqual(filtered["record_id"].tolist(), ["a"])
        self.assertEqual(filtered["preserve_duplicate"].tolist(), ["Yes"])

    def test_data_filter_can_evaluate_dataframe_method_chain_conditions(self):
        frame = pd.DataFrame(
            {
                "ValueA": [0.0, 2.0, 0.5],
                "ValueB": [0.0, 0.0, 1.5],
                "label": ["x", "y", "z"],
            }
        )

        filtered = data_filter(
            frame,
            "filter(regex='^Value').abs().gt(1).any(axis=1)",
        )

        self.assertEqual(filtered["label"].tolist(), ["y", "z"])

    def test_data_sorter_supports_multi_column_inputs(self):
        frame = pd.DataFrame(
            {
                "group": ["b", "a", "a"],
                "score": [2, 3, 1],
            }
        )

        sorted_frame = data_sorter(frame, by=["group", "score"], ascending=[True, False])

        self.assertEqual(
            sorted_frame.reset_index(drop=True).to_dict(orient="records"),
            [
                {"group": "a", "score": 3},
                {"group": "a", "score": 1},
                {"group": "b", "score": 2},
            ],
        )

    def test_data_deduplicator_supports_subset_and_keep(self):
        frame = pd.DataFrame(
            {
                "record_id": [1, 2, 3, 4],
                "group": ["a", "a", "b", "b"],
            }
        )

        deduped = data_deduplicator(frame, subset=["group"], keep="last", ignore_index=True)

        self.assertEqual(
            deduped.to_dict(orient="records"),
            [
                {"record_id": 2, "group": "a"},
                {"record_id": 4, "group": "b"},
            ],
        )

    def test_data_transformer_supports_general_shift_and_melt_operations(self):
        frame = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "left": [10, None, 30],
                "right": [None, 20, 40],
            }
        )

        transformed = data_transformer(
            frame,
            transformations=[
                {"type": "circular_shift", "column": "id", "shift": 1},
                {"type": "shift_non_nulls_left", "columns": ["left", "right"], "fill_value": None},
                {
                    "type": "melt",
                    "id_vars": ["id"],
                    "value_vars": ["left", "right"],
                    "var_name": "side",
                    "value_name": "value",
                },
            ],
        )

        self.assertEqual(list(transformed.columns), ["id", "side", "value"])
        self.assertEqual(transformed["id"].tolist()[:3], [3, 1, 2])

    def test_data_transformer_can_fallback_from_df_eval_to_rowwise_expression(self):
        frame = pd.DataFrame(
            {
                "url": ["a", "a", "b"],
                "keep_if_dup": ["Yes", "No", "No"],
            }
        )

        transformed = data_transformer(
            frame,
            transformations=[
                {
                    "type": "derive_column",
                    "new_column": "url_key",
                    "expression": "url + '_' + str(row.name) if keep_if_dup == 'Yes' else url",
                }
            ],
        )

        self.assertEqual(transformed["url_key"].tolist(), ["a_0", "a", "b"])


if __name__ == "__main__":
    unittest.main()
