import importlib.util
import unittest
from pathlib import Path

import pandas as pd

from core.compiler import compile_output
from core.planner import build_node_summary, suggest_pandas_nodes
from core.validator import MAX_PLAN_NODES, validate_plan
from nodes.registry import NODE_REGISTRY


REPO_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_TEMPLATE_STEMS = [
    "rename_columns",
    "drop_nulls",
    "fill_nulls",
    "add_constant_column",
    "arithmetic_column",
    "map_column",
    "where_column",
    "mask_column",
    "coalesce_columns",
    "strip_text",
    "lower_text",
    "upper_text",
    "replace_text",
    "split_text",
    "extract_text",
    "contains_text",
    "to_datetime",
    "extract_year",
    "extract_month",
    "extract_day",
    "date_diff",
    "pivot_frame",
    "pivot_table_frame",
    "melt_frame",
    "explode_column",
    "set_index",
    "reset_index",
    "sort_index",
    "reindex_frame",
    "rank_values",
    "cumulative_sum",
    "cumulative_count",
    "rolling_aggregate",
]


def _load_template_module(stem: str):
    module_path = REPO_ROOT / "nodes" / "templates" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"test_{stem}", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load template module at {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PandasSubsetNodeTests(unittest.TestCase):
    def test_removed_broad_nodes_are_no_longer_registered(self):
        removed = {
            "DataTransformer",
            "Aggregator",
            "AssignOps",
            "StringOps",
            "DateTimeOps",
            "ReshapeOps",
            "IndexOps",
            "RankingWindowOps",
            "NullHandler",
        }
        for node_name in removed:
            with self.subTest(node_name=node_name):
                self.assertNotIn(node_name, NODE_REGISTRY)

    def test_pandas_planner_summary_uses_semantic_nodes(self):
        summary = build_node_summary("pandas")

        self.assertIn("RenameColumns", summary)
        self.assertIn("DropNulls", summary)
        self.assertIn("GroupByAgg", summary)
        self.assertNotIn("DataTransformer", summary)
        self.assertNotIn("Aggregator", summary)
        self.assertNotIn("AssignOps", summary)

    def test_suggest_pandas_nodes_maps_common_intents(self):
        task = (
            "Rename columns, filter rows where salary > 50000, fill nulls, "
            "extract year from signup_date, then group by department and compute value counts."
        )

        suggestions = suggest_pandas_nodes(task)

        self.assertEqual(
            suggestions[:6],
            [
                "RenameColumns",
                "DataFilter",
                "FillNulls",
                "ExtractYear",
                "GroupByAgg",
                "ValueCountsOps",
            ],
        )

    def test_validate_plan_rejects_bad_param_shapes(self):
        bad_plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "RenameColumns", "params": {"columns": ["bad"]}},
                {"id": "n3", "type": "DataFilter", "params": {"condition": "salary > 1", "bogus": 1}},
            ],
            "edges": [["n1", "n2"], ["n2", "n3"]],
        }

        is_valid, errors = validate_plan(bad_plan)

        self.assertFalse(is_valid)
        self.assertTrue(any("INVALID_PARAM_TYPE" in error for error in errors), errors)
        self.assertTrue(any("UNEXPECTED_PARAM" in error for error in errors), errors)

    def test_validate_plan_rejects_overly_long_micro_plans(self):
        plan = {
            "nodes": [
                {"id": f"n{i}", "type": "CSVParser", "params": {"file_path": f"data_{i}.csv"}}
                for i in range(1, MAX_PLAN_NODES + 2)
            ],
            "edges": [],
        }

        is_valid, errors = validate_plan(plan)

        self.assertFalse(is_valid)
        self.assertTrue(any("PLAN_TOO_LONG" in error for error in errors), errors)

    def test_compile_output_supports_long_linear_small_node_chain(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "RenameColumns", "params": {"columns": {"Name": "name"}}},
                {"id": "n3", "type": "LowerText", "params": {"column": "name"}},
                {"id": "n4", "type": "ReplaceText", "params": {"column": "name", "pattern": "smith", "repl": "jones"}},
                {"id": "n5", "type": "DataFilter", "params": {"condition": "salary > 10"}},
                {"id": "n6", "type": "DataFilter", "params": {"condition": "salary < 1000"}},
                {"id": "n7", "type": "DataSorter", "params": {"by": ["salary"], "ascending": [False]}},
                {"id": "n8", "type": "ReduceOutput", "params": {"method": "head", "n": 3}},
            ],
            "edges": [
                ["n1", "n2"],
                ["n2", "n3"],
                ["n3", "n4"],
                ["n4", "n5"],
                ["n5", "n6"],
                ["n6", "n7"],
                ["n7", "n8"],
            ],
        }

        is_valid, errors = validate_plan(plan)
        self.assertTrue(is_valid, msg=f"Unexpected validation errors: {errors}")

        code = compile_output(plan)
        self.assertIn("out_data_filter_1 = data_filter(", code)
        self.assertIn("out_data_filter_2 = data_filter(", code)
        self.assertIn("out_reduce_output = reduce_output(", code)

    def test_runtime_for_all_new_semantic_templates(self):
        text_df = pd.DataFrame(
            {
                "name": ["  Alice ", "Bob", "Cara"],
                "status": ["Active", "inactive", "ACTIVE"],
                "code": ["A-10", "B-20", "C-30"],
                "tags": [["x", "y"], ["z"], []],
                "signup_date": ["2024-01-10", "2024-02-15", "2024-03-20"],
                "end_date": ["2024-01-12", "2024-02-20", "2024-03-25"],
                "salary": [10, 20, 30],
                "bonus": [1, None, 3],
                "fallback": [100, 200, 300],
                "department": ["eng", "eng", "sales"],
                "sales": [5, 10, 15],
                "score": [3, 2, 1],
                "quarter": ["Q1", "Q2", "Q1"],
            }
        )

        pivot_df = pd.DataFrame(
            {
                "department": ["eng", "eng", "sales", "sales"],
                "quarter": ["Q1", "Q2", "Q1", "Q2"],
                "sales": [10, 20, 30, 40],
            }
        )

        runtime_cases = {
            "rename_columns": lambda module: self.assertIn(
                "full_name",
                module.rename_columns(text_df, {"name": "full_name"}).columns,
            ),
            "drop_nulls": lambda module: self.assertEqual(
                len(module.drop_nulls(text_df, subset=["bonus"])),
                2,
            ),
            "fill_nulls": lambda module: self.assertEqual(
                module.fill_nulls(text_df, 0, columns=["bonus"])["bonus"].tolist(),
                [1.0, 0.0, 3.0],
            ),
            "add_constant_column": lambda module: self.assertEqual(
                module.add_constant_column(text_df, "constant", 7)["constant"].tolist(),
                [7, 7, 7],
            ),
            "arithmetic_column": lambda module: self.assertEqual(
                module.arithmetic_column(text_df, "total", "salary", "+", "sales")["total"].tolist(),
                [15, 30, 45],
            ),
            "map_column": lambda module: self.assertEqual(
                module.map_column(text_df, "department", "dept_code", {"eng": 1}, default=0)["dept_code"].tolist(),
                [1, 1, 0],
            ),
            "where_column": lambda module: self.assertEqual(
                module.where_column(text_df, "salary", "bucketed", "salary > 10", 0)["bucketed"].tolist(),
                [0, 20, 30],
            ),
            "mask_column": lambda module: self.assertEqual(
                module.mask_column(text_df, "salary", "masked", "salary > 10", -1)["masked"].tolist(),
                [10, -1, -1],
            ),
            "coalesce_columns": lambda module: self.assertEqual(
                module.coalesce_columns(text_df, "bonus_filled", ["bonus", "fallback"])["bonus_filled"].tolist(),
                [1.0, 200.0, 3.0],
            ),
            "strip_text": lambda module: self.assertEqual(
                module.strip_text(text_df, "name")["name"].tolist(),
                ["Alice", "Bob", "Cara"],
            ),
            "lower_text": lambda module: self.assertEqual(
                module.lower_text(text_df, "status")["status"].tolist(),
                ["active", "inactive", "active"],
            ),
            "upper_text": lambda module: self.assertEqual(
                module.upper_text(text_df, "status")["status"].tolist(),
                ["ACTIVE", "INACTIVE", "ACTIVE"],
            ),
            "replace_text": lambda module: self.assertEqual(
                module.replace_text(text_df, "code", "-", "_")["code"].tolist(),
                ["A_10", "B_20", "C_30"],
            ),
            "split_text": lambda module: self.assertEqual(
                module.split_text(text_df, "code", pattern="-", target="parts")["parts"].iloc[0],
                ["A", "10"],
            ),
            "extract_text": lambda module: self.assertEqual(
                module.extract_text(text_df, "code", r"([A-Z])-(\d+)", targets=["prefix", "digits"])["digits"].tolist(),
                ["10", "20", "30"],
            ),
            "contains_text": lambda module: self.assertEqual(
                module.contains_text(text_df, "status", "act", target="has_act", case=False)["has_act"].tolist(),
                [True, True, True],
            ),
            "to_datetime": lambda module: self.assertTrue(
                pd.api.types.is_datetime64_any_dtype(module.to_datetime(text_df, "signup_date")["signup_date"])
            ),
            "extract_year": lambda module: self.assertEqual(
                module.extract_year(text_df, "signup_date")["signup_date_year"].tolist(),
                [2024, 2024, 2024],
            ),
            "extract_month": lambda module: self.assertEqual(
                module.extract_month(text_df, "signup_date")["signup_date_month"].tolist(),
                [1, 2, 3],
            ),
            "extract_day": lambda module: self.assertEqual(
                module.extract_day(text_df, "signup_date")["signup_date_day"].tolist(),
                [10, 15, 20],
            ),
            "date_diff": lambda module: self.assertEqual(
                module.date_diff(text_df, "signup_date", "end_date", "days_open")["days_open"].tolist(),
                [2, 5, 5],
            ),
            "pivot_frame": lambda module: self.assertIn(
                "Q1",
                module.pivot_frame(pivot_df, index="department", columns="quarter", values="sales").columns,
            ),
            "pivot_table_frame": lambda module: self.assertEqual(
                int(module.pivot_table_frame(pivot_df, index="department", values="sales", aggfunc="sum").loc[0, "sales"]),
                30,
            ),
            "melt_frame": lambda module: self.assertEqual(
                len(module.melt_frame(pivot_df, id_vars=["department"], value_vars=["quarter", "sales"])),
                8,
            ),
            "explode_column": lambda module: self.assertEqual(
                len(module.explode_column(text_df, "tags")),
                4,
            ),
            "set_index": lambda module: self.assertEqual(
                module.set_index(text_df, "department").index.name,
                "department",
            ),
            "reset_index": lambda module: self.assertIn(
                "department",
                module.reset_index(text_df.set_index("department")).columns,
            ),
            "sort_index": lambda module: self.assertEqual(
                module.sort_index(text_df.set_index("department")).index.tolist(),
                sorted(text_df.set_index("department").index.tolist()),
            ),
            "reindex_frame": lambda module: self.assertEqual(
                module.reindex_frame(text_df[["salary"]], index=[0, 1, 4], fill_value=0).iloc[2]["salary"],
                0,
            ),
            "rank_values": lambda module: self.assertEqual(
                module.rank_values(text_df, "salary", "salary_rank", ascending=False)["salary_rank"].tolist(),
                [3.0, 2.0, 1.0],
            ),
            "cumulative_sum": lambda module: self.assertEqual(
                module.cumulative_sum(text_df, "salary", "running_salary")["running_salary"].tolist(),
                [10, 30, 60],
            ),
            "cumulative_count": lambda module: self.assertEqual(
                module.cumulative_count(text_df, "running_count")["running_count"].tolist(),
                [0, 1, 2],
            ),
            "rolling_aggregate": lambda module: self.assertEqual(
                module.rolling_aggregate(text_df, "salary", "rolling_salary", window=2, min_periods=1)["rolling_salary"].tolist(),
                [10.0, 15.0, 25.0],
            ),
        }

        for stem in SEMANTIC_TEMPLATE_STEMS:
            with self.subTest(stem=stem):
                module = _load_template_module(stem)
                runtime_cases[stem](module)

    def test_groupby_and_value_counts_and_reduce_output_runtime(self):
        group_module = _load_template_module("group_by_agg")
        counts_module = _load_template_module("value_counts_ops")
        reduce_module = _load_template_module("reduce_output")

        df = pd.DataFrame(
            {
                "department": ["eng", "eng", "sales"],
                "salary": [100, 150, 80],
            }
        )

        grouped = group_module.group_by_agg(
            df,
            group_by=["department"],
            aggregations=[
                {"output": "salary_mean", "column": "salary", "agg": "mean"},
                {"output": "row_count", "agg": "size"},
            ],
        )
        counts = counts_module.value_counts_ops(df, column="department")
        scalar = reduce_module.reduce_output(df, method="scalar_agg", column="salary", agg="sum")

        self.assertEqual(grouped.loc[grouped["department"] == "eng", "row_count"].iloc[0], 2)
        self.assertEqual(counts.loc["eng"], 2)
        self.assertEqual(scalar, 330)


if __name__ == "__main__":
    unittest.main()
