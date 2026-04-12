import unittest

import pandas as pd

from benchmark.ds1000.comparator import compare_ds1000_result
from benchmark.ds1000.loader import DS1000Case, DS1000Task
from benchmark.ds1000.runner import execute_compiled_plan, run_ds1000_task
from core.compiler import compile_output
from core.validator import validate_plan


class DS1000RunnerTests(unittest.TestCase):
    def test_compile_output_can_assign_result(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput", "params": {"source_name": "df"}},
                {"id": "n2", "type": "ReduceOutput", "params": {"method": "scalar_agg", "column": "score", "agg": "sum"}},
            ],
            "edges": [["n1", "n2"]],
        }

        is_valid, errors = validate_plan(plan)
        self.assertTrue(is_valid, msg=f"Unexpected validation errors: {errors}")

        code = compile_output(plan, emit_mode="result")
        self.assertIn("result = out_reduce_output", code)
        self.assertNotIn("print(out_reduce_output)", code)

    def test_comparator_supports_dataframe_series_and_scalar(self):
        frame_result = compare_ds1000_result(
            pd.DataFrame({"score": [1, 2]}),
            pd.DataFrame({"score": [1, 2]}),
            expected_kind="dataframe",
        )
        series_result = compare_ds1000_result(
            pd.Series([1, 2], name="score"),
            pd.Series([1, 2], name="score"),
            expected_kind="series",
        )
        scalar_result = compare_ds1000_result(3, 3, expected_kind="scalar")

        self.assertTrue(frame_result.passed, msg=frame_result.failures)
        self.assertTrue(series_result.passed, msg=series_result.failures)
        self.assertTrue(scalar_result.passed, msg=scalar_result.failures)

    def test_comparator_reports_wrong_scalar_kind_without_raising(self):
        comparison = compare_ds1000_result(
            pd.DataFrame({"score": [1, 2]}),
            3,
            expected_kind="scalar",
        )

        self.assertFalse(comparison.passed)
        self.assertTrue(any("scalar mismatch" in failure for failure in comparison.failures))

    def test_execute_compiled_plan_injects_df_and_captures_result(self):
        task = DS1000Task(
            task_id="manual_df_identity",
            prompt="Return the dataframe unchanged.",
            dataframe=pd.DataFrame({"score": [1, 2, 3]}),
            expected_result=pd.DataFrame({"score": [1, 2, 3]}),
            expected_result_kind="dataframe",
        )
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput", "params": {"source_name": "df"}},
            ],
            "edges": [],
        }

        code = compile_output(plan, emit_mode="result")
        result, _namespace = execute_compiled_plan(code, task)

        pd.testing.assert_frame_equal(result, task.expected_result)

    def test_dataframe_input_requires_requested_binding(self):
        task = DS1000Task(
            task_id="manual_df_alias",
            prompt="Return the dataframe unchanged.",
            dataframe=pd.DataFrame({"score": [1, 2, 3]}),
            expected_result=pd.DataFrame({"score": [1, 2, 3]}),
            expected_result_kind="dataframe",
        )
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput", "params": {"source_name": "input_df"}},
            ],
            "edges": [],
        }

        code = compile_output(plan, emit_mode="result")
        result, _namespace = execute_compiled_plan(code, task)

        pd.testing.assert_frame_equal(result, task.expected_result)

    def test_execute_compiled_plan_uses_explicit_source_binding(self):
        task = DS1000Task(
            task_id="manual_named_df",
            prompt="Return the dataframe unchanged.",
            dataframe=pd.DataFrame({"score": [1, 2, 3]}),
            expected_result=pd.DataFrame({"score": [1, 2, 3]}),
            expected_result_kind="dataframe",
            source_name="input_df",
        )
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput", "params": {"source_name": "input_df"}},
            ],
            "edges": [],
        }

        code = compile_output(plan, emit_mode="result")
        result, _namespace = execute_compiled_plan(code, task)

        pd.testing.assert_frame_equal(result, task.expected_result)

    def test_run_ds1000_task_supports_series_and_scalar_outputs(self):
        frame = pd.DataFrame({"score": [1, 2, 3], "group": ["a", "b", "c"]})

        series_task = DS1000Task(
            task_id="series_task",
            prompt="Return the score column.",
            dataframe=frame,
            expected_result=frame["score"],
            expected_result_kind="series",
        )
        series_plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput", "params": {"source_name": "df"}},
                {"id": "n2", "type": "ReduceOutput", "params": {"method": "column", "column": "score"}},
            ],
            "edges": [["n1", "n2"]],
        }

        scalar_task = DS1000Task(
            task_id="scalar_task",
            prompt="Return the sum of score.",
            dataframe=frame,
            expected_result=6,
            expected_result_kind="scalar",
        )
        scalar_plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput", "params": {"source_name": "df"}},
                {"id": "n2", "type": "ReduceOutput", "params": {"method": "scalar_agg", "column": "score", "agg": "sum"}},
            ],
            "edges": [["n1", "n2"]],
        }

        series_outcome = run_ds1000_task(series_task, plan=series_plan)
        scalar_outcome = run_ds1000_task(scalar_task, plan=scalar_plan)

        self.assertTrue(series_outcome.comparison_passed, msg=series_outcome.comparison.failures if series_outcome.comparison else None)
        self.assertTrue(scalar_outcome.comparison_passed, msg=scalar_outcome.comparison.failures if scalar_outcome.comparison else None)

    def test_run_ds1000_task_executes_all_cases(self):
        frame_one = pd.DataFrame({"score": [1, 2]})
        frame_two = pd.DataFrame({"score": [3, 4]})
        task = DS1000Task(
            task_id="multi_case_task",
            prompt="Return the score column.",
            dataframe=frame_one,
            expected_result=frame_one["score"],
            expected_result_kind="series",
            cases=[
                DS1000Case(test_case_id=1, dataframe=frame_one, expected_result=frame_one["score"], source_name="df"),
                DS1000Case(test_case_id=2, dataframe=frame_two, expected_result=frame_two["score"], source_name="df"),
            ],
        )
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput", "params": {"source_name": "df"}},
                {"id": "n2", "type": "ReduceOutput", "params": {"method": "column", "column": "score"}},
            ],
            "edges": [["n1", "n2"]],
        }

        outcome = run_ds1000_task(task, plan=plan)

        self.assertTrue(outcome.comparison_passed, msg=outcome.comparison.failures if outcome.comparison else None)
        self.assertEqual(outcome.result.tolist(), [3, 4])

    def test_compile_output_strips_future_imports_from_multiple_templates(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput"},
                {
                    "id": "n2",
                    "type": "DateRangeExpander",
                    "params": {"group_keys": ["user"], "date_column": "dt"},
                },
                {
                    "id": "n3",
                    "type": "Aggregator",
                    "params": {
                        "group_keys": ["user"],
                        "aggregations": [{"column": "value", "op": "sum", "output": "total"}],
                    },
                },
            ],
            "edges": [["n1", "n2"], ["n2", "n3"]],
        }

        code = compile_output(plan, emit_mode="result")

        self.assertNotIn("from __future__ import annotations", code)


if __name__ == "__main__":
    unittest.main()
