import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from benchmark.ds1000.loader import DS1000Task, load_ds1000_tasks, normalize_ds1000_task
from benchmark.ds1000.subset import (
    apply_linear_pandas_manifest,
    build_linear_pandas_manifest,
    classify_linear_pandas_task,
    filter_linear_pandas_tasks,
    is_branching_task,
    is_pandas_task,
    load_linear_pandas_manifest,
    write_linear_pandas_manifest,
)


class DS1000LoaderTests(unittest.TestCase):
    def test_normalize_task_builds_dataframe_and_series_result(self):
        raw_task = {
            "task_id": "ds_task_01",
            "question": "Using pandas, clean df and return the salary Series.",
            "input_df": {
                "columns": ["name", "salary"],
                "data": [["A", 10], ["B", 20]],
            },
            "expected_result_kind": "series",
            "expected_result": {
                "index": [0, 1],
                "data": [10, 20],
                "name": "salary",
            },
            "metadata": {"libraries": ["pandas"]},
        }

        task = normalize_ds1000_task(raw_task)

        self.assertIsInstance(task, DS1000Task)
        self.assertEqual(task.task_id, "ds_task_01")
        self.assertEqual(task.prompt, "Using pandas, clean df and return the salary Series.")
        self.assertEqual(list(task.dataframe.columns), ["name", "salary"])
        self.assertTrue(isinstance(task.expected_result, pd.Series))
        self.assertEqual(task.expected_result_kind, "series")
        self.assertEqual(task.source_name, "df")

    def test_load_tasks_supports_tasks_wrapper(self):
        payload = {
            "tasks": [
                {
                    "id": "ds_task_02",
                    "prompt": "Use pandas to sort the dataframe by score.",
                    "df": [{"name": "A", "score": 3}, {"name": "B", "score": 1}],
                    "expected": [
                        {"name": "B", "score": 1},
                        {"name": "A", "score": 3},
                    ],
                    "expected_result_kind": "dataframe",
                    "metadata": {"tags": ["pandas"]},
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tasks.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            tasks = load_ds1000_tasks(path)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_id, "ds_task_02")
        self.assertEqual(tasks[0].expected_result_kind, "dataframe")
        self.assertTrue(isinstance(tasks[0].expected_result, pd.DataFrame))
        self.assertEqual(tasks[0].expected_result.iloc[0]["name"], "B")

    def test_normalize_official_code_context_record_extracts_dataframe(self):
        raw_task = {
            "prompt": "Use pandas to reorder rows and count differences.",
            "metadata": {"library": "Pandas", "problem_id": 9},
            "code_context": """
import pandas as pd

def generate_test_case(test_case_id):
    df = pd.DataFrame({"value": [1, 2, 3]})
    order = [2, 0, 1]
    expected = (df.iloc[order].reset_index(drop=True)["value"] != df["value"]).sum()
    return (df, order), expected

exec_context = r'''
import pandas as pd
df, order = test_input
[insert]
'''
""",
        }

        task = normalize_ds1000_task(raw_task)

        self.assertEqual(task.task_id, "ds1000_pandas_9")
        self.assertEqual(task.expected_result_kind, "scalar")
        self.assertEqual(task.source_name, "df")
        self.assertEqual(task.additional_inputs["order"], [2, 0, 1])
        self.assertEqual(list(task.dataframe.columns), ["value"])
        self.assertEqual(task.metadata["test_case_count"], 1)
        self.assertEqual(len(task.cases), 1)

    def test_load_tasks_supports_jsonl_gz(self):
        record = {
            "task_id": "ds_task_08",
            "prompt": "Use pandas to return the dataframe unchanged.",
            "df": [{"name": "A", "score": 1}],
            "expected_result_kind": "dataframe",
            "expected_result": [{"name": "A", "score": 1}],
            "metadata": {"library": "Pandas"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tasks.jsonl.gz"
            import gzip

            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            tasks = load_ds1000_tasks(path)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_id, "ds_task_08")
        self.assertEqual(tasks[0].expected_result_kind, "dataframe")

    def test_load_tasks_can_filter_by_library(self):
        payload = {
            "tasks": [
                {
                    "id": "ds_task_09",
                    "prompt": "Use pandas to sort a dataframe.",
                    "df": [{"score": 1}],
                    "expected_result_kind": "dataframe",
                    "expected_result": [{"score": 1}],
                    "metadata": {"library": "Pandas"},
                },
                {
                    "id": "ds_task_10",
                    "prompt": "Use numpy to sum an array.",
                    "df": [{"score": 1}],
                    "expected_result_kind": "scalar",
                    "expected_result": 1,
                    "metadata": {"library": "NumPy"},
                },
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tasks.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            tasks = load_ds1000_tasks(path, library="Pandas")

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_id, "ds_task_09")

    def test_official_style_nested_object_result_is_classified(self):
        task = normalize_ds1000_task(
            {
                "prompt": "Use pandas to convert a dataframe into a nested dictionary.",
                "metadata": {"library": "Pandas", "problem_id": 10},
                "code_context": """
import pandas as pd

def generate_test_case(test_case_id):
    df = pd.DataFrame({"name": ["A"], "v1": ["A1"], "v2": [1]})
    expected = {"A": {"A1": 1}}
    return df, expected

exec_context = r'''
import pandas as pd
df = test_input
[insert]
'''
""",
            }
        )

        self.assertEqual(task.expected_result_kind, "object")


class DS1000SubsetTests(unittest.TestCase):
    def test_pandas_task_is_selected(self):
        task = normalize_ds1000_task(
            {
                "task_id": "ds_task_03",
                "description": "Using pandas, groupby department and return a dataframe.",
                "dataframe": [{"department": "A", "salary": 10}],
                "expected_result": [{"department": "A", "salary": 10}],
                "expected_result_kind": "dataframe",
                "metadata": {"libraries": ["pandas"]},
            }
        )

        self.assertTrue(is_pandas_task(task))
        self.assertFalse(is_branching_task(task))

        decision = classify_linear_pandas_task(task)
        self.assertTrue(decision.include)
        self.assertEqual(decision.reasons, [])

    def test_non_pandas_task_is_rejected(self):
        task = {
            "task_id": "ds_task_04",
            "prompt": "Use numpy to compute the eigenvalues of the matrix.",
            "metadata": {"libraries": ["numpy"]},
        }

        self.assertFalse(is_pandas_task(task))
        decision = classify_linear_pandas_task(task)
        self.assertFalse(decision.include)
        self.assertIn("not_pandas", decision.reasons)

    def test_branching_task_is_rejected(self):
        task = {
            "task_id": "ds_task_05",
            "prompt": "Use pandas to merge df1 and df2 on employee_id.",
            "metadata": {
                "libraries": ["pandas"],
                "input_dataframes": ["df1", "df2"],
            },
        }

        self.assertTrue(is_pandas_task(task))
        self.assertTrue(is_branching_task(task))

        decision = classify_linear_pandas_task(task)
        self.assertFalse(decision.include)
        self.assertIn("multiple_inputs", decision.reasons)
        self.assertIn("branching_required", decision.reasons)

    def test_task_with_auxiliary_inputs_is_rejected_for_df_only_subset(self):
        task = normalize_ds1000_task(
            {
                "prompt": "Use pandas to reorder rows with an auxiliary list.",
                "metadata": {"library": "Pandas", "problem_id": 12},
                "code_context": """
import pandas as pd

def generate_test_case(test_case_id):
    df = pd.DataFrame({"value": [1, 2, 3]})
    order = [1, 2, 0]
    return (df, order), df.iloc[order]

exec_context = r'''
import pandas as pd
df, order = test_input
[insert]
'''
""",
            }
        )

        decision = classify_linear_pandas_task(task)

        self.assertFalse(decision.include)
        self.assertIn("multiple_inputs", decision.reasons)
        self.assertIn("requires_auxiliary_inputs", decision.reasons)

    def test_task_with_object_result_is_rejected_for_first_pass_subset(self):
        task = normalize_ds1000_task(
            {
                "prompt": "Use pandas to convert a dataframe into a nested dictionary.",
                "metadata": {"library": "Pandas", "problem_id": 13},
                "code_context": """
import pandas as pd

def generate_test_case(test_case_id):
    df = pd.DataFrame({"name": ["A"], "value": [1]})
    expected = {"A": 1}
    return df, expected

exec_context = r'''
import pandas as pd
df = test_input
[insert]
'''
""",
            }
        )

        decision = classify_linear_pandas_task(task)

        self.assertFalse(decision.include)
        self.assertIn("unsupported_result_kind", decision.reasons)

    def test_filter_returns_selected_and_rejected(self):
        selected_task = normalize_ds1000_task(
            {
                "task_id": "ds_task_06",
                "prompt": "Using pandas, fillna and sort the dataframe.",
                "df": [{"name": "A", "score": 1}],
                "expected": [{"name": "A", "score": 1}],
                "expected_result_kind": "dataframe",
                "metadata": {"tags": ["pandas"]},
            }
        )
        rejected_task = {
            "task_id": "ds_task_07",
            "prompt": "Use pandas to concat two dataframes and compare them.",
            "metadata": {"libraries": ["pandas"], "dataframes": ["left", "right"]},
        }

        selected, rejected = filter_linear_pandas_tasks(
            [selected_task, rejected_task],
            include_rejections=True,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].task_id, "ds_task_06")
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].task_id, "ds_task_07")
        self.assertIn("branching_required", rejected[0].reasons)

    def test_manifest_round_trip_freezes_selected_and_rejected_ids(self):
        selected_task = normalize_ds1000_task(
            {
                "task_id": "ds_task_manifest_keep",
                "prompt": "Using pandas, fillna and sort the dataframe.",
                "df": [{"name": "A", "score": 1}],
                "expected": [{"name": "A", "score": 1}],
                "expected_result_kind": "dataframe",
                "metadata": {"tags": ["pandas"]},
            }
        )
        rejected_task = {
            "task_id": "ds_task_manifest_drop",
            "prompt": "Use pandas to concat two dataframes and compare them.",
            "metadata": {"libraries": ["pandas"], "dataframes": ["left", "right"]},
        }

        manifest = build_linear_pandas_manifest([selected_task, rejected_task])
        self.assertEqual(manifest["selected_task_ids"], ["ds_task_manifest_keep"])
        self.assertEqual(manifest["rejections"][0]["task_id"], "ds_task_manifest_drop")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subset_manifest.json"
            write_linear_pandas_manifest([selected_task, rejected_task], path)
            loaded_manifest = load_linear_pandas_manifest(path)

        selected, rejected = apply_linear_pandas_manifest(
            [selected_task, rejected_task],
            loaded_manifest,
            include_rejections=True,
        )

        self.assertEqual([task.task_id for task in selected], ["ds_task_manifest_keep"])
        self.assertEqual([decision.task_id for decision in rejected], ["ds_task_manifest_drop"])


if __name__ == "__main__":
    unittest.main()
