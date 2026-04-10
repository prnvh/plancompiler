import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from core.compiler import compile_output
from core.validator import validate_plan
from nodes.registry import NODE_REGISTRY
from nodes.types import NodeType


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_template_module(stem: str):
    module_path = REPO_ROOT / "nodes" / "templates" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"test_{stem}", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load template module at {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ValidatorContractTests(unittest.TestCase):
    def test_source_nodes_are_explicitly_marked(self):
        self.assertTrue(NODE_REGISTRY["CSVParser"].is_source)
        self.assertEqual(NODE_REGISTRY["CSVParser"].min_inputs, 0)
        self.assertEqual(NODE_REGISTRY["CSVParser"].max_inputs, 0)

        self.assertTrue(NODE_REGISTRY["DataFrameInput"].is_source)
        self.assertEqual(NODE_REGISTRY["DataFrameInput"].input_type, NodeType.ANY)
        self.assertEqual(NODE_REGISTRY["DataFrameInput"].min_inputs, 0)
        self.assertEqual(NODE_REGISTRY["DataFrameInput"].max_inputs, 0)

        self.assertTrue(NODE_REGISTRY["SQLiteReader"].is_source)
        self.assertEqual(NODE_REGISTRY["SQLiteReader"].min_inputs, 0)
        self.assertEqual(NODE_REGISTRY["SQLiteReader"].max_inputs, 0)

    def test_validator_uses_contract_metadata_for_arity(self):
        valid_plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "CSVExporter", "params": {"output_path": "results.csv"}},
            ],
            "edges": [["n1", "n2"]],
        }

        is_valid, errors = validate_plan(valid_plan)

        self.assertTrue(is_valid, msg=f"Unexpected validation errors: {errors}")

        invalid_plan = {
            "nodes": [
                {"id": "n1", "type": "DataFilter", "params": {"condition": "score > 0.5"}},
            ],
            "edges": [],
        }

        is_valid, errors = validate_plan(invalid_plan)

        self.assertFalse(is_valid)
        self.assertTrue(any("INVALID_ARITY" in error for error in errors), errors)

    def test_series_output_can_feed_csv_exporter(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "ValueCountsOps", "params": {"column": "label"}},
                {"id": "n3", "type": "CSVExporter", "params": {"output_path": "counts.csv"}},
            ],
            "edges": [["n1", "n2"], ["n2", "n3"]],
        }

        is_valid, errors = validate_plan(plan)
        self.assertTrue(is_valid, msg=f"Unexpected validation errors: {errors}")

        code = compile_output(plan)
        self.assertIn("out_csv_exporter = csv_exporter(out_value_counts_ops, output_path='counts.csv')", code)

    def test_csv_exporter_converts_series_to_tabular_output(self):
        module = _load_template_module("csv_exporter")
        series = pd.Series([3, 1], index=pd.Index(["eng", "sales"], name="department"), name="count")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "counts.csv"
            module.csv_exporter(series, str(output_path))

            exported = pd.read_csv(output_path)

        self.assertEqual(exported.columns.tolist(), ["department", "count"])
        self.assertEqual(exported["department"].tolist(), ["eng", "sales"])
        self.assertEqual(exported["count"].tolist(), [3, 1])


if __name__ == "__main__":
    unittest.main()
