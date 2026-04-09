import unittest

from benchmark.ablations.ablations_harness import _normalize_no_linearize
from core.compiler import compile_output
from core.validator import validate_plan


class RefactorRegressionTests(unittest.TestCase):
    def test_validate_plan_still_reports_type_mismatch(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "CSVParser", "params": {"file_path": "data.csv"}},
                {"id": "n2", "type": "QueryEngine", "params": {"query": "SELECT * FROM data"}},
            ],
            "edges": [["n1", "n2"]],
        }

        is_valid, errors = validate_plan(plan)

        self.assertFalse(is_valid)
        self.assertTrue(
            any("TYPE_MISMATCH" in error for error in errors),
            msg=f"Expected type mismatch, got: {errors}",
        )

    def test_validate_plan_still_reports_invalid_arity(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "CSVParser", "params": {"file_path": "left.csv"}},
                {"id": "n2", "type": "JSONParser", "params": {"file_path": "right.json"}},
                {"id": "n3", "type": "DataFilter", "params": {"condition": "score > 10"}},
            ],
            "edges": [["n1", "n3"], ["n2", "n3"]],
        }

        is_valid, errors = validate_plan(plan)

        self.assertFalse(is_valid)
        self.assertTrue(
            any("INVALID_ARITY" in error for error in errors),
            msg=f"Expected invalid arity, got: {errors}",
        )

    def test_validate_plan_still_reports_missing_param(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "CSVParser"},
            ],
            "edges": [],
            "parameters": {},
        }

        is_valid, errors = validate_plan(plan)

        self.assertFalse(is_valid)
        self.assertTrue(
            any("MISSING_PARAM" in error for error in errors),
            msg=f"Expected missing-param error, got: {errors}",
        )

    def test_compile_output_still_uses_custom_glue_code(self):
        glue_code = "if __name__ == '__main__':\n    print('custom-run')"
        plan = {
            "nodes": [
                {"id": "n1", "type": "CSVParser", "params": {"file_path": "data.csv"}},
            ],
            "edges": [],
            "parameters": {"n1": {"file_path": "data.csv"}},
            "glue_code": glue_code,
        }

        code = compile_output(plan)

        self.assertIn("# --- Execution (LLM-generated) ---", code)
        self.assertIn(glue_code, code)
        self.assertNotIn("# --- Execution (auto-generated) ---", code)

    def test_compile_output_still_rejects_invalid_plan(self):
        invalid_plan = {
            "nodes": [
                {"id": "n1", "type": "CSVExporter"},
            ],
            "edges": [],
            "parameters": {"n1": {"output_path": "results.csv"}},
        }

        with self.assertRaises(ValueError):
            compile_output(invalid_plan)

    def test_ablation_no_linearize_path_still_preserves_forward_edges(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "CSVParser"},
                {"id": "n2", "type": "DataFilter"},
                {"id": "n3", "type": "CSVExporter"},
            ],
            "edges": [["n1", "n3"], ["n1", "n2"]],
            "parameters": {
                "n1": {"file_path": "data.csv"},
                "n2": {"condition": "score > 10"},
                "n3": {"output_path": "results.csv"},
            },
        }

        normalized = _normalize_no_linearize(plan)

        self.assertEqual(normalized["edges"], [["n1", "n3"], ["n1", "n2"]])


if __name__ == "__main__":
    unittest.main()
