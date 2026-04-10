import unittest

from core.compiler import compile_output
from core.plan_utils import normalize_plan_shape
from core.planner import normalize_plan, plan_from_nodes
from core.validator import validate_plan


class RepeatNodeSupportTests(unittest.TestCase):
    def test_legacy_plan_is_normalized_to_instance_ids(self):
        legacy_plan = {
            "nodes": ["CSVParser", "DataFilter", "CSVExporter"],
            "edges": [["CSVParser", "DataFilter"], ["DataFilter", "CSVExporter"]],
            "parameters": {
                "CSVParser": {"file_path": "data.csv"},
                "DataFilter": {"condition": "salary > 35000"},
                "CSVExporter": {"output_path": "results.csv"},
            },
        }

        normalized = normalize_plan_shape(legacy_plan)

        self.assertEqual(
            normalized["nodes"],
            [
                {"id": "n1", "type": "CSVParser"},
                {"id": "n2", "type": "DataFilter"},
                {"id": "n3", "type": "CSVExporter"},
            ],
        )
        self.assertEqual(normalized["edges"], [["n1", "n2"], ["n2", "n3"]])
        self.assertEqual(
            normalized["parameters"],
            {
                "n1": {"file_path": "data.csv"},
                "n2": {"condition": "salary > 35000"},
                "n3": {"output_path": "results.csv"},
            },
        )

    def test_repeat_node_types_validate_and_compile(self):
        plan = {
            "nodes": [
                {"id": "n1", "type": "CSVParser", "params": {"file_path": "data.csv"}},
                {"id": "n2", "type": "DataFilter", "params": {"condition": "salary > 35000"}},
                {"id": "n3", "type": "DataFilter", "params": {"condition": "salary < 80000"}},
                {"id": "n4", "type": "CSVExporter", "params": {"output_path": "results.csv"}},
            ],
            "edges": [["n1", "n2"], ["n2", "n3"], ["n3", "n4"]],
        }

        is_valid, errors = validate_plan(plan)
        self.assertTrue(is_valid, msg=f"Unexpected validation errors: {errors}")

        code = compile_output(plan)
        self.assertIn("out_data_filter_1 = data_filter(out_csv_parser, condition='salary > 35000')", code)
        self.assertIn(
            "out_data_filter_2 = data_filter(out_data_filter_1, condition='salary < 80000')",
            code,
        )
        self.assertEqual(code.count("def data_filter("), 1)

    def test_legacy_plan_keeps_legacy_variable_names_when_types_are_unique(self):
        legacy_plan = {
            "nodes": ["CSVParser", "DataFilter", "CSVExporter"],
            "edges": [["CSVParser", "DataFilter"], ["DataFilter", "CSVExporter"]],
            "parameters": {
                "CSVParser": {"file_path": "data.csv"},
                "DataFilter": {"condition": "salary > 35000"},
                "CSVExporter": {"output_path": "results.csv"},
            },
        }

        code = compile_output(legacy_plan)
        self.assertIn("out_csv_parser = csv_parser(file_path='data.csv')", code)
        self.assertIn(
            "out_data_filter = data_filter(out_csv_parser, condition='salary > 35000')",
            code,
        )
        self.assertIn(
            "out_csv_exporter = csv_exporter(out_data_filter, output_path='results.csv')",
            code,
        )

    def test_duplicate_node_ids_are_rejected(self):
        bad_plan = {
            "nodes": [
                {"id": "n1", "type": "CSVParser"},
                {"id": "n1", "type": "CSVExporter"},
            ],
            "edges": [["n1", "n1"]],
            "parameters": {
                "n1": {"file_path": "data.csv", "output_path": "results.csv"},
            },
        }

        is_valid, errors = validate_plan(bad_plan)
        self.assertFalse(is_valid)
        self.assertTrue(
            any("DUPLICATE_NODE_ID" in error for error in errors),
            msg=f"Expected duplicate-id error, got: {errors}",
        )

    def test_normalize_plan_linearizes_with_instance_ids(self):
        messy_plan = {
            "nodes": [
                {"id": "step1", "type": "CSVParser"},
                {"id": "step2", "type": "DataFilter"},
                {"id": "step3", "type": "CSVExporter"},
            ],
            "edges": [["step2", "step3"]],
            "parameters": {
                "step1": {"file_path": "data.csv"},
                "step2": {"condition": "salary > 35000"},
                "step3": {"output_path": "results.csv"},
            },
        }

        normalized = normalize_plan(messy_plan)
        self.assertEqual(
            normalized["edges"],
            [["step1", "step2"], ["step2", "step3"]],
        )

    def test_plan_from_nodes_generates_unique_instance_ids(self):
        plan = plan_from_nodes(["CSVParser", "DataFilter", "DataFilter"])

        self.assertEqual(
            plan["nodes"],
            [
                {"id": "n1", "type": "CSVParser"},
                {"id": "n2", "type": "DataFilter"},
                {"id": "n3", "type": "DataFilter"},
            ],
        )
        self.assertEqual(plan["edges"], [["n1", "n2"], ["n2", "n3"]])


if __name__ == "__main__":
    unittest.main()
