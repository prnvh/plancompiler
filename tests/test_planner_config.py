import os
import unittest
from unittest.mock import patch

from core.planner import (
    PARAMETER_SYSTEM_PROMPT,
    DEFAULT_PLANNER_MODEL,
    _extract_response_error_message,
    build_parameter_response_template,
    build_planner_payload,
    build_selected_node_details,
    get_plan,
    get_planner_model,
    get_planner_pricing,
    planner_supports_custom_temperature,
    planner_cost,
)


class PlannerConfigTests(unittest.TestCase):
    class _FakeResponse:
        def __init__(self, body: dict, status_code: int = 200):
            self._body = body
            self.status_code = status_code
            self.text = ""

        def json(self):
            return self._body

        def raise_for_status(self):
            return None

    def test_extract_response_error_message_prefers_api_error_payload(self):
        class FakeResponse:
            text = ""

            @staticmethod
            def json():
                return {
                    "error": {
                        "message": "The model does not exist",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                }

        self.assertEqual(
            _extract_response_error_message(FakeResponse()),
            "The model does not exist | type=invalid_request_error | code=model_not_found",
        )

    def test_extract_response_error_message_falls_back_to_text(self):
        class FakeResponse:
            text = "plain text failure"

            @staticmethod
            def json():
                raise ValueError("not json")

        self.assertEqual(
            _extract_response_error_message(FakeResponse()),
            "plain text failure",
        )

    def test_default_planner_model_is_used_when_env_var_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_planner_model(), DEFAULT_PLANNER_MODEL)

    def test_planner_model_can_be_overridden_with_env_var(self):
        with patch.dict(os.environ, {"PLANNER_MODEL": "gpt-5-mini"}, clear=True):
            self.assertEqual(get_planner_model(), "gpt-5-mini")
            self.assertEqual(
                get_planner_pricing(),
                {"input": 0.25, "output": 2.00},
            )

    def test_gpt5_models_do_not_send_temperature_override(self):
        payload = build_planner_payload("system", "user", "gpt-5-mini")

        self.assertFalse(planner_supports_custom_temperature("gpt-5-mini"))
        self.assertNotIn("temperature", payload)

    def test_non_gpt5_models_keep_temperature_zero(self):
        payload = build_planner_payload("system", "user", "gpt-4.1")

        self.assertTrue(planner_supports_custom_temperature("gpt-4.1"))
        self.assertEqual(payload["temperature"], 0)

    def test_unknown_model_can_use_explicit_price_overrides(self):
        with patch.dict(
            os.environ,
            {
                "PLANNER_MODEL": "custom-planner-model",
                "PLANNER_INPUT_PRICE_PER_1M": "1.5",
                "PLANNER_OUTPUT_PRICE_PER_1M": "6.5",
            },
            clear=True,
        ):
            self.assertEqual(
                get_planner_pricing(),
                {"input": 1.5, "output": 6.5},
            )
            self.assertEqual(planner_cost(1_000_000, 1_000_000), 8.0)

    def test_selected_node_details_only_include_chosen_nodes(self):
        details = build_selected_node_details(
            [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "ValueTransformer"},
            ],
            [["n1", "n2"]],
        )

        self.assertIn("n1: DataFrameInput", details)
        self.assertIn("n2: ValueTransformer", details)
        self.assertIn("allowed operation types", details)
        self.assertIn("replace_substring", details)
        self.assertIn("graph position:", details)
        self.assertNotIn("CSVParser", details)

    def test_parameter_response_template_includes_all_selected_node_ids(self):
        template = build_parameter_response_template(
            [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "Aggregator"},
                {"id": "n3", "type": "ReduceOutput"},
            ]
        )

        self.assertIn('"n1": {}', template)
        self.assertIn('"n2": {}', template)
        self.assertIn('"n3": {}', template)

    def test_parameter_prompt_mentions_parameters_only(self):
        self.assertIn("only to fill node parameters", PARAMETER_SYSTEM_PROMPT)
        self.assertNotIn("glue_code", PARAMETER_SYSTEM_PROMPT)

    @patch("requests.post")
    def test_get_plan_runs_architecture_then_parameter_stage(self, mock_post):
        architecture_body = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"nodes":[{"id":"n1","type":"DataFrameInput"},'
                            '{"id":"n2","type":"ValueTransformer"},'
                            '{"id":"n3","type":"ReduceOutput"}],'
                            '"edges":[["n1","n2"],["n2","n3"]],"flags":[]}'
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }
        parameter_body = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"parameters":{"n2":{"operations":[{"type":"replace_values",'
                            '"column":"city","mapping":{"ny":"New York"}}]}}}'
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 60, "completion_tokens": 15, "total_tokens": 75},
        }
        mock_post.side_effect = [
            self._FakeResponse(architecture_body),
            self._FakeResponse(parameter_body),
        ]

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            plan, usage = get_plan("Normalize the city values and return the result.")

        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(
            plan["nodes"],
            [
                {"id": "n1", "type": "DataFrameInput"},
                {"id": "n2", "type": "ValueTransformer"},
                {"id": "n3", "type": "ReduceOutput"},
            ],
        )
        self.assertEqual(plan["edges"], [["n1", "n2"], ["n2", "n3"]])
        self.assertEqual(plan["parameters"]["n1"], {})
        self.assertEqual(
            plan["parameters"]["n2"],
            {
                "operations": [
                    {
                        "type": "replace_values",
                        "column": "city",
                        "mapping": {"ny": "New York"},
                    }
                ]
            },
        )
        self.assertEqual(plan["parameters"]["n3"], {})
        self.assertEqual(usage["input_tokens"], 160)
        self.assertEqual(usage["output_tokens"], 35)
        self.assertEqual(usage["total_tokens"], 195)
        self.assertIn("architecture", usage["stages"])
        self.assertIn("parameters", usage["stages"])


if __name__ == "__main__":
    unittest.main()
