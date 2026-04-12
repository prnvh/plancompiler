from __future__ import annotations

import copy
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from benchmark.ds1000.comparator import DS1000ComparisonResult, compare_ds1000_result
from benchmark.ds1000.loader import DS1000Case, DS1000Task, infer_result_kind
from core.compiler import compile_output
from core.planner import PlanningContext, get_plan
from core.validator import validate_plan
from nodes.registry import NODE_REGISTRY
from nodes.types import NodeType


PlannerFn = Callable[[str], dict | tuple[dict, dict]]


@dataclass(slots=True)
class DS1000RunResult:
    task_id: str
    prompt: str
    plan_success: bool = False
    validation_success: bool = False
    compile_success: bool = False
    run_success: bool = False
    comparison_passed: bool = False
    plan: dict | None = None
    planner_usage: dict | None = None
    validation_errors: list[str] = field(default_factory=list)
    compile_error: str | None = None
    runtime_error: str | None = None
    code: str | None = None
    result: Any = None
    comparison: DS1000ComparisonResult | None = None


def _iter_task_cases(task: DS1000Task) -> list[DS1000Case]:
    if task.cases:
        return list(task.cases)
    return [
        DS1000Case(
            test_case_id=None,
            dataframe=task.dataframe,
            expected_result=task.expected_result,
            source_name=task.source_name,
            additional_inputs=task.additional_inputs,
        )
    ]


def build_execution_namespace(task: DS1000Task, case: DS1000Case | None = None) -> dict[str, Any]:
    case = case or _iter_task_cases(task)[0]
    namespace = {
        "__name__": "__main__",
        case.source_name: case.dataframe.copy(),
    }

    for key, value in case.additional_inputs.items():
        namespace[key] = copy.deepcopy(value)

    return namespace


def _build_planning_context(task: DS1000Task) -> PlanningContext:
    binding_names = {
        task.source_name,
        *[case.source_name for case in _iter_task_cases(task)],
    }
    allowed_nodes = [
        name
        for name, node in NODE_REGISTRY.items()
        if node.planner_enabled and (
            name in {"DataFrameInput", "ReduceOutput"} or node.input_type == NodeType.DATA_FRAME
        )
    ]
    notes = [
        "This workflow starts from an in-memory pandas DataFrame, not a file path or external service.",
        "Use DataFrameInput as the source node and do not introduce file readers, connectors, or exporters.",
    ]
    if binding_names:
        notes.append(
            "If DataFrameInput needs an explicit source_name, use one of the runtime bindings listed in the planning context."
        )

    return PlanningContext(
        allowed_nodes=allowed_nodes,
        source_type="in_memory_dataframe",
        desired_output_kind=task.expected_result_kind,
        workflow_mode="in_memory_dataframe_transform",
        source_binding_names=sorted(name for name in binding_names if name),
        notes=notes,
    )


def execute_compiled_plan(code: str, task: DS1000Task, case: DS1000Case | None = None) -> tuple[Any, dict[str, Any]]:
    namespace = build_execution_namespace(task, case)
    exec(code, namespace)

    if "result" not in namespace:
        raise RuntimeError("Compiled plan did not assign a value to 'result'.")

    return namespace["result"], namespace


def run_ds1000_task(
    task: DS1000Task,
    *,
    plan: dict | None = None,
    planner: PlannerFn | None = None,
) -> DS1000RunResult:
    outcome = DS1000RunResult(task_id=task.task_id, prompt=task.prompt)

    try:
        if plan is None:
            planner_fn = planner or get_plan
            if planner is None:
                planner_result = planner_fn(task.prompt, context=_build_planning_context(task))
            else:
                planner_result = planner_fn(task.prompt)
            if isinstance(planner_result, tuple) and len(planner_result) == 2:
                plan, usage = planner_result
                outcome.planner_usage = usage
            else:
                plan = planner_result

        outcome.plan = plan
        outcome.plan_success = True
    except Exception:
        outcome.runtime_error = f"Planner error:\n{traceback.format_exc()}"
        return outcome

    try:
        is_valid, errors = validate_plan(plan)
        outcome.validation_errors = errors
        outcome.validation_success = is_valid
        if not is_valid:
            return outcome
    except Exception:
        outcome.runtime_error = f"Validator error:\n{traceback.format_exc()}"
        return outcome

    try:
        code = compile_output(plan, emit_mode="result")
        outcome.code = code
        outcome.compile_success = True
    except Exception:
        outcome.compile_error = traceback.format_exc()
        return outcome

    try:
        last_result = None
        for case in _iter_task_cases(task):
            result, _namespace = execute_compiled_plan(code, task, case)
            last_result = result

            comparison = compare_ds1000_result(
                result,
                case.expected_result,
                expected_kind=task.expected_result_kind,
            )
            if not comparison.passed:
                outcome.result = result
                outcome.run_success = True
                outcome.comparison = comparison
                if case.test_case_id is not None:
                    comparison.failures = [
                        f"[test_case_id={case.test_case_id}] {failure}"
                        for failure in comparison.failures
                    ]
                outcome.comparison_passed = False
                return outcome

        outcome.result = last_result
        outcome.run_success = True
    except Exception:
        outcome.runtime_error = traceback.format_exc()
        return outcome

    comparison = DS1000ComparisonResult(
        passed=True,
        expected_kind=task.expected_result_kind,
        actual_kind=infer_result_kind(outcome.result),
        failures=[],
    )
    outcome.comparison = comparison
    outcome.comparison_passed = comparison.passed
    return outcome
