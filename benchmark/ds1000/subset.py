from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.ds1000.loader import DS1000Task

_PANDAS_METHOD_HINTS = (
    "dataframe",
    "series",
    "groupby",
    "pivot",
    "pivot_table",
    "melt",
    "explode",
    "fillna",
    "astype",
    "sort_values",
    "value_counts",
    "rolling",
    "cumsum",
    "cumcount",
    "reset_index",
    "set_index",
    "to_datetime",
    "iloc",
    "loc",
)

_NON_PANDAS_HINTS = (
    "numpy",
    "np.",
    "matplotlib",
    "seaborn",
    "plotly",
    "sklearn",
    "scikit-learn",
    "scipy",
    "tensorflow",
    "pytorch",
    "sql",
    "sqlite",
    "database",
    "api",
    "http",
    "requests",
)

_BRANCHING_PATTERNS = (
    r"\bmerge\b",
    r"\bjoin\b",
    r"\bconcat\b",
    r"\bappend\b",
    r"\bcombine\b",
    r"\balign\b",
    r"\bcompare two dataframes\b",
    r"\bcompare two series\b",
    r"\bdf1\b.*\bdf2\b",
    r"\bleft dataframe\b.*\bright dataframe\b",
    r"\btwo dataframes\b",
    r"\bmultiple dataframes\b",
)


@dataclass(slots=True)
class TaskSelection:
    task_id: str
    include: bool
    reasons: list[str] = field(default_factory=list)


def _task_id(task: DS1000Task | dict[str, Any]) -> str:
    if isinstance(task, DS1000Task):
        return task.task_id
    return str(task.get("task_id") or task.get("id") or "unknown")


def _record(task: DS1000Task | dict[str, Any]) -> dict[str, Any]:
    if isinstance(task, DS1000Task):
        return task.raw_record
    return task


def _metadata(task: DS1000Task | dict[str, Any]) -> dict[str, Any]:
    if isinstance(task, DS1000Task):
        return task.metadata

    metadata = task.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return {}


def _prompt_text(task: DS1000Task | dict[str, Any]) -> str:
    if isinstance(task, DS1000Task):
        return task.prompt.lower()

    record = _record(task)
    prompt = (
        record.get("prompt")
        or record.get("question")
        or record.get("description")
        or record.get("instruction")
        or ""
    )
    if isinstance(prompt, dict):
        prompt = prompt.get("text") or prompt.get("content") or ""
    return str(prompt).lower()


def _iter_classifier_values(task: DS1000Task | dict[str, Any]) -> list[str]:
    record = _record(task)
    metadata = _metadata(task)

    values: list[str] = []
    for container in (record, metadata):
        for key in ("library", "libraries", "framework", "frameworks", "tag", "tags", "domain", "domains"):
            raw = container.get(key)
            if isinstance(raw, list):
                values.extend(str(item).lower() for item in raw)
            elif raw is not None:
                values.append(str(raw).lower())

    return values


def _count_explicit_sources(task: DS1000Task | dict[str, Any]) -> int | None:
    if isinstance(task, DS1000Task):
        return 1 + len(task.additional_inputs)

    record = _record(task)
    metadata = _metadata(task)

    for container in (record, metadata):
        for key in (
            "source_count",
            "input_count",
            "dataframe_count",
            "num_inputs",
        ):
            value = container.get(key)
            if isinstance(value, int):
                return value

        for key in (
            "input_dataframes",
            "dataframes",
            "frames",
            "tables",
            "sources",
            "inputs",
            "datasets",
        ):
            value = container.get(key)
            if isinstance(value, list):
                return len(value)

    return None


def _requires_auxiliary_inputs(task: DS1000Task | dict[str, Any]) -> bool:
    if isinstance(task, DS1000Task):
        return bool(task.additional_inputs)

    metadata = _metadata(task)
    additional_names = metadata.get("additional_input_names")
    if isinstance(additional_names, list) and len(additional_names) > 0:
        return True

    return False


def is_pandas_task(task: DS1000Task | dict[str, Any]) -> bool:
    labels = _iter_classifier_values(task)
    prompt = _prompt_text(task)

    if any("pandas" in label for label in labels):
        return True

    explicit_labels_exist = bool(labels)
    if explicit_labels_exist and not any("pandas" in label for label in labels):
        return False

    has_pandas_signal = "pandas" in prompt or any(hint in prompt for hint in _PANDAS_METHOD_HINTS)
    has_non_pandas_signal = any(hint in prompt for hint in _NON_PANDAS_HINTS)

    return has_pandas_signal and not has_non_pandas_signal


def is_branching_task(task: DS1000Task | dict[str, Any]) -> bool:
    source_count = _count_explicit_sources(task)
    if source_count is not None and source_count > 1:
        return True

    prompt = _prompt_text(task)
    return any(re.search(pattern, prompt) for pattern in _BRANCHING_PATTERNS)


def classify_linear_pandas_task(task: DS1000Task | dict[str, Any]) -> TaskSelection:
    reasons: list[str] = []

    if not is_pandas_task(task):
        reasons.append("not_pandas")

    source_count = _count_explicit_sources(task)
    if source_count is not None and source_count > 1:
        reasons.append("multiple_inputs")

    if _requires_auxiliary_inputs(task):
        reasons.append("requires_auxiliary_inputs")

    if isinstance(task, DS1000Task) and task.expected_result_kind not in {"dataframe", "series", "scalar"}:
        reasons.append("unsupported_result_kind")

    if is_branching_task(task):
        reasons.append("branching_required")

    return TaskSelection(
        task_id=_task_id(task),
        include=not reasons,
        reasons=reasons,
    )


def filter_linear_pandas_tasks(
    tasks: list[DS1000Task | dict[str, Any]],
    *,
    include_rejections: bool = False,
) -> list[DS1000Task | dict[str, Any]] | tuple[list[DS1000Task | dict[str, Any]], list[TaskSelection]]:
    selected: list[DS1000Task | dict[str, Any]] = []
    rejected: list[TaskSelection] = []

    for task in tasks:
        decision = classify_linear_pandas_task(task)
        if decision.include:
            selected.append(task)
        else:
            rejected.append(decision)

    if include_rejections:
        return selected, rejected

    return selected


def build_linear_pandas_manifest(tasks: list[DS1000Task | dict[str, Any]]) -> dict[str, Any]:
    selected, rejected = filter_linear_pandas_tasks(tasks, include_rejections=True)
    return {
        "manifest_version": 1,
        "subset_name": "ds1000_pandas_linear_non_branching",
        "selected_task_ids": [_task_id(task) for task in selected],
        "rejections": [
            {
                "task_id": decision.task_id,
                "reasons": list(decision.reasons),
            }
            for decision in rejected
        ],
    }


def write_linear_pandas_manifest(tasks: list[DS1000Task | dict[str, Any]], path: str | Path) -> Path:
    manifest = build_linear_pandas_manifest(tasks)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def load_linear_pandas_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def apply_linear_pandas_manifest(
    tasks: list[DS1000Task | dict[str, Any]],
    manifest: dict[str, Any],
    *,
    include_rejections: bool = False,
) -> list[DS1000Task | dict[str, Any]] | tuple[list[DS1000Task | dict[str, Any]], list[TaskSelection]]:
    selected_ids = set(manifest.get("selected_task_ids", []))
    rejection_map = {
        entry["task_id"]: list(entry.get("reasons", []))
        for entry in manifest.get("rejections", [])
        if isinstance(entry, dict) and "task_id" in entry
    }

    selected = [task for task in tasks if _task_id(task) in selected_ids]

    if not include_rejections:
        return selected

    selected_id_lookup = {_task_id(task) for task in selected}
    rejected = [
        TaskSelection(
            task_id=_task_id(task),
            include=False,
            reasons=list(rejection_map.get(_task_id(task), ["not_in_manifest"])),
        )
        for task in tasks
        if _task_id(task) not in selected_id_lookup
    ]
    return selected, rejected
