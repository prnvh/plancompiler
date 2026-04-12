from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class DS1000Case:
    test_case_id: int | None
    dataframe: pd.DataFrame
    expected_result: Any
    source_name: str = "df"
    additional_inputs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DS1000Task:
    task_id: str
    prompt: str
    dataframe: pd.DataFrame
    expected_result: Any
    expected_result_kind: str
    source_name: str = "df"
    additional_inputs: dict[str, Any] = field(default_factory=dict)
    cases: list[DS1000Case] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_record: dict[str, Any] = field(default_factory=dict)


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _looks_like_task_record(value: Any) -> bool:
    return isinstance(value, dict) and any(
        key in value
        for key in (
            "task_id",
            "id",
            "prompt",
            "question",
            "description",
            "instruction",
            "df",
            "dataframe",
            "input_df",
        )
    )


def _library_matches(record: dict[str, Any], library: str | None) -> bool:
    if library is None:
        return True

    target = library.strip().lower()
    if not target:
        return True

    metadata = record.get("metadata")
    metadata_library = ""
    if isinstance(metadata, dict):
        metadata_library = str(metadata.get("library") or "")

    record_library = str(
        _first_present(record, "library", "domain") or metadata_library
    ).strip().lower()
    return record_library == target


def _coerce_records(payload: Any, records_key: str | None = None) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        if records_key:
            records = payload.get(records_key, [])
        else:
            records = _first_present(payload, "tasks", "records", "examples", "items", "data")
            if records is None and _looks_like_task_record(payload):
                records = [payload]
        if not isinstance(records, list):
            raise ValueError("DS-1000 payload must contain a list of task records.")
    else:
        raise TypeError("DS-1000 payload must be a dict or list.")

    normalized_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"DS-1000 record at index {index} must be a dict.")
        normalized_records.append(record)

    return normalized_records


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    records: list[dict[str, Any]] = []

    with opener(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue

            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise TypeError(f"DS-1000 JSONL record at index {index} must be a dict.")
            records.append(record)

    return records


def _resolve_path(value: str, base_dir: Path | None) -> Path:
    path = Path(value)
    if path.is_absolute() or base_dir is None:
        return path
    return (base_dir / path).resolve()


def _load_dataframe_from_path(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported dataframe file format for DS-1000 loader: {path.suffix}")


def _coerce_dataframe(value: Any, base_dir: Path | None) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()

    if isinstance(value, str):
        path = _resolve_path(value, base_dir)
        return _load_dataframe_from_path(path)

    if isinstance(value, list):
        if value and not all(isinstance(item, dict) for item in value):
            raise TypeError("List-based DataFrame payloads must contain dict rows.")
        return pd.DataFrame(value)

    if isinstance(value, dict):
        if "path" in value:
            path = _resolve_path(str(value["path"]), base_dir)
            return _load_dataframe_from_path(path)
        if "file_path" in value:
            path = _resolve_path(str(value["file_path"]), base_dir)
            return _load_dataframe_from_path(path)
        if "dataframe" in value:
            return _coerce_dataframe(value["dataframe"], base_dir)
        if "records" in value:
            return _coerce_dataframe(value["records"], base_dir)
        if "columns" in value and "data" in value:
            return pd.DataFrame(value["data"], columns=value["columns"])
        if "columns" in value and "values" in value:
            return pd.DataFrame(value["values"], columns=value["columns"])
        if value and all(isinstance(key, str) for key in value):
            return pd.DataFrame(value)

    raise TypeError("Could not coerce task dataframe into a pandas DataFrame.")


def _coerce_series(value: Any, base_dir: Path | None) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.copy()

    if isinstance(value, str):
        path = _resolve_path(value, base_dir)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            return _coerce_series(payload, path.parent)
        raise ValueError(f"Unsupported series file format for DS-1000 loader: {path.suffix}")

    if isinstance(value, dict):
        if "series" in value:
            return _coerce_series(value["series"], base_dir)
        if "data" in value:
            return pd.Series(
                value["data"],
                index=value.get("index"),
                name=value.get("name"),
            )
        if "values" in value:
            return pd.Series(
                value["values"],
                index=value.get("index"),
                name=value.get("name"),
            )
        return pd.Series(value)

    if isinstance(value, list):
        return pd.Series(value)

    raise TypeError("Could not coerce task expected result into a pandas Series.")


def _parse_exec_input_bindings(exec_context: str | None) -> list[str]:
    if not exec_context:
        return []

    for raw_line in exec_context.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("import ") or line.startswith("from "):
            continue
        if "= test_input" not in line:
            continue

        lhs = line.split("=", 1)[0].strip()
        parts = [part.strip() for part in lhs.split(",") if part.strip()]
        if parts:
            return parts

    return []


def _normalize_runtime_inputs(
    test_input: Any,
    binding_names: list[str],
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    if isinstance(test_input, pd.DataFrame):
        source_name = binding_names[0] if binding_names else "df"
        return test_input.copy(), source_name, {}

    if isinstance(test_input, tuple):
        values = list(test_input)
    elif isinstance(test_input, list):
        values = list(test_input)
    elif isinstance(test_input, dict):
        dataframe_keys = [key for key, value in test_input.items() if isinstance(value, pd.DataFrame)]
        if len(dataframe_keys) != 1:
            raise ValueError("DS-1000 task must expose exactly one DataFrame input for this subset.")
        source_name = str(dataframe_keys[0])
        dataframe = test_input[source_name].copy()
        additional_inputs = {
            str(key): value
            for key, value in test_input.items()
            if key != source_name
        }
        return dataframe, source_name, additional_inputs
    else:
        raise TypeError("Unsupported DS-1000 test_input shape.")

    dataframe_indexes = [
        index
        for index, value in enumerate(values)
        if isinstance(value, pd.DataFrame)
    ]
    if len(dataframe_indexes) != 1:
        raise ValueError("DS-1000 task must expose exactly one DataFrame input for this subset.")

    dataframe_index = dataframe_indexes[0]
    names = list(binding_names)
    if len(names) != len(values):
        names = [f"input_{index + 1}" for index in range(len(values))]

    source_name = names[dataframe_index]
    dataframe = values[dataframe_index].copy()
    additional_inputs = {
        names[index]: value
        for index, value in enumerate(values)
        if index != dataframe_index
    }
    return dataframe, source_name, additional_inputs


def infer_result_kind(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        return "dataframe"
    if isinstance(value, pd.Series):
        return "series"
    if pd.api.types.is_scalar(value):
        return "scalar"
    return "object"


def _infer_test_case_ids_from_code_context(code_context: str) -> list[int] | None:
    matches = [
        int(match.group(1))
        for match in re.finditer(r"for\s+\w+\s+in\s+range\((\d+)\)\s*:", code_context)
    ]
    if not matches:
        return None

    count = max(matches)
    if count <= 0:
        return None
    return list(range(1, count + 1))


def _derive_cases_from_code_context(record: dict[str, Any]) -> tuple[list[DS1000Case], dict[str, Any]]:
    code_context = record.get("code_context")
    if not isinstance(code_context, str) or not code_context.strip():
        raise ValueError("DS-1000 record is missing a usable code_context.")

    runtime: dict[str, Any] = {}
    exec(code_context, runtime)

    generate_test_case = runtime.get("generate_test_case")
    if not callable(generate_test_case):
        raise ValueError("DS-1000 code_context does not define generate_test_case().")

    exec_context = runtime.get("exec_context")
    binding_names = _parse_exec_input_bindings(exec_context if isinstance(exec_context, str) else None)

    cases: list[DS1000Case] = []
    inferred_test_case_ids = _infer_test_case_ids_from_code_context(code_context)

    if inferred_test_case_ids is None:
        explicit_max_test_cases = record.get("max_test_cases")
        if explicit_max_test_cases is not None:
            test_case_ids = list(range(1, int(explicit_max_test_cases) + 1))
        else:
            test_case_ids = [1]
    else:
        test_case_ids = inferred_test_case_ids

    for test_case_id in test_case_ids:
        try:
            test_input, expected_result = generate_test_case(test_case_id)
        except Exception:
            if inferred_test_case_ids is None and record.get("max_test_cases") is not None and cases:
                break
            raise

        dataframe, source_name, additional_inputs = _normalize_runtime_inputs(test_input, binding_names)
        cases.append(
            DS1000Case(
                test_case_id=test_case_id,
                dataframe=dataframe,
                expected_result=expected_result,
                source_name=source_name,
                additional_inputs=additional_inputs,
            )
        )

    if not cases:
        raise ValueError("DS-1000 code_context did not yield any runnable test cases.")

    all_additional_names = sorted(
        {
            name
            for case in cases
            for name in case.additional_inputs.keys()
        }
    )

    derived_metadata = {
        "derived_from_code_context": True,
        "test_case_ids": [case.test_case_id for case in cases],
        "test_case_count": len(cases),
        "input_bindings": binding_names,
        "additional_input_names": all_additional_names,
    }
    return cases, derived_metadata


def _coerce_expected_result(value: Any, explicit_kind: str | None, base_dir: Path | None) -> tuple[Any, str]:
    kind = (explicit_kind or "").strip().lower()

    if kind == "dataframe":
        result = _coerce_dataframe(value, base_dir)
        return result, "dataframe"

    if kind == "series":
        result = _coerce_series(value, base_dir)
        return result, "series"

    if kind == "scalar":
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        if not pd.api.types.is_scalar(value):
            raise TypeError("Scalar expected result must be a scalar value.")
        return value, "scalar"

    if isinstance(value, (pd.DataFrame, list)):
        result = _coerce_dataframe(value, base_dir)
        return result, "dataframe"

    if isinstance(value, dict):
        marker = str(value.get("kind") or value.get("type") or "").strip().lower()
        if marker in {"dataframe", "frame"}:
            result = _coerce_dataframe(value.get("data", value), base_dir)
            return result, "dataframe"
        if marker == "series":
            result = _coerce_series(value.get("data", value), base_dir)
            return result, "series"
        if marker == "scalar":
            scalar = value.get("value")
            if not pd.api.types.is_scalar(scalar):
                raise TypeError("Scalar expected result must be a scalar value.")
            return scalar, "scalar"
        if "columns" in value and ("data" in value or "values" in value):
            result = _coerce_dataframe(value, base_dir)
            return result, "dataframe"
        if "index" in value and ("data" in value or "values" in value):
            result = _coerce_series(value, base_dir)
            return result, "series"
        if set(value.keys()) == {"value"}:
            return value["value"], "scalar"

    if pd.api.types.is_scalar(value):
        return value, "scalar"

    raise TypeError("Could not infer DS-1000 expected result kind.")


def normalize_ds1000_task(
    record: dict[str, Any],
    *,
    index: int = 0,
    base_dir: str | Path | None = None,
) -> DS1000Task:
    base_path = Path(base_dir) if base_dir is not None else None
    record_metadata = dict(record.get("metadata") or {})
    metadata_problem_id = _first_present(record_metadata, "problem_id", "library_problem_id")

    task_id = str(
        _first_present(record, "task_id", "id", "qid", "problem_id", "name")
        or (
            f"ds1000_{str(record_metadata.get('library', 'task')).lower()}_{metadata_problem_id}"
            if metadata_problem_id is not None
            else f"ds1000_{index + 1:04d}"
        )
    )

    prompt = _first_present(record, "prompt", "question", "description", "instruction")
    if isinstance(prompt, dict):
        prompt = _first_present(prompt, "text", "content")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Task '{task_id}' is missing a usable prompt/description.")

    dataframe_payload = _first_present(
        record,
        "dataframe",
        "df",
        "input_df",
        "frame",
        "table",
    )
    if dataframe_payload is None:
        input_block = record.get("input")
        if isinstance(input_block, dict):
            dataframe_payload = _first_present(
                input_block,
                "dataframe",
                "df",
                "input_df",
                "frame",
                "table",
            )
    expected_payload = _first_present(
        record,
        "expected_result",
        "expected",
        "result",
        "answer",
        "gold",
        "output",
    )
    derived_metadata: dict[str, Any] = {}
    additional_inputs: dict[str, Any] = {}
    cases: list[DS1000Case] = []

    if dataframe_payload is None or expected_payload is None:
        cases, derived_metadata = _derive_cases_from_code_context(record)
        dataframe = cases[0].dataframe.copy()
        expected_result = cases[0].expected_result
        source_name = cases[0].source_name
        additional_inputs = dict(cases[0].additional_inputs)
        kinds = {infer_result_kind(case.expected_result) for case in cases}
        if len(kinds) != 1:
            raise ValueError("DS-1000 task produced inconsistent result kinds across test cases.")
        expected_result_kind = kinds.pop()
    else:
        dataframe = _coerce_dataframe(dataframe_payload, base_path)
        expected_result, expected_result_kind = _coerce_expected_result(
            expected_payload,
            _first_present(record, "expected_result_kind", "result_kind", "answer_type", "output_type"),
            base_path,
        )
        source_name = str(record.get("source_name") or "df")
        cases = [
            DS1000Case(
                test_case_id=None,
                dataframe=dataframe.copy(),
                expected_result=expected_result,
                source_name=source_name,
                additional_inputs={},
            )
        ]

    metadata = record_metadata
    metadata.update(derived_metadata)
    metadata.setdefault("source_path", str(base_path) if base_path is not None else None)
    metadata.setdefault("raw_keys", sorted(record.keys()))

    return DS1000Task(
        task_id=task_id,
        prompt=prompt.strip(),
        dataframe=dataframe,
        expected_result=expected_result,
        expected_result_kind=expected_result_kind,
        source_name=source_name,
        additional_inputs=additional_inputs,
        cases=cases,
        metadata=metadata,
        raw_record=dict(record),
    )


def load_ds1000_tasks(
    path: str | Path,
    *,
    records_key: str | None = None,
    library: str | None = None,
    skip_failures: bool = False,
) -> list[DS1000Task]:
    source_path = Path(path)
    source_name = source_path.name.lower()

    if source_name.endswith(".jsonl") or source_name.endswith(".jsonl.gz"):
        records = _load_jsonl_records(source_path)
    else:
        with source_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = _coerce_records(payload, records_key=records_key)

    tasks: list[DS1000Task] = []
    for index, record in enumerate(records):
        if not _library_matches(record, library):
            continue

        try:
            task = normalize_ds1000_task(record, index=index, base_dir=source_path.parent)
        except Exception:
            if skip_failures:
                continue
            raise

        tasks.append(task)

    return tasks
