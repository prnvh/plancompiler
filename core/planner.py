import json
import os
import re
from dataclasses import dataclass, field
from openai import OpenAI
from nodes.contracts import get_node_contract, get_representative_contract_examples
from nodes.registry import NODE_REGISTRY
from core.plan_utils import normalize_plan_shape
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(timeout=30.0, max_retries=1)

DEFAULT_PLANNER_MODEL = "gpt-4.1"
PLANNER_PRICING_BY_MODEL = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
}


@dataclass(slots=True)
class PlanningContext:
    allowed_nodes: list[str] | None = None
    source_type: str | None = None
    desired_output_kind: str | None = None
    workflow_mode: str | None = None
    notes: list[str] = field(default_factory=list)


def get_planner_model() -> str:
    return os.getenv("PLANNER_MODEL", DEFAULT_PLANNER_MODEL).strip() or DEFAULT_PLANNER_MODEL


def get_planner_pricing(model: str | None = None) -> dict[str, float]:
    resolved_model = (model or get_planner_model()).strip()
    if resolved_model in PLANNER_PRICING_BY_MODEL:
        return dict(PLANNER_PRICING_BY_MODEL[resolved_model])

    input_override = os.getenv("PLANNER_INPUT_PRICE_PER_1M")
    output_override = os.getenv("PLANNER_OUTPUT_PRICE_PER_1M")
    if input_override and output_override:
        return {
            "input": float(input_override),
            "output": float(output_override),
        }

    return {"input": 0.0, "output": 0.0}


def planner_cost(input_tokens: int, output_tokens: int, model: str | None = None) -> float:
    pricing = get_planner_pricing(model)
    return round(
        (input_tokens / 1_000_000) * pricing["input"]
        + (output_tokens / 1_000_000) * pricing["output"],
        6,
    )


def planner_supports_custom_temperature(model: str | None = None) -> bool:
    resolved_model = (model or get_planner_model()).strip().lower()
    return not resolved_model.startswith("gpt-5")


def build_planner_payload(system_prompt: str, user_message: str, model: str | None = None) -> dict:
    planner_model = model or get_planner_model()
    payload = {
        "model": planner_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }

    if planner_supports_custom_temperature(planner_model):
        payload["temperature"] = 0

    return payload


def _tokenize_relevance_text(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", (text or "").lower())
        if len(token) > 2
    }


def _node_relevance_score(task_description: str, node_name: str, node) -> int:
    if not task_description.strip():
        return 0

    task_tokens = _tokenize_relevance_text(task_description)
    if not task_tokens:
        return 0

    node_text = " ".join(
        [
            node_name,
            node.name,
            node.description,
            node.when_to_use,
            " ".join(node.required_params),
            " ".join(node.canonical_params.keys()),
        ]
    )
    node_tokens = _tokenize_relevance_text(node_text)
    overlap = task_tokens & node_tokens
    if not overlap:
        return 0

    score = len(overlap) * 10
    if node_name in task_description:
        score += 10
    if node.name.lower() in task_description.lower():
        score += 10

    for token in overlap:
        if len(token) >= 6:
            score += 2

    return score


def build_node_summary(context: PlanningContext | None = None, task_description: str = "") -> str:
    lines = []
    allowed_nodes = set(context.allowed_nodes) if context and context.allowed_nodes else None

    registry_items = list(NODE_REGISTRY.items())
    if allowed_nodes is not None:
        registry_items = [
            (name, node)
            for name, node in NODE_REGISTRY.items()
            if name in allowed_nodes and node.planner_enabled
        ]
    else:
        registry_items = [
            (name, node)
            for name, node in NODE_REGISTRY.items()
            if node.planner_enabled
        ]

    scored_items = []
    for index, (name, node) in enumerate(registry_items):
        score = _node_relevance_score(task_description, name, node)
        scored_items.append((score, index, name, node))

    ordered_items = sorted(scored_items, key=lambda item: (-item[0], item[1]))
    if task_description.strip() and len(ordered_items) > 14:
        source_items = [item for item in ordered_items if item[3].is_source]
        non_source_items = [item for item in ordered_items if not item[3].is_source]

        detailed_items = source_items[:4]
        for item in non_source_items:
            if len(detailed_items) >= 14:
                break
            detailed_items.append(item)

        detailed_identity = {(name, index) for _score, index, name, _node in detailed_items}
        remaining_names = [
            name
            for _score, index, name, _node in ordered_items
            if (name, index) not in detailed_identity
        ]
    else:
        detailed_items = ordered_items
        remaining_names = []

    for _score, _index, name, node in detailed_items:
        required_params = ", ".join(node.required_params) if node.required_params else "none"
        line = (
            f"- {name} [{node.name}]: {node.description} | use when: {node.when_to_use} | avoid for: {node.avoid_for} | "
            f"input: {node.input_type} | output: {node.output_type} | required params: {required_params}"
        )

        if node.canonical_params:
            canonical_params = ", ".join(
                f"{param}={param_type}" for param, param_type in node.canonical_params.items()
            )
            line += f" | canonical params: {canonical_params}"

        if node.param_examples:
            if name == "DataFrameInput":
                example_candidates = node.param_examples[:2]
            else:
                example_candidates = get_representative_contract_examples(name, max_examples=2) or node.param_examples[:2]
            serialized_examples = [
                json.dumps(example, ensure_ascii=True)
                for example in example_candidates[:2]
            ]
            if len(serialized_examples) == 1:
                line += f" | example: {serialized_examples[0]}"
            else:
                line += f" | examples: {' || '.join(serialized_examples)}"

        if name == "Aggregator":
            line += " | allowed aggregation ops: sum, mean, min, max, count, size, first, last, nunique, collect_list, collect_set, collect_rows"
        if name in {"ColumnTransformer", "DataTransformer", "DataFilter", "Aggregator"}:
            line += " | expression environment: pd, np, df, index, col(name), column(name), and dataframe column names directly; row is available only for rowwise rules"

        lines.append(line)

    if remaining_names:
        lines.append("- Other available nodes: " + ", ".join(remaining_names))

    return "\n".join(lines)


def _build_selected_node_neighbors(nodes: list[dict[str, str]], edges: list[list[str]]) -> dict[str, dict[str, list[str]]]:
    node_type_by_id = {node["id"]: node["type"] for node in nodes}
    neighbors = {
        node["id"]: {"inputs": [], "outputs": []}
        for node in nodes
    }

    for source, target in edges:
        if source in neighbors and target in neighbors:
            neighbors[target]["inputs"].append(node_type_by_id[source])
            neighbors[source]["outputs"].append(node_type_by_id[target])

    return neighbors


def build_selected_node_details(nodes: list[dict[str, str]], edges: list[list[str]] | None = None) -> str:
    lines: list[str] = []
    neighbors = _build_selected_node_neighbors(nodes, edges or [])

    for node in nodes:
        node_id = node["id"]
        node_type = node["type"]
        registry_node = NODE_REGISTRY[node_type]
        contract = get_node_contract(node_type)
        node_neighbors = neighbors.get(node_id, {"inputs": [], "outputs": []})

        lines.append(
            f"- {node_id}: {node_type} [{registry_node.name}] | input: {registry_node.input_type} | "
            f"output: {registry_node.output_type} | description: {registry_node.description}"
        )
        lines.append(
            "  graph position: "
            f"inputs=[{', '.join(node_neighbors['inputs']) or 'none'}], "
            f"outputs=[{', '.join(node_neighbors['outputs']) or 'none'}]"
        )

        if registry_node.required_params:
            lines.append(
                f"  required params: {', '.join(registry_node.required_params)}"
            )

        top_level_params = contract.get("top_level_params", {})
        if top_level_params:
            lines.append(
                "  canonical top-level params: "
                + ", ".join(f"{key}={value}" for key, value in top_level_params.items())
            )

        operation_types = contract.get("operation_types", {})
        if operation_types:
            lines.append(
                "  allowed operation types: " + ", ".join(sorted(operation_types))
            )
            for operation_name, spec in sorted(operation_types.items()):
                required = spec.get("required", [])
                optional = spec.get("optional", [])
                lines.append(
                    f"    - {operation_name}: required=[{', '.join(required) or 'none'}], "
                    f"optional=[{', '.join(optional) or 'none'}]"
                )

        examples = get_representative_contract_examples(node_type, max_examples=2)
        if examples:
            serialized_examples = [
                json.dumps(example, ensure_ascii=True)
                for example in examples[:2]
            ]
            lines.append("  examples: " + " || ".join(serialized_examples))

        if node_type == "DataFrameInput":
            lines.append(
                "  note: omit source_name unless the task explicitly names a runtime binding; use {} to accept the default source."
            )
        if node_type == "ValueTransformer":
            lines.append(
                "  note: for replace_infrequent, use columns=[...] when the same threshold and replacement apply to all named columns; use column_rules when thresholds differ by column or when preserve_values are needed."
            )
        if node_type == "DataFilter":
            lines.append(
                "  note: use one boolean row-retention condition; duplicate-aware keep/drop rules can often be expressed with duplicated() inside the condition."
            )
        if node_type == "DataSorter":
            lines.append(
                "  note: when the task asks to sort an existing datetime or scalar column, sort directly by that column instead of creating helper sort columns unless the output explicitly requires them."
            )
        if node_type == "Aggregator":
            lines.append(
                "  allowed aggregation ops: sum, mean, min, max, count, size, first, last, nunique, collect_list, collect_set, collect_rows"
            )
        if node_type in {"ColumnTransformer", "DataTransformer", "DataFilter", "Aggregator"}:
            lines.append(
                "  expression environment: pd, np, df, index, col(name), column(name), and dataframe column names directly; row is available only for rowwise rules."
            )

    return "\n".join(lines)


def build_parameter_response_template(nodes: list[dict[str, str]]) -> str:
    template = {"parameters": {node["id"]: {} for node in nodes}}
    return json.dumps(template, ensure_ascii=True, indent=2)


def build_planning_context_summary(context: PlanningContext | None = None) -> str:
    if context is None:
        return ""

    lines: list[str] = []
    if context.allowed_nodes:
        lines.append(f"- Allowed nodes: {', '.join(context.allowed_nodes)}")
    if context.source_type:
        lines.append(f"- Source type: {context.source_type}")
    if context.desired_output_kind:
        lines.append(f"- Desired output kind: {context.desired_output_kind}")
    if context.workflow_mode:
        lines.append(f"- Workflow mode: {context.workflow_mode}")
    for note in context.notes:
        lines.append(f"- Constraint: {note}")

    return "\n".join(lines)


def plan_from_nodes(nodes: list[str]) -> dict:
    edges = []
    parameters = {}

    if len(nodes) == 0:
        return normalize_plan({
            "nodes": [],
            "edges": [],
            "parameters": {},
        })

    for i in range(len(nodes) - 1):
        edges.append((f"n{i + 1}", f"n{i + 2}"))
        parameters.setdefault(f"n{i + 1}", {})

    parameters.setdefault(f"n{len(nodes)}", {})

    return normalize_plan({
        "nodes": [
            {"id": f"n{i + 1}", "type": node_type}
            for i, node_type in enumerate(nodes)
        ],
        "edges": edges,
        "parameters": parameters,
    })


def load_plan(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_response_error_message(response) -> str:
    try:
        payload = response.json()
    except Exception:
        payload = None

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            code = error.get("code")
            error_type = error.get("type")
            detail_parts = [part for part in [message, f"type={error_type}" if error_type else None, f"code={code}" if code else None] if part]
            if detail_parts:
                return " | ".join(detail_parts)

    text = getattr(response, "text", "") or ""
    text = text.strip()
    if text:
        return text[:500]

    return "No error details returned by API."


ARCHITECTURE_SYSTEM_PROMPT = """
You are a code graph planner.

You select nodes from a fixed library and connect them to solve the user's task.

Output STRICTLY raw JSON.

Response format must be:

{
  "nodes": [{"id": "n1", "type": "CSVParser"}],
  "edges": [["n1", "n2"]],
  "flags": []
}

Rules:

- Only use nodes from the provided library.
- Never invent nodes.
- Every node instance must have a unique id.
- The node type string is the stable internal id. The bracketed display name is only a human-friendly label.
- Every edge must be type-compatible.
- Edges must reference node ids, not node type names.
- Prefer the most specific node whose purpose directly matches the task.
- Respect the planning context block. It defines allowed nodes, source shape, desired output kind, and workflow mode.
- Every valid plan must start with an explicit source node. Never make a transformer, reducer, or exporter the first node.
- Plan only the graph architecture in this stage: nodes, edges, and flags.
- Do not include parameters in this stage.
- Prefer the smallest valid architecture that can express the task.
- Do not insert placeholder, no-op, or cleanup nodes unless the task explicitly requires them.
- Do not use a generic fallback node when a more specific node can directly express the task.
- If the task is only about keeping or dropping rows by a condition or duplicate-aware exception rule, prefer a filter-style architecture over adding helper transformation nodes.
- Plans should have a single final terminal output node.
- If a required node is missing, add flag MISSING_NODE.
- If credentials are needed, add flag REQUIRED_CREDENTIAL.
- Return raw JSON only.
"""

SYSTEM_PROMPT = ARCHITECTURE_SYSTEM_PROMPT


PARAMETER_SYSTEM_PROMPT = """
You are a code graph parameter planner.

You are given a fixed graph architecture that already contains the chosen nodes and edges.

Your job is only to fill node parameters that make the architecture solve the task.

Output STRICTLY raw JSON.

Response format must be:

{
  "parameters": {
    "n1": {},
    "n2": {}
  }
}

Rules:

- Do not add, remove, rename, or reorder nodes.
- Do not change edges.
- Only return a parameters object keyed by the existing node ids.
- Return an object for every existing node id. Use {} when a node needs no explicit parameters.
- Use only the node contracts provided for the selected nodes.
- Never invent top-level parameter names.
- For transformer nodes, use only the listed operation types and only the listed fields for each operation type.
- Never put prose instructions, notes, scripts, code snippets, or alternate aliases into the final JSON.
- Omit optional parameters unless the task actually requires them.
- Parameterize the general transformation described by the task, not only the visibly changed values in one example table.
- When the task names multiple columns or fields for the same operation, include the full named set unless the task explicitly excludes some.
- Do not exclude a named column just because one shown example happens to produce no visible change for that column.
- When a task says a rule applies to named columns and also notes that one illustrated example has no change for one of them, still include that named column unless the task explicitly says to exclude it.
- Do not add helper columns, intermediate derived columns, or extra transforms unless the task explicitly asks for them or a downstream node requires them.
- Do not leave transformer nodes with empty operations lists.
- For source nodes, do not invent binding names that are not explicitly stated in the task.
- Return raw JSON only.
"""


def _parse_planner_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _call_planner_stage(system_prompt: str, user_message: str, planner_model: str) -> tuple[dict, dict]:
    import time
    import requests as req

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
    }
    payload = build_planner_payload(system_prompt, user_message, planner_model)

    for attempt in range(4):
        try:
            resp = req.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=45,
            )

            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  [planner] rate limited, waiting {wait}s (attempt {attempt+1}/4)...")
                time.sleep(wait)
                continue

            if 500 <= resp.status_code < 600:
                wait = 5 * (attempt + 1)
                print(f"  [planner] server error {resp.status_code}, waiting {wait}s (attempt {attempt+1}/4)...")
                time.sleep(wait)
                continue

            if 400 <= resp.status_code < 500:
                detail = _extract_response_error_message(resp)
                raise RuntimeError(
                    f"Planner API returned HTTP {resp.status_code} for model '{planner_model}': {detail}"
                )

            resp.raise_for_status()

            body = resp.json()
            raw = body["choices"][0]["message"]["content"].strip()
            usage = body.get("usage", {})
            stage_usage = {
                "model": planner_model,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "cost_usd": planner_cost(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    planner_model,
                ),
            }
            return _parse_planner_json(raw), stage_usage

        except req.exceptions.Timeout:
            if attempt == 3:
                raise RuntimeError(
                    f"Planner API call timed out after 45 seconds for model '{planner_model}'"
                )
            print(f"  [planner] timeout, retrying (attempt {attempt+1}/4)...")
            time.sleep(5)

        except req.exceptions.ConnectionError as e:
            if attempt == 3:
                raise RuntimeError(
                    f"Planner connection error after 4 attempts for model '{planner_model}': {e}"
                ) from e
            print(
                f"  [planner] connection error ({type(e).__name__}), retrying (attempt {attempt+1}/4)..."
            )
            time.sleep(5)

        except req.exceptions.RequestException as e:
            if attempt == 3:
                raise RuntimeError(
                    f"Planner request failed for model '{planner_model}': {type(e).__name__}: {e}"
                ) from e
            print(
                f"  [planner] request error ({type(e).__name__}), retrying (attempt {attempt+1}/4)..."
            )
            time.sleep(5)

    raise RuntimeError("Planner failed after 4 attempts")


def _normalize_architecture_plan(plan: dict) -> dict:
    normalized = normalize_plan_shape(
        {
            "nodes": plan.get("nodes", []),
            "edges": plan.get("edges", []),
            "flags": plan.get("flags", []),
            "parameters": {},
        }
    )
    return {
        "nodes": normalized.get("nodes", []),
        "edges": normalized.get("edges", []),
        "parameters": {node["id"]: {} for node in normalized.get("nodes", [])},
        "flags": normalized.get("flags", []),
    }


def _merge_architecture_and_parameters(
    architecture_plan: dict,
    parameter_payload: dict,
) -> dict:
    raw_parameters = parameter_payload.get("parameters", {})
    if not isinstance(raw_parameters, dict):
        raw_parameters = {}

    merged = {
        "nodes": architecture_plan.get("nodes", []),
        "edges": architecture_plan.get("edges", []),
        "parameters": raw_parameters,
        "flags": architecture_plan.get("flags", []),
    }
    return normalize_plan(merged)


def _combine_stage_usage(
    planner_model: str,
    architecture_usage: dict,
    parameter_usage: dict,
) -> dict:
    input_tokens = architecture_usage.get("input_tokens", 0) + parameter_usage.get("input_tokens", 0)
    output_tokens = architecture_usage.get("output_tokens", 0) + parameter_usage.get("output_tokens", 0)
    total_tokens = architecture_usage.get("total_tokens", 0) + parameter_usage.get("total_tokens", 0)

    return {
        "model": planner_model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": planner_cost(input_tokens, output_tokens, planner_model),
        "stages": {
            "architecture": architecture_usage,
            "parameters": parameter_usage,
        },
    }


def get_plan(task_description: str, context: PlanningContext | None = None) -> dict:
    node_summary = build_node_summary(context, task_description=task_description)
    context_summary = build_planning_context_summary(context)

    architecture_user_message = f"""
Available Nodes:
{node_summary}

Planning Context:
{context_summary or "- No additional planning context provided."}

Task:
{task_description}
"""
    planner_model = get_planner_model()
    architecture_raw, architecture_usage = _call_planner_stage(
        ARCHITECTURE_SYSTEM_PROMPT,
        architecture_user_message,
        planner_model,
    )
    architecture_plan = _normalize_architecture_plan(architecture_raw)

    selected_node_details = build_selected_node_details(
        architecture_plan["nodes"],
        architecture_plan["edges"],
    )
    parameter_response_template = build_parameter_response_template(architecture_plan["nodes"])
    parameter_user_message = f"""
Task:
{task_description}

Planning Context:
{context_summary or "- No additional planning context provided."}

Fixed Architecture:
{json.dumps(
        {
            "nodes": architecture_plan["nodes"],
            "edges": architecture_plan["edges"],
            "flags": architecture_plan.get("flags", []),
        },
        ensure_ascii=True,
        indent=2,
    )}

Selected Node Contracts:
{selected_node_details or "- No nodes were selected."}

Required Response Shape:
{parameter_response_template}
"""

    parameter_raw, parameter_usage = _call_planner_stage(
        PARAMETER_SYSTEM_PROMPT,
        parameter_user_message,
        planner_model,
    )

    merged_plan = _merge_architecture_and_parameters(architecture_plan, parameter_raw)
    combined_usage = _combine_stage_usage(
        planner_model,
        architecture_usage,
        parameter_usage,
    )
    return merged_plan, combined_usage


def _to_snake_case(name: str) -> str:
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    return s.lower()


def normalize_plan(plan: dict) -> dict:
    normalized_plan = normalize_plan_shape(plan)
    normalized_plan.pop("glue_code", None)
    return normalized_plan
