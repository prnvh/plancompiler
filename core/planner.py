import json
import os
import re
from openai import OpenAI
from nodes.registry import NODE_REGISTRY
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(timeout=30.0, max_retries=1)


def build_node_summary() -> str:
    lines = []

    for name, node in NODE_REGISTRY.items():
        lines.append(
            f"- {name}: {node.description} | input: {node.input_type} | output: {node.output_type} | required params: {node.required_params}"
        )

    return "\n".join(lines)


def plan_from_nodes(nodes: list[str]) -> dict:
    edges = []
    parameters = {}

    if len(nodes) == 0:
        return {"nodes": [], "edges": [], "parameters": {}, "glue_code": ""}

    for i in range(len(nodes) - 1):
        edges.append((nodes[i], nodes[i + 1]))
        parameters.setdefault(nodes[i], {})

    parameters.setdefault(nodes[-1], {})

    return {
        "nodes": nodes,
        "edges": edges,
        "parameters": parameters,
        "glue_code": ""
    }


def load_plan(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


SYSTEM_PROMPT = """
You are a code graph planner.

You select nodes from a fixed library and connect them to solve the user's task.

Output STRICTLY raw JSON.

Response format must be:

{
  "nodes": [],
  "edges": [],
  "parameters": {},
  "flags": [],
  "glue_code": ""
}

Rules:

- Only use nodes from the provided library.
- Never invent nodes.
- Every edge must be type-compatible.
- If a required node is missing, add flag MISSING_NODE.
- If credentials are needed, add flag REQUIRED_CREDENTIAL.
- Glue code must follow topological execution order.
- Return raw JSON only.
"""


def get_plan(task_description: str) -> dict:
    node_summary = build_node_summary()

    user_message = f"""
Available Nodes:
{node_summary}

Task:
{task_description}
"""

    import time
    import requests as req

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0,
    }

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

            resp.raise_for_status()

            raw = resp.json()["choices"][0]["message"]["content"].strip()
            break

        except req.exceptions.Timeout:
            if attempt == 3:
                raise RuntimeError("Planner API call timed out after 45 seconds")
            print(f"  [planner] timeout, retrying (attempt {attempt+1}/4)...")
            time.sleep(5)

        except req.exceptions.RequestException as e:
            if attempt == 3:
                raise RuntimeError(f"Planner API call failed: {e}") from e
            print(f"  [planner] network error, retrying (attempt {attempt+1}/4)...")
            time.sleep(5)

    else:
        raise RuntimeError("Planner failed after 4 attempts")

    plan = json.loads(raw)
    return normalize_plan(plan)


def _to_snake_case(name: str) -> str:
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    return s.lower()


def normalize_plan(plan: dict) -> dict:
    snake_to_camel = {node.function_name: name for name, node in NODE_REGISTRY.items()}

    raw_nodes = plan.get("nodes", [])
    if raw_nodes and isinstance(raw_nodes[0], dict):
        plan["nodes"] = [n["type"] for n in raw_nodes]
        if not plan.get("parameters"):
            plan["parameters"] = {n["type"]: n.get("params", {}) for n in raw_nodes}

    raw_nodes = plan.get("nodes", [])
    raw_edges = plan.get("edges", [])
    normalized_edges = []

    def resolve_node_ref(ref, nodes_list):
        if ref is None:
            return None
        if isinstance(ref, int):
            idx = ref - 1
            if 0 <= idx < len(nodes_list):
                return nodes_list[idx]
            if 0 <= ref < len(nodes_list):
                return nodes_list[ref]
            return None
        if isinstance(ref, str) and ref.isdigit():
            return resolve_node_ref(int(ref), nodes_list)
        return snake_to_camel.get(ref, ref)

    for edge in raw_edges:
        try:
            if isinstance(edge, dict):
                source = edge.get("from") or edge.get("source") or edge.get("start")
                target = edge.get("to") or edge.get("target") or edge.get("end")
            elif isinstance(edge, (list, tuple)) and len(edge) == 2:
                source, target = edge
            elif isinstance(edge, str) and "->" in edge:
                parts = edge.split("->")
                source, target = parts[0].strip(), parts[1].strip()
            else:
                continue

            source = resolve_node_ref(source, raw_nodes)
            target = resolve_node_ref(target, raw_nodes)

            if source is None or target is None:
                continue

            normalized_edges.append([source, target])

        except Exception:
            continue

    plan["edges"] = [e for e in normalized_edges if e[0] != e[1]]

    # Strictly linearize edges for benchmark stability.
    # Some model outputs create back-edges/cycles that can stall execution planning.
    # For this compiler, tasks are modeled as ordered pipelines, so we enforce
    # node[i] -> node[i+1] regardless of model-proposed edge shape.
    if len(plan.get("nodes", [])) > 1:
        index = {name: i for i, name in enumerate(plan["nodes"])}
        expected_edges = len(plan["nodes"]) - 1

        def is_forward_edge(edge: list[str]) -> bool:
            source, target = edge
            if source not in index or target not in index:
                return False
            return index[source] < index[target]

        has_non_forward = any(not is_forward_edge(e) for e in plan["edges"])

        if len(plan["edges"]) != expected_edges or has_non_forward:
            plan["edges"] = [
                [plan["nodes"][i], plan["nodes"][i + 1]]
                for i in range(len(plan["nodes"]) - 1)
            ]

    return plan
