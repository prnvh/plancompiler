import json
import os
import re
from openai import OpenAI
from nodes.registry import NODE_REGISTRY
from core.plan_utils import linearize_plan_edges, normalize_plan_shape
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
        edges.append((f"n{i + 1}", f"n{i + 2}"))
        parameters.setdefault(f"n{i + 1}", {})

    parameters.setdefault(f"n{len(nodes)}", {})

    return normalize_plan_shape({
        "nodes": [
            {"id": f"n{i + 1}", "type": node_type}
            for i, node_type in enumerate(nodes)
        ],
        "edges": edges,
        "parameters": parameters,
        "glue_code": ""
    })


def load_plan(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


SYSTEM_PROMPT = """
You are a code graph planner.

You select nodes from a fixed library and connect them to solve the user's task.

Output STRICTLY raw JSON.

Response format must be:

{
  "nodes": [{"id": "n1", "type": "CSVParser"}],
  "edges": [["n1", "n2"]],
  "parameters": {},
  "flags": [],
  "glue_code": ""
}

Rules:

- Only use nodes from the provided library.
- Never invent nodes.
- Every node instance must have a unique id.
- Every edge must be type-compatible.
- Edges must reference node ids, not node type names.
- Parameters must be keyed by node id.
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

            body = resp.json()
            raw  = body["choices"][0]["message"]["content"].strip()
            _usage = body.get("usage", {})
            plan_usage = {
                "input_tokens":  _usage.get("prompt_tokens", 0),
                "output_tokens": _usage.get("completion_tokens", 0),
                "total_tokens":  _usage.get("total_tokens", 0),
                "cost_usd": round(
                    (_usage.get("prompt_tokens", 0) / 1_000_000) * 0.15 +
                    (_usage.get("completion_tokens", 0) / 1_000_000) * 0.60, 6
                ),
            }
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
    return normalize_plan(plan), plan_usage


def _to_snake_case(name: str) -> str:
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    return s.lower()


def normalize_plan(plan: dict) -> dict:
    return linearize_plan_edges(normalize_plan_shape(plan))
