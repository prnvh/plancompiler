import json
import os
import re
import time

import requests as req
from dotenv import load_dotenv

from core.plan_utils import linearize_plan_edges, normalize_plan_shape
from nodes.registry import NODE_REGISTRY

load_dotenv()


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


def _extract_json_candidates(raw: str) -> list[str]:
    stripped = raw.strip()
    candidates: list[str] = []

    if stripped:
        candidates.append(stripped)

    for match in re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL):
        candidate = match.strip()
        if candidate:
            candidates.append(candidate)

    decoder = json.JSONDecoder()
    for text in list(candidates):
        for index, char in enumerate(text):
            if char not in "{[":
                continue
            try:
                parsed, end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, dict):
                candidates.append(text[index:index + end])

    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)

    return deduped


def _parse_plan_json(raw: str) -> dict:
    last_error = None

    for candidate in _extract_json_candidates(raw):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as error:
            last_error = error
            continue

        if isinstance(parsed, dict):
            return parsed

        last_error = ValueError("Planner returned JSON that was not an object.")

    if last_error is None:
        raise ValueError("Planner returned no JSON content.")

    raise last_error


def get_plan(task_description: str, domain: str | None = None) -> dict:
    node_summary = build_node_summary()

    user_message = f"""
Available Nodes:
{node_summary}

Task:
{task_description}
"""

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
            raw = body["choices"][0]["message"]["content"].strip()
            usage = body.get("usage", {})
            plan_usage = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "cost_usd": round(
                    (usage.get("prompt_tokens", 0) / 1_000_000) * 0.15
                    + (usage.get("completion_tokens", 0) / 1_000_000) * 0.60,
                    6,
                ),
            }

            try:
                plan = _parse_plan_json(raw)
                break
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == 3:
                    raise RuntimeError(f"Planner returned invalid JSON: {e}") from e
                print(f"  [planner] invalid JSON, retrying (attempt {attempt+1}/4)...")
                time.sleep(2)
                continue

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

    return normalize_plan(plan), plan_usage


def normalize_plan(plan: dict) -> dict:
    return linearize_plan_edges(normalize_plan_shape(plan))
