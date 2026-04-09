from __future__ import annotations

import copy
import re

from nodes.registry import NODE_REGISTRY


def _snake_to_camel_map() -> dict[str, str]:
    return {node.function_name: name for name, node in NODE_REGISTRY.items()}


def _normalize_node_type(node_type):
    if not isinstance(node_type, str):
        return node_type

    cleaned = node_type.strip()
    return _snake_to_camel_map().get(cleaned, cleaned)


def _default_node_id(index: int) -> str:
    return f"n{index + 1}"


def _coerce_nodes(raw_nodes: list) -> tuple[list[dict[str, str]], dict[str, dict]]:
    nodes: list[dict[str, str]] = []
    inline_parameters: dict[str, dict] = {}

    for index, raw_node in enumerate(raw_nodes):
        node_id = None
        node_type = None
        params = {}

        if isinstance(raw_node, dict):
            node_id = (
                raw_node.get("id")
                or raw_node.get("node_id")
                or raw_node.get("instance_id")
                or raw_node.get("alias")
            )
            node_type = (
                raw_node.get("type")
                or raw_node.get("node_type")
                or raw_node.get("name")
            )
            params = raw_node.get("params") or raw_node.get("parameters") or {}
        else:
            node_type = raw_node

        if not node_id:
            node_id = _default_node_id(index)

        node_id = str(node_id)
        node_type = _normalize_node_type(node_type)
        nodes.append({"id": node_id, "type": node_type})

        if isinstance(params, dict) and params:
            inline_parameters[node_id] = dict(params)

    return nodes, inline_parameters


def _merge_parameters(
    raw_parameters: dict,
    nodes: list[dict[str, str]],
    inline_parameters: dict[str, dict],
) -> dict[str, dict]:
    parameters = {
        node_id: dict(node_params)
        for node_id, node_params in inline_parameters.items()
    }

    node_ids = {node["id"] for node in nodes}
    ids_by_type: dict[str, list[str]] = {}

    for node in nodes:
        ids_by_type.setdefault(node["type"], []).append(node["id"])

    per_type_cursor = {node_type: 0 for node_type in ids_by_type}

    for raw_key, raw_value in (raw_parameters or {}).items():
        if not isinstance(raw_value, dict):
            continue

        key = str(raw_key).strip()
        target_id = None

        if key in node_ids:
            target_id = key
        else:
            normalized_key = _normalize_node_type(key)
            matches = ids_by_type.get(normalized_key, [])

            if len(matches) == 1:
                target_id = matches[0]
            elif matches:
                cursor = per_type_cursor[normalized_key]
                if cursor < len(matches):
                    target_id = matches[cursor]
                    per_type_cursor[normalized_key] += 1

        if target_id is None:
            continue

        merged = dict(parameters.get(target_id, {}))
        merged.update(raw_value)
        parameters[target_id] = merged

    return parameters


def _resolve_node_ref(ref, nodes: list[dict[str, str]]) -> str | None:
    node_ids = [node["id"] for node in nodes]
    node_id_set = set(node_ids)
    ids_by_type: dict[str, list[str]] = {}

    for node in nodes:
        ids_by_type.setdefault(node["type"], []).append(node["id"])

    if ref is None:
        return None

    if isinstance(ref, dict):
        return _resolve_node_ref(ref.get("id") or ref.get("type"), nodes)

    if isinstance(ref, int):
        idx = ref - 1
        if 0 <= idx < len(node_ids):
            return node_ids[idx]
        if 0 <= ref < len(node_ids):
            return node_ids[ref]
        return None

    if isinstance(ref, str):
        cleaned = ref.strip()

        if cleaned.isdigit():
            return _resolve_node_ref(int(cleaned), nodes)

        if cleaned in node_id_set:
            return cleaned

        # Allow refs like DataFilter#2 or DataFilter:2 when a type repeats.
        match = re.match(r"^(.*?)(?:#|:|\[)(\d+)\]?$", cleaned)
        if match:
            node_type = _normalize_node_type(match.group(1).strip())
            ordinal = int(match.group(2)) - 1
            matches = ids_by_type.get(node_type, [])
            if 0 <= ordinal < len(matches):
                return matches[ordinal]

        normalized_type = _normalize_node_type(cleaned)
        matches = ids_by_type.get(normalized_type, [])
        if len(matches) == 1:
            return matches[0]

    return None


def normalize_plan_shape(plan: dict) -> dict:
    raw_nodes = plan.get("nodes", [])
    nodes, inline_parameters = _coerce_nodes(raw_nodes)
    parameters = _merge_parameters(plan.get("parameters", {}), nodes, inline_parameters)

    normalized_edges = []
    for edge in plan.get("edges", []):
        try:
            if isinstance(edge, dict):
                source = edge.get("from") or edge.get("source") or edge.get("start")
                target = edge.get("to") or edge.get("target") or edge.get("end")
            elif isinstance(edge, (list, tuple)) and len(edge) == 2:
                source, target = edge
            elif isinstance(edge, str) and "->" in edge:
                source, target = [part.strip() for part in edge.split("->", 1)]
            else:
                continue

            source_id = _resolve_node_ref(source, nodes)
            target_id = _resolve_node_ref(target, nodes)

            if source_id is None or target_id is None or source_id == target_id:
                continue

            normalized_edges.append([source_id, target_id])
        except Exception:
            continue

    return {
        **copy.deepcopy(plan),
        "nodes": nodes,
        "edges": normalized_edges,
        "parameters": parameters,
        "flags": copy.deepcopy(plan.get("flags", [])),
        "glue_code": plan.get("glue_code", ""),
    }


def linearize_plan_edges(plan: dict) -> dict:
    plan = copy.deepcopy(plan)
    nodes = plan.get("nodes", [])

    if len(nodes) <= 1:
        return plan

    ordered_ids = [node["id"] for node in nodes]
    index = {node_id: idx for idx, node_id in enumerate(ordered_ids)}
    expected_edges = len(nodes) - 1

    def is_forward_edge(edge: list[str]) -> bool:
        source, target = edge
        if source not in index or target not in index:
            return False
        return index[source] < index[target]

    has_non_forward = any(not is_forward_edge(edge) for edge in plan.get("edges", []))

    if len(plan.get("edges", [])) != expected_edges or has_non_forward:
        plan["edges"] = [
            [ordered_ids[i], ordered_ids[i + 1]]
            for i in range(len(ordered_ids) - 1)
        ]

    return plan
