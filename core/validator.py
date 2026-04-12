from collections import defaultdict, deque
from core.plan_utils import normalize_plan_shape
from nodes.contracts import validate_canonical_node_parameters
from nodes.registry import NODE_REGISTRY
from nodes.types import NodeType


def validate_plan(plan: dict) -> tuple[bool, list[str]]:
    if not isinstance(plan, dict):
        return False, ["INVALID_PLAN: plan must be a JSON object."]

    plan = normalize_plan_shape(plan)
    errors: list[str] = []

    nodes = plan.get("nodes", [])
    edges = plan.get("edges", [])
    parameters = plan.get("parameters", {})
    node_ids = [node["id"] for node in nodes]
    node_type_by_id = {node["id"]: node["type"] for node in nodes}

    # ---- CHECK 1: Unique ids + node existence ----
    seen_ids = set()
    for node in nodes:
        node_id = node["id"]
        node_type = node["type"]

        if node_id in seen_ids:
            errors.append(f"DUPLICATE_NODE_ID: '{node_id}' appears more than once.")
            continue

        seen_ids.add(node_id)

        if node_type not in NODE_REGISTRY:
            errors.append(f"NODE_NOT_FOUND: '{node_type}' not in registry.")

    if errors:
        return False, errors

    # ---- CHECK 2: Edge references valid nodes ----
    for source, target in edges:
        if source not in node_ids or target not in node_ids:
            errors.append(
                f"EDGE_INVALID: [{source} -> {target}] references undefined node."
            )

    # ---- CHECK 3: Type compatibility ----
    for source, target in edges:
        if source not in node_type_by_id or target not in node_type_by_id:
            continue

        source_output = NODE_REGISTRY[node_type_by_id[source]].output_type
        target_input = NODE_REGISTRY[node_type_by_id[target]].input_type

        if (
            target_input != NodeType.ANY
            and source_output != NodeType.ANY
            and source_output != target_input
        ):
            errors.append(
                f"TYPE_MISMATCH: [{source} -> {target}] "
                f"{source_output} != {target_input}"
            )

    # ---- CHECK 4: Cycle detection (topological sort) ----
    adj = defaultdict(list)
    in_degree = {node_id: 0 for node_id in node_ids}

    for source, target in edges:
        adj[source].append(target)
        in_degree[target] += 1

    queue = deque([node_id for node_id in node_ids if in_degree[node_id] == 0])
    visited = 0

    while queue:
        node = queue.popleft()
        visited += 1
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited != len(node_ids):
        errors.append("CYCLE_DETECTED: graph contains a cycle.")

    # ---- CHECK 5: Orphan nodes ----
    connected = set()
    out_degree = {node_id: 0 for node_id in node_ids}
    for source, target in edges:
        connected.add(source)
        connected.add(target)
        out_degree[source] += 1

    for node_id in node_ids:
        if len(node_ids) > 1 and node_id not in connected:
            errors.append(f"ORPHAN_NODE: '{node_id}' is disconnected.")

    # ---- CHECK 5b: Single terminal sink ----
    sink_nodes = [node_id for node_id in node_ids if out_degree[node_id] == 0]
    if len(node_ids) > 1 and len(sink_nodes) != 1:
        errors.append(
            "INVALID_TERMINAL_SHAPE: executable plans must have exactly one terminal output node; "
            f"found {len(sink_nodes)} sink nodes ({', '.join(sink_nodes)})."
        )

    # ---- CHECK 6: Input arity (branch-safe structural integrity) ----
    # Recompute in-degree cleanly (cycle step mutated it)
    in_degree = {node_id: 0 for node_id in node_ids}
    for source, target in edges:
        in_degree[target] += 1

    for node_id in node_ids:
        node_type = node_type_by_id[node_id]
        node = NODE_REGISTRY[node_type]

        if node.is_source:
            if in_degree[node_id] != 0:
                errors.append(
                    f"INVALID_ARITY: '{node_id}' expects 0 inbound edges "
                    f"(source node), got {in_degree[node_id]}."
                )
        else:
            if in_degree[node_id] != 1:
                errors.append(
                    f"INVALID_ARITY: '{node_id}' expects 1 inbound edge "
                    f"(input_type={node.input_type}), got {in_degree[node_id]}."
                )

    # ---- CHECK 7: Required parameters present ----
    for node_id in node_ids:
        node_type = node_type_by_id[node_id]
        node = NODE_REGISTRY[node_type]
        provided = parameters.get(node_id, {})

        for param in node.required_params:
            if param not in provided:
                errors.append(
                    f"MISSING_PARAM: '{node_id}' ({node_type}) requires '{param}'."
                )

        errors.extend(validate_canonical_node_parameters(node_type, provided))

    return len(errors) == 0, errors

def topological_sort(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    """
    Returns nodes in deterministic topological order.
    Raises ValueError if cycle exists.
    """

    adj = defaultdict(list)
    in_degree = {node: 0 for node in nodes}

    for source, target in edges:
        adj[source].append(target)
        in_degree[target] += 1

    queue = deque([n for n in nodes if in_degree[n] == 0])
    ordered: list[str] = []

    while queue:
        node = queue.popleft()
        ordered.append(node)

        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(ordered) != len(nodes):
        raise ValueError("Graph contains a cycle.")

    return ordered
