from collections import defaultdict, deque

from core.plan_utils import normalize_plan_shape
from nodes.registry import NODE_REGISTRY
from nodes.types import NodeType

MAX_PLAN_NODES = 20


def _value_matches_type(value, expected_type: str) -> bool:
    if expected_type == "any":
        return True
    if expected_type == "str":
        return isinstance(value, str)
    if expected_type == "bool":
        return isinstance(value, bool)
    if expected_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "dict":
        return isinstance(value, dict)
    if expected_type == "list":
        return isinstance(value, list)
    if expected_type == "scalar":
        return not isinstance(value, (dict, list, tuple, set))
    return False


def _validate_param_contract(node_id: str, node_type: str, provided: dict) -> list[str]:
    node = NODE_REGISTRY[node_type]
    errors: list[str] = []
    schema = node.param_schema or {}

    if not isinstance(provided, dict):
        return [f"INVALID_PARAM_BAG: '{node_id}' ({node_type}) parameters must be a dict."]

    if not node.allow_extra_params:
        unexpected = sorted(set(provided) - set(schema))
        for key in unexpected:
            errors.append(
                f"UNEXPECTED_PARAM: '{node_id}' ({node_type}) does not accept '{key}'."
            )

    for param_name, spec in schema.items():
        if param_name not in provided:
            continue

        value = provided[param_name]
        if value is None and spec.get("allow_none"):
            continue

        expected_types = spec.get("types", ["any"])
        if not any(_value_matches_type(value, expected_type) for expected_type in expected_types):
            errors.append(
                f"INVALID_PARAM_TYPE: '{node_id}' ({node_type}) param '{param_name}' "
                f"expected {expected_types}, got {type(value).__name__}."
            )
            continue

        if spec.get("choices") is not None and value not in spec["choices"]:
            errors.append(
                f"INVALID_PARAM_VALUE: '{node_id}' ({node_type}) param '{param_name}' "
                f"must be one of {spec['choices']}, got {value!r}."
            )

        if isinstance(value, list):
            min_items = spec.get("min_items")
            if min_items is not None and len(value) < min_items:
                errors.append(
                    f"INVALID_PARAM_VALUE: '{node_id}' ({node_type}) param '{param_name}' "
                    f"must contain at least {min_items} item(s)."
                )

            item_types = spec.get("item_types")
            if item_types:
                for index, item in enumerate(value):
                    if not any(_value_matches_type(item, item_type) for item_type in item_types):
                        errors.append(
                            f"INVALID_PARAM_TYPE: '{node_id}' ({node_type}) param '{param_name}[{index}]' "
                            f"expected {item_types}, got {type(item).__name__}."
                        )

    return errors


def _edge_types_are_compatible(source_type: NodeType, target_node) -> bool:
    accepted_inputs = target_node.accepted_input_types or [target_node.input_type]

    return (
        NodeType.ANY in accepted_inputs
        or source_type == NodeType.ANY
        or source_type in accepted_inputs
    )


def _describe_expected_inputs(node) -> str:
    min_inputs = node.min_inputs
    max_inputs = node.max_inputs

    if max_inputs is None:
        return f"at least {min_inputs} inbound edge(s)"
    if min_inputs == max_inputs:
        return f"{min_inputs} inbound edge(s)"
    return f"between {min_inputs} and {max_inputs} inbound edge(s)"


def validate_plan(plan: dict) -> tuple[bool, list[str]]:
    plan = normalize_plan_shape(plan)
    errors: list[str] = []

    nodes = plan.get("nodes", [])
    edges = plan.get("edges", [])
    parameters = plan.get("parameters", {})
    node_ids = [node["id"] for node in nodes]
    node_type_by_id = {node["id"]: node["type"] for node in nodes}

    # ---- CHECK 0: Plan length guard ----
    if len(nodes) > MAX_PLAN_NODES:
        errors.append(
            f"PLAN_TOO_LONG: plan has {len(nodes)} nodes, maximum supported is {MAX_PLAN_NODES}."
        )

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
        target_node = NODE_REGISTRY[node_type_by_id[target]]
        accepted_inputs = target_node.accepted_input_types or [target_node.input_type]

        if not _edge_types_are_compatible(source_output, target_node):
            errors.append(
                f"TYPE_MISMATCH: [{source} -> {target}] "
                f"{source_output} not in {accepted_inputs}"
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
    for source, target in edges:
        connected.add(source)
        connected.add(target)

    for node_id in node_ids:
        if len(node_ids) > 1 and node_id not in connected:
            errors.append(f"ORPHAN_NODE: '{node_id}' is disconnected.")

    # ---- CHECK 6: Input arity ----
    in_degree = {node_id: 0 for node_id in node_ids}
    for source, target in edges:
        in_degree[target] += 1

    for node_id in node_ids:
        node_type = node_type_by_id[node_id]
        node = NODE_REGISTRY[node_type]
        inbound_count = in_degree[node_id]
        min_inputs = node.min_inputs
        max_inputs = node.max_inputs

        if inbound_count < min_inputs or (max_inputs is not None and inbound_count > max_inputs):
            contract_hint = "source node" if node.is_source else "node contract"
            errors.append(
                f"INVALID_ARITY: '{node_id}' expects {_describe_expected_inputs(node)} "
                f"({contract_hint}), got {inbound_count}."
            )

    # ---- CHECK 7: Required parameters + contract checks ----
    for node_id in node_ids:
        node_type = node_type_by_id[node_id]
        node = NODE_REGISTRY[node_type]
        provided = parameters.get(node_id, {})

        for param in node.required_params:
            if param not in provided:
                errors.append(
                    f"MISSING_PARAM: '{node_id}' ({node_type}) requires '{param}'."
                )

        errors.extend(_validate_param_contract(node_id, node_type, provided))

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
