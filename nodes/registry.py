from pydantic import BaseModel, Field

from nodes.registry_catalog import build_registry
from nodes.types import NodeType


class Node(BaseModel):
    name: str
    description: str
    input_type: NodeType
    output_type: NodeType
    template_path: str
    function_name: str
    required_params: list[str] = Field(default_factory=list)
    domain: str = "general"
    category: str = "misc"
    planner_enabled: bool = True
    deprecated: bool = False
    keywords: list[str] = Field(default_factory=list)
    param_schema: dict[str, dict] = Field(default_factory=dict)
    allow_extra_params: bool = True
    is_source: bool = False
    min_inputs: int = 1
    max_inputs: int | None = 1
    accepted_input_types: list[NodeType] = Field(default_factory=list)


def _spec(
    *types: str,
    required: bool = False,
    item_types: list[str] | None = None,
    choices: list | None = None,
    min_items: int | None = None,
    allow_none: bool = False,
) -> dict:
    spec: dict = {"types": list(types) or ["any"]}
    if required:
        spec["required"] = True
    if item_types is not None:
        spec["item_types"] = item_types
    if choices is not None:
        spec["choices"] = choices
    if min_items is not None:
        spec["min_items"] = min_items
    if allow_none:
        spec["allow_none"] = True
    return spec


def _node(
    *,
    name: str,
    description: str,
    input_type: NodeType,
    output_type: NodeType,
    template_path: str,
    function_name: str,
    domain: str,
    category: str,
    keywords: list[str],
    param_schema: dict[str, dict] | None = None,
    required_params: list[str] | None = None,
    planner_enabled: bool = True,
    deprecated: bool = False,
    allow_extra_params: bool = True,
    is_source: bool = False,
    min_inputs: int | None = None,
    max_inputs: int | None = None,
    accepted_input_types: list[NodeType] | None = None,
) -> Node:
    schema = param_schema or {}
    required = (
        required_params
        if required_params is not None
        else [param for param, spec in schema.items() if spec.get("required")]
    )
    resolved_min_inputs = 0 if is_source and min_inputs is None else (1 if min_inputs is None else min_inputs)
    resolved_max_inputs = 0 if is_source and max_inputs is None else (1 if max_inputs is None else max_inputs)
    resolved_accepted_input_types = accepted_input_types or [input_type]

    return Node(
        name=name,
        description=description,
        input_type=input_type,
        output_type=output_type,
        template_path=template_path,
        function_name=function_name,
        required_params=required,
        domain=domain,
        category=category,
        planner_enabled=planner_enabled,
        deprecated=deprecated,
        keywords=keywords,
        param_schema=schema,
        allow_extra_params=allow_extra_params,
        is_source=is_source,
        min_inputs=resolved_min_inputs,
        max_inputs=resolved_max_inputs,
        accepted_input_types=resolved_accepted_input_types,
    )


NODE_REGISTRY = build_registry(NodeType, _node, _spec)
