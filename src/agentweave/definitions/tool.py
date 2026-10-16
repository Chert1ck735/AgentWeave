"""Python tool declarations and schema derivation, with no execution engine."""
from collections.abc import Callable
from dataclasses import dataclass
import functools
import hashlib
import inspect
import json
from typing import get_type_hints
from pydantic import BaseModel, Field, TypeAdapter, model_validator, create_model, ConfigDict
from ..messages import StrictModel, LogicalName, JSONValue

class ToolSchema(StrictModel):
    tool_id: LogicalName
    revision: str
    description: str
    input_schema: dict[str, JSONValue]
    output_schema: dict[str, JSONValue]

class RetryPolicy(StrictModel):
    safe: bool = False
    max_attempts: int = Field(default=3, ge=1)
    min_delay: float = Field(default=1, ge=0)
    max_delay: float = Field(default=10, ge=0)
    timeout: float = Field(default=30, gt=0)
    jitter: bool = True

    @model_validator(mode='after')
    def ordered(self):
        if self.max_delay < self.min_delay:
            raise ValueError('max_delay must be >= min_delay')
        return self

class ToolDefinition(StrictModel):
    name: LogicalName
    callable: Callable
    schema_: ToolSchema = Field(alias='schema', serialization_alias='schema')
    retry: RetryPolicy = RetryPolicy()
    allowed_callers: tuple[LogicalName, ...] | None = None

    @property
    def schema(self) -> ToolSchema:
        return self.schema_


def retry_policy(value) -> RetryPolicy:
    if value is None:
        return RetryPolicy()
    if value == 'safe':
        return RetryPolicy(safe=True)
    if isinstance(value, RetryPolicy):
        return value
    raise ValueError('retry must be RetryPolicy, safe, or None')


@dataclass(frozen=True)
class ToolSignature:
    """Typed view of a Python tool: strict input model, injected context and output type."""
    input_model: type
    context_parameter: str | None
    output: TypeAdapter

    def parse(self, arguments: dict) -> BaseModel:
        """Validate JSON arguments exactly as the published schema describes them.

        Strict JSON mode accepts a JSON array for a tuple or an ISO string for a
        datetime, which strict Python-mode validation would reject.
        """
        return self.input_model.model_validate_json(json.dumps(arguments), strict=True)

    def keywords(self, arguments: dict) -> dict:
        model = self.parse(arguments)
        return {name: getattr(model, name) for name in type(model).model_fields}


def _is_tool_context(annotation) -> bool:
    return (getattr(annotation, '__name__', '') == 'ToolContext' and
            getattr(annotation, '__module__', '') == 'agentweave.context')


@functools.cache
def tool_signature(function: Callable) -> ToolSignature:
    """Derive the model-facing input model; ToolContext is runtime-injected, not input."""
    hints = get_type_hints(function)
    parameters, context_parameter = {}, None
    for name, param in inspect.signature(function).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD, param.POSITIONAL_ONLY):
            raise ValueError('tool parameters must support named arguments')
        if name not in hints:
            raise ValueError(f'tool parameter {name} needs a type annotation')
        annotation = hints[name]
        if _is_tool_context(annotation):
            context_parameter = name
            continue
        if param.default is inspect.Parameter.empty:
            parameters[name] = (annotation, ...)
        else:
            TypeAdapter(JSONValue).validate_python(param.default)
            # A default the model may rely on must itself be a valid JSON argument.
            TypeAdapter(annotation).validate_json(json.dumps(param.default), strict=True)
            parameters[name] = (annotation, param.default)
    if 'return' not in hints:
        raise ValueError('tool needs a return annotation')
    input_model = create_model(function.__name__ + 'Input', __config__=ConfigDict(extra='forbid', strict=True), **parameters)
    return ToolSignature(input_model, context_parameter, TypeAdapter(hints['return']))


def schema_revision(input_schema: dict, output_schema: dict, description: str) -> str:
    """The revision callers pin: it changes whenever the model-facing contract changes."""
    return hashlib.sha256(json.dumps([input_schema, output_schema, description], sort_keys=True).encode()).hexdigest()


def tool(fn=None, *, retry=None):
    def decorate(function):
        if not callable(function):
            raise ValueError('tool requires a callable')
        signature = tool_signature(function)
        input_schema = signature.input_model.model_json_schema()
        input_schema.pop('title', None)
        output_schema = signature.output.json_schema()
        description = inspect.getdoc(function) or ''
        revision = schema_revision(input_schema, output_schema, description)
        return ToolDefinition(name=function.__name__, callable=function,
            schema=ToolSchema(tool_id=function.__name__, revision=revision, description=description,
                              input_schema=input_schema, output_schema=output_schema), retry=retry_policy(retry))
    return decorate(fn) if fn is not None else decorate
