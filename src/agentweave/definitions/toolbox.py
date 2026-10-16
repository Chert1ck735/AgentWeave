"""Validated HTTP/MCP descriptions and named tool sets, without adapters."""
import re
from typing import Literal as Kind
from pydantic import Field, field_validator, model_validator
from ..messages import StrictModel, LogicalName, JSONValue, validate_name
from ..models import validate_http_url
from ..config import SecretRef
from .tool import ToolDefinition, RetryPolicy, ToolSchema, schema_revision

_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
# Credentials in these headers must come from the environment, never from configuration.
_SECRET_HEADERS = frozenset(('authorization', 'proxy-authorization', 'x-api-key', 'api-key', 'cookie',
                             'x-auth-token', 'x-access-token', 'x-amz-security-token', 'x-csrf-token'))
# Headers that route, frame or re-target a request: never filled by a model argument.
_ROUTING_HEADERS = frozenset(('host', 'content-length', 'transfer-encoding', 'connection', 'te', 'upgrade',
                              'forwarded', 'content-type', 'content-encoding', 'accept-encoding', 'expect',
                              'keep-alive', 'proxy-connection', 'x-http-method-override', 'x-http-method',
                              'x-method-override'))


class Literal(StrictModel):
    """A fixed value supplied by the configuration."""
    value: JSONValue


class Argument(StrictModel):
    """The value of one model argument; an absent optional argument is left out."""
    name: str = Field(min_length=1)


Mapping = Literal | Argument | SecretRef


class HttpToolSpec(StrictModel):
    """A JSON HTTP(S) tool defined by configuration alone.

    Method and URL are fixed. The model fills only the declared input schema,
    which is closed (additionalProperties false), and mappings place literals,
    arguments and environment secrets into query, headers and body. Secrets are
    referenced by environment variable name and resolved only when called.
    timeout bounds a whole attempt (retry.timeout does not apply); retry decides
    whether and how often a failed attempt may run again. Configured URLs are
    trusted: nothing blocks private addresses an operator chose deliberately.
    """
    kind: Kind['http'] = 'http'
    name: LogicalName
    method: Kind['GET', 'POST', 'PUT', 'PATCH', 'DELETE']
    url: str
    description: str = ''
    input_schema: dict[str, JSONValue]
    output_schema: dict[str, JSONValue] = {}
    query: dict[str, Mapping] = {}
    headers: dict[str, Mapping] = {}
    body: dict[str, Mapping] = {}
    timeout: float = Field(default=30, gt=0)
    max_response_bytes: int = Field(default=1048576, gt=0, le=1048576)
    idempotency_header: str | None = None
    retry: RetryPolicy = RetryPolicy()
    allowed_callers: tuple[LogicalName, ...] | None = None

    @model_validator(mode='before')
    @classmethod
    def closed_input(cls, data):
        if isinstance(data, dict) and isinstance(data.get('input_schema'), dict):
            schema = data['input_schema']
            if schema.get('additionalProperties', False) is not False:
                raise ValueError('the input schema must not allow undeclared arguments')
            data = data | {'input_schema': schema | {'additionalProperties': False}}
        return data

    @field_validator('url')
    @classmethod
    def http(cls, value):
        return validate_http_url(value)

    @model_validator(mode='after')
    def consistent(self):
        from jsonschema import Draft202012Validator, SchemaError
        if self.input_schema.get('type') != 'object':
            raise ValueError('the input schema must describe a JSON object')
        try:
            Draft202012Validator.check_schema(self.input_schema)
            Draft202012Validator.check_schema(self.output_schema)
        except SchemaError as exc:
            raise ValueError(f'invalid JSON Schema: {exc.message}') from exc
        declared = set(self.input_schema.get('properties', {}))
        for part in (self.query, self.headers, self.body):
            for value in part.values():
                if isinstance(value, Argument) and value.name not in declared:
                    raise ValueError(f'argument {value.name} is not declared in the input schema')
        if self.body and self.method in ('GET', 'DELETE'):
            raise ValueError(f'{self.method} requests carry no body')
        names = [*self.headers, *([self.idempotency_header] if self.idempotency_header else [])]
        if len({name.lower() for name in names}) != len(names) or not all(_HEADER_NAME.fullmatch(n) for n in names):
            raise ValueError('header names must be distinct HTTP tokens')
        for name, value in self.headers.items():
            lowered = name.lower()
            if lowered in _SECRET_HEADERS and not isinstance(value, SecretRef):
                raise ValueError('credential headers require SecretRef')
            if isinstance(value, Argument) and (lowered in _ROUTING_HEADERS or lowered.startswith('x-forwarded-')):
                raise ValueError(f'header {name} routes or frames the request and cannot come from an argument')
        return self

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(tool_id=self.name, description=self.description, input_schema=self.input_schema,
                          output_schema=self.output_schema,
                          revision=schema_revision(self.input_schema, self.output_schema, self.description))


class McpToolboxSpec(StrictModel):
    """A toolbox whose tools come from one MCP server, served as an independent tool service.

    stdio starts command with args (an argv list, never a shell line) as this
    service's own child; http connects to url and never owns that server.
    Secrets reach the child environment and HTTP headers only by reference.
    allow keeps only the named server tools; each is published as
    namespace.name and the catalog follows the server's list as it changes,
    by notification or by polling every refresh_seconds.
    """
    kind: Kind['mcp'] = 'mcp'
    namespace: LogicalName
    transport: Kind['stdio', 'http']
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, SecretRef] = {}
    url: str | None = None
    headers: dict[str, SecretRef] = {}
    allow: tuple[str, ...] | None = None
    refresh_seconds: float = Field(default=30, gt=0)
    timeout: float = Field(default=30, gt=0)
    retry: RetryPolicy = RetryPolicy()
    allowed_callers: tuple[LogicalName, ...] | None = None

    @field_validator('args', 'allow', mode='before')
    @classmethod
    def sequences(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode='after')
    def transport_config(self):
        if self.transport == 'stdio':
            if not self.command or self.url is not None or self.headers:
                raise ValueError('stdio requires command and forbids URL/headers')
        else:
            if self.url is None or self.command is not None or self.args or self.env:
                raise ValueError('HTTP requires URL and forbids process configuration')
            validate_http_url(self.url)
        return self

class PythonBinding(StrictModel):
    kind: Kind['python'] = 'python'
    target: str

    @field_validator('target')
    @classmethod
    def module_object(cls, value):
        parts = value.split(':')
        if len(parts) != 2 or not all(p and all(x.isidentifier() for x in p.split('.')) for p in parts):
            raise ValueError('Python binding must be module:object')
        return value

class ToolboxDefinition(StrictModel):
    namespace: LogicalName
    bindings: dict[str, ToolDefinition | PythonBinding | HttpToolSpec]

    @field_validator('bindings', mode='before')
    @classmethod
    def http_names(cls, values):
        """An HTTP binding in a toolbox is named by its key unless it names itself."""
        if not isinstance(values, dict):
            return values
        return {name: binding | {'name': binding.get('name', name)}
                if isinstance(binding, dict) and binding.get('kind') == 'http' else binding
                for name, binding in values.items()}

    @field_validator('bindings')
    @classmethod
    def aliases(cls, values):
        aliases = set()
        for name in values:
            validate_name(name)
            alias = name.replace('.', '_').replace('-', '_')
            if alias in aliases:
                raise ValueError('tool aliases collide')
            aliases.add(alias)
        return values


def configured_toolbox(namespace: str, values: dict) -> ToolboxDefinition | McpToolboxSpec:
    """A toolbox from agentweave.toml: an MCP server when kind = "mcp", named bindings otherwise."""
    values = {'namespace': namespace} | values
    if values.get('kind') == 'mcp':
        return McpToolboxSpec.model_validate(values)
    return ToolboxDefinition.model_validate(values)
