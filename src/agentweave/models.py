"""Portable model DTOs and lazy, declarative provider selection."""
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any, Literal, Protocol
import re
from urllib.parse import urlsplit
from pydantic import Field, field_validator, model_validator
from .messages import StrictModel, Message, JSONValue, UTCDateTime, LogicalName
from .definitions.tool import ToolSchema

class ModelCapabilities(StrictModel):
    tools: bool = True
    streaming: bool = True

class ProposedToolCall(StrictModel):
    """A complete call proposed by the model; it runs only after Runtime validation."""
    provider_call_id: str
    name: str
    arguments: dict[str, JSONValue] = {}
    arguments_error: str | None = None

class ModelRequest(StrictModel):
    messages: list[Message]
    tools: list[ToolSchema]
    model_id: str
    deadline: UTCDateTime

class ModelTurn(StrictModel):
    messages: list[Message] = []
    tool_calls: list[ProposedToolCall] = []
    usage: dict[str, JSONValue] = {}
    adapter_metadata: dict[str, JSONValue] = {}

_DECIMAL = re.compile(r'^\d+(\.\d+)?$')


def _decimal(value: str) -> str:
    if type(value) is not str or not _DECIMAL.fullmatch(value):
        raise ValueError('money values are decimal strings such as "0.01"')
    return value


class ModelPricing(StrictModel):
    """Explicit, versioned tariff; never inferred from the model name."""
    revision: str = Field(min_length=1)
    input_per_million_usd: str
    output_per_million_usd: str

    @field_validator('input_per_million_usd', 'output_per_million_usd')
    @classmethod
    def decimal(cls, value):
        return _decimal(value)

    def cost(self, input_tokens: int | None, output_tokens: int | None) -> Decimal | None:
        """Cost of one attempt; unknown usage gives an unknown cost, never zero."""
        if input_tokens is None or output_tokens is None:
            return None
        return (Decimal(input_tokens) * Decimal(self.input_per_million_usd) +
                Decimal(output_tokens) * Decimal(self.output_per_million_usd)) / Decimal(1_000_000)


class ModelBudget(StrictModel):
    """Optional root-task token and cost budgets, inherited by child tasks."""
    max_tokens: int | None = Field(default=None, gt=0)
    max_cost_usd: str | None = None
    pricing: ModelPricing | None = None

    @field_validator('max_cost_usd')
    @classmethod
    def decimal(cls, value):
        return value if value is None else _decimal(value)

    @model_validator(mode='after')
    def cost_needs_pricing(self):
        if self.max_cost_usd is not None and self.pricing is None:
            raise ValueError('max_cost_usd requires explicit versioned pricing')
        return self


class TextDelta(StrictModel):
    text: str

class TurnComplete(StrictModel):
    turn: ModelTurn

ModelEvent = TextDelta | TurnComplete

class _Adapter(Protocol):
    """Structural declaration typing; the adapter protocol is providers.ModelAdapter."""
    def capabilities(self) -> ModelCapabilities: ...
    def validate_request(self, request: ModelRequest) -> None: ...
    def generate(self, request: ModelRequest) -> AsyncIterator[ModelEvent]: ...
    async def close(self) -> None: ...


def validate_env(value):
    if type(value) is not str or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', value):
        raise ValueError('secret reference must be an environment variable name')
    return value


def validate_http_url(value):
    parts = urlsplit(value)
    if parts.scheme not in ('http', 'https') or not parts.netloc or parts.username or parts.password:
        raise ValueError('expected HTTP(S) URL without embedded credentials')
    return value

class ProviderOptions(StrictModel):
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    context_budget: int | None = Field(default=None, gt=0)

class ModelSpec(StrictModel):
    provider: Literal['openai', 'compatible', 'anthropic', 'gemini', 'custom']
    model_id: str = Field(min_length=1)
    api_key_env: str | None = None
    base_url: str | None = None
    capabilities: ModelCapabilities = ModelCapabilities()
    factory_entrypoint: str | None = None
    options: dict[str, JSONValue] = {}
    custom_adapter: Any = Field(default=None, exclude=True)

    @field_validator('api_key_env')
    @classmethod
    def env(cls, value):
        return validate_env(value) if value is not None else None

    @field_validator('base_url')
    @classmethod
    def url(cls, value):
        return validate_http_url(value) if value is not None else None

    @model_validator(mode='after')
    def provider_options(self):
        if self.provider != 'custom':
            ProviderOptions.model_validate(self.options)
            if self.custom_adapter is not None or self.factory_entrypoint is not None:
                raise ValueError('built-in models cannot have a custom factory or adapter')
            if self.api_key_env is None:
                raise ValueError('built-in model requires api_key_env')
        if self.provider == 'compatible' and self.base_url is None:
            raise ValueError('compatible model requires base_url')
        if self.provider == 'custom':
            if (self.custom_adapter is None) == (self.factory_entrypoint is None):
                raise ValueError('custom model requires exactly one adapter or factory entrypoint')
            reference = self.factory_entrypoint
            if reference is not None:
                # A registered/installed adapter name, or an explicit module:object reference.
                parts = reference.split(':')
                named = len(parts) == 1 and re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,199}', reference)
                imported = len(parts) == 2 and all(p and all(x.isidentifier() for x in p.split('.')) for p in parts)
                if not (named or imported):
                    raise ValueError('factory entrypoint must be an adapter name or module:object')
        if self.provider == 'custom' and self.custom_adapter is not None:
            for method in ('capabilities', 'validate_request', 'generate', 'close'):
                if not callable(getattr(self.custom_adapter, method, None)):
                    raise ValueError(f'custom adapter requires {method}')
        return self

class Model:
    @staticmethod
    def openai(model_id: str, *, api_key_env='OPENAI_API_KEY', options=None) -> ModelSpec:
        return ModelSpec(provider='openai', model_id=model_id, api_key_env=api_key_env, options={} if options is None else options)

    @staticmethod
    def compatible(model_id: str, *, base_url: str, api_key_env='OPENAI_API_KEY', capabilities=None, options=None) -> ModelSpec:
        return ModelSpec(provider='compatible', model_id=model_id, api_key_env=api_key_env,
            base_url=base_url, capabilities=ModelCapabilities() if capabilities is None else capabilities, options={} if options is None else options)

    @staticmethod
    def anthropic(model_id: str, *, api_key_env='ANTHROPIC_API_KEY', options=None) -> ModelSpec:
        return ModelSpec(provider='anthropic', model_id=model_id, api_key_env=api_key_env, options={} if options is None else options)

    @staticmethod
    def gemini(model_id: str, *, api_key_env='GEMINI_API_KEY', options=None) -> ModelSpec:
        return ModelSpec(provider='gemini', model_id=model_id, api_key_env=api_key_env, options={} if options is None else options)

    @staticmethod
    def custom(adapter: _Adapter, *, name='custom', revision=None) -> ModelSpec:
        caps = adapter.capabilities()
        return ModelSpec(provider='custom', model_id=name, capabilities=caps,
            options={} if revision is None else {'revision':revision}, custom_adapter=adapter)
