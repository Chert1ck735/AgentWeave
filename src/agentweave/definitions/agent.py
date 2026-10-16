"""Agent declarations and unambiguous handler registration."""
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Literal
from pydantic import Field, PrivateAttr, field_validator, model_validator
from ..messages import CLIENT_PREFIX, StrictModel, LogicalName, is_client, validate_name
from ..models import ModelBudget, ModelSpec
from .selections import PeerSelection, ToolSelection
from .tool import ToolDefinition, RetryPolicy, retry_policy
from .toolbox import HttpToolSpec

class SubscriptionPolicy(StrictModel):
    queue_limit: int = Field(default=64, gt=0)
    concurrency: int = Field(default=1, gt=0)
    ordering: Literal['key', 'unordered'] = 'key'
    overflow: Literal['wait', 'reject', 'latest'] = 'wait'
    replaceable: bool = False

    @model_validator(mode='after')
    def valid_latest(self):
        if self.overflow == 'latest' and not self.replaceable:
            raise ValueError('latest requires replaceable snapshots')
        return self

class HandlerDefinition(StrictModel):
    callable: Callable
    retry: RetryPolicy
    policy: SubscriptionPolicy = SubscriptionPolicy()

class Agent(StrictModel):
    name: LogicalName
    instructions: str = ''
    model: ModelSpec | None = None
    budget: ModelBudget | None = None
    tools: list[ToolDefinition | HttpToolSpec] | ToolSelection | None = None
    peers: PeerSelection | None = None
    memory: Literal['stateless', 'stateful'] = 'stateless'
    concurrency: int = Field(default=1, ge=1)
    revision: str | None = None
    max_inflight: int = Field(default=64, gt=0)
    subscription_policies: dict[LogicalName, SubscriptionPolicy] = {}
    _handlers: dict[str, HandlerDefinition] = PrivateAttr(default_factory=dict)
    _request: HandlerDefinition | None = PrivateAttr(default=None)
    _before_tool: list[Callable] = PrivateAttr(default_factory=list)
    _after_tool: list[Callable] = PrivateAttr(default_factory=list)
    # Hooks from agentweave.toml are kept apart so each resolution replaces them.
    _configured_before_tool: list[Callable] = PrivateAttr(default_factory=list)
    _configured_after_tool: list[Callable] = PrivateAttr(default_factory=list)
    _resolved_root: object = PrivateAttr(default=None)
    _source_hashes: dict[str, str] = PrivateAttr(default_factory=dict)
    _package_versions: dict[str, str] = PrivateAttr(default_factory=dict)

    def __init__(self, name, instructions='', model=None, tools=None, peers=None,
                 memory='stateless', concurrency=1, revision=None, max_inflight=64, subscription_policies=None,
                 budget=None, **kwargs):
        super().__init__(name=name, instructions=instructions, model=model, budget=budget, tools=tools,
                         peers=peers, memory=memory, concurrency=concurrency, revision=revision, max_inflight=max_inflight,
                         subscription_policies={} if subscription_policies is None else subscription_policies, **kwargs)

    @field_validator('name')
    @classmethod
    def not_client_name(cls, value):
        if is_client(value):
            raise ValueError(f'names starting with {CLIENT_PREFIX!r} are reserved for clients')
        return value

    @field_validator('tools')
    @classmethod
    def unique_tools(cls, tools):
        if isinstance(tools, list):
            names = [x.name for x in tools]
            aliases = [x.replace('.', '_').replace('-', '_') for x in names]
            if len(set(aliases)) != len(aliases):
                raise ValueError('duplicate or colliding tool aliases')
        return tools

    @property
    def handlers(self) -> Mapping[str, HandlerDefinition]:
        return MappingProxyType(self._handlers.copy())

    @property
    def request_handler(self) -> HandlerDefinition | None:
        return self._request

    @property
    def before_tool_hooks(self) -> tuple[Callable, ...]:
        """Python-declared hooks first, then the ones configured in agentweave.toml."""
        return tuple(self._before_tool + self._configured_before_tool)

    @property
    def after_tool_hooks(self) -> tuple[Callable, ...]:
        return tuple(self._after_tool + self._configured_after_tool)

    def before_tool(self, hook):
        """Register a check run before every tool effect; see agentweave.hooks."""
        if not callable(hook):
            raise ValueError('before_tool hook must be callable')
        self._before_tool.append(hook)
        return hook

    def after_tool(self, hook):
        """Register a result check run after every successful tool effect."""
        if not callable(hook):
            raise ValueError('after_tool hook must be callable')
        self._after_tool.append(hook)
        return hook

    def on(self, event_type, retry=None, *, policy=None):
        event_type = validate_name(event_type)
        retry = retry_policy(retry)
        policy = self.subscription_policies.get(event_type, policy or SubscriptionPolicy())
        if not isinstance(policy, SubscriptionPolicy):
            policy = SubscriptionPolicy.model_validate(policy)
        def decorate(fn):
            if event_type in self._handlers:
                raise ValueError(f'duplicate handler for {event_type}')
            self._handlers[event_type] = HandlerDefinition(callable=fn, retry=retry, policy=policy)
            return fn
        return decorate

    def on_request(self, fn=None, *, retry=None):
        policy = retry_policy(retry)
        def decorate(function):
            if self._request is not None:
                raise ValueError('duplicate request handler')
            self._request = HandlerDefinition(callable=function, retry=policy)
            return function
        return decorate(fn) if fn is not None else decorate
