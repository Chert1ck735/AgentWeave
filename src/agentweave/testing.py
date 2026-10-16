"""Deterministic test doubles for agents: a scripted model and turn builders."""
import asyncio
from collections.abc import AsyncIterator, Callable
from .errors import ModelError
from .messages import JSONValue, Message
from .models import (ModelCapabilities, ModelEvent, ModelRequest, ModelTurn, ProposedToolCall,
                     TextDelta, TurnComplete)

DEFAULT_USAGE = {'input_tokens': 1, 'output_tokens': 1}


def text_turn(text: str, *, usage: dict | None = None) -> ModelTurn:
    """A final assistant answer."""
    return ModelTurn(messages=[Message(role='assistant', content=text)],
                     usage=DEFAULT_USAGE if usage is None else usage)


def tool_turn(name: str, arguments: dict[str, JSONValue], *, call_id: str = 'call_1',
              usage: dict | None = None) -> ModelTurn:
    """An assistant turn proposing one tool call."""
    call = ProposedToolCall(provider_call_id=call_id, name=name, arguments=arguments)
    return ModelTurn(messages=[Message(role='assistant', tool_calls=[{'id': call_id, 'name': name,
                                                                      'arguments': arguments}])],
                     tool_calls=[call], usage=DEFAULT_USAGE if usage is None else usage)


class FakeModel:
    """A model adapter that replays a script of turns or errors, with no network.

    It records every request and the highest number of overlapping requests,
    so tests can assert what the runtime sent and that pools were respected.
    """

    def __init__(self, script: list[ModelTurn | Exception], *, capabilities: ModelCapabilities | None = None,
                 latency: float = 0, on_generate: Callable[[ModelRequest], None] | None = None) -> None:
        self.script = list(script)
        self._capabilities = capabilities or ModelCapabilities()
        self.latency, self.on_generate = latency, on_generate
        self.requests: list[ModelRequest] = []
        self.active = self.peak_concurrency = 0

    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def validate_request(self, request: ModelRequest) -> None:
        if request.tools and not self._capabilities.tools:
            raise ModelError('unsupported', 'this model does not support tools')

    async def generate(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        self.active += 1
        self.peak_concurrency = max(self.peak_concurrency, self.active)
        try:
            if self.on_generate is not None:
                self.on_generate(request)
            if self.latency:
                await asyncio.sleep(self.latency)
            assert self.script, 'FakeModel script is exhausted'
            step = self.script.pop(0)
            if isinstance(step, Exception):
                raise step
            for message in step.messages:
                if isinstance(message.content, str) and message.content:
                    yield TextDelta(text=message.content)
            yield TurnComplete(turn=step)
        finally:
            self.active -= 1

    async def close(self) -> None:
        return None


async def check_model_adapter(adapter, *, request: ModelRequest | None = None) -> ModelTurn:
    """Check the ModelAdapter contract any adapter must keep; return the turn it produced.

    Capabilities are ModelCapabilities, validate_request accepts the request, a
    tool request is refused when tools are unsupported, generate yields only
    TextDelta progress followed by exactly one final TurnComplete, and close
    can be awaited. Raises AssertionError naming the broken rule.
    """
    from datetime import datetime, timedelta, timezone
    from .definitions.tool import ToolSchema
    capabilities = adapter.capabilities()
    assert isinstance(capabilities, ModelCapabilities), 'capabilities() must return ModelCapabilities'
    request = request or ModelRequest(messages=[Message(role='user', content='hello')], tools=[], model_id='check',
                                      deadline=datetime.now(timezone.utc) + timedelta(seconds=60))
    adapter.validate_request(request)
    if not capabilities.tools:
        probe = request.model_copy(update={'tools': [ToolSchema(tool_id='probe', revision='r', description='',
                                                                input_schema={'type': 'object'}, output_schema={})]})
        try:
            adapter.validate_request(probe)
        except ModelError:
            pass
        else:
            raise AssertionError('validate_request must refuse tools the model does not support')
    events = [event async for event in adapter.generate(request)]
    finals = [event for event in events if isinstance(event, TurnComplete)]
    assert len(finals) == 1 and events[-1] is finals[0], 'generate must end with exactly one TurnComplete'
    assert all(isinstance(event, TextDelta) for event in events[:-1]), 'only TextDelta may precede TurnComplete'
    await adapter.close()
    return finals[0].turn


__all__ = ['FakeModel', 'check_model_adapter', 'text_turn', 'tool_turn']
