"""OpenAI-compatible endpoints over Chat Completions: streamed text and indexed tool calls.

Compatible servers implement Chat Completions; the Responses API is never
assumed. Tool call fragments are joined per index (a new call ID starts a new call)
and parsed only when the turn ends. The SDK never retries (max_retries=0) and never runs tools.
"""
import json
from collections.abc import AsyncIterator
from typing import Any
import httpx
from ...errors import ModelError
from ...messages import Message
from ...models import ModelCapabilities, ModelEvent, ModelRequest, ModelSpec, ModelTurn, ProposedToolCall, TextDelta, TurnComplete
from ...providers import check_capabilities
from .openai import _model_error

_COMPACT = {'separators': (',', ':'), 'sort_keys': True, 'ensure_ascii': False}
_FINISHED = frozenset(('stop', 'tool_calls', 'function_call'))


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, **_COMPACT)


def chat_messages(messages: list[Message]) -> list[dict]:
    """The portable history as Chat Completions messages; tool pairs keep their call IDs."""
    translated = []
    for message in messages:
        if message.role == 'tool':
            for result in message.tool_results:
                body = {key: value for key, value in result.items() if key != 'id'}
                translated.append({'role': 'tool', 'tool_call_id': result['id'], 'content': json.dumps(body, **_COMPACT)})
            continue
        item = {'role': message.role, 'content': None if message.content is None else _text(message.content)}
        if message.role == 'assistant' and message.tool_calls:
            item['tool_calls'] = [{'id': call['id'], 'type': 'function',
                                   'function': {'name': call['name'], 'arguments': json.dumps(call['arguments'], **_COMPACT)}}
                                  for call in message.tool_calls]
        translated.append(item)
    return translated


class CompatibleAdapter:
    def __init__(self, spec: ModelSpec, secret: str, *, http_client=None) -> None:
        from openai import AsyncOpenAI
        self.spec = spec
        self.client = AsyncOpenAI(api_key=secret, base_url=spec.base_url, max_retries=0, http_client=http_client)

    def capabilities(self) -> ModelCapabilities:
        return self.spec.capabilities

    def validate_request(self, request: ModelRequest) -> None:
        check_capabilities(self.spec.capabilities, request)

    async def close(self) -> None:
        await self.client.close()

    def _arguments(self, request: ModelRequest) -> dict:
        arguments = {'model': request.model_id, 'messages': chat_messages(request.messages), 'stream': True,
                     'stream_options': {'include_usage': True}}
        if request.tools:
            arguments['tools'] = [{'type': 'function', 'function': {'name': tool.tool_id, 'description': tool.description,
                                                                    'parameters': tool.input_schema}}
                                  for tool in request.tools]
        for option in ('temperature', 'max_tokens'):
            if self.spec.options.get(option) is not None:
                arguments[option] = self.spec.options[option]
        return arguments

    async def generate(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.validate_request(request)
        import openai
        texts, calls, by_index, finish, usage, started = [], [], {}, None, None, False
        try:
            stream = await self.client.chat.completions.create(**self._arguments(request))
            async for chunk in stream:
                started = True
                if getattr(chunk, 'usage', None) is not None:
                    usage = {'input_tokens': chunk.usage.prompt_tokens, 'output_tokens': chunk.usage.completion_tokens}
                for choice in chunk.choices or []:
                    delta = choice.delta
                    if delta is not None and delta.content:
                        texts.append(delta.content)
                        yield TextDelta(text=delta.content)
                    for fragment in (delta.tool_calls or []) if delta is not None else []:
                        call = _open_call(calls, by_index, fragment)
                        if fragment.function is not None:
                            call['name'] = fragment.function.name or call['name']
                            call['arguments'] += fragment.function.arguments or ''
                    finish = choice.finish_reason or finish
        except openai.APIError as error:
            failure = _model_error(error)
            failure.usage_unknown = failure.usage_unknown or started  # tokens may already be billed
            raise failure from error
        except (httpx.TransportError, ValueError) as error:
            raise ModelError('unavailable', f'the stream broke off ({type(error).__name__})', retryable=True,
                             usage_unknown=True) from error
        known = usage or {'input_tokens': None, 'output_tokens': None}  # not reported: unknown, not zero
        if finish == 'length':
            raise ModelError('incomplete', 'the answer stopped at the token limit', usage=usage, usage_unknown=usage is None)
        if finish == 'content_filter':
            raise ModelError('invalid_request', 'the endpoint filtered the answer', usage=usage, usage_unknown=usage is None)
        if finish not in _FINISHED:
            # None or an unknown reason such as 'error': the answer is not known to be complete.
            raise ModelError('unavailable', 'the stream ended without a completed answer', retryable=True, usage_unknown=True)
        proposed = [_proposed(call, position) for position, call in enumerate(calls)]
        text = ''.join(texts) or None
        yield TurnComplete(turn=ModelTurn(
            messages=[Message(role='assistant', content=text,
                              tool_calls=[{'id': c.provider_call_id, 'name': c.name, 'arguments': c.arguments} for c in proposed])],
            tool_calls=proposed, usage=known))


def _open_call(calls: list, by_index: dict, fragment) -> dict:
    """The call a fragment continues; a new id, or a new index, starts the next call."""
    key = fragment.index
    current = by_index.get(key) if key is not None else (calls[-1] if calls else None)
    if current is None or (fragment.id and current['id'] and fragment.id != current['id']):
        current = {'id': fragment.id, 'name': '', 'arguments': ''}
        calls.append(current)
        if key is not None:
            by_index[key] = current
    current['id'] = current['id'] or fragment.id
    return current


def _proposed(call: dict, position: int) -> ProposedToolCall:
    identity = call['id'] or f'call_{position}'  # unique even for two calls of one tool
    try:
        arguments = json.loads(call['arguments'] or '{}')
        if not isinstance(arguments, dict):
            raise ValueError('arguments must be a JSON object')
    except ValueError as error:
        return ProposedToolCall(provider_call_id=identity, name=call['name'], arguments_error=f'unparseable arguments: {error}')
    return ProposedToolCall(provider_call_id=identity, name=call['name'], arguments=arguments)


__all__ = ['CompatibleAdapter', 'chat_messages']
