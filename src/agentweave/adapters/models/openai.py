"""OpenAI Responses API adapter: streamed text and complete function calls.

The SDK never retries (max_retries=0) and never runs tools; function call
arguments are taken only from the final response, never from deltas.
"""
import json
from collections.abc import AsyncIterator
from typing import Any
from ...errors import ModelError
from ...messages import Message
from ...models import (ModelCapabilities, ModelEvent, ModelRequest, ModelSpec, ModelTurn,
                       ProposedToolCall, TextDelta, TurnComplete)
from ...providers import check_capabilities

_COMPACT = {'separators': (',', ':'), 'sort_keys': True, 'ensure_ascii': False}
# Portable fields describe the turn for any adapter; these items replay it exactly for OpenAI.
_NATIVE_OUTPUT = 'openai_output'


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, **_COMPACT)


class OpenAIAdapter:
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
        instructions = '\n\n'.join(m.content for m in request.messages
                                   if m.role == 'system' and isinstance(m.content, str) and m.content)
        # Nothing is stored by the provider, so opaque reasoning state travels with the history.
        arguments = {'model': request.model_id, 'input': self._input(request.messages), 'stream': True,
                     'store': False, 'include': ['reasoning.encrypted_content']}
        if instructions:
            arguments['instructions'] = instructions
        if request.tools:
            arguments['tools'] = [{'type': 'function', 'name': tool.tool_id, 'description': tool.description,
                                   'parameters': tool.input_schema, 'strict': False} for tool in request.tools]
        options = self.spec.options
        if options.get('temperature') is not None:
            arguments['temperature'] = options['temperature']
        if options.get('max_tokens') is not None:
            arguments['max_output_tokens'] = options['max_tokens']
        return arguments

    @staticmethod
    def _input(messages: list[Message]) -> list[dict]:
        items = []
        for message in messages:
            native = message.adapter_metadata.get(_NATIVE_OUTPUT) if message.role == 'assistant' else None
            if native is not None:
                items.extend(native)  # this adapter's own turn: replay it with its opaque items
                continue
            if message.role in ('user', 'assistant') and message.content is not None:
                items.append({'role': message.role, 'content': _text(message.content)})
            for call in message.tool_calls:
                items.append({'type': 'function_call', 'call_id': call['id'], 'name': call['name'],
                              'arguments': json.dumps(call['arguments'], **_COMPACT)})
            for result in message.tool_results:
                body = {key: value for key, value in result.items() if key != 'id'}
                items.append({'type': 'function_call_output', 'call_id': result['id'],
                              'output': json.dumps(body, **_COMPACT)})
        return items

    async def generate(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.validate_request(request)
        import openai
        try:
            stream = await self.client.responses.create(**self._arguments(request))
            async for event in stream:
                if event.type == 'response.output_text.delta':
                    yield TextDelta(text=event.delta)
                elif event.type == 'response.completed':
                    yield TurnComplete(turn=self._turn(event.response))
                    return
                elif event.type in ('response.failed', 'response.incomplete'):
                    raise _ended_early(event.type, event.response)
                elif event.type == 'error':
                    raise ModelError('unavailable', 'the provider reported a stream error',
                                     retryable=True, usage_unknown=True)
        except openai.APIError as error:
            raise _model_error(error) from error
        raise ModelError('unavailable', 'stream ended without a completed response',
                         retryable=True, usage_unknown=True)

    @staticmethod
    def _turn(response) -> ModelTurn:
        texts, calls, native = [], [], []
        for item in response.output or []:
            if item.type == 'message':
                text = ''.join(part.text for part in item.content if getattr(part, 'type', '') == 'output_text')
                texts.append(text)
                native.append({'role': 'assistant', 'content': text})
            elif item.type == 'function_call':
                calls.append(_proposed(item))
                native.append({'type': 'function_call', 'call_id': item.call_id, 'name': item.name,
                               'arguments': item.arguments})
            elif item.type == 'reasoning':
                # Opaque continuation state; its content is never logged or shown.
                dumped = item.model_dump(mode='json', exclude_none=True)
                native.append({key: dumped[key] for key in ('type', 'id', 'summary', 'encrypted_content')
                               if key in dumped})
        return ModelTurn(
            messages=[Message(role='assistant', content=''.join(texts) or None,
                              tool_calls=[{'id': c.provider_call_id, 'name': c.name, 'arguments': c.arguments}
                                          for c in calls],
                              adapter_metadata={_NATIVE_OUTPUT: native})],
            tool_calls=calls, adapter_metadata={'response_id': response.id}, usage=_usage(response))


def _usage(response) -> dict:
    usage = getattr(response, 'usage', None)
    return {'input_tokens': getattr(usage, 'input_tokens', None),
            'output_tokens': getattr(usage, 'output_tokens', None)}


def _ended_early(event_type: str, response) -> ModelError:
    """A token limit or content filter repeats on retry; only server errors may pass."""
    usage = _usage(response)
    known = None not in usage.values()
    if event_type == 'response.incomplete':
        reason = getattr(getattr(response, 'incomplete_details', None), 'reason', None) or 'unknown'
        return ModelError('incomplete', f'the response stopped early: {reason}',
                          usage=usage if known else None, usage_unknown=not known)
    code = getattr(getattr(response, 'error', None), 'code', None)
    retryable = code in ('server_error', 'rate_limit_exceeded')
    return ModelError('unavailable' if retryable else 'invalid_request', f'the response failed: {code}',
                      retryable=retryable, usage=usage if known else None, usage_unknown=not known)


def _proposed(item) -> ProposedToolCall:
    try:
        arguments = json.loads(item.arguments or '{}')
        if not isinstance(arguments, dict):
            raise ValueError('arguments must be a JSON object')
    except ValueError as error:
        return ProposedToolCall(provider_call_id=item.call_id, name=item.name,
                                arguments_error=f'unparseable arguments: {error}')
    return ProposedToolCall(provider_call_id=item.call_id, name=item.name, arguments=arguments)


def _model_error(error) -> ModelError:
    """Map SDK errors to stable kinds; the provider message is not echoed."""
    import openai
    if isinstance(error, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return ModelError('auth', 'the provider rejected the credentials')
    if isinstance(error, openai.RateLimitError):
        if getattr(error, 'code', None) == 'insufficient_quota':
            return ModelError('quota', 'the provider account has no remaining quota')
        return ModelError('rate_limit', 'the provider is rate limiting requests', retryable=True)
    if isinstance(error, openai.InternalServerError):
        return ModelError('unavailable', 'the provider failed to serve the request', retryable=True)
    if isinstance(error, openai.APITimeoutError):
        return ModelError('timeout', 'the provider request timed out', retryable=True, usage_unknown=True)
    if isinstance(error, openai.APIConnectionError):
        return ModelError('unavailable', 'the provider connection failed', retryable=True, usage_unknown=True)
    if getattr(error, 'status_code', None) in (408, 409):
        return ModelError('unavailable', 'the provider asked to retry the request', retryable=True,
                          usage_unknown=True)
    return ModelError('invalid_request', 'the provider rejected the request')
