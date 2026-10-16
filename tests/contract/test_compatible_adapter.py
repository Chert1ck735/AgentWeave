"""OpenAI-compatible Chat Completions endpoints through a mock transport: no key, no network."""
import json
from datetime import datetime, timedelta, timezone
import httpx
import pytest


def chunk(delta=None, finish=None, usage=None, index=0):
    body = {'id': 'chat_1', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'local-model',
            'choices': [] if delta is None and finish is None else
            [{'index': index, 'delta': delta or {}, 'finish_reason': finish}]}
    if usage is not None:
        body['usage'] = usage
    return body


def stream(chunks):
    return b''.join(f'data: {json.dumps(item)}\n\n'.encode() for item in chunks) + b'data: [DONE]\n\n'


class ChatServer:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    def handler(self, request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=stream(reply))

    def body(self, index=-1):
        return json.loads(self.requests[index].content)


def adapter_for(server, **spec_changes):
    from agentweave import Model
    from agentweave.adapters.models.compatible import CompatibleAdapter
    spec = Model.compatible('local-model', base_url='http://llm.local:8080/v1', **spec_changes)
    return CompatibleAdapter(spec, 'test-key', http_client=httpx.AsyncClient(transport=httpx.MockTransport(server.handler)))


def request(messages=None, tools=()):
    from agentweave.messages import Message
    from agentweave.models import ModelRequest
    return ModelRequest(messages=messages or [Message(role='user', content='hi')], tools=list(tools),
                        model_id='local-model', deadline=datetime.now(timezone.utc) + timedelta(seconds=60))


def add_schema():
    from agentweave.definitions.tool import ToolSchema
    return ToolSchema(tool_id='add', revision='r1', description='Add.', output_schema={},
                      input_schema={'type': 'object', 'properties': {'a': {'type': 'integer'}}})


async def collect(adapter, req):
    from agentweave.models import TextDelta, TurnComplete
    deltas, turns = [], []
    async for event in adapter.generate(req):
        (deltas if isinstance(event, TextDelta) else turns).append(event)
    assert len(turns) == 1 and isinstance(turns[0], TurnComplete)
    return [d.text for d in deltas], turns[0].turn


async def test_streamed_text_becomes_deltas_and_one_turn_with_usage():
    server = ChatServer([[chunk({'role': 'assistant', 'content': 'Hel'}), chunk({'content': 'lo'}),
                          chunk(finish='stop'), chunk(usage={'prompt_tokens': 7, 'completion_tokens': 2})]])
    adapter = adapter_for(server)
    deltas, turn = await collect(adapter, request())
    assert deltas == ['Hel', 'lo'] and turn.messages[0].content == 'Hello'
    assert turn.usage == {'input_tokens': 7, 'output_tokens': 2}
    sent = server.requests[0]
    assert (sent.url.host, sent.url.path) == ('llm.local', '/v1/chat/completions')  # never the Responses API
    assert server.body()['stream'] is True and server.body()['stream_options'] == {'include_usage': True}
    await adapter.close()


async def test_indexed_tool_call_fragments_are_assembled_per_call():
    server = ChatServer([[
        chunk({'tool_calls': [{'index': 0, 'id': 'call_a', 'type': 'function',
                               'function': {'name': 'add', 'arguments': '{"a"'}}]}),
        chunk({'tool_calls': [{'index': 1, 'id': 'call_b', 'type': 'function',
                               'function': {'name': 'add', 'arguments': '{"a": 2}'}}]}),
        chunk({'tool_calls': [{'index': 0, 'function': {'arguments': ': 1}'}}]}),
        chunk(finish='tool_calls')]])
    _, turn = await collect(adapter_for(server), request(tools=[add_schema()]))
    assert [(c.provider_call_id, c.name, c.arguments) for c in turn.tool_calls] == [
        ('call_a', 'add', {'a': 1}), ('call_b', 'add', {'a': 2})]
    assert turn.usage == {'input_tokens': None, 'output_tokens': None}  # not reported: unknown, not zero
    assert server.body()['tools'][0]['function']['name'] == 'add'


async def test_unparseable_tool_arguments_are_reported_not_guessed():
    server = ChatServer([[chunk({'tool_calls': [{'index': 0, 'id': 'c', 'function': {'name': 'add', 'arguments': '{oops'}}]}),
                          chunk(finish='tool_calls')]])
    _, turn = await collect(adapter_for(server), request(tools=[add_schema()]))
    assert turn.tool_calls[0].arguments_error


async def test_history_with_closed_tool_pairs_translates_to_chat_messages():
    from agentweave.messages import Message
    server = ChatServer([[chunk({'content': 'done'}), chunk(finish='stop')]])
    history = [Message(role='system', content='Be brief.'), Message(role='user', content='add 1'),
               Message(role='assistant', tool_calls=[{'id': 'c1', 'name': 'add', 'arguments': {'a': 1}}]),
               Message(role='tool', tool_results=[{'id': 'c1', 'status': 'succeeded', 'output': 1, 'error': None}])]
    await collect(adapter_for(server), request(history, tools=[add_schema()]))
    messages = server.body()['messages']
    assert [m['role'] for m in messages] == ['system', 'user', 'assistant', 'tool']
    assert messages[2]['tool_calls'][0] == {'id': 'c1', 'type': 'function',
                                            'function': {'name': 'add', 'arguments': '{"a":1}'}}
    assert messages[3]['tool_call_id'] == 'c1' and json.loads(messages[3]['content'])['output'] == 1


async def test_a_model_without_tools_refuses_tool_requests_before_sending():
    from agentweave.errors import ModelError
    from agentweave.models import ModelCapabilities
    server = ChatServer([])
    adapter = adapter_for(server, capabilities=ModelCapabilities(tools=False))
    with pytest.raises(ModelError) as refused:
        adapter.validate_request(request(tools=[add_schema()]))
    assert refused.value.kind == 'unsupported' and server.requests == []


def test_an_endpoint_without_streaming_is_a_configuration_error():
    from agentweave import Model
    from agentweave.errors import ConfigError
    from agentweave.models import ModelCapabilities
    from agentweave.providers import check_available
    spec = Model.compatible('local-model', base_url='http://llm.local/v1', capabilities=ModelCapabilities(streaming=False))
    with pytest.raises(ConfigError):
        check_available(spec, {'OPENAI_API_KEY': 'k'})


def test_a_compatible_spec_creates_its_adapter_from_the_key_reference():
    from agentweave import Model
    from agentweave.adapters.models.compatible import CompatibleAdapter
    from agentweave.errors import ConfigError
    from agentweave.providers import create_adapter
    spec = Model.compatible('local-model', base_url='http://llm.local/v1', api_key_env='LOCAL_KEY')
    with pytest.raises(ConfigError):
        create_adapter(spec, {})
    assert isinstance(create_adapter(spec, {'LOCAL_KEY': 'k'}), CompatibleAdapter)


@pytest.mark.parametrize('reply, kind, retryable', [
    (httpx.Response(401, json={'error': {'message': 'no'}}), 'auth', False),
    (httpx.Response(429, json={'error': {'message': 'slow down'}}), 'rate_limit', True),
    (httpx.Response(500, json={'error': {'message': 'boom'}}), 'unavailable', True),
    (httpx.ReadTimeout('late'), 'timeout', True)])
async def test_endpoint_errors_map_to_model_errors_without_sdk_retries(reply, kind, retryable):
    from agentweave.errors import ModelError
    server = ChatServer([reply, reply, reply])
    with pytest.raises(ModelError) as failed:
        await collect(adapter_for(server), request())
    assert (failed.value.kind, failed.value.retryable) == (kind, retryable)
    assert len(server.requests) == 1  # the SDK never retries on its own


async def test_a_length_cut_answer_is_incomplete_and_not_retried():
    from agentweave.errors import ModelError
    server = ChatServer([[chunk({'content': 'partial'}), chunk(finish='length'),
                          chunk(usage={'prompt_tokens': 3, 'completion_tokens': 9})]])
    with pytest.raises(ModelError) as cut:
        await collect(adapter_for(server), request())
    assert (cut.value.kind, cut.value.retryable) == ('incomplete', False)
    assert cut.value.usage == {'input_tokens': 3, 'output_tokens': 9}


async def test_options_are_forwarded():
    server = ChatServer([[chunk({'content': 'x'}), chunk(finish='stop')]])
    await collect(adapter_for(server, options={'temperature': 0.2, 'max_tokens': 50}), request())
    assert (server.body()['temperature'], server.body()['max_tokens']) == (0.2, 50)


@pytest.mark.parametrize('indexes', [(None, None), (0, 0)])
async def test_calls_without_distinct_indexes_are_split_by_their_ids(indexes):
    first, second = indexes
    server = ChatServer([[
        chunk({'tool_calls': [{'index': first, 'id': 'call_a', 'function': {'name': 'add', 'arguments': '{"a": 1}'}}]}),
        chunk({'tool_calls': [{'index': second, 'id': 'call_b', 'function': {'name': 'add', 'arguments': '{"a": 2}'}}]}),
        chunk(finish='tool_calls')]])
    _, turn = await collect(adapter_for(server), request(tools=[add_schema()]))
    assert [(c.provider_call_id, c.arguments) for c in turn.tool_calls] == [('call_a', {'a': 1}), ('call_b', {'a': 2})]


async def test_calls_without_ids_get_distinct_ids():
    server = ChatServer([[
        chunk({'tool_calls': [{'index': 0, 'function': {'name': 'add', 'arguments': '{"a": 1}'}}]}),
        chunk({'tool_calls': [{'index': 1, 'function': {'name': 'add', 'arguments': '{"a": 2}'}}]}),
        chunk(finish='tool_calls')]])
    _, turn = await collect(adapter_for(server), request(tools=[add_schema()]))
    ids = [c.provider_call_id for c in turn.tool_calls]
    assert len(set(ids)) == 2


async def test_a_connection_dropped_mid_answer_is_retryable_with_unknown_usage():
    from agentweave.errors import ModelError

    async def broken():
        yield f'data: {json.dumps(chunk({"content": "Hel"}))}\n\n'.encode()
        raise httpx.ReadError('connection reset')
    server = ChatServer([httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=broken())])
    with pytest.raises(ModelError) as dropped:
        await collect(adapter_for(server), request())
    assert dropped.value.retryable and dropped.value.usage_unknown


async def test_an_error_after_the_stream_started_leaves_usage_unknown():
    from agentweave.errors import ModelError
    body = (f'data: {json.dumps(chunk({"content": "Hel"}))}\n\n'
            f'data: {json.dumps({"error": {"message": "upstream", "type": "server_error"}})}\n\n').encode()
    server = ChatServer([httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=body)])
    with pytest.raises(ModelError) as failed:
        await collect(adapter_for(server), request())
    assert failed.value.usage_unknown


async def test_an_unknown_finish_reason_is_not_a_success():
    from agentweave.errors import ModelError
    server = ChatServer([[chunk({'content': 'half'}), chunk(finish='error')]])
    with pytest.raises(ModelError) as failed:
        await collect(adapter_for(server), request())
    assert failed.value.kind == 'unavailable' and failed.value.retryable
