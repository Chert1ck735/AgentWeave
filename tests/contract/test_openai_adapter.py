"""OpenAI Responses adapter over a mock transport: translation, streaming, errors."""
from datetime import datetime, timedelta, timezone
import pytest
from tests.fixtures.provider_http import OpenAIServer, error, text_response, tool_response


def request(messages=None, tools=None):
    from agentweave.messages import Message
    from agentweave.models import ModelRequest
    return ModelRequest(messages=messages or [Message(role='system', content='Be brief.'),
                                              Message(role='user', content='hi')],
                        tools=tools or [], model_id='test-model',
                        deadline=datetime.now(timezone.utc) + timedelta(seconds=60))


def adapter(server, **options):
    from agentweave.adapters.models.openai import OpenAIAdapter
    from agentweave.models import Model
    return OpenAIAdapter(Model.openai('test-model', options=options), 'sk-test', http_client=server.client())


async def collect(stream):
    return [event async for event in stream]


def add_schema():
    from agentweave import tool

    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
    return add.schema


async def test_streams_text_deltas_then_a_complete_turn_with_usage():
    from agentweave.models import TextDelta, TurnComplete
    server = OpenAIServer([text_response('Hello there')])
    events = await collect(adapter(server).generate(request()))
    assert ''.join(e.text for e in events if isinstance(e, TextDelta)) == 'Hello there'
    turn = events[-1].turn
    assert isinstance(events[-1], TurnComplete)
    assert turn.messages[-1].role == 'assistant' and turn.messages[-1].content == 'Hello there'
    assert turn.usage == {'input_tokens': 7, 'output_tokens': 3}
    body = server.requests[0]
    assert body['stream'] is True and body['model'] == 'test-model'
    assert body['instructions'] == 'Be brief.'
    assert body['input'] == [{'role': 'user', 'content': 'hi'}]


async def test_function_call_arguments_are_parsed_only_from_the_final_event():
    server = OpenAIServer([tool_response('add', '{"a": 12, "b": 7}')])
    events = await collect(adapter(server).generate(request(tools=[add_schema()])))
    turn = events[-1].turn
    assert [(c.provider_call_id, c.name, c.arguments) for c in turn.tool_calls] == [('call_1', 'add', {'a': 12, 'b': 7})]
    tool = server.requests[0]['tools'][0]
    assert tool['type'] == 'function' and tool['name'] == 'add' and tool['parameters']['required'] == ['a', 'b']


async def test_unparseable_function_arguments_are_reported_not_executed():
    server = OpenAIServer([tool_response('add', '{"a": 12, ')])
    turn = (await collect(adapter(server).generate(request(tools=[add_schema()]))))[-1].turn
    assert turn.tool_calls[0].arguments == {} and turn.tool_calls[0].arguments_error


async def test_history_round_trips_closed_tool_pairs():
    from agentweave.messages import Message
    history = [Message(role='user', content='12 + 7?'),
               Message(role='assistant', content=None,
                       tool_calls=[{'id': 'call_1', 'name': 'add', 'arguments': {'a': 12, 'b': 7}}]),
               Message(role='tool', tool_results=[{'id': 'call_1', 'output': 19, 'status': 'succeeded'}])]
    server = OpenAIServer([text_response('19')])
    await collect(adapter(server).generate(request(messages=history, tools=[add_schema()])))
    items = server.requests[0]['input']
    assert items[0] == {'role': 'user', 'content': '12 + 7?'}
    assert items[1]['type'] == 'function_call' and items[1]['call_id'] == 'call_1'
    assert items[2] == {'type': 'function_call_output', 'call_id': 'call_1',
                        'output': '{"output":19,"status":"succeeded"}'}


async def test_missing_usage_is_unknown_not_zero():
    server = OpenAIServer([text_response('ok', usage=None)])
    turn = (await collect(adapter(server).generate(request())))[-1].turn
    assert turn.usage == {'input_tokens': None, 'output_tokens': None}


@pytest.mark.parametrize('status,kind,retryable', [(401, 'auth', False), (403, 'auth', False),
                                                   (404, 'invalid_request', False), (400, 'invalid_request', False),
                                                   (429, 'rate_limit', True), (500, 'unavailable', True)])
async def test_provider_errors_map_to_model_error_without_sdk_retries(status, kind, retryable):
    from agentweave.errors import ModelError
    server = OpenAIServer([error(status, 'secret-free message'), text_response('should not be requested')])
    with pytest.raises(ModelError) as raised:
        await collect(adapter(server).generate(request()))
    assert raised.value.kind == kind and raised.value.retryable is retryable
    assert len(server.requests) == 1  # max_retries=0: the runtime owns retries


async def test_options_are_forwarded_and_unknown_tools_capability_is_rejected():
    from agentweave.errors import ModelError
    server = OpenAIServer([text_response('ok')])
    await collect(adapter(server, temperature=0.2, max_tokens=50).generate(request()))
    assert server.requests[0]['temperature'] == 0.2 and server.requests[0]['max_output_tokens'] == 50
    from agentweave.adapters.models.openai import OpenAIAdapter
    from agentweave.models import Model, ModelCapabilities
    spec = Model.openai('test-model').model_copy(update={'capabilities': ModelCapabilities(tools=False)})
    no_tools = OpenAIAdapter(spec, 'sk-test', http_client=OpenAIServer([]).client())
    with pytest.raises(ModelError) as raised:
        no_tools.validate_request(request(tools=[add_schema()]))
    assert raised.value.kind == 'unsupported' and not raised.value.retryable


def _incomplete(reason, usage=None):
    response = {'id': 'resp_1', 'object': 'response', 'created_at': 0, 'model': 'test-model',
                'status': 'incomplete', 'output': [], 'parallel_tool_calls': True, 'tool_choice': 'auto',
                'tools': [], 'incomplete_details': {'reason': reason}}
    if usage:
        response['usage'] = usage | {'total_tokens': sum(usage.values()), 'input_tokens_details': {'cached_tokens': 0},
                                     'output_tokens_details': {'reasoning_tokens': 0}}
    return [{'type': 'response.incomplete', 'response': response, 'sequence_number': 1}]


async def test_a_token_limited_incomplete_response_is_not_retried_and_keeps_usage():
    from agentweave.errors import ModelError
    server = OpenAIServer([_incomplete('max_output_tokens', {'input_tokens': 9, 'output_tokens': 50})])
    with pytest.raises(ModelError) as raised:
        await collect(adapter(server).generate(request()))
    assert raised.value.kind == 'incomplete' and not raised.value.retryable
    assert raised.value.usage == {'input_tokens': 9, 'output_tokens': 50}


async def test_a_failed_server_response_is_retryable():
    from agentweave.errors import ModelError
    failed = _incomplete('server_error')
    failed[0]['type'] = 'response.failed'
    failed[0]['response']['status'] = 'failed'
    failed[0]['response']['error'] = {'code': 'server_error', 'message': 'internal'}
    with pytest.raises(ModelError) as raised:
        await collect(adapter(OpenAIServer([failed])).generate(request()))
    assert raised.value.kind == 'unavailable' and raised.value.retryable


async def test_exhausted_quota_is_not_retried():
    import httpx
    from agentweave.errors import ModelError
    quota = httpx.Response(429, json={'error': {'message': 'quota', 'type': 'insufficient_quota',
                                                'code': 'insufficient_quota'}})
    with pytest.raises(ModelError) as raised:
        await collect(adapter(OpenAIServer([quota])).generate(request()))
    assert raised.value.kind == 'quota' and not raised.value.retryable


async def test_reasoning_continuation_is_kept_and_sent_back():
    from tests.fixtures.provider_http import _completed
    reasoning = {'id': 'rs_1', 'type': 'reasoning', 'summary': [], 'encrypted_content': 'opaque-state'}
    call = {'id': 'fc_1', 'type': 'function_call', 'call_id': 'call_1', 'name': 'add',
            'arguments': '{"a": 1, "b": 2}', 'status': 'completed'}
    server = OpenAIServer([[_completed([reasoning, call], {'input_tokens': 5, 'output_tokens': 5})],
                           text_response('3')])
    model = adapter(server)
    first = (await collect(model.generate(request(tools=[add_schema()]))))[-1].turn
    assert server.requests[0]['include'] == ['reasoning.encrypted_content']
    from agentweave.messages import Message
    history = [Message(role='user', content='1 + 2?'), *first.messages,
               Message(role='tool', tool_results=[{'id': 'call_1', 'output': 3, 'status': 'succeeded'}])]
    await collect(model.generate(request(messages=history, tools=[add_schema()])))
    items = server.requests[1]['input']
    assert {'type': 'reasoning', 'id': 'rs_1', 'summary': [], 'encrypted_content': 'opaque-state'} in items
    assert [i.get('type') for i in items].index('reasoning') < [i.get('type') for i in items].index('function_call')
