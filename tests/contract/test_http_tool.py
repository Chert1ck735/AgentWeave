"""Declarative HTTP tools against a real local HTTP server: mappings, limits, failures, secrets."""
import asyncio
import json
import sqlite3
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4
import pytest
from tests.integration.test_remote_tools import eventually, remote_tools  # noqa: F401

SECRET = 'Bearer s3cret-value'
INSTANCE = '00000000-0000-0000-0000-0000000000d1'
WEATHER = {'temperature': 21, 'city': 'Paris'}


class HttpServer:
    """A minimal HTTP/1.1 server that records requests and replays scripted responses."""

    def __init__(self):
        self.requests, self.responses = [], []
        self.server = None

    async def start(self):
        self.server = await asyncio.start_server(self._serve, '127.0.0.1', 0)
        self.url = 'http://127.0.0.1:%d' % self.server.sockets[0].getsockname()[1]

    def respond(self, status=200, body=WEATHER, *, content_type='application/json', delay=0.0, raw=None):
        payload = raw if raw is not None else json.dumps(body).encode()
        self.responses.append((status, payload, content_type, delay))

    async def _serve(self, reader, writer):
        try:
            head = await reader.readuntil(b'\r\n\r\n')
            lines = head.decode('latin-1').split('\r\n')
            method, target, _ = lines[0].split(' ', 2)
            headers = {k.strip().lower(): v.strip() for k, v in (line.split(':', 1) for line in lines[1:] if ':' in line)}
            body = await reader.readexactly(int(headers.get('content-length', '0')))
            parts = urlsplit(target)
            self.requests.append({'method': method, 'path': parts.path, 'query': parse_qs(parts.query),
                                  'headers': headers, 'body': json.loads(body) if body else None})
            status, payload, content_type, delay = (self.responses.pop(0) if self.responses else
                                                    (200, json.dumps(WEATHER).encode(), 'application/json', 0))
            if delay:
                await asyncio.sleep(delay)
            writer.write(b'HTTP/1.1 %d X\r\nContent-Type: %s\r\nContent-Length: %d\r\nConnection: close\r\n\r\n'
                         % (status, content_type.encode(), len(payload)) + payload)
            await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()

    async def close(self):
        self.server.close()
        await self.server.wait_closed()


@pytest.fixture
async def http_server():
    server = HttpServer()
    await server.start()
    try:
        yield server
    finally:
        await server.close()


INPUT = {'type': 'object', 'properties': {'city': {'type': 'string'}, 'days': {'type': 'integer'}},
         'required': ['city']}
OUTPUT = {'type': 'object', 'properties': {'temperature': {'type': 'number'}}, 'required': ['temperature']}


def weather_spec(url, **changes):
    from agentweave.config import SecretRef
    from agentweave.definitions import Argument, HttpToolSpec, Literal
    values = dict(name='forecast', method='POST', url=url + '/forecast', description='Get a forecast.',
                  input_schema=INPUT, output_schema=OUTPUT,
                  query={'city': Argument(name='city'), 'units': Literal(value='metric')},
                  headers={'Authorization': SecretRef(env='WEATHER_TOKEN'), 'X-Client': Literal(value='agentweave')},
                  body={'days': Argument(name='days'), 'source': Literal(value='agent')})
    return HttpToolSpec(**(values | changes))


class HttpHarness:
    def __init__(self, server, store, clock, envelope_factory):
        self.server, self.store, self.clock, self.envelope_factory = server, store, clock, envelope_factory
        self.spec, self.task, self.lease = weather_spec(server.url), None, None

    @property
    def requests(self):
        return self.server.requests

    def tool_call(self, arguments, **changes):
        from agentweave.messages import ToolCall
        values = dict(tool_call_id=str(uuid4()), tool_id=self.spec.name, revision=self.spec.schema.revision,
                      arguments=arguments, caller='assistant', deadline=self.task.deadline)
        return ToolCall(**(values | changes))

    async def begin(self, **changes):
        if changes:
            self.spec = type(self.spec).model_validate(self.spec.model_dump(by_alias=True) | changes)
        event = self.envelope_factory()
        await self.store.admit(event, 'def-v1')
        self.lease = await self.store.claim(event.task_id, INSTANCE, 30)
        self.task = await self.store.get_task(event.task_id)
        return self

    async def call(self, arguments):
        """The binding alone, as the executor calls it with approved arguments."""
        from agentweave.adapters.tools.http import HttpBinding
        from agentweave.context import ToolContext
        call = self.tool_call(arguments)
        context = ToolContext(call, task_id=self.task.task_id, session_id=None, attempt=1, clock=self.clock)
        return await asyncio.wait_for(HttpBinding(self.spec).invoke(call, context), 10)

    async def execute(self, arguments, *, call=None):
        """The whole journaled path an agent's tool call takes."""
        from agentweave import Agent
        from agentweave._runtime.tools import ToolExecutor

        async def no_sleep(seconds):
            self.clock.advance(seconds)
        executor = ToolExecutor(Agent('assistant', tools=[self.spec]), self.store, instance_id=INSTANCE,
                                clock=self.clock, sleep=no_sleep, random=lambda: 0.0)
        try:
            return await asyncio.wait_for(executor.execute(call or self.tool_call(arguments), lease=self.lease,
                                                           task=self.task), 10)
        finally:
            executor.close()

    def journal(self):
        with sqlite3.connect(self.store.path) as connection:
            return connection.execute('SELECT * FROM tool_executions').fetchall()


@pytest.fixture
async def http_harness(http_server, store, fake_clock, envelope_factory, monkeypatch):
    monkeypatch.setenv('WEATHER_TOKEN', SECRET)
    return await HttpHarness(http_server, store, fake_clock, envelope_factory).begin()


async def test_http_address_cannot_be_changed_by_model(http_harness):
    from agentweave.errors import ToolArgumentsInvalid
    with pytest.raises(ToolArgumentsInvalid):
        await http_harness.call({"url": "https://unexpected.invalid"})
    assert http_harness.requests == []
    result = await http_harness.execute({'city': 'Paris', 'url': 'https://unexpected.invalid'})
    assert result.status == 'invalid' and http_harness.requests == [] and http_harness.journal() == []


async def test_mappings_fill_query_headers_and_body_from_arguments_literals_and_secrets(http_harness):
    assert await http_harness.call({'city': 'Paris', 'days': 3}) == WEATHER
    [request] = http_harness.requests
    assert (request['method'], request['path']) == ('POST', '/forecast')
    assert request['query'] == {'city': ['Paris'], 'units': ['metric']}
    assert request['headers']['authorization'] == SECRET and request['headers']['x-client'] == 'agentweave'
    assert request['body'] == {'days': 3, 'source': 'agent'}


async def test_an_absent_optional_argument_is_left_out(http_harness):
    await http_harness.call({'city': 'Paris'})
    assert http_harness.requests[0]['body'] == {'source': 'agent'}


async def test_a_missing_secret_stops_the_call_before_any_request(http_harness, monkeypatch):
    from agentweave.errors import ToolNotStarted
    monkeypatch.delenv('WEATHER_TOKEN')
    with pytest.raises(ToolNotStarted):
        await http_harness.call({'city': 'Paris'})
    assert http_harness.requests == []
    result = await http_harness.execute({'city': 'Paris'})
    assert result.status == 'failed' and result.error['type'] == 'ToolNotStarted'


async def test_an_oversized_response_is_refused_before_parsing(http_harness):
    from agentweave.errors import ToolResponseInvalid
    await http_harness.begin(max_response_bytes=64)
    http_harness.server.respond(body={'temperature': 21, 'padding': 'x' * 500})
    with pytest.raises(ToolResponseInvalid, match='exceeds'):
        await http_harness.call({'city': 'Paris'})


@pytest.mark.parametrize('content_type, raw', [('text/html', b'<html></html>'),
                                               ('application/json', b'{not json'),
                                               ('application/json', b'{"t": NaN}')])
async def test_a_response_that_is_not_strict_json_is_refused(http_harness, content_type, raw):
    from agentweave.errors import ToolResponseInvalid
    http_harness.server.respond(content_type=content_type, raw=raw)
    with pytest.raises(ToolResponseInvalid):
        await http_harness.call({'city': 'Paris'})


async def test_a_client_error_is_a_known_refusal_and_a_server_error_is_unknown(http_harness):
    from agentweave.errors import ToolRequestRejected
    http_harness.server.respond(404, {'error': 'no such city'})
    with pytest.raises(ToolRequestRejected) as rejected:
        await http_harness.call({'city': 'Atlantis'})
    assert rejected.value.status == 404
    http_harness.server.respond(422, {'error': 'bad'})
    refused = await http_harness.execute({'city': 'Atlantis'})
    assert refused.status == 'failed' and refused.error['type'] == 'ToolRequestRejected'
    http_harness.server.respond(500, {'error': 'boom'})
    unknown = await http_harness.execute({'city': 'Paris'})
    assert unknown.status == 'outcome_unknown'  # the server may have acted before failing


async def test_a_retry_safe_call_keeps_one_idempotency_key_across_attempts(http_harness):
    await http_harness.begin(idempotency_header='Idempotency-Key',
                             retry={'safe': True, 'min_delay': 1, 'max_delay': 1})
    http_harness.server.respond(503, {'error': 'busy'})
    http_harness.server.respond(200, WEATHER)
    call = http_harness.tool_call({'city': 'Paris'})
    result = await http_harness.execute(None, call=call)
    assert result.status == 'succeeded' and result.output == WEATHER
    keys = [request['headers']['idempotency-key'] for request in http_harness.requests]
    assert keys == [call.tool_call_id, call.tool_call_id]


async def test_the_timeout_bounds_a_slow_endpoint_and_an_unanswered_effect_is_unknown(http_harness):
    await http_harness.begin(timeout=0.2)
    http_harness.server.respond(delay=2)
    result = await http_harness.execute({'city': 'Paris'})
    assert result.status == 'outcome_unknown' and len(http_harness.requests) == 1


async def test_an_output_outside_the_declared_schema_is_unknown(http_harness):
    http_harness.server.respond(body={'temperature': 'warm'})
    result = await http_harness.execute({'city': 'Paris'})
    assert result.status == 'outcome_unknown' and result.error['type'] == 'InvalidToolOutput'


async def test_secrets_stay_out_of_definitions_journal_and_logs(http_harness, caplog):
    from agentweave.definitions import fingerprint
    await http_harness.begin(query={'city': {'name': 'city'}, 'key': {'env': 'WEATHER_TOKEN'}})
    http_harness.server.respond(500, {'error': 'boom'})
    with caplog.at_level('DEBUG'):
        result = await http_harness.execute({'city': 'Paris'})
    assert result.status == 'outcome_unknown'
    assert http_harness.requests[0]['query']['key'] == [SECRET]  # the endpoint did get it
    spec = http_harness.spec
    for text in (repr(spec), spec.model_dump_json(), json.dumps(http_harness.journal()), caplog.text,
                 json.dumps(result.model_dump(mode='json'))):
        assert 's3cret' not in text
    assert fingerprint(spec)


async def test_an_agent_uses_a_configured_http_tool_without_python_code(http_server, store, fake_clock,
                                                                        envelope_factory, monkeypatch):
    from agentweave import Agent, Model
    from agentweave._runtime.steps import ModelLoop
    from agentweave._runtime.tools import ToolExecutor
    from agentweave.testing import FakeModel, text_turn, tool_turn
    monkeypatch.setenv('WEATHER_TOKEN', SECRET)
    model = FakeModel([tool_turn('forecast', {'city': 'Paris', 'url': 'x'}), tool_turn('forecast', {'city': 'Paris'},
                       call_id='call_2'), text_turn('21 degrees')])
    agent = Agent('assistant', model=Model.custom(model), tools=[weather_spec(http_server.url)])
    event = envelope_factory()
    await store.admit(event, 'def-v1')
    lease = await store.claim(event.task_id, INSTANCE, 30)
    task = await store.get_task(event.task_id)
    tools = ToolExecutor(agent, store, instance_id=INSTANCE, clock=fake_clock)
    try:
        outcome = await ModelLoop(agent, store, model, instance_id=INSTANCE, clock=fake_clock, tools=tools).run(task, lease)
    finally:
        tools.close()
    assert (outcome.status, outcome.output) == ('succeeded', '21 degrees')
    assert model.requests[0].tools[0].input_schema['additionalProperties'] is False
    assert model.requests[1].messages[-1].tool_results[0]['status'] == 'invalid'  # 'url' is not a parameter
    assert model.requests[2].messages[-1].tool_results[0]['output'] == WEATHER
    assert len(http_server.requests) == 1


def test_toml_configures_agent_and_toolbox_http_tools(tmp_path):
    import shutil
    from pathlib import Path
    from agentweave.definitions import HttpToolSpec, ToolboxDefinition, load_definition
    shutil.copy(Path(__file__).resolve().parents[2] / 'examples' / 'http_tool.toml', tmp_path / 'agentweave.toml')
    toolbox = load_definition('weather', tmp_path)
    assert isinstance(toolbox, ToolboxDefinition)
    forecast = toolbox.bindings['forecast']
    assert isinstance(forecast, HttpToolSpec) and forecast.name == 'forecast'
    agent = load_definition('forecaster', tmp_path)
    assert [type(t) for t in agent.tools] == [HttpToolSpec]


def test_http_specs_reject_dynamic_targets_and_plain_credentials():
    from agentweave.definitions import Argument, HttpToolSpec, Literal
    base = dict(name='forecast', method='POST', url='https://api.example.test/forecast', input_schema=INPUT)
    with pytest.raises(ValueError):
        HttpToolSpec(**base | {'url': 'file:///etc/passwd'})
    with pytest.raises(ValueError):
        HttpToolSpec(**base | {'headers': {'Authorization': Literal(value='Bearer plain')}})
    with pytest.raises(ValueError):
        HttpToolSpec(**base | {'query': {'q': Argument(name='undeclared')}})
    with pytest.raises(ValueError):
        HttpToolSpec(**base | {'method': 'GET', 'body': {'days': Argument(name='days')}})
    with pytest.raises(ValueError):
        HttpToolSpec(**base | {'input_schema': INPUT | {'additionalProperties': True}})
    assert HttpToolSpec(**base).schema.input_schema['additionalProperties'] is False


async def test_a_toolbox_service_publishes_an_http_tool_to_agents(http_server, remote_tools, monkeypatch):
    from tests.e2e import test_dev_cli
    from tests.e2e.test_dev_cli import ok
    h = remote_tools
    monkeypatch.setitem(test_dev_cli.ENV, 'WEATHER_TOKEN', SECRET)  # the tool service process's environment
    (h.root / 'agentweave.toml').write_text(f'''
[toolboxes.weather.bindings.forecast]
kind = "http"
method = "POST"
url = "{http_server.url}/forecast"
description = "Get a forecast."
input_schema = {{ type = "object", properties = {{ city = {{ type = "string" }} }}, required = ["city"] }}
query = {{ city = {{ name = "city" }} }}
headers = {{ Authorization = {{ env = "WEATHER_TOKEN" }} }}
''')
    await asyncio.to_thread(ok, h.root, 'dev', 'run', 'weather', '--detach')
    try:
        async def live():
            return h.resources.directory.snapshot().get('weather.forecast')
        snapshot = (await eventually(live))[0]
        call = h.call(snapshot, {'city': 'Paris'})
        assert await asyncio.wait_for(h.binding(snapshot).invoke(call), 60) == WEATHER
        [request] = http_server.requests
        assert request['query']['city'] == ['Paris'] and request['headers']['authorization'] == SECRET
    finally:
        await asyncio.to_thread(ok, h.root, 'tool', 'stop', 'weather')


@pytest.mark.parametrize('status', [302, 303, 307])
async def test_a_redirect_after_a_possible_effect_is_unknown_not_a_clean_failure(http_harness, status):
    http_harness.server.respond(status, {'moved': True})
    result = await http_harness.execute({'city': 'Paris'})
    assert result.status == 'outcome_unknown' and len(http_harness.requests) == 1


@pytest.mark.parametrize('status', [408, 429])
async def test_retry_later_answers_retry_a_safe_call_and_fail_cleanly_otherwise(http_harness, status):
    http_harness.server.respond(status, {'error': 'later'})
    refused = await http_harness.execute({'city': 'Paris'})
    assert refused.status == 'failed'  # the endpoint did not process it
    await http_harness.begin(retry={'safe': True, 'min_delay': 1, 'max_delay': 1})
    http_harness.server.respond(status, {'error': 'later'})
    http_harness.server.respond(200, WEATHER)
    assert (await http_harness.execute({'city': 'Paris'})).status == 'succeeded'


async def test_compressed_responses_are_neither_requested_nor_accepted(http_harness):
    from agentweave.errors import ToolResponseInvalid
    http_harness.server.respond(raw=b'\x1f\x8b\x08\x00compressed', content_type='application/json')
    http_harness.server.responses[-1] = (200, b'\x1f\x8b\x08\x00compressed', 'application/json\r\nContent-Encoding: gzip', 0)
    with pytest.raises(ToolResponseInvalid, match='encoded'):
        await http_harness.call({'city': 'Paris'})
    assert http_harness.requests[0]['headers']['accept-encoding'] == 'identity'


async def test_the_spec_timeout_bounds_the_whole_attempt(http_harness):
    from agentweave._runtime.tools import ToolExecutor
    # A slow-drip endpoint answers each read in time; the attempt limit still ends it.
    await http_harness.begin(timeout=45)
    assert ToolExecutor._attempt_limit(http_harness.spec) == 45  # not capped by retry.timeout (30)


@pytest.mark.parametrize('header', ['Host', 'X-Forwarded-For', 'X-HTTP-Method-Override', 'Content-Length',
                                    'Transfer-Encoding', 'Forwarded'])
def test_the_model_cannot_steer_routing_or_framing_headers(header):
    from agentweave.definitions import Argument, HttpToolSpec
    with pytest.raises(ValueError):
        HttpToolSpec(name='forecast', method='POST', url='https://api.example.test/forecast', input_schema=INPUT,
                     headers={header: Argument(name='city')})


async def test_a_header_value_with_line_breaks_is_refused_before_sending(http_harness):
    from agentweave.definitions import Argument
    await http_harness.begin(headers={'X-City': {'name': 'city'}, 'Authorization': {'env': 'WEATHER_TOKEN'}})
    result = await http_harness.execute({'city': 'Paris\r\nX-Injected: 1'})
    assert result.status == 'failed' and result.error['type'] == 'ToolArgumentsInvalid'
    assert http_harness.requests == []
    assert Argument


@pytest.mark.parametrize('raw', [b'{"temperature": 1, "temperature": 2}', b'{"temperature": 1e999}',
                                 b'[' * 100000 + b']' * 100000])
async def test_duplicate_keys_infinities_and_deep_nesting_are_not_strict_json(http_harness, raw):
    from agentweave.errors import ToolResponseInvalid
    http_harness.server.respond(raw=raw)
    with pytest.raises(ToolResponseInvalid):
        await http_harness.call({'city': 'Paris'})


async def test_a_declared_body_is_sent_even_when_every_argument_is_absent(http_harness):
    await http_harness.begin(body={'days': {'name': 'days'}})
    await http_harness.call({'city': 'Paris'})
    assert http_harness.requests[0]['body'] == {}
