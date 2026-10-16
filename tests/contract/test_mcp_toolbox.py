"""MCP toolboxes served as tool services, against real stdio and Streamable HTTP test servers."""
import asyncio
import json
import socket
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from uuid import uuid4
import psutil
import pytest
from tests.integration.test_python_runtime import eventually

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures' / 'mcp_server.py'
REPO = Path(__file__).resolve().parents[2]


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def server_processes(state):
    found = []
    for process in psutil.process_iter(['cmdline']):
        command = process.info['cmdline'] or []
        if str(FIXTURE) in command and str(state) in command:
            found.append(process)
    return found


class HttpMcpServer:
    """An MCP server this test owns, reached by URL: the toolbox never owns it."""

    def __init__(self, state, port):
        self.state, self.port, self.process = state, port, None
        self.url = f'http://127.0.0.1:{port}/mcp'

    async def start(self):
        self.process = subprocess.Popen([sys.executable, str(FIXTURE), str(self.state), 'http', str(self.port)],
                                        cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        async def listening():
            try:
                _, writer = await asyncio.open_connection('127.0.0.1', self.port)
            except OSError:
                return False
            writer.close()
            return True
        await eventually(listening, 20)

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(3)
            except subprocess.TimeoutExpired:  # an open change stream keeps the server alive
                self.process.kill()
                self.process.wait(10)


class McpHarness:
    def __init__(self, runtime, state):
        self.runtime, self.state = runtime, state
        self.worker = self.run = None
        self.agent_starts = 0

    def spec(self, **changes):
        from agentweave.definitions import McpToolboxSpec
        values = dict(namespace='support', transport='stdio', command=sys.executable,
                      args=[str(FIXTURE), str(self.state), 'stdio'], refresh_seconds=30)
        return McpToolboxSpec(**(values | changes))

    async def start(self, spec=None, **changes):
        from agentweave._runtime.tool_service import ToolWorker
        r = self.runtime
        self.worker = ToolWorker(spec or self.spec(**changes), r.config, r.store, r.bus, r.directory)
        self.run = asyncio.create_task(self.worker.run())
        await asyncio.wait_for(self.worker.ready.wait(), 30)
        return self.worker

    async def stop(self):
        if self.worker is not None:
            await self.worker.stop()
            await asyncio.gather(self.run, return_exceptions=True)
            self.worker = None

    def tools(self):
        return {name for name, instances in self.runtime.directory.snapshot().items() if instances[0].kind == 'tool'}

    async def tool(self, name, *, changed_from=None):
        async def live():
            found = self.runtime.directory.snapshot().get(name)
            return found if found and found[0].revision != changed_from else None
        return (await eventually(live, 30))[0]

    def set_tools(self, tools):
        (self.state / 'tools.json').write_text(json.dumps(tools))

    @property
    def effects(self):
        path = self.state / 'effects.log'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def call(self, snapshot, arguments, **changes):
        from agentweave.messages import ToolCall
        values = dict(tool_call_id=str(uuid4()), tool_id=snapshot.logical_id, revision=snapshot.revision,
                      arguments=arguments, caller='assistant',
                      deadline=self.runtime.store.clock.now() + timedelta(seconds=60))
        return ToolCall(**(values | changes))

    async def invoke(self, snapshot, arguments):
        from agentweave._runtime.tools import RemoteToolBinding
        r = self.runtime
        binding = RemoteToolBinding(snapshot, r.bus, r.store, directory=r.directory)
        return await asyncio.wait_for(binding.invoke(self.call(snapshot, arguments)), 30)

    async def agent(self, model, name='assistant'):
        from agentweave import Agent, Model, Tools
        self.agent_starts += 1
        return await self.runtime.start(Agent(name, model=Model.custom(model), tools=Tools.discover()))


@pytest.fixture
async def system_runtime(broker_process, tmp_path):
    """Real broker and journal on the system clock: a republished revision needs time to move on."""
    from uuid import UUID
    from agentweave._directory import LiveDirectory
    from agentweave.adapters.broker import BrokerBus
    from agentweave.adapters.storage.sqlite import SQLiteStore
    from agentweave.config import ProfileConfig
    from tests.integration.test_python_runtime import RuntimeHarness
    store = SQLiteStore(tmp_path / 'journal.sqlite3')
    await store.migrate()
    config = ProfileConfig(profile='dev', root=tmp_path, project_id=str(uuid4()),
        network_id='00000000-0000-0000-0000-000000000010', broker_url=broker_process.url,
        database_url='sqlite:///' + str(store.path), broker_epoch=str(UUID(broker_process.epoch)))
    bus = BrokerBus(config.broker_url, config.network_id, profile='dev', broker_epoch=config.broker_epoch)
    directory = LiveDirectory(bus, store, config.network_id, broker_epoch=config.broker_epoch)
    harness = RuntimeHarness(config, store, bus, directory)
    try:
        yield harness
    finally:
        await harness.close()
        await directory.close()
        await bus.close()
        await store.close()


@pytest.fixture
async def mcp_harness(system_runtime, tmp_path):
    state = tmp_path / 'mcp-state'
    state.mkdir()
    harness = McpHarness(system_runtime, state)
    try:
        yield harness
    finally:
        await harness.stop()


@pytest.fixture
async def http_mcp_server(tmp_path, mcp_harness):
    server = HttpMcpServer(mcp_harness.state, free_port())
    await server.start()
    try:
        yield server
    finally:
        server.stop()


async def test_a_stdio_toolbox_publishes_and_executes_its_mcp_tools(mcp_harness):
    h = mcp_harness
    await h.start()
    lookup = await h.tool('support.lookup')
    await h.tool('support.fail')
    await h.tool('support.big')
    assert lookup.input_schema['properties']['query']['type'] == 'string'
    assert lookup.description == 'Look up a support article.'
    assert await h.invoke(lookup, {'query': 'refund'}) == {'result': 'article:refund'}
    assert h.effects == [['lookup', 'refund']]


async def test_new_mcp_tool_appears_without_agent_restart(mcp_harness):
    from agentweave.testing import FakeModel, text_turn, tool_turn
    h = mcp_harness
    model = FakeModel([tool_turn('support_search', {'text': 'printer'}), text_turn('found it')])
    await h.agent(model)
    await h.start()  # the server notifies; polling is 30 s away
    await h.tool('support.lookup')
    h.set_tools({'search': 'Search the knowledge base.'})
    await h.tool('support.search')
    result = await asyncio.wait_for(h.runtime.ask('assistant', 'find printer help'), 30)
    assert (result.status, result.output) == ('succeeded', 'found it')
    assert 'support_search' in [t.tool_id for t in model.requests[0].tools]
    assert h.effects == [['search', 'printer']]
    assert h.agent_starts == 1


async def test_polling_finds_tools_a_silent_server_never_announces(mcp_harness):
    h = mcp_harness
    await h.start(args=[str(FIXTURE), str(h.state), 'stdio', '--silent'], refresh_seconds=0.3)
    await h.tool('support.lookup')
    h.set_tools({'search': 'Search the knowledge base.'})
    await h.tool('support.search')


async def test_a_changed_schema_rejects_an_old_call_before_dispatch(mcp_harness):
    from agentweave.errors import ToolRevisionChanged
    h = mcp_harness
    h.set_tools({'search': 'Search v1.'})
    await h.start()
    old = await h.tool('support.search')
    h.set_tools({'search': 'Search v2.'})
    new = await h.tool('support.search', changed_from=old.revision)
    assert new.description == 'Search v2.'
    with pytest.raises(ToolRevisionChanged):
        await h.invoke(old, {'text': 'stale'})
    assert h.effects == []
    assert await h.invoke(new, {'text': 'fresh'}) == {'result': 'search:fresh'}


async def test_a_removed_tool_is_withdrawn_and_the_allowlist_limits_publication(mcp_harness):
    h = mcp_harness
    h.set_tools({'search': 'Search.', 'admin': 'Dangerous.'})
    await h.start(allow=['lookup', 'search'], refresh_seconds=0.3)
    await h.tool('support.search')
    assert 'support.admin' not in h.tools() and 'support.big' not in h.tools()
    h.set_tools({'admin': 'Dangerous.'})

    async def withdrawn():
        return 'support.search' not in h.tools()
    await eventually(withdrawn, 30)
    assert 'support.admin' not in h.tools()


async def test_mcp_errors_and_oversized_results_follow_the_common_contract(mcp_harness):
    from agentweave.errors import ToolCallFailed
    h = mcp_harness
    await h.start()
    fail, big = await h.tool('support.fail'), await h.tool('support.big')
    with pytest.raises(ToolCallFailed) as failed:
        await h.invoke(fail, {'query': 'x'})
    assert failed.value.result.status == 'outcome_unknown'  # the tool may have acted before failing
    # Server-written error text stays out of the journal: it may echo a secret it received.
    assert failed.value.result.error['message'] == 'the MCP tool reported an error'
    with pytest.raises(ToolCallFailed) as oversized:
        await h.invoke(big, {'size': 1_200_000})
    assert oversized.value.result.status == 'outcome_unknown'
    assert oversized.value.result.error['type'] == 'InvalidToolOutput'


async def test_a_streamable_http_toolbox_reconnects_and_refetches_schemas(mcp_harness, http_mcp_server):
    h = mcp_harness
    await h.start(transport='http', url=http_mcp_server.url, command=None, args=[], refresh_seconds=0.3)
    lookup = await h.tool('support.lookup')
    assert await h.invoke(lookup, {'query': 'a'}) == {'result': 'article:a'}
    http_mcp_server.stop()
    h.set_tools({'search': 'Search.'})
    await http_mcp_server.start()  # same address, a new server process
    search = await h.tool('support.search')
    assert await h.invoke(search, {'text': 'b'}) == {'result': 'search:b'}
    assert await h.invoke(lookup, {'query': 'c'}) == {'result': 'article:c'}


async def test_stopping_closes_its_own_stdio_child_but_never_a_url_server(mcp_harness):
    h = mcp_harness
    await h.start()
    await h.tool('support.lookup')
    assert len(server_processes(h.state)) == 1
    await h.stop()

    async def gone():
        return not server_processes(h.state)
    await eventually(gone, 15)
    foreign = HttpMcpServer(h.state, free_port())
    await foreign.start()
    try:
        await h.start(transport='http', url=foreign.url, command=None, args=[])
        await h.tool('support.lookup')
        await h.stop()
        assert foreign.process.poll() is None  # not ours to stop
    finally:
        foreign.stop()


def test_http_headers_come_only_from_environment_references(monkeypatch):
    from agentweave.adapters.tools.mcp import McpBinding
    from agentweave.config import SecretRef
    from agentweave.definitions import McpToolboxSpec
    from agentweave.errors import ToolNotStarted
    with pytest.raises(ValueError):
        McpToolboxSpec(namespace='support', transport='http', url='https://mcp.example.test/mcp',
                       headers={'Authorization': 'Bearer plain'})
    with pytest.raises(ValueError):
        McpToolboxSpec(namespace='support', transport='stdio', url='https://mcp.example.test/mcp')
    spec = McpToolboxSpec(namespace='support', transport='http', url='https://mcp.example.test/mcp',
                          headers={'Authorization': SecretRef(env='SUPPORT_TOKEN')})
    monkeypatch.delenv('SUPPORT_TOKEN', raising=False)
    with pytest.raises(ToolNotStarted):
        McpBinding(spec).headers()
    monkeypatch.setenv('SUPPORT_TOKEN', 'Bearer s3cret')
    assert McpBinding(spec).headers() == {'Authorization': 'Bearer s3cret'}
    assert 's3cret' not in repr(McpBinding(spec)) and 's3cret' not in spec.model_dump_json()


def test_toml_configures_an_mcp_toolbox(tmp_path):
    import shutil
    from agentweave.definitions import McpToolboxSpec, load_definition
    shutil.copy(REPO / 'examples' / 'mcp_toolbox.toml', tmp_path / 'agentweave.toml')
    toolbox = load_definition('support', tmp_path)
    assert isinstance(toolbox, McpToolboxSpec) and toolbox.namespace == 'support'
    assert toolbox.transport == 'stdio' and toolbox.allow == ('lookup', 'search')


async def test_the_list_changing_between_turn_and_execute_is_rejected_then_refreshed(mcp_harness):
    from agentweave.testing import FakeModel, text_turn, tool_turn
    h = mcp_harness
    h.set_tools({'search': 'Search v1.'})
    gate, entered = asyncio.Event(), asyncio.Event()

    class GatedModel(FakeModel):
        async def generate(self, request):
            if not self.requests:
                entered.set()
                await gate.wait()  # the catalog of this turn is already frozen
            async for event in super().generate(request):
                yield event
    model = GatedModel([tool_turn('support_search', {'text': 'a'}),
                        tool_turn('support_search', {'text': 'b'}, call_id='call_2'), text_turn('done')])
    await h.start()
    old = await h.tool('support.search')
    await h.agent(model)
    asking = asyncio.create_task(h.runtime.ask('assistant', 'search'))
    await asyncio.wait_for(entered.wait(), 30)
    h.set_tools({'search': 'Search v2.'})
    new = await h.tool('support.search', changed_from=old.revision)
    gate.set()
    result = await asyncio.wait_for(asking, 30)
    assert (result.status, result.output) == ('succeeded', 'done')
    rejected = model.requests[1].messages[-1].tool_results[0]
    assert rejected['status'] == 'invalid' and rejected['error']['type'] == 'ToolRevisionChanged'
    assert [t.revision for t in model.requests[1].tools if t.tool_id == 'support_search'] == [new.revision]
    assert h.effects == [['search', 'b']]


async def test_the_cli_serves_an_mcp_toolbox_and_tool_stop_closes_its_server(tmp_path):
    from tests.integration.test_remote_tools import RemoteToolHarness, _cleanup
    from tests.e2e.test_dev_cli import ok
    root, state = tmp_path / 'tools', tmp_path / 'mcp-state'
    root.mkdir()
    state.mkdir()
    (root / 'agentweave.toml').write_text(f'''
[toolboxes.support]
kind = "mcp"
transport = "stdio"
command = {json.dumps(sys.executable)}
args = [{json.dumps(str(FIXTURE))}, {json.dumps(str(state))}, "stdio"]
''')
    h = RemoteToolHarness(root)
    try:
        await h.open()
        await asyncio.to_thread(ok, root, 'dev', 'run', 'support', '--detach')

        async def live():
            return h.resources.directory.snapshot().get('support.lookup')
        snapshot = (await eventually(live, 30))[0]
        call = h.call(snapshot, {'query': 'cli'})
        assert await asyncio.wait_for(h.binding(snapshot).invoke(call), 60) == {'result': 'article:cli'}
        assert len(server_processes(state)) == 1
        await asyncio.to_thread(ok, root, 'tool', 'stop', 'support')

        async def gone():
            return not server_processes(state)
        await eventually(gone, 15)
    finally:
        await h.close()
        await asyncio.to_thread(_cleanup, root)


def _stub_binding(**changes):
    from agentweave.adapters.tools.mcp import McpBinding
    from agentweave.definitions import McpToolboxSpec
    return McpBinding(McpToolboxSpec(**({'namespace': 'support', 'transport': 'stdio', 'command': 'server'} | changes)))


def _schema(revision):
    from agentweave.definitions.tool import ToolSchema
    return ToolSchema(tool_id='lookup', revision=revision, description='', input_schema={'type': 'object'},
                      output_schema={})


async def test_a_call_for_an_outdated_revision_never_reaches_the_server():
    from types import SimpleNamespace
    from agentweave.adapters.tools.mcp import McpTool
    from agentweave.errors import ToolRevisionChanged
    binding, sent = _stub_binding(), []

    class Session:
        async def call_tool(self, *args, **kwargs):
            sent.append(args)
    binding._client = SimpleNamespace(session=Session())
    binding._schemas = {'lookup': _schema('r2')}  # the server already moved on
    stale = McpTool('support.lookup', 'lookup', _schema('r1'), binding.spec, binding)
    with pytest.raises(ToolRevisionChanged):
        await stale.invoke(SimpleNamespace(arguments={}), None)
    assert sent == []


@pytest.mark.parametrize('code, expected', [(-32001, 'TimeoutError'), (-32000, 'ConnectionError'),
                                            (-32603, 'ConnectionError'), (-32602, 'ToolRequestRejected'),
                                            (-32601, 'ToolRequestRejected'), (-32020, 'ToolRevisionChanged')])
async def test_only_definite_protocol_refusals_count_as_not_run(code, expected):
    from types import SimpleNamespace
    from mcp import MCPError
    from agentweave.adapters.tools.mcp import McpTool
    binding = _stub_binding()

    class Session:
        async def call_tool(self, *args, **kwargs):
            raise MCPError(code, 'server says no')
    binding._client = SimpleNamespace(session=Session())
    binding._schemas = {'lookup': _schema('r1')}
    with pytest.raises(Exception) as raised:
        await McpTool('support.lookup', 'lookup', _schema('r1'), binding.spec, binding).invoke(
            SimpleNamespace(arguments={}), None)
    assert type(raised.value).__name__ == expected


async def test_a_hung_listing_times_out_instead_of_freezing_the_catalog():
    binding = _stub_binding(timeout=0.2)

    class Hung:
        async def list_tools(self, **kwargs):
            await asyncio.Event().wait()
    started = asyncio.get_running_loop().time()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(binding._refresh(Hung()), 5)
    assert asyncio.get_running_loop().time() - started < 2  # its own timeout, not the test's


async def test_a_flapping_server_is_reconnected_with_growing_delays(monkeypatch):
    import mcp_types
    binding, connections = _stub_binding(), []

    class Flapping:
        async def __aenter__(self):
            connections.append(asyncio.get_running_loop().time())
            return self

        async def __aexit__(self, *exc):
            return False

        async def list_tools(self, **kwargs):
            return mcp_types.ListToolsResult(tools=[])

        def listen(self, **kwargs):
            raise ConnectionError('the change stream ended at once')
    monkeypatch.setattr(binding, '_client_for', lambda stack: Flapping())
    await binding.connect()
    await asyncio.sleep(1.3)
    await binding.close()
    assert len(connections) <= 4  # 0, 0.2, 0.6 s and so on, not five times a second


def test_a_replica_leaves_calls_of_a_revision_another_replica_serves():
    from types import SimpleNamespace
    from agentweave._runtime.tool_service import ToolWorker
    live = {'support.lookup': (SimpleNamespace(fingerprint='new'), SimpleNamespace(fingerprint='old'))}
    worker = SimpleNamespace(directory=SimpleNamespace(snapshot=lambda: live))
    task = SimpleNamespace(agent_id='support.lookup', definition_hash='new')
    assert not ToolWorker._superseded(worker, task)
    assert ToolWorker._superseded(worker, SimpleNamespace(agent_id='support.lookup', definition_hash='gone'))
