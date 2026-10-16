"""Independent Python tool services: discovery, remote calls, revisions, receipts and replicas.

Tool services run as real CLI child processes; the caller shares only the
project journal and broker with them and never imports the tool's code.
"""
import asyncio
import json
import sqlite3
import sys
from datetime import timedelta
from pathlib import Path
from uuid import uuid4
import pytest
from tests.e2e.test_dev_cli import CREATED, cli, ok

TOOL = 'calculator.add'
ALIAS = 'calculator_add'
APP = '''from pathlib import Path
from agentweave import tool

EFFECTS = Path(__file__).with_name('effects.log')


@tool
def add(a: int, b: int) -> int:
    """{doc}"""
    with EFFECTS.open('a') as stream:
        stream.write(f'{{a}}+{{b}}\\n')
    return a + b
{extra}
'''


async def eventually(check, timeout=30):
    async with asyncio.timeout(timeout):
        while True:
            value = await check()
            if value:
                return value
            await asyncio.sleep(.05)


class RemoteToolHarness:
    """A calculator tool service in its own process and a caller sharing its journal."""

    def __init__(self, root):
        self.root, self.resources, self.snapshot = root, None, None
        self.workers = []

    def write(self, doc='Add two integers.', extra=''):
        (self.root/'app.py').write_text(APP.format(doc=doc, extra=extra))

    async def open(self):
        from agentweave._infrastructure.bootstrap import Bootstrap
        self.resources = await Bootstrap(self.root).ensure()

    @property
    def store(self):
        return self.resources.store

    async def launch(self, *flags):
        return await asyncio.to_thread(ok, self.root, 'dev', 'run', 'app:add', '--namespace', 'calculator',
                                       '--detach', *flags)

    async def stop_tool(self):
        await asyncio.to_thread(ok, self.root, 'tool', 'stop', TOOL)

    async def tool(self, *, instances=1, revision=None, changed_from=None):
        async def live():
            found = self.resources.directory.snapshot().get(TOOL, ())
            if len(found) < instances:
                return None
            if revision is not None and found[0].revision != revision:
                return None
            if changed_from is not None and found[0].revision == changed_from:
                return None
            return found
        return (await eventually(live))[0]

    async def update_schema(self):
        old = self.snapshot.revision
        await self.stop_tool()
        self.write(doc='Add two integers exactly.')
        await self.launch()
        self.snapshot = await self.tool(changed_from=old)

    def call(self, snapshot, arguments=None, *, caller='assistant', seconds=60, **changes):
        from agentweave.messages import ToolCall
        values = dict(tool_call_id=str(uuid4()), tool_id=snapshot.logical_id, revision=snapshot.revision,
                      arguments={'a': 2, 'b': 3} if arguments is None else arguments, caller=caller,
                      deadline=self.resources.store.clock.now() + timedelta(seconds=seconds))
        return ToolCall(**(values | changes))

    def binding(self, snapshot, *, directory=True):
        from agentweave._runtime.tools import RemoteToolBinding
        return RemoteToolBinding(snapshot, self.resources.bus, self.resources.store,
                                 directory=self.resources.directory if directory else None)

    async def invoke(self, snapshot, arguments=None, **changes):
        call = changes.pop('call', None) or self.call(snapshot, arguments, **changes)
        return await asyncio.wait_for(self.binding(snapshot).invoke(call), 60)

    @property
    def external_calls(self):
        effects = self.root/'effects.log'
        return len(effects.read_text().splitlines()) if effects.exists() else 0

    def rows(self, sql, values=()):
        with sqlite3.connect(self.store.path) as connection:
            return connection.execute(sql, values).fetchall()

    async def agent(self, definition):
        from agentweave import Worker
        r = self.resources
        worker = Worker(definition, r.config, r.store, r.bus, r.directory)
        run = asyncio.create_task(worker.run())
        self.workers.append((worker, run))
        await asyncio.wait_for(worker.ready.wait(), 20)
        async def announced():
            return bool(r.directory.snapshot().get(definition.name))
        await eventually(announced)
        return worker

    async def ask(self, name, value):
        from agentweave import Client
        r = self.resources
        client = Client(r.config, r.store, r.bus, r.directory)
        try:
            return await asyncio.wait_for(client.agent(name).ask(value), 60)
        finally:
            await client.close()

    async def close(self):
        for worker, run in self.workers:
            await worker.stop()
            await asyncio.gather(run, return_exceptions=True)
        if self.resources is not None:
            await self.resources.close()


def _cleanup(root):
    from agentweave._infrastructure.local import cleanup_epoch
    from agentweave._infrastructure.processes import ProcessRegistry
    if (root/'.agentweave/dev').exists():
        cli(root, 'tool', 'stop', TOOL, timeout=70)
        cli(root, 'infra', 'down', timeout=70)
    registry = ProcessRegistry(root)
    for record in CREATED.pop(root.resolve(), {}).values():
        if registry.running(record):
            assert registry.verify(record), 'unverifiable live test process: ' + str(record.pid)
            registry.terminate(record, grace=2)
        if record.logical_id == '$broker' and Path(record.epoch_dir).exists():
            cleanup_epoch(record, registry.paths)
        registry.remove(record)


@pytest.fixture
async def remote_tools(tmp_path):
    root = tmp_path/'tools'
    root.mkdir()
    harness = RemoteToolHarness(root)
    harness.write()
    try:
        await harness.open()
        yield harness
    finally:
        try:
            await harness.close()
        finally:
            await asyncio.to_thread(_cleanup, root)


@pytest.fixture
async def remote_tool_harness(remote_tools):
    await remote_tools.launch()
    remote_tools.snapshot = await remote_tools.tool()
    return remote_tools


def discovering_agent(model, name='assistant', **tools):
    from agentweave import Agent, Model, Tools
    return Agent(name, instructions='Use the calculator.', model=Model.custom(model),
                 tools=Tools.discover(**tools))


async def test_tool_published_after_agent_start_is_discovered_without_import_or_restart(remote_tools):
    from agentweave.testing import FakeModel, text_turn, tool_turn
    h = remote_tools
    model = FakeModel([text_turn('no calculator yet'), tool_turn(ALIAS, {'a': 2, 'b': 3}), text_turn('5')])
    await h.agent(discovering_agent(model))
    assert (await h.ask('assistant', 'add 2 and 3')).output == 'no calculator yet'
    assert model.requests[0].tools == []
    published = await h.launch()
    assert published['logical_id'] == TOOL
    snapshot = await h.tool()
    assert (snapshot.kind, snapshot.alias, snapshot.allowed_callers) == ('tool', ALIAS, None)
    result = await h.ask('assistant', 'add 2 and 3')
    assert (result.status, result.output) == ('succeeded', '5')
    offered = model.requests[1].tools
    assert [(t.tool_id, t.revision) for t in offered] == [(ALIAS, snapshot.revision)]
    assert offered[0].input_schema == snapshot.input_schema
    assert model.requests[2].messages[-1].tool_results[0]['output'] == 5
    assert h.external_calls == 1
    # The caller learned the tool from the catalog alone: no project module was imported here.
    assert not [name for name in sys.modules if name.startswith('_agentweave_project_')
                and str(h.root) in str(getattr(sys.modules[name], '__file__', '') or '')]


async def test_old_revision_does_not_execute(remote_tool_harness):
    from agentweave.errors import ToolRevisionChanged
    old = remote_tool_harness.snapshot
    await remote_tool_harness.update_schema()
    with pytest.raises(ToolRevisionChanged):
        await remote_tool_harness.invoke(old)
    assert remote_tool_harness.external_calls == 0
    # Without a catalog to compare, the executor itself rejects the old revision before any effect.
    with pytest.raises(ToolRevisionChanged):
        await asyncio.wait_for(remote_tool_harness.binding(old, directory=False).invoke(
            remote_tool_harness.call(old)), 60)
    assert remote_tool_harness.external_calls == 0
    assert await remote_tool_harness.invoke(remote_tool_harness.snapshot) == 5
    assert remote_tool_harness.external_calls == 1


async def test_duplicate_and_lost_receipt_deliveries_run_the_tool_once(remote_tool_harness):
    h = remote_tool_harness
    call = h.call(h.snapshot)
    assert await h.invoke(h.snapshot, call=call) == 5
    [(event_id, raw)] = h.rows("SELECT event_id, envelope FROM outbox WHERE json_extract(envelope,'$.kind')='tool_call' "
                               "AND json_extract(envelope,'$.payload.tool_call_id')=?", (call.tool_call_id,))
    async def delivered():
        return h.rows('SELECT delivered FROM outbox WHERE event_id=?', (event_id,)) == [(1,)]
    await eventually(delivered)  # the executor's receipt confirmed the call
    from agentweave.messages import decode
    envelope = decode(raw.encode())
    for _ in range(2):
        await h.resources.bus.publish(envelope)  # duplicate deliveries of one call
    # A lost receipt: the caller's outbox publishes the same call again.
    with sqlite3.connect(h.store.path) as connection:
        connection.execute("UPDATE outbox SET published_at=NULL, delivered=0, received_targets='[]', "
                           "retry_at='2000-01-01T00:00:00.000000+00:00', version=version+1 WHERE event_id=?", (event_id,))
    await eventually(delivered)
    await asyncio.sleep(.5)
    assert h.external_calls == 1
    assert await h.invoke(h.snapshot, call=call) == 5  # a repeated call returns the saved result
    assert h.external_calls == 1
    assert h.rows('SELECT status, attempts FROM tool_executions WHERE tool_call_id=?',
                  (call.tool_call_id,)) == [('succeeded', 1)]


async def test_executor_checks_allowed_callers_independently(remote_tools):
    from agentweave.errors import ToolCallFailed
    h = remote_tools
    h.write(extra="add = add.model_copy(update={'allowed_callers': ('auditor',)})")
    await h.launch()
    snapshot = await h.tool()
    assert snapshot.allowed_callers == ['auditor']
    # Even a caller that ignores the published restriction is refused by the executor.
    forged = snapshot.model_copy(update={'allowed_callers': None})
    with pytest.raises(ToolCallFailed) as refused:
        await h.invoke(forged, caller='assistant')
    assert refused.value.result.status == 'denied' and h.external_calls == 0
    assert await h.invoke(snapshot, caller='auditor') == 5
    assert h.external_calls == 1


async def test_catalog_offers_only_tools_both_sides_allow(remote_tools):
    from agentweave.testing import FakeModel, text_turn
    h = remote_tools
    h.write(extra="add = add.model_copy(update={'allowed_callers': ('auditor',)})")
    await h.launch()
    await h.tool()
    refused, auditor, outside = (FakeModel([text_turn('ok')]) for _ in range(3))
    await h.agent(discovering_agent(refused, 'assistant'))
    await h.agent(discovering_agent(auditor, 'auditor'))
    await h.agent(discovering_agent(outside, 'outsider', allow=['weather.*']))
    for name in ('assistant', 'auditor', 'outsider'):
        assert (await h.ask(name, 'hi')).output == 'ok'
    assert refused.requests[0].tools == [] and outside.requests[0].tools == []
    assert [t.tool_id for t in auditor.requests[0].tools] == [ALIAS]


async def test_two_replica_processes_share_calls_without_double_effects(remote_tool_harness):
    h = remote_tool_harness
    await h.launch('--replica')
    snapshot = await h.tool(instances=2)
    instances = {p.instance_id for p in h.resources.directory.snapshot()[TOOL]}
    assert len(instances) == 2
    calls = [h.call(snapshot, {'a': n, 'b': n}) for n in range(8)]
    results = await asyncio.gather(*(h.invoke(snapshot, call=call) for call in calls))
    assert results == [2 * n for n in range(8)]
    # Duplicates of every call reach the shared inbox group again.
    for (raw,) in h.rows("SELECT envelope FROM outbox WHERE json_extract(envelope,'$.kind')='tool_call'"):
        from agentweave.messages import decode
        await h.resources.bus.publish(decode(raw.encode()))
    await asyncio.sleep(1)
    assert h.external_calls == 8
    assert sorted(await asyncio.gather(*(h.invoke(snapshot, call=call) for call in calls))) == sorted(results)
    assert h.external_calls == 8
    owners = {row[0] for row in h.rows('SELECT instance_id FROM tool_executions')}
    assert owners <= instances


async def test_vanished_executor_is_reported_unavailable_within_the_deadline(remote_tool_harness):
    from agentweave.errors import ToolUnavailable
    h = remote_tool_harness
    snapshot = h.snapshot
    await h.stop_tool()
    async def withdrawn():
        return not h.resources.directory.snapshot().get(TOOL)
    await eventually(withdrawn)
    with pytest.raises(ToolUnavailable):
        await h.invoke(snapshot)  # the catalog has no live executor: nothing is sent
    assert h.rows("SELECT COUNT(*) FROM outbox WHERE json_extract(envelope,'$.kind')='tool_call'") == [(0,)]
    # Sent while no executor runs: the caller gives up at the deadline, not later.
    call = h.call(snapshot, seconds=2)
    started = asyncio.get_running_loop().time()
    with pytest.raises(ToolUnavailable):
        await asyncio.wait_for(h.binding(snapshot, directory=False).invoke(call), 10)
    assert asyncio.get_running_loop().time() - started < 6
    await h.launch()
    await h.tool()
    await asyncio.sleep(1.5)
    assert h.external_calls == 0  # an expired call is never executed later
    assert h.rows('SELECT COUNT(*) FROM tool_executions') == [(0,)]


async def test_schema_change_rejects_the_turn_and_the_next_step_gets_a_new_catalog(remote_tool_harness):
    from agentweave.testing import FakeModel, text_turn, tool_turn
    h = remote_tool_harness
    old = h.snapshot.revision
    gate, entered = asyncio.Event(), asyncio.Event()

    class GatedModel(FakeModel):
        async def generate(self, request):
            if not self.requests:
                entered.set()
                await gate.wait()  # the schema changes after this turn's catalog was frozen
            async for event in super().generate(request):
                yield event
    model = GatedModel([tool_turn(ALIAS, {'a': 2, 'b': 3}), tool_turn(ALIAS, {'a': 2, 'b': 3}, call_id='call_2'),
                        text_turn('5')])
    await h.agent(discovering_agent(model))
    asking = asyncio.create_task(h.ask('assistant', 'add'))
    await asyncio.wait_for(entered.wait(), 30)
    await h.update_schema()
    gate.set()
    result = await asking
    assert (result.status, result.output) == ('succeeded', '5')
    assert model.requests[0].tools[0].revision == old
    assert model.requests[1].tools[0].revision == h.snapshot.revision != old
    rejected = model.requests[1].messages[-1].tool_results[0]
    assert rejected['status'] == 'invalid' and rejected['error']['type'] == 'ToolRevisionChanged'
    assert h.external_calls == 1


async def test_tool_cli_lists_describes_and_stops_only_its_own_processes(remote_tool_harness):
    h = remote_tool_harness
    listed = await asyncio.to_thread(ok, h.root, 'tool', 'list')
    [entry] = [t for t in listed['tools'] if t['tool_id'] == TOOL]
    assert entry['revision'] == h.snapshot.revision and entry['instances'] == 1 and entry['owned']
    described = await asyncio.to_thread(ok, h.root, 'tool', 'describe', TOOL)
    assert described['input_schema'] == h.snapshot.input_schema
    assert described['output_schema'] == h.snapshot.output_schema
    missing = await asyncio.to_thread(cli, h.root, 'tool', 'stop', 'weather.forecast')
    assert missing.returncode == 2 and json.loads(missing.stdout)['error_code'] == 'tool_not_found'
    assert len(h.resources.directory.snapshot()[TOOL]) == 1
    stopped = await asyncio.to_thread(ok, h.root, 'tool', 'stop', TOOL)
    assert len(stopped['stopped']) == 1


async def test_a_step_catalog_is_frozen_by_its_first_writer(store, envelope_factory):
    event = envelope_factory()
    await store.admit(event, 'def-v1')
    lease = await store.claim(event.task_id, '00000000-0000-0000-0000-0000000000c1', 30)
    first = [{'tool_id': TOOL, 'revision': 'r1'}]
    assert await store.freeze_tool_catalog(lease, step=0, catalog=first) == first
    assert await store.freeze_tool_catalog(lease, step=0, catalog=[{'tool_id': TOOL, 'revision': 'r2'}]) == first
    assert await store.freeze_tool_catalog(lease, step=1, catalog=[]) == []


async def test_caller_policy_refuses_a_remote_call_before_sending(store, envelope_factory, fake_clock):
    from agentweave import Agent, Tools
    from agentweave._records import Participant
    from agentweave._runtime.tools import ToolExecutor
    from agentweave.messages import ToolCall
    agent = Agent('assistant', tools=Tools.discover(allow=['weather.*']))
    event = envelope_factory()
    await store.admit(event, 'def-v1')
    lease = await store.claim(event.task_id, '00000000-0000-0000-0000-0000000000c1', 30)
    task = await store.get_task(event.task_id)
    participant = Participant(network_id=event.network_id, logical_id=TOOL,
        instance_id=str(uuid4()), kind='tool', fingerprint='f', configuration_version='r1', broker_epoch='',
        heartbeat_at=fake_clock.now(), revision='r1', input_schema={'type': 'object'}, output_schema={})
    executor = ToolExecutor(agent, store, instance_id='00000000-0000-0000-0000-0000000000c1', clock=fake_clock)
    try:
        call = ToolCall(tool_call_id=str(uuid4()), tool_id=TOOL, revision='r1', arguments={'a': 1},
                        caller='assistant', deadline=task.deadline)
        result = await executor.execute_remote(call, participant, bus=None, lease=lease, task=task)
    finally:
        executor.close()
    assert result.status == 'denied'
    with sqlite3.connect(store.path) as connection:
        assert connection.execute('SELECT COUNT(*) FROM outbox').fetchone()[0] == 1  # only the receipt


def _stub_tool(network_id, now):
    from agentweave._records import Participant
    return Participant(network_id=network_id, logical_id=TOOL, instance_id=str(uuid4()), kind='tool',
                       fingerprint='f', configuration_version='r1', broker_epoch='', heartbeat_at=now,
                       revision='r1', input_schema={'type': 'object'}, output_schema={})


def _stub_call(clock, **changes):
    from agentweave.messages import ToolCall
    values = dict(tool_call_id=str(uuid4()), tool_id=TOOL, revision='r1', arguments={'a': 1},
                  caller='assistant', deadline=clock.now() + timedelta(seconds=60))
    return ToolCall(**(values | changes))


async def test_a_call_finishing_between_journal_reads_reports_its_result(fake_clock):
    # The row is read while running, then the executor settles it and commits the task.
    from agentweave._records import TaskRecord, ToolLease
    from agentweave._runtime.tools import RemoteToolBinding, remote_task_id
    call = _stub_call(fake_clock)
    running = ToolLease(tool_call_id=call.tool_call_id, token=1, attempt=1, status='running', arguments={'a': 1})
    done = running.model_copy(update={'status': 'succeeded', 'result': 2})
    task = TaskRecord(task_id=remote_task_id(call.tool_call_id), agent_id=TOOL, status='succeeded', version=3,
                      input=None, session_id=None, root_task_id=remote_task_id(call.tool_call_id),
                      definition_hash='f', deadline=call.deadline, result=None)

    class RacingStore:
        clock = fake_clock
        rows = [running, done]

        async def get_tool_execution(self, tool_call_id):
            return self.rows.pop(0) if len(self.rows) > 1 else self.rows[0]

        async def get_task(self, task_id):
            return task

        async def abandon_tool(self, tool_call_id):
            return self.rows[0]  # like the journal: an outcome recorded first is returned as is
    binding = RemoteToolBinding(_stub_tool('00000000-0000-0000-0000-000000000010', fake_clock.now()), None,
                                RacingStore())
    result = await asyncio.wait_for(binding._outcome(call), 5)
    assert (result.status, result.output) == ('succeeded', 2)


async def test_the_caller_settles_a_call_still_running_at_its_deadline_once(store, envelope_factory, fake_clock):
    from agentweave._runtime.tools import REMOTE_SETTLE_GRACE_SECONDS, RemoteToolBinding, remote_task_id
    from agentweave.errors import LeaseLost
    call = _stub_call(fake_clock, deadline=fake_clock.now() + timedelta(seconds=5))
    event = envelope_factory(kind='tool_call', sender='assistant', recipient=TOOL,
                             task_id=remote_task_id(call.tool_call_id), deadline=call.deadline,
                             payload=call.model_dump(mode='json'))
    await store.admit(event, 'f')
    lease = await store.claim(event.task_id, '00000000-0000-0000-0000-0000000000c1', 30)
    attempt = await store.reserve_tool(call, call.arguments, lease=lease,
                                       instance_id='00000000-0000-0000-0000-0000000000c1', retry_safe=False,
                                       max_attempts=1)
    binding = RemoteToolBinding(_stub_tool(event.network_id, fake_clock.now()), None, store)
    fake_clock.advance(5 + REMOTE_SETTLE_GRACE_SECONDS)
    result = await asyncio.wait_for(binding._outcome(call), 5)
    assert result.status == 'outcome_unknown'
    # The late executor cannot overwrite the answer the caller already gave.
    with pytest.raises(LeaseLost):
        await store.finish_tool(attempt, status='succeeded', result=2)
    assert (await store.get_tool_execution(call.tool_call_id)).status == 'outcome_unknown'


async def test_an_unadmitted_call_expires_at_its_expiry_not_its_deadline(store, fake_clock):
    from agentweave._runtime.tools import RemoteToolBinding
    call = _stub_call(fake_clock, expires_at=fake_clock.now() + timedelta(seconds=1))
    binding = RemoteToolBinding(_stub_tool('00000000-0000-0000-0000-000000000010', fake_clock.now()), None, store)
    pending = asyncio.create_task(binding.execute(call))
    await asyncio.sleep(.2)
    fake_clock.advance(2)
    result = await asyncio.wait_for(pending, 5)
    assert result.status == 'expired' and result.error['type'] == 'InputExpired'


async def test_a_transient_renewal_failure_does_not_abandon_a_running_call(monkeypatch, fake_clock):
    from types import SimpleNamespace
    from agentweave._records import Lease
    from agentweave._runtime import engine
    from agentweave._runtime.tool_service import ToolWorker
    from agentweave.errors import LeaseLost, StoreUnavailable
    monkeypatch.setattr(engine, 'LEASE_RENEW_SECONDS', 0.01)
    lease = Lease(task_id=str(uuid4()), instance_id=str(uuid4()), token=1,
                  expires_at=fake_clock.now() + timedelta(seconds=30))
    outcomes = [StoreUnavailable('busy'), lease, LeaseLost('replaced')]
    calls = []

    async def renew(current):
        calls.append(current)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    worker = SimpleNamespace(store=SimpleNamespace(renew=renew), clock=fake_clock)
    with pytest.raises(LeaseLost):
        await asyncio.wait_for(ToolWorker._renew(worker, lease), 5)
    assert len(calls) == 3  # the transient failure was retried, only the lost lease ends it


async def test_agent_commands_do_not_stop_tool_services(remote_tool_harness):
    h = remote_tool_harness
    stopped = await asyncio.to_thread(ok, h.root, 'agent', 'stop', TOOL)
    assert stopped['stopped'] == []
    listed = await asyncio.to_thread(ok, h.root, 'agent', 'list')
    assert all(agent['logical_id'] != TOOL for agent in listed['agents'])
    await asyncio.sleep(.5)
    assert len(h.resources.directory.snapshot()[TOOL]) == 1
    assert await h.invoke(h.snapshot) == 5
