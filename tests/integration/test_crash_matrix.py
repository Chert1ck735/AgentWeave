"""Crash matrix: a worker process dies right after each durable step and a restart finishes the work.

Each scenario arms one failpoint in a real worker process (tests/fixtures/crash.py),
waits for the process to die, moves the shared clock past the lease and starts a
fresh worker. The test then checks the task result and how often the external
effect really happened.
"""
import asyncio
import json
import multiprocessing
import sqlite3
from uuid import UUID, uuid4
import pytest
from tests.fixtures.crash import CRASH_EXIT, effects_of, run_crash_worker
from tests.integration.test_python_runtime import eventually

NETWORK = '00000000-0000-0000-0000-000000000010'
LEASE_EXPIRY_SECONDS = 31
START_SECONDS = 30


class CrashHarness:
    def __init__(self, broker, store, clock, tmp_path):
        self.broker, self.store, self.clock, self.tmp_path = broker, store, clock, tmp_path
        self.context = multiprocessing.get_context('spawn')
        self.ready, self.stop = self.context.Queue(), self.context.Event()
        self.processes, self.effects = [], tmp_path / 'effects.jsonl'
        self.clock_path = tmp_path / 'clock.txt'
        self.clock_path.write_text(clock.now().isoformat())
        self.project_id = str(uuid4())
        self.bus = self.directory = self.client = self.config = None
        self.limits = {}

    async def connect(self, **limits):
        """Client, bus and directory for the broker's current epoch; limits apply to later workers."""
        from agentweave import Client
        from agentweave.adapters.broker import BrokerBus
        from agentweave.config import Limits, ProfileConfig
        self.limits = limits or self.limits
        from agentweave._directory import LiveDirectory
        await self._disconnect()
        self.config = ProfileConfig(profile='dev', root=self.tmp_path, project_id=self.project_id, network_id=NETWORK,
                                    broker_url=self.broker.url, database_url='sqlite:///' + str(self.store.path),
                                    broker_epoch=str(UUID(self.broker.epoch)), limits=Limits(**self.limits))
        self.bus = BrokerBus(self.config.broker_url, NETWORK, profile='dev', broker_epoch=self.config.broker_epoch)
        self.directory = LiveDirectory(self.bus, self.store, NETWORK, broker_epoch=self.config.broker_epoch,
                                       clock=self.clock)
        await self.directory.start()
        self.client = Client(self.config, self.store, self.bus, self.directory)

    async def _disconnect(self):
        if self.client is not None:
            await self.client.close()
        if self.directory is not None:
            await self.directory.close()
        if self.bus is not None:
            await self.bus.close()
        self.client = self.directory = self.bus = None

    async def start(self, kind, failpoint=None, **options):
        options = {'effects': str(self.effects), 'marker': str(self.tmp_path / 'fired'), 'failpoint': failpoint} | options
        process = self.context.Process(target=run_crash_worker, args=(
            self.config.model_dump_json(), str(self.store.path), str(self.clock_path), kind, json.dumps(options),
            self.ready, self.stop))
        process.start()
        self.processes.append(process)
        outcome = await asyncio.to_thread(self.ready.get, True, START_SECONDS)
        assert outcome[0] == 'ready', outcome
        name = options.get('name', kind)

        async def visible():
            return any(p.instance_id == outcome[2] for p in self.directory.snapshot().get(name, ()))
        await eventually(visible)
        return process

    async def crashed(self, process):
        await asyncio.to_thread(process.join, START_SECONDS)
        assert process.exitcode == CRASH_EXIT, f'the failpoint did not fire (exit {process.exitcode})'

    async def kill(self, process):
        process.kill()
        await asyncio.to_thread(process.join, 10)

    def advance(self, seconds=LEASE_EXPIRY_SECONDS):
        self.clock.advance(seconds)
        self.clock_path.write_text(self.clock.now().isoformat())

    async def submit(self, agent, payload='go'):
        return (await self.client.agent(agent).submit(payload)).task_id

    async def result(self, task_id, seconds=60):
        return await asyncio.wait_for(self.client.result(task_id), seconds)

    def external_effects(self, kind):
        return len(effects_of(self.effects, kind))

    async def close(self):
        self.stop.set()
        for process in self.processes:
            await asyncio.to_thread(process.join, 15)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 5)
        self.ready.close()
        await self._disconnect()


@pytest.fixture
async def crash_harness(broker_process, store, fake_clock, tmp_path):
    harness = CrashHarness(broker_process, store, fake_clock, tmp_path)
    await harness.connect()
    try:
        yield harness
    finally:
        await harness.close()


def journal(store, sql, *values):
    with sqlite3.connect(store.path) as connection:
        return connection.execute(sql, values).fetchall()


async def test_committed_task_survives_broker_and_worker_loss(crash_harness):
    h = crash_harness
    worker = await h.start('loop', model_seconds=5)  # the model is still answering when everything dies
    task_id = await h.submit('loop')

    async def admitted():
        task = await h.store.get_task(task_id)
        return task is not None and task.status == 'running'
    await eventually(admitted)
    await h.kill(worker)
    h.broker.restart()  # a new epoch: every unadmitted message of the old one is gone
    await h.connect()
    h.advance()
    await h.start('loop')
    assert (await h.result(task_id)).status == 'succeeded'
    assert h.external_effects('tool') == 1


@pytest.mark.parametrize('failpoint', ['input_commit', 'model_response', 'step_commit', 'outbox_publish', 'in_model'])
async def test_a_crash_after_each_durable_step_finishes_once(crash_harness, failpoint):
    h = crash_harness
    worker = await h.start('loop', failpoint)
    task_id = await h.submit('loop')
    await h.crashed(worker)
    h.advance()
    await h.start('loop')
    result = await h.result(task_id)
    assert (result.status, result.output) == ('succeeded', 'done')
    assert h.external_effects('tool') == 1
    if failpoint == 'outbox_publish':
        async def delivered():
            rows = journal(h.store, "SELECT published_at FROM outbox WHERE json_extract(envelope,'$.kind')='request'")
            return bool(rows) and all(row[0] is not None for row in rows)
        await eventually(delivered)
    if failpoint == 'in_model':
        usage = await h.store.model_usage(task_id)
        assert usage.unknown_usage >= 1  # the attempt that died may have been billed: never counted as zero


@pytest.mark.parametrize('safe_tool, status, effects', [(False, 'outcome_unknown', 1), (True, 'succeeded', 2)])
async def test_a_crash_after_the_tool_effect_is_unknown_unless_the_tool_is_retry_safe(crash_harness, safe_tool,
                                                                                     status, effects):
    h = crash_harness
    worker = await h.start('loop', 'tool_effect', safe_tool=safe_tool)
    task_id = await h.submit('loop')
    await h.crashed(worker)
    h.advance()
    await h.start('loop', safe_tool=safe_tool)
    assert (await h.result(task_id)).status == status
    assert h.external_effects('tool') == effects  # a retry-safe tool may run again; an unsafe one never does


async def test_a_crash_after_a_receipt_commit_does_not_repeat_the_consult(crash_harness):
    h = crash_harness
    await h.start('analyst')
    asker = await h.start('asker', 'receipt_commit')
    task_id = await h.submit('asker', {'question': 'q'})
    await h.crashed(asker)
    h.advance()
    await h.start('asker')
    result = await h.result(task_id)
    assert (result.status, result.output) == ('succeeded', {'consulted': 'analysed'})
    assert h.external_effects('analyst') == 1


async def test_a_crash_right_after_an_ownership_transfer_leaves_one_owner(crash_harness):
    h = crash_harness
    await h.start('analyst')
    general = await h.start('general', 'ownership_transfer')
    task_id = await h.submit('general')
    await h.crashed(general)
    h.advance()
    await h.start('general')
    result = await h.result(task_id)
    assert (result.status, result.responding_agent) == ('succeeded', 'analyst')
    assert h.external_effects('analyst') == 1
