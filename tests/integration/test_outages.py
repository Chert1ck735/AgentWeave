"""Infrastructure limits: what the dev broker loses, a full disk, overload, shared pools and clean logs."""
import asyncio
import logging
import re
import sqlite3
import time
from uuid import uuid4
import pytest
from tests.fixtures.crash import effects_of
from tests.integration.test_crash_matrix import CrashHarness, crash_harness  # noqa: F401
from tests.integration.test_python_runtime import eventually, runtime_harness  # noqa: F401

UUID_TEXT = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')


async def test_dev_broker_loses_an_acknowledged_event_nobody_admitted_and_says_so(crash_harness):
    """The documented dev limitation, shown honestly rather than called durable recovery."""
    from agentweave.messages import event_task_id
    h = crash_harness
    capabilities = h.bus.capabilities()
    assert (capabilities.durable_stream, capabilities.persistent_consumer_offsets) == (False, False)
    event_id = await h.client.publish('work.item', {'n': 1})  # the broker acknowledged it
    h.broker.restart()
    await h.connect()
    await h.start('listener')
    await asyncio.sleep(3)
    assert effects_of(h.effects, 'listener') == []
    assert await h.store.get_task(event_task_id(event_id, 'listener')) is None  # lost, and nothing pretends otherwise
    control = await h.client.publish('work.item', {'n': 2})  # the same subscriber does handle what it can see

    async def handled():
        return [row['event'] for row in effects_of(h.effects, 'listener')] == [control]
    await eventually(handled, timeout=30)


async def test_a_full_disk_admits_nothing_and_confirms_nothing(store, envelope_factory):
    from agentweave.errors import StoreUnavailable
    pages = (await (await store._connection.execute('PRAGMA page_count')).fetchone())[0]
    await store._connection.execute(f'PRAGMA max_page_count={pages}')  # SQLITE_FULL on the next new page
    event = envelope_factory(payload='x' * 200_000)
    with pytest.raises(StoreUnavailable):
        await store.admit(event, 'def-v1')
    assert await store.get_task(event.task_id) is None
    with sqlite3.connect(store.path) as connection:
        assert connection.execute('SELECT COUNT(*) FROM receipts').fetchone() == (0,)
    await store._connection.execute('PRAGMA max_page_count=1073741823')
    assert not (await store.admit(event, 'def-v1')).duplicate  # the same input is admitted once space returns


async def test_a_sync_timeout_after_the_effect_is_unknown_and_never_repeated(runtime_harness):
    from agentweave import Agent
    from agentweave.definitions.tool import RetryPolicy
    effects = []
    agent = Agent('sync')

    @agent.on_request(retry=RetryPolicy(timeout=0.2))
    def slow(ctx, request):
        effects.append(request.input)
        time.sleep(1)  # the effect already happened when the runtime gives up waiting
        return 'late'
    await runtime_harness.start(agent)
    result = await runtime_harness.ask('sync', 'charge')
    assert result.status == 'outcome_unknown' and effects == ['charge']
    await runtime_harness.restart('sync')
    await asyncio.sleep(1)
    assert effects == ['charge']  # recovery leaves an unknown outcome to the operator


async def _finished(store, task_id):
    return (await store.get_task(task_id)).status == 'succeeded'


async def test_an_overloaded_subscriber_is_bounded_and_the_other_keeps_working(runtime_harness):
    from agentweave import Agent, SubscriptionPolicy
    from agentweave.messages import event_task_id
    release, fast_seen = asyncio.Event(), []
    agent = Agent('busy', concurrency=4, subscription_policies={
        'slow.work': SubscriptionPolicy(queue_limit=2, overflow='reject')})

    @agent.on('slow.work', retry='safe')
    async def slow(ctx, event):
        await release.wait()
        return 'slow'

    @agent.on('fast.work', retry='safe')
    async def fast(ctx, event):
        fast_seen.append(event['n'])
        return 'fast'

    @agent.on_request
    async def answer(ctx, request):
        return 'still answering'
    worker = await runtime_harness.start(agent)
    client = runtime_harness.client
    slow_ids = [await client.publish('slow.work', {'n': n}) for n in range(10)]
    fast_ids = [await client.publish('fast.work', {'n': n}) for n in range(20)]

    async def fast_done():
        return sorted(fast_seen) == list(range(20))
    await eventually(fast_done, timeout=30)
    assert (await runtime_harness.ask('busy', 'ping')).output == 'still answering'  # requests are still served
    assert worker.metrics.queues['slow.work']['depth'] <= 2  # bounded by the policy, not by memory

    async def slow_accounted():
        admitted = [await runtime_harness.store.get_task(event_task_id(e, 'busy')) for e in slow_ids]
        queued = worker.metrics.queues['slow.work']['depth']
        return sum(task is not None for task in admitted) + worker.metrics.counters['rejected'] + queued == 10
    # Every slow event is admitted, waiting in the bounded queue, or an explicit policy skip: none unexplained.
    await eventually(slow_accounted, timeout=30)
    assert worker.metrics.counters['rejected'] > 0
    release.set()

    async def slow_drained():
        admitted = [await runtime_harness.store.get_task(event_task_id(e, 'busy')) for e in slow_ids]
        return sum(task is not None for task in admitted) + worker.metrics.counters['rejected'] == 10
    await eventually(slow_drained, timeout=30)
    for event_id in slow_ids + fast_ids:
        task = await runtime_harness.store.get_task(event_task_id(event_id, 'busy'))
        if task is not None:  # an admitted task is never lost
            await eventually(lambda: _finished(runtime_harness.store, task.task_id), timeout=30)


async def test_the_provider_pool_is_shared_by_worker_processes(crash_harness, tmp_path):
    h = crash_harness
    await h.connect(provider_pool_limit=1)
    model_effects = str(tmp_path / 'model.jsonl')
    for name in ('loop', 'loop2'):
        await h.start('loop', name=name, no_tools=True, model_seconds=0.3, model_effects=model_effects)
    task_ids = [await h.submit(name) for name in ('loop', 'loop2') for _ in range(3)]
    results = await asyncio.gather(*(h.result(task_id) for task_id in task_ids))
    assert all(result.status == 'succeeded' for result in results)
    calls = effects_of(model_effects)
    edges = sorted([(row['at'], 1 if row['kind'] == 'model_start' else -1) for row in calls])
    running = peak = 0
    for _, step in edges:
        running += step
        peak = max(peak, running)
    assert peak == 1 and len({row['pid'] for row in calls}) == 2  # two processes, one request at a time


async def test_logs_carry_ids_not_payloads_secrets_or_hidden_reasoning(runtime_harness, caplog):
    from agentweave import Agent, Model, tool
    from agentweave.models import ModelTurn, ProposedToolCall
    from agentweave.messages import Message
    from agentweave.testing import FakeModel, text_turn

    @tool
    def lookup(query: str) -> str:
        """Look something up."""
        return 'found'
    call = ProposedToolCall(provider_call_id='c1', name='lookup', arguments={'query': 'SECRET_PAYLOAD'})
    thinking = ModelTurn(messages=[Message(role='assistant', tool_calls=[{'id': 'c1', 'name': 'lookup',
                                                                          'arguments': {'query': 'SECRET_PAYLOAD'}}])],
                         tool_calls=[call], adapter_metadata={'reasoning': 'HIDDEN_REASONING'})
    model = FakeModel([thinking, text_turn('answer')])
    worker = await runtime_harness.start(Agent('logged', model=Model.custom(model), tools=[lookup]))
    caplog.set_level(logging.INFO)  # ordinary logs; every SDK logger also at DEBUG
    caplog.set_level(logging.DEBUG, logger='agentweave')
    result = await runtime_harness.ask('logged', {'card': 'SECRET_PAYLOAD'}, session_id=str(uuid4()))
    assert result.status == 'succeeded'
    for record in caplog.records:
        text = record.getMessage() + ' ' + ' '.join(str(value) for value in vars(record).values())
        assert 'SECRET_PAYLOAD' not in text and 'HIDDEN_REASONING' not in text, record.name
    names = set(worker.metrics.counters) | set(worker.metrics.queues)
    assert not any(UUID_TEXT.search(name) for name in names)  # metric names never carry task or event IDs
    with sqlite3.connect(runtime_harness.store.path) as connection:  # the journal links the input to its tool call
        [(tool_task,)] = connection.execute('SELECT task_id FROM tool_executions').fetchall()
    assert tool_task == result.task_id
