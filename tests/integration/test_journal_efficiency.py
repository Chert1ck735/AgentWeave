"""Journal efficiency: few write transactions per task, no busy recovery polling, one call per transaction."""
import asyncio
import sqlite3
from collections import Counter
import pytest
from tests.integration.test_python_runtime import eventually, runtime_harness  # noqa: F401

# Journal writes on the path of one request, from submission to the delivered reply.
TASK_WRITES = ('submit', 'admit', 'claim', 'gate', 'start_handler', 'begin_attempt', 'commit_step', 'lease_outbox',
               'record_broker_ack', 'confirm_receipt', 'resolve_submission_expiry')


def _count_writes(store):
    writes = Counter()
    original = store._transaction

    async def counted(operation):
        name = operation.__qualname__.split('.<locals>')[0].split('.')[-1]
        writes[name] += 1
        return await original(operation)
    store._transaction = counted
    return writes


async def test_a_request_needs_few_journal_writes(runtime_harness):
    from agentweave import Agent
    agent = Agent('cheap')
    agent.on_request(lambda ctx, request: 'ok')
    await runtime_harness.start(agent)
    await asyncio.sleep(1)
    writes = _count_writes(runtime_harness.store)
    result = await runtime_harness.ask('cheap', 'hi')
    assert result.status == 'succeeded'

    async def delivered():  # a separate read-only connection: nothing here may lease or write
        with sqlite3.connect(runtime_harness.store.path) as connection:
            return connection.execute('SELECT COUNT(*) FROM outbox WHERE delivered=0').fetchone() == (0,)
    await eventually(delivered, timeout=20)
    await asyncio.sleep(0.5)
    task_writes = {name: count for name, count in writes.items()
                   if any(name.startswith(prefix) or prefix in name for prefix in TASK_WRITES)}
    print('journal writes per request:', dict(writes))
    assert sum(task_writes.values()) <= 9, task_writes  # 14 before the journal optimization


async def test_waiting_for_a_result_does_not_write_the_journal(runtime_harness, envelope_factory):
    """Client.result polls every 20 ms; under load hundreds of waiting clients each took the write lock."""
    from agentweave.messages import Envelope
    store = runtime_harness.store
    event = envelope_factory(recipient='nobody', sender=runtime_harness.client.sender)
    await store.submit(event)  # nobody will ever answer before its deadline
    writes = _count_writes(store)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(runtime_harness.client.result(event.task_id), 1)
    assert sum(writes.values()) == 0, dict(writes)


async def test_idle_recovery_does_not_poll_the_journal_continuously(runtime_harness):
    """Recovery used to read the journal every 50 ms per worker; the engine now wakes it when work frees up."""
    from agentweave import Agent
    agent = Agent('quiet')
    agent.on_request(lambda ctx, request: 'ok')
    await runtime_harness.start(agent)
    store, reads = runtime_harness.store, []
    original = store.recoverable_page

    async def counted(*args, **kwargs):
        reads.append(args)
        return await original(*args, **kwargs)
    store.recoverable_page = counted
    await asyncio.sleep(2)
    assert len(reads) <= 4, len(reads)  # was about 40


async def test_a_journal_transaction_is_one_call_on_the_journal_thread(store, envelope_factory):
    """Each statement used to be its own hop to the database thread while the write lock was held."""
    calls = []
    original = store._in_thread

    async def counted(operation):
        calls.append(operation)
        return await original(operation)
    store._in_thread = counted
    await store.admit(envelope_factory(), 'def-v1')
    assert len(calls) == 1  # BEGIN IMMEDIATE ... COMMIT ran as one unit on the journal thread


async def test_many_waiting_results_cost_few_journal_reads(runtime_harness, envelope_factory):
    """Found by the load run: each waiting Client.result read the journal twice every 20 ms, so a
    backlog of waiting requests saturated the client and collapsed request/reply over a minute."""
    store, reads = runtime_harness.store, []
    for name in ('get_task', 'resolve_submission_expiry'):
        original = getattr(store, name)

        async def counted(*args, original=original, **kwargs):
            reads.append(args)
            return await original(*args, **kwargs)
        setattr(store, name, counted)
    events = [envelope_factory(recipient='nobody', sender=runtime_harness.client.sender) for _ in range(20)]
    for event in events:
        await store.submit(event)  # nobody answers: every client keeps waiting
    waiting = [asyncio.create_task(runtime_harness.client.result(event.task_id)) for event in events]
    await asyncio.sleep(1)
    for task in waiting:
        task.cancel()
    await asyncio.gather(*waiting, return_exceptions=True)
    assert len(reads) <= 20 * 12, len(reads)  # was 20 * 2 * 50 = 2000


async def test_a_topic_event_receipt_is_kept_but_never_published(store, envelope_factory):
    """Found by the load run: every topic event published a receipt nobody waits for (pub/sub
    publishers finish on the broker ACK), and every worker of the network consumed and committed it."""
    event = envelope_factory(kind='orders.created', topic='orders.created', recipient=None, task_id=None)
    await store.admit(event, 'def-v1', agent_id='assistant')
    request = envelope_factory()
    await store.admit(request, 'def-v1')
    with sqlite3.connect(store.path) as connection:
        rows = dict(connection.execute("SELECT json_extract(envelope,'$.correlation_id'), delivered FROM outbox "
                                       "WHERE json_extract(envelope,'$.kind')='receipt'").fetchall())
    assert rows == {event.event_id: 1, request.event_id: 0}  # only the addressed request's sender awaits it
    assert (await store.admit(event, 'def-v1', agent_id='assistant')).duplicate  # dedup still answers
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT delivered FROM outbox WHERE json_extract(envelope,'$.correlation_id')=?",
                                  (event.event_id,)).fetchone() == (1,)  # and a duplicate does not republish it


async def test_replies_nobody_reads_are_journaled_not_published(runtime_harness):
    """Found by the load run: replies to clients (who read results from the journal) and to topic
    publishers (who wait for nothing) were published to inboxes nobody consumes."""
    from agentweave import Agent
    from agentweave.messages import event_task_id
    agent = Agent('replier')
    agent.on_request(lambda ctx, request: 'ok')
    agent.on('orders.created')(lambda ctx, event: 'seen')
    await runtime_harness.start(agent)
    result = await runtime_harness.ask('replier', 'hi')
    event_id = await runtime_harness.client.publish('orders.created', {'n': 1})
    done = await asyncio.wait_for(runtime_harness.client.event_result(event_id, 'replier'), 20)
    assert (result.status, done.status) == ('succeeded', 'succeeded')
    with sqlite3.connect(runtime_harness.store.path) as connection:
        replies = connection.execute("SELECT json_extract(envelope,'$.task_id'), delivered FROM outbox "
                                     "WHERE json_extract(envelope,'$.kind')='reply'").fetchall()
    assert replies == [(result.task_id, 1)]  # the client's reply is kept, already delivered; none for the event
    assert event_task_id(event_id, 'replier') not in [task for task, _ in replies]
