"""Regressions found by reviewing the runtime contracts."""
import asyncio
import logging
import sqlite3
from datetime import timedelta
from uuid import uuid4
import pytest
from tests.integration.test_python_runtime import eventually, runtime_harness  # noqa: F401
from tests.integration.test_discovery import catalog  # noqa: F401

NETWORK = '00000000-0000-0000-0000-000000000010'
OWNER = '00000000-0000-0000-0000-000000000001'


@pytest.fixture
async def bus(broker_url):
    from agentweave.adapters.broker import BrokerBus
    value = BrokerBus(broker_url, NETWORK, profile='dev')
    yield value
    await value.close()


def request_agent(handler, name='prober', retry=None):
    from agentweave import Agent
    agent = Agent(name=name)
    agent.on_request(retry=retry)(handler)
    return agent


def rows(path, sql, values=()):
    with sqlite3.connect(path) as connection:
        return connection.execute(sql, values).fetchall()


@pytest.mark.integration
async def test_ack_after_group_rebalance_is_a_typed_stale_delivery(bus, envelope_factory):
    from agentweave.errors import StaleDelivery
    for key in ('a', 'b', 'c', 'd'):
        await bus.publish(envelope_factory(kind='rebalance.probe', topic='rebalance.probe',
                                           recipient=None, task_id=None, key=key))
    group = str(uuid4())
    first = await bus.subscribe([bus.events_topic('rebalance.probe')], group)
    second = None
    try:
        delivery = await asyncio.wait_for(anext(first), 20)
        second = await bus.subscribe([bus.events_topic('rebalance.probe')], group)
        await asyncio.wait_for(anext(second), 30)  # the rebalance has assigned the joiner
        with pytest.raises(StaleDelivery):
            await bus.ack(delivery)
    finally:
        if second is not None:
            await second.aclose()
        await first.aclose()


async def test_worker_survives_stale_ack_after_durable_admission(runtime_harness, monkeypatch):
    from agentweave.errors import StaleDelivery
    h = runtime_harness
    original, stale = h.bus.ack, []

    async def ack(delivery):
        if delivery.envelope.kind == 'request' and not stale:
            stale.append(delivery)
            raise StaleDelivery('partition was reassigned')
        return await original(delivery)

    monkeypatch.setattr(h.bus, 'ack', ack)

    async def answer(ctx, request):
        return request.input
    worker = await h.start(request_agent(answer))
    result = await h.ask('prober', 'hello')
    assert stale and result.status == 'succeeded' and result.output == 'hello'
    assert not h.runs['prober'].done() and not worker.stopping


async def test_handler_exception_is_recorded_and_logged(runtime_harness, caplog):
    async def boom(ctx, request):
        raise RuntimeError('boom detail')
    caplog.set_level(logging.ERROR, logger='agentweave.runtime')
    await runtime_harness.start(request_agent(boom))
    result = await runtime_harness.ask('prober', 'hi')
    assert result.status == 'failed'
    assert result.error == {'type': 'RuntimeError', 'message': 'boom detail'}
    assert (await runtime_harness.store.get_task(result.task_id)).error == result.error
    assert any(record.levelno == logging.ERROR and record.exc_info is not None
               and 'boom detail' in str(record.exc_info[1]) for record in caplog.records)


async def test_exhausted_safe_retries_keep_the_last_error(runtime_harness):
    from agentweave.definitions import RetryPolicy
    calls = []

    async def flaky(ctx, request):
        calls.append(1)
        raise ValueError(f'attempt {len(calls)}')
    await runtime_harness.start(request_agent(flaky, retry=RetryPolicy(safe=True, max_attempts=2, min_delay=0)))
    result = await runtime_harness.ask('prober', 'hi')
    assert result.status == 'failed' and len(calls) == 2
    assert result.error == {'type': 'ValueError', 'message': 'attempt 2'}


async def test_user_timeout_error_is_a_failure_not_unknown(runtime_harness):
    async def upstream(ctx, request):
        raise TimeoutError('upstream http timeout')
    await runtime_harness.start(request_agent(upstream))
    result = await runtime_harness.ask('prober', 'hi')
    assert result.status == 'failed'
    assert result.error == {'type': 'TimeoutError', 'message': 'upstream http timeout'}


async def test_unknown_outcome_does_not_consume_admission_capacity(store, envelope_factory, fake_clock):
    from agentweave._records import Admission
    first = envelope_factory()
    await store.admit(first, 'def-v1')
    lease = await store.claim_runtime(first.task_id, str(uuid4()), 5, concurrency=1)
    assert await store.start_handler(lease, retry_safe=False)
    fake_clock.advance(200)
    assert (await store.expire_task(first.task_id)).status == 'outcome_unknown'
    later = envelope_factory(created_at=fake_clock.now(), deadline=fake_clock.now() + timedelta(seconds=120))
    decision = await store.admit_with_policy(later, 'def-v1', agent_id='assistant', max_inflight=1)
    assert isinstance(decision, Admission)


async def test_reply_to_client_is_journaled_and_never_published(store, envelope_factory, fake_clock):
    """Clients read results from the journal; publishing a reply nobody consumes only loaded the broker."""
    from agentweave._records import StepCommit
    request = envelope_factory(sender='client.abc')
    await store.admit(request, 'def-v1')
    lease = await store.claim(request.task_id, OWNER, 30)
    task = await store.get_task(request.task_id)
    reply = envelope_factory(kind='reply', sender='assistant', recipient='client.abc',
        task_id=request.task_id, correlation_id=request.event_id,
        payload={'task_id': request.task_id, 'output': 1, 'responding_agent': 'assistant', 'status': 'succeeded'})
    await store.commit_step(lease, StepCommit(step_id='s', expected_version=task.version,
                                              status='succeeded', result=1, outgoing=[reply]))
    published = 0
    for _ in range(10):
        for row in await store.lease_outbox(OWNER, 16):
            published += row.envelope.event_id == reply.event_id
            await store.record_broker_ack(row.envelope.event_id, lease_token=row.lease_token,
                expected_version=row.version, retry_at=fake_clock.now() + timedelta(seconds=1))
        fake_clock.advance(31)
    assert published == 0
    assert rows(store.path, 'SELECT delivered FROM outbox WHERE event_id=?', (reply.event_id,)) == [(1,)]


async def test_client_names_are_reserved_for_clients():
    from agentweave import Agent
    with pytest.raises(ValueError):
        Agent(name='client.impostor')


async def test_outbox_lease_uses_pending_index(store, envelope_factory):
    for _ in range(3):
        envelope = envelope_factory(recipient=None, topic='news')
        await store.submit(envelope)
        await store.record_broker_ack(envelope.event_id)
    assert rows(store.path, 'SELECT COUNT(*) FROM outbox WHERE delivered=0') == [(0,)]
    plan = ' '.join(str(row) for row in rows(store.path,
        'EXPLAIN QUERY PLAN SELECT * FROM outbox WHERE delivered=0 AND deadline>? '
        'AND (retry_at IS NULL OR retry_at<=?) AND (lease_expires_at IS NULL OR lease_expires_at<=?) '
        'ORDER BY rowid LIMIT 16', ('', '', '')))
    assert 'outbox_pending' in plan


async def test_receipt_completion_marks_request_delivered(store, envelope_factory):
    envelope = envelope_factory()
    await store.submit(envelope)
    row = (await store.lease_outbox(OWNER, 10))[0]
    await store.record_broker_ack(envelope.event_id, lease_token=row.lease_token, expected_version=row.version)
    assert rows(store.path, 'SELECT delivered FROM outbox WHERE event_id=?', (envelope.event_id,)) == [(0,)]
    await store.confirm_receipt(envelope.event_id, 'assistant')
    assert rows(store.path, 'SELECT delivered FROM outbox WHERE event_id=?', (envelope.event_id,)) == [(1,)]


async def test_recovery_does_not_execute_unknown_outcomes(runtime_harness, monkeypatch):
    from agentweave.definitions import RetryPolicy
    h = runtime_harness

    async def stuck(ctx, request):
        await asyncio.Event().wait()
    worker = await h.start(request_agent(stuck, name='stuck', retry=RetryPolicy(timeout=.02)))
    result = await h.ask('stuck', 'x')
    assert result.status == 'outcome_unknown'
    submitted = await h.store.get_submission_envelope(result.task_id)

    async def request_delivered():
        # Until the receipt arrives the outbox may redeliver the request; a
        # deduplicated redelivery legitimately calls execute once more.
        return rows(h.store.path, 'SELECT delivered FROM outbox WHERE event_id=?',
                    (submitted.event_id,)) == [(1,)]
    await eventually(request_delivered)
    executed = []
    original = worker.engine.execute

    async def execute(task_id):
        executed.append(task_id)
        return await original(task_id)
    monkeypatch.setattr(worker.engine, 'execute', execute)
    await asyncio.sleep(.3)
    assert result.task_id not in executed


async def test_session_history_queries_are_indexed(store):
    plan = ' '.join(str(row) for row in rows(store.path,
        'EXPLAIN QUERY PLAN SELECT message FROM messages WHERE session_id=? ORDER BY id', ('x',)))
    assert 'messages_session' in plan


async def test_client_publish_and_event_result(runtime_harness):
    from tests.fixtures.agents import counter_agent
    h = runtime_harness
    await h.start(counter_agent())
    event_id = await h.client.publish('numbers.received', {'numbers': [2, 3]}, session_id=OWNER)
    result = await asyncio.wait_for(h.client.event_result(event_id, 'counter'), 20)
    assert result.status == 'succeeded' and result.output == 2  # counts the two numbers


def test_owned_partitions_forget_offsets_only_after_commit():
    from types import SimpleNamespace
    from agentweave.adapters.broker import BrokerPosition, _OwnedPartitions
    from agentweave.errors import StaleDelivery
    partitions = _OwnedPartitions()
    partitions.assign([SimpleNamespace(topic='t', partition=0)])
    generation = [partitions.track(('t', 0), offset) for offset in (5, 6, 7)][0]

    def at(offset, generation=generation):
        return BrokerPosition('s', 'g', 't', 0, offset, generation, 'e', None)
    assert partitions.complete(at(6)) is None  # 5 is still unacknowledged
    assert partitions.complete(at(5)) == ('t', 0, 7, 2)
    # A failed commit leaves the prefix tracked, so the next ack still covers it.
    assert partitions.complete(at(7)) == ('t', 0, 8, 3)
    partitions.committed(('t', 0), 3)
    assert partitions.outstanding_count() == 0
    with pytest.raises(StaleDelivery):
        partitions.complete(at(7))
    partitions.assign([SimpleNamespace(topic='t', partition=0)])
    with pytest.raises(StaleDelivery):
        partitions.complete(at(9, generation=generation))


@pytest.mark.parametrize('topic', ['request', 'reply', 'receipt', 'receipts', 'error', 'errors',
                                   'progress', 'directory', 'tool_call'])
def test_events_cannot_use_control_kinds(topic):
    from agentweave.messages import validate_event_topic
    with pytest.raises(ValueError):
        validate_event_topic(topic)


async def test_reply_on_the_bus_carries_only_the_error_type(runtime_harness):
    h = runtime_harness

    async def boom(ctx, request):
        raise RuntimeError('token=secret-value')
    await h.start(request_agent(boom))
    result = await h.ask('prober', 'hi')
    assert result.error == {'type': 'RuntimeError', 'message': 'token=secret-value'}
    replies = [row[0] for row in rows(h.store.path, "SELECT envelope FROM outbox "
                                      "WHERE json_extract(envelope,'$.kind')='reply'")]
    assert replies and all('secret-value' not in reply for reply in replies)


async def test_migration_stops_republishing_existing_client_replies(tmp_path, fake_clock, envelope_factory):
    from agentweave.adapters.storage.migrations import statements
    from agentweave.adapters.storage.sqlite import SQLiteStore
    path = tmp_path / 'legacy.sqlite3'
    reply = envelope_factory(kind='reply', sender='assistant', recipient='client.abc',
        payload={'task_id': str(uuid4()), 'output': 1, 'responding_agent': 'assistant', 'status': 'succeeded'})
    with sqlite3.connect(path) as connection:
        for version in (1, 2, 3):
            for statement in statements(version):
                connection.execute(statement)
            connection.execute('INSERT INTO schema_migrations VALUES(?,?)', (version, fake_clock.now().isoformat()))
        connection.execute('INSERT INTO outbox(event_id,envelope,targets,retry_at,deadline) VALUES(?,?,?,?,?)',
            (reply.event_id, reply.model_dump_json(), '["client.abc"]', fake_clock.now().isoformat(),
             reply.deadline.isoformat(timespec='microseconds')))
    store = SQLiteStore(path, clock=fake_clock)
    await store.migrate()
    try:
        row = (await store.lease_outbox(OWNER, 16))[0]
        assert row.targets == []
        await store.record_broker_ack(reply.event_id, lease_token=row.lease_token, expected_version=row.version)
        fake_clock.advance(31)
        assert await store.lease_outbox(OWNER, 16) == []
    finally:
        await store.close()


async def test_withdraw_without_a_broker_still_releases_the_instance(catalog, monkeypatch, caplog):
    from agentweave.errors import BrokerUnavailable
    from tests.integration.test_discovery import participant
    directory, bus = catalog
    mine = participant(directory.clock)
    await directory.publish(mine)

    async def unavailable(envelope):
        raise BrokerUnavailable('broker operation failed temporarily')
    monkeypatch.setattr(bus, 'publish', unavailable)
    caplog.set_level(logging.WARNING, logger='agentweave.directory')
    await directory.withdraw(mine.logical_id, mine.instance_id)  # must not raise
    assert rows(directory.store.path, 'SELECT COUNT(*) FROM definition_instances WHERE instance_id=?',
                (mine.instance_id,)) == [(0,)]
    assert any('withdrawal' in record.getMessage() for record in caplog.records)
    await directory.close()  # nothing left to withdraw, so close does not retry the broker


async def _started_safe_attempt(store, envelope_factory):
    event = envelope_factory()
    await store.admit(event, 'def-v1')
    lease = await store.claim_runtime(event.task_id, str(uuid4()), 5, concurrency=1)
    assert await store.start_handler(lease, retry_safe=True)
    return event, lease


async def test_interrupted_safe_attempt_past_deadline_is_unknown(store, envelope_factory, fake_clock):
    event, _ = await _started_safe_attempt(store, envelope_factory)
    fake_clock.advance(200)  # the worker died mid-effect; no retry can start before the deadline
    assert (await store.expire_task(event.task_id)).status == 'outcome_unknown'


async def test_known_safe_failure_past_deadline_is_expired_with_its_error(store, envelope_factory, fake_clock):
    event, lease = await _started_safe_attempt(store, envelope_factory)
    await store.release_claim(lease, error={'type': 'ConnectionError', 'message': 'down'})
    fake_clock.advance(200)
    task = await store.expire_task(event.task_id)
    assert task.status == 'expired' and task.error == {'type': 'ConnectionError', 'message': 'down'}


async def test_exhausted_safe_retries_ending_in_an_interruption_are_unknown(store, envelope_factory, fake_clock):
    event, lease = await _started_safe_attempt(store, envelope_factory)
    for _ in range(2):
        fake_clock.advance(6)  # each attempt's lease lapses without a recorded outcome
        lease = await store.claim_runtime(event.task_id, str(uuid4()), 5, concurrency=1)
        await store.start_handler(lease, retry_safe=True, max_attempts=3)
    fake_clock.advance(6)
    lease = await store.claim_runtime(event.task_id, str(uuid4()), 5, concurrency=1)
    assert not await store.start_handler(lease, retry_safe=True, max_attempts=3)
    assert (await store.get_task(event.task_id)).status == 'outcome_unknown'


async def test_retry_delay_is_journaled_and_binds_every_replica(store, envelope_factory, fake_clock):
    event, lease = await _started_safe_attempt(store, envelope_factory)
    await store.release_claim(lease, error={'type': 'E', 'message': 'x'},
                              retry_at=fake_clock.now() + timedelta(seconds=1))
    assert await store.claim_runtime(event.task_id, str(uuid4()), 5, concurrency=1) is None
    assert [t.task_id for t in await store.recoverable_page('assistant')] == []
    fake_clock.advance(1.01)
    assert await store.claim_runtime(event.task_id, str(uuid4()), 5, concurrency=1) is not None


async def test_worker_waits_for_the_journaled_retry_time(runtime_harness):
    from agentweave.definitions import RetryPolicy
    h, calls = runtime_harness, []

    async def flaky(ctx, request):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError('first attempt fails')
        return 'ok'
    await h.start(request_agent(flaky, retry=RetryPolicy(safe=True, min_delay=1)))
    submission = await h.client.agent('prober').submit('hi')
    async def first_failed():
        return calls == [1]
    await eventually(first_failed)
    await asyncio.sleep(1.5)  # longer than min_delay in real time
    assert calls == [1]  # the journaled retry time on the frozen test clock has not come
    h.store.clock.advance(1.01)
    result = await asyncio.wait_for(h.client.result(submission.task_id), 20)
    assert result.status == 'succeeded' and calls == [1, 1]


async def test_lease_loss_on_the_retry_path_does_not_stop_the_worker(runtime_harness, monkeypatch):
    from agentweave.definitions import RetryPolicy
    from agentweave.errors import LeaseLost
    h = runtime_harness

    async def fails(ctx, request):
        raise ConnectionError('boom')
    worker = await h.start(request_agent(fails, retry=RetryPolicy(safe=True, min_delay=0)))
    original, lost = h.store.release_claim, []

    async def release_claim(lease, **kwargs):
        if kwargs.get('error') is not None and not lost:
            lost.append(lease)
            raise LeaseLost('execution lease is no longer current')
        return await original(lease, **kwargs)
    monkeypatch.setattr(h.store, 'release_claim', release_claim)
    await h.client.agent('prober').submit('hi')
    async def released():
        return bool(lost)
    await eventually(released)
    await asyncio.sleep(.2)
    assert not h.runs['prober'].done() and not worker.stopping


async def test_slow_adapter_close_respects_the_shutdown_budget(runtime_harness):
    from agentweave import Agent, Model
    from agentweave.testing import FakeModel
    release = asyncio.Event()

    class SlowClose(FakeModel):
        async def close(self):
            await release.wait()
    worker = await runtime_harness.start(Agent('slowclose', model=Model.custom(SlowClose([]))))
    worker._shutdown_deadline = asyncio.get_running_loop().time() + 0.03
    started = asyncio.get_running_loop().time()
    with pytest.raises(BaseException):
        await asyncio.wait_for(asyncio.shield(worker.stop()), 2)
    assert asyncio.get_running_loop().time() - started < 1  # stop returned within its budget
    release.set()
    runtime_harness.runs.pop('slowclose').cancel()
    runtime_harness.workers.pop('slowclose')


@pytest.mark.integration
async def test_subscribe_racing_close_never_leaves_a_live_consumer(bus):
    await bus.ensure_topics([])
    await bus._producer_lock.acquire()  # a publication is in flight
    closing = asyncio.create_task(bus.close())
    late = None
    try:
        await asyncio.sleep(.1)
        try:
            late = await bus.subscribe([bus.events_topic('late.topic')], str(uuid4()))
        except RuntimeError:
            pass
    finally:
        bus._producer_lock.release()
        await closing
    if late is not None:
        await late.aclose()  # cleanup for the failing case; the assertion below reports it
    assert late is None and bus._subscriptions == {}


async def test_client_events_reads_progress_written_with_the_terminal_result():
    from datetime import datetime, timezone
    from agentweave.client import Client
    from agentweave._records import TaskRecord
    progress = Envelope_progress()

    class Store:
        reads = 0
        async def get_task(self, task_id):
            return TaskRecord(task_id=task_id, agent_id='a', status='succeeded', version=1, input='x',
                              session_id=None, root_task_id=task_id, definition_hash='h',
                              deadline=datetime(2030, 1, 1, tzinfo=timezone.utc))
        async def get_submission_envelope(self, task_id):
            return None
        async def read_progress_page(self, task_id, *, after_sequence=0, limit=64):
            self.reads += 1
            # The last progress commits together with the result, after the first empty read.
            return [] if self.reads == 1 else ([(1, progress)] if after_sequence < 1 else [])

    class Subscription:
        def __aiter__(self):
            return self
        async def __anext__(self):
            await asyncio.Event().wait()
        async def aclose(self):
            return None

    class Bus:
        def control_topic(self, kind):
            return 'progress'
        async def subscribe(self, topics, group):
            return Subscription()
    client = Client(config=None, store=Store(), bus=Bus(), directory=None)
    events = [event async for event in client.events(progress.task_id)]
    assert events == [progress]


def Envelope_progress():
    from datetime import datetime, timedelta, timezone
    from agentweave.messages import Envelope
    now = datetime.now(timezone.utc)
    return Envelope(event_id=str(uuid4()), network_id=NETWORK, kind='progress', sender='a', recipient=None,
                    topic='progress', task_id=str(uuid4()), parent_task_id=None, correlation_id=None,
                    session_id=None, created_at=now, deadline=now + timedelta(seconds=60), traceparent=None,
                    payload={'text': 'last words'})


async def test_an_acknowledgement_does_not_wait_for_the_next_idle_poll(bus, envelope_factory):
    """Found by the load run: every ack waited for the consumer's 50 ms poll, capping a worker near 20/s."""
    import time
    topic = bus.events_topic('ack.latency')
    subscription = await bus.subscribe([topic], 'ack-latency')
    try:
        for n in range(20):
            await bus.publish(envelope_factory(kind='ack.latency', topic='ack.latency', recipient=None, task_id=None,
                                               payload={'n': n}))
        durations = []
        for _ in range(20):
            delivery = await asyncio.wait_for(anext(subscription), 20)
            started = time.perf_counter()
            await bus.ack(delivery)
            durations.append(time.perf_counter() - started)
    finally:
        await subscription.aclose()
    assert sum(durations) / len(durations) < 0.02, durations


async def test_an_idle_outbox_pump_takes_no_write_lock(store):
    """Found by the load run: ten idle pumps opened ~100 empty write transactions per second."""
    writes = []
    original = store._transaction

    async def counted(operation):
        writes.append(operation)
        return await original(operation)
    store._transaction = counted
    assert await store.lease_outbox('00000000-0000-0000-0000-0000000000a9', 16) == []
    assert writes == []


async def test_each_receipt_is_written_once_however_many_workers_run(runtime_harness):
    """Found by the load run: every worker wrote every receipt, one write transaction per worker.

    Every worker still sees every receipt (none depends on another being alive); only the first writes.
    """
    from agentweave import Agent
    store = runtime_harness.store
    calls, writes = [], []
    confirm, transaction = store.confirm_receipt, store._transaction

    async def counted(event_id, recipient):
        calls.append(event_id)
        return await confirm(event_id, recipient)

    async def written(operation):
        if 'confirm_receipt' in operation.__qualname__:
            writes.append(operation)
        return await transaction(operation)
    store.confirm_receipt, store._transaction = counted, written
    for name in ('first', 'second', 'third'):
        agent = Agent(name)
        agent.on_request(lambda ctx, request: 'ok')
        await runtime_harness.start(agent)
    for name in ('first', 'second', 'third'):
        assert (await runtime_harness.ask(name, 'hi')).status == 'succeeded'

    async def seen_by_every_worker():
        return len(set(calls)) >= 3 and min(calls.count(event_id) for event_id in set(calls)) == 3
    await eventually(seen_by_every_worker, timeout=30)
    assert len(writes) == len(set(calls))  # one write per receipt, not one per worker



async def test_a_record_published_while_a_fetch_is_held_arrives_quickly(bus, envelope_factory):
    """Found by the load run: the dev broker holds a fetch for fetch.wait.max.ms and does not end it
    early for new data, so a record waited up to 500 ms (request p95 ~560 ms with no load at all)."""
    import time
    topic = bus.events_topic('fetch.latency')
    subscription = await bus.subscribe([topic], 'fetch-latency', low_latency=True)  # as an inbox subscribes
    try:
        await asyncio.sleep(2)  # assigned and idle
        latencies = []
        for n in range(8):
            await asyncio.sleep(0.3)  # publish in the middle of a held fetch
            started = time.perf_counter()
            await bus.publish(envelope_factory(kind='fetch.latency', topic='fetch.latency', recipient=None,
                                               task_id=None, payload={'n': n}))
            delivery = await asyncio.wait_for(anext(subscription), 20)
            latencies.append(time.perf_counter() - started)
            await bus.ack(delivery)
    finally:
        await subscription.aclose()
    assert sorted(latencies)[len(latencies) // 2] < 0.12, latencies
