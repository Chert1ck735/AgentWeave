"""Live catalog and upgrade barriers on the actual broker and SQLite journal."""
import asyncio
from datetime import timedelta
import multiprocessing
import sqlite3
from uuid import uuid4
import pytest

NETWORK = '00000000-0000-0000-0000-000000000010'


def participant(clock, *, logical_id='assistant', instance_id=None, **changes):
    from agentweave._records import Participant
    values = dict(network_id=NETWORK, logical_id=logical_id, instance_id=instance_id or str(uuid4()),
        kind='agent', fingerprint='hash-v1', configuration_version='v1',
        broker_epoch='epoch', heartbeat_at=clock.now(), description='Public description')
    return Participant(**(values | changes))


async def eventually(predicate, timeout=20):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(.05)


@pytest.fixture
async def catalog(broker_process, store, fake_clock):
    from agentweave.adapters.broker import BrokerBus
    from agentweave._directory import LiveDirectory
    bus = BrokerBus(broker_process.url, NETWORK, profile='dev', broker_epoch=broker_process.epoch)
    directory = LiveDirectory(bus, store, NETWORK, broker_epoch=broker_process.epoch, clock=fake_clock)
    await directory.start()
    try:
        yield directory, bus
    finally:
        await directory.close()
        await bus.close()


@pytest.mark.integration
async def test_new_participants_replicas_and_clean_stop_update_without_restart(catalog, store_factory, fake_clock, tmp_path):
    directory, bus = catalog
    from agentweave._directory import LiveDirectory
    from agentweave.adapters.broker import BrokerBus
    other_bus = BrokerBus(bus.broker_url, NETWORK, profile='dev', broker_epoch=bus.epoch)
    other_store = await store_factory(tmp_path / 'publisher.sqlite3')
    publisher = LiveDirectory(other_bus, other_store, NETWORK, broker_epoch=bus.epoch, clock=fake_clock)
    await publisher.start()
    first = participant(fake_clock)
    second = participant(fake_clock)
    try:
        await publisher.publish(first)
        await publisher.publish(second)
        await eventually(lambda: len(directory.snapshot().get('assistant', ())) == 2)
        assert set(directory.snapshot()) == {'assistant'}
        await publisher.withdraw(first.logical_id, first.instance_id)
        await eventually(lambda: len(directory.snapshot().get('assistant', ())) == 1)
        assert directory.snapshot()['assistant'][0].instance_id == second.instance_id
        await publisher.close()
        await eventually(lambda: directory.snapshot() == {})
    finally:
        await publisher.close()
        await other_bus.close()


@pytest.mark.integration
async def test_heartbeat_expiry_replay_and_epoch_do_not_use_db_as_liveness(catalog, broker_process, store, fake_clock):
    directory, bus = catalog
    value = participant(fake_clock)
    await directory.publish(value)
    await eventually(lambda: 'assistant' in directory.snapshot())
    fake_clock.advance(14.9)
    assert 'assistant' in directory.snapshot()
    fake_clock.advance(.1)
    assert directory.snapshot() == {}
    await directory.heartbeat()
    await eventually(lambda: 'assistant' in directory.snapshot())
    broker_process.restart()
    await bus.broker_epoch_changed(broker_process.epoch)
    await directory.broker_epoch_changed(broker_process.epoch)
    assert directory.snapshot() == {}
    await directory.heartbeat()
    await eventually(lambda: 'assistant' in directory.snapshot())
    assert directory.snapshot()['assistant'][0].broker_epoch == broker_process.epoch


@pytest.mark.integration
@pytest.mark.parametrize('elapsed_before_stop', [0, 1])
async def test_replayed_announcement_cannot_resurrect_a_clean_stop(catalog, fake_clock, elapsed_before_stop):
    directory, bus = catalog
    value = participant(fake_clock)
    await directory.publish(value)
    await eventually(lambda: 'assistant' in directory.snapshot())
    announcement = directory._envelope(value.logical_id, value.instance_id,
        {'participant': directory.snapshot()['assistant'][0].model_dump(mode='json'),
         'broker_epoch': bus.epoch})
    if elapsed_before_stop:
        fake_clock.advance(elapsed_before_stop)
    await directory.withdraw(value.logical_id, value.instance_id)
    await eventually(lambda: directory.snapshot() == {})
    await bus.publish(announcement)
    # The marker is on the same one-partition stream, proving replay was read.
    await directory.publish(participant(fake_clock, logical_id='marker'))
    await eventually(lambda: 'marker' in directory.snapshot())
    assert 'assistant' not in directory.snapshot()


@pytest.mark.integration
async def test_withdrawn_key_accepts_new_instance_and_strictly_newer_publication(catalog, fake_clock):
    directory, _ = catalog
    first = participant(fake_clock)
    await directory.publish(first)
    await eventually(lambda: 'assistant' in directory.snapshot())
    await directory.withdraw(first.logical_id, first.instance_id)
    await eventually(lambda: directory.snapshot() == {})
    # A new process key remains discoverable at the exact same timestamp.
    replica = participant(fake_clock)
    await directory.publish(replica)
    await eventually(lambda: len(directory.snapshot().get('assistant', ())) == 1)
    assert directory.snapshot()['assistant'][0].instance_id == replica.instance_id
    # Deliberate reuse of the withdrawn key requires a demonstrably newer time.
    fake_clock.advance(1)
    await directory.publish(first)
    await eventually(lambda: len(directory.snapshot().get('assistant', ())) == 2)
    assert {value.instance_id for value in directory.snapshot()['assistant']} == {
        first.instance_id, replica.instance_id}


@pytest.mark.integration
async def test_future_tombstone_cannot_remove_a_live_participant(catalog, fake_clock):
    directory, bus = catalog
    value = participant(fake_clock)
    await directory.publish(value)
    await eventually(lambda: 'assistant' in directory.snapshot())
    tombstone = directory._envelope(value.logical_id, value.instance_id,
        {'participant': None, 'broker_epoch': bus.epoch},
        now=fake_clock.now() + timedelta(seconds=60))
    await bus.publish(tombstone)
    await directory.publish(participant(fake_clock, logical_id='marker'))
    await eventually(lambda: 'marker' in directory.snapshot())
    assert 'assistant' in directory.snapshot()


@pytest.mark.integration
async def test_tool_revision_snapshot_mutation_and_alias_resolution(catalog, fake_clock):
    from agentweave.errors import DefinitionError, StoreConflict
    directory, _ = catalog
    value = participant(fake_clock, logical_id='calculator.add', kind='tool', name='add', revision='r1',
        input_schema={'type': 'object', 'properties': {'x': {'type': 'integer'}}}, output_schema={'type': 'integer'})
    await directory.publish(value)
    await eventually(lambda: 'calculator.add' in directory.snapshot())
    snapshot = directory.snapshot()
    snapshot['calculator.add'][0].input_schema['properties']['x']['type'] = 'string'
    assert directory.resolve_tool('calculator_add', revision='r1').input_schema['properties']['x']['type'] == 'integer'
    assert directory.resolve_tool('calculator_add', revision='wrong') is None
    assert directory.resolve_tool('calculator.add') is None
    value.input_schema['properties']['x']['type'] = 'boolean'
    assert directory.resolve_tool('calculator_add').input_schema['properties']['x']['type'] == 'integer'
    changed = participant(fake_clock, logical_id='calculator.add', kind='tool', name='add', revision='r2',
        input_schema={'type': 'object'}, output_schema={'type': 'string'}, fingerprint='hash-v2')
    with pytest.raises(StoreConflict):
        await directory.publish(changed)
    await directory.withdraw(value.logical_id, value.instance_id)
    await eventually(lambda: directory.snapshot() == {})
    await directory.publish(changed)
    await eventually(lambda: directory.resolve_tool('calculator_add', revision='r2') is not None)
    assert snapshot['calculator.add'][0].revision == 'r1'
    colliding = participant(fake_clock, logical_id='calculator_add', kind='tool', name='add', revision='r1',
        input_schema={}, output_schema={})
    with pytest.raises(DefinitionError):
        await directory.publish(colliding)


@pytest.mark.parametrize('status', ['accepted', 'running', 'waiting_peer', 'waiting_tool', 'outcome_unknown'])
async def test_atomic_registration_and_upgrade_barrier_for_waiting_work(store, store_factory, fake_clock, envelope_factory, status):
    from agentweave.errors import StoreConflict
    from agentweave._records import StepCommit
    first, second = str(uuid4()), str(uuid4())
    rival = await store_factory(store.path)
    outcomes = await asyncio.gather(store.register_definition('assistant', first, 'hash-v1'),
        rival.register_definition('assistant', second, 'hash-v2'), return_exceptions=True)
    assert sum(isinstance(result, StoreConflict) for result in outcomes) == 1
    winner = 'hash-v1' if outcomes[0] is None else 'hash-v2'
    await store.register_definition('assistant', str(uuid4()), winner)
    event = envelope_factory()
    await store.admit(event, winner)
    if status != 'accepted':
        lease = await store.claim(event.task_id, first, 60)
        current = await store.get_task(event.task_id)
        if status != 'running':
            await store.commit_step(lease, StepCommit(step_id='wait', expected_version=current.version, status=status))
    fake_clock.advance(16)
    with pytest.raises(StoreConflict):
        await rival.register_definition('assistant', second, 'new-hash')
    assert (await store.get_task(event.task_id)).definition_hash == winner


async def test_replaced_definition_rejects_old_admission_and_allows_new_work(store, fake_clock, envelope_factory):
    from agentweave.errors import StoreConflict
    owner = str(uuid4())
    await store.register_definition('assistant', owner, 'hash-v1')
    await store.unregister_definition('assistant', owner)
    await store.register_definition('assistant', str(uuid4()), 'hash-v2')
    event = envelope_factory()
    with pytest.raises(StoreConflict):
        await store.admit(event, 'hash-v1')
    assert await store.get_task(event.task_id) is None
    assert not (await store.admit(event, 'hash-v2')).duplicate


@pytest.mark.parametrize('status', ['succeeded', 'failed', 'cancelled', 'expired'])
async def test_settled_work_does_not_block_next_definition(store, fake_clock, envelope_factory, status):
    from agentweave._records import StepCommit
    owner = str(uuid4())
    await store.register_definition('assistant', owner, 'hash-v1')
    event = envelope_factory()
    await store.admit(event, 'hash-v1')
    lease = await store.claim(event.task_id, owner, 30)
    current = await store.get_task(event.task_id)
    await store.commit_step(lease, StepCommit(step_id='settled', expected_version=current.version, status=status))
    await store.unregister_definition('assistant', owner)
    await store.register_definition('assistant', str(uuid4()), 'hash-v2')
    duplicate = await store.admit(event, 'hash-v1')
    assert duplicate.duplicate
    assert (await store.get_task(event.task_id)).definition_hash == 'hash-v1'


async def test_stale_definition_may_be_replaced_but_old_publisher_is_fenced(store, fake_clock):
    from agentweave.errors import StoreConflict
    first = participant(fake_clock)
    await store.register_definition(first.logical_id, first.instance_id, first.fingerprint)
    await store.publish_participant(first)
    fake_clock.advance(15)
    await store.register_definition('assistant', str(uuid4()), 'hash-v2')
    with pytest.raises(StoreConflict):
        await store.publish_participant(first.model_copy(update={'heartbeat_at':fake_clock.now()}))


async def test_schema1_upgrade_preserves_journal_and_rejects_future_version(tmp_path, envelope_factory, fake_clock):
    from agentweave.adapters.storage.sqlite import SQLiteStore
    from agentweave.adapters.storage.migrations import initial_statements
    from agentweave.errors import StoreConflict
    path = tmp_path / 'old.sqlite3'
    legacy = envelope_factory()
    with sqlite3.connect(path) as connection:
        for sql in initial_statements():
            connection.execute(sql)
        connection.execute('INSERT INTO schema_migrations VALUES(1,?)', (fake_clock.now().isoformat(),))
        connection.execute("INSERT INTO sessions VALUES('assistant',?,'{\"kept\":true}',?)", (str(uuid4()), fake_clock.now().isoformat()))
        connection.execute('INSERT INTO tasks(task_id,agent_id,status,version,input,input_envelope,session_id,root_task_id,definition_hash,deadline,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
            (legacy.task_id, 'assistant', 'waiting_tool', 3, '{"preserved":true}', legacy.model_dump_json(),
             legacy.session_id, legacy.task_id, 'hash-v1', legacy.deadline.isoformat(), fake_clock.now().isoformat()))
        connection.execute('INSERT INTO outbox(event_id,envelope,targets,deadline) VALUES(?,?,?,?)',
            (legacy.event_id, legacy.model_dump_json(), '["assistant"]', legacy.deadline.isoformat()))
    store = SQLiteStore(path, clock=fake_clock)
    try:
        await store.migrate()
        event = envelope_factory()
        await store.admit(event, 'hash-v1')
        await store.submit(envelope_factory())
        await store.register_definition('assistant', str(uuid4()), 'hash-v1')
    finally:
        await store.close()
    reopened = SQLiteStore(path, clock=fake_clock)
    try:
        await reopened.migrate()
        await reopened.migrate()
        assert (await reopened.get_task(event.task_id)).definition_hash == 'hash-v1'
        preserved = await reopened.get_task(legacy.task_id)
        assert preserved.status == 'waiting_tool'
        assert preserved.version == 3
        assert preserved.input == {'preserved': True}
        assert await reopened.get_input_envelope(legacy.task_id) == legacy
        with sqlite3.connect(path) as connection:
            assert connection.execute('SELECT version FROM schema_migrations ORDER BY version').fetchall() == [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,)]
            assert connection.execute('SELECT state FROM sessions').fetchall() == [('{"kept":true}',)]
            assert connection.execute('SELECT count(*) FROM outbox').fetchone()[0] == 3
            assert connection.execute('SELECT envelope FROM outbox WHERE event_id=?', (legacy.event_id,)).fetchone()[0] == legacy.model_dump_json()
            connection.execute('INSERT INTO schema_migrations VALUES(99,?)', (fake_clock.now().isoformat(),))
    finally:
        await reopened.close()
    future = SQLiteStore(path)
    try:
        with pytest.raises(StoreConflict):
            await future.migrate()
    finally:
        await future.close()


async def test_unsupported_compaction_rejected_before_broker_io():
    from agentweave.adapters.broker import BrokerBus
    from agentweave.ports import TopicSpec
    bus = BrokerBus('127.0.0.1:1', NETWORK, profile='dev')
    try:
        with pytest.raises(ValueError, match='compact'):
            await bus.ensure_topics([TopicSpec(bus.control_topic('directory'), partitions=1, cleanup_policy='compact')])
        with pytest.raises(ValueError, match='cleanup'):
            await bus.ensure_topics([TopicSpec(bus.control_topic('directory'), cleanup_policy='unknown')])
    finally:
        await bus.close()


@pytest.mark.parametrize('profile,cleanup', [('dev', 'delete'), ('server', 'compact')])
async def test_directory_provisions_cleanup_from_explicit_bus_capability(monkeypatch, store, profile, cleanup):
    """Check the native admin request boundary; this is not server acceptance."""
    from concurrent.futures import Future
    from types import SimpleNamespace
    import confluent_kafka.admin
    from agentweave.adapters.broker import BrokerBus
    from agentweave._directory import LiveDirectory
    requests = []
    class Admin:
        def __init__(self, config):
            pass
        def create_topics(self, topics, **options):
            requests.extend(topics)
            futures = {}
            for topic in topics:
                future = Future()
                future.set_result(None)
                futures[topic.topic] = future
            return futures
        def list_topics(self, **options):
            return SimpleNamespace(topics={topic.topic: SimpleNamespace(
                error=None, partitions={0: None}) for topic in requests})
    class Subscription:
        def __aiter__(self):
            return self.deliveries()
        async def deliveries(self):
            if False:
                yield
        async def aclose(self):
            pass
    async def subscribe(topics, group_id):
        return Subscription()
    monkeypatch.setattr(confluent_kafka.admin, 'AdminClient', Admin)
    bus = BrokerBus('127.0.0.1:1', NETWORK, profile=profile)
    monkeypatch.setattr(bus, 'subscribe', subscribe)
    directory = LiveDirectory(bus, store, NETWORK)
    try:
        await directory.start()
        assert len(requests) == 1
        assert requests[0].topic == 'aw.00000000-0000-0000-0000-000000000010.directory'
        assert requests[0].num_partitions == 1
        assert requests[0].config['cleanup.policy'] == cleanup
    finally:
        await directory.close()
        await bus.close()


@pytest.mark.parametrize('renewal', ['registration', 'publication'])
async def test_same_definition_renewal_keeps_upgrade_reserved(store, fake_clock, renewal):
    from agentweave.errors import StoreConflict
    value = participant(fake_clock)
    await store.register_definition(value.logical_id, value.instance_id, value.fingerprint)
    fake_clock.advance(14)
    if renewal == 'registration':
        await store.register_definition(value.logical_id, value.instance_id, value.fingerprint)
    else:
        await store.publish_participant(value.model_copy(update={'heartbeat_at': fake_clock.now()}))
    fake_clock.advance(1)
    with pytest.raises(StoreConflict):
        await store.register_definition(value.logical_id, str(uuid4()), 'hash-v2')
    fake_clock.advance(14)
    await store.register_definition(value.logical_id, str(uuid4()), 'hash-v2')


def test_participant_only_accepts_public_fields(fake_clock):
    from pydantic import ValidationError
    for private in ('system_prompt', 'env', 'endpoint', 'command'):
        with pytest.raises(ValidationError):
            participant(fake_clock, **{private:'secret'})
    with pytest.raises(ValidationError):
        participant(fake_clock, kind='tool')
    with pytest.raises(ValidationError):
        participant(fake_clock, schema_version=2)


def child_publish(url, path, epoch, ready, stop, queue):
    from agentweave._records import SystemClock
    from agentweave._directory import LiveDirectory
    from agentweave.adapters.broker import BrokerBus
    from agentweave.adapters.storage.sqlite import SQLiteStore
    async def run():
        clock = SystemClock()
        store = SQLiteStore(path, clock=clock)
        bus = BrokerBus(url, NETWORK, profile='dev', broker_epoch=epoch)
        directory = LiveDirectory(bus, store, NETWORK, broker_epoch=epoch, clock=clock)
        try:
            await store.migrate()
            await directory.start()
            value = participant(clock, logical_id='calculator.add', kind='tool', name='add', revision='r1',
                input_schema={'type':'object'}, output_schema={'type':'integer'})
            await directory.publish(value)
            ready.set()
            async with asyncio.timeout(25):
                while not stop.is_set():
                    await asyncio.sleep(.1)
            queue.put('ok')
        except Exception as error:
            queue.put(repr(error))
        finally:
            await directory.close()
            await bus.close()
            await store.close()
    asyncio.run(run())


@pytest.mark.integration
async def test_independent_process_discovered_and_removed_without_restart(broker_process, tmp_path):
    from agentweave._records import SystemClock
    from agentweave._directory import LiveDirectory
    from agentweave.adapters.broker import BrokerBus
    from agentweave.adapters.storage.sqlite import SQLiteStore
    ctx = multiprocessing.get_context('spawn')
    ready = [ctx.Event(), ctx.Event()]
    stops = [ctx.Event(), ctx.Event()]
    queue = ctx.Queue()
    bus = BrokerBus(broker_process.url, NETWORK, profile='dev', broker_epoch=broker_process.epoch)
    store = SQLiteStore(tmp_path / 'observer.sqlite3')
    directory = LiveDirectory(bus, store, NETWORK, broker_epoch=broker_process.epoch, clock=SystemClock())
    processes = [ctx.Process(target=child_publish, args=(bus.broker_url,
        str(tmp_path / 'replicas.sqlite3'), bus.epoch, ready[index], stops[index], queue))
        for index in range(2)]
    try:
        await store.migrate()
        await directory.start()
        assert directory.snapshot() == {}
        for process in processes:
            process.start()
        for signal in ready:
            assert await asyncio.to_thread(signal.wait, 15)
        await eventually(lambda: len(directory.snapshot().get('calculator.add', ())) == 2)
        # Automatic heartbeat keeps the child visible beyond a complete expiry window.
        await asyncio.sleep(16)
        assert len(directory.snapshot()['calculator.add']) == 2
        assert directory.resolve_tool('calculator_add', revision='r1') is not None
        stops[0].set()
        await eventually(lambda: len(directory.snapshot().get('calculator.add', ())) == 1)
        assert directory.resolve_tool('calculator_add', revision='r1') is not None
        stops[1].set()
        await eventually(lambda: directory.snapshot() == {})
        for _ in processes:
            assert await asyncio.to_thread(queue.get, True, 10) == 'ok'
    finally:
        for stop in stops:
            stop.set()
        for process in processes:
            if process.pid is not None:
                await asyncio.to_thread(process.join, 10)
                if process.is_alive():
                    process.kill()
                    await asyncio.to_thread(process.join)
        queue.close()
        await directory.close()
        await bus.close()
        await store.close()
