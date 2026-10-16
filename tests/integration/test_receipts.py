"""File-backed admission/outbox against the actual packaged broker."""
import asyncio
import sqlite3
from uuid import uuid4
import pytest
from agentweave.errors import StoreUnavailable

OWNER = '00000000-0000-0000-0000-000000000001'
NETWORK = '00000000-0000-0000-0000-000000000010'

def saved(store, sql):
    with sqlite3.connect(store.path) as c:
        return c.execute(sql).fetchall()

async def take(sub):
    return await asyncio.wait_for(anext(sub), 20)

@pytest.fixture
async def delivery_components(broker_process, store, store_factory, tmp_path, fake_clock):
    from agentweave.adapters.broker import BrokerBus
    from agentweave._runtime.delivery import AdmissionService, OutboxPump
    bus = BrokerBus(broker_process.url, NETWORK, profile='dev', broker_epoch=broker_process.epoch)
    receiver = await store_factory(tmp_path / 'receiver.sqlite3')
    yield bus, receiver, OutboxPump(store, bus, OWNER, clock=fake_clock), OutboxPump(receiver, bus, str(uuid4()), clock=fake_clock), AdmissionService(receiver, bus), AdmissionService(store, bus)
    await bus.close()

@pytest.mark.integration
async def test_publish_ack_is_not_durable_receipt(delivery_components, broker_process, store, fake_clock, envelope_factory):
    bus, receiver, sender_pump, receiver_pump, admission, receipts = delivery_components
    event = envelope_factory()
    await store.submit(event)
    assert await sender_pump.run_once() == 1
    broker_process.stop()
    assert saved(store, 'SELECT received_targets FROM outbox') == [('[]',)]
    broker_process.start()
    await bus.broker_epoch_changed(broker_process.epoch)
    fake_clock.advance(1)
    assert await sender_pump.run_once() == 1
    sub = await bus.subscribe([bus.topic_for(event)], str(uuid4()))
    controls = await bus.subscribe(['aw.' + NETWORK + '.receipts'], str(uuid4()))
    try:
        result = await admission.accept(await take(sub), 'v1')
        assert not result.duplicate
        assert await receiver_pump.run_once() == 1
        await receipts.accept_receipt(await take(controls))
        fake_clock.advance(5)
        assert await store.lease_outbox(OWNER, 10) == []
        assert len(saved(receiver, 'SELECT * FROM tasks')) == 1
        assert len(saved(store, 'SELECT * FROM tasks')) == 0
        assert len(saved(store, 'SELECT * FROM outbox')) == 1
    finally:
        await sub.aclose()
        await controls.aclose()

@pytest.mark.integration
async def test_lost_receipt_repeats_same_receipt_without_second_task(delivery_components, broker_process, store, fake_clock, envelope_factory):
    bus, receiver, sender_pump, receiver_pump, admission, receipts = delivery_components
    event = envelope_factory()
    await store.submit(event)
    await sender_pump.run_once()
    sub = await bus.subscribe([bus.topic_for(event)], str(uuid4()))
    first = await admission.accept(await take(sub), 'original')
    await receiver_pump.run_once()
    await sub.aclose()
    broker_process.restart()
    await bus.broker_epoch_changed(broker_process.epoch)
    fake_clock.advance(1)
    await sender_pump.run_once()
    sub = await bus.subscribe([bus.topic_for(event)], str(uuid4()))
    controls = await bus.subscribe(['aw.' + NETWORK + '.receipts'], str(uuid4()))
    try:
        duplicate = await admission.accept(await take(sub), 'changed')
        assert duplicate.duplicate and duplicate.receipt == first.receipt
        assert (await receiver.get_task(event.task_id)).definition_hash == 'original'
        await receiver_pump.run_once()
        await receipts.accept_receipt(await take(controls))
        assert saved(store, 'SELECT received_targets FROM outbox') == [('["assistant"]',)]
        assert len(saved(receiver, 'SELECT * FROM tasks')) == 1
    finally:
        await sub.aclose()
        await controls.aclose()

@pytest.mark.integration
async def test_admission_crash_before_ack_deduplicates_replayed_input(delivery_components, envelope_factory):
    bus, receiver, _, _, admission, _ = delivery_components
    event = envelope_factory()
    await bus.publish(event)
    group = str(uuid4())
    sub = await bus.subscribe([bus.topic_for(event)], group)
    delivery = await take(sub)
    await sub.aclose()
    with pytest.raises(ValueError):
        await admission.accept(delivery, 'v1')
    assert (await receiver.get_task(event.task_id)).status == 'accepted'
    sub = await bus.subscribe([bus.topic_for(event)], group)
    try:
        duplicate = await admission.accept(await take(sub), 'v2')
        assert duplicate.duplicate
        assert len(saved(receiver, 'SELECT * FROM tasks')) == 1
    finally:
        await sub.aclose()

@pytest.mark.integration
async def test_store_failure_does_not_commit_input(delivery_components, envelope_factory):
    import confluent_kafka as ck
    bus, receiver, _, _, admission, _ = delivery_components
    event = envelope_factory()
    await bus.publish(event)
    group = str(uuid4())
    sub = await bus.subscribe([bus.topic_for(event)], group)
    try:
        delivery = await take(sub)
        await receiver.close()
        with pytest.raises(StoreUnavailable):
            await admission.accept(delivery, 'v1')
        inspector = ck.Consumer({'bootstrap.servers':bus.broker_url, 'group.id':group, 'log_level':0})
        try:
            committed = await asyncio.to_thread(inspector.committed,
                [ck.TopicPartition(delivery.position.topic, delivery.position.partition)], 6)
            assert committed[0].offset < delivery.position.offset + 1
        finally:
            await asyncio.to_thread(inspector.close)
    finally:
        await sub.aclose()

@pytest.mark.integration
async def test_addressed_backoff_and_deadline_are_persisted(delivery_components, store, fake_clock, envelope_factory):
    _, _, pump, _, _, _ = delivery_components
    event = envelope_factory()
    await store.submit(event)
    for delay in (1, 2, 4, 5, 5):
        before = fake_clock.now()
        assert await pump.run_once() == 1
        from datetime import datetime
        due = datetime.fromisoformat(saved(store, 'SELECT retry_at FROM outbox')[0][0])
        assert (due - before).total_seconds() == delay
        fake_clock.advance(delay - .1)
        assert await pump.run_once() == 0
        fake_clock.advance(.1)
    fake_clock.advance(120)
    assert await pump.run_once() == 0

@pytest.mark.integration
async def test_failed_publish_releases_and_persists_retry(delivery_components, broker_process, store, fake_clock, envelope_factory):
    bus, _, pump, _, _, _ = delivery_components
    event = envelope_factory()
    # Topic provision succeeds before simulating a temporary transport failure.
    await bus.publish(event)
    await store.submit(event)
    broker_process.stop()
    assert await pump.run_once() == 0
    assert saved(store, 'SELECT published_at,lease_instance_id,received_targets FROM outbox') == [(None, None, '[]')]
    broker_process.start()
    await bus.broker_epoch_changed(broker_process.epoch)
    fake_clock.advance(1)
    assert await pump.run_once() == 1


@pytest.mark.integration
@pytest.mark.parametrize('status', ['succeeded', 'failed', 'cancelled', 'expired'])
async def test_terminal_task_result_reply_confirms_original_input(delivery_components, store, envelope_factory, status):
    bus, _, _, _, _, receipts = delivery_components
    event = envelope_factory()
    await store.submit(event)
    reply = envelope_factory(kind='reply', sender='assistant', recipient='client.one',
        task_id=event.task_id, correlation_id=event.event_id,
        payload={'task_id':event.task_id, 'output':{'answer':42}, 'responding_agent':'assistant', 'status':status})
    await bus.publish(reply)
    sub = await bus.subscribe([bus.topic_for(reply)], str(uuid4()))
    try:
        await receipts.accept_receipt(await take(sub))
        assert saved(store, 'SELECT received_targets FROM outbox') == [('["assistant"]',)]
        assert saved(store, 'SELECT * FROM tasks') == []
    finally:
        await sub.aclose()


@pytest.mark.integration
@pytest.mark.parametrize('change', [dict(status='outcome_unknown'), dict(status='running'),
    dict(task_id='00000000-0000-0000-0000-000000000099'), dict(responding_agent='other'),
    dict(status='succeeded', extra='unexpected'), dict(responding_agent=None)])
async def test_invalid_or_nonterminal_reply_does_not_confirm_input(delivery_components, store, envelope_factory, change):
    bus, _, _, _, _, receipts = delivery_components
    event = envelope_factory()
    await store.submit(event)
    result = {'task_id':event.task_id, 'output':None, 'responding_agent':'assistant', 'status':'succeeded'} | change
    reply = envelope_factory(kind='reply', sender='assistant', recipient='client.one',
        task_id=event.task_id, correlation_id=event.event_id, payload=result)
    await bus.publish(reply)
    sub = await bus.subscribe([bus.topic_for(reply)], str(uuid4()))
    try:
        delivery = await take(sub)
        with pytest.raises(ValueError):
            await receipts.accept_receipt(delivery)
        assert saved(store, 'SELECT received_targets FROM outbox') == [('[]',)]
        await bus.ack(delivery)
    finally:
        await sub.aclose()


@pytest.mark.integration
async def test_controls_do_not_recursively_admit_tasks(delivery_components, store, envelope_factory):
    bus, _, _, _, admission, _ = delivery_components
    control = envelope_factory(kind='progress', recipient=None, topic='progress')
    await bus.publish(control)
    sub = await bus.subscribe([bus.topic_for(control)], str(uuid4()))
    try:
        delivery = await take(sub)
        with pytest.raises(ValueError):
            await admission.accept(delivery, 'v1', agent_id='assistant')
        assert saved(store, 'SELECT * FROM tasks') == []
        await bus.ack(delivery)
    finally:
        await sub.aclose()


@pytest.mark.integration
async def test_pubsub_ack_finishes_outbox_without_subscriber_receipts(delivery_components, store, fake_clock, envelope_factory):
    _, _, pump, _, _, _ = delivery_components
    event = envelope_factory(recipient=None, topic='news')
    await store.submit(event)
    assert await pump.run_once() == 1
    fake_clock.advance(10)
    assert await pump.run_once() == 0
    assert saved(store, 'SELECT received_targets FROM outbox') == [('[]',)]


@pytest.mark.integration
async def test_running_pump_stops_after_real_publication(delivery_components, store, envelope_factory):
    bus, _, pump, _, _, _ = delivery_components
    event = envelope_factory()
    await store.submit(event)
    sub = await bus.subscribe([bus.topic_for(event)], str(uuid4()))
    running = asyncio.create_task(pump.run())
    try:
        delivery = await take(sub)
        pump.stop()
        await asyncio.wait_for(running, 15)
        await bus.ack(delivery)
        assert saved(store, 'SELECT received_targets FROM outbox') == [('[]',)]
    finally:
        pump.stop()
        await running
        await sub.aclose()


@pytest.mark.integration
async def test_unavailable_topic_provisioning_keeps_pump_retryable(delivery_components, broker_process, store, fake_clock, envelope_factory):
    bus, _, pump, _, _, _ = delivery_components
    event = envelope_factory()
    await store.submit(event)
    broker_process.stop()
    assert await pump.run_once() == 0
    assert saved(store, 'SELECT lease_instance_id,published_at FROM outbox') == [(None, None)]
    broker_process.start()
    await bus.broker_epoch_changed(broker_process.epoch)
    fake_clock.advance(1)
    assert await pump.run_once() == 1


@pytest.mark.integration
async def test_topic_groups_admit_once_per_consuming_agent(delivery_components, store, envelope_factory):
    bus, receiver, pump, _, admission, _ = delivery_components
    event = envelope_factory(recipient=None, topic='news', task_id=None)
    # Topic publication via the actual bus does not require a request task_id.
    await bus.publish(event)
    left = await bus.subscribe([bus.topic_for(event)], str(uuid4()))
    right = await bus.subscribe([bus.topic_for(event)], str(uuid4()))
    try:
        first = await admission.accept(await take(left), 'v1', agent_id='first')
        second = await admission.accept(await take(right), 'v2', agent_id='second')
        assert first.task_id != second.task_id
        assert sorted(saved(receiver, 'SELECT agent_id FROM tasks')) == [('first',), ('second',)]
        assert (await receiver.get_input_envelope(first.task_id)) == event
        assert (await receiver.get_input_envelope(second.task_id)) == event
    finally:
        await left.aclose()
        await right.aclose()


@pytest.mark.integration
async def test_pump_respects_false_ack_after_real_receipt_reactivation(delivery_components, envelope_factory, store_factory, monkeypatch):
    bus, receiver, _, pump, _, _ = delivery_components
    event = envelope_factory()
    admission = await receiver.admit(event, 'v1')
    rival = await store_factory(receiver.path)
    native_publish = bus.publish

    async def publish_then_reactivate(outgoing):
        # This timing wrapper delegates publication to the real broker and
        # reactivation to another real SQLite connection. Neither guarantee is
        # replaced by a stub; it fixes the crash-window ordering deterministically.
        await native_publish(outgoing)
        await rival.admit(event, 'v1')

    monkeypatch.setattr(bus, 'publish', publish_then_reactivate)
    assert await pump.run_once() == 0
    assert saved(receiver, 'SELECT published_at,lease_instance_id FROM outbox') == [(None, None)]
    assert len(saved(receiver, 'SELECT * FROM tasks')) == 1
    monkeypatch.setattr(bus, 'publish', native_publish)
    assert await pump.run_once() == 1
    assert saved(receiver, 'SELECT event_id FROM outbox') == [(admission.receipt.event_id,)]
    assert saved(receiver, 'SELECT published_at FROM outbox')[0][0] is not None


@pytest.mark.integration
async def test_broadcast_admission_receipt_advances_without_false_target(delivery_components, store, envelope_factory):
    bus, receiver, sender_pump, receiver_pump, admission, receipts = delivery_components
    event = envelope_factory(recipient=None, topic='news')
    await store.submit(event)
    await sender_pump.run_once()
    sub = await bus.subscribe([bus.topic_for(event)], str(uuid4()))
    try:
        await admission.accept(await take(sub), 'v1', agent_id='listener')
        before = saved(store, 'SELECT received_targets,version,published_at FROM outbox')
        # A topic publisher finished on the broker ACK: the receipt is journaled for dedup, never sent.
        assert saved(receiver, "SELECT delivered FROM outbox WHERE json_extract(envelope,'$.kind')='receipt'") == [(1,)]
        assert await receiver_pump.run_once() == 0
        assert saved(store, 'SELECT received_targets,version,published_at FROM outbox') == before
        assert await sender_pump.run_once() == 0
    finally:
        await sub.aclose()


def bus_control_topics(bus):
    return [bus.control_topic('receipt')]


@pytest.mark.integration
async def test_unmatched_receipt_does_not_block_matching_receipt(delivery_components, store, envelope_factory):
    bus, receiver, _, _, _, receipts = delivery_components
    event = envelope_factory()
    await store.submit(event)
    match = (await receiver.admit(event, 'v1')).receipt
    unknown_id = str(uuid4())
    foreign = envelope_factory(kind='receipt', sender='other', recipient='client.other',
        correlation_id=unknown_id, payload={'event_id':unknown_id, 'recipient':'other'})
    await bus.publish(foreign)
    await bus.publish(match)
    controls = await bus.subscribe(bus_control_topics(bus), str(uuid4()))
    try:
        await receipts.accept_receipt(await take(controls))
        assert saved(store, 'SELECT received_targets FROM outbox') == [('[]',)]
        await receipts.accept_receipt(await take(controls))
        assert saved(store, 'SELECT received_targets FROM outbox') == [('["assistant"]',)]
        assert saved(store, 'SELECT * FROM tasks') == []
    finally:
        await controls.aclose()


@pytest.mark.integration
async def test_receipt_store_failure_is_not_ignored_or_acknowledged(delivery_components, store, envelope_factory):
    import confluent_kafka as ck
    bus, receiver, _, _, _, receipts = delivery_components
    event = envelope_factory()
    await store.submit(event)
    receipt = (await receiver.admit(event, 'v1')).receipt
    await bus.publish(receipt)
    group = str(uuid4())
    controls = await bus.subscribe(bus_control_topics(bus), group)
    try:
        delivery = await take(controls)
        await store.close()
        with pytest.raises(StoreUnavailable):
            await receipts.accept_receipt(delivery)
        inspector = ck.Consumer({'bootstrap.servers':bus.broker_url, 'group.id':group, 'log_level':0})
        try:
            committed = await asyncio.to_thread(inspector.committed,
                [ck.TopicPartition(delivery.position.topic, delivery.position.partition)], 6)
            assert committed[0].offset < delivery.position.offset + 1
        finally:
            await asyncio.to_thread(inspector.close)
    finally:
        await controls.aclose()


@pytest.mark.integration
@pytest.mark.parametrize('changes', [dict(correlation_id='00000000-0000-0000-0000-000000000099'),
    dict(payload={'event_id':'bad', 'recipient':'assistant'}),
    dict(payload={'event_id':'00000000-0000-0000-0000-000000000099', 'recipient':'other'}),
    dict(payload=['not', 'a', 'receipt'])])
async def test_malformed_receipt_frame_is_not_treated_as_unmatched(delivery_components, store, envelope_factory, changes):
    from dataclasses import replace
    bus, receiver, _, _, _, receipts = delivery_components
    event = envelope_factory()
    await store.submit(event)
    receipt = (await receiver.admit(event, 'v1')).receipt
    await bus.publish(receipt)
    controls = await bus.subscribe(bus_control_topics(bus), str(uuid4()))
    try:
        delivery = await take(controls)
        invalid = replace(delivery, envelope=receipt.model_copy(update=changes))
        with pytest.raises(ValueError):
            await receipts.accept_receipt(invalid)
        assert saved(store, 'SELECT received_targets FROM outbox') == [('[]',)]
        await bus.ack(delivery)
    finally:
        await controls.aclose()
