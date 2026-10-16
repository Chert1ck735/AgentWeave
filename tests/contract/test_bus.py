"""Real transport checks, including committed positions and bounded isolation."""
import asyncio
from uuid import uuid4
import pytest

NETWORK = '00000000-0000-0000-0000-000000000010'

@pytest.fixture
async def bus(broker_url):
    from agentweave.adapters.broker import BrokerBus
    value = BrokerBus(broker_url, NETWORK, profile='dev', handoff_capacity=2)
    yield value
    await value.close()

async def take(subscription):
    return await asyncio.wait_for(anext(subscription), 20)

@pytest.mark.integration
async def test_key_routing_namespace_and_capabilities(bus, envelope_factory):
    from agentweave.ports import TopicSpec
    name = 'aw.' + NETWORK + '.inbox.agent.assistant'
    await bus.ensure_topics([TopicSpec(name=name, keyed=True, partitions=2, retention_ms=60000)])
    await bus.ensure_topics([TopicSpec(name=name, keyed=True, partitions=2, retention_ms=60000)])
    with pytest.raises(ValueError):
        await bus.ensure_topics([TopicSpec(name='aw.foreign.events.test', keyed=True, partitions=2, retention_ms=60000)])
    capabilities = bus.capabilities()
    assert capabilities.keyed_ordering and not capabilities.durable_stream
    assert not capabilities.persistent_consumer_offsets
    sub = await bus.subscribe([name], str(uuid4()))
    try:
        events = [envelope_factory(key='same-key') for _ in range(2)]
        for event in events:
            await bus.publish(event)
        received = [await take(sub), await take(sub)]
        assert [d.envelope for d in received] == events
        assert received[0].position.partition == received[1].position.partition
        assert received[0].position.key == b'same-key'
        for delivery in received:
            await bus.ack(delivery)
    finally:
        await sub.aclose()

@pytest.mark.integration
async def test_ack_cannot_skip_unaccepted_offset(bus, envelope_factory):
    import confluent_kafka as ck
    first, second = envelope_factory(key='ordered'), envelope_factory(key='ordered')
    await bus.publish(first)
    await bus.publish(second)
    group = str(uuid4())
    sub = await bus.subscribe([bus.topic_for(first)], group)
    try:
        one, two = await take(sub), await take(sub)
        await bus.ack(two)
        inspector = ck.Consumer({'bootstrap.servers':bus.broker_url, 'group.id':group, 'log_level':0})
        try:
            committed = await asyncio.to_thread(inspector.committed,
                [ck.TopicPartition(one.position.topic, one.position.partition)], 6)
            assert committed[0].offset < one.position.offset + 1
            await bus.ack(one)
            committed = await asyncio.to_thread(inspector.committed,
                [ck.TopicPartition(one.position.topic, one.position.partition)], 6)
            assert committed[0].offset == two.position.offset + 1
        finally:
            await asyncio.to_thread(inspector.close)
    finally:
        await sub.aclose()

@pytest.mark.integration
async def test_full_subscription_does_not_block_control_consumer(bus, envelope_factory):
    busy = envelope_factory(key='busy')
    await bus.publish(busy)
    sub = await bus.subscribe([bus.topic_for(busy)], str(uuid4()))
    control = envelope_factory(kind='progress', recipient='client.one', key='control')
    await bus.publish(control)
    other = await bus.subscribe([bus.topic_for(control)], str(uuid4()))
    try:
        for _ in range(8):
            await bus.publish(envelope_factory(key='busy'))
        await asyncio.sleep(.5)
        delivery = await take(other)
        assert delivery.envelope.event_id == control.event_id
        assert sub.buffered <= 2
        await bus.ack(delivery)
    finally:
        await sub.aclose()
        await other.aclose()


@pytest.mark.integration
async def test_epoch_change_invalidates_buffered_delivery_and_reprovisions(bus, broker_process, envelope_factory):
    event = envelope_factory()
    await bus.publish(event)
    sub = await bus.subscribe([bus.topic_for(event)], str(uuid4()))
    old = await take(sub)
    await bus.publish(envelope_factory(key=event.session_id))
    await asyncio.sleep(.2)
    broker_process.restart()
    await bus.broker_epoch_changed(broker_process.epoch)
    with pytest.raises(ValueError):
        await bus.ack(old)
    with pytest.raises(StopAsyncIteration):
        await anext(sub)
    await bus.publish(event)
    fresh = await bus.subscribe([bus.topic_for(event)], str(uuid4()))
    try:
        current = await take(fresh)
        assert current.envelope == event
        await bus.ack(current)
    finally:
        await fresh.aclose()


@pytest.mark.integration
async def test_malformed_input_reports_sanitized_error_and_continues(bus, envelope_factory):
    import confluent_kafka as ck
    from agentweave.ports import TopicSpec
    event = envelope_factory(key='malformed-test')
    name = bus.topic_for(event)
    await bus.ensure_topics([TopicSpec(name)])
    errors = await bus.subscribe(['aw.' + NETWORK + '.errors'], str(uuid4()))
    sub = await bus.subscribe([name], str(uuid4()))
    producer = ck.Producer({'bootstrap.servers':bus.broker_url, 'log_level':0})
    producer.produce(name, key=b'malformed-test', value=b'not-json SECRET-PAYLOAD')
    assert await asyncio.to_thread(producer.flush, 10) == 0
    await bus.publish(event)
    try:
        delivery = await take(sub)
        assert delivery.envelope == event
        error = await take(errors)
        assert error.envelope.kind == 'error'
        assert error.envelope.payload['code'] == 'invalid_envelope'
        assert 'SECRET' not in str(error.envelope.payload)
        await bus.ack(delivery)
        await bus.ack(error)
        # Malformed records already in the error stream must be consumed without
        # recursively reporting another error to that same stream.
        producer.produce(error.position.topic, value=b'bad SECRET-ERROR')
        assert await asyncio.to_thread(producer.flush, 10) == 0
        sentinel = envelope_factory(kind='error', recipient=None, topic='errors', payload={'marker':'next'})
        await bus.publish(sentinel)
        resumed = await take(errors)
        assert resumed.envelope == sentinel
        await bus.ack(resumed)
        inspector = ck.Consumer({'bootstrap.servers':bus.broker_url, 'group.id':str(uuid4()), 'log_level':0})
        try:
            low, high = await asyncio.to_thread(inspector.get_watermark_offsets,
                ck.TopicPartition(error.position.topic, 0), 6)
            assert high == 3  # diagnostic, malformed error record, sentinel only
        finally:
            await asyncio.to_thread(inspector.close)
    finally:
        await sub.aclose()
        await errors.aclose()


@pytest.mark.parametrize('changes', [dict(kind='receipt', recipient=None, topic='receipts'),
    dict(kind='receipt', payload={'event_id':'bad', 'recipient':'assistant'}),
    dict(kind='directory', recipient=None, topic='arbitrary'),
    dict(kind='error', recipient=None, topic='arbitrary')])
async def test_invalid_control_shapes_are_never_published(bus, envelope_factory, changes):
    with pytest.raises(ValueError):
        await bus.publish(envelope_factory(**changes))


async def test_named_topic_helpers_route_validated_destinations(bus, envelope_factory):
    assert bus.agent_topic('assistant') == 'aw.' + NETWORK + '.inbox.agent.assistant'
    assert bus.tool_topic('tools.lookup') == 'aw.' + NETWORK + '.inbox.tool.tools.lookup'
    assert bus.events_topic('weather') == 'aw.' + NETWORK + '.events.weather'
    assert bus.control_topic('receipt') == 'aw.' + NETWORK + '.receipts'
    assert bus.topic_for(envelope_factory(kind='tool_call', recipient='tools.lookup')) == 'aw.' + NETWORK + '.inbox.tool.tools.lookup'
    with pytest.raises(ValueError):
        bus.agent_topic('assistant/foreign')
    with pytest.raises(ValueError):
        bus.control_topic('custom')


@pytest.mark.integration
async def test_consumer_caps_unacknowledged_handoff_without_committing(bus, envelope_factory):
    events = [envelope_factory(key='bounded') for _ in range(3)]
    for event in events:
        await bus.publish(event)
    sub = await bus.subscribe([bus.topic_for(events[0])], str(uuid4()))
    try:
        first, second = await take(sub), await take(sub)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(anext(sub), 2)
        await bus.ack(first)
        third = await take(sub)
        assert third.envelope == events[2]
        await bus.ack(second)
        await bus.ack(third)
    finally:
        await sub.aclose()


@pytest.mark.integration
async def test_concurrent_incompatible_topic_specs_cannot_both_succeed(bus):
    import confluent_kafka as ck
    from confluent_kafka.admin import AdminClient
    from agentweave.ports import TopicSpec
    name = bus.events_topic('concurrent')
    outcomes = await asyncio.gather(
        bus.ensure_topics([TopicSpec(name, partitions=2)]),
        bus.ensure_topics([TopicSpec(name, partitions=1)]), return_exceptions=True)
    assert outcomes[0] is None
    assert isinstance(outcomes[1], ValueError)
    metadata = await asyncio.to_thread(AdminClient({'bootstrap.servers':bus.broker_url,
        'log_level':0}).list_topics, timeout=6)
    assert len(metadata.topics[name].partitions) == 2


@pytest.mark.integration
async def test_contradictory_topic_specs_in_one_batch_are_rejected(bus):
    from confluent_kafka.admin import AdminClient
    from agentweave.ports import TopicSpec
    name = bus.events_topic('contradictory')
    with pytest.raises(ValueError):
        await bus.ensure_topics([TopicSpec(name, partitions=2), TopicSpec(name, partitions=1)])
    metadata = await asyncio.to_thread(AdminClient({'bootstrap.servers':bus.broker_url,
        'log_level':0}).list_topics, timeout=6)
    assert name not in metadata.topics


@pytest.fixture
async def malformed_triplet(bus, envelope_factory):
    import confluent_kafka as ck
    from agentweave.messages import encode
    from agentweave.ports import TopicSpec
    first, following = envelope_factory(key='recover'), envelope_factory(key='recover')
    name = bus.topic_for(first)
    await bus.ensure_topics([TopicSpec(name)])
    producer = ck.Producer({'bootstrap.servers':bus.broker_url, 'log_level':0})
    for raw in (b'bad SECRET', encode(first), encode(following)):
        producer.produce(name, partition=0, key=b'recover', value=raw)
        assert await asyncio.to_thread(producer.flush, 10) == 0
    group = str(uuid4())
    sub = await bus.subscribe([name], group)
    async with asyncio.timeout(15):
        while sub.buffered != 2:
            await asyncio.sleep(.01)
    try:
        yield sub, group, first, following
    finally:
        await sub.aclose()


async def committed_offset(bus, group, position):
    import confluent_kafka as ck
    inspector = ck.Consumer({'bootstrap.servers':bus.broker_url, 'group.id':group, 'log_level':0})
    try:
        actual = await asyncio.to_thread(inspector.committed,
            [ck.TopicPartition(position.topic, position.partition)], 6)
        return actual[0].offset
    finally:
        await asyncio.to_thread(inspector.close)


async def assert_recovered_records_commit(bus, malformed_triplet):
    sub, group, first, following = malformed_triplet
    resumed = await take(sub)
    assert resumed.envelope == first
    await bus.ack(resumed)
    assert await committed_offset(bus, group, resumed.position) == 2
    later = await take(sub)
    assert later.envelope == following
    await bus.ack(later)
    assert await committed_offset(bus, group, later.position) == 3


@pytest.mark.integration
async def test_cancelled_malformed_diagnostic_retries_before_later_delivery(bus, malformed_triplet):
    sub, _, _, _ = malformed_triplet
    errors = await bus.subscribe([bus.control_topic('error')], str(uuid4()))
    await bus._producer_lock.acquire()
    interrupted = asyncio.create_task(anext(sub))
    try:
        async with asyncio.timeout(10):
            while sub.buffered != 1:
                await asyncio.sleep(.01)
        interrupted.cancel()
        with pytest.raises(asyncio.CancelledError):
            await interrupted
    finally:
        bus._producer_lock.release()
        if not interrupted.done():
            interrupted.cancel()
            await asyncio.gather(interrupted, return_exceptions=True)
    try:
        await assert_recovered_records_commit(bus, malformed_triplet)
        error = await take(errors)
        assert error.envelope.payload['offset'] == 0
        assert 'SECRET' not in str(error.envelope.payload)
        await bus.ack(error)
    finally:
        await errors.aclose()


@pytest.mark.integration
async def test_unavailable_malformed_diagnostic_retries_before_later_delivery(bus, malformed_triplet):
    import socket
    from agentweave.errors import BrokerUnavailable
    sub, _, _, _ = malformed_triplet
    healthy = bus.broker_url
    # The native input consumer already owns its healthy connection. A bound,
    # non-listening local socket makes only new publication/provisioning clients
    # fail with a real native transport timeout; no broker/store substitute.
    with socket.socket() as unavailable:
        unavailable.bind(('127.0.0.1', 0))
        bus.broker_url = f'127.0.0.1:{unavailable.getsockname()[1]}'
        try:
            with pytest.raises(BrokerUnavailable):
                await take(sub)
        finally:
            bus.broker_url = healthy
    await assert_recovered_records_commit(bus, malformed_triplet)
    errors = await bus.subscribe([bus.control_topic('error')], str(uuid4()))
    try:
        diagnostic = await take(errors)
        assert diagnostic.envelope.payload['offset'] == 0
        await bus.ack(diagnostic)
    finally:
        await errors.aclose()


@pytest.mark.integration
async def test_cancelled_malformed_ack_retains_its_real_committed_command(bus, malformed_triplet, monkeypatch):
    import confluent_kafka as ck
    sub, _, _, _ = malformed_triplet
    errors = await bus.subscribe([bus.control_topic('error')], str(uuid4()))
    native_ack = sub.ack
    committed = asyncio.Event()
    release = asyncio.Event()

    async def acknowledge_then_pause(position):
        # The real owner/native command commits first; the gate only delays
        # observation of its completion to reproduce iterator cancellation.
        await native_ack(position)
        committed.set()
        await release.wait()

    monkeypatch.setattr(sub, 'ack', acknowledge_then_pause)
    interrupted = asyncio.create_task(anext(sub))
    try:
        await asyncio.wait_for(committed.wait(), 15)
        interrupted.cancel()
        with pytest.raises(asyncio.CancelledError):
            await interrupted
    finally:
        release.set()
        monkeypatch.setattr(sub, 'ack', native_ack)
        if not interrupted.done():
            interrupted.cancel()
            await asyncio.gather(interrupted, return_exceptions=True)
    try:
        await assert_recovered_records_commit(bus, malformed_triplet)
        diagnostic = await take(errors)
        await bus.ack(diagnostic)
        inspector = ck.Consumer({'bootstrap.servers':bus.broker_url, 'group.id':str(uuid4()), 'log_level':0})
        try:
            _, high = await asyncio.to_thread(inspector.get_watermark_offsets,
                ck.TopicPartition(bus.control_topic('error'), 0), 6)
            assert high == 1
        finally:
            await asyncio.to_thread(inspector.close)
    finally:
        await errors.aclose()


@pytest.mark.integration
@pytest.mark.parametrize('invalidate', ['close', 'epoch'])
async def test_retained_malformed_position_is_invalidated_on_close_or_epoch(bus, malformed_triplet, broker_process, invalidate):
    from agentweave.ports import TopicSpec
    sub, _, first, _ = malformed_triplet
    await bus.ensure_topics([TopicSpec(bus.control_topic('error'), partitions=1)])
    async with bus._producer_lock:
        interrupted = asyncio.create_task(anext(sub))
        try:
            async with asyncio.timeout(10):
                while sub.buffered != 1:
                    await asyncio.sleep(.01)
            interrupted.cancel()
            with pytest.raises(asyncio.CancelledError):
                await interrupted
        finally:
            if not interrupted.done():
                interrupted.cancel()
                await asyncio.gather(interrupted, return_exceptions=True)
    if invalidate == 'epoch':
        broker_process.restart()
        await bus.broker_epoch_changed(broker_process.epoch)
    else:
        await sub.aclose()
    with pytest.raises(StopAsyncIteration):
        await anext(sub)
    # A replacement iterator publishes/consumes normally; stale retained work
    # cannot commit or suppress delivery in the new subscription/epoch.
    await bus.publish(first)
    replacement = await bus.subscribe([bus.topic_for(first)], str(uuid4()))
    try:
        # For close, old records remain in the stream; consume them through the
        # normal diagnostic path until the requested known valid event appears.
        delivery = await take(replacement)
        assert delivery.envelope == first
        await bus.ack(delivery)
    finally:
        await replacement.aclose()


@pytest.mark.integration
async def test_native_batched_source_keeps_following_record_after_backpressure(bus, envelope_factory):
    import confluent_kafka as ck
    from agentweave.messages import encode
    from agentweave.ports import TopicSpec
    events = [envelope_factory(key='batch-source') for _ in range(3)]
    topic = bus.topic_for(events[0])
    await bus.ensure_topics([TopicSpec(topic)])
    producer = ck.Producer({'bootstrap.servers':bus.broker_url, 'log_level':0})
    for event in events:
        producer.produce(topic, partition=0, key=b'batch-source', value=encode(event))
    assert await asyncio.to_thread(producer.flush, 10) == 0
    sub = await bus.subscribe([topic], str(uuid4()))
    try:
        one, two = await take(sub), await take(sub)
        assert [one.envelope, two.envelope] == events[:2]
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(anext(sub), .5)
        await bus.ack(one)
        await bus.ack(two)
        third = await asyncio.wait_for(anext(sub), 3)
        assert third.envelope == events[2]
        await bus.ack(third)
        assert await committed_offset(bus, one.position.group_id, third.position) == 3
    finally:
        await sub.aclose()
