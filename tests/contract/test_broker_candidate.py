"""Catch missing protocol support in the actual packaged broker, never a report stub."""
import pytest


@pytest.mark.integration
def test_candidate_supports_required_protocol(candidate_report):
    assert candidate_report == dict(create=True, keyed=True, two_groups=True,
        rebalance=True, commit=True, reconnect=True, payload_1mib=True)


@pytest.mark.integration
def test_native_batch_latest_and_group_reopen_preserve_record_offsets(broker_url):
    import time
    from uuid import uuid4
    import confluent_kafka as ck
    from confluent_kafka.admin import AdminClient, NewTopic
    name = 'aw.batch.' + uuid4().hex
    admin = AdminClient({'bootstrap.servers':broker_url, 'log_level':0})
    admin.create_topics([NewTopic(name, num_partitions=1, replication_factor=1)])[name].result(10)
    producer = ck.Producer({'bootstrap.servers':broker_url, 'linger.ms':1000, 'log_level':0})
    published = []
    for value in (b'zero', b'one', b'two'):
        producer.produce(name, partition=0, key=b'batch', value=value,
            on_delivery=lambda error, message: published.append((error, message.offset())))
    assert producer.flush(10) == 0
    assert published == [(None, 0), (None, 1), (None, 2)]
    group = 'aw-batch-' + uuid4().hex
    config = {'bootstrap.servers':broker_url, 'group.id':group, 'group.protocol':'classic',
        'enable.auto.commit':False, 'enable.auto.offset.store':False, 'auto.offset.reset':'earliest',
        'session.timeout.ms':6000, 'heartbeat.interval.ms':1000, 'max.poll.interval.ms':60000,
        'partition.assignment.strategy':'range', 'log_level':0}

    def take(consumer, count):
        received = []
        end = time.monotonic() + 12
        while len(received) < count and time.monotonic() < end:
            message = consumer.poll(.1)
            if message is None:
                continue
            if message.error():
                raise ck.KafkaException(message.error())
            received.append((message.offset(), message.value()))
        return received

    consumer = ck.Consumer(config)
    try:
        assert consumer.get_watermark_offsets(ck.TopicPartition(name, 0), timeout=6) == (0, 3)
        consumer.subscribe([name])
        assert take(consumer, 2) == [(0, b'zero'), (1, b'one')]
        consumer.commit(offsets=[ck.TopicPartition(name, 0, 2)], asynchronous=False)
        assert consumer.committed([ck.TopicPartition(name, 0)], timeout=6)[0].offset == 2
    finally:
        consumer.close()
    consumer = ck.Consumer(config)
    try:
        consumer.subscribe([name])
        assert take(consumer, 1) == [(2, b'two')]
        assert consumer.get_watermark_offsets(ck.TopicPartition(name, 0), timeout=6) == (0, 3)
    finally:
        consumer.close()
