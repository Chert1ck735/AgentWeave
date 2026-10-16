"""Native broker transport; each consumer is accessed by exactly one owner thread."""
import asyncio
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import queue
import threading
import time
from uuid import UUID, uuid4

from ..messages import Envelope, decode, encode, validate_name
from ..errors import InvalidEnvelope, BrokerUnavailable, StaleDelivery
from ..ports import BusCapabilities, Delivery, TopicSpec


def _raise_broker_error(error):
    import confluent_kafka as ck
    if error.retriable() or error.code() in (ck.KafkaError._MSG_TIMED_OUT,
            ck.KafkaError._TIMED_OUT, ck.KafkaError._TRANSPORT,
            ck.KafkaError._ALL_BROKERS_DOWN):
        raise BrokerUnavailable('broker operation failed temporarily')
    raise ck.KafkaException(error)


# A consumer polls this long when idle, and this briefly while a delivery awaits its ack.
IDLE_POLL_SECONDS = 0.05
ACK_POLL_SECONDS = 0.005
# Short polls last this long after a delivery: a handler blocked longer gets idle polls again.
ACK_WINDOW_SECONDS = 1.0
# Bounds how long a new record can wait behind an empty fetch the dev broker is holding, for
# subscriptions where latency matters (inboxes). Elsewhere the default keeps idle fetches rare:
# a short wait on every consumer halved the dev broker's throughput.
FETCH_WAIT_MS = 100


@dataclass(frozen=True)
class BrokerPosition:
    subscription_id: str
    group_id: str
    topic: str
    partition: int
    offset: int
    generation: int
    epoch: str
    key: bytes | None


@dataclass
class _MalformedDelivery:
    position: BrokerPosition
    diagnostic: Envelope | None
    published: bool = False
    ack_task: asyncio.Task | None = None


class BrokerBus:
    """Transport for the dev memory broker or durable Redpanda server profile.

    Runtime bootstrap must notify broker_epoch_changed when a dev broker restarts.
    Existing iterators then end; their callers must subscribe again.
    """
    def __init__(self, broker_url: str, network_id: str, *, profile: str,
                 broker_epoch: str = '', handoff_capacity: int = 64,
                 native_prefetch_messages: int = 64, native_prefetch_kib: int = 2048):
        if str(UUID(network_id)) != network_id:
            raise ValueError('network_id must be a canonical UUID')
        if profile not in ('dev', 'server'):
            raise ValueError('profile must be dev or server')
        if any(type(x) is not int or x < 1 for x in
               (handoff_capacity, native_prefetch_messages, native_prefetch_kib)):
            raise ValueError('transport buffer limits must be positive integers')
        self.broker_url, self.network_id, self.profile = broker_url, network_id, profile
        self.epoch = broker_epoch
        self.prefix = 'aw.' + network_id + '.'
        self.capacity = handoff_capacity
        self.prefetch_messages, self.prefetch_kib = native_prefetch_messages, native_prefetch_kib
        self._subscriptions = {}
        self._specs = {}
        self._provisioned = set()
        self._topics_lock = asyncio.Lock()
        self._producer_lock = asyncio.Lock()
        self._producer = None
        self._closed = False

    def capabilities(self) -> BusCapabilities:
        persistent = self.profile == 'server'
        return BusCapabilities(True, persistent, persistent, topic_compaction=persistent)

    def _validate_topic(self, name):
        if not name.startswith(self.prefix):
            raise ValueError('topic is outside the configured network namespace')
        suffix = name[len(self.prefix):]
        controls = ('directory', 'receipts', 'progress', 'errors')
        roots = ('inbox.agent.', 'inbox.tool.', 'events.')
        if suffix in controls:
            return
        for root in roots:
            if suffix.startswith(root):
                validate_name(suffix[len(root):])
                return
        raise ValueError('unsupported network topic')

    def topic_for(self, envelope: Envelope) -> str:
        if envelope.network_id != self.network_id:
            raise ValueError('envelope is outside the configured network')
        control = {'receipt':'receipts', 'directory':'directory',
                   'progress':'progress', 'error':'errors'}
        if envelope.kind in control:
            if envelope.kind == 'receipt':
                payload = envelope.payload
                if (envelope.recipient is None or envelope.correlation_id is None or
                    not isinstance(payload, dict) or set(payload) != {'event_id', 'recipient'} or
                    payload['event_id'] != envelope.correlation_id or payload['recipient'] != envelope.sender):
                    raise ValueError('receipt must be addressed with matching correlation and sender')
            elif envelope.recipient is None and envelope.topic != control[envelope.kind]:
                raise ValueError('broadcast control topic must match its reserved kind')
            return self.control_topic(envelope.kind)
        if envelope.topic is not None:
            return self.events_topic(envelope.topic)
        return self.tool_topic(envelope.recipient) if envelope.kind == 'tool_call' else self.agent_topic(envelope.recipient)

    def agent_topic(self, logical_id: str) -> str:
        return self.prefix + 'inbox.agent.' + validate_name(logical_id)

    def tool_topic(self, logical_id: str) -> str:
        return self.prefix + 'inbox.tool.' + validate_name(logical_id)

    def events_topic(self, topic: str) -> str:
        return self.prefix + 'events.' + validate_name(topic)

    def control_topic(self, kind: str) -> str:
        names = {'receipt':'receipts', 'directory':'directory', 'progress':'progress', 'error':'errors'}
        if kind not in names:
            raise ValueError('unsupported control kind')
        return self.prefix + names[kind]

    async def ensure_topics(self, specs: list[TopicSpec]) -> None:
        if self._closed:
            raise RuntimeError('bus is closed')
        for spec in specs:
            if spec.cleanup_policy == 'compact' and not self.capabilities().topic_compaction:
                raise ValueError('topic compaction is unsupported by this bus')
            self._validate_topic(spec.name)
            if type(spec.partitions) is not int or spec.partitions < 1 or type(spec.retention_ms) is not int or spec.retention_ms < 1:
                raise ValueError('topic partitions and retention must be positive integers')
        async with self._topics_lock:
            requested = {}
            for spec in specs:
                prior = requested.get(spec.name, self._specs.get(spec.name))
                if prior is not None and prior != spec:
                    raise ValueError('topic already requested with another specification')
                requested[spec.name] = spec
            pending = [s for s in requested.values() if s.name not in self._provisioned]
            if pending:
                await asyncio.to_thread(self._provision, pending)
                for spec in pending:
                    self._specs[spec.name] = spec
                    self._provisioned.add(spec.name)

    def _provision(self, specs):
        import confluent_kafka as ck
        from confluent_kafka.admin import AdminClient, NewTopic
        admin = AdminClient({'bootstrap.servers':self.broker_url, 'socket.timeout.ms':6000, 'log_level':0})
        futures = admin.create_topics([NewTopic(s.name, num_partitions=s.partitions,
            replication_factor=1, config={'retention.ms':str(s.retention_ms),
                'cleanup.policy':s.cleanup_policy}) for s in specs], request_timeout=6)
        for name, future in futures.items():
            try:
                future.result(10)
            except ck.KafkaException as exc:
                if exc.args[0].code() != ck.KafkaError.TOPIC_ALREADY_EXISTS:
                    _raise_broker_error(exc.args[0])
        try:
            metadata = admin.list_topics(timeout=6)
        except ck.KafkaException as exc:
            _raise_broker_error(exc.args[0])
        for spec in specs:
            topic = metadata.topics.get(spec.name)
            if topic is None or topic.error or len(topic.partitions) != spec.partitions:
                raise ValueError('broker topic partition count does not match requested specification')

    async def publish(self, envelope: Envelope) -> None:
        name = self.topic_for(envelope)
        raw = encode(envelope)
        if name not in self._provisioned:
            partitions = 1 if name[len(self.prefix):] in ('directory', 'receipts', 'progress', 'errors') else 2
            await self.ensure_topics([TopicSpec(name, partitions=partitions)])
        key = (envelope.key or envelope.session_id or envelope.event_id).encode('utf-8')
        async with self._producer_lock:
            if self._closed:
                raise RuntimeError('bus is closed')
            await asyncio.to_thread(self._publish, name, key, raw)

    def _publish(self, name, key, raw):
        import confluent_kafka as ck
        if self._producer is None:
            self._producer = ck.Producer({'bootstrap.servers':self.broker_url,
                'enable.idempotence':False, 'message.max.bytes':2097152,
                'delivery.timeout.ms':10000, 'socket.timeout.ms':6000,
                'queue.buffering.max.messages':64, 'queue.buffering.max.kbytes':4096,
                'log_level':0})
        result = []
        self._producer.produce(name, key=key, value=raw,
                               on_delivery=lambda error, message: result.append(error))
        self._producer.flush(12)
        if not result:
            raise TimeoutError('broker did not acknowledge publication')
        if result[0] is not None:
            _raise_broker_error(result[0])

    async def subscribe(self, topics: list[str], group_id: str, *, low_latency: bool = False):
        if self._closed:
            raise RuntimeError('bus is closed')
        if not topics or not group_id:
            raise ValueError('topics and group_id are required')
        for topic in topics:
            self._validate_topic(topic)
        await self.ensure_topics([TopicSpec(topic, partitions=1 if topic[len(self.prefix):]
            in ('directory', 'receipts', 'progress', 'errors') else 2)
            for topic in topics if topic not in self._specs])
        subscription = _Subscription(self, topics, group_id, low_latency=low_latency)
        self._subscriptions[subscription.id] = subscription
        subscription.thread.start()
        await asyncio.wrap_future(subscription.ready)
        if self._closed:
            await subscription.aclose()
            raise RuntimeError('bus is closed')
        return subscription

    async def ack(self, delivery: Delivery) -> None:
        position = delivery.position
        if not isinstance(position, BrokerPosition):
            raise ValueError('delivery does not have a broker position')
        subscription = self._subscriptions.get(position.subscription_id)
        if subscription is None or position.epoch != self.epoch:
            raise StaleDelivery('delivery belongs to an inactive subscription or broker epoch')
        await subscription.ack(position)

    async def broker_epoch_changed(self, epoch: str) -> None:
        if epoch == self.epoch:
            return
        for subscription in list(self._subscriptions.values()):
            await subscription.aclose()
        self.epoch = epoch
        self._provisioned.clear()
        await self.ensure_topics(list(self._specs.values()))

    async def close(self) -> None:
        # Refuse new resources before the first await, so a racing subscribe cannot outlive close.
        self._closed = True
        for subscription in list(self._subscriptions.values()):
            await subscription.aclose()
        async with self._producer_lock:
            if self._producer is not None:
                await asyncio.to_thread(self._producer.flush, 12)
                self._producer = None


class _Subscription:
    def __init__(self, bus, topics, group, *, low_latency=False):
        self.bus, self.topics, self.group, self.low_latency = bus, topics, group, low_latency
        self.id = str(uuid4())
        self.loop = asyncio.get_running_loop()
        self.wake = asyncio.Event()
        self.output = queue.Queue(maxsize=bus.capacity)
        self.commands = queue.Queue()
        self.stop = threading.Event()
        self.ready = Future()
        self.thread = threading.Thread(target=self._own, name='aw-consumer-' + self.id, daemon=True)
        self.failure = None
        self.closed = False
        self._next_lock = asyncio.Lock()
        self._malformed = None

    @property
    def buffered(self):
        return self.output.qsize()

    def __aiter__(self):
        return self

    async def __anext__(self):
        async with self._next_lock:
            return await self._next_delivery()

    async def _next_delivery(self):
        while True:
            if self.closed:
                raise StopAsyncIteration
            if self._malformed is not None:
                await self._complete_malformed()
                continue
            try:
                raw, position = self.output.get_nowait()
            except queue.Empty:
                if self.stop.is_set():
                    if self.failure is not None:
                        raise self.failure
                    raise StopAsyncIteration
                self.wake.clear()
                if self.output.empty() and not self.stop.is_set():
                    await self.wake.wait()
                continue
            try:
                event = decode(raw)
                if self.bus.topic_for(event) != position.topic:
                    raise InvalidEnvelope('topic mismatch')
            except (InvalidEnvelope, ValueError):
                if position.topic == self.bus.control_topic('error'):
                    self._malformed = _MalformedDelivery(position, None)
                    continue
                now = datetime.now(timezone.utc)
                error = Envelope(event_id=str(uuid4()), network_id=self.bus.network_id,
                    kind='error', sender='transport', recipient=None, topic='errors',
                    task_id=None, parent_task_id=None, correlation_id=None, session_id=None,
                    created_at=now, deadline=now + timedelta(seconds=120), traceparent=None,
                    payload={'code':'invalid_envelope', 'topic':position.topic[:256],
                             'partition':position.partition, 'offset':position.offset})
                self._malformed = _MalformedDelivery(position, error)
                continue
            return Delivery(event, position)

    async def _complete_malformed(self):
        pending = self._malformed
        if self.closed or pending.position.epoch != self.bus.epoch:
            self._malformed = None
            return
        if pending.diagnostic is not None and not pending.published:
            await self.bus.publish(pending.diagnostic)
            pending.published = True
        if self.closed or pending.position.epoch != self.bus.epoch:
            self._malformed = None
            return
        if pending.ack_task is None:
            pending.ack_task = asyncio.create_task(self.ack(pending.position))
        try:
            # Retain this one ACK task across iterator cancellation: its native
            # command may already have committed, so retrying a new command
            # would mistake a completed offset for a stale assignment.
            await asyncio.shield(pending.ack_task)
        except ValueError:
            # The owner rejected an invalidated position (reassignment/close).
            # Its old proof cannot commit the current assignment's offsets.
            self._malformed = None
        except asyncio.CancelledError:
            raise
        except Exception:
            pending.ack_task = None
            raise
        else:
            self._malformed = None

    async def ack(self, position):
        if self.stop.is_set():
            raise StaleDelivery('consumer is closed')
        future = Future()
        self.commands.put((position, future))
        await asyncio.wrap_future(future)

    async def aclose(self):
        self.closed = True
        pending_ack = self._malformed.ack_task if self._malformed is not None else None
        self._malformed = None
        self.stop.set()
        await asyncio.to_thread(self.thread.join, 15)
        if self.thread.is_alive():
            raise TimeoutError('consumer owner failed to stop')
        if pending_ack is not None:
            await asyncio.gather(pending_ack, return_exceptions=True)
        self.bus._subscriptions.pop(self.id, None)
        self.wake.set()

    def _signal(self):
        self.loop.call_soon_threadsafe(self.wake.set)

    def _consumer_config(self) -> dict:
        return {'bootstrap.servers':self.bus.broker_url, 'group.id':self.group,
            'group.protocol':'classic', 'enable.auto.commit':False, 'enable.auto.offset.store':False,
            'auto.offset.reset':'earliest', 'session.timeout.ms':6000, 'heartbeat.interval.ms':1000,
            'max.poll.interval.ms':300000, 'partition.assignment.strategy':'range',
            'fetch.message.max.bytes':2097152, 'fetch.max.bytes':2097152,
            'queued.min.messages':self.bus.prefetch_messages,
            'queued.max.messages.kbytes':self.bus.prefetch_kib, 'socket.timeout.ms':6000, 'log_level':0} | (
            # The dev broker holds an empty fetch for this long and does not end it early for new data.
            {'fetch.wait.max.ms':FETCH_WAIT_MS} if self.low_latency and self.bus.profile == 'dev' else {})

    def _apply_acks(self, consumer, partitions, ck) -> None:
        """Commit every contiguous acknowledged prefix requested since the last poll."""
        while True:
            try:
                position, future = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                if (position.group_id != self.group or position.subscription_id != self.id or
                        position.epoch != self.bus.epoch):
                    raise StaleDelivery('stale or foreign delivery position')
                commit = partitions.complete(position)
                if commit is not None:
                    topic, partition, next_offset, count = commit
                    try:
                        consumer.commit(offsets=[ck.TopicPartition(topic, partition, next_offset)],
                                        asynchronous=False)
                    except ck.KafkaException as exc:
                        if exc.args[0].code() in _rebalance_codes(ck):
                            raise StaleDelivery('partition is being reassigned') from exc
                        raise
                    partitions.committed((topic, partition), count)
                if not future.cancelled():
                    future.set_result(None)
            except Exception as exc:
                if not future.cancelled():
                    future.set_exception(exc)

    def _own(self):
        consumer = None
        partitions = _OwnedPartitions()
        pending = None
        emitted = 0.0
        try:
            import confluent_kafka as ck
            consumer = ck.Consumer(self._consumer_config())

            def assigned(owner, assignment):
                partitions.assign(assignment)
                owner.assign(assignment)

            def revoked(owner, assignment):
                partitions.revoke(assignment)
                owner.unassign()

            consumer.subscribe(self.topics, on_assign=assigned, on_revoke=revoked)
            self.ready.set_result(None)
            while not self.stop.is_set():
                self._apply_acks(consumer, partitions, ck)
                if pending is not None:
                    try:
                        self.output.put_nowait(pending)
                        pending = None
                        self._signal()
                    except queue.Full:
                        self.stop.wait(.02)
                        continue
                partitions.apply_backpressure(consumer,
                    self.output.full() or partitions.outstanding_count() >= self.bus.capacity)
                # Acks are applied between polls: while a delivery awaits its ack, poll briefly
                # so the ack is not held for a whole idle poll (that capped a worker near 20/s).
                recent = partitions.outstanding_count() > 0 and not partitions.paused and (
                    time.monotonic() - emitted < ACK_WINDOW_SECONDS)
                waiting = recent or not self.commands.empty()
                message = consumer.poll(ACK_POLL_SECONDS if waiting else IDLE_POLL_SECONDS)
                if message is None:
                    continue
                if message.error():
                    if message.error().code() == ck.KafkaError._PARTITION_EOF:
                        continue
                    raise ck.KafkaException(message.error())
                key = (message.topic(), message.partition())
                position = BrokerPosition(self.id, self.group, *key, message.offset(),
                    partitions.track(key, message.offset()), self.bus.epoch, message.key())
                pending = (message.value(), position)
                emitted = time.monotonic()
        except BaseException as exc:
            self.failure = exc
            if not self.ready.done():
                self.ready.set_exception(exc)
        finally:
            if consumer is not None:
                consumer.close()
            self.stop.set()
            while not self.commands.empty():
                _, future = self.commands.get_nowait()
                if not future.done():
                    future.set_exception(RuntimeError('consumer owner stopped'))
            self._signal()


def _rebalance_codes(ck) -> frozenset[int]:
    """Commit errors meaning the group moved on; the record will be redelivered."""
    return frozenset((ck.KafkaError.ILLEGAL_GENERATION, ck.KafkaError.UNKNOWN_MEMBER_ID,
                      ck.KafkaError.REBALANCE_IN_PROGRESS))


class _OwnedPartitions:
    """Assignment generations and unacknowledged offsets; owned by one consumer thread.

    An offset is committed only after it and every earlier delivered offset of
    its partition were acknowledged, so a commit never skips unaccepted input.
    """
    def __init__(self):
        self.generation = 0
        self.generations, self.outstanding, self.completed = {}, {}, {}
        self.paused = False

    def assign(self, assignment) -> None:
        self.generation += 1
        self.paused = False
        for part in assignment:
            key = (part.topic, part.partition)
            self.generations[key] = self.generation
            self.outstanding[key], self.completed[key] = deque(), set()

    def revoke(self, assignment) -> None:
        for part in assignment:
            key = (part.topic, part.partition)
            self.generations.pop(key, None)
            self.outstanding.pop(key, None)
            self.completed.pop(key, None)

    def track(self, key, offset) -> int:
        self.outstanding[key].append(offset)
        return self.generations[key]

    def outstanding_count(self) -> int:
        return sum(map(len, self.outstanding.values()))

    def complete(self, position) -> tuple[str, int, int, int] | None:
        """Mark position acknowledged and return the committable prefix.

        The result is (topic, partition, next offset, count). Offsets stay
        tracked until committed() confirms the broker accepted the commit.
        """
        key = (position.topic, position.partition)
        if self.generations.get(key) != position.generation:
            raise StaleDelivery('stale or foreign delivery position')
        offsets = self.outstanding[key]
        if position.offset not in offsets:
            raise StaleDelivery('delivery position is not outstanding')
        done = self.completed[key]
        done.add(position.offset)
        count = 0
        for offset in offsets:
            if offset not in done:
                break
            count += 1
        return None if not count else (*key, offsets[count - 1] + 1, count)

    def committed(self, key, count: int) -> None:
        offsets, done = self.outstanding.get(key), self.completed.get(key)
        if offsets is None:
            return  # revoked while committing; the next owner tracks its own offsets
        for _ in range(count):
            done.discard(offsets.popleft())

    def apply_backpressure(self, consumer, full: bool) -> None:
        if full and not self.paused:
            consumer.pause(consumer.assignment())
            self.paused = True
        elif not full and self.paused:
            consumer.resume(consumer.assignment())
            self.paused = False
