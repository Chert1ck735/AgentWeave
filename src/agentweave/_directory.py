"""Broker announcements are the sole authority for observed participant liveness."""
import asyncio
from datetime import timedelta
import logging
from uuid import uuid4
from pydantic import ValidationError
from ._records import Clock, Participant, SystemClock
from .errors import BrokerUnavailable, DefinitionError, StoreConflict
from .messages import Envelope, validate_name
from .ports import EventBus, ExecutionStore, TopicSpec

_logger = logging.getLogger('agentweave.directory')

HEARTBEAT_SECONDS = 5
STALE_SECONDS = 15


class LiveDirectory:
    """One catalog subscription per observer, independent of inbox consumer groups.

    Bootstrap owns the bus/store and calls broker_epoch_changed after changing
    the bus epoch. No persisted participant is loaded as proof of availability.
    Background failures remain visible via check_health() and the next operation.
    A withdrawn key requires a strictly newer timestamp to become live again.
    """
    def __init__(self, bus: EventBus, store: ExecutionStore, network_id: str, *,
                 broker_epoch: str = '', clock: Clock | None = None):
        from pydantic import TypeAdapter
        from .messages import UUIDString
        self.network_id = TypeAdapter(UUIDString).validate_python(network_id)
        self.bus, self.store, self.epoch = bus, store, broker_epoch
        self.clock = clock if clock is not None else SystemClock()
        self._entries, self._owned, self._versions = {}, {}, {}
        self._subscription = None
        self._reader = self._ticker = None
        self._lock = asyncio.Lock()
        self._started = self._closed = False
        self._failure = None

    def check_health(self) -> None:
        if self._failure is not None:
            raise self._failure

    def _require_open(self):
        self.check_health()
        if not self._started or self._closed:
            raise RuntimeError('directory is not started or has closed')

    async def _subscribe(self):
        topic = 'aw.' + self.network_id + '.directory'
        compact = self.bus.capabilities().topic_compaction
        await self.bus.ensure_topics([TopicSpec(topic, partitions=1,
            cleanup_policy='compact' if compact else 'delete')])
        # A fresh observer must fold retained announcements, never inherit
        # another process's committed offset in the directory broadcast topic.
        self._subscription = await self.bus.subscribe([topic], 'aw.directory.' + str(uuid4()))
        self._reader = asyncio.create_task(self._consume(), name='aw-directory-reader')

    async def start(self) -> None:
        async with self._lock:
            if self._closed:
                raise RuntimeError('directory is closed')
            if self._started:
                self.check_health()
                return
            await self._subscribe()
            self._started = True
            self._ticker = asyncio.create_task(self._run_heartbeats(), name='aw-directory-heartbeat')

    def _fresh(self, value):
        now = self.clock.now()
        return value.broker_epoch == self.epoch and now - timedelta(seconds=STALE_SECONDS) < value.heartbeat_at <= now

    def snapshot(self) -> dict[str, tuple[Participant, ...]]:
        self.check_health()
        grouped = {}
        for (logical_id, _), value in sorted(self._entries.items()):
            if self._fresh(value):
                grouped.setdefault(logical_id, []).append(value.model_copy(deep=True))
        return {logical_id:tuple(instances) for logical_id, instances in grouped.items()}

    def resolve_tool(self, alias: str, *, revision: str | None = None) -> Participant | None:
        matches = [instances[0] for instances in self.snapshot().values()
            if instances[0].kind == 'tool' and instances[0].alias == alias]
        if len(matches) > 1:
            raise DefinitionError('tool alias is ambiguous')
        if not matches or (revision is not None and matches[0].revision != revision):
            return None
        return matches[0]

    def _validate_announcement(self, value):
        for instances in self.snapshot().values():
            old = instances[0]
            if old.logical_id == value.logical_id:
                # Compare all public definition fields, not heartbeat winner.
                dynamic = {'instance_id', 'heartbeat_at', 'broker_epoch'}
                if old.model_dump(exclude=dynamic) != value.model_dump(exclude=dynamic):
                    raise StoreConflict('conflicting live participant definition')
            elif old.kind == value.kind == 'tool' and old.alias == value.alias:
                raise DefinitionError('tool aliases collide')

    async def _announce(self, value):
        now = self.clock.now()
        value = value.model_copy(update={'heartbeat_at':now, 'broker_epoch':self.epoch}, deep=True)
        await self.store.publish_participant(value)
        await self.bus.publish(self._envelope(value.logical_id, value.instance_id,
            {'participant':value.model_dump(mode='json'), 'broker_epoch':self.epoch}, now=now))
        return value

    def _envelope(self, logical_id, instance_id, payload, *, now=None):
        now = self.clock.now() if now is None else now
        return Envelope(event_id=str(uuid4()), network_id=self.network_id, kind='directory',
            sender=logical_id, recipient=None, topic='directory', task_id=None,
            parent_task_id=None, correlation_id=None, session_id=None, created_at=now,
            deadline=now + timedelta(seconds=STALE_SECONDS), traceparent=None,
            key=logical_id + '/' + instance_id, payload=payload)

    async def publish(self, participant: Participant) -> None:
        value = Participant.model_validate(participant.model_dump(mode='python')).model_copy(deep=True)
        if value.network_id != self.network_id:
            raise ValueError('participant is outside the directory network')
        async with self._lock:
            self._require_open()
            self._validate_announcement(value)
            await self.store.register_definition(value.logical_id, value.instance_id, value.fingerprint)
            key = (value.logical_id, value.instance_id)
            # Retain ownership after a publication failure so close can release
            # the reservation and a heartbeat can retry the same definition.
            self._owned[key] = value
            self._owned[key] = await self._announce(value)

    async def heartbeat(self) -> None:
        async with self._lock:
            self._require_open()
            for key, value in list(self._owned.items()):
                self._owned[key] = await self._announce(value)

    async def withdraw(self, logical_id: str, instance_id: str) -> None:
        logical_id = validate_name(logical_id)
        async with self._lock:
            self._require_open()
            await self._withdraw(logical_id, instance_id)

    async def _withdraw(self, logical_id, instance_id):
        key = (logical_id, instance_id)
        if key not in self._owned:
            return
        try:
            await self.bus.publish(self._envelope(logical_id, instance_id,
                {'participant':None, 'broker_epoch':self.epoch}))
        except (BrokerUnavailable, TimeoutError) as error:
            # Withdrawal is best effort on the wire: peers expire the stale
            # announcement after the heartbeat timeout, while the journal
            # below releases the definition instance immediately.
            _logger.warning('directory withdrawal not published: %s', error,
                            extra={'logical_id':logical_id, 'instance_id':instance_id})
        await self.store.unregister_definition(logical_id, instance_id)
        self._owned.pop(key, None)

    async def _consume(self):
        try:
            async for delivery in self._subscription:
                envelope = delivery.envelope
                try:
                    self._apply(envelope)
                except (ValueError, ValidationError, DefinitionError, StoreConflict):
                    # A rejected announcement cannot alter authority or prevent
                    # subsequent valid keyed snapshots from being processed.
                    pass
                await self.bus.ack(delivery)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._failure = error

    def _apply(self, envelope):
        if envelope.network_id != self.network_id or envelope.kind != 'directory' or envelope.topic != 'directory':
            raise ValueError('invalid directory destination')
        payload = envelope.payload
        if not isinstance(payload, dict) or set(payload) != {'participant', 'broker_epoch'}:
            raise ValueError('invalid announcement payload')
        if payload['broker_epoch'] != self.epoch:
            return
        if not envelope.key or '/' not in envelope.key:
            raise ValueError('directory key requires logical and instance ID')
        logical_id, instance_id = envelope.key.split('/', 1)
        key = (logical_id, instance_id)
        if envelope.sender != logical_id:
            raise ValueError('announcement sender mismatch')
        if envelope.created_at > self.clock.now():
            raise ValueError('announcement timestamp is in the future')
        previous = self._versions.get(key)
        if previous is not None:
            previous_at, withdrawn = previous
            if envelope.created_at < previous_at or (withdrawn and envelope.created_at == previous_at):
                return
        if payload['participant'] is None:
            self._entries.pop(key, None)
            self._versions[key] = (envelope.created_at, True)
            return
        value = Participant.model_validate(payload['participant'])
        if (value.logical_id, value.instance_id) != key or value.network_id != self.network_id or value.heartbeat_at != envelope.created_at:
            raise ValueError('announcement identity or heartbeat mismatch')
        if not self._fresh(value):
            return
        self._validate_announcement(value)
        self._entries[key] = value.model_copy(deep=True)
        self._versions[key] = (envelope.created_at, False)

    async def _run_heartbeats(self):
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_SECONDS)
                await self.heartbeat()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._failure = error

    async def _stop_reader(self):
        if self._reader is not None:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)
            self._reader = None
        if self._subscription is not None:
            await self._subscription.aclose()
            self._subscription = None

    async def broker_epoch_changed(self, epoch: str) -> None:
        async with self._lock:
            if epoch == self.epoch:
                return
            if not self._started or self._closed:
                raise RuntimeError('directory is not started or has closed')
            await self._stop_reader()
            self.epoch = epoch
            self._entries.clear()
            self._versions.clear()
            self._failure = None
            await self._subscribe()
            if self._ticker is None or self._ticker.done():
                self._ticker = asyncio.create_task(self._run_heartbeats(), name='aw-directory-heartbeat')

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            if self._ticker is not None:
                self._ticker.cancel()
                await asyncio.gather(self._ticker, return_exceptions=True)
                self._ticker = None
            errors = []
            for logical_id, instance_id in list(self._owned):
                try:
                    await self._withdraw(logical_id, instance_id)
                except Exception as error:
                    errors.append(error)
            await self._stop_reader()
            self._entries.clear()
            self._versions.clear()
            self._closed = True
            if errors:
                raise errors[0]
