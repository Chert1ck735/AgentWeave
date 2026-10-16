"""Durable admission and leased outbox publication, independent of storage backend."""
import asyncio
import logging
from datetime import timedelta
from pydantic import TypeAdapter

from .._records import Admission, Clock, SystemClock
from ..errors import BrokerUnavailable, ReceiptNotExpected, StaleDelivery
from ..messages import TaskResult, UUIDString, validate_name
from ..ports import Delivery, EventBus, ExecutionStore

_CONTROL_KINDS = frozenset(('receipt', 'directory', 'progress', 'error'))
_TERMINAL = frozenset(('succeeded', 'failed', 'cancelled', 'expired'))
_uuid = TypeAdapter(UUIDString)


async def complete_or_undo(coroutine, undo):
    """Await a journal write that completes even if the caller is cancelled; undo it if the caller was.

    A store operation runs to its commit in the journal thread whatever happens to
    the awaiting task, so a cancelled caller must still learn what was written.
    """
    job = asyncio.ensure_future(coroutine)
    try:
        return await asyncio.shield(job)
    except asyncio.CancelledError:
        try:
            result = await job
        except BaseException:
            raise asyncio.CancelledError from None
        if result:
            await asyncio.shield(undo(result))
        raise


# Workers other than the receipt's own agents confirm it this much later, normally finding it recorded.
RECEIPT_DEFER_SECONDS = 1.0


async def confirm_receipts(subscription, service, prefers, on_outcome=None) -> None:
    """Confirm every receipt this worker sees, writing it once in the common case.

    Every worker sees every receipt, so none depends on another being alive. The
    agents a receipt concerns (prefers) confirm it at once; every other worker
    checks it RECEIPT_DEFER_SECONDS later and writes only if it is still unrecorded.
    """
    deferred, failures, consumer = set(), [], asyncio.current_task()

    async def later(delivery):
        await asyncio.sleep(RECEIPT_DEFER_SECONDS)
        try:
            outcome = await service.accept_confirmation(delivery)
        except Exception as error:  # a storage failure stops the consumer at once, visibly
            logging.getLogger(__name__).error('deferred receipt confirmation failed', exc_info=error)
            failures.append(error)
            consumer.cancel()
            return
        if on_outcome is not None:
            on_outcome(outcome, delivery)
    try:
        async for delivery in subscription:
            if prefers(delivery.envelope):
                outcome = await service.accept_confirmation(delivery)
                if on_outcome is not None:
                    on_outcome(outcome, delivery)
                continue
            job = asyncio.create_task(later(delivery))
            deferred.add(job)
            job.add_done_callback(deferred.discard)
    except asyncio.CancelledError:
        if failures:
            raise failures[0] from None
        raise
    finally:
        for job in deferred:
            job.cancel()
        await asyncio.gather(*deferred, return_exceptions=True)


class AdmissionService:
    def __init__(self, store: ExecutionStore, bus: EventBus):
        self.store, self.bus = store, bus

    async def accept(self, delivery: Delivery, definition_hash: str, *, agent_id=None) -> Admission:
        if delivery.envelope.kind in _CONTROL_KINDS:
            raise ValueError('control traffic requires its dedicated consumer')
        admitted = await self.store.admit(delivery.envelope, definition_hash, agent_id=agent_id)
        await self.bus.ack(delivery)
        return admitted

    async def accept_receipt(self, delivery: Delivery) -> None:
        event = delivery.envelope
        if event.recipient is None or event.topic is not None or event.correlation_id is None:
            raise ValueError('receipt confirmation must be addressed and correlated')
        if event.kind == 'receipt':
            payload = event.payload
            if not isinstance(payload, dict) or set(payload) != {'event_id', 'recipient'}:
                raise ValueError('invalid receipt payload')
            original = _uuid.validate_python(payload['event_id'])
            recipient = validate_name(payload['recipient'])
            if original != event.correlation_id or recipient != event.sender:
                raise ValueError('receipt correlation and sender do not match payload')
        elif event.kind in ('reply', 'tool_result'):
            # A terminal answer also proves the input arrived, e.g. after a lost receipt.
            result = TaskResult.model_validate(event.payload)
            if (result.task_id != event.task_id or result.responding_agent != event.sender or
                    result.status not in _TERMINAL):
                raise ValueError('only a matching terminal task result confirms input')
            original, recipient = event.correlation_id, event.sender
        else:
            raise ValueError('delivery is not a receipt or terminal reply')
        try:
            await self.store.confirm_receipt(original, recipient)
        except ReceiptNotExpected:
            logging.getLogger(__name__).info('Ignoring unmatched receipt',
                extra={'event_id':original, 'recipient':recipient})
        await self.bus.ack(delivery)

    async def accept_confirmation(self, delivery: Delivery) -> str | None:
        """accept_receipt for a long-running consumer; returns the outcome worth counting.

        A stale position is redelivered to the current owner, and one malformed
        record is acknowledged and skipped so it cannot stop the consumer.
        """
        try:
            await self.accept_receipt(delivery)
        except StaleDelivery:
            return 'stale_ack'
        except ValueError as error:
            logging.getLogger(__name__).warning('ignoring invalid control record: %s', error,
                extra={'event_id': delivery.envelope.event_id, 'kind': delivery.envelope.kind})
            try:
                await self.bus.ack(delivery)
            except StaleDelivery:
                pass
            return 'invalid_control'
        return None


class OutboxPump:
    """Publish leased rows. A broker ACK never substitutes for a target receipt.

    Retry count is process local and safely resets to one second on restart;
    retry_at is saved atomically so competing pumps respect the due time.
    """
    def __init__(self, store: ExecutionStore, bus: EventBus, instance_id: str, *,
                 clock: Clock | None = None, batch_size: int = 16, idle_seconds: float = .1):
        self.store, self.bus = store, bus
        self.instance_id = _uuid.validate_python(instance_id)
        if type(batch_size) is not int or batch_size < 1 or idle_seconds <= 0:
            raise ValueError('pump batch and idle interval must be positive')
        self.clock = clock or SystemClock()
        self.batch_size, self.idle_seconds = batch_size, idle_seconds
        self._attempts = {}
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()
        self.last_error = None

    def stop(self):
        self._stop.set()

    async def run(self) -> None:
        while not self._stop.is_set():
            await self.run_once()
            try:
                await asyncio.wait_for(self._stop.wait(), self.idle_seconds)
            except TimeoutError:
                pass

    async def run_once(self) -> int:
        async with self._lock:
            now = self.clock.now()
            self._attempts = {key:value for key,value in self._attempts.items() if value[1] > now}
            rows = await complete_or_undo(self.store.lease_outbox(self.instance_id, self.batch_size),
                                          self.store.release_outbox)
            acks, handled, results = [], set(), []
            try:
                await self._publish_rows(rows, acks, handled)
            finally:
                # Acks of rows already published are written even if a later row failed or the pump
                # was cancelled; leased rows not published now are given back, not left until expiry.
                pending = [row for row in rows if row.envelope.event_id not in handled]
                results = await asyncio.shield(self._settle([ack for _, ack in acks], pending))
            published = 0
            for (row, _), acknowledged in zip(acks, results):
                # False means the lease/version changed during publish: respect the store's newer
                # receipt or reactivation facts without retrying an unscoped ACK.
                if acknowledged:
                    published += 1
                    if not row.targets:
                        self._attempts.pop(row.envelope.event_id, None)
            return published

    async def _settle(self, acks, pending):
        results = await self.store.record_broker_acks(acks) if acks else []
        await self.store.release_outbox(pending)
        return results

    async def _publish_rows(self, rows, acks, handled):
        for row in rows:
            event = row.envelope
            if event.deadline <= self.clock.now():
                await self.store.reschedule_outbox(event.event_id, event.deadline,
                    lease_token=row.lease_token, expected_version=row.version)
                handled.add(event.event_id)
                continue
            attempt = self._attempts.get(event.event_id, (0, event.deadline))[0]
            delay = (1, 2, 4, 5)[min(attempt, 3)]
            self._attempts[event.event_id] = (min(attempt + 1, 3), event.deadline)
            try:
                await self.bus.publish(event)
            except (BrokerUnavailable, TimeoutError) as exc:
                self.last_error = exc
                await self.store.reschedule_outbox(event.event_id,
                    min(self.clock.now() + timedelta(seconds=delay), event.deadline),
                    lease_token=row.lease_token, expected_version=row.version)
                handled.add(event.event_id)
                continue
            handled.add(event.event_id)
            acks.append((row, (event.event_id, row.lease_token, row.version,
                               min(self.clock.now() + timedelta(seconds=delay), event.deadline))))
