"""Local disk SQLite journal with short, fenced mutation transactions.

Constructing a store has no I/O. Call migrate() before use. Each store owns one
connection; a local asyncio lock prevents transaction interleaving on that
connection while BEGIN IMMEDIATE arbitrates independent processes.
"""
import asyncio
from datetime import timedelta
import logging
from decimal import Decimal
import json
import math
from pathlib import Path
import sqlite3
from uuid import UUID, uuid4, uuid5

from pydantic import TypeAdapter

from ..._records import Admission, ConversationRecord, EventRejection, Clock, Lease, OutboxRecord, Participant, StepCommit, SystemClock, TaskRecord, ToolLease, ModelAttempt, ModelUsage
from ...errors import (HandoffRefused, LeaseLost, ModelBudgetExceeded, ModelCapacity, ModelError, ReceiptNotExpected,
                       ReplayExpired, StoreConflict, StoreUnavailable, TaskCancelled, VersionConflict)
from ...messages import (MAX_ENVELOPE_BYTES, Envelope, TaskResult, TaskSubmission, ToolCall, UUIDString, UTCDateTime,
                         decode, encode, event_task_id, is_client, validate_name)
from .migrations import statements
from .sqlite_operations import OperatorOperations
from .sqlite_thread import ThreadConnection

OUTBOX_LEASE_SECONDS = 30
# An unreadable outbox row is leased to this owner forever, so no publisher selects it again.
QUARANTINE_OWNER = 'quarantined'
QUARANTINE_UNTIL = '9999-12-31T23:59:59.999999+00:00'
_logger = logging.getLogger('agentweave.storage')
BUSY_BACKOFF_SECONDS = (0.05, 0.1)
# How far ahead of this clock an event's created_at may be (same-host processes).
FUTURE_TOLERANCE = timedelta(minutes=5)
SCHEMA_VERSIONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
TERMINAL_STATUSES = ('succeeded', 'failed', 'cancelled', 'expired')
_EXPIRE = object()  # a read found a submission that must now be recorded as expired


class _Busy(Exception):
    """Rolls back a begin_attempt whose ordering gate is held by an earlier task."""
_uuid = TypeAdapter(UUIDString)
_utc = TypeAdapter(UTCDateTime)


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _stamp(value):
    # Uniform precision keeps SQLite lexicographic UTC comparisons correct.
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('clock must return an aware UTC timestamp')
    from datetime import timezone
    return value.astimezone(timezone.utc).isoformat(timespec='microseconds')


def _task(row):
    if row is None:
        return None
    return TaskRecord(task_id=row['task_id'], agent_id=row['agent_id'], status=row['status'],
        version=row['version'], input=json.loads(row['input']), session_id=row['session_id'],
        root_task_id=row['root_task_id'], definition_hash=row['definition_hash'], deadline=row['deadline'],
        result=json.loads(row['result']), error=json.loads(row['error']), admitted_at=row['admitted_at'],
        waiting_for=json.loads(row['waiting_for']), transfer_context=json.loads(row['transfer_context']),
        handoffs=row['handoffs'])


class SQLiteStore(OperatorOperations):
    def __init__(self, path: Path | str, clock: Clock | None = None):
        self.path = Path(path)
        if str(path) == ':memory:':
            raise ValueError('SQLiteStore requires a real local file')
        self.clock = clock if clock is not None else SystemClock()
        self._connection = None
        self._lock = asyncio.Lock()
        self._ready = False
        self._closed = False

    async def _fetch(self, sql, values=()):
        async with self._connection.execute(sql, values) as cursor:
            return await cursor.fetchone()

    async def _all(self, sql, values=()):
        async with self._connection.execute(sql, values) as cursor:
            return await cursor.fetchall()

    async def _in_thread(self, operation):
        """Run a whole operation (its statements, and a transaction's BEGIN to COMMIT) in one call."""
        return await self._connection.run(operation)

    async def _retry(self, operation):
        for attempt in range(3):
            try:
                return await self._in_thread(operation)
            except sqlite3.Error as exc:
                code = getattr(exc, 'sqlite_errorcode', 0) & 0xff
                if code in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) and attempt < 2:
                    await asyncio.sleep(BUSY_BACKOFF_SECONDS[attempt])
                    continue
                if isinstance(exc, sqlite3.IntegrityError):
                    raise StoreConflict('journal identity or constraint conflict') from exc
                raise StoreUnavailable('SQLite persistence operation failed') from exc

    async def _transaction(self, operation):
        async def attempt():
            # Runs as one call on the journal thread: the write lock is held only while SQLite
            # works, and a cancelled caller cannot leave the transaction half done.
            locked = self._connection.acquire_writer()
            try:
                await self._connection.execute('BEGIN IMMEDIATE')
                try:
                    value = await operation()
                    await self._connection.commit()
                except BaseException:
                    try:  # also after a failed COMMIT, so no transaction stays open on the connection
                        await self._connection.rollback()
                    except sqlite3.Error:
                        pass  # the original error is the one to report
                    raise
                return value
            finally:
                if locked:
                    self._connection.release_writer()
        return await self._retry(attempt)

    def _require_ready(self):
        if not self._ready or self._closed:
            raise StoreUnavailable('store is not migrated or has been closed')

    async def migrate(self) -> None:
        async with self._lock:
            if self._closed:
                raise StoreUnavailable('store has been closed')
            if self._ready:
                return
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self._connection is None:
                    self._connection = await ThreadConnection(self.path).open()
                await self._connection.execute('PRAGMA busy_timeout=5000')
                async def configure():
                    row = await self._fetch('PRAGMA journal_mode=WAL')
                    if row[0] != 'wal':
                        raise StoreUnavailable('SQLite WAL mode is required on local disk')
                await self._retry(configure)
                await self._connection.execute('PRAGMA synchronous=FULL')
                await self._connection.execute('PRAGMA foreign_keys=ON')
                async def migration():
                    exists = await self._fetch("SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'")
                    applied = []
                    if exists:
                        versions = await self._all('SELECT version FROM schema_migrations ORDER BY version')
                        applied = [row[0] for row in versions]
                        if not applied or applied != list(SCHEMA_VERSIONS[:len(applied)]):
                            raise StoreConflict('unsupported SQLite schema version')
                    for version in SCHEMA_VERSIONS:
                        if version not in applied:
                            for statement in statements(version):
                                await self._connection.execute(statement)
                            await self._connection.execute('INSERT INTO schema_migrations VALUES(?,?)', (version, _stamp(self.clock.now())))
                await self._transaction(migration)
                self._ready = True
            except (sqlite3.Error, OSError) as exc:
                raise StoreUnavailable('SQLite initialization failed') from exc

    async def _mutate(self, operation):
        async with self._lock:
            self._require_ready()
            return await self._transaction(operation)

    async def register_definition(self, logical_id: str, instance_id: str, fingerprint: str) -> None:
        logical_id, instance_id = validate_name(logical_id), _uuid.validate_python(instance_id)
        if type(fingerprint) is not str or not fingerprint:
            raise ValueError('fingerprint must be a nonempty string')
        async def operation():
            now = self.clock.now()
            registered = await self._fetch('SELECT fingerprint FROM logical_definitions WHERE logical_id=?', (logical_id,))
            # Tasks are a barrier even when all owners stopped, including waiting
            # or outcome_unknown work whose external effects remain unresolved.
            unfinished = await self._fetch("SELECT 1 FROM tasks WHERE agent_id=? AND definition_hash<>? AND status NOT IN ('succeeded','failed','cancelled','expired') LIMIT 1", (logical_id, fingerprint))
            if unfinished:
                raise StoreConflict('definition conflicts with accepted nonterminal work')
            if registered and registered['fingerprint'] != fingerprint:
                live = await self._fetch('SELECT 1 FROM definition_instances WHERE logical_id=? AND expires_at>? LIMIT 1', (logical_id, _stamp(now)))
                if live:
                    raise StoreConflict('definition conflicts with a live instance')
                await self._connection.execute('DELETE FROM definition_instances WHERE logical_id=?', (logical_id,))
            await self._connection.execute('INSERT INTO logical_definitions VALUES(?,?) ON CONFLICT(logical_id) DO UPDATE SET fingerprint=excluded.fingerprint', (logical_id, fingerprint))
            await self._connection.execute('INSERT INTO definition_instances VALUES(?,?,?,?) ON CONFLICT(logical_id,instance_id) DO UPDATE SET fingerprint=excluded.fingerprint,expires_at=excluded.expires_at',
                (logical_id, instance_id, fingerprint, _stamp(now + timedelta(seconds=15))))
        await self._mutate(operation)

    async def publish_participant(self, participant: Participant) -> None:
        participant = Participant.model_validate(participant.model_dump(mode='python'))
        async def operation():
            now = self.clock.now()
            row = await self._fetch('SELECT d.fingerprint FROM logical_definitions d JOIN definition_instances i ON d.logical_id=i.logical_id WHERE i.logical_id=? AND i.instance_id=?', (participant.logical_id, participant.instance_id))
            if row is None or row['fingerprint'] != participant.fingerprint:
                raise StoreConflict('participant has no matching registered definition')
            if not now - timedelta(seconds=15) < participant.heartbeat_at <= now:
                raise ValueError('participant heartbeat must be fresh and not in the future')
            await self._connection.execute('UPDATE definition_instances SET expires_at=? WHERE logical_id=? AND instance_id=? AND fingerprint=?',
                (_stamp(participant.heartbeat_at + timedelta(seconds=15)), participant.logical_id, participant.instance_id, participant.fingerprint))
            await self._connection.execute('INSERT INTO participant_announcements VALUES(?,?,?) ON CONFLICT(logical_id,instance_id) DO UPDATE SET snapshot=excluded.snapshot',
                (participant.logical_id, participant.instance_id, participant.model_dump_json()))
        await self._mutate(operation)

    async def unregister_definition(self, logical_id: str, instance_id: str) -> None:
        logical_id, instance_id = validate_name(logical_id), _uuid.validate_python(instance_id)
        async def operation():
            await self._connection.execute('DELETE FROM definition_instances WHERE logical_id=? AND instance_id=?', (logical_id, instance_id))
        await self._mutate(operation)

    @staticmethod
    def _receipt_targets(envelope):
        # Receipt-of-receipt would create an endless acknowledgement cascade, and
        # clients read results from the journal instead of confirming replies.
        # A remote tool's caller also reads the outcome from the journal.
        if envelope.kind in ('receipt', 'tool_result', 'handoff') or envelope.recipient is None:
            return []
        if envelope.kind == 'reply' and is_client(envelope.recipient):
            return []
        return [envelope.recipient]

    async def _insert_outbox(self, envelope, now, *, publish=True):
        """Queue an envelope for publication; publish=False keeps it in the journal only (already delivered)."""
        targets = self._receipt_targets(envelope)
        if envelope.kind == 'reply' and is_client(envelope.recipient):
            publish = False  # clients read results from the journal; nobody consumes a client's inbox
        if not publish:
            await self._connection.execute('INSERT INTO outbox(event_id,envelope,targets,published_at,deadline,delivered) VALUES(?,?,?,?,?,1)',
                (envelope.event_id, encode(envelope).decode('utf-8'), _json(targets), now, _stamp(envelope.deadline)))
            return
        await self._connection.execute('INSERT INTO outbox(event_id,envelope,targets,retry_at,deadline) VALUES(?,?,?,?,?)',
            (envelope.event_id, encode(envelope).decode('utf-8'), _json(targets), now, _stamp(envelope.deadline)))

    async def submit(self, envelope: Envelope) -> TaskSubmission:
        envelope = decode(encode(envelope))
        if envelope.task_id is None:
            raise ValueError('submit requires a task_id')
        async def operation():
            existing = await self._fetch('SELECT event_id FROM submissions WHERE task_id=?', (envelope.task_id,))
            if existing:
                original = await self._fetch('SELECT envelope FROM outbox WHERE event_id=?', (existing['event_id'],))
                if original['envelope'] != encode(envelope).decode('utf-8'):
                    raise StoreConflict('submission task_id already has a different envelope')
            else:
                now = _stamp(self.clock.now())
                await self._insert_outbox(envelope, now)
                await self._connection.execute('INSERT INTO submissions VALUES(?,?,?)', (envelope.task_id, envelope.event_id, now))
            return TaskSubmission(task_id=envelope.task_id)
        return await self._mutate(operation)

    async def get_submission_envelope(self, task_id):
        task_id = _uuid.validate_python(task_id)
        async with self._lock:
            self._require_ready()
            async def operation():
                row = await self._fetch('SELECT o.envelope FROM submissions s JOIN outbox o ON o.event_id=s.event_id WHERE s.task_id=?', (task_id,))
                return decode(row['envelope'].encode()) if row else None
            return await self._retry(operation)

    async def resolve_submission_expiry(self, task_id):
        """The terminal result of a submission, persisting an expiry once; polled by every waiting client.

        Waiting only reads: the write lock is taken once, when the submission must be recorded as expired.
        """
        task_id = _uuid.validate_python(task_id)
        async def look():
            task = _task(await self._fetch('SELECT * FROM tasks WHERE task_id=?', (task_id,)))
            if task:
                if task.status in ('succeeded','failed','cancelled','expired','outcome_unknown'):
                    return TaskResult(task_id=task_id,output=task.result,responding_agent=task.agent_id,status=task.status,error=task.error)
                return None
            saved = await self._fetch('SELECT result FROM submission_outcomes WHERE task_id=?', (task_id,))
            if saved:
                return TaskResult.model_validate_json(saved['result'])
            row = await self._fetch('SELECT o.envelope FROM submissions s JOIN outbox o ON o.event_id=s.event_id WHERE s.task_id=?', (task_id,))
            if row is None:
                raise ValueError('unknown task or submission')
            event = decode(row['envelope'].encode())
            now = self.clock.now()
            if event.deadline > now and (event.expires_at is None or event.expires_at > now):
                return None
            return _EXPIRE
        async with self._lock:
            self._require_ready()
            found = await self._retry(look)
        if found is not _EXPIRE:
            return found
        async def operation():
            task = _task(await self._fetch('SELECT * FROM tasks WHERE task_id=?', (task_id,)))
            if task:
                if task.status in ('succeeded','failed','cancelled','expired','outcome_unknown'):
                    return TaskResult(task_id=task_id,output=task.result,responding_agent=task.agent_id,status=task.status,error=task.error)
                return None
            saved = await self._fetch('SELECT result FROM submission_outcomes WHERE task_id=?', (task_id,))
            if saved:
                return TaskResult.model_validate_json(saved['result'])
            row = await self._fetch('SELECT o.envelope FROM submissions s JOIN outbox o ON o.event_id=s.event_id WHERE s.task_id=?', (task_id,))
            if row is None:
                raise ValueError('unknown task or submission')
            event = decode(row['envelope'].encode())
            now = self.clock.now()
            if event.deadline > now and (event.expires_at is None or event.expires_at > now):
                return None
            result = TaskResult(task_id=task_id,output=None,responding_agent=None,status='expired')
            await self._connection.execute('INSERT INTO submission_outcomes VALUES(?,?)', (task_id,result.model_dump_json()))
            return result
        return await self._mutate(operation)

    async def _replay_expired(self, envelope):
        """Inside a transaction: older than the replay window, or no newer than a removed dedup ID."""
        if envelope.created_at < self.clock.now() - timedelta(days=self.retention.replay_window_days):
            return True
        floor = await self.dedup_floor()
        return floor is not None and envelope.created_at <= floor

    async def admit(self, envelope: Envelope, definition_hash: str, *, agent_id: str | None = None) -> Admission:
        return await self._admit(envelope, definition_hash, agent_id=agent_id)

    async def admit_with_policy(self, envelope, definition_hash, *, agent_id, max_inflight, snapshot=False):
        if type(max_inflight) is not int or max_inflight < 1:
            raise ValueError('max_inflight must be positive')
        if snapshot and (envelope.topic is None or envelope.key is None or envelope.source_version is None or envelope.kind in ('request','reply','receipt','progress','directory','error','tool_call')):
            raise ValueError('snapshot admission requires a user topic, key and source_version')
        return await self._admit(envelope, definition_hash, agent_id=agent_id,
                                 max_inflight=max_inflight, snapshot=snapshot)

    async def _admit(self, envelope, definition_hash, *, agent_id=None, max_inflight=None, snapshot=False):
        envelope = decode(encode(envelope))
        if type(definition_hash) is not str:
            raise ValueError('definition_hash must be a string')
        if envelope.recipient is not None:
            if agent_id is not None and agent_id != envelope.recipient:
                raise ValueError('agent_id must match addressed recipient')
            agent_id = envelope.recipient
            if envelope.task_id is None:
                raise ValueError('addressed admission requires task_id')
            task_id = envelope.task_id
        else:
            if agent_id is None:
                raise ValueError('topic admission requires the consuming agent_id')
            task_id = event_task_id(envelope.event_id, agent_id)
        agent_id = validate_name(agent_id)
        receipt_name = 'receipt:' + agent_id
        async def operation():
            duplicate = await self._fetch('SELECT task_id,receipt_event_id FROM processed_events WHERE agent_id=? AND event_id=?', (agent_id, envelope.event_id))
            if duplicate:
                saved = await self._fetch('SELECT envelope FROM receipts WHERE agent_id=? AND event_id=?', (agent_id, envelope.event_id))
                # A prior broker ACK can survive a lost receipt. Requeue the
                # exact durable envelope without disturbing a current publisher.
                outbox = await self._fetch('SELECT version FROM outbox WHERE event_id=?', (duplicate['receipt_event_id'],))
                if outbox is not None and envelope.recipient is not None:
                    await self._connection.execute('UPDATE outbox SET published_at=NULL,delivered=0,retry_at=?,version=version+1 WHERE event_id=? AND version=?',
                        (_stamp(self.clock.now()), duplicate['receipt_event_id'], outbox['version']))
                return Admission(duplicate=True, task_id=duplicate['task_id'], receipt=decode(saved['envelope'].encode('utf-8')))
            if envelope.created_at > self.clock.now() + FUTURE_TOLERANCE:
                # Its dedup ID could outlive no replay window: a skewed clock must not move the floor.
                if max_inflight is not None:
                    return EventRejection(reason='created_in_future', retryable=False)
                raise ReplayExpired('the event was created in the future; fix the sender clock')
            if await self._replay_expired(envelope):
                # Its dedup ID may be gone: running it could repeat work already done.
                if max_inflight is not None:
                    return EventRejection(reason='replay_expired', retryable=False)
                raise ReplayExpired('the event is older than the replay window; it is not run again')
            if max_inflight is not None:
                terminal = await self._fetch('SELECT result FROM submission_outcomes WHERE task_id=?', (task_id,))
                if terminal:
                    return EventRejection(reason='expired', retryable=False)
            registered = await self._fetch('SELECT fingerprint FROM logical_definitions WHERE logical_id=?', (agent_id,))
            if registered is not None and registered['fingerprint'] != definition_hash:
                raise StoreConflict('admission conflicts with the current definition')
            now = self.clock.now()
            if max_inflight is not None:
                if envelope.deadline <= now or (envelope.expires_at is not None and envelope.expires_at <= now):
                    return EventRejection(reason='expired', retryable=False)
                if snapshot:
                    old = await self._fetch('SELECT source_version FROM source_watermarks WHERE agent_id=? AND sender=? AND topic=? AND key=?',
                        (agent_id, envelope.sender, envelope.topic, envelope.key))
                    if old is not None and old['source_version'] >= envelope.source_version:
                        return EventRejection(reason='stale_version', retryable=False)
                # outcome_unknown awaits an operator, not execution capacity.
                # A parent waiting for a consult holds no execution: counting it could deadlock two agents.
                count = await self._fetch("SELECT COUNT(*) AS count FROM tasks WHERE agent_id=? AND status IN ('accepted','running','waiting_tool')", (agent_id,))
                if count['count'] >= max_inflight:
                    return EventRejection(reason='capacity', retryable=True)
                if snapshot:
                    await self._connection.execute('INSERT INTO source_watermarks VALUES(?,?,?,?,?) ON CONFLICT(agent_id,sender,topic,key) DO UPDATE SET source_version=excluded.source_version',
                        (agent_id, envelope.sender, envelope.topic, envelope.key, envelope.source_version))
            root_id = task_id
            if envelope.parent_task_id is not None:
                parent = await self._fetch('SELECT root_task_id FROM tasks WHERE task_id=?', (envelope.parent_task_id,))
                if parent is None:
                    raise StoreConflict('child admission requires a known parent task')
                root_id = parent['root_task_id']
            receipt = Envelope(event_id=str(uuid5(UUID(envelope.event_id), receipt_name)), network_id=envelope.network_id,
                kind='receipt', sender=agent_id, recipient=envelope.sender, topic=None, task_id=task_id,
                parent_task_id=envelope.parent_task_id, correlation_id=envelope.event_id, session_id=envelope.session_id,
                created_at=now, deadline=envelope.deadline, traceparent=envelope.traceparent,
                payload={'event_id': envelope.event_id, 'recipient': agent_id})
            await self._connection.execute('INSERT INTO tasks(task_id,agent_id,status,version,input,input_envelope,session_id,root_task_id,definition_hash,deadline,updated_at,admitted_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (task_id, agent_id, 'accepted', 0, _json(envelope.payload), envelope.model_dump_json(), envelope.session_id,
                 root_id, definition_hash, _stamp(envelope.deadline), _stamp(now), _stamp(now)))
            # Only an addressed input's sender waits for its receipt; a topic publisher finished on the
            # broker ACK. The receipt is still journaled: duplicates of this event are answered with it.
            await self._insert_outbox(receipt, _stamp(now), publish=envelope.recipient is not None)
            await self._connection.execute('INSERT INTO processed_events VALUES(?,?,?,?)', (agent_id, envelope.event_id, task_id, receipt.event_id))
            await self._connection.execute('INSERT INTO receipts VALUES(?,?,?,?)', (agent_id, envelope.event_id, receipt.event_id, encode(receipt).decode('utf-8')))
            await self._connection.execute('INSERT INTO task_events(task_id,kind,version,recorded_at) VALUES(?,?,?,?)', (task_id, 'accepted', 0, _stamp(now)))
            return Admission(duplicate=False, task_id=task_id, receipt=receipt)
        return await self._mutate(operation)

    async def get_task(self, task_id: str) -> TaskRecord | None:
        task_id = _uuid.validate_python(task_id)
        async with self._lock:
            self._require_ready()
            async def operation():
                return _task(await self._fetch('SELECT * FROM tasks WHERE task_id=?', (task_id,)))
            return await self._retry(operation)

    async def get_input_envelope(self, task_id: str) -> Envelope | None:
        task_id = _uuid.validate_python(task_id)
        async with self._lock:
            self._require_ready()
            async def operation():
                row = await self._fetch('SELECT input_envelope FROM tasks WHERE task_id=?', (task_id,))
                return decode(row['input_envelope'].encode('utf-8')) if row else None
            return await self._retry(operation)

    async def claim(self, task_id: str, instance_id: str, ttl_seconds: float) -> Lease | None:
        return await self._claim(task_id, instance_id, ttl_seconds)

    async def claim_runtime(self, task_id, instance_id, ttl_seconds, *, concurrency, agent_id=None):
        """Lease a runnable task; with agent_id only while that agent still owns it (not after a handoff)."""
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError('concurrency must be positive')
        return await self._claim(task_id, instance_id, ttl_seconds, concurrency=concurrency, agent_id=agent_id)

    async def _claim(self, task_id, instance_id, ttl_seconds, *, concurrency=None, agent_id=None):
        task_id, instance_id = _uuid.validate_python(task_id), _uuid.validate_python(instance_id)
        if type(ttl_seconds) not in (float, int) or not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError('lease TTL must be positive and finite')
        return await self._mutate(lambda: self._claim_row(task_id, instance_id, ttl_seconds, concurrency, agent_id))

    async def begin_attempt(self, task_id, instance_id, ttl_seconds, *, concurrency, agent_id, gates,
                            retry_safe, max_attempts=3):
        """claim_runtime, every key gate and start_handler in one journal transaction.

        Returns the lease when the handler may run. None when the task cannot run here
        now: nothing changed if it was not claimable or a gate was busy; a task that
        start_handler settled (unknown, failed or expired) keeps that outcome.
        """
        task_id, instance_id = _uuid.validate_python(task_id), _uuid.validate_python(instance_id)
        if type(ttl_seconds) not in (float, int) or not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError('lease TTL must be positive and finite')
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError('concurrency must be positive')
        if type(retry_safe) is not bool or type(max_attempts) is not int or max_attempts < 1:
            raise ValueError('invalid handler retry policy')
        gates = [(subscription, key) for subscription, key in gates]
        if any(not isinstance(part, str) or not part for gate in gates for part in gate):
            raise ValueError('gate subscription and key must be nonempty strings')
        async def operation():
            lease = await self._claim_row(task_id, instance_id, ttl_seconds, concurrency, agent_id)
            if lease is None:
                return None
            for subscription, key in gates:
                if not await self._gate_row(task_id, subscription, key, lease):
                    raise _Busy()  # roll the claim back: an earlier task of this key or session runs first
            return lease if await self._start_row(lease, retry_safe, max_attempts) else None
        try:
            return await self._mutate(operation)
        except _Busy:
            return None

    async def _claim_row(self, task_id, instance_id, ttl_seconds, concurrency, agent_id):
        """Inside a transaction: lease a runnable task, or None."""
        now = self.clock.now()
        row = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (task_id,))
        if row is None or row['status'] not in ('accepted', 'running') or row['deadline'] <= _stamp(now):
            return None
        if agent_id is not None and row['agent_id'] != agent_id:
            return None  # handed off: the new owner runs it
        if row['lease_expires_at'] is not None and row['lease_expires_at'] > _stamp(now):
            return None
        if row['not_before'] is not None and row['not_before'] > _stamp(now):
            return None  # a journaled retry delay binds every replica
        if row['cancel_requested']:
            # Cancelled while a step ran. An attempt that may have acted stays unknown;
            # otherwise the task ends cancelled and its requester gets the reply saved then.
            if await self.effect_in_flight(task_id):
                await self._finish_unleased(row, 'outcome_unknown')
                return None
            await self._finish_unleased(row, 'cancelled', error=_json({'type': 'Cancelled',
                'message': 'an operator cancelled the task'}))
            if row['cancel_reply'] is not None:
                await self._insert_outbox(decode(row['cancel_reply'].encode()), _stamp(now))
            return None
        if concurrency is not None:
            active = await self._fetch("SELECT COUNT(*) AS count FROM tasks WHERE agent_id=? AND status='running' AND lease_expires_at>?", (row['agent_id'], _stamp(now)))
            if active['count'] >= concurrency:
                return None
        expires = now + timedelta(seconds=ttl_seconds)
        lease = Lease(task_id=task_id, instance_id=instance_id, token=row['lease_token'] + 1, expires_at=expires)
        cursor = await self._connection.execute("UPDATE tasks SET status='running',version=version+1,lease_instance_id=?,lease_token=?,lease_expires_at=?,lease_ttl=?,updated_at=? WHERE task_id=? AND version=? AND lease_token=?",
            (instance_id, lease.token, _stamp(expires), ttl_seconds, _stamp(now), task_id, row['version'], row['lease_token']))
        if cursor.rowcount != 1:
            raise VersionConflict('task changed during claim')
        await self._connection.execute('INSERT INTO task_events(task_id,kind,version,recorded_at) VALUES(?,?,?,?)', (task_id, 'claimed', row['version'] + 1, _stamp(now)))
        return lease

    def _check_lease(self, row, lease, now, *, waiting=False):
        # A Python handler awaiting a consult keeps its lease while its task is waiting_peer.
        active = row is not None and (row['status'] == 'running' or (waiting and row['status'] == 'waiting_peer'))
        if not active or row['lease_instance_id'] != lease.instance_id or row['lease_token'] != lease.token or row['lease_expires_at'] is None or row['lease_expires_at'] <= _stamp(now):
            raise LeaseLost('execution lease is no longer current')

    async def renew(self, lease: Lease) -> Lease:
        lease = Lease.model_validate(lease.model_dump())
        async def operation():
            now = self.clock.now()
            row = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (lease.task_id,))
            self._check_lease(row, lease, now, waiting=True)
            if row['deadline'] <= _stamp(now):
                raise LeaseLost('task deadline has expired')
            renewed = Lease(task_id=lease.task_id, instance_id=lease.instance_id, token=lease.token,
                            expires_at=now + timedelta(seconds=row['lease_ttl']))
            cursor = await self._connection.execute('UPDATE tasks SET lease_expires_at=? WHERE task_id=? AND lease_instance_id=? AND lease_token=? AND version=?',
                (_stamp(renewed.expires_at), lease.task_id, lease.instance_id, lease.token, row['version']))
            if cursor.rowcount != 1:
                raise LeaseLost('lease changed during renewal')
            return renewed
        return await self._mutate(operation)

    async def commit_step(self, lease: Lease, change: StepCommit) -> TaskRecord:
        lease = Lease.model_validate(lease.model_dump())
        change = StepCommit.model_validate(change.model_dump())
        if change.status in ('submitted', 'accepted'):
            raise ValueError('a running step cannot return to submission or admission')
        # Validate/serialize user values before taking a transaction lock.
        serialized = change.model_dump(mode='json')
        for envelope in change.outgoing:
            encode(envelope)
        async def operation():
            now = self.clock.now()
            row = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (lease.task_id,))
            self._check_lease(row, lease, now)
            if row['version'] != change.expected_version:
                raise VersionConflict('step expected_version differs from current task')
            if row['deadline'] <= _stamp(now) and change.status != 'expired':
                raise LeaseLost('task deadline has expired')
            if change.state is not None and row['session_id'] is None:
                raise ValueError('session state requires a session_id')
            version = row['version'] + 1
            await self._connection.execute('INSERT INTO steps VALUES(?,?,?,?,?)', (lease.task_id, change.step_id, version, _json(serialized), _stamp(now)))
            if change.state is not None:
                await self._connection.execute('INSERT INTO sessions VALUES(?,?,?,?) ON CONFLICT(agent_id,session_id) DO UPDATE SET state=excluded.state,updated_at=excluded.updated_at',
                    (row['agent_id'], row['session_id'], _json(change.state), _stamp(now)))
            for message in change.messages:
                await self._connection.execute('INSERT INTO messages(task_id,session_id,agent_id,step_id,message) VALUES(?,?,?,?,?)',
                    (lease.task_id, row['session_id'], row['agent_id'], change.step_id, message.model_dump_json()))
            for envelope in change.outgoing:
                await self._insert_outbox(envelope, _stamp(now))
            keep_owner = change.status == 'running'
            cursor = await self._connection.execute('UPDATE tasks SET status=?,version=?,result=?,error=?,waiting_for=?,lease_instance_id=?,lease_expires_at=?,updated_at=? WHERE task_id=? AND version=? AND lease_token=? AND lease_instance_id=?',
                (change.status, version, _json(change.result), _json(change.error), _json(change.waiting_for), lease.instance_id if keep_owner else None,
                 row['lease_expires_at'] if keep_owner else None, _stamp(now), lease.task_id, change.expected_version, lease.token, lease.instance_id))
            if cursor.rowcount != 1:
                raise VersionConflict('task changed during step commit')
            await self._connection.execute('INSERT INTO task_events(task_id,kind,version,recorded_at) VALUES(?,?,?,?)', (lease.task_id, change.status, version, _stamp(now)))
            if change.status in ('succeeded','failed','cancelled','expired'):
                await self._connection.execute('DELETE FROM execution_gates WHERE task_id=?', (lease.task_id,))
            return _task(await self._fetch('SELECT * FROM tasks WHERE task_id=?', (lease.task_id,)))
        return await self._mutate(operation)

    async def pending_runtime_count(self, agent_id):
        agent_id = validate_name(agent_id)
        async with self._lock:
            self._require_ready()
            async def operation():
                row = await self._fetch("SELECT COUNT(*) AS count FROM tasks WHERE agent_id=? AND status NOT IN ('succeeded','failed','cancelled','expired')", (agent_id,))
                return row['count']
            return await self._retry(operation)

    async def recoverable_page(self, agent_id, *, after_task_id=None, limit=64):
        agent_id = validate_name(agent_id)
        if type(limit) is not int or not 1 <= limit <= 64:
            raise ValueError('recovery page limit must be between 1 and 64')
        if after_task_id is not None:
            after_task_id = _uuid.validate_python(after_task_id)
        async with self._lock:
            self._require_ready()
            async def operation():
                now = _stamp(self.clock.now())
                rows = await self._all("SELECT * FROM tasks WHERE agent_id=? AND task_id>? AND status IN ('accepted','running','waiting_peer','waiting_tool','outcome_unknown') AND (lease_expires_at IS NULL OR lease_expires_at<=?) AND (not_before IS NULL OR not_before<=?) ORDER BY task_id LIMIT ?",
                    (agent_id, after_task_id or '', now, now, limit))
                return [_task(row) for row in rows]
            return await self._retry(operation)

    async def acquire_key_gate(self, task_id, subscription, key, *, lease):
        task_id = _uuid.validate_python(task_id)
        if not isinstance(subscription, str) or not subscription or not isinstance(key, str) or not key:
            raise ValueError('gate subscription and key must be nonempty strings')
        return await self._mutate(lambda: self._gate_row(task_id, subscription, key, lease))

    async def _gate_row(self, task_id, subscription, key, lease):
        """Inside a transaction: take the ordering gate of key for task_id under lease, or False."""
        row = await self._fetch('SELECT rowid AS admission_order,* FROM tasks WHERE task_id=?', (task_id,))
        if lease.task_id != task_id:
            raise LeaseLost('gate task differs from lease')
        self._check_lease(row, lease, self.clock.now())
        # The global session namespace serializes history across logical agents.
        scope = '$session' if subscription == '$session' else row['agent_id'] + ':' + subscription
        if subscription == '$session':
            prior = await self._fetch("SELECT 1 FROM tasks WHERE session_id=? AND rowid<? AND status NOT IN ('succeeded','failed','cancelled','expired') LIMIT 1", (key, row['admission_order']))
        else:
            prior = await self._fetch("SELECT 1 FROM tasks WHERE agent_id=? AND COALESCE(json_extract(input_envelope,'$.topic'),json_extract(input_envelope,'$.kind'))=? AND COALESCE(json_extract(input_envelope,'$.key'),session_id,json_extract(input_envelope,'$.event_id'))=? AND rowid<? AND status NOT IN ('succeeded','failed','cancelled','expired') LIMIT 1",
                (row['agent_id'], subscription, key, row['admission_order']))
        if prior:
            return False
        gate = await self._fetch('SELECT * FROM execution_gates WHERE subscription=? AND key=?', (scope, key))
        if gate and gate['task_id'] != task_id:
            return False
        await self._connection.execute('INSERT INTO execution_gates VALUES(?,?,?,?) ON CONFLICT(subscription,key) DO UPDATE SET lease_token=excluded.lease_token', (scope, key, task_id, row['lease_token']))
        return True

    async def release_key_gate(self, task_id, *, lease):
        task_id = _uuid.validate_python(task_id)
        async def operation():
            row = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (task_id,))
            if row is None or lease.task_id != task_id or row['lease_token'] != lease.token:
                raise LeaseLost('gate lease proof is no longer current')
            if row and row['status'] not in ('succeeded','failed','cancelled','expired'):
                # Active gates are released by the fenced commit, never an unscoped caller.
                raise LeaseLost('nonterminal gates require fenced task completion')
            await self._connection.execute('DELETE FROM execution_gates WHERE task_id=?', (task_id,))
        await self._mutate(operation)

    async def abandon_attempt(self, lease) -> None:
        """Undo a begin_attempt whose handler never ran (its caller was cancelled right after).

        The lease and its gates are released and the recorded attempt is taken back,
        so the next owner starts the handler instead of finding an interrupted attempt.
        """
        lease = Lease.model_validate(lease.model_dump())
        async def operation():
            row = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (lease.task_id,))
            if row is None or row['lease_token'] != lease.token or row['lease_instance_id'] != lease.instance_id:
                return
            attempt = await self._fetch('SELECT * FROM handler_attempts WHERE task_id=?', (lease.task_id,))
            if attempt is not None and attempt['lease_token'] == lease.token:
                if attempt['attempts'] <= 1:
                    await self._connection.execute('DELETE FROM handler_attempts WHERE task_id=?', (lease.task_id,))
                else:
                    await self._connection.execute("UPDATE handler_attempts SET attempts=attempts-1,last_outcome='failed' WHERE task_id=?",
                        (lease.task_id,))
            await self._connection.execute('UPDATE tasks SET lease_instance_id=NULL,lease_expires_at=NULL WHERE task_id=? AND lease_token=?',
                (lease.task_id, lease.token))
            await self._connection.execute('DELETE FROM execution_gates WHERE task_id=? AND lease_token=?', (lease.task_id, lease.token))
        await self._mutate(operation)

    async def release_outbox(self, rows: list) -> None:
        """Give back outbox rows this pump leased but will not publish now."""
        leased = [(row.envelope.event_id, row.lease_token, row.version) for row in rows]
        if not leased:
            return
        async def operation():
            for event_id, token, version in leased:
                await self._connection.execute('UPDATE outbox SET lease_instance_id=NULL,lease_expires_at=NULL,version=version+1 WHERE event_id=? AND lease_token=? AND version=?',
                    (event_id, token, version))
        await self._mutate(operation)

    async def release_claim(self, lease, *, error=None, retry_at=None):
        """Give up ownership; error records why a retry-safe attempt will run again.

        retry_at is journaled on the task, so no replica claims it earlier.
        """
        serialized_error = None if error is None else _json(error)
        not_before = None if retry_at is None else _stamp(_utc.validate_python(retry_at))
        async def operation():
            row = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (lease.task_id,))
            self._check_lease(row, lease, self.clock.now())
            await self._connection.execute("UPDATE tasks SET lease_instance_id=NULL,lease_expires_at=NULL WHERE task_id=?", (lease.task_id,))
            await self._connection.execute('DELETE FROM execution_gates WHERE task_id=? AND lease_token=?', (lease.task_id, lease.token))
            if serialized_error is not None:
                await self._connection.execute("UPDATE handler_attempts SET last_error=?,last_outcome='failed' WHERE task_id=?", (serialized_error, lease.task_id))
            if not_before is not None:
                await self._connection.execute('UPDATE tasks SET not_before=? WHERE task_id=?', (not_before, lease.task_id))
        await self._mutate(operation)

    async def start_handler(self, lease, *, retry_safe, max_attempts=3):
        if type(retry_safe) is not bool or type(max_attempts) is not int or max_attempts < 1:
            raise ValueError('invalid handler retry policy')
        return await self._mutate(lambda: self._start_row(lease, retry_safe, max_attempts))

    async def _start_row(self, lease, retry_safe, max_attempts):
        """Inside a transaction: record a handler attempt, or settle a task that may not run again."""
        row = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (lease.task_id,))
        self._check_lease(row, lease, self.clock.now())
        attempt = await self._fetch('SELECT * FROM handler_attempts WHERE task_id=?', (lease.task_id,))
        if attempt and attempt['lease_token'] == lease.token:
            return False
        if attempt and not attempt['retry_safe']:
            await self._finish_unleased(row, 'outcome_unknown')
            return False
        if attempt and attempt['attempts'] >= max_attempts:
            # An interrupted last attempt may have acted: unknown, not a clean failure.
            exhausted = 'outcome_unknown' if attempt['last_outcome'] == 'started' else 'failed'
            await self._finish_unleased(row, exhausted, error=attempt['last_error'])
            return False
        event = decode(row['input_envelope'].encode())
        if row['deadline'] <= _stamp(self.clock.now()) or (event.expires_at and event.expires_at <= self.clock.now()):
            await self._finish_unleased(row, 'expired')
            return False
        await self._connection.execute('INSERT INTO handler_attempts(task_id,lease_token,retry_safe,attempts,last_outcome) VALUES(?,?,?,1,\'started\') ON CONFLICT(task_id) DO UPDATE SET lease_token=excluded.lease_token,attempts=handler_attempts.attempts+1,last_outcome=\'started\'', (lease.task_id, lease.token, int(retry_safe)))
        return True

    async def _finish_unleased(self, row, status, *, error=None):
        """Finish a task outside a lease; error is an already serialized JSON value."""
        await self._connection.execute('UPDATE tasks SET status=?,error=COALESCE(?,error),version=version+1,lease_instance_id=NULL,lease_expires_at=NULL,updated_at=? WHERE task_id=?',
            (status, error, _stamp(self.clock.now()), row['task_id']))
        if status != 'outcome_unknown':
            await self._connection.execute('DELETE FROM execution_gates WHERE task_id=?', (row['task_id'],))
        await self._connection.execute('INSERT INTO task_events(task_id,kind,version,recorded_at) VALUES(?,?,?,?)', (row['task_id'], status, row['version']+1, _stamp(self.clock.now())))

    def _needs_expiry(self, row):
        if row is None or row['status'] not in ('accepted','running','waiting_peer','waiting_tool'):
            return False
        now = self.clock.now()
        if row['lease_expires_at'] is not None and row['lease_expires_at'] > _stamp(now):
            return False
        event = decode(row['input_envelope'].encode())
        return row['deadline'] <= _stamp(now) or (event.expires_at is not None and event.expires_at <= now)

    async def expire_task(self, task_id):
        task_id = _uuid.validate_python(task_id)
        # Most calls find a live task; decide that without the write lock.
        async with self._lock:
            self._require_ready()
            async def read():
                return await self._fetch('SELECT * FROM tasks WHERE task_id=?', (task_id,))
            row = await self._retry(read)
        if not self._needs_expiry(row):
            return _task(row)
        async def operation():
            row = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (task_id,))
            if not self._needs_expiry(row):
                return _task(row)
            attempt = await self._fetch('SELECT retry_safe,last_outcome,last_error FROM handler_attempts WHERE task_id=?', (task_id,))
            # A started attempt without a recorded outcome may have acted, even if retry-safe:
            # with no time left to retry it, its effect is unknown rather than expired.
            unknown = attempt is not None and (not attempt['retry_safe'] or attempt['last_outcome'] == 'started')
            # So may a tool attempt left running, e.g. by a remote tool service that died.
            unknown = unknown or await self._fetch("SELECT 1 FROM tool_executions WHERE task_id=? AND status='running' LIMIT 1", (task_id,)) is not None
            await self._finish_unleased(row, 'outcome_unknown' if unknown else 'expired',
                                        error=attempt['last_error'] if attempt and not unknown else None)
            return _task(await self._fetch('SELECT * FROM tasks WHERE task_id=?', (task_id,)))
        return await self._mutate(operation)

    async def load_session(self, agent_id, session_id):
        agent_id, session_id = validate_name(agent_id), _uuid.validate_python(session_id)
        from ...messages import Message
        async with self._lock:
            self._require_ready()
            async def operation():
                row = await self._fetch('SELECT state FROM sessions WHERE agent_id=? AND session_id=?', (agent_id, session_id))
                # An agent remembers its own turns; what others said in a shared session arrives by handoff.
                history = await self._all('SELECT message FROM messages WHERE session_id=? AND agent_id=? ORDER BY id', (session_id, agent_id))
                return json.loads(row['state']) if row else {}, [Message.model_validate_json(x['message']) for x in history]
            return await self._retry(operation)

    async def append_progress(self, task_id, envelope):
        task_id, envelope = _uuid.validate_python(task_id), decode(encode(envelope))
        if envelope.task_id != task_id or envelope.kind != 'progress':
            raise ValueError('progress must match task_id and kind')
        async def operation():
            row = await self._fetch('SELECT id,envelope FROM task_events WHERE task_id=? AND event_id=?', (task_id, envelope.event_id))
            if row:
                if row['envelope'] != envelope.model_dump_json():
                    raise StoreConflict('progress event_id conflicts with persisted envelope')
                return row['id']
            task = await self._fetch('SELECT version FROM tasks WHERE task_id=?', (task_id,))
            if task is None:
                raise StoreConflict('progress requires an admitted task')
            cursor = await self._connection.execute('INSERT INTO task_events(task_id,kind,version,recorded_at,event_id,envelope) VALUES(?,?,?,?,?,?)', (task_id, 'progress', task['version'], _stamp(self.clock.now()), envelope.event_id, envelope.model_dump_json()))
            return cursor.lastrowid
        return await self._mutate(operation)

    async def read_progress_page(self, task_id, *, after_sequence=0, limit=64):
        task_id = _uuid.validate_python(task_id)
        if type(after_sequence) is not int or after_sequence < 0 or type(limit) is not int or not 1 <= limit <= 64:
            raise ValueError('invalid progress cursor or limit')
        async with self._lock:
            self._require_ready()
            async def operation():
                rows = await self._all('SELECT id,envelope FROM task_events WHERE task_id=? AND id>? AND envelope IS NOT NULL ORDER BY id LIMIT ?', (task_id, after_sequence, limit))
                return [(row['id'], decode(row['envelope'].encode())) for row in rows]
            return await self._retry(operation)

    async def read_progress(self, task_id, after_sequence=0):
        task_id = _uuid.validate_python(task_id)
        if type(after_sequence) is not int or after_sequence < 0:
            raise ValueError('after_sequence must be nonnegative')
        async with self._lock:
            self._require_ready()
            async def operation():
                rows = await self._all('SELECT envelope FROM task_events WHERE task_id=? AND id>? AND envelope IS NOT NULL ORDER BY id', (task_id, after_sequence))
                return [decode(row['envelope'].encode()) for row in rows]
            return await self._retry(operation)

    async def list_recoverable(self, agent_id: str) -> list[TaskRecord]:
        agent_id = validate_name(agent_id)
        async with self._lock:
            self._require_ready()
            async def operation():
                now = _stamp(self.clock.now())
                found = await self._all("SELECT * FROM tasks WHERE agent_id=? AND deadline>? AND (status IN ('waiting_peer','waiting_tool') OR (status IN ('accepted','running') AND (lease_expires_at IS NULL OR lease_expires_at<=?))) ORDER BY task_id", (agent_id, now, now))
                return [_task(row) for row in found]
            return await self._retry(operation)

    async def lease_outbox(self, instance_id: str, limit: int) -> list[OutboxRecord]:
        instance_id = _uuid.validate_python(instance_id)
        if type(limit) is not int or limit < 1:
            raise ValueError('outbox limit must be a positive integer')
        async def due():
            now = _stamp(self.clock.now())
            return await self._fetch('SELECT 1 FROM outbox WHERE delivered=0 AND deadline>? AND (retry_at IS NULL OR retry_at<=?) AND (lease_expires_at IS NULL OR lease_expires_at<=?) LIMIT 1',
                (now, now, now))
        # An idle pump only reads: the write lock is taken when something is due (shared by every process).
        async with self._lock:
            self._require_ready()
            if await self._retry(due) is None:
                return []
        async def operation():
            now = self.clock.now()
            # The partial outbox_pending index skips delivered rows entirely.
            candidates = await self._all('SELECT * FROM outbox WHERE delivered=0 AND deadline>? AND (retry_at IS NULL OR retry_at<=?) AND (lease_expires_at IS NULL OR lease_expires_at<=?) ORDER BY rowid LIMIT ?',
                (_stamp(now), _stamp(now), _stamp(now), limit))
            result = []
            for row in candidates:
                token = str(uuid4())
                try:
                    record = OutboxRecord(envelope=decode(row['envelope'].encode('utf-8')),
                        targets=json.loads(row['targets']), received_targets=json.loads(row['received_targets']),
                        published_at=row['published_at'], retry_at=row['retry_at'], lease_token=token,
                        version=row['version'] + 1)
                except (ValueError, TypeError) as error:
                    # One unreadable row must not stop every publisher of this journal.
                    await self._quarantine_outbox(row, error)
                    continue
                cursor = await self._connection.execute('UPDATE outbox SET lease_instance_id=?,lease_token=?,lease_expires_at=?,version=version+1 WHERE event_id=? AND version=? AND lease_token IS ?',
                    (instance_id, token, _stamp(now + timedelta(seconds=OUTBOX_LEASE_SECONDS)), row['event_id'], row['version'], row['lease_token']))
                if cursor.rowcount != 1:
                    raise VersionConflict('outbox changed during lease')
                result.append(record)
            return result
        return await self._mutate(operation)

    async def _quarantine_outbox(self, row, error):
        """Lease an unreadable row forever so no publisher selects it again; it stays undelivered."""
        _logger.error('quarantined an unreadable outbox row: %s', type(error).__name__,
                      extra={'event_id': row['event_id']})
        await self._connection.execute('UPDATE outbox SET lease_instance_id=?,lease_token=?,lease_expires_at=?,version=version+1 WHERE event_id=? AND version=?',
            (QUARANTINE_OWNER, QUARANTINE_OWNER, QUARANTINE_UNTIL, row['event_id'], row['version']))

    async def record_broker_ack(self, event_id: str, *, lease_token: str | None = None,
                                expected_version: int | None = None, retry_at=None) -> bool:
        ack = self._checked_ack(event_id, lease_token, expected_version, retry_at)
        return await self._mutate(lambda: self._ack_row(*ack))

    async def record_broker_acks(self, acks: list) -> list[bool]:
        """record_broker_ack for a pump's whole batch in one transaction.

        acks holds (event_id, lease_token, expected_version, retry_at) per published row.
        """
        checked = [self._checked_ack(*ack) for ack in acks]
        if not checked:
            return []
        async def operation():
            return [await self._ack_row(*ack) for ack in checked]
        return await self._mutate(operation)

    @staticmethod
    def _checked_ack(event_id, lease_token, expected_version, retry_at):
        event_id = _uuid.validate_python(event_id)
        retry_at = None if retry_at is None else _utc.validate_python(retry_at)
        if (lease_token is None) != (expected_version is None):
            raise ValueError('broker ACK ownership requires both token and version')
        if lease_token is not None and (type(lease_token) is not str or not lease_token or
                type(expected_version) is not int or expected_version < 0):
            raise ValueError('broker ACK requires an opaque token and nonnegative version')
        return event_id, lease_token, expected_version, retry_at

    async def _ack_row(self, event_id, lease_token, expected_version, retry_at):
        """Inside a transaction: record one broker ACK; False when the row changed owner or version."""
        row = await self._fetch('SELECT * FROM outbox WHERE event_id=?', (event_id,))
        if row is None:
            raise StoreConflict('broker acknowledgment references unknown event')
        now = self.clock.now()
        if lease_token is None:
            if row['lease_instance_id'] is not None or row['lease_expires_at'] is not None:
                return False
        else:
            if lease_token != row['lease_token']:
                return False
            if expected_version != row['version']:
                # The same publisher finished, but a duplicate input may
                # have reactivated its receipt. Release only that owner;
                # leave pending publication facts/schedule intact.
                if row['lease_instance_id'] is not None or row['lease_expires_at'] is not None:
                    await self._connection.execute('UPDATE outbox SET lease_instance_id=NULL,lease_expires_at=NULL,version=version+1 WHERE event_id=? AND lease_token=? AND version=?',
                        (event_id, lease_token, row['version']))
                return False
        # Without expected receipts a broker ACK completes delivery.
        delivered = int(row['targets'] == '[]')
        await self._connection.execute('UPDATE outbox SET published_at=COALESCE(published_at,?),retry_at=?,delivered=MAX(delivered,?),lease_instance_id=NULL,lease_expires_at=NULL,version=version+1 WHERE event_id=? AND version=? AND lease_token IS ?',
            (_stamp(now), _stamp(retry_at if retry_at is not None else now + timedelta(seconds=1)), delivered, event_id, row['version'], row['lease_token']))
        return True

    async def reschedule_outbox(self, event_id: str, retry_at, *, lease_token: str,
                                expected_version: int) -> bool:
        event_id = _uuid.validate_python(event_id)
        retry_at = _utc.validate_python(retry_at)
        if (type(lease_token) is not str or not lease_token or
                type(expected_version) is not int or expected_version < 0):
            raise ValueError('reschedule requires an opaque token and nonnegative version')
        async def operation():
            row = await self._fetch('SELECT * FROM outbox WHERE event_id=?', (event_id,))
            if row is None:
                raise StoreConflict('reschedule references unknown event')
            if row['lease_token'] != lease_token:
                return False
            if row['version'] != expected_version:
                if row['lease_instance_id'] is not None or row['lease_expires_at'] is not None:
                    await self._connection.execute('UPDATE outbox SET lease_instance_id=NULL,lease_expires_at=NULL,version=version+1 WHERE event_id=? AND lease_token=? AND version=?',
                        (event_id, lease_token, row['version']))
                return False
            await self._connection.execute('UPDATE outbox SET retry_at=?,lease_instance_id=NULL,lease_expires_at=NULL,version=version+1 WHERE event_id=? AND lease_token=? AND version=?',
                (_stamp(retry_at), event_id, lease_token, expected_version))
            return True
        return await self._mutate(operation)

    async def confirm_receipt(self, event_id: str, recipient: str) -> None:
        """Record that recipient took event_id; idempotent and safe from every worker of the network.

        Every worker sees every receipt (so none depends on another being alive),
        but only the first confirmation writes: the others find it recorded with a read.
        """
        event_id, recipient = _uuid.validate_python(event_id), validate_name(recipient)
        async def recorded():
            row = await self._fetch('SELECT targets,received_targets FROM outbox WHERE event_id=?', (event_id,))
            if row is None or recipient not in json.loads(row['targets']):
                raise ReceiptNotExpected('receipt references unknown event or recipient')
            return recipient in json.loads(row['received_targets'])
        async with self._lock:
            self._require_ready()
            if await self._retry(recorded):
                return
        async def operation():
            row = await self._fetch('SELECT * FROM outbox WHERE event_id=?', (event_id,))
            if row is None or recipient not in json.loads(row['targets']):
                raise ReceiptNotExpected('receipt references unknown event or recipient')
            received = json.loads(row['received_targets'])
            if recipient in received:
                return
            received.append(recipient)
            delivered = int(set(json.loads(row['targets'])) <= set(received))
            await self._connection.execute('UPDATE outbox SET received_targets=?,delivered=?,version=version+1 WHERE event_id=? AND version=?',
                (_json(received), delivered, event_id, row['version']))
        await self._mutate(operation)

    @staticmethod
    def _tool_lease(row):
        if row is None:
            return None
        return ToolLease(tool_call_id=row['tool_call_id'], token=row['token'], attempt=row['attempts'],
            status=row['status'], arguments=json.loads(row['arguments']), requested=json.loads(row['requested']),
            retry_safe=bool(row['retry_safe']),
            max_attempts=row['max_attempts'], result=json.loads(row['result']),
            error=json.loads(row['error']), hook_status=row['hook_status'],
            hook_result=json.loads(row['hook_result']), hook_error=json.loads(row['hook_error']))

    async def _tool_row(self, tool_call_id):
        """Read one tool row inside the caller's transaction."""
        return self._tool_lease(await self._fetch('SELECT * FROM tool_executions WHERE tool_call_id=?', (tool_call_id,)))

    @staticmethod
    def _resumed_status(row):
        """Next status of a pending call: a new attempt, or a final outcome without one."""
        if row['status'] == 'running' and not row['retry_safe']:
            return 'outcome_unknown'  # an attempt may have acted before the interruption
        if row['attempts'] >= row['max_attempts']:
            return 'failed'
        return 'running'

    async def get_tool_execution(self, tool_call_id: str) -> ToolLease | None:
        tool_call_id = _uuid.validate_python(tool_call_id)
        async with self._lock:
            self._require_ready()
            async def operation():
                return await self._tool_row(tool_call_id)
            return await self._retry(operation)

    async def reserve_tool(self, call: ToolCall, arguments: dict, *, lease: Lease, instance_id: str,
                           retry_safe: bool, max_attempts: int, output_schema: dict | None = None) -> ToolLease:
        """Start an attempt fenced by the task lease, or return the call's settled outcome.

        A row left running by an interrupted attempt becomes outcome_unknown
        unless the tool is retry-safe and has attempts left.
        """
        call = ToolCall.model_validate(call.model_dump())
        lease = Lease.model_validate(lease.model_dump())
        instance_id = _uuid.validate_python(instance_id)
        if type(retry_safe) is not bool or type(max_attempts) is not int or max_attempts < 1:
            raise ValueError('invalid tool retry policy')
        requested = _json(call.model_dump(mode='json'))
        effective = _json(arguments)
        schema = None if output_schema is None else _json(output_schema)
        async def operation():
            now = self.clock.now()
            task = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (lease.task_id,))
            self._check_lease(task, lease, now)
            if task['cancel_requested']:
                settled = await self._fetch('SELECT status FROM tool_executions WHERE tool_call_id=?', (call.tool_call_id,))
                if settled is None or settled['status'] in ('running', 'retry'):
                    raise TaskCancelled('an operator cancelled the task; no new tool call starts')
            # Checked in the transaction: once a call's deadline or expiry passed,
            # no attempt can start, so a caller reading after it sees the final word.
            stale = now >= call.deadline or (call.expires_at is not None and now >= call.expires_at)
            row = await self._fetch('SELECT * FROM tool_executions WHERE tool_call_id=?', (call.tool_call_id,))
            if row is None:
                status = 'expired' if stale else 'running'
                error = _json({'type': 'InputExpired', 'message': 'input expired before the tool started'} if stale else None)
                await self._connection.execute('INSERT INTO tool_executions(tool_call_id,task_id,tool_id,revision,requested,arguments,retry_safe,max_attempts,status,attempts,token,instance_id,updated_at,error,output_schema) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (call.tool_call_id, lease.task_id, call.tool_id, call.revision, requested, effective,
                     int(retry_safe), max_attempts, status, 0 if stale else 1, 1, instance_id, _stamp(now), error, schema))
                return await self._tool_row(call.tool_call_id)
            if row['task_id'] != lease.task_id or row['requested'] != requested:
                raise StoreConflict('tool_call_id already identifies a different call')
            if row['status'] not in ('running', 'retry'):
                return self._tool_lease(row)
            if stale:
                return await self._expire_pending_tool(row, now)
            status = self._resumed_status(row)
            if status != 'running':
                error = row['error'] if row['status'] == 'retry' else _json({'type': 'Interrupted',
                    'message': 'an attempt was interrupted before its outcome was recorded'})
                await self._connection.execute('UPDATE tool_executions SET status=?,error=?,updated_at=? WHERE tool_call_id=?',
                    (status, error, _stamp(now), call.tool_call_id))
            else:
                await self._connection.execute('UPDATE tool_executions SET status=?,attempts=attempts+1,token=token+1,instance_id=?,updated_at=? WHERE tool_call_id=?',
                    ('running', instance_id, _stamp(now), call.tool_call_id))
            return await self._tool_row(call.tool_call_id)
        return await self._mutate(operation)

    async def _settle_tool(self, tool_lease, status, *, result=None, error=None):
        result, error = _json(result), _json(error)
        async def operation():
            cursor = await self._connection.execute("UPDATE tool_executions SET status=?,result=?,error=?,updated_at=? WHERE tool_call_id=? AND token=? AND status='running'",
                (status, result, error, _stamp(self.clock.now()), tool_lease.tool_call_id, tool_lease.token))
            if cursor.rowcount != 1:
                raise LeaseLost('tool attempt is no longer current')
            return await self._tool_row(tool_lease.tool_call_id)
        return await self._mutate(operation)

    async def finish_tool(self, tool_lease: ToolLease, *, status: str, result=None, error=None) -> ToolLease:
        """Record an attempt's outcome; the attempt token fences out late writers."""
        if status not in ('succeeded', 'failed', 'expired', 'outcome_unknown'):
            raise ValueError('invalid final tool status')
        return await self._settle_tool(tool_lease, status, result=result, error=error)

    async def mark_tool_unknown(self, tool_lease: ToolLease, *, error=None) -> ToolLease:
        return await self._settle_tool(tool_lease, 'outcome_unknown', error=error)

    async def release_tool(self, tool_lease: ToolLease, *, error) -> None:
        """A retry-safe attempt failed; the next reserve may start another one."""
        await self._settle_tool(tool_lease, 'retry', error=error)

    async def expire_tool(self, tool_call_id: str, *, lease: Lease) -> ToolLease | None:
        """Close a pending call whose input went stale before another attempt could start."""
        tool_call_id = _uuid.validate_python(tool_call_id)
        async def operation():
            now = self.clock.now()
            task = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (lease.task_id,))
            self._check_lease(task, lease, now)
            row = await self._fetch('SELECT * FROM tool_executions WHERE tool_call_id=?', (tool_call_id,))
            if row is None or row['status'] not in ('running', 'retry'):
                return self._tool_lease(row)
            return await self._expire_pending_tool(row, now)
        return await self._mutate(operation)

    async def _expire_pending_tool(self, row, now):
        """Settle a pending call that may not start another attempt; inside a transaction."""
        if row['status'] == 'running' and not row['retry_safe']:
            status, error = 'outcome_unknown', _json({'type': 'Interrupted',
                'message': 'an attempt was interrupted before its outcome was recorded'})
        else:
            status, error = 'expired', _json({'type': 'InputExpired',
                'message': 'input expired before another attempt could start'})
        await self._connection.execute('UPDATE tool_executions SET status=?,error=?,updated_at=? WHERE tool_call_id=?',
            (status, error, _stamp(now), row['tool_call_id']))
        return await self._tool_row(row['tool_call_id'])

    async def abandon_tool(self, tool_call_id: str) -> ToolLease | None:
        """Settle a call whose deadline passed with no outcome, once: a later finish is refused.

        A running attempt may have acted, so it becomes outcome_unknown; a call
        waiting for a retry becomes expired. A settled call is returned as is.
        """
        tool_call_id = _uuid.validate_python(tool_call_id)
        async def operation():
            row = await self._fetch('SELECT * FROM tool_executions WHERE tool_call_id=?', (tool_call_id,))
            if row is None or row['status'] not in ('running', 'retry'):
                return self._tool_lease(row)
            if row['status'] == 'running':
                status, error = 'outcome_unknown', _json({'type': 'DeadlineExceeded',
                    'message': 'the tool attempt did not report its outcome before the deadline'})
            else:
                status, error = 'expired', row['error']
            await self._connection.execute("UPDATE tool_executions SET status=?,error=?,updated_at=? WHERE tool_call_id=? AND status IN ('running','retry')",
                (status, error, _stamp(self.clock.now()), tool_call_id))
            return await self._tool_row(tool_call_id)
        return await self._mutate(operation)

    async def _submit_children(self, envelopes, now):
        """Record consult requests in the outbox once; a replayed one is already there."""
        for envelope in envelopes:
            if await self._fetch('SELECT 1 FROM submissions WHERE task_id=?', (envelope.task_id,)) is None:
                await self._insert_outbox(envelope, _stamp(now))
                await self._connection.execute('INSERT INTO submissions VALUES(?,?,?)', (envelope.task_id, envelope.event_id, _stamp(now)))

    async def begin_consult(self, lease: Lease, envelopes: list, *, children: list) -> TaskRecord:
        """A managed loop waits for its consult children: requests, waiting_peer and released lease at once.

        The execution slot is free while it waits; a session gate stays with the task.
        """
        lease = Lease.model_validate(lease.model_dump())
        envelopes = [decode(encode(envelope)) for envelope in envelopes]
        waiting = _json({'kind': 'loop', 'children': list(children)})
        async def operation():
            now = self.clock.now()
            row = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (lease.task_id,))
            self._check_lease(row, lease, now)
            await self._submit_children(envelopes, now)
            # Children that already ended (e.g. before a restart) leave nothing to wait for.
            status = 'accepted' if await self._children_ended(children) else 'waiting_peer'
            await self._connection.execute("UPDATE tasks SET status=?,waiting_for=?,version=version+1,lease_instance_id=NULL,lease_expires_at=NULL,updated_at=? WHERE task_id=?",
                (status, waiting if status == 'waiting_peer' else 'null', _stamp(now), lease.task_id))
            if status == 'waiting_peer':
                # Waiting is not an interrupted attempt: expiry then means expired, not unknown.
                await self._connection.execute("UPDATE handler_attempts SET last_outcome='waiting' WHERE task_id=?", (lease.task_id,))
            else:
                await self._connection.execute('DELETE FROM handler_attempts WHERE task_id=?', (lease.task_id,))
            await self._connection.execute('INSERT INTO task_events(task_id,kind,version,recorded_at) VALUES(?,?,?,?)',
                (lease.task_id, 'waiting' if status == 'waiting_peer' else 'resumed', row['version'] + 1, _stamp(now)))
            return _task(await self._fetch('SELECT * FROM tasks WHERE task_id=?', (lease.task_id,)))
        return await self._mutate(operation)

    async def wait_for_child(self, lease: Lease, envelope: Envelope) -> None:
        """A Python handler awaits one consult: it keeps its lease, but no longer counts as running."""
        lease = Lease.model_validate(lease.model_dump())
        envelope = decode(encode(envelope))
        async def operation():
            now = self.clock.now()
            row = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (lease.task_id,))
            self._check_lease(row, lease, now)
            await self._submit_children([envelope], now)
            # Same lease, same attempt: the version stays, so the handler's step still commits.
            await self._connection.execute("UPDATE tasks SET status='waiting_peer',waiting_for=?,updated_at=? WHERE task_id=?",
                (_json({'kind': 'handler', 'children': [envelope.task_id]}), _stamp(now), lease.task_id))
        await self._mutate(operation)

    async def resume_waiting(self, lease: Lease) -> None:
        """The awaiting Python handler continues under the same lease."""
        lease = Lease.model_validate(lease.model_dump())
        async def operation():
            now = self.clock.now()
            row = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (lease.task_id,))
            self._check_lease(row, lease, now, waiting=True)
            await self._connection.execute("UPDATE tasks SET status='running',waiting_for='null',updated_at=? WHERE task_id=?",
                (_stamp(now), lease.task_id))
        await self._mutate(operation)

    async def _children_ended(self, children) -> bool:
        if not children:
            return False
        marks = ','.join('?' * len(children))
        ended = await self._fetch(f"SELECT COUNT(*) AS count FROM tasks WHERE task_id IN ({marks}) AND status IN ('succeeded','failed','cancelled','expired','outcome_unknown')", tuple(children))
        return ended['count'] == len(children)

    async def resume_parent(self, parent_id: str, child_id: str, *, receipt: Envelope | None = None) -> bool:
        """Make a waiting managed loop runnable once all its children ended; True only the first time.

        receipt, the parent's confirmation of the child's reply, is recorded either way.
        """
        parent_id, child_id = _uuid.validate_python(parent_id), _uuid.validate_python(child_id)
        receipt = None if receipt is None else decode(encode(receipt))
        if receipt is None and not await self._resumable(parent_id, child_id):
            return False  # recovery polls often; decide the common no-op without a write transaction
        async def operation():
            now = self.clock.now()
            if receipt is not None and await self._fetch('SELECT 1 FROM outbox WHERE event_id=?', (receipt.event_id,)) is None:
                await self._insert_outbox(receipt, _stamp(now))
            row = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (parent_id,))
            if row is None or row['status'] != 'waiting_peer':
                return False
            waiting = json.loads(row['waiting_for'])
            if not isinstance(waiting, dict) or waiting.get('kind') != 'loop' or child_id not in waiting.get('children', ()):
                return False
            if not await self._children_ended(waiting['children']):
                return False
            await self._connection.execute("UPDATE tasks SET status='accepted',waiting_for='null',version=version+1,updated_at=? WHERE task_id=?", (_stamp(now), parent_id))
            # The loop continues from its journal; the wait is not another handler attempt.
            await self._connection.execute('DELETE FROM handler_attempts WHERE task_id=?', (parent_id,))
            await self._connection.execute('INSERT INTO task_events(task_id,kind,version,recorded_at) VALUES(?,?,?,?)', (parent_id, 'resumed', row['version'] + 1, _stamp(now)))
            return True
        return await self._mutate(operation)

    async def _resumable(self, parent_id, child_id) -> bool:
        async with self._lock:
            self._require_ready()
            async def read():
                row = await self._fetch('SELECT status, waiting_for FROM tasks WHERE task_id=?', (parent_id,))
                if row is None or row['status'] != 'waiting_peer':
                    return False
                waiting = json.loads(row['waiting_for'])
                return (isinstance(waiting, dict) and waiting.get('kind') == 'loop'
                        and child_id in waiting.get('children', ()) and await self._children_ended(waiting['children']))
            return await self._retry(read)

    async def reclaim_handler_wait(self, task_id: str) -> TaskRecord | None:
        """A Python handler that died while awaiting a consult: run it again if retry-safe, else unknown."""
        task_id = _uuid.validate_python(task_id)
        async def operation():
            now = self.clock.now()
            row = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (task_id,))
            if row is None or row['status'] != 'waiting_peer' or (row['lease_expires_at'] is not None and row['lease_expires_at'] > _stamp(now)):
                return _task(row)
            waiting = json.loads(row['waiting_for'])
            if not isinstance(waiting, dict) or waiting.get('kind') != 'handler':
                return _task(row)
            attempt = await self._fetch('SELECT retry_safe FROM handler_attempts WHERE task_id=?', (task_id,))
            if attempt is not None and attempt['retry_safe']:
                await self._connection.execute("UPDATE tasks SET status='accepted',waiting_for='null',version=version+1,lease_instance_id=NULL,lease_expires_at=NULL,updated_at=? WHERE task_id=?", (_stamp(now), task_id))
                await self._connection.execute("UPDATE handler_attempts SET last_outcome='interrupted' WHERE task_id=?", (task_id,))
            else:
                await self._finish_unleased(row, 'outcome_unknown')
            return _task(await self._fetch('SELECT * FROM tasks WHERE task_id=?', (task_id,)))
        return await self._mutate(operation)

    async def open_conversation(self, agent_id: str, session_id: str | None = None) -> ConversationRecord:
        """Start a conversation answered by agent_id, or return the existing one with that ID."""
        agent_id = validate_name(agent_id)
        session_id = str(uuid4()) if session_id is None else _uuid.validate_python(session_id)
        async def operation():
            now = _stamp(self.clock.now())
            await self._connection.execute('INSERT INTO conversations(session_id,active_agent,version,created_at,updated_at) VALUES(?,?,0,?,?) ON CONFLICT(session_id) DO NOTHING',
                (session_id, agent_id, now, now))
            return self._conversation(await self._fetch('SELECT * FROM conversations WHERE session_id=?', (session_id,)))
        return await self._mutate(operation)

    async def get_conversation(self, session_id: str) -> ConversationRecord | None:
        session_id = _uuid.validate_python(session_id)
        async with self._lock:
            self._require_ready()
            async def operation():
                return self._conversation(await self._fetch('SELECT * FROM conversations WHERE session_id=?', (session_id,)))
            return await self._retry(operation)

    @staticmethod
    def _conversation(row):
        if row is None:
            return None
        return ConversationRecord(session_id=row['session_id'], active_agent=row['active_agent'],
                                  version=row['version'], current_task_id=row['current_task_id'])

    async def transfer(self, lease: Lease, target: str, context: dict, scope: str, *,
                       counted: bool = True, max_handoffs: int = 8) -> TaskRecord:
        """Give a task to target in one transaction: owner, fencing, conversation owner and notice.

        task_id stays; the old lease can no longer commit. scope 'conversation'
        also makes target the conversation's active agent; 'task' moves only
        this task. An uncounted transfer re-addresses a queued conversation message.
        """
        lease = Lease.model_validate(lease.model_dump())
        target = validate_name(target)
        if scope not in ('task', 'conversation'):
            raise ValueError('handoff scope must be task or conversation')
        serialized = _json(context)
        if len(serialized.encode()) > MAX_ENVELOPE_BYTES:
            raise HandoffRefused('the handoff context exceeds 1 MiB')  # never truncated silently
        async def operation():
            now = self.clock.now()
            row = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (lease.task_id,))
            self._check_lease(row, lease, now)
            if counted and row['handoffs'] >= max_handoffs:
                raise HandoffRefused(f'a task may be handed off at most {max_handoffs} times')
            registered = await self._fetch('SELECT fingerprint FROM logical_definitions WHERE logical_id=?', (target,))
            if registered is None:
                raise HandoffRefused(f'{target} has no registered definition')
            event = decode(row['input_envelope'].encode())
            moved = event.model_copy(update={'recipient': target}) if event.recipient is not None else event
            if scope == 'conversation':
                conversation = await self._fetch('SELECT 1 FROM conversations WHERE session_id=?', (row['session_id'],))
                if conversation is None:
                    raise HandoffRefused('a conversation handoff needs a conversation')
                await self._connection.execute('UPDATE conversations SET active_agent=?,version=version+1,current_task_id=?,updated_at=? WHERE session_id=?',
                    (target, row['task_id'], _stamp(now), row['session_id']))
            await self._connection.execute("UPDATE tasks SET agent_id=?,definition_hash=?,status='accepted',version=version+1,lease_instance_id=NULL,lease_expires_at=NULL,input_envelope=?,transfer_context=?,handoffs=handoffs+?,updated_at=? WHERE task_id=?",
                (target, registered['fingerprint'], moved.model_dump_json(), serialized, int(counted), _stamp(now), row['task_id']))
            await self._connection.execute('DELETE FROM handler_attempts WHERE task_id=?', (row['task_id'],))
            # The conversation's session gate stays with the task; the old owner's own gates go.
            await self._connection.execute("DELETE FROM execution_gates WHERE task_id=? AND subscription<>'$session'", (row['task_id'],))
            await self._connection.execute('INSERT INTO task_events(task_id,kind,version,recorded_at) VALUES(?,?,?,?)',
                (row['task_id'], 'handoff' if counted else 'redirected', row['version'] + 1, _stamp(now)))
            notice = Envelope(event_id=str(uuid5(UUID(row['task_id']), f"handoff:{row['handoffs']}:{target}:{row['version']}")),
                network_id=event.network_id, kind='handoff', sender=row['agent_id'], recipient=target, topic=None,
                task_id=row['task_id'], parent_task_id=event.parent_task_id, correlation_id=event.event_id,
                session_id=event.session_id, created_at=now, deadline=event.deadline, traceparent=event.traceparent,
                payload={'task_id': row['task_id']})
            await self._insert_outbox(notice, _stamp(now))
            return _task(await self._fetch('SELECT * FROM tasks WHERE task_id=?', (row['task_id'],)))
        return await self._mutate(operation)

    async def freeze_tool_catalog(self, lease: Lease, *, step: int, catalog: list) -> list:
        """Record the discovered tools a model step offers; the first recorded catalog wins."""
        lease = Lease.model_validate(lease.model_dump())
        if type(step) is not int or step < 0 or not isinstance(catalog, list):
            raise ValueError('a catalog needs a nonnegative step and a list of tools')
        serialized = _json(catalog)
        async def operation():
            now = self.clock.now()
            task = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (lease.task_id,))
            self._check_lease(task, lease, now)
            row = await self._fetch('SELECT catalog FROM model_step_catalogs WHERE task_id=? AND step=?', (lease.task_id, step))
            if row is not None:
                return json.loads(row['catalog'])
            await self._connection.execute('INSERT INTO model_step_catalogs VALUES(?,?,?,?)',
                (lease.task_id, step, serialized, _stamp(now)))
            return json.loads(serialized)
        return await self._mutate(operation)

    async def record_tool_hook(self, tool_call_id: str, *, status: str, result=None, error=None) -> ToolLease:
        """Journal the after-hook outcome once, separately from the external result."""
        tool_call_id = _uuid.validate_python(tool_call_id)
        if status not in ('succeeded', 'failed'):
            raise ValueError('invalid hook status')
        result, error = _json(result), _json(error)
        async def operation():
            await self._connection.execute("UPDATE tool_executions SET hook_status=?,hook_result=?,hook_error=?,updated_at=? WHERE tool_call_id=? AND status='succeeded' AND hook_status IS NULL",
                (status, result, error, _stamp(self.clock.now()), tool_call_id))
            return await self._tool_row(tool_call_id)
        return await self._mutate(operation)

    MODEL_ATTEMPTS_PER_STEP = 3

    @staticmethod
    def _model_attempt(row):
        if row is None:
            return None
        return ModelAttempt(attempt_id=row['attempt_id'], task_id=row['task_id'], root_task_id=row['root_task_id'],
            step=row['step'], attempt=row['attempt'], request_hash=row['request_hash'], status=row['status'],
            turn=json.loads(row['turn']), error=json.loads(row['error']))

    async def _root_usage(self, root_task_id):
        rows = await self._all('SELECT status, input_tokens, output_tokens, usage_known, cost_usd FROM model_attempts WHERE root_task_id=?', (root_task_id,))
        usage = {'calls': len(rows), 'tokens': 0, 'unknown_usage': 0, 'cost': Decimal(0), 'unknown_cost': 0}
        for row in rows:
            if row['status'] == 'running':
                continue  # in flight: neither spent nor unknown yet
            if row['usage_known']:
                usage['tokens'] += (row['input_tokens'] or 0) + (row['output_tokens'] or 0)
            else:
                usage['unknown_usage'] += 1
            if row['cost_usd'] is None:
                usage['unknown_cost'] += 1
            else:
                usage['cost'] += Decimal(row['cost_usd'])
        return usage

    @staticmethod
    def _budget_refusal(usage, budget, max_calls):
        if usage['calls'] >= max_calls:
            return ModelBudgetExceeded('the root task used its model call budget')
        if budget is None:
            return None
        if budget.max_tokens is not None and (usage['unknown_usage'] or usage['tokens'] >= budget.max_tokens):
            return ModelBudgetExceeded('the root task token budget is exhausted or has unknown usage')
        if budget.max_cost_usd is not None and (usage['unknown_cost'] or usage['cost'] >= Decimal(budget.max_cost_usd)):
            return ModelBudgetExceeded('the root task cost budget is exhausted or has unknown cost')
        return None

    async def reserve_model_attempt(self, lease, *, step, request_hash, pool_name, pool_limit, budget,
                                    max_calls, instance_id, ttl_seconds=60):
        """Replay a completed step, or atomically check budget and pool before a new request.

        A running attempt left by a crashed or replaced owner becomes unknown:
        it may have been billed. Refusals commit that bookkeeping first.
        """
        lease = Lease.model_validate(lease.model_dump())
        instance_id = _uuid.validate_python(instance_id)
        if type(step) is not int or step < 0 or type(pool_limit) is not int or pool_limit < 1:
            raise ValueError('invalid model attempt step or pool limit')
        async def operation():
            now = self.clock.now()
            task = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (lease.task_id,))
            self._check_lease(task, lease, now)
            rows = await self._all('SELECT * FROM model_attempts WHERE task_id=? AND step=? ORDER BY attempt', (lease.task_id, step))
            for row in rows:
                if row['status'] == 'completed':
                    if row['request_hash'] != request_hash:
                        raise StoreConflict('a completed model step differs from the replayed request')
                    return self._model_attempt(row)
            if task['cancel_requested']:
                raise TaskCancelled('an operator cancelled the task; no new model step starts')
            await self._connection.execute("UPDATE model_attempts SET status='unknown',updated_at=? WHERE task_id=? AND step=? AND status='running'",
                (_stamp(now), lease.task_id, step))
            if len(rows) >= self.MODEL_ATTEMPTS_PER_STEP:
                return ModelError('exhausted', 'the model step failed on every allowed attempt')
            refusal = self._budget_refusal(await self._root_usage(task['root_task_id']), budget, max_calls)
            if refusal is not None:
                return refusal
            busy = await self._fetch("SELECT COUNT(*) AS count FROM model_attempts WHERE pool_name=? AND status='running' AND lease_expires_at>?",
                (pool_name, _stamp(now)))
            if busy['count'] >= pool_limit:
                return ModelCapacity('the provider pool is full')
            number = len(rows) + 1
            attempt_id = str(uuid5(UUID(lease.task_id), f'model:{step}:{number}'))
            await self._connection.execute('INSERT INTO model_attempts(attempt_id,task_id,root_task_id,step,attempt,request_hash,pool_name,status,instance_id,lease_expires_at,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (attempt_id, lease.task_id, task['root_task_id'], step, number, request_hash, pool_name, 'running',
                 instance_id, _stamp(now + timedelta(seconds=ttl_seconds)), _stamp(now), _stamp(now)))
            return self._model_attempt(await self._fetch('SELECT * FROM model_attempts WHERE attempt_id=?', (attempt_id,)))
        outcome = await self._mutate(operation)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def renew_model_attempt(self, attempt_id, ttl_seconds=60):
        attempt_id = _uuid.validate_python(attempt_id)
        async def operation():
            await self._connection.execute("UPDATE model_attempts SET lease_expires_at=? WHERE attempt_id=? AND status='running'",
                (_stamp(self.clock.now() + timedelta(seconds=ttl_seconds)), attempt_id))
        await self._mutate(operation)

    async def finish_model_attempt(self, attempt_id, turn, *, input_tokens, output_tokens, cost_usd=None,
                                   pricing_revision=None):
        """Journal a completed turn once; a duplicate finish never counts usage twice."""
        attempt_id = _uuid.validate_python(attempt_id)
        serialized = _json(turn.model_dump(mode='json') if hasattr(turn, 'model_dump') else turn)
        known = int(input_tokens is not None and output_tokens is not None)
        async def operation():
            await self._connection.execute("UPDATE model_attempts SET status='completed',turn=?,input_tokens=?,output_tokens=?,usage_known=?,cost_usd=?,pricing_revision=?,updated_at=? WHERE attempt_id=? AND status='running'",
                (serialized, input_tokens, output_tokens, known, cost_usd, pricing_revision, _stamp(self.clock.now()), attempt_id))
            return self._model_attempt(await self._fetch('SELECT * FROM model_attempts WHERE attempt_id=?', (attempt_id,)))
        return await self._mutate(operation)

    async def fail_model_attempt(self, attempt_id, error, *, usage_unknown, input_tokens=None,
                                 output_tokens=None, cost_usd=None):
        """Journal a failed attempt with whatever usage the provider reported.

        A rejected request spent nothing; a billed failure keeps its reported
        tokens; an interrupted one may have been billed and stays unknown.
        """
        attempt_id = _uuid.validate_python(attempt_id)
        if usage_unknown:
            status, known, values = 'unknown', 0, (None, None, None)
        elif input_tokens is None and output_tokens is None:
            status, known, values = 'failed', 1, (0, 0, '0')
        else:
            status, known, values = 'failed', 1, (input_tokens, output_tokens, cost_usd)
        error = _json(error)
        async def operation():
            await self._connection.execute("UPDATE model_attempts SET status=?,usage_known=?,input_tokens=?,output_tokens=?,cost_usd=?,error=?,updated_at=? WHERE attempt_id=? AND status='running'",
                (status, known, *values, error, _stamp(self.clock.now()), attempt_id))
            return self._model_attempt(await self._fetch('SELECT * FROM model_attempts WHERE attempt_id=?', (attempt_id,)))
        return await self._mutate(operation)

    async def model_usage(self, root_task_id):
        root_task_id = _uuid.validate_python(root_task_id)
        async with self._lock:
            self._require_ready()
            async def operation():
                usage = await self._root_usage(root_task_id)
                return ModelUsage(calls=usage['calls'], tokens=usage['tokens'], unknown_usage=usage['unknown_usage'],
                                  cost_usd=str(usage['cost']), unknown_cost=usage['unknown_cost'])
            return await self._retry(operation)

    async def close(self) -> None:
        async with self._lock:
            if self._connection is not None:
                await self._connection.close()
                self._connection = None
            self._closed = True
            self._ready = False
