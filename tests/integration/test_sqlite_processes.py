"""Disk-backed, independent-process lock, durability and write failure checks."""
import asyncio
import multiprocessing
import sqlite3
import time
import pytest
from tests.contract.test_store import OWNER, OTHER, rows


def compete(path, envelope_json, instance_id, barrier, queue):
    from agentweave.adapters.storage.sqlite import SQLiteStore
    from agentweave.messages import Envelope
    async def run():
        store = SQLiteStore(path)
        try:
            barrier.wait(timeout=20)
            await store.migrate()
            envelope = Envelope.model_validate_json(envelope_json)
            await store.admit(envelope, 'v1')
            lease = await store.claim(envelope.task_id, instance_id, 30)
            queue.put(('ok', lease.token if lease else None))
        except Exception as exc:
            queue.put(('error', repr(exc)))
        finally:
            await store.close()
    asyncio.run(run())


def test_concurrent_migrations_admission_and_claim_have_one_owner(tmp_path, envelope_factory):
    ctx = multiprocessing.get_context('spawn')
    barrier, queue = ctx.Barrier(2), ctx.Queue()
    event = envelope_factory()
    # The child runs on the system clock: the event is created now and its deadline is live.
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    event = event.model_copy(update={'created_at': now, 'deadline': now + timedelta(minutes=5)})
    path = tmp_path / 'shared.sqlite3'
    processes = [ctx.Process(target=compete, args=(str(path), event.model_dump_json(), owner, barrier, queue)) for owner in (OWNER, OTHER)]
    for process in processes:
        process.start()
    try:
        outcomes = [queue.get(timeout=30) for _ in processes]
        assert sorted(outcomes, key=str) == sorted([('ok', 1), ('ok', None)], key=str)
        assert rows(path, 'SELECT version FROM schema_migrations ORDER BY version') == [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,)]
        assert len(rows(path, 'SELECT * FROM tasks')) == 1
        assert len(rows(path, 'SELECT * FROM receipts')) == 1
    finally:
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        queue.close()


async def test_real_busy_is_finite_and_does_not_admit_or_send_receipt(store, envelope_factory):
    from agentweave.errors import StoreUnavailable
    event = envelope_factory()
    blocker = sqlite3.connect(store.path)
    blocker.execute('BEGIN IMMEDIATE')
    start = time.monotonic()
    receipt_was_sent = False
    try:
        with pytest.raises(StoreUnavailable) as error:
            admission = await store.admit(event, 'v1')
            receipt_was_sent = admission is not None
        assert error.value.__cause__.sqlite_errorcode & 0xff == sqlite3.SQLITE_BUSY
    finally:
        blocker.rollback()
        blocker.close()
    assert 14 <= time.monotonic() - start < 18
    assert await store.get_task(event.task_id) is None
    assert receipt_was_sent is False
    assert rows(store.path, 'SELECT * FROM outbox') == []
    assert (await store.admit(event, 'v1')).duplicate is False


def fail_writes(path, input_json, queue):
    import resource
    import signal
    from agentweave.adapters.storage.sqlite import SQLiteStore
    from agentweave._records import StepCommit
    from agentweave.messages import Envelope
    from agentweave.errors import StoreUnavailable
    async def run():
        store = SQLiteStore(path)
        try:
            await store.migrate()
            event = Envelope.model_validate_json(input_json)
            await store.admit(event, 'v1')
            lease = await store.claim(event.task_id, OWNER, 30)
            current = await store.get_task(event.task_id)
            await store.commit_step(lease, StepCommit(step_id='initial', expected_version=current.version, status='running', state={'previous': True}))
            current = await store.get_task(event.task_id)
            from pathlib import Path
            wal_size = Path(str(path) + '-wal').stat().st_size
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (wal_size + 8192, wal_size + 8192))
            fresh = event.model_copy(update={'event_id': OTHER, 'task_id': OTHER, 'payload': {'large': 'x' * 200_000}})
            receipt_was_sent = False
            errors = []
            try:
                receipt_was_sent = (await store.admit(fresh, 'v1')) is not None
            except StoreUnavailable as exc:
                errors.append(exc.__cause__.sqlite_errorcode & 0xff)
            try:
                await store.commit_step(lease, StepCommit(step_id='too-large', expected_version=current.version, status='succeeded', state={'large': 'x' * 200_000}))
            except StoreUnavailable as exc:
                errors.append(exc.__cause__.sqlite_errorcode & 0xff)
            queue.put((errors, receipt_was_sent, await store.get_task(OTHER) is None,
                       (await store.get_task(event.task_id)).status,
                       rows(path, 'SELECT state FROM sessions')))
        except Exception as exc:
            queue.put(('error', repr(exc)))
        finally:
            await store.close()
    asyncio.run(run())


def test_real_disk_write_failure_rolls_back_input_receipt_and_state(tmp_path, envelope_factory):
    from datetime import datetime, timedelta, timezone
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue()
    now = datetime.now(timezone.utc)  # the child runs on the system clock
    event = envelope_factory(deadline=now + timedelta(minutes=5)).model_copy(update={'created_at': now})
    path = tmp_path / 'limited.sqlite3'
    process = ctx.Process(target=fail_writes, args=(str(path), event.model_dump_json(), queue))
    process.start()
    try:
        result = queue.get(timeout=30)
        assert result == ([sqlite3.SQLITE_IOERR, sqlite3.SQLITE_IOERR], False, True, 'running', [('{"previous":true}',)])
        assert len(rows(path, 'SELECT * FROM processed_events')) == 1
        assert len(rows(path, 'SELECT * FROM receipts')) == 1
        assert len(rows(path, 'SELECT * FROM steps')) == 1
    finally:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        queue.close()
