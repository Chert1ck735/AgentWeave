"""Replicas share one logical execution limit, never run one task twice, and survive store or broker loss."""
import asyncio
import sqlite3
from datetime import timedelta
from uuid import uuid4
import pytest
from tests.fixtures.crash import effects_of
from tests.integration.test_crash_matrix import CrashHarness, crash_harness  # noqa: F401
from tests.integration.test_python_runtime import eventually


def max_overlap(effects):
    """The most handler runs that were in progress at the same moment."""
    edges = sorted([(row['at'], 1) for row in effects if row['kind'] == 'start'] +
                   [(row['at'], -1) for row in effects if row['kind'] == 'end'])
    running = peak = 0
    for _, step in edges:
        running += step
        peak = max(peak, running)
    return peak


def journal(store, sql, *values):
    with sqlite3.connect(store.path) as connection:
        return connection.execute(sql, values).fetchall()


@pytest.mark.parametrize('concurrency', [1, 2])
async def test_two_replicas_run_each_task_once_within_the_shared_limit(crash_harness, concurrency):
    h = crash_harness
    for _ in range(2):
        await h.start('slow', concurrency=concurrency, seconds=0.2)
    task_ids = [await h.submit('slow', n) for n in range(12)]
    results = await asyncio.gather(*(h.result(task_id) for task_id in task_ids))
    assert all(result.status == 'succeeded' for result in results)
    starts = effects_of(h.effects, 'start')
    assert sorted(row['task'] for row in starts) == sorted(task_ids)  # each task ran once, on one replica
    assert max_overlap(effects_of(h.effects)) <= concurrency  # the limit is shared, not per process


async def test_a_stale_lease_cannot_commit_after_another_replica_took_over(crash_harness, tmp_path):
    h = crash_harness
    release = tmp_path / 'release'
    first = await h.start('blocking', release=str(release))
    task_id = await h.submit('blocking')

    async def started():
        return len(effects_of(h.effects, 'start')) == 1
    await eventually(started)
    h.advance()  # the first owner's lease ran out while its handler still runs
    await h.start('blocking', release=str(release))

    async def taken_over():
        return len(effects_of(h.effects, 'start')) == 2
    await eventually(taken_over)
    release.write_text('go')
    result = await h.result(task_id)
    assert result.status == 'succeeded'
    assert result.output != first.pid  # the late first owner did not commit
    assert journal(h.store, "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='succeeded'", task_id) == [(1,)]


async def test_a_lost_journal_stops_admission_and_effects_until_it_returns(crash_harness, tmp_path):
    from agentweave.messages import event_task_id
    h = crash_harness
    outage = tmp_path / 'db_down'
    worker = await h.start('listener', outage=str(outage))
    outage.write_text('down')
    event_id = await h.client.publish('work.item', {'n': 1})
    await asyncio.sleep(3)
    assert effects_of(h.effects, 'listener') == []  # nothing runs without a journal
    assert await h.store.get_task(event_task_id(event_id, 'listener')) is None  # and nothing was admitted
    await asyncio.to_thread(worker.join, 60)  # cleanup has its own 30-second budget
    assert not worker.is_alive()  # storage loss stops the worker visibly instead of acting blind
    stopped = await asyncio.to_thread(h.ready.get, True, 5)
    assert stopped[0] == 'error' and 'StoreUnavailable' in stopped[2], stopped
    outage.unlink()
    h.advance()
    await h.start('listener', outage=str(outage))  # the unacknowledged event is delivered again

    async def handled():  # the effect lands before the handler returns; success commits after it
        task = await h.store.get_task(event_task_id(event_id, 'listener'))
        return task is not None and task.status == 'succeeded'
    await eventually(handled, timeout=30)
    assert len(effects_of(h.effects, 'listener')) == 1  # exactly once, after the journal returned


async def test_a_lost_broker_keeps_the_outbox_and_the_journal(crash_harness):
    from agentweave.messages import Envelope
    h = crash_harness
    worker = await h.start('analyst')
    h.broker.stop()
    now = h.clock.now()
    envelope = Envelope(event_id=str(uuid4()), network_id=h.config.network_id, kind='request', sender=h.client.sender,
                        recipient='analyst', topic=None, task_id=str(uuid4()), parent_task_id=None, correlation_id=None,
                        session_id=str(uuid4()), created_at=now, deadline=now + timedelta(seconds=300),
                        traceparent=None, payload='while the broker is down')
    await h.store.submit(envelope)  # durable even though nothing can be published now
    assert journal(h.store, 'SELECT published_at FROM outbox WHERE event_id=?', envelope.event_id) == [(None,)]
    await h.kill(worker)
    h.broker.restart()
    await h.connect()
    h.advance()
    await h.start('analyst')
    result = await h.result(envelope.task_id)
    assert result.status == 'succeeded' and len(effects_of(h.effects, 'analyst')) == 1
