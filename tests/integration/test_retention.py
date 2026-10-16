"""Retention: finished work expires on schedule, unfinished work never, and dedup outlives replay."""
import sqlite3
from datetime import timedelta
import pytest
from agentweave._records import EventRejection, StepCommit

INSTANCE = '00000000-0000-0000-0000-0000000000d1'
DAY = 86400


async def finish(store, envelope, *, state=None, messages=()):
    await store.admit(envelope, 'def-v1')
    lease = await store.claim(envelope.task_id, INSTANCE, 30)
    task = await store.get_task(envelope.task_id)
    await store.commit_step(lease, StepCommit(step_id='done', expected_version=task.version, status='succeeded',
                                              result={'answer': 42}, state=state, messages=list(messages)))
    return envelope


def exchange(text):
    from agentweave.messages import Message
    return [Message(role='user', content=text), Message(role='assistant', content='ok')]


async def test_cleanup_purges_finished_results_after_seven_days_and_keeps_unfinished_work(store, fake_clock,
                                                                                            envelope_factory):
    from agentweave._runtime.maintenance import cleanup
    old = await finish(store, envelope_factory())
    unfinished = envelope_factory()
    await store.admit(unfinished, 'def-v1')
    fake_clock.advance(7.5 * DAY)
    recent = await finish(store, envelope_factory())
    report = await cleanup(store)
    assert report['dry_run'] is True and report['purged_tasks'] == 1
    assert (await store.get_task(old.task_id)).result == {'answer': 42}  # a dry run changes nothing
    report = await cleanup(store, dry_run=False)
    assert report['purged_tasks'] == 1 and report['expired_ids'] == 0
    purged = await store.get_task(old.task_id)
    assert (purged.status, purged.result, purged.input) == ('succeeded', None, None)
    assert (await store.get_task(recent.task_id)).result == {'answer': 42}
    assert (await store.get_task(unfinished.task_id)).status == 'accepted'
    admitted = await store.admit(old, 'def-v1')
    assert admitted.duplicate and admitted.task_id == old.task_id  # still deduplicated, never run again


async def test_old_events_are_refused_once_their_ids_may_be_gone(store, fake_clock, envelope_factory):
    from agentweave._runtime.maintenance import cleanup
    from agentweave.errors import ReplayExpired
    old = await finish(store, envelope_factory())
    never_seen = envelope_factory(recipient='assistant')
    with sqlite3.connect(store.path) as connection:  # the outbox pump delivered every receipt
        connection.execute('UPDATE outbox SET delivered=1')
    fake_clock.advance(9 * DAY)
    report = await cleanup(store, dry_run=False)
    assert report['expired_ids'] == 1 and report['deleted_tasks'] == 1
    assert await store.get_task(old.task_id) is None
    later = {'deadline': fake_clock.now() + timedelta(seconds=60)}
    with pytest.raises(ReplayExpired):
        await store.admit(old.model_copy(update=later), 'def-v1')
    with pytest.raises(ReplayExpired):  # older than the replay window, even if never processed
        await store.admit(never_seen.model_copy(update=later), 'def-v1')
    rejected = await store.admit_with_policy(old.model_copy(update=later), 'def-v1',
                                             agent_id='assistant', max_inflight=8)
    assert rejected == EventRejection(reason='replay_expired', retryable=False)
    fresh = envelope_factory()
    assert not (await store.admit(fresh, 'def-v1')).duplicate


async def test_a_longer_replay_window_extends_the_dedup_horizon(store, fake_clock, envelope_factory):
    from agentweave._runtime.maintenance import cleanup
    from agentweave.config import Retention
    old = await finish(store, envelope_factory())
    store.retention = Retention(replay_window_days=14)
    fake_clock.advance(9 * DAY)
    report = await cleanup(store, dry_run=False)
    assert report['expired_ids'] == 0 and report['purged_tasks'] == 1
    assert (await store.admit(old, 'def-v1')).duplicate


async def test_idle_stateful_history_expires_after_thirty_days_unless_a_task_is_open(store, fake_clock,
                                                                                    envelope_factory):
    from agentweave._runtime.maintenance import cleanup, inspect_state
    idle = await finish(store, envelope_factory(), state={'n': 1}, messages=exchange('first'))
    open_session = envelope_factory()
    await finish(store, open_session.model_copy(update={'event_id': '00000000-0000-4000-8000-00000000aa01',
                                                        'task_id': '00000000-0000-4000-8000-00000000aa02'}),
                 state={'n': 2}, messages=exchange('kept'))
    await store.admit(open_session, 'def-v1')  # still waiting in the same session
    fake_clock.advance(25 * DAY)
    active = await finish(store, envelope_factory(), state={'n': 3}, messages=exchange('recent'))
    fake_clock.advance(6 * DAY)
    report = await cleanup(store, dry_run=False)
    assert report['idle_sessions'] == 1
    assert await store.load_session('assistant', idle.session_id) == ({}, [])
    assert (await store.load_session('assistant', active.session_id))[0] == {'n': 3}
    kept = await inspect_state(store, open_session.session_id)
    assert kept['sessions'] == [{'agent_id': 'assistant', 'state': {'n': 2}, 'messages': 2,
                                 'updated_at': kept['sessions'][0]['updated_at']}]
    assert [t['status'] for t in kept['tasks']] == ['succeeded', 'accepted']


async def test_export_uses_the_backup_api_with_a_live_wal(store, envelope_factory, tmp_path):
    from agentweave._runtime.maintenance import export_state
    from agentweave.errors import OperatorActionRefused
    for _ in range(3):
        await finish(store, envelope_factory())
    wal = store.path.with_name(store.path.name + '-wal')
    assert wal.exists() and wal.stat().st_size > 0  # committed pages still live only in the WAL
    target = tmp_path / 'backup.sqlite3'
    report = await export_state(store, target)
    assert report['tasks'] == 3
    with sqlite3.connect(target) as copy:
        assert copy.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert copy.execute('SELECT COUNT(*) FROM tasks').fetchone()[0] == 3
    with pytest.raises(OperatorActionRefused, match='exists'):
        await export_state(store, target)


async def test_children_of_an_open_root_keep_their_journal(store, fake_clock, envelope_factory):
    """A remote tool call of a task still open must replay from its journal, never run again."""
    from agentweave._runtime.maintenance import cleanup
    parent = envelope_factory()
    await store.admit(parent, 'def-v1')  # still open, e.g. waiting for an operator
    child = await finish(store, envelope_factory(kind='tool_call', recipient='weather', sender='assistant',
                                                 parent_task_id=parent.task_id, session_id=None))
    fake_clock.advance(30 * 86400)
    report = await cleanup(store, dry_run=False)
    assert report['purged_tasks'] == 0 and report['expired_ids'] == 0
    assert (await store.get_task(child.task_id)).result == {'answer': 42}


async def test_an_id_whose_receipt_is_still_undelivered_is_kept(store, fake_clock, envelope_factory):
    from agentweave._runtime.maintenance import cleanup
    old = await finish(store, envelope_factory(sender='client.far', recipient='assistant'))
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE outbox SET delivered=0 WHERE json_extract(envelope,'$.kind')='receipt'")
    fake_clock.advance(9 * DAY)
    report = await cleanup(store, dry_run=False)
    assert report['expired_ids'] == 0
    assert (await store.admit(old, 'def-v1')).duplicate


async def test_an_event_from_the_future_is_refused_and_never_moves_the_dedup_floor(store, fake_clock,
                                                                                  envelope_factory):
    from agentweave.errors import ReplayExpired
    ahead = envelope_factory(created_at=fake_clock.now() + timedelta(days=3))
    with pytest.raises(ReplayExpired, match='future'):
        await store.admit(ahead, 'def-v1')
    rejected = await store.admit_with_policy(ahead, 'def-v1', agent_id='assistant', max_inflight=8)
    assert rejected == EventRejection(reason='created_in_future', retryable=False)


async def test_an_export_is_private_from_its_creation(store, envelope_factory, tmp_path):
    from agentweave._runtime.maintenance import export_state
    await finish(store, envelope_factory())
    target = tmp_path / 'private.sqlite3'
    await export_state(store, target)
    assert target.stat().st_mode & 0o777 == 0o600
