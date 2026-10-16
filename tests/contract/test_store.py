"""Real persistence contracts; each assertion catches a missing durable effect or fence."""
import sqlite3
import uuid
import pytest
from agentweave._records import Admission, StepCommit
from agentweave.messages import Message

OWNER = '00000000-0000-0000-0000-000000000001'
OTHER = '00000000-0000-0000-0000-000000000002'


def rows(path, sql, args=()):
    with sqlite3.connect(path) as connection:
        return connection.execute(sql, args).fetchall()


async def test_duplicate_admit_returns_original_durable_receipt(store, envelope_factory):
    envelope = envelope_factory()
    first = await store.admit(envelope, 'v1')
    assert isinstance(first, Admission)
    duplicate = await store.admit(envelope, 'v2')
    assert first.duplicate is False
    assert duplicate.duplicate is True
    assert duplicate.receipt == first.receipt
    assert first.receipt.correlation_id == envelope.event_id
    task = await store.get_task(envelope.task_id)
    assert task.input == {'question': 'hello'}
    assert task.definition_hash == 'v1'
    assert task.status == 'accepted'
    assert len(rows(store.path, 'SELECT * FROM tasks')) == 1
    assert len(rows(store.path, 'SELECT * FROM outbox')) == 1


async def test_submit_is_durable_but_does_not_admit(store, envelope_factory, store_factory):
    envelope = envelope_factory()
    submission = await store.submit(envelope)
    assert submission.task_id == envelope.task_id
    assert await store.get_task(envelope.task_id) is None
    await store.close()
    reopened = await store_factory(store.path)
    pending = await reopened.lease_outbox(OWNER, 5)
    assert [row.envelope.event_id for row in pending] == [envelope.event_id]
    assert pending[0].targets == ['assistant']
    assert len(rows(store.path, 'SELECT * FROM submissions')) == 1
    await reopened.admit(envelope, 'v1')
    assert (await reopened.get_task(envelope.task_id)).status == 'accepted'


async def test_commit_atomically_persists_state_result_messages_and_outbox(store, accepted_task, envelope_factory, store_factory):
    lease = await store.claim(accepted_task.task_id, OWNER, 30)
    task = await store.get_task(accepted_task.task_id)
    outgoing = envelope_factory(kind='event', recipient=None, topic='updates', payload={'done': True})
    result = await store.commit_step(lease, StepCommit(step_id='final', expected_version=task.version,
        status='succeeded', result=None, state={'count': 2}, messages=[Message(role='assistant', content='done')], outgoing=[outgoing]))
    assert result.status == 'succeeded'
    assert result.result is None
    assert result.version == task.version + 1
    await store.close()
    reopened = await store_factory(store.path)
    assert (await reopened.get_task(task.task_id)).status == 'succeeded'
    assert rows(store.path, 'SELECT state FROM sessions WHERE agent_id=? AND session_id=?', ('assistant', task.session_id)) == [('{"count":2}',)]
    assert len(rows(store.path, 'SELECT * FROM messages')) == 1
    assert len(rows(store.path, 'SELECT * FROM steps')) == 1
    assert outgoing.event_id in [row.envelope.event_id for row in await reopened.lease_outbox(OWNER, 10)]


async def test_old_owner_cannot_commit_or_renew_after_expiry(store, accepted_task, fake_clock, completed_step):
    from agentweave.errors import LeaseLost
    old = await store.claim(accepted_task.task_id, OWNER, 30)
    fake_clock.advance(31)
    new = await store.claim(accepted_task.task_id, OTHER, 30)
    assert new.token > old.token
    with pytest.raises(LeaseLost):
        await store.commit_step(old, completed_step)
    with pytest.raises(LeaseLost):
        await store.renew(old)
    assert (await store.get_task(old.task_id)).status == 'running'


async def test_version_conflict_and_failed_outgoing_roll_back_every_effect(store, accepted_task, envelope_factory):
    from agentweave.errors import VersionConflict, StoreConflict
    lease = await store.claim(accepted_task.task_id, OWNER, 30)
    with pytest.raises(VersionConflict):
        await store.commit_step(lease, StepCommit(step_id='bad-version', expected_version=0, status='succeeded'))
    duplicate = envelope_factory()
    await store.submit(duplicate)
    current = await store.get_task(lease.task_id)
    with pytest.raises(StoreConflict):
        await store.commit_step(lease, StepCommit(step_id='bad-outbox', expected_version=current.version,
            status='succeeded', state={'changed': True}, messages=[Message(role='assistant', content='partial')], outgoing=[duplicate]))
    assert await store.get_task(lease.task_id) == current
    assert rows(store.path, 'SELECT * FROM steps') == []
    assert rows(store.path, 'SELECT * FROM messages') == []
    assert rows(store.path, 'SELECT * FROM sessions') == []


async def test_renew_extends_original_ttl_without_changing_fence(store, accepted_task, fake_clock):
    lease = await store.claim(accepted_task.task_id, OWNER, 30)
    fake_clock.advance(10)
    renewed = await store.renew(lease)
    assert renewed.token == lease.token
    assert (renewed.expires_at - lease.expires_at).total_seconds() == 10
    fake_clock.advance(21)
    assert await store.claim(lease.task_id, OTHER, 30) is None


@pytest.mark.parametrize('status', ['succeeded', 'failed', 'cancelled', 'expired', 'outcome_unknown'])
async def test_non_replayable_status_cannot_be_claimed(store, accepted_task, fake_clock, status):
    lease = await store.claim(accepted_task.task_id, OWNER, 30)
    current = await store.get_task(lease.task_id)
    await store.commit_step(lease, StepCommit(step_id='pause', expected_version=current.version, status=status))
    fake_clock.advance(31)
    assert await store.claim(lease.task_id, OTHER, 30) is None
    recovered = await store.list_recoverable('assistant')
    assert lease.task_id not in [task.task_id for task in recovered]


async def test_state_none_preserves_but_empty_mapping_clears(store, accepted_task):
    lease = await store.claim(accepted_task.task_id, OWNER, 30)
    for index, state in enumerate([{'count': 1}, None, {}]):
        task = await store.get_task(lease.task_id)
        await store.commit_step(lease, StepCommit(step_id=str(index), expected_version=task.version, status='running', state=state))
        expected = '{"count":1}' if index < 2 else '{}'
        assert rows(store.path, 'SELECT state FROM sessions') == [(expected,)]


async def test_outbox_ack_keeps_request_until_receipt_and_reclaims_after_timeout(store, envelope_factory, fake_clock):
    envelope = envelope_factory()
    await store.submit(envelope)
    first = (await store.lease_outbox(OWNER, 10))[0]
    assert first.lease_token is not None
    assert await store.lease_outbox(OTHER, 10) == []
    await store.record_broker_ack(envelope.event_id, lease_token=first.lease_token, expected_version=first.version)
    fake_clock.advance(31)
    second = (await store.lease_outbox(OTHER, 10))[0]
    assert second.lease_token != first.lease_token
    assert second.version > first.version
    assert second.published_at is not None
    await store.confirm_receipt(envelope.event_id, 'assistant')
    await store.confirm_receipt(envelope.event_id, 'assistant')
    fake_clock.advance(31)
    assert await store.lease_outbox(OWNER, 10) == []


async def test_broadcast_and_receipt_complete_after_broker_ack(store, envelope_factory, fake_clock):
    envelope = envelope_factory(recipient=None, topic='news')
    await store.submit(envelope)
    await store.record_broker_ack(envelope.event_id)
    admitted = await store.admit(envelope_factory(), 'v1')
    await store.record_broker_ack(admitted.receipt.event_id)
    fake_clock.advance(31)
    assert await store.lease_outbox(OWNER, 10) == []


async def test_admit_task_collision_rolls_back_processed_event_and_receipt(store, envelope_factory):
    from agentweave.errors import StoreConflict
    envelope = envelope_factory()
    await store.admit(envelope, 'v1')
    with pytest.raises(StoreConflict):
        await store.admit(envelope_factory(task_id=envelope.task_id), 'v1')
    assert len(rows(store.path, 'SELECT * FROM processed_events')) == 1
    assert len(rows(store.path, 'SELECT * FROM receipts')) == 1
    assert len(rows(store.path, 'SELECT * FROM outbox')) == 1


async def test_topic_fanout_deduplicates_per_agent_and_preserves_original_envelope(store, envelope_factory, store_factory):
    from uuid import UUID, uuid5
    event = envelope_factory(recipient=None, topic='news', task_id=None)
    first = await store.admit(event, 'v1', agent_id='first')
    second = await store.admit(event, 'v2', agent_id='second')
    assert first.task_id == str(uuid5(UUID(event.event_id), 'task:first'))
    assert second.task_id == str(uuid5(UUID(event.event_id), 'task:second'))
    assert first.receipt.event_id != second.receipt.event_id
    assert (await store.get_task(first.task_id)).agent_id == 'first'
    assert (await store.get_task(second.task_id)).agent_id == 'second'
    assert rows(store.path, 'SELECT input_envelope FROM tasks') == [(event.model_dump_json(),), (event.model_dump_json(),)]
    await store.close()
    reopened = await store_factory(store.path)
    assert (await reopened.admit(event, 'changed', agent_id='first')).duplicate is True
    assert len(rows(store.path, 'SELECT * FROM tasks')) == 2


async def test_admission_requires_unambiguous_target_and_known_parent(store, envelope_factory):
    from agentweave.errors import StoreConflict
    with pytest.raises(ValueError):
        await store.admit(envelope_factory(recipient=None, topic='news'), 'v1')
    with pytest.raises(ValueError):
        await store.admit(envelope_factory(), 'v1', agent_id='somebody-else')
    with pytest.raises(StoreConflict):
        await store.admit(envelope_factory(parent_task_id=str(uuid.uuid4())), 'v1')
    assert rows(store.path, 'SELECT * FROM tasks') == []
    parent = envelope_factory()
    await store.admit(parent, 'v1')
    child = envelope_factory(parent_task_id=parent.task_id)
    await store.admit(child, 'v1')
    assert (await store.get_task(child.task_id)).root_task_id == parent.task_id


async def test_expired_task_and_invalid_lease_arguments_have_no_owner(store, envelope_factory, fake_clock):
    event = envelope_factory()
    await store.admit(event, 'v1')
    with pytest.raises(ValueError):
        await store.claim(event.task_id, OWNER, 0)
    with pytest.raises(ValueError):
        await store.claim(event.task_id, 'not-a-uuid', 30)
    fake_clock.advance(121)
    assert await store.claim(event.task_id, OWNER, 30) is None
    assert await store.lease_outbox(OWNER, 10) == []
    assert (await store.get_task(event.task_id)).status == 'accepted'


async def test_session_state_without_session_is_rejected_before_commit(store, envelope_factory):
    event = envelope_factory(session_id=None)
    await store.admit(event, 'v1')
    lease = await store.claim(event.task_id, OWNER, 30)
    with pytest.raises(ValueError):
        await store.commit_step(lease, StepCommit(step_id='bad', expected_version=1, status='succeeded', state={}))
    assert (await store.get_task(event.task_id)).status == 'running'
    assert rows(store.path, 'SELECT * FROM steps') == []


async def test_store_configuration_and_schema_are_durable(store):
    from agentweave.ports import ExecutionStore
    assert isinstance(store, ExecutionStore)
    async with store._connection.execute('PRAGMA journal_mode') as cursor:
        assert (await cursor.fetchone())[0] == 'wal'
    for pragma, expected in [('foreign_keys', 1), ('synchronous', 2), ('busy_timeout', 5000)]:
        async with store._connection.execute('PRAGMA ' + pragma) as cursor:
            assert (await cursor.fetchone())[0] == expected
    names = {row[0] for row in rows(store.path, "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {'definitions', 'tasks', 'steps', 'tool_calls', 'sessions', 'messages', 'task_events',
            'processed_events', 'outbox', 'receipts', 'submissions', 'participants', 'schema_migrations'} <= names


@pytest.mark.parametrize('status', ['waiting_peer', 'waiting_tool'])
async def test_waiting_tasks_are_recoverable_but_not_runnable(store, accepted_task, status):
    lease = await store.claim(accepted_task.task_id, OWNER, 30)
    await store.commit_step(lease, StepCommit(step_id='wait', expected_version=1, status=status, waiting_for={'event_id': OTHER}))
    assert [task.task_id for task in await store.list_recoverable('assistant')] == [lease.task_id]
    assert await store.claim(lease.task_id, OTHER, 30) is None
    assert rows(store.path, 'SELECT waiting_for FROM tasks') == [('{"event_id":"00000000-0000-0000-0000-000000000002"}',)]


async def test_recovery_returns_original_envelope_and_stable_receipt_identity(store, envelope_factory):
    from uuid import UUID, uuid5
    event = envelope_factory()
    admitted = await store.admit(event, 'v1')
    assert admitted.receipt.event_id == str(uuid5(UUID(event.event_id), 'receipt:assistant'))
    assert await store.get_input_envelope(admitted.task_id) == event
    assert await store.get_input_envelope(OTHER) is None


async def test_duplicate_input_reactivates_lost_receipt_without_new_task(store, envelope_factory, fake_clock):
    event = envelope_factory()
    admitted = await store.admit(event, 'v1')
    receipt = (await store.lease_outbox(OWNER, 5))[0]
    await store.record_broker_ack(receipt.envelope.event_id, lease_token=receipt.lease_token, expected_version=receipt.version)
    fake_clock.advance(31)
    assert await store.lease_outbox(OTHER, 5) == []
    duplicate = await store.admit(event, 'v2')
    assert duplicate.receipt == admitted.receipt
    ready = await store.lease_outbox(OTHER, 5)
    assert [row.envelope.event_id for row in ready] == [receipt.envelope.event_id]
    assert ready[0].published_at is None
    assert ready[0].version > receipt.version
    assert len(rows(store.path, 'SELECT * FROM tasks')) == 1


async def test_duplicate_input_preserves_active_receipt_publisher_ownership(store, envelope_factory, fake_clock):
    event = envelope_factory()
    await store.admit(event, 'v1')
    receipt = (await store.lease_outbox(OWNER, 5))[0]
    ownership = rows(store.path, 'SELECT lease_token,lease_instance_id,lease_expires_at FROM outbox')
    await store.admit(event, 'v1')
    assert rows(store.path, 'SELECT lease_token,lease_instance_id,lease_expires_at FROM outbox') == ownership
    assert await store.lease_outbox(OTHER, 5) == []
    fake_clock.advance(31)
    reclaimed = (await store.lease_outbox(OTHER, 5))[0]
    assert reclaimed.lease_token != receipt.lease_token


async def test_duplicate_input_does_not_change_completed_task(store, envelope_factory):
    event = envelope_factory()
    await store.admit(event, 'v1')
    lease = await store.claim(event.task_id, OWNER, 30)
    completed = await store.commit_step(lease, StepCommit(step_id='done', expected_version=1, status='succeeded', result=42))
    await store.admit(event, 'v2')
    assert await store.get_task(event.task_id) == completed


async def test_ack_and_receipt_reject_unknown_event_or_recipient(store, envelope_factory):
    from agentweave.errors import StoreConflict
    event = envelope_factory()
    await store.submit(event)
    with pytest.raises(StoreConflict):
        await store.record_broker_ack(OTHER)
    with pytest.raises(StoreConflict):
        await store.confirm_receipt(event.event_id, 'wrong-recipient')
    with pytest.raises(StoreConflict):
        await store.confirm_receipt(OTHER, 'assistant')
    assert rows(store.path, 'SELECT received_targets,published_at FROM outbox') == [('[]', None)]


async def test_recovery_excludes_current_owner_and_includes_expired_running(store, accepted_task, fake_clock):
    assert [row.task_id for row in await store.list_recoverable('assistant')] == [accepted_task.task_id]
    lease = await store.claim(accepted_task.task_id, OWNER, 30)
    assert await store.list_recoverable('assistant') == []
    fake_clock.advance(30)
    assert [row.task_id for row in await store.list_recoverable('assistant')] == [lease.task_id]


async def test_unscoped_ack_does_not_release_active_publisher(store, envelope_factory):
    event = envelope_factory()
    await store.submit(event)
    await store.lease_outbox(OWNER, 5)
    before = rows(store.path, 'SELECT lease_token,lease_instance_id,lease_expires_at,version,published_at,retry_at FROM outbox')
    result = await store.record_broker_ack(event.event_id)
    assert rows(store.path, 'SELECT lease_token,lease_instance_id,lease_expires_at,version,published_at,retry_at FROM outbox') == before
    assert result is False


async def test_old_ack_cannot_release_new_outbox_owner(store, envelope_factory, fake_clock):
    event = envelope_factory()
    await store.submit(event)
    old = (await store.lease_outbox(OWNER, 5))[0]
    fake_clock.advance(31)
    new = (await store.lease_outbox(OTHER, 5))[0]
    before = rows(store.path, 'SELECT lease_token,lease_instance_id,lease_expires_at,version,published_at FROM outbox')
    assert await store.record_broker_ack(event.event_id, lease_token=old.lease_token, expected_version=old.version) is False
    assert rows(store.path, 'SELECT lease_token,lease_instance_id,lease_expires_at,version,published_at FROM outbox') == before
    assert new.lease_token != old.lease_token
    assert await store.record_broker_ack(event.event_id, lease_token=new.lease_token, expected_version=new.version) is True
    fake_clock.advance(1)
    assert len(await store.lease_outbox(OWNER, 5)) == 1


async def test_scoped_ack_does_not_suppress_receipt_reactivated_during_publication(store, envelope_factory):
    event = envelope_factory()
    await store.admit(event, 'v1')
    publishing = (await store.lease_outbox(OWNER, 5))[0]
    await store.admit(event, 'v1')
    before = rows(store.path, 'SELECT published_at,retry_at FROM outbox')
    assert await store.record_broker_ack(publishing.envelope.event_id,
        lease_token=publishing.lease_token, expected_version=publishing.version) is False
    assert rows(store.path, 'SELECT published_at,retry_at FROM outbox') == before
    assert rows(store.path, 'SELECT lease_instance_id,lease_expires_at FROM outbox') == [(None, None)]
    released = rows(store.path, 'SELECT version,published_at,retry_at FROM outbox')
    assert await store.record_broker_ack(publishing.envelope.event_id,
        lease_token=publishing.lease_token, expected_version=publishing.version) is False
    assert rows(store.path, 'SELECT version,published_at,retry_at FROM outbox') == released
    republishing = (await store.lease_outbox(OTHER, 5))[0]
    assert republishing.lease_token != publishing.lease_token
    assert republishing.published_at is None
    assert await store.record_broker_ack(republishing.envelope.event_id,
        lease_token=republishing.lease_token, expected_version=republishing.version) is True
    assert await store.lease_outbox(OWNER, 5) == []


async def test_unscoped_ack_rejects_expired_owner_and_preserves_reactivated_receipt(store, envelope_factory, fake_clock):
    event = envelope_factory()
    await store.admit(event, 'v1')
    publisher = (await store.lease_outbox(OWNER, 5))[0]
    await store.admit(event, 'v1')
    fake_clock.advance(31)
    before = rows(store.path, 'SELECT lease_token,lease_instance_id,lease_expires_at,version,published_at,retry_at FROM outbox')
    result = await store.record_broker_ack(publisher.envelope.event_id)
    assert result is False
    assert rows(store.path, 'SELECT lease_token,lease_instance_id,lease_expires_at,version,published_at,retry_at FROM outbox') == before
    pending = await store.lease_outbox(OTHER, 5)
    assert [row.envelope.event_id for row in pending] == [publisher.envelope.event_id]
    assert pending[0].published_at is None
    assert pending[0].lease_token != publisher.lease_token


async def test_task4_scoped_retry_schedule_survives_publish_and_failure(store, envelope_factory, fake_clock):
    from datetime import timedelta
    event = envelope_factory()
    await store.submit(event)
    first = (await store.lease_outbox(OWNER, 1))[0]
    due = fake_clock.now() + timedelta(seconds=4)
    assert await store.record_broker_ack(event.event_id, lease_token=first.lease_token,
                                        expected_version=first.version, retry_at=due)
    fake_clock.advance(3)
    assert await store.lease_outbox(OTHER, 1) == []
    fake_clock.advance(1)
    second = (await store.lease_outbox(OTHER, 1))[0]
    assert second.retry_at == due
    due = fake_clock.now() + timedelta(seconds=5)
    assert await store.reschedule_outbox(event.event_id, due, lease_token=second.lease_token,
                                         expected_version=second.version)
    assert await store.lease_outbox(OWNER, 1) == []
    fake_clock.advance(5)
    third = (await store.lease_outbox(OWNER, 1))[0]
    assert third.retry_at == due
    assert third.published_at == first.envelope.created_at


async def test_task4_stale_reschedule_preserves_reactivated_receipt(store, envelope_factory, fake_clock):
    from datetime import timedelta
    event = envelope_factory()
    admission = await store.admit(event, 'v1')
    first = (await store.lease_outbox(OWNER, 1))[0]
    await store.admit(event, 'v1')
    before = rows(store.path, 'SELECT published_at,retry_at,version FROM outbox')
    assert not await store.reschedule_outbox(admission.receipt.event_id, fake_clock.now() + timedelta(seconds=5),
        lease_token=first.lease_token, expected_version=first.version)
    after = rows(store.path, 'SELECT published_at,retry_at,version FROM outbox')
    assert after[0][:2] == before[0][:2]
    second = (await store.lease_outbox(OTHER, 1))[0]
    unchanged = rows(store.path, 'SELECT * FROM outbox')
    assert not await store.reschedule_outbox(admission.receipt.event_id, fake_clock.now() + timedelta(seconds=5),
        lease_token=first.lease_token, expected_version=first.version)
    assert rows(store.path, 'SELECT * FROM outbox') == unchanged
    assert second.lease_token != first.lease_token


@pytest.mark.parametrize('operation', ['ack', 'failure'])
async def test_task4_retry_schedule_rejects_naive_time_without_mutation(store, envelope_factory, operation):
    from datetime import datetime
    event = envelope_factory()
    await store.submit(event)
    row = (await store.lease_outbox(OWNER, 1))[0]
    before = rows(store.path, 'SELECT * FROM outbox')
    with pytest.raises(ValueError):
        if operation == 'ack':
            await store.record_broker_ack(event.event_id, retry_at=datetime(2025, 1, 1),
                lease_token=row.lease_token, expected_version=row.version)
        else:
            await store.reschedule_outbox(event.event_id, datetime(2025, 1, 1),
                lease_token=row.lease_token, expected_version=row.version)
    assert rows(store.path, 'SELECT * FROM outbox') == before


@pytest.mark.parametrize('column, value', [('retry_at', '0000-bad'), ('envelope', '{"broken": true}')])
async def test_one_malformed_outbox_row_is_quarantined_not_fatal(store, envelope_factory, column, value):
    broken, healthy = envelope_factory(), envelope_factory()
    await store.submit(broken)
    await store.submit(healthy)
    with sqlite3.connect(store.path) as connection:
        connection.execute(f'UPDATE outbox SET {column}=? WHERE event_id=?', (value, broken.event_id))
    leased = await store.lease_outbox(OWNER, 10)
    assert [row.envelope.event_id for row in leased] == [healthy.event_id]
    await store.record_broker_ack(healthy.event_id, lease_token=leased[0].lease_token,
                                  expected_version=leased[0].version)
    assert await store.lease_outbox(OTHER, 10) == []  # the quarantined row is never retried
    assert rows(store.path, 'SELECT delivered FROM outbox WHERE event_id=?', (broken.event_id,)) == [(0,)]
