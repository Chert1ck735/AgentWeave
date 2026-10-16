"""task and state commands work on the profile journal alone, with versions and stable exit codes."""
import asyncio
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from uuid import uuid4
import pytest
from tests.e2e.test_dev_cli import ENV, REPO

INSTANCE = '00000000-0000-0000-0000-0000000000e2'


def run(root, *args):
    return subprocess.run([sys.executable, '-m', 'agentweave', '--root', str(root), '--json', *args],
                          cwd=REPO, env=ENV, capture_output=True, text=True, timeout=60)


def ok(root, *args):
    result = run(root, *args)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return json.loads(result.stdout.splitlines()[-1])


def refused(root, code, *args, exit_code=2):
    result = run(root, *args)
    assert result.returncode == exit_code, (result.stdout, result.stderr)
    assert json.loads(result.stdout.splitlines()[-1])['error_code'] == code
    return result


async def _unknown_handler(path):
    """A non-retry-safe handler whose worker died mid-run, journaled as the runtime would."""
    from agentweave.adapters.storage.sqlite import SQLiteStore
    from agentweave.messages import Envelope
    store = SQLiteStore(path)
    await store.migrate()
    try:
        now = datetime.now(timezone.utc)
        event = Envelope(event_id=str(uuid4()), network_id='00000000-0000-0000-0000-000000000010', kind='request',
                         sender='client.one', recipient='counter', topic=None, task_id=str(uuid4()),
                         parent_task_id=None, correlation_id=None, session_id=str(uuid4()), created_at=now,
                         deadline=now + timedelta(minutes=10), traceparent=None, payload={'add': 3})
        await store.admit(event, 'def-v1')
        lease = await store.claim_runtime(event.task_id, INSTANCE, 0.2, concurrency=8)
        assert await store.start_handler(lease, retry_safe=False)
        await asyncio.sleep(0.3)  # the worker died; its lease ran out
        lease = await store.claim_runtime(event.task_id, INSTANCE, 5, concurrency=8)
        assert not await store.start_handler(lease, retry_safe=False)
        return event
    finally:
        await store.close()


@pytest.fixture
def journal_project(tmp_path):
    from agentweave.config import ProfileConfig, write_config
    root = tmp_path / 'operator project'
    journal = root / '.agentweave' / 'dev' / 'state.sqlite3'
    journal.parent.mkdir(parents=True)
    write_config(ProfileConfig(profile='dev', project_id=str(uuid4()), network_id=str(uuid4()), root=root,
                               database_url='sqlite:///' + str(journal)), journal.parent / 'config.toml')
    return root, journal


def test_an_operator_resolves_an_unknown_handler_with_its_version(journal_project):
    root, journal = journal_project
    event = asyncio.run(_unknown_handler(journal))
    status = ok(root, 'task', 'status', event.task_id)
    assert status['status'] == 'outcome_unknown'
    assert [(s['step_id'], s['kind'], s['status']) for s in status['steps']] == [
        ('handler', 'python_handler', 'outcome_unknown')]
    version = status['version']
    refused(root, 'version_conflict', 'task', 'resolve', event.task_id, '--step', 'handler',
            '--expected-version', str(version - 1), '--result', '{"count": 3}')
    refused(root, 'invalid_json', 'task', 'resolve', event.task_id, '--step', 'handler',
            '--expected-version', str(version), '--result', '{oops')
    resolved = ok(root, 'task', 'resolve', event.task_id, '--step', 'handler', '--expected-version', str(version),
                  '--result', '{"count": 3}', '--state', '{"count": 3}')
    assert (resolved['status'], resolved['version']) == ('succeeded', version + 1)
    result = ok(root, 'task', 'result', event.task_id)
    assert (result['status'], result['output'], result['terminal']) == ('succeeded', {'count': 3}, True)
    refused(root, 'version_conflict', 'task', 'fail', event.task_id, '--expected-version', str(version + 1),
            '--reason', 'too late')
    inspected = ok(root, 'state', 'inspect', event.session_id)
    assert inspected['sessions'][0]['state'] == {'count': 3}


def test_a_failed_task_result_exits_4_and_unknown_ids_are_input_errors(journal_project):
    root, journal = journal_project
    event = asyncio.run(_unknown_handler(journal))
    version = ok(root, 'task', 'status', event.task_id)['version']
    refused(root, 'action_refused', 'task', 'retry', event.task_id, '--step', 'handler',
            '--expected-version', str(version))  # without --allow-unknown
    ok(root, 'task', 'fail', event.task_id, '--expected-version', str(version), '--reason', 'refunded by hand')
    result = run(root, 'task', 'result', event.task_id)
    assert result.returncode == 4 and json.loads(result.stdout)['error']['type'] == 'OperatorFailed'
    refused(root, 'task_not_found', 'task', 'status', str(uuid4()))
    refused(root, 'invalid_input', 'task', 'status', 'not-a-uuid')
    refused(root, 'usage_error', 'task', 'fail', event.task_id, '--reason', 'no version given')


def test_state_cleanup_is_a_dry_run_by_default_and_export_never_overwrites(journal_project, tmp_path):
    root, journal = journal_project
    asyncio.run(_unknown_handler(journal))
    report = ok(root, 'state', 'cleanup')
    assert report['dry_run'] is True and report['purged_tasks'] == 0
    assert ok(root, 'state', 'cleanup', '--apply')['dry_run'] is False
    target = tmp_path / 'backup.sqlite3'
    assert ok(root, 'state', 'export', str(target))['tasks'] == 1
    refused(root, 'action_refused', 'state', 'export', str(target))


def test_operator_commands_need_an_existing_journal(tmp_path):
    refused(tmp_path, 'journal_unavailable', 'task', 'status', str(uuid4()), exit_code=3)
    refused(tmp_path, 'journal_unavailable', 'state', 'cleanup', exit_code=3)
    assert not (tmp_path / '.agentweave').exists()
