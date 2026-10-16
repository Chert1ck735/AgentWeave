"""event publish/subscribe and agent request over a real dev broker, with stable exit codes."""
import json
import signal
import subprocess
import sys
from pathlib import Path
import pytest
from tests.e2e.test_dev_cli import CREATED, ENV, REPO, cli, ok

SESSION = '6f1c1d2e-0000-4000-8000-000000000007'

APP = '''from agentweave import Agent
listener = Agent('listener')


@listener.on('orders.created')
async def created(ctx, event):
    return {'payload': event, 'key': ctx.event.key, 'source_version': ctx.event.source_version,
            'session_id': ctx.event.session_id}


@listener.on_request
async def answer(ctx, request):
    if request.input == 'fail':
        raise RuntimeError('the handler failed on purpose')
    return {'echo': request.input}
'''


def _cleanup(root):
    from agentweave._infrastructure.local import cleanup_epoch
    from agentweave._infrastructure.processes import ProcessRegistry
    cli(root, 'agent', 'stop', 'listener', timeout=70)
    cli(root, 'infra', 'down', timeout=70)
    registry = ProcessRegistry(root)
    for record in CREATED.pop(root.resolve(), {}).values():
        if registry.running(record):
            registry.terminate(record, grace=2)
        if record.logical_id == '$broker' and Path(record.epoch_dir).exists():
            cleanup_epoch(record, registry.paths)
        registry.remove(record)


@pytest.fixture(scope='module')
def listening(tmp_path_factory):
    root = tmp_path_factory.mktemp('events') / 'event project'
    root.mkdir()
    (root / 'app.py').write_text(APP)
    try:
        ok(root, 'agent', 'start', 'app:listener')
        yield root
    finally:
        _cleanup(root)


def run(root, *args, input=None, timeout=60):
    return subprocess.run([sys.executable, '-m', 'agentweave', '--root', str(root), '--json', *args],
                          input=input, cwd=REPO, env=ENV, capture_output=True, text=True, timeout=timeout)


def lines(result):
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def subscriber(root, *args):
    process = subprocess.Popen([sys.executable, '-m', 'agentweave', '--root', str(root), '--json', 'event',
                                'subscribe', 'orders.created', *args], cwd=REPO, env=ENV, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    first = json.loads(process.stdout.readline())
    assert first['subscribed'] == 'orders.created', first
    return process


def failure(result, code, exit_code):
    assert result.returncode == exit_code, (result.returncode, result.stdout, result.stderr)
    payload = lines(result)[-1]
    assert payload['error_code'] == code, payload
    return payload


def test_publish_input_needs_exactly_one_valid_source_before_any_connection(tmp_path):
    root = tmp_path / 'no infra'
    root.mkdir()
    payload = tmp_path / 'payload.json'
    payload.write_text('{"order": 1}')
    failure(run(root, 'event', 'publish', 'orders.created', '{"order": 1}', '--file', str(payload)), 'invalid_input', 2)
    failure(run(root, 'event', 'publish', 'orders.created', input=''), 'invalid_input', 2)
    failure(run(root, 'event', 'publish', 'orders.created', '{oops'), 'invalid_json', 2)
    failure(run(root, 'event', 'publish', 'orders.created', '{}', '--event-time', 'yesterday'), 'invalid_input', 2)
    failure(run(root, 'event', 'publish', 'Bad Topic!', '{}'), 'invalid_input', 2)
    failure(run(root, 'event', 'publish', 'orders.created', '{}', '--session-id', 'not-a-uuid'), 'invalid_input', 2)
    binary = tmp_path / 'binary.json'
    binary.write_bytes(b'\xff\xfe')
    failure(run(root, 'event', 'publish', 'orders.created', '--file', str(binary)), 'invalid_input', 2)
    failure(run(root, 'event', 'publish', 'orders.created', '1e999'), 'invalid_json', 2)
    failure(run(root, 'event', 'publish', 'orders.created', '-', input='[' * 100000), 'invalid_json', 2)
    failure(run(root, 'event', 'publish', 'orders.created', '{}', '--wait', 'bad name'), 'invalid_input', 2)
    for timeout in ('nan', 'inf', '0', '-1'):
        failure(run(root, 'event', 'publish', 'orders.created', '{}', '--wait', 'listener', '--timeout', timeout),
                'invalid_input', 2)
    failure(run(root, 'agent', 'request', 'listener', '1e999'), 'invalid_json', 2)
    # Valid input, but nothing runs: the CLI reports it instead of starting a broker.
    failure(run(root, 'event', 'publish', 'orders.created', '--file', str(payload)), 'infrastructure_unavailable', 3)
    failure(run(root, 'agent', 'request', 'listener', '"hi"'), 'infrastructure_unavailable', 3)
    assert lines(run(root, 'infra', 'status'))[-1]['running'] is False
    profile = root / '.agentweave' / 'dev'
    profile.mkdir(parents=True)
    (profile / 'config.toml').write_text('profile = "dev"\nunknown = 1\n')
    failure(run(root, 'event', 'publish', 'orders.created', '{}'), 'profile_invalid', 2)


def test_a_payload_over_the_envelope_limit_is_an_input_error(listening):
    failure(run(listening, 'event', 'publish', 'orders.created', '-', input=json.dumps('x' * 1_100_000)),
            'invalid_input', 2)


def test_a_published_event_reaches_subscribers_with_its_metadata(listening):
    watcher = subscriber(listening, '--limit', '1')
    try:
        published = run(listening, 'event', 'publish', 'orders.created', '--key', 'order-7', '--source-version', '3',
                        '--event-time', '2025-01-16T10:00:00Z', '--expires-at', '2099-01-01T00:00:00Z',
                        '--session-id', SESSION, '--wait', 'listener', input='{"order": 7}')
        assert published.returncode == 0, (published.stdout, published.stderr)
        sent, processed = lines(published)
        assert sent['topic'] == 'orders.created' and sent['event_id']
        assert processed['status'] == 'succeeded' and processed['responding_agent'] == 'listener'
        assert processed['output'] == {'payload': {'order': 7}, 'key': 'order-7', 'source_version': 3,
                                       'session_id': SESSION}
        out, err = watcher.communicate(timeout=30)
    finally:
        if watcher.poll() is None:
            watcher.kill()
            watcher.communicate()
    assert watcher.returncode == 0, err
    [seen] = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert (seen['event_id'], seen['payload'], seen['key'], seen['source_version']) == (
        sent['event_id'], {'order': 7}, 'order-7', 3)
    assert seen['event_time'].startswith('2025-01-16T10:00:00')


def test_a_request_reports_acceptance_then_the_result_and_failures_exit_4(listening):
    answered = run(listening, 'agent', 'request', 'listener', '"hi"')
    assert answered.returncode == 0, (answered.stdout, answered.stderr)
    accepted, result = lines(answered)
    assert accepted['accepted'] is True and accepted['task_id'] == result['task_id']
    assert (result['status'], result['output']) == ('succeeded', {'echo': 'hi'})
    failed = run(listening, 'agent', 'request', 'listener', '"fail"')
    assert failed.returncode == 4, (failed.stdout, failed.stderr)
    assert lines(failed)[-1]['status'] == 'failed'
    detached = run(listening, 'agent', 'request', 'listener', '"later"', '--no-wait')
    assert detached.returncode == 0 and [set(x) for x in lines(detached)] == [{'task_id', 'accepted'}]


@pytest.mark.parametrize('stop', [signal.SIGINT, signal.SIGTERM])
def test_subscribe_ends_cleanly_on_interrupt(listening, stop):
    watcher = subscriber(listening)
    watcher.send_signal(stop)
    out, err = watcher.communicate(timeout=30)
    assert watcher.returncode == 0, err
    assert 'Traceback' not in err and out.strip() == ''
