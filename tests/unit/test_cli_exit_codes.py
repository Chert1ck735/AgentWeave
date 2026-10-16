"""SDK failures inside CLI commands keep their meaning: infrastructure is 3, bad input is 2."""
import json
import pytest


def _validation_error():
    from agentweave.messages import TaskSubmission
    try:
        TaskSubmission(task_id='not a uuid')
    except ValueError as exc:
        return exc
    raise AssertionError('expected a validation error')


def _sdk_error(name, message):
    from agentweave import errors
    return getattr(errors, name)(message)


@pytest.mark.parametrize('make, code, exit_code', [
    (lambda: _sdk_error('BrokerUnavailable', 'broker down'), 'broker_unavailable', 3),
    (lambda: _sdk_error('StoreUnavailable', 'disk'), 'store_unavailable', 3),
    (lambda: _sdk_error('EnvelopeTooLarge', 'big'), 'invalid_input', 2),
    (_validation_error, 'invalid_input', 2)])
def test_sdk_failures_map_to_stable_exit_codes(monkeypatch, capsys, tmp_path, make, code, exit_code):
    from agentweave.cli import app, event

    def failing(root, profile, args, emit):
        raise make()
    monkeypatch.setattr(event, 'publish', failing)
    assert app.main(['--root', str(tmp_path), '--json', 'event', 'publish', 'orders.created', '{}']) == exit_code
    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert payload['error_code'] == code
    assert 'not a uuid' not in json.dumps(payload)  # the input value is never echoed
