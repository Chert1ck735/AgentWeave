"""Wire tests catch dropped validation, unsafe JSON and byte-size errors."""
from datetime import datetime, timezone, timedelta
import importlib
import json
from uuid import uuid4
import pytest


def api():
    return importlib.import_module('agentweave.messages')


def fields():
    return dict(event_id=str(uuid4()), network_id=str(uuid4()), kind='request', sender='client',
                recipient='assistant', topic=None, task_id=str(uuid4()), parent_task_id=None,
                correlation_id=None, session_id=None, created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                deadline=datetime(2025, 1, 1, 0, 2, tzinfo=timezone.utc), traceparent=None, payload={'text':'Привет'})


def test_wire_round_trip_preserves_unicode_and_utc():
    m = api()
    e = m.Envelope(**fields())
    assert m.decode(m.encode(e)) == e
    assert json.loads(m.encode(e))['schema_version'] == 1
    assert e.created_at.utcoffset() == timedelta(0)
    with pytest.raises(Exception):
        e.event_id = str(uuid4())


@pytest.mark.parametrize('change', [dict(event_id='bad'), dict(network_id='bad'),
    dict(created_at=datetime(2025, 1, 1)), dict(payload=float('nan')), dict(payload={'n':float('inf')}),
    dict(payload={1:'bad'}), dict(payload=(1, 2)), dict(schema_version=2), dict(unknown=True),
    dict(recipient=None, topic=None), dict(recipient='assistant', topic='events.test')])
def test_rejects_invalid_envelopes(change):
    with pytest.raises(ValueError):
        api().Envelope(**(fields() | change))


def test_wire_decode_missing_fields_unknown_version_and_invalid_json():
    m = api()
    for raw in [b'{}', b'{"schema_version":2}', b'{"payload":NaN}', b'\xff', b'[]']:
        with pytest.raises(m.InvalidEnvelope):
            m.decode(raw)


def test_encode_revalidates_copies_and_checks_utf8_limit():
    m = api()
    e = m.Envelope(**fields())
    with pytest.raises(m.EnvelopeTooLarge):
        m.encode(e.model_copy(update={'payload':'я' * 524288}))
    with pytest.raises(m.InvalidEnvelope):
        m.encode(e.model_copy(update={'payload':float('nan')}))
    with pytest.raises(m.EnvelopeTooLarge):
        m.decode(b' ' * 1048577)


def test_model_and_storage_dtos_use_real_shared_contracts():
    m = api()
    models = importlib.import_module('agentweave.models')
    records = importlib.import_module('agentweave._records')
    call = models.ProposedToolCall(provider_call_id='native-1', name='calc.add', arguments={'a':1})
    turn = models.ModelTurn(messages=[m.Message(role='assistant', content='hello')], tool_calls=[call])
    assert turn.tool_calls[0].provider_call_id == 'native-1'
    with pytest.raises(ValueError):
        models.ProposedToolCall(provider_call_id='x', name='a', arguments={}, caller='spoof')
    task = records.TaskRecord(task_id=fields()['task_id'], agent_id='assistant', status='accepted',
        version=0, input={'a':1}, session_id=None, root_task_id=fields()['task_id'],
        definition_hash='abc', deadline=fields()['deadline'], result=None, error=None)
    assert task.status == 'accepted'
    assert records.SystemClock().now().utcoffset() == timedelta(0)
    with pytest.raises(ValueError):
        records.StepCommit(step_id='s', expected_version=-1, status='running')


def test_schema_version_cannot_be_boolean_or_coerced():
    for version in [True, '1', 1.0]:
        with pytest.raises(ValueError):
            api().Envelope(**fields(), schema_version=version)


def test_exact_wire_limit_and_duplicate_json_keys():
    m = api()
    e = m.Envelope(**(fields() | {'payload':''}))
    overhead = len(m.encode(e))
    exact = e.model_copy(update={'payload':'x' * (1048576-overhead)})
    assert len(m.encode(exact)) == 1048576
    assert m.decode(m.encode(exact)).payload == exact.payload
    with pytest.raises(m.EnvelopeTooLarge):
        m.encode(exact.model_copy(update={'payload':exact.payload+'x'}))
    with pytest.raises(m.InvalidEnvelope):
        m.decode(b'{"schema_version":1,"schema_version":2}')


def test_lease_token_is_numeric_monotonic_fencing_value():
    r = importlib.import_module('agentweave._records')
    common = dict(task_id=fields()['task_id'], instance_id=str(uuid4()), expires_at=fields()['deadline'])
    old = r.Lease(**common, token=1)
    new = r.Lease(**common, token=2)
    assert new.token > old.token
    for token in ['opaque', '1', True, 0]:
        with pytest.raises(ValueError):
            r.Lease(**common, token=token)


def test_cyclic_json_is_rejected_at_the_boundary():
    values = []; values.append(values)
    with pytest.raises(ValueError):
        api().Envelope(**(fields() | {'payload':values}))


def test_optional_metadata_roundtrip_and_old_wire(envelope_factory):
    from agentweave.messages import encode, decode
    import json
    event = envelope_factory(key='диалог', event_time=envelope_factory().created_at,
                             expires_at=envelope_factory().deadline, source_version=0)
    assert decode(encode(event)) == event
    old = json.loads(encode(event))
    for field in ('key', 'event_time', 'expires_at', 'source_version'):
        del old[field]
    assert decode(json.dumps(old).encode()).key is None


@pytest.mark.parametrize('changes', [dict(key=''), dict(key='я' * 129),
    dict(source_version=True), dict(source_version=-1),
    dict(event_time=datetime(2025, 1, 1)), dict(expires_at=datetime(2025, 1, 1))])
def test_metadata_rejects_invalid_values(envelope_factory, changes):
    with pytest.raises(ValueError):
        envelope_factory(**changes)


def test_submission_only_terminal_outcome_has_no_responding_agent():
    from agentweave.messages import TaskResult
    result = TaskResult(task_id='00000000-0000-0000-0000-000000000001',
        output=None,responding_agent=None,status='expired')
    assert TaskResult.model_validate_json(result.model_dump_json()) == result
