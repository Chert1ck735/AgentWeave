"""Operator actions and journal maintenance over store operations; no SQL here.

An operator first inspects a task: its unknown step with kind (tool,
python_handler, model) and arguments without secrets. Every mutation then names
the version it was decided on. A tool result is checked against the schema the
call was dispatched with; finishing a task sends its requester a reply, as the
runtime would.
"""
import json
import re
from typing import Any
from pydantic import TypeAdapter, ValidationError
from ..errors import OperatorActionRefused
from ..messages import JSONValue, MAX_ERROR_MESSAGE_CHARS
from .engine import reply_envelope
from .tools import schema_error

HANDLER_STEP = 'handler'
REDACTED = '***'
_SECRET_KEY = re.compile(r'secret|token|passw|api[_-]?key|authori[sz]ation|credential|private[_-]?key|cookie', re.I)
_json_value = TypeAdapter(JSONValue)
_json_object = TypeAdapter(dict[str, JSONValue])


def redact(value: Any) -> Any:
    """Arguments as an operator may see them: values under secret-like keys are hidden."""
    if isinstance(value, dict):
        return {key: REDACTED if _SECRET_KEY.search(str(key)) else redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def _steps(view: dict) -> list[dict]:
    task = view['task']
    unknown_tools = [row for row in view['tools'] if row['status'] == 'outcome_unknown']
    steps = [{'step_id': row['attempt_id'], 'kind': 'model', 'step': row['step'], 'attempt': row['attempt'],
              'status': row['status']} for row in view['models']]
    steps += [{'step_id': row['tool_call_id'], 'kind': 'tool', 'tool_id': row['tool_id'], 'status': row['status'],
               'attempts': row['attempts'], 'retry_safe': bool(row['retry_safe']),
               'remote': row['task_id'] != task.task_id, 'arguments': redact(json.loads(row['arguments']))}
              for row in view['tools']]
    handler_unknown = task.status == 'outcome_unknown' and not unknown_tools
    if view['handler'] is not None or handler_unknown:
        steps.append({'step_id': HANDLER_STEP, 'kind': 'python_handler',
                      'status': 'outcome_unknown' if handler_unknown else view['handler']['last_outcome'],
                      'attempts': view['handler']['attempts'] if view['handler'] else 0})
    return steps


async def describe(store, task_id: str) -> dict:
    """The task and its steps, the unknown one marked, without secrets."""
    view = await store.operator_view(task_id)
    if view is None:
        raise OperatorActionRefused('no such task')
    task = view['task']
    return {'task_id': task.task_id, 'agent_id': task.agent_id, 'status': task.status, 'version': task.version,
            'deadline': task.deadline.isoformat(), 'cancel_requested': view['cancel_requested'],
            'purged': view['purged'], 'error': task.error, 'steps': _steps(view)}


def _reason(reason: str) -> str:
    if not isinstance(reason, str) or not reason.strip():
        raise OperatorActionRefused('a reason is required')
    return reason.strip()[:MAX_ERROR_MESSAGE_CHARS]


def _json(value, what):
    try:
        return _json_value.validate_python(value)
    except ValidationError as error:
        raise OperatorActionRefused(f'{what} must be JSON') from error


async def _reply(store, task_id: str, status: str, *, output=None, error=None) -> list:
    """The reply finishing this task sends; nothing for a task that is gone."""
    task = await store.get_task(task_id)
    event = await store.get_input_envelope(task_id) if task is not None else None
    if task is None or event is None:
        return []
    return [reply_envelope(task, event, store.clock.now(), status=status, output=output, error=error)]


async def resolve(store, task_id: str, step_id: str, expected_version: int, *, result: Any,
                  state: dict | None = None, new_deadline=None):
    """Record the confirmed outcome of the unknown step.

    A tool result must match the call's result schema; the loop then resumes
    and replays the journal. A Python handler is finished as a whole; state,
    when given, replaces its session state.
    """
    result = _json(result, 'the result')
    if state is not None:
        try:
            state = _json_object.validate_python(state)
        except ValidationError as error:
            raise OperatorActionRefused('the state must be a JSON object') from error
    outgoing = []
    if step_id == HANDLER_STEP:
        outgoing = await _reply(store, task_id, 'succeeded', output=result)
    else:
        schema = await store.tool_output_schema(step_id)
        if schema is None:
            raise OperatorActionRefused('the step has no journaled result schema to check the result against')
        problem = schema_error(schema, result)
        if problem is not None:
            raise OperatorActionRefused(f'the result does not match the tool result schema: {problem}')
    return await store.resolve_unknown(task_id, step_id, expected_version, result, state=state,
                                       new_deadline=new_deadline, outgoing=outgoing)


async def retry(store, task_id: str, step_id: str, expected_version: int, *, allow_unknown: bool = False,
                new_deadline=None):
    """Run only the unknown step again; allow_unknown acknowledges its effect may repeat."""
    return await store.retry_unknown(task_id, step_id, expected_version, allow_unknown, new_deadline=new_deadline)


async def fail(store, task_id: str, expected_version: int, *, reason: str):
    reason = _reason(reason)
    error = {'type': 'OperatorFailed', 'message': reason}
    outgoing = await _reply(store, task_id, 'failed', error=error)
    return await store.fail_task(task_id, expected_version, reason, outgoing=outgoing)


async def cancel(store, task_id: str, expected_version: int, *, reason: str):
    """Cancel a task; a step in flight still reports its real outcome first."""
    reason = _reason(reason)
    outgoing = await _reply(store, task_id, 'cancelled', error={'type': 'Cancelled', 'message': reason})
    return await store.cancel_task(task_id, expected_version, reason, outgoing=outgoing)


async def inspect_state(store, session_id: str) -> dict:
    return await store.inspect_state(session_id)


async def export_state(store, path) -> dict:
    return await store.export_state(path)


async def cleanup(store, *, dry_run: bool = True, retention=None) -> dict:
    """Retention cleanup; a dry run (the default) reports and changes nothing."""
    return await store.cleanup(dry_run=dry_run, retention=retention)


__all__ = ['cancel', 'cleanup', 'describe', 'export_state', 'fail', 'inspect_state', 'redact', 'resolve', 'retry']
