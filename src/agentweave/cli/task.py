"""task status/result/events/cancel/resolve/fail/retry: the journal only, no broker needed."""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from .._infrastructure.processes import InfrastructureError

_TERMINAL = ('succeeded', 'failed', 'cancelled', 'expired', 'outcome_unknown')


@asynccontextmanager
async def journal(root, profile):
    """The profile's SQLite journal, opened for an operator; it must exist already."""
    from ..adapters.storage.sqlite import SQLiteStore
    from ..config import resolve
    from ..errors import ConfigError
    from .._infrastructure.bootstrap import Paths
    try:
        config = resolve(root, profile)
    except ConfigError as exc:
        if Paths(root, profile).config.exists():
            raise InfrastructureError('profile_invalid', 'the profile configuration is invalid') from exc
        raise InfrastructureError('journal_unavailable', 'this profile has no journal yet') from exc
    url = config.database_url or ''
    path = Path(url.removeprefix('sqlite:///'))
    if not url.startswith('sqlite:///') or not path.is_file():
        raise InfrastructureError('journal_unavailable', 'this profile has no journal yet')
    from ..errors import StoreConflict
    store = SQLiteStore(path)
    store.retention = config.retention
    try:
        try:
            await store.migrate()
        except StoreConflict as exc:
            raise InfrastructureError('journal_unavailable', 'the journal has a schema this SDK does not know') from exc
        yield store
    finally:
        await store.close()


def operator_errors(action):
    """Run an operator coroutine; refusals and stale versions are input errors, never tracebacks."""
    from ..errors import OperatorActionRefused, VersionConflict
    try:
        return asyncio.run(action)
    except OperatorActionRefused as exc:
        raise InfrastructureError('action_refused', str(exc)) from exc
    except VersionConflict as exc:
        raise InfrastructureError('version_conflict', str(exc)) from exc


def _record(task):
    return {'task_id': task.task_id, 'agent_id': task.agent_id, 'status': task.status, 'version': task.version,
            'deadline': task.deadline.isoformat()}


def command(root, profile, args, emit):
    """Dispatch one task command; returns the exit code."""
    return operator_errors(_command(root, profile, args, emit))


async def _command(root, profile, args, emit):
    from .._runtime import maintenance
    async with journal(root, profile) as store:
        action = args.action
        if action == 'status':
            await _task(store, args.task_id)
            emit(await maintenance.describe(store, args.task_id))
        elif action == 'result':
            task = await _task(store, args.task_id)
            emit({'task_id': task.task_id, 'status': task.status, 'output': task.result, 'error': task.error,
                  'responding_agent': task.agent_id, 'terminal': task.status in _TERMINAL})
            return 4 if task.status in _TERMINAL and task.status != 'succeeded' else 0
        elif action == 'events':
            await _task(store, args.task_id)
            for event in await store.read_progress(args.task_id):
                emit(event.model_dump(mode='json', include={'event_id', 'created_at', 'payload'}))
        elif action == 'resolve':
            result, state = _resolution(args)
            await _task(store, args.task_id)
            emit(_record(await maintenance.resolve(store, args.task_id, args.step, args.expected_version,
                                                   result=result, state=state, new_deadline=_deadline(args))))
        elif action == 'retry':
            await _task(store, args.task_id)
            emit(_record(await maintenance.retry(store, args.task_id, args.step, args.expected_version,
                                                 allow_unknown=args.allow_unknown, new_deadline=_deadline(args))))
        else:  # cancel or fail
            await _task(store, args.task_id)
            run = maintenance.cancel if action == 'cancel' else maintenance.fail
            emit(_record(await run(store, args.task_id, args.expected_version, reason=args.reason)))
    return 0


async def _task(store, task_id):
    from .event import session
    try:
        session(task_id)
    except InfrastructureError as exc:
        raise InfrastructureError('invalid_input', 'the task ID must be a UUID') from exc
    task = await store.get_task(task_id)
    if task is None:
        raise InfrastructureError('task_not_found', 'no such task in this profile')
    return task


def _resolution(args):
    from .event import parse_json, read_text
    if (args.result is None) == (args.result_file is None):
        raise InfrastructureError('invalid_input', 'give the confirmed result once: --result or --result-file')
    result = parse_json(args.result if args.result is not None else read_text(args.result_file), 'the result')
    state = None if args.state is None else parse_json(args.state, 'the state')
    if state is not None and not isinstance(state, dict):
        raise InfrastructureError('invalid_input', 'the state must be a JSON object')
    return result, state


def _deadline(args):
    from .event import parse_moment
    return parse_moment(args.deadline, '--deadline')
