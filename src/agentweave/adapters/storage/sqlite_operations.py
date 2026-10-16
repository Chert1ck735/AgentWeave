"""Operator actions, state inspection, backup and retention of the SQLite journal.

Every operator mutation names the task version it was decided on: a stale
version is a VersionConflict, so two operators never create two continuations.
Each one is recorded in the operator_actions audit journal in the same
transaction. Retention removes only finished work and never a dedup ID an
admitted replay could still need.
"""
import asyncio
from datetime import datetime, timedelta
import json
import os
import sqlite3
from pathlib import Path

from ...config import Retention
from ...errors import OperatorActionRefused, VersionConflict
from ...messages import ToolCall, decode, encode

TERMINAL = ('succeeded', 'failed', 'cancelled', 'expired')
_TERMINAL_SQL = "('succeeded','failed','cancelled','expired')"
HANDLER_STEP = 'handler'
DEDUP_FLOOR = 'dedup_floor'
# A task is kept whole while anything of its root is open: a replay of the root reads its children.
_ROOT_CLOSED = (f'NOT EXISTS (SELECT 1 FROM tasks r WHERE r.root_task_id=tasks.root_task_id AND '
                f'r.status NOT IN {_TERMINAL_SQL})')
_RETAINED_DETAILS = ('tool_calls', 'task_events', 'tool_executions', 'model_attempts', 'model_step_catalogs',
                     'handler_attempts', 'execution_gates')


class _DryRun(Exception):
    """Rolls a cleanup transaction back after counting what it would remove."""

    def __init__(self, report):
        super().__init__('dry run')
        self.report = report


class OperatorOperations:
    """Mixed into SQLiteStore: uses its connection, clock, transactions and helpers."""

    retention = Retention()

    # -- reading ---------------------------------------------------------------------------

    async def operator_view(self, task_id: str) -> dict | None:
        """The task, its tool calls (also those of its remote tool tasks) and its model attempts."""
        from .sqlite import _task, _uuid
        task_id = _uuid.validate_python(task_id)
        async with self._lock:
            self._require_ready()
            async def operation():
                row = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (task_id,))
                if row is None:
                    return None
                tools = await self._tool_rows(task_id)
                models = await self._all('SELECT attempt_id,step,attempt,status,input_tokens,output_tokens,cost_usd '
                                         'FROM model_attempts WHERE task_id=? ORDER BY step,attempt', (task_id,))
                attempt = await self._fetch('SELECT attempts,last_outcome,retry_safe FROM handler_attempts WHERE task_id=?',
                                            (task_id,))
                return {'task': _task(row), 'cancel_requested': bool(row['cancel_requested']),
                        'purged': row['purged_at'] is not None,
                        'tools': [dict(r) for r in tools], 'models': [dict(r) for r in models],
                        'handler': dict(attempt) if attempt else None}
            return await self._retry(operation)

    async def _tool_rows(self, task_id):
        """Tool rows of the task and of the remote tool tasks it submitted."""
        return await self._all(
            "SELECT * FROM tool_executions WHERE task_id=? OR task_id IN (SELECT task_id FROM tasks WHERE "
            "json_extract(input_envelope,'$.parent_task_id')=? AND json_extract(input_envelope,'$.kind')='tool_call') "
            'ORDER BY updated_at,tool_call_id', (task_id, task_id))

    async def tool_output_schema(self, tool_call_id: str) -> dict | None:
        from .sqlite import _uuid
        tool_call_id = _uuid.validate_python(tool_call_id)
        async with self._lock:
            self._require_ready()
            async def operation():
                row = await self._fetch('SELECT output_schema FROM tool_executions WHERE tool_call_id=?', (tool_call_id,))
                return None if row is None or row['output_schema'] is None else json.loads(row['output_schema'])
            return await self._retry(operation)

    async def inspect_state(self, session_id: str) -> dict:
        from .sqlite import _uuid
        session_id = _uuid.validate_python(session_id)
        async with self._lock:
            self._require_ready()
            async def operation():
                sessions = await self._all(
                    'SELECT s.agent_id,s.state,s.updated_at,(SELECT COUNT(*) FROM messages m WHERE m.session_id=s.session_id '
                    'AND m.agent_id=s.agent_id) AS messages FROM sessions s WHERE s.session_id=? ORDER BY s.agent_id',
                    (session_id,))
                conversation = await self._fetch('SELECT active_agent,version,updated_at FROM conversations '
                                                 'WHERE session_id=?', (session_id,))
                tasks = await self._all('SELECT task_id,agent_id,status,version,updated_at FROM tasks WHERE session_id=? '
                                        'ORDER BY rowid', (session_id,))
                return {'session_id': session_id,
                        'sessions': [{'agent_id': r['agent_id'], 'state': json.loads(r['state']),
                                      'messages': r['messages'], 'updated_at': r['updated_at']} for r in sessions],
                        'conversation': dict(conversation) if conversation else None,
                        'tasks': [dict(r) for r in tasks]}
            return await self._retry(operation)

    async def export_state(self, path) -> dict:
        """A consistent copy through SQLite's backup API, which also reads pages still in the WAL."""
        target = Path(path)
        if not target.parent.is_dir():
            raise OperatorActionRefused('the export directory does not exist')
        self._require_ready()
        try:
            # Created private and exclusively: a copy of the journal is never readable by others.
            os.close(os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        except FileExistsError as error:
            raise OperatorActionRefused('the export target already exists; choose a new file') from error

        def copy():
            source = sqlite3.connect(self.path)
            try:
                destination = sqlite3.connect(target)
                try:
                    source.backup(destination)
                    return destination.execute('SELECT COUNT(*) FROM tasks').fetchone()[0]
                finally:
                    destination.close()
            finally:
                source.close()
        count = await asyncio.to_thread(copy)
        return {'path': str(target), 'tasks': count}

    # -- operator mutations ----------------------------------------------------------------

    async def resolve_unknown(self, task_id, step_id, expected_version, result, state=None, new_deadline=None, *,
                              outgoing=()):
        """Record the confirmed outcome of the unknown step.

        A tool call gets its result and the task continues (accepted again);
        an opaque handler is finished as a whole with result and, if given,
        its full confirmed state; outgoing is then its reply.
        """
        from .sqlite import _json, _stamp
        async def operation():
            row = await self._operable(task_id, expected_version, 'outcome_unknown')
            kind, tool = await self._unknown_step(row, step_id)
            now = self.clock.now()
            if kind == 'tool':
                deadline = self._continuation_deadline(row, new_deadline, now)
                await self._connection.execute(
                    "UPDATE tool_executions SET status='succeeded',result=?,error='null',hook_status=NULL,"
                    "hook_result='null',hook_error='null',updated_at=? WHERE tool_call_id=?",
                    (_json(result), _stamp(now), tool['tool_call_id']))
                if tool['task_id'] != row['task_id']:  # a remote tool task: it finished with this result
                    await self._finish(tool['task_id'], 'succeeded', now, result={
                        'tool_call_id': tool['tool_call_id'], 'tool_id': tool['tool_id'], 'status': 'succeeded',
                        'output': result, 'error': None})
                version = await self._continue(row, deadline, now)
            else:
                if state is not None and row['session_id'] is None:
                    raise OperatorActionRefused('this task has no session to hold a state')
                if state is not None:
                    await self._connection.execute(
                        'INSERT INTO sessions VALUES(?,?,?,?) ON CONFLICT(agent_id,session_id) DO UPDATE SET '
                        'state=excluded.state,updated_at=excluded.updated_at',
                        (row['agent_id'], row['session_id'], _json(state), _stamp(now)))
                version = await self._finish(row['task_id'], 'succeeded', now, result=result, outgoing=outgoing)
            await self._audit(row, 'resolve', step_id, version, now,
                              {'kind': kind, 'state': state is not None, 'new_deadline': _stamped(new_deadline)})
            return await self._task_row(row['task_id'])
        return await self._mutate(operation)

    async def retry_unknown(self, task_id, step_id, expected_version, allow_unknown, new_deadline=None):
        """Run the unknown step again: only that tool call, or the opaque handler from its start."""
        from .sqlite import _stamp
        async def operation():
            row = await self._operable(task_id, expected_version, 'outcome_unknown')
            if allow_unknown is not True:
                raise OperatorActionRefused('retrying an unknown outcome may repeat its effect: allow_unknown is required')
            kind, tool = await self._unknown_step(row, step_id)
            now = self.clock.now()
            deadline = self._continuation_deadline(row, new_deadline, now)
            if kind == 'tool':
                # One more attempt than already made, whatever the journaled retry policy said.
                await self._connection.execute("UPDATE tool_executions SET status='retry',max_attempts=attempts+1,"
                                               'updated_at=? WHERE tool_call_id=?', (_stamp(now), tool['tool_call_id']))
                if tool['task_id'] != row['task_id']:
                    child = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (tool['task_id'],))
                    await self._continue(child, deadline, now)
            version = await self._continue(row, deadline, now)
            await self._audit(row, 'retry', step_id, version, now, {'kind': kind, 'new_deadline': _stamped(new_deadline)})
            return await self._task_row(row['task_id'])
        return await self._mutate(operation)

    async def fail_task(self, task_id, expected_version, reason, *, outgoing=()):
        """End a task that is not running with an operator error."""
        async def operation():
            row = await self._operable(task_id, expected_version)
            now = self.clock.now()
            if self._leased(row, now):
                raise OperatorActionRefused('the task is running; cancel it, then fail it if it ends unknown')
            version = await self._finish(row['task_id'], 'failed', now, outgoing=outgoing,
                                         error={'type': 'OperatorFailed', 'message': reason})
            await self._audit(row, 'fail', None, version, now, {'reason': reason})
            return await self._task_row(row['task_id'])
        return await self._mutate(operation)

    async def cancel_task(self, task_id, expected_version, reason, *, outgoing=()):
        """Cancel now if nothing runs; otherwise forbid new steps and let the running one report."""
        async def operation():
            row = await self._operable(task_id, expected_version)
            if row['status'] == 'outcome_unknown':
                raise OperatorActionRefused('the outcome is unknown: resolve, retry or fail the task instead')
            if json.loads(row['input_envelope']).get('kind') == 'tool_call':
                raise OperatorActionRefused('a tool call is cancelled through its caller: cancel the calling task')
            now = self.clock.now()
            reply = encode(outgoing[0]).decode('utf-8') if outgoing else None
            await self._connection.execute('UPDATE tasks SET cancel_requested=1,cancel_reply=? WHERE task_id=?',
                                           (reply, row['task_id']))
            if self._leased(row, now):
                await self._audit(row, 'cancel_requested', None, row['version'], now, {'reason': reason})
            elif await self.effect_in_flight(row['task_id']):
                # A crashed attempt may have acted: its outcome stays unknown, never "cancelled".
                version = await self._finish(row['task_id'], 'outcome_unknown', now)
                await self._audit(row, 'cancel', None, version, now, {'reason': reason, 'outcome': 'outcome_unknown'})
            else:
                version = await self._finish(row['task_id'], 'cancelled', now, outgoing=outgoing,
                                             error={'type': 'Cancelled', 'message': reason})
                await self._audit(row, 'cancel', None, version, now, {'reason': reason})
            return await self._task_row(row['task_id'])
        return await self._mutate(operation)

    # -- helpers for mutations -------------------------------------------------------------

    async def effect_in_flight(self, task_id) -> bool:
        """Inside a transaction: an interrupted attempt that is not safe to repeat may have acted."""
        tools = await self._tool_rows(task_id)
        if any(row['status'] == 'running' and not row['retry_safe'] for row in tools):
            return True
        attempt = await self._fetch("SELECT 1 FROM handler_attempts WHERE task_id=? AND last_outcome='started' "
                                    'AND retry_safe=0', (task_id,))
        return attempt is not None

    async def _operable(self, task_id, expected_version, status=None):
        from .sqlite import _uuid
        task_id = _uuid.validate_python(task_id)
        if type(expected_version) is not int:
            raise ValueError('an operator action needs the expected task version')
        row = await self._fetch('SELECT * FROM tasks WHERE task_id=?', (task_id,))
        if row is None:
            raise OperatorActionRefused('no such task')
        if row['version'] != expected_version:
            raise VersionConflict('the task changed since it was inspected; inspect it again')
        if row['purged_at'] is not None or row['status'] in TERMINAL or (status and row['status'] != status):
            raise VersionConflict(f'the task is {row["status"]}; this action does not apply')
        return row

    async def _unknown_step(self, row, step_id):
        tools = [r for r in await self._tool_rows(row['task_id']) if r['status'] == 'outcome_unknown']
        if step_id == HANDLER_STEP:
            if tools:
                raise OperatorActionRefused('the unknown step is a tool call; act on its tool_call_id')
            return 'python_handler', None
        match = [r for r in tools if r['tool_call_id'] == step_id]
        if not match:
            raise OperatorActionRefused('the step is not an unknown step of this task')
        return 'tool', match[0]

    @staticmethod
    def _continuation_deadline(row, new_deadline, now):
        from .sqlite import _stamp
        expires_at = decode(row['input_envelope'].encode()).expires_at
        if expires_at is not None and expires_at <= now:
            raise OperatorActionRefused('the input itself has expired; nothing can continue it, fail the task instead')
        if new_deadline is None:
            if row['deadline'] <= _stamp(now):
                raise OperatorActionRefused('the task deadline has passed; a continuation needs an explicit new deadline')
            return None
        if new_deadline <= now:
            raise OperatorActionRefused('the new deadline must be in the future')
        return new_deadline

    @staticmethod
    def _leased(row, now):
        from .sqlite import _stamp
        return row['lease_expires_at'] is not None and row['lease_expires_at'] > _stamp(now)

    async def _continue(self, row, deadline, now):
        """Make the task runnable again, with a fresh handler attempt and an optional new deadline."""
        from .sqlite import _stamp
        envelope = decode(row['input_envelope'].encode())
        if deadline is not None:
            await self._extend(row, deadline)
            envelope = envelope.model_copy(update={'deadline': deadline})
        version = row['version'] + 1
        cursor = await self._connection.execute(
            "UPDATE tasks SET status='accepted',version=?,error='null',deadline=?,input_envelope=?,lease_instance_id=NULL,"
            'lease_expires_at=NULL,not_before=NULL,updated_at=? WHERE task_id=? AND version=?',
            (version, _stamp(envelope.deadline), envelope.model_dump_json(), _stamp(now), row['task_id'], row['version']))
        if cursor.rowcount != 1:
            raise VersionConflict('the task changed during the operator action')
        await self._connection.execute('DELETE FROM handler_attempts WHERE task_id=?', (row['task_id'],))
        await self._event(row['task_id'], 'accepted', version, now)
        return version

    async def _extend(self, row, deadline):
        """Carry a new deadline into every journaled call of the task, so replays still match them."""
        from .sqlite import _json, _stamp
        for tool in await self._tool_rows(row['task_id']):
            requested = ToolCall.model_validate_json(tool['requested']).model_copy(update={'deadline': deadline})
            await self._connection.execute('UPDATE tool_executions SET requested=? WHERE tool_call_id=?',
                                           (_json(requested.model_dump(mode='json')), tool['tool_call_id']))
        children = await self._all("SELECT * FROM tasks WHERE json_extract(input_envelope,'$.parent_task_id')=? AND "
                                   "json_extract(input_envelope,'$.kind')='tool_call'", (row['task_id'],))
        for child in children:
            event = _with_deadline(decode(child['input_envelope'].encode()), deadline)
            await self._connection.execute('UPDATE tasks SET deadline=?,input=?,input_envelope=? WHERE task_id=?',
                                           (_stamp(deadline), _json(event.payload), event.model_dump_json(),
                                            child['task_id']))
        submissions = await self._all(
            "SELECT o.event_id,o.envelope FROM submissions s JOIN outbox o ON o.event_id=s.event_id "
            "WHERE json_extract(o.envelope,'$.parent_task_id')=? AND json_extract(o.envelope,'$.kind')='tool_call'",
            (row['task_id'],))
        for submitted in submissions:
            event = _with_deadline(decode(submitted['envelope'].encode()), deadline)
            await self._connection.execute('UPDATE outbox SET envelope=?,deadline=? WHERE event_id=?',
                                           (encode(event).decode('utf-8'), _stamp(deadline), submitted['event_id']))

    async def _finish(self, task_id, status, now, *, result=None, error=None, outgoing=()):
        from .sqlite import _json, _stamp
        row = await self._fetch('SELECT version FROM tasks WHERE task_id=?', (task_id,))
        version = row['version'] + 1
        await self._connection.execute(
            'UPDATE tasks SET status=?,version=?,result=?,error=?,lease_instance_id=NULL,lease_expires_at=NULL,'
            'updated_at=? WHERE task_id=?', (status, version, _json(result), _json(error), _stamp(now), task_id))
        await self._connection.execute('DELETE FROM execution_gates WHERE task_id=?', (task_id,))
        for envelope in outgoing:
            await self._insert_outbox(envelope, _stamp(now))
        await self._event(task_id, status, version, now)
        return version

    async def _event(self, task_id, kind, version, now):
        from .sqlite import _stamp
        await self._connection.execute('INSERT INTO task_events(task_id,kind,version,recorded_at) VALUES(?,?,?,?)',
                                       (task_id, kind, version, _stamp(now)))

    async def _audit(self, row, action, step_id, version, now, details):
        from .sqlite import _json, _stamp
        kept = {key: value for key, value in details.items() if value not in (None, False)}
        await self._connection.execute(
            'INSERT INTO operator_actions(task_id,action,step_id,expected_version,new_version,details,recorded_at) '
            'VALUES(?,?,?,?,?,?,?)', (row['task_id'], action, step_id, row['version'], version, _json(kept), _stamp(now)))

    async def _task_row(self, task_id):
        from .sqlite import _task
        return _task(await self._fetch('SELECT * FROM tasks WHERE task_id=?', (task_id,)))

    # -- retention -------------------------------------------------------------------------

    async def dedup_floor(self) -> datetime | None:
        """Inside a transaction: the newest event created_at whose dedup ID was removed."""
        row = await self._fetch('SELECT value FROM maintenance_marks WHERE name=?', (DEDUP_FLOOR,))
        return None if row is None else datetime.fromisoformat(row['value'])

    async def cleanup(self, now=None, dry_run=True, retention=None) -> dict:
        """Remove finished work past its retention; a dry run reports the same counts and keeps everything."""
        retention = retention or self.retention
        now = now or self.clock.now()
        cutoffs = {'terminal': now - timedelta(days=retention.terminal_days),
                   'idle_session': now - timedelta(days=retention.idle_session_days),
                   'dedup': now - timedelta(days=retention.replay_window_days + 1)}
        async def operation():
            report = {'dry_run': dry_run, 'idle_sessions': await self._expire_sessions(cutoffs['idle_session']),
                      'purged_tasks': await self._purge_tasks(cutoffs['terminal'], now),
                      'expired_ids': await self._expire_ids(cutoffs['dedup']),
                      'deleted_tasks': await self._delete_tasks(),
                      'cutoffs': {name: value.isoformat() for name, value in cutoffs.items()}}
            if dry_run:
                raise _DryRun(report)
            return report
        try:
            return await self._mutate(operation)
        except _DryRun as rolled_back:
            return rolled_back.report

    async def _expire_sessions(self, cutoff):
        """Stateful history idle since cutoff, with no open task and no session gate held."""
        from .sqlite import _stamp
        quiet = (f'NOT EXISTS (SELECT 1 FROM tasks t WHERE t.session_id={{0}}.session_id AND '
                 f'(t.status NOT IN {_TERMINAL_SQL} OR t.updated_at>=?))')
        idle = await self._all(
            'SELECT s.agent_id,s.session_id FROM sessions s WHERE s.updated_at<? AND ' + quiet.format('s') +
            " AND NOT EXISTS (SELECT 1 FROM execution_gates g WHERE g.subscription='$session' AND g.key=s.session_id)",
            (_stamp(cutoff), _stamp(cutoff)))
        for row in idle:
            for table in ('messages', 'sessions'):
                await self._connection.execute(f'DELETE FROM {table} WHERE session_id=? AND agent_id=?',
                                               (row['session_id'], row['agent_id']))
        await self._connection.execute('DELETE FROM conversations WHERE updated_at<? AND ' + quiet.format('conversations'),
                                       (_stamp(cutoff), _stamp(cutoff)))
        return len(idle)

    async def _purge_tasks(self, cutoff, now):
        """Drop the payloads and journal details of tasks finished before cutoff; keep their identity."""
        from .sqlite import _stamp
        selected = (f'SELECT task_id FROM tasks WHERE status IN {_TERMINAL_SQL} AND updated_at<? AND purged_at IS NULL '
                    f'AND {_ROOT_CLOSED}')
        values = (_stamp(cutoff),)
        count = (await self._fetch(f'SELECT COUNT(*) AS count FROM ({selected})', values))['count']
        for table in _RETAINED_DETAILS:
            await self._connection.execute(f'DELETE FROM {table} WHERE task_id IN ({selected})', values)
        # Steps carrying a live session's messages stay until that history expires.
        await self._connection.execute(
            f'DELETE FROM steps WHERE task_id IN ({selected}) AND NOT EXISTS (SELECT 1 FROM messages m '
            'WHERE m.task_id=steps.task_id AND m.step_id=steps.step_id)', values)
        await self._connection.execute(
            f"DELETE FROM outbox WHERE delivered=1 AND json_extract(envelope,'$.task_id') IN ({selected}) AND event_id "
            'NOT IN (SELECT receipt_event_id FROM receipts) AND event_id NOT IN (SELECT event_id FROM submissions)', values)
        await self._connection.execute(
            "UPDATE tasks SET input='null',result='null',error='null',waiting_for='null',transfer_context='null',"
            f"input_envelope=json_set(input_envelope,'$.payload',json('null')),purged_at=? WHERE task_id IN ({selected})",
            (_stamp(now),) + values)
        return count

    async def _expire_ids(self, cutoff):
        """Dedup IDs of purged tasks admitted before cutoff; later replays of older events are refused."""
        from .sqlite import _stamp
        expired = await self._all(
            "SELECT p.agent_id,p.event_id,p.receipt_event_id,json_extract(tasks.input_envelope,'$.created_at') AS created_at "
            'FROM processed_events p JOIN tasks ON tasks.task_id=p.task_id WHERE tasks.purged_at IS NOT NULL '
            f'AND tasks.admitted_at<? AND {_ROOT_CLOSED} AND NOT EXISTS (SELECT 1 FROM outbox o WHERE '
            'o.event_id=p.receipt_event_id AND o.delivered=0)', (_stamp(cutoff),))
        if not expired:
            return 0
        floor = max(datetime.fromisoformat(row['created_at'].replace('Z', '+00:00')) for row in expired)
        current = await self.dedup_floor()
        if current is None or floor > current:
            await self._connection.execute(
                'INSERT INTO maintenance_marks VALUES(?,?) ON CONFLICT(name) DO UPDATE SET value=excluded.value',
                (DEDUP_FLOOR, _stamp(floor)))
        for row in expired:
            for table in ('receipts', 'processed_events'):
                await self._connection.execute(f'DELETE FROM {table} WHERE agent_id=? AND event_id=?',
                                               (row['agent_id'], row['event_id']))
            await self._connection.execute('DELETE FROM outbox WHERE event_id=? AND delivered=1 AND event_id NOT IN '
                                           '(SELECT event_id FROM submissions)', (row['receipt_event_id'],))
        return len(expired)

    async def _delete_tasks(self):
        """Purged tasks nothing refers to any more: no dedup ID and no session message."""
        removable = ('SELECT task_id FROM tasks WHERE purged_at IS NOT NULL AND NOT EXISTS (SELECT 1 FROM processed_events p '
                     'WHERE p.task_id=tasks.task_id) AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.task_id=tasks.task_id) '
                     f'AND {_ROOT_CLOSED}')
        count = (await self._fetch(f'SELECT COUNT(*) AS count FROM ({removable})'))['count']
        events = [row['event_id'] for row in await self._all(f'SELECT event_id FROM submissions WHERE task_id IN ({removable})')]
        for table in ('submission_outcomes', 'submissions'):
            await self._connection.execute(f'DELETE FROM {table} WHERE task_id IN ({removable})')
        for event_id in events:
            await self._connection.execute('DELETE FROM outbox WHERE event_id=? AND delivered=1 AND event_id NOT IN '
                                           '(SELECT receipt_event_id FROM receipts)', (event_id,))
        for table in ('steps', 'tasks'):
            await self._connection.execute(f'DELETE FROM {table} WHERE task_id IN ({removable})')
        return count


def _with_deadline(event, deadline):
    """A tool_call envelope whose call and envelope carry the new deadline."""
    call = ToolCall.model_validate(event.payload).model_copy(update={'deadline': deadline})
    return event.model_copy(update={'deadline': deadline, 'payload': call.model_dump(mode='json')})


def _stamped(value):
    return None if value is None else value.isoformat()


__all__ = ['HANDLER_STEP', 'OperatorOperations']
