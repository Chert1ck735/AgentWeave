"""event publish/subscribe: JSON events through the public Client and EventBus, never around admission."""
import asyncio
from datetime import datetime
import json
import math
from pathlib import Path
import signal
import sys
from uuid import UUID, uuid4
from .._infrastructure.processes import InfrastructureError

def _reject_constant(name):
    raise ValueError(f'{name} is not JSON')


def parse_json(text, what):
    """Strict JSON: no NaN or infinities (also from overflowing numbers) and no unbounded nesting."""
    try:
        value = json.loads(text, parse_constant=_reject_constant)
        json.dumps(value, allow_nan=False)
    except (ValueError, RecursionError) as exc:
        raise InfrastructureError('invalid_json', f'{what} must be one JSON value') from exc
    return value


def read_text(file):
    try:
        return Path(file).read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise InfrastructureError('invalid_input', 'cannot read the file as UTF-8 text') from exc


def read_payload(value, file):
    """One JSON value from exactly one source: the argument, --file, or stdin when neither is given."""
    if value is not None and file is not None:
        raise InfrastructureError('invalid_input', 'give the payload once: as an argument, with --file or on stdin')
    if file is not None:
        text = read_text(file)
    elif value is None or value == '-':
        text = sys.stdin.read()
    else:
        text = value
    if not text.strip():
        raise InfrastructureError('invalid_input', 'no payload: pass JSON as an argument, with --file or on stdin')
    return parse_json(text, 'the payload')


def parse_moment(value, flag):
    if value is None:
        return None
    try:
        moment = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise InfrastructureError('invalid_input', f'{flag} must be an ISO 8601 time with a zone') from exc
    if moment.tzinfo is None:
        raise InfrastructureError('invalid_input', f'{flag} must name its time zone')
    return moment


def session(value):
    if value is None:
        return None
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise InfrastructureError('invalid_input', '--session-id must be a UUID') from exc


def _topic(value):
    from ..messages import validate_event_topic
    try:
        return validate_event_topic(value)
    except ValueError as exc:
        raise InfrastructureError('invalid_input', f'invalid event topic: {exc}') from exc


def _metadata(args):
    """Every publish option checked before any connection is made."""
    key = args.key
    if key is not None and not 1 <= len(key.encode('utf-8')) <= 256:
        raise InfrastructureError('invalid_input', '--key must contain 1 to 256 UTF-8 bytes')
    if args.source_version is not None and args.source_version < 0:
        raise InfrastructureError('invalid_input', '--source-version must not be negative')
    return {'session_id': session(args.session_id), 'key': key, 'source_version': args.source_version,
            'event_time': parse_moment(args.event_time, '--event-time'), 'expires_at': parse_moment(args.expires_at, '--expires-at')}


def _waiting(args):
    """--wait AGENT and --timeout, checked before the event is published."""
    if args.wait is None:
        return None
    from ..messages import validate_name
    try:
        validate_name(args.wait)
    except ValueError as exc:
        raise InfrastructureError('invalid_input', '--wait must name an agent') from exc
    if not (math.isfinite(args.timeout) and args.timeout > 0):
        raise InfrastructureError('invalid_input', '--timeout must be a positive number of seconds')
    return args.wait


def publish(root, profile, args, emit):
    """Publish one event; with --wait AGENT also report that agent's result. Returns the exit code."""
    topic = _topic(args.topic)
    payload = read_payload(args.payload, args.file)
    metadata = _metadata(args)
    agent = _waiting(args)
    from .infra import require_running
    require_running(root, profile)
    return asyncio.run(_publish(root, profile, topic, payload, metadata, agent, args.timeout, emit))


async def _publish(root, profile, topic, payload, metadata, agent, timeout, emit):
    from ..client import Client
    async with Client.connect(root, profile) as client:
        event_id = await client.publish(topic, payload, **metadata)
        emit({'event_id': event_id, 'topic': topic})
        if agent is None:
            return 0
        try:
            async with asyncio.timeout(timeout):
                result = await client.event_result(event_id, agent)
        except TimeoutError as exc:
            raise InfrastructureError('task_unknown', f'{agent} reported no result within {timeout:g} seconds') from exc
        emit(result.model_dump(mode='json'))
        return 0 if result.status == 'succeeded' else 4


def _view(envelope):
    return envelope.model_dump(mode='json', include={'event_id', 'topic', 'sender', 'key', 'session_id', 'created_at',
                                                     'event_time', 'expires_at', 'source_version', 'payload'})


def subscribe(root, profile, args, emit):
    """Print events of a topic as JSON lines until --limit events or Ctrl-C. Returns the exit code."""
    topic = _topic(args.topic)
    if args.limit is not None and args.limit < 1:
        raise InfrastructureError('invalid_input', '--limit must be positive')
    from .infra import require_running
    require_running(root, profile)
    try:
        asyncio.run(_subscribe(root, profile, topic, args.limit, args.from_retained, emit))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass  # Ctrl-C or SIGTERM is how a subscription normally ends
    return 0


async def _subscribe(root, profile, topic, limit, from_retained, emit):
    """An own consumer group: the subscriber never takes events from the agents subscribed to topic.

    An event is acknowledged only after its line is written, so nothing is
    confirmed that was not shown.
    """
    from ..client import Client
    # SIGTERM cancels like Ctrl-C does, so the subscription and connections close.
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    async with Client.connect(root, profile) as client:
        since = client.clock.now()
        subscription = await client.bus.subscribe([client.bus.events_topic(topic)], 'cli.subscribe.' + uuid4().hex)
        try:
            emit({'subscribed': topic, 'since': since.isoformat()})
            shown = 0
            async for delivery in subscription:
                if from_retained or delivery.envelope.created_at >= since:
                    emit(_view(delivery.envelope))
                    shown += 1
                await client.bus.ack(delivery)
                if limit is not None and shown >= limit:
                    break
        finally:
            await subscription.aclose()
