"""Pure JSON messages and the versioned transport boundary."""
from datetime import datetime, timezone
import json
import math
import re
from typing import Annotated, Literal
from uuid import UUID, uuid5
from pydantic import BaseModel, ConfigDict, BeforeValidator, JsonValue, model_validator, field_validator
from .errors import EnvelopeTooLarge, InvalidEnvelope

MAX_ENVELOPE_BYTES = 1048576


def _json(value):
    active = set()
    def walk(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) not in (list, dict):
            raise ValueError('expected finite JSON values and string object keys')
        if type(item) is dict and not all(type(key) is str for key in item):
            raise ValueError('JSON object keys must be strings')
        identity = id(item)
        if identity in active:
            raise ValueError('cyclic values are not JSON')
        active.add(identity)
        try:
            for child in item.values() if type(item) is dict else item:
                walk(child)
        finally:
            active.remove(identity)
    try:
        walk(value)
    except RecursionError as exc:
        raise ValueError('JSON nesting exceeds the supported depth') from exc
    return value


def _uuid(value):
    if type(value) is not str:
        raise ValueError('ID must be a UUID string')
    try:
        canonical = str(UUID(value))
    except (ValueError, AttributeError) as exc:
        raise ValueError('ID must be a UUID string') from exc
    if value != canonical:
        raise ValueError('ID must be a canonical UUID string')
    return value


def validate_name(value: str) -> str:
    """Validate logical names and topic segments, with no lossy normalization."""
    if type(value) is not str or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_][A-Za-z0-9_-]*)*', value):
        raise ValueError('expected a dotted logical name with letters, numbers, underscores or hyphens')
    if len(value.encode('utf-8')) > 200:
        raise ValueError('logical name is too long')
    return value


def _utc(value):
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError('invalid UTC timestamp') from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('timestamp must be timezone aware')
    return value.astimezone(timezone.utc)

JSONValue = Annotated[JsonValue, BeforeValidator(_json)]
UUIDString = Annotated[str, BeforeValidator(_uuid)]
LogicalName = Annotated[str, BeforeValidator(validate_name)]
UTCDateTime = Annotated[datetime, BeforeValidator(_utc)]
TaskStatus = Literal['submitted', 'accepted', 'running', 'waiting_peer', 'waiting_tool',
                     'outcome_unknown', 'succeeded', 'failed', 'cancelled', 'expired']

class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', frozen=True, allow_inf_nan=False, serialize_by_alias=True)

class Envelope(StrictModel):
    schema_version: Literal[1] = 1
    event_id: UUIDString
    network_id: UUIDString
    kind: LogicalName
    sender: LogicalName
    recipient: LogicalName | None
    topic: LogicalName | None
    task_id: UUIDString | None
    parent_task_id: UUIDString | None
    correlation_id: UUIDString | None
    session_id: UUIDString | None
    created_at: UTCDateTime
    deadline: UTCDateTime
    traceparent: str | None
    payload: JSONValue
    key: str | None = None
    event_time: UTCDateTime | None = None
    expires_at: UTCDateTime | None = None
    source_version: int | None = None

    @field_validator('key')
    @classmethod
    def valid_key(cls, value):
        if value is not None and not 1 <= len(value.encode('utf-8')) <= 256:
            raise ValueError('key must contain 1 to 256 UTF-8 bytes')
        return value

    @field_validator('source_version', mode='before')
    @classmethod
    def valid_source_version(cls, value):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError('source_version must be a nonnegative integer')
        return value

    @field_validator('schema_version', mode='before')
    @classmethod
    def strict_version(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError('unsupported schema_version')
        return value

    @model_validator(mode='after')
    def destination(self):
        if (self.recipient is None) == (self.topic is None):
            raise ValueError('exactly one recipient or topic is required')
        return self

class ToolCall(StrictModel):
    tool_call_id: UUIDString
    tool_id: LogicalName
    revision: str
    arguments: dict[str, JSONValue]
    caller: LogicalName
    deadline: UTCDateTime
    expires_at: UTCDateTime | None = None

ToolStatus = Literal['succeeded', 'failed', 'denied', 'invalid', 'expired', 'outcome_unknown']

class ToolResult(StrictModel):
    """Outcome of one tool call; only succeeded and outcome_unknown may follow an effect."""
    tool_call_id: UUIDString
    tool_id: LogicalName
    status: ToolStatus
    output: JSONValue = None
    error: JSONValue = None

class Message(StrictModel):
    role: Literal['system', 'user', 'assistant', 'tool']
    content: JSONValue = None
    tool_calls: list[JSONValue] = []
    tool_results: list[JSONValue] = []
    adapter_metadata: dict[str, JSONValue] = {}

class TaskResult(StrictModel):
    task_id: UUIDString
    output: JSONValue
    responding_agent: LogicalName | None
    status: TaskStatus
    error: JSONValue = None

class TaskSubmission(StrictModel):
    task_id: UUIDString


CLIENT_PREFIX = 'client.'
MAX_ERROR_MESSAGE_CHARS = 1000
# Kinds the runtime interprets and the topics carrying control traffic. User
# events may not use them, or a topic event could pose as a request or receipt.
RESERVED_EVENT_NAMES = frozenset(('request', 'reply', 'receipt', 'receipts', 'progress',
                                  'directory', 'error', 'errors', 'tool_call', 'tool_result', 'consult', 'handoff'))


def validate_event_topic(topic: str) -> str:
    topic = validate_name(topic)
    if topic in RESERVED_EVENT_NAMES:
        raise ValueError('public publish cannot use a reserved control kind or topic')
    return topic


def public_error(error: JSONValue) -> JSONValue:
    """The part of a journal error that may leave this process: its type only."""
    return {'type': error['type']} if isinstance(error, dict) and 'type' in error else None


def is_client(name: str | None) -> bool:
    """Clients read results from the journal and never confirm their replies."""
    return name is not None and name.startswith(CLIENT_PREFIX)


def error_payload(error: BaseException) -> dict[str, str]:
    """A bounded JSON description of a handler failure for the journal."""
    return {'type': type(error).__name__, 'message': str(error)[:MAX_ERROR_MESSAGE_CHARS]}


def event_task_id(event_id: str, agent_id: str) -> str:
    """Deterministic task identity of one agent handling one broadcast event."""
    return str(uuid5(UUID(event_id), 'task:' + agent_id))


def encode(envelope: Envelope) -> bytes:
    try:
        checked = Envelope.model_validate(envelope.model_dump(mode='python'))
        raw = json.dumps(checked.model_dump(mode='json'), ensure_ascii=False,
                         allow_nan=False, separators=(',', ':')).encode('utf-8')
    except (ValueError, TypeError, AttributeError, RecursionError) as exc:
        raise InvalidEnvelope(str(exc)) from exc
    if len(raw) > MAX_ENVELOPE_BYTES:
        raise EnvelopeTooLarge('envelope exceeds 1 MiB')
    return raw


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON object key')
        result[key] = value
    return result


def decode(raw: bytes) -> Envelope:
    if not isinstance(raw, bytes):
        raise InvalidEnvelope('wire input must be bytes')
    if len(raw) > MAX_ENVELOPE_BYTES:
        raise EnvelopeTooLarge('envelope exceeds 1 MiB')
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError('non-finite JSON')))
        if not isinstance(value, dict) or 'schema_version' not in value:
            raise ValueError('schema_version is required on the wire')
        return Envelope.model_validate(value)
    except (ValueError, TypeError, RecursionError) as exc:
        raise InvalidEnvelope(str(exc)) from exc
