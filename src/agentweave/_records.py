"""Shared storage records; no backend or runtime imports."""
from datetime import datetime, timezone
from typing import Literal, Protocol
from pydantic import Field, field_validator, model_validator
from .messages import StrictModel, UUIDString, LogicalName, UTCDateTime, JSONValue, TaskStatus, Message, Envelope

class Clock(Protocol):
    def now(self) -> datetime: ...

class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class Participant(StrictModel):
    """Public per-instance announcement; never serialize execution configuration."""
    schema_version: Literal[1] = 1
    network_id: UUIDString
    logical_id: LogicalName
    instance_id: UUIDString
    kind: Literal['agent', 'tool', 'toolbox']
    fingerprint: str = Field(min_length=1)
    configuration_version: str
    broker_epoch: str
    heartbeat_at: UTCDateTime
    description: str = ''
    capabilities: list[LogicalName] = []
    name: str = ''
    revision: str | None = None
    input_schema: dict[str, JSONValue] | None = None
    output_schema: dict[str, JSONValue] | None = None
    allowed_callers: list[LogicalName] | None = None

    @field_validator('schema_version', mode='before')
    @classmethod
    def valid_version(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError('unsupported participant schema_version')
        return value

    @model_validator(mode='after')
    def tool_schema(self):
        if self.kind == 'tool':
            if not self.revision or self.input_schema is None or self.output_schema is None:
                raise ValueError('tool requires revision, input_schema and output_schema')
        elif any(x is not None for x in (self.revision, self.input_schema, self.output_schema, self.allowed_callers)):
            raise ValueError('tool schemas and allowed_callers require kind=tool')
        return self

    @property
    def alias(self) -> str:
        return self.logical_id.replace('.', '_').replace('-', '_')

class TaskRecord(StrictModel):
    task_id: UUIDString
    agent_id: LogicalName
    status: TaskStatus
    version: int = Field(ge=0)
    input: JSONValue
    session_id: UUIDString | None
    root_task_id: UUIDString
    definition_hash: str
    deadline: UTCDateTime
    result: JSONValue = None
    error: JSONValue = None
    admitted_at: UTCDateTime | None = None
    waiting_for: JSONValue = None  # what a waiting_peer task waits for: its consult children
    transfer_context: JSONValue = None  # what a handed-off task brings to its owner: messages, root budget
    handoffs: int = Field(default=0, ge=0)

class ConversationRecord(StrictModel):
    """A session with one answering agent at a time; a handoff moves active_agent."""
    session_id: UUIDString
    active_agent: LogicalName
    version: int = Field(ge=0)
    current_task_id: UUIDString | None = None


class Lease(StrictModel):
    task_id: UUIDString
    instance_id: UUIDString
    token: int = Field(ge=1)
    expires_at: UTCDateTime

class Admission(StrictModel):
    duplicate: bool
    task_id: UUIDString
    receipt: Envelope

class StepCommit(StrictModel):
    step_id: str
    expected_version: int = Field(ge=0)
    status: TaskStatus
    result: JSONValue = None
    state: dict[str, JSONValue] | None = None
    messages: list[Message] = []
    outgoing: list[Envelope] = []
    waiting_for: JSONValue = None
    error: JSONValue = None

class OutboxRecord(StrictModel):
    envelope: Envelope
    targets: list[LogicalName]
    received_targets: list[LogicalName] = []
    published_at: UTCDateTime | None = None
    retry_at: UTCDateTime | None = None
    lease_token: str | None = None
    version: int = Field(default=0, ge=0)


class EventRejection(StrictModel):
    reason: str
    retryable: bool


class ToolLease(StrictModel):
    """One tool call's journal row; status running means an effect may be in progress."""
    tool_call_id: UUIDString
    token: int = Field(ge=0)
    attempt: int = Field(ge=0)
    status: Literal['running', 'retry', 'succeeded', 'failed', 'expired', 'outcome_unknown']
    arguments: dict[str, JSONValue]
    requested: JSONValue = None  # the call as first requested: its identity
    retry_safe: bool = False
    max_attempts: int = Field(default=1, ge=1)
    result: JSONValue = None
    error: JSONValue = None
    hook_status: Literal['succeeded', 'failed'] | None = None
    hook_result: JSONValue = None
    hook_error: JSONValue = None


class ModelAttempt(StrictModel):
    """One sent model request; a completed attempt carries its whole turn for replay."""
    attempt_id: UUIDString
    task_id: UUIDString
    root_task_id: UUIDString
    step: int = Field(ge=0)
    attempt: int = Field(ge=1)
    request_hash: str
    status: Literal['running', 'completed', 'failed', 'unknown']
    turn: JSONValue = None
    error: JSONValue = None


class ModelUsage(StrictModel):
    """Root-task model spending; unknown counts are never folded into the totals."""
    calls: int = 0
    tokens: int = 0
    unknown_usage: int = 0
    cost_usd: str = '0'
    unknown_cost: int = 0
