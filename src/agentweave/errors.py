"""Stable errors at the declaration and wire boundaries."""

class AgentWeaveError(Exception):
    """Base SDK error."""

class InvalidEnvelope(AgentWeaveError, ValueError):
    """Malformed or unsupported wire data."""

class EnvelopeTooLarge(InvalidEnvelope):
    """UTF-8 envelope exceeds the 1 MiB transport limit."""

class ConfigError(AgentWeaveError, ValueError):
    """Invalid or incomplete project configuration."""

class DefinitionError(AgentWeaveError, ValueError):
    """Invalid or ambiguous participant definition."""

class CapabilityMismatch(ConfigError):
    """The model cannot do what the definition asks of it, such as calling tools."""

class DefinitionExists(ConfigError):
    """A configured definition already has that name; replacing it must be explicit."""

class StoreUnavailable(AgentWeaveError):
    """Persistence failed; callers must not acknowledge unsaved input."""

class StoreConflict(AgentWeaveError):
    """An identity or journal invariant conflicts with durable data."""

class ReceiptNotExpected(StoreConflict):
    """Valid receipt references no pending target in this store."""

class LeaseLost(StoreConflict):
    """Execution ownership expired or was replaced by a newer fence."""

class VersionConflict(StoreConflict):
    """The task changed since the proposed step was prepared."""

class BrokerUnavailable(AgentWeaveError):
    """Temporary broker transport failure; durable rows remain pending."""

class InputExpired(AgentWeaveError, TimeoutError):
    """The input deadline or freshness passed; no new effect may start.

    It subclasses TimeoutError, so a handler catching TimeoutError around
    ctx.publish also catches it; re-raise it to report the task as expired.
    """

class StaleDelivery(AgentWeaveError, ValueError):
    """A rebalance, epoch change or close invalidated this delivery position.

    The broker redelivers the record to the current owner, so a caller whose
    input is already durably admitted may ignore this error.
    """


class ModelError(AgentWeaveError):
    """A provider request failed; kind is stable across adapters.

    kind is one of auth, quota, invalid_request, unsupported, incomplete,
    rate_limit, unavailable, timeout or exhausted. retryable tells the loop
    whether another attempt may help. usage carries reported token counts of a
    billed failure; usage_unknown means the request may have been billed.
    """
    def __init__(self, kind: str, message: str, *, retryable: bool = False, usage_unknown: bool = False,
                 usage: dict | None = None):
        super().__init__(message)
        self.kind, self.retryable, self.usage_unknown, self.usage = kind, retryable, usage_unknown, usage

class ModelCapacity(AgentWeaveError):
    """The shared provider pool is full; wait and reserve again."""

class ModelBudgetExceeded(AgentWeaveError):
    """The root task's call, token or cost budget allows no further attempt."""


class ToolUnavailable(AgentWeaveError):
    """No live executor took the remote tool call before its deadline; it never ran."""


class ToolRevisionChanged(AgentWeaveError):
    """The tool's schema changed after the caller chose it; the call ran no effect.

    Refresh the catalog and call again with the current revision.
    """


class ToolCallFailed(AgentWeaveError):
    """A remote tool call ended without a result; result is its ToolResult."""
    def __init__(self, result):
        super().__init__(f'tool call ended with status {result.status}')
        self.result = result


class ToolArgumentsInvalid(AgentWeaveError, ValueError):
    """Tool arguments break the tool's declared input schema; nothing was sent."""


class ToolNotStarted(AgentWeaveError):
    """The tool could not start its effect, e.g. a missing secret or an unreachable host.

    The executor records it as failed, never as outcome_unknown.
    """


class ToolRequestRejected(AgentWeaveError):
    """The endpoint answered and refused the request (HTTP 3xx or 4xx); status is its code."""
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class ToolResponseInvalid(AgentWeaveError):
    """The endpoint answered, but not with a bounded strict JSON body; it may have acted."""


class ConsultRefused(AgentWeaveError):
    """A consult was refused before anything was sent: peer not allowed, unavailable or too deep."""


class CycleDetected(ConsultRefused):
    """The consulted agent is already waiting further up the same chain of consults."""


class HandoffRefused(AgentWeaveError):
    """A handoff was refused before the owner changed: recipient, limit, size or scope."""


class OperatorActionRefused(AgentWeaveError, ValueError):
    """An operator action cannot apply to the task as it is; nothing was changed."""


class TaskCancelled(AgentWeaveError):
    """An operator cancelled the task: no new step may start."""


class ReplayExpired(AgentWeaveError):
    """The event is older than the replay window or its dedup ID may already be gone; it is not run."""


class ConversationNotFound(AgentWeaveError, ValueError):
    """No conversation has that ID; a new one is opened only on request."""
