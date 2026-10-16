-- One row per tool call: the requested call is its identity, the effective
-- arguments are what a before hook approved, and status records whether an
-- external effect may have started. A row stays running from the start of an
-- attempt until its outcome commits, so a crash in between is detectable.
CREATE TABLE tool_executions (
 tool_call_id TEXT PRIMARY KEY,
 task_id TEXT NOT NULL REFERENCES tasks(task_id),
 tool_id TEXT NOT NULL,
 revision TEXT NOT NULL,
 requested TEXT NOT NULL,
 arguments TEXT NOT NULL,
 retry_safe INTEGER NOT NULL,
 max_attempts INTEGER NOT NULL,
 status TEXT NOT NULL CHECK (status IN ('running','retry','succeeded','failed','expired','outcome_unknown')),
 attempts INTEGER NOT NULL,
 token INTEGER NOT NULL,
 instance_id TEXT NOT NULL,
 result TEXT NOT NULL DEFAULT 'null',
 error TEXT NOT NULL DEFAULT 'null',
 hook_status TEXT CHECK (hook_status IS NULL OR hook_status IN ('succeeded','failed')),
 hook_result TEXT NOT NULL DEFAULT 'null',
 hook_error TEXT NOT NULL DEFAULT 'null',
 updated_at TEXT NOT NULL
);
CREATE INDEX tool_executions_task ON tool_executions(task_id);
