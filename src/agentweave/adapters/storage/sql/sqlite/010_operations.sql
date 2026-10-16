-- The result schema a tool call was dispatched with, so an operator resolution is checked against it.
ALTER TABLE tool_executions ADD COLUMN output_schema TEXT;

-- An operator asked to stop a task whose step was in flight: no new step may start.
ALTER TABLE tasks ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0;
-- The reply a cancel sends once the running step has ended without committing.
ALTER TABLE tasks ADD COLUMN cancel_reply TEXT;

-- When retention removed a finished task's payloads, keeping only its identity.
ALTER TABLE tasks ADD COLUMN purged_at TEXT;
CREATE INDEX tasks_retention ON tasks(status, updated_at);

-- Audit journal of operator actions: what was done, to which step, from which version.
CREATE TABLE operator_actions (
 id INTEGER PRIMARY KEY,
 task_id TEXT NOT NULL,
 action TEXT NOT NULL,
 step_id TEXT,
 expected_version INTEGER NOT NULL,
 new_version INTEGER NOT NULL,
 details TEXT NOT NULL,
 recorded_at TEXT NOT NULL
);
CREATE INDEX operator_actions_task ON operator_actions(task_id, id);

-- Durable maintenance marks, such as the newest event whose dedup ID cleanup removed.
CREATE TABLE maintenance_marks (
 name TEXT PRIMARY KEY,
 value TEXT NOT NULL
);
