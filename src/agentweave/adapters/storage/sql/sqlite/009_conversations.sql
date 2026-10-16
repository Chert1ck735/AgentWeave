-- A conversation is a session with one active agent at a time, and a handoff
-- moves that ownership in the same transaction as the task's new owner.
CREATE TABLE conversations(
    session_id TEXT PRIMARY KEY,
    active_agent TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 0,
    current_task_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- What a handed-off task carries to its next owner, and how often it moved.
ALTER TABLE tasks ADD COLUMN transfer_context TEXT NOT NULL DEFAULT 'null';
ALTER TABLE tasks ADD COLUMN handoffs INTEGER NOT NULL DEFAULT 0;
