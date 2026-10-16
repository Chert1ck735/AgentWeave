CREATE TABLE source_watermarks (
 agent_id TEXT NOT NULL, sender TEXT NOT NULL, topic TEXT NOT NULL, key TEXT NOT NULL,
 source_version INTEGER NOT NULL, PRIMARY KEY(agent_id,sender,topic,key)
);
CREATE TABLE execution_gates (
 subscription TEXT NOT NULL, key TEXT NOT NULL, task_id TEXT NOT NULL REFERENCES tasks(task_id),
 lease_token INTEGER NOT NULL, PRIMARY KEY(subscription,key)
);
CREATE TABLE handler_attempts (
 task_id TEXT PRIMARY KEY REFERENCES tasks(task_id), lease_token INTEGER NOT NULL,
 retry_safe INTEGER NOT NULL, attempts INTEGER NOT NULL DEFAULT 1
);
ALTER TABLE task_events ADD COLUMN event_id TEXT;
ALTER TABLE task_events ADD COLUMN envelope TEXT;
CREATE UNIQUE INDEX task_progress_identity ON task_events(task_id,event_id);
CREATE INDEX runtime_recovery_page ON tasks(agent_id,task_id);

CREATE TABLE submission_outcomes (
 task_id TEXT PRIMARY KEY REFERENCES submissions(task_id), result TEXT NOT NULL
);

ALTER TABLE tasks ADD COLUMN admitted_at TEXT;
UPDATE tasks SET admitted_at=COALESCE((SELECT recorded_at FROM task_events WHERE task_events.task_id=tasks.task_id AND kind='accepted' ORDER BY id LIMIT 1),updated_at);
