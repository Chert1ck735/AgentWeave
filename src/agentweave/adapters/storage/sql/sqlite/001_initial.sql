CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
CREATE TABLE definitions (definition_hash TEXT PRIMARY KEY, snapshot TEXT NOT NULL);
CREATE TABLE participants (instance_id TEXT PRIMARY KEY, logical_id TEXT NOT NULL, snapshot TEXT NOT NULL);
CREATE TABLE tasks (
 task_id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, status TEXT NOT NULL
 CHECK(status IN ('submitted','accepted','running','waiting_peer','waiting_tool','outcome_unknown','succeeded','failed','cancelled','expired')),
 version INTEGER NOT NULL CHECK(version>=0), input TEXT NOT NULL, input_envelope TEXT NOT NULL,
 session_id TEXT, root_task_id TEXT NOT NULL, definition_hash TEXT NOT NULL, deadline TEXT NOT NULL,
 result TEXT NOT NULL DEFAULT 'null', error TEXT NOT NULL DEFAULT 'null', waiting_for TEXT NOT NULL DEFAULT 'null',
 lease_instance_id TEXT, lease_token INTEGER NOT NULL DEFAULT 0 CHECK(lease_token>=0),
 lease_expires_at TEXT, lease_ttl REAL, updated_at TEXT NOT NULL
);
CREATE INDEX tasks_recovery ON tasks(agent_id,status,lease_expires_at);
CREATE TABLE sessions (
 agent_id TEXT NOT NULL, session_id TEXT NOT NULL, state TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL,
 PRIMARY KEY(agent_id,session_id)
);
CREATE TABLE steps (
 task_id TEXT NOT NULL REFERENCES tasks(task_id), step_id TEXT NOT NULL,
 version INTEGER NOT NULL, change TEXT NOT NULL, committed_at TEXT NOT NULL, PRIMARY KEY(task_id,step_id)
);
CREATE TABLE messages (
 id INTEGER PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(task_id), session_id TEXT,
 agent_id TEXT NOT NULL, step_id TEXT NOT NULL, message TEXT NOT NULL,
 FOREIGN KEY(task_id,step_id) REFERENCES steps(task_id,step_id)
);
CREATE TABLE tool_calls (tool_call_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(task_id), record TEXT NOT NULL);
CREATE TABLE task_events (
 id INTEGER PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(task_id), kind TEXT NOT NULL,
 version INTEGER NOT NULL, recorded_at TEXT NOT NULL
);
CREATE TABLE processed_events (
 agent_id TEXT NOT NULL, event_id TEXT NOT NULL, task_id TEXT NOT NULL REFERENCES tasks(task_id),
 receipt_event_id TEXT NOT NULL, PRIMARY KEY(agent_id,event_id)
);
CREATE TABLE outbox (
 event_id TEXT PRIMARY KEY, envelope TEXT NOT NULL, targets TEXT NOT NULL, received_targets TEXT NOT NULL DEFAULT '[]',
 published_at TEXT, retry_at TEXT, deadline TEXT NOT NULL, lease_instance_id TEXT, lease_token TEXT,
 lease_expires_at TEXT, version INTEGER NOT NULL DEFAULT 0 CHECK(version>=0)
);
CREATE INDEX outbox_ready ON outbox(retry_at,lease_expires_at,deadline);
CREATE TABLE receipts (
 agent_id TEXT NOT NULL, event_id TEXT NOT NULL, receipt_event_id TEXT NOT NULL UNIQUE REFERENCES outbox(event_id),
 envelope TEXT NOT NULL, PRIMARY KEY(agent_id,event_id),
 FOREIGN KEY(agent_id,event_id) REFERENCES processed_events(agent_id,event_id)
);
CREATE TABLE submissions (
 task_id TEXT PRIMARY KEY, event_id TEXT NOT NULL UNIQUE REFERENCES outbox(event_id), submitted_at TEXT NOT NULL
);
