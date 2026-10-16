-- One row per model request actually sent. A completed row holds the whole
-- turn, so recovery replays it instead of asking the model again. Usage that
-- was not reported stays unknown (usage_known=0) and is never counted as zero.
-- Running rows with a live lease hold a slot of their provider pool.
CREATE TABLE model_attempts (
 attempt_id TEXT PRIMARY KEY,
 task_id TEXT NOT NULL REFERENCES tasks(task_id),
 root_task_id TEXT NOT NULL,
 step INTEGER NOT NULL,
 attempt INTEGER NOT NULL,
 request_hash TEXT NOT NULL,
 pool_name TEXT NOT NULL,
 status TEXT NOT NULL CHECK (status IN ('running','completed','failed','unknown')),
 instance_id TEXT NOT NULL,
 lease_expires_at TEXT NOT NULL,
 turn TEXT NOT NULL DEFAULT 'null',
 input_tokens INTEGER,
 output_tokens INTEGER,
 usage_known INTEGER NOT NULL DEFAULT 0,
 cost_usd TEXT,
 pricing_revision TEXT,
 error TEXT NOT NULL DEFAULT 'null',
 created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL,
 UNIQUE(task_id, step, attempt)
);
CREATE INDEX model_attempts_root ON model_attempts(root_task_id);
CREATE INDEX model_attempts_pool ON model_attempts(pool_name, status, lease_expires_at);
