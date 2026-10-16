-- The discovered tool catalog a model step was built from. The first writer
-- wins, so replaying a step after a restart offers the model the same tools
-- and revisions even when the live directory has changed since.
CREATE TABLE model_step_catalogs(
    task_id TEXT NOT NULL REFERENCES tasks(task_id),
    step INTEGER NOT NULL,
    catalog TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY(task_id, step)
);
