CREATE TABLE logical_definitions (
 logical_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL
);
CREATE TABLE definition_instances (
 logical_id TEXT NOT NULL REFERENCES logical_definitions(logical_id), instance_id TEXT NOT NULL,
 fingerprint TEXT NOT NULL, expires_at TEXT NOT NULL, PRIMARY KEY(logical_id,instance_id)
);
CREATE TABLE participant_announcements (
 logical_id TEXT NOT NULL, instance_id TEXT NOT NULL, snapshot TEXT NOT NULL,
 PRIMARY KEY(logical_id,instance_id),
 FOREIGN KEY(logical_id,instance_id) REFERENCES definition_instances(logical_id,instance_id) ON DELETE CASCADE
);
