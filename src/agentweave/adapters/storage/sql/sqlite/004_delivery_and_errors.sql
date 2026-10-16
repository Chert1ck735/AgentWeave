-- Pending outbox rows are found through a partial index instead of scanning
-- every row whose deadline has not passed. A row is delivered after a broker
-- ACK when no receipt is expected, or once every target confirmed it.
ALTER TABLE outbox ADD COLUMN delivered INTEGER NOT NULL DEFAULT 0;
-- Clients read results from the journal and never confirm replies.
UPDATE outbox SET targets='[]' WHERE json_extract(envelope,'$.kind')='reply'
  AND json_extract(envelope,'$.recipient') LIKE 'client.%';
UPDATE outbox SET delivered=1 WHERE published_at IS NOT NULL AND targets='[]';
UPDATE outbox SET delivered=1 WHERE targets<>'[]' AND NOT EXISTS (
  SELECT 1 FROM json_each(outbox.targets) AS target
  WHERE target.value NOT IN (SELECT value FROM json_each(outbox.received_targets))
);
CREATE INDEX outbox_pending ON outbox(delivered) WHERE delivered=0;

-- The last failure of a retry-safe handler survives until retries are exhausted.
ALTER TABLE handler_attempts ADD COLUMN last_error TEXT;

-- Session history and session gates look tasks and messages up by session.
CREATE INDEX messages_session ON messages(session_id,id);
CREATE INDEX tasks_session ON tasks(session_id);
