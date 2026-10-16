-- A started handler attempt stays 'started' until a known failure is recorded,
-- so an interrupted attempt can be told apart from a reported one. Existing
-- rows conservatively count as started: their outcome was never journaled.
ALTER TABLE handler_attempts ADD COLUMN last_outcome TEXT NOT NULL DEFAULT 'started';

-- A released retry may not be claimed by any replica before this time.
ALTER TABLE tasks ADD COLUMN not_before TEXT;
