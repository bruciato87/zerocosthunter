-- Level 7.2: Distributed Lock Migration
-- Adds columns to user_settings to handle command deduplication across distributed instances via DB lock.

-- Add last_command_ts and last_command_hash to user_settings table
-- These will serve as a distributed mutex.

ALTER TABLE user_settings 
ADD COLUMN IF NOT EXISTS last_command_ts TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS last_command_hash TEXT;

-- Index for faster lookups (though user_settings is small usually)
CREATE INDEX IF NOT EXISTS idx_user_settings_last_command_ts ON user_settings(last_command_ts);

-- Comment
COMMENT ON COLUMN user_settings.last_command_ts IS 'Timestamp of the last successfully processed command to prevent duplicates';
COMMENT ON COLUMN user_settings.last_command_hash IS 'Hash of the last command content (e.g. user_id + command_text)';
