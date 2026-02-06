-- Add 'is_confirmed' column to handle Draft state from Telegram
ALTER TABLE portfolio ADD COLUMN IF NOT EXISTS is_confirmed boolean DEFAULT true;

-- Optional: Add a 'chat_id' to know who owns the draft (for multi-user support in future)
ALTER TABLE portfolio ADD COLUMN IF NOT EXISTS chat_id bigint;

-- Update RLS to only show confirmed items to the 'public' (the analysis engine)
-- Dropping old policy to replace it
DROP POLICY IF EXISTS "Allow public read access" ON portfolio;

CREATE POLICY "Allow public read access" ON portfolio
  FOR SELECT USING (is_confirmed = true);

-- Service role still has full access to write drafts
