-- Migration v18: Ticker Cache Expansion
-- Adds missing columns used by DBHandler but not present in initial schema

ALTER TABLE ticker_cache ADD COLUMN IF NOT EXISTS last_price FLOAT;
ALTER TABLE ticker_cache ADD COLUMN IF NOT EXISTS last_price_at TIMESTAMP WITH TIME ZONE;

-- Add comment
COMMENT ON COLUMN ticker_cache.last_price IS 'Last discovered price (Smart Price EUR)';
COMMENT ON COLUMN ticker_cache.last_price_at IS 'Timestamp of the last price discovery';
