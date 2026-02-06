
-- Schema v19: Add price caching to ticker_cache
ALTER TABLE ticker_cache 
ADD COLUMN IF NOT EXISTS last_price numeric,
ADD COLUMN IF NOT EXISTS last_price_at timestamptz;
