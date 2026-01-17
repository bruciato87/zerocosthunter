-- Migration v11: Self-Learning Ticker Cache
-- Automatically caches successful ticker resolutions to avoid repeated 404s

CREATE TABLE IF NOT EXISTS ticker_cache (
    user_ticker TEXT PRIMARY KEY,           -- What user enters (e.g., "JAZZ", "TCT")
    resolved_ticker TEXT NOT NULL,          -- What works on Yahoo (e.g., "JAZZ", "0700.HK")
    source TEXT DEFAULT 'yahoo',            -- 'yahoo', 'coingecko', etc.
    is_crypto BOOLEAN DEFAULT FALSE,
    currency TEXT DEFAULT 'USD',            -- USD, EUR, HKD, etc.
    last_verified_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    fail_count INTEGER DEFAULT 0,           -- For cache invalidation
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_ticker_cache_user ON ticker_cache(user_ticker);

-- Comment
COMMENT ON TABLE ticker_cache IS 'Auto-learning cache for ticker resolution. Avoids repeated 404 errors by remembering successful ticker formats.';
