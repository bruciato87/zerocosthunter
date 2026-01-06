-- Transaction Tracking Table for Zero Cost Hunter
-- Run this in Supabase SQL Editor

CREATE TABLE IF NOT EXISTS transactions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL CHECK (action IN ('BUY', 'SELL')),
    quantity DECIMAL(18,8) NOT NULL,
    price_per_unit DECIMAL(18,4) NOT NULL,
    total_value DECIMAL(18,4) NOT NULL,
    realized_pnl DECIMAL(18,4),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_transactions_ticker ON transactions(ticker);
CREATE INDEX IF NOT EXISTS idx_transactions_action ON transactions(action);
CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions(created_at DESC);

-- Enable RLS (optional, for multi-user)
-- ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;
