-- =============================================================================
-- schema_v9_strategy.sql
-- Level 8: Strategy Governance - Rules of Engagement
-- =============================================================================
-- This migration creates the strategy_rules table for per-asset trading rules.
-- Rules define allocation targets, strategy types, and profit/loss thresholds.
-- =============================================================================

-- Create ENUM for strategy types (if not exists)
DO $$ BEGIN
    CREATE TYPE strategy_type AS ENUM ('ACCUMULATE', 'SWING', 'LONG_TERM');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Strategy Rules Table
CREATE TABLE IF NOT EXISTS strategy_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker VARCHAR(20) NOT NULL UNIQUE,
    
    -- Strategy Type
    strategy_type strategy_type NOT NULL DEFAULT 'ACCUMULATE',
    
    -- Allocation Targets (% of total portfolio)
    target_allocation_pct DECIMAL(5,2) DEFAULT 10.00,  -- Target: 10%
    max_allocation_cap DECIMAL(5,2) DEFAULT 20.00,     -- Hard Cap: 20%
    
    -- Profit/Loss Thresholds (%)
    take_profit_threshold DECIMAL(5,2) DEFAULT NULL,   -- e.g. +20% -> Trim
    stop_loss_threshold DECIMAL(5,2) DEFAULT NULL,     -- e.g. -15% -> Sell
    
    -- Tax Efficiency
    min_net_profit_eur DECIMAL(10,2) DEFAULT 50.00,    -- Min net profit to trigger SELL
    
    -- Metadata
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for fast ticker lookup
CREATE INDEX IF NOT EXISTS idx_strategy_rules_ticker ON strategy_rules(ticker);

-- Auto-update timestamp trigger
CREATE OR REPLACE FUNCTION update_strategy_rules_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_strategy_rules_timestamp ON strategy_rules;
CREATE TRIGGER update_strategy_rules_timestamp
    BEFORE UPDATE ON strategy_rules
    FOR EACH ROW EXECUTE FUNCTION update_strategy_rules_timestamp();

-- =============================================================================
-- Default Rules (Pre-populate with sensible defaults for common assets)
-- =============================================================================
INSERT INTO strategy_rules (ticker, strategy_type, target_allocation_pct, max_allocation_cap, take_profit_threshold, stop_loss_threshold, min_net_profit_eur, notes)
VALUES 
    -- ETFs (Long Term - Never Sell)
    ('EUNL.DE', 'LONG_TERM', 40.00, 50.00, NULL, NULL, 100.00, 'World ETF - PAC Strategy'),
    
    -- Crypto (Swing Trading)
    ('BTC-USD', 'SWING', 10.00, 15.00, 30.00, -20.00, 50.00, 'Bitcoin - Swing with Take Profit'),
    ('ETH-USD', 'SWING', 10.00, 15.00, 30.00, -20.00, 50.00, 'Ethereum - Swing with Take Profit'),
    ('SOL-USD', 'ACCUMULATE', 5.00, 10.00, 50.00, -25.00, 30.00, 'Solana - Higher risk'),
    
    -- Tech Stocks (Accumulate)
    ('AAPL', 'ACCUMULATE', 5.00, 10.00, 25.00, -15.00, 50.00, 'Apple - Core holding'),
    ('GOOGL', 'ACCUMULATE', 5.00, 10.00, 25.00, -15.00, 50.00, 'Google - Core holding'),
    ('NVDA', 'SWING', 5.00, 10.00, 20.00, -15.00, 50.00, 'Nvidia - High volatility'),
    ('META', 'ACCUMULATE', 5.00, 10.00, 20.00, -15.00, 50.00, 'Meta - Core holding')
ON CONFLICT (ticker) DO NOTHING;

-- =============================================================================
-- RLS Policies (Optional - Enable if using Supabase Auth)
-- =============================================================================
-- ALTER TABLE strategy_rules ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "Allow all for authenticated users" ON strategy_rules FOR ALL USING (true);

-- =============================================================================
-- Grant Permissions (for Supabase anon key)
-- =============================================================================
GRANT SELECT, INSERT, UPDATE, DELETE ON strategy_rules TO anon;
GRANT SELECT, INSERT, UPDATE, DELETE ON strategy_rules TO authenticated;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO anon;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO authenticated;
