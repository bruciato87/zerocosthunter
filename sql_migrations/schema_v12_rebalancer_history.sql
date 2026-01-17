-- Migration v12: Rebalancer Learning System
-- Tracks AI suggestions and their outcomes for continuous learning

CREATE TABLE IF NOT EXISTS rebalancer_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- AI Suggestion
    ticker TEXT NOT NULL,
    action TEXT NOT NULL,           -- 'BUY', 'SELL', 'HOLD', 'TRIM', 'ACCUMULATE'
    suggested_amount_eur DECIMAL,
    confidence DECIMAL,
    reasoning TEXT,
    
    -- Market Context at suggestion time
    regime TEXT,                    -- 'BULL', 'BEAR', 'SIDEWAYS'
    sector_rotation TEXT,           -- 'RISK_ON', 'RISK_OFF'
    ticker_rsi DECIMAL,
    ticker_pnl_pct DECIMAL,
    portfolio_value_eur DECIMAL,
    
    -- Outcome tracking (updated later)
    was_executed BOOLEAN DEFAULT NULL,
    executed_at TIMESTAMP WITH TIME ZONE,
    price_at_suggestion DECIMAL,
    price_after_7d DECIMAL,
    price_after_30d DECIMAL,
    outcome_pnl_pct DECIMAL,
    
    -- Learning score
    was_good_advice BOOLEAN DEFAULT NULL  -- true if followed AND profitable
);

-- Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_rebalancer_history_ticker ON rebalancer_history(ticker);
CREATE INDEX IF NOT EXISTS idx_rebalancer_history_created ON rebalancer_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rebalancer_history_action ON rebalancer_history(action);

COMMENT ON TABLE rebalancer_history IS 'Tracks AI rebalancing suggestions and their outcomes for self-learning improvement.';
