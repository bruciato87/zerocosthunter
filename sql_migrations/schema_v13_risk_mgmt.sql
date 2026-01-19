-- Migration v13: Risk Management (The Trader Path)

-- 1. Create Paper Accounts table (Cash Balance)
CREATE TABLE IF NOT EXISTS paper_accounts (
    chat_id BIGINT PRIMARY KEY, -- Telegram Chat ID (linked to user)
    cash_balance DECIMAL(18, 2) DEFAULT 10000.00, -- Start with 10k virtual cash
    total_deposit DECIMAL(18, 2) DEFAULT 10000.00, -- For ROI calculation
    last_trade_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Add Stop Loss and Take Profit to Paper Portfolio
ALTER TABLE paper_portfolio 
ADD COLUMN IF NOT EXISTS stop_loss DECIMAL(18, 2) DEFAULT NULL,
ADD COLUMN IF NOT EXISTS take_profit DECIMAL(18, 2) DEFAULT NULL,
ADD COLUMN IF NOT EXISTS entry_notes TEXT DEFAULT NULL;

-- 3. Comments for documentation
COMMENT ON TABLE paper_accounts IS 'Tracks virtual cash balance for paper trading';
COMMENT ON COLUMN paper_portfolio.stop_loss IS 'Price at which to auto-sell to limit loss';
COMMENT ON COLUMN paper_portfolio.take_profit IS 'Price at which to auto-sell to secure profit';
