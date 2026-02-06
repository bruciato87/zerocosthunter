-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table: Portfolio
-- Stores the user's current holdings
CREATE TABLE IF NOT EXISTS portfolio (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(20) NOT NULL UNIQUE,
    quantity NUMERIC(18, 8) DEFAULT 0,
    average_buy_price NUMERIC(18, 2),
    current_price NUMERIC(18, 2), -- Optional cache
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table: Predictions
-- Stores AI analysis results for tickers
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(20) NOT NULL,
    sentiment VARCHAR(10) CHECK (sentiment IN ('BUY', 'SELL', 'HOLD', 'ACCUMULATE', 'PANIC SELL', 'WATCH', 'AVOID')),
    reasoning TEXT,
    prediction_sentence TEXT,
    confidence_score NUMERIC(3, 2), -- 0.00 to 1.00
    source_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table: Logs
-- General application logs for stateless debugging
CREATE TABLE IF NOT EXISTS logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    level VARCHAR(10) CHECK (level IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    module VARCHAR(50),
    message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for faster queries on predictions by ticker and date
CREATE INDEX IF NOT EXISTS idx_predictions_ticker_date ON predictions (ticker, created_at);
