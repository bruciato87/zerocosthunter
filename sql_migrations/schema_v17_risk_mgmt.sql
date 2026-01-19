-- Migration v17: Risk Management (SL/TP)
-- Adds Stop Loss and Take Profit columns to prediction/signal tables

-- 1. Add columns to predictions (Live Signals)
ALTER TABLE predictions
ADD COLUMN IF NOT EXISTS stop_loss FLOAT DEFAULT NULL,
ADD COLUMN IF NOT EXISTS take_profit FLOAT DEFAULT NULL;

-- 2. Add columns to signal_tracking (Historical Tracking)
ALTER TABLE signal_tracking
ADD COLUMN IF NOT EXISTS stop_loss FLOAT DEFAULT NULL,
ADD COLUMN IF NOT EXISTS take_profit FLOAT DEFAULT NULL;

COMMENT ON COLUMN predictions.stop_loss IS 'ATR-based Stop Loss';
COMMENT ON COLUMN predictions.take_profit IS 'ATR-based Take Profit';
