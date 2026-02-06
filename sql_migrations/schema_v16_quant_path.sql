-- Migration v16: Quant Path (ML Enrichment)
-- Adds Sentiment Score and Market Regime to ML data for smarter training

-- 1. Add columns to ml_predictions (Live Inference storage)
ALTER TABLE ml_predictions
ADD COLUMN IF NOT EXISTS sentiment_score INT DEFAULT NULL,
ADD COLUMN IF NOT EXISTS market_regime TEXT DEFAULT NULL;

-- 2. Add columns to signal_tracking (Historical Training Data)
ALTER TABLE signal_tracking
ADD COLUMN IF NOT EXISTS sentiment_score INT DEFAULT NULL,
ADD COLUMN IF NOT EXISTS market_regime TEXT DEFAULT NULL;

-- 3. Comments
COMMENT ON COLUMN ml_predictions.sentiment_score IS 'AI Sentiment Confidence (0-100)';
COMMENT ON COLUMN ml_predictions.market_regime IS 'Macro Regime context (BULL, BEAR, SIDEWAYS)';
