-- Migration v10: Trailing Stop Support

-- 1. Add trailing_stop_pct to strategy_rules
ALTER TABLE strategy_rules 
ADD COLUMN IF NOT EXISTS trailing_stop_pct DECIMAL(5, 2) DEFAULT NULL;

-- 2. Add high_watermark_price to signal_tracking for trailing calculation
ALTER TABLE signal_tracking
ADD COLUMN IF NOT EXISTS high_watermark_price DECIMAL(10, 2) DEFAULT NULL;

-- 3. Update existing open signals to set high_watermark = entry_price (init)
UPDATE signal_tracking 
SET high_watermark_price = entry_price 
WHERE status = 'OPEN' AND high_watermark_price IS NULL;

-- 4. Comment on columns
COMMENT ON COLUMN strategy_rules.trailing_stop_pct IS 'Percentage drop from High Watermark to trigger sell (e.g. 5.0 for 5%)';
COMMENT ON COLUMN signal_tracking.high_watermark_price IS 'Highest price reached since trade entry (for trailing stop)';
