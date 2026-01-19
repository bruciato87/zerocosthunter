-- Migration V15: Digital Twin / Risk Management for Real Portfolio
-- Adds Stop Loss and Take Profit tracking to the main portfolio table

DO $$ 
BEGIN 
    -- Add stop_loss column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'portfolio' AND column_name = 'stop_loss') THEN 
        ALTER TABLE portfolio ADD COLUMN stop_loss DECIMAL DEFAULT 0; 
    END IF;

    -- Add take_profit column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'portfolio' AND column_name = 'take_profit') THEN 
        ALTER TABLE portfolio ADD COLUMN take_profit DECIMAL DEFAULT 0; 
    END IF;
END $$;
