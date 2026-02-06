-- Migration v21: Position Targets & Strategic Alignment
-- Adds columns for dynamic and manual exit targets.

DO $$
BEGIN
    -- 1. Target Price (Profit Goal)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'portfolio' AND column_name = 'target_price') THEN
        ALTER TABLE public.portfolio ADD COLUMN target_price numeric;
    END IF;

    -- 2. Stop Loss Price (Renaming SL if already exists or adding)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'portfolio' AND column_name = 'stop_loss_price') THEN
        ALTER TABLE public.portfolio ADD COLUMN stop_loss_price numeric;
    END IF;

    -- 3. Target Horizon in Days
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'portfolio' AND column_name = 'target_horizon_days') THEN
        ALTER TABLE public.portfolio ADD COLUMN target_horizon_days integer DEFAULT 30;
    END IF;

    -- 4. Entry Reason / Investment Thesis
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'portfolio' AND column_name = 'entry_reason') THEN
        ALTER TABLE public.portfolio ADD COLUMN entry_reason text;
    END IF;

    -- 5. Track if target was set automatically
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'portfolio' AND column_name = 'target_type') THEN
        ALTER TABLE public.portfolio ADD COLUMN target_type varchar(20) DEFAULT 'AUTO'; -- 'AUTO' or 'MANUAL'
    END IF;

END $$;
