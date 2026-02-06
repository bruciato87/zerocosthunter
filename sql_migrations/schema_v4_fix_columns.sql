-- Fix missing columns in portfolio table
-- The original schema might have missed avg_price or used a different name. 
-- This migration ensures columns exist and are correct type.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'portfolio' AND column_name = 'avg_price') THEN
        ALTER TABLE public.portfolio ADD COLUMN avg_price numeric DEFAULT 0;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'portfolio' AND column_name = 'quantity') THEN
        ALTER TABLE public.portfolio ADD COLUMN quantity numeric DEFAULT 0;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'portfolio' AND column_name = 'sector') THEN
        ALTER TABLE public.portfolio ADD COLUMN sector text DEFAULT 'Unknown';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'portfolio' AND column_name = 'is_confirmed') THEN
        ALTER TABLE public.portfolio ADD COLUMN is_confirmed boolean DEFAULT true;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'portfolio' AND column_name = 'chat_id') THEN
        ALTER TABLE public.portfolio ADD COLUMN chat_id bigint;
    END IF;
END $$;
