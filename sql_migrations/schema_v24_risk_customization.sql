-- Migration V24: Risk Customization & Core Assets
-- Add risk_profile to user_settings
ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS risk_profile TEXT DEFAULT 'BALANCED';

-- Add is_core to portfolio
ALTER TABLE portfolio ADD COLUMN IF NOT EXISTS is_core BOOLEAN DEFAULT FALSE;
