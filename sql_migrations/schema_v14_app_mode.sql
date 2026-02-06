-- Migration v14: App Mode & Settings Repair

-- 1. Ensure Extension for UUIDs (if needed for ID generation)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. Ensure Table Exists
CREATE TABLE IF NOT EXISTS user_settings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    min_confidence DECIMAL(5, 2) DEFAULT 0.70,
    only_portfolio BOOLEAN DEFAULT FALSE,
    app_mode VARCHAR(20) DEFAULT 'PROD',
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Ensure 'app_mode' column exists (safe add)
ALTER TABLE user_settings 
ADD COLUMN IF NOT EXISTS app_mode VARCHAR(20) DEFAULT 'PROD';

-- 4. Seed Default Row (Critical: Brain needs 1 row to read)
INSERT INTO user_settings (min_confidence, only_portfolio, app_mode)
SELECT 0.70, FALSE, 'PROD'
WHERE NOT EXISTS (SELECT 1 FROM user_settings);

-- Comments
COMMENT ON COLUMN user_settings.app_mode IS 'PROD (Hybrid) or PREPROD (Gemini Only)';
