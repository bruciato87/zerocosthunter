-- Phase 2: ML Regression Model Schema Update
-- Add model_type column to distinguish classifier vs regressor models

-- Add model_type column (default 'classifier' for backward compatibility)
ALTER TABLE ml_model_state 
ADD COLUMN IF NOT EXISTS model_type VARCHAR(20) DEFAULT 'classifier';

-- Create index for efficient model type lookups
CREATE INDEX IF NOT EXISTS idx_ml_model_type ON ml_model_state(model_type, trained_at DESC);

-- Update existing rows to have model_type = 'classifier'
UPDATE ml_model_state 
SET model_type = 'classifier' 
WHERE model_type IS NULL;
