-- Level 4: Machine Learning Schema
-- Tables for ML predictions tracking and model state

-- ML Predictions Table: Track all predictions for accuracy measurement
CREATE TABLE IF NOT EXISTS ml_predictions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    predicted_direction VARCHAR(10),  -- UP, DOWN, HOLD
    ml_confidence FLOAT,
    features JSONB,  -- Store feature values for debugging
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for ticker lookups
CREATE INDEX IF NOT EXISTS idx_ml_predictions_ticker ON ml_predictions(ticker);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_created ON ml_predictions(created_at DESC);

-- ML Model State: Track model versions, accuracy, and weights
CREATE TABLE IF NOT EXISTS ml_model_state (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    accuracy FLOAT,
    samples_count INT,
    model_weights TEXT,  -- Serialized model JSON for Pure Python GB
    trained_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for version lookups
CREATE INDEX IF NOT EXISTS idx_ml_model_version ON ml_model_state(trained_at DESC);

-- Grant permissions (adjust as needed for your Supabase setup)
-- ALTER TABLE ml_predictions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE ml_model_state ENABLE ROW LEVEL SECURITY;
