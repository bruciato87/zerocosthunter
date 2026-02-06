-- Create user_settings table
CREATE TABLE IF NOT EXISTS user_settings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID, -- Optional linkage if we had auth, but single user for now
    min_confidence DECIMAL(3,2) DEFAULT 0.70, -- Filter signals below 70%
    only_portfolio BOOLEAN DEFAULT FALSE, -- If TRUE, only alerts for owned assets
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default settings row if not exists (Single User Mode)
INSERT INTO user_settings (min_confidence, only_portfolio)
SELECT 0.70, FALSE
WHERE NOT EXISTS (SELECT 1 FROM user_settings);
