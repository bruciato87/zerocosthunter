-- Migration: Add Critic Agent fields to predictions table
-- Date: 2026-01-20
-- Author: Antigravity

ALTER TABLE predictions
ADD COLUMN critic_verdict VARCHAR(20), -- 'APPROVE', 'REJECT', 'SKIPPED'
ADD COLUMN critic_score INT,           -- 0-100 Quality Score
ADD COLUMN critic_reasoning TEXT;      -- Risk Manager's notes
