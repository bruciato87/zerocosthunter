-- SCHEMA V15: SECURITY HARDENING (Enable RLS)
-- Sanates "RLS Disabled" vulnerabilities detected by Supabase/GitGuardian.
-- Enabling RLS restricts access to only:
-- 1. Service Role Key (used by our Python Bot) - BYPASSES RLS (Full Access)
-- 2. Postgres Admin - Full Access
-- 3. Users explicitly granted access via POLICY (currently none, which is secure by default)

-- 1. Portfolio (Policy Existed but RLS was disabled)
ALTER TABLE public.portfolio ENABLE ROW LEVEL SECURITY;

-- 2. Core Tables
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.run_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.alerts ENABLE ROW LEVEL SECURITY;

-- 3. Cache Tables
ALTER TABLE public.ticker_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.news_cache ENABLE ROW LEVEL SECURITY;

-- 4. Strategy & Analysis
ALTER TABLE public.strategy_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rebalancer_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.signal_tracking ENABLE ROW LEVEL SECURITY;

-- 5. ML Tables
ALTER TABLE public.ml_model_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ml_predictions ENABLE ROW LEVEL SECURITY;

-- 6. Paper Trading Tables
ALTER TABLE public.paper_accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.paper_portfolio ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.paper_trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.backtest_results ENABLE ROW LEVEL SECURITY;

-- Verification Note:
-- After running this, the "RLS Disabled" errors in Supabase should disappear.
-- The Python Bot will continue to work because it uses the SERVICE_ROLE key.
