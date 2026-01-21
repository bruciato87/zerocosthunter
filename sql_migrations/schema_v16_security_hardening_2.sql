-- SCHEMA V16: SECURITY HARDENING PART 2
-- Fixes "Function Search Path Mutable" and "Extension in Public" warnings.

-- 1. Fix Mutable Search Path for Function
-- This prevents malicious users from hijacking the search_path to execute custom code.
ALTER FUNCTION public.update_strategy_rules_timestamp() SET search_path = public;

-- 2. Move 'vector' extension to 'extensions' schema
-- Best practice is to keep 'public' clean for user tables.

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS extensions;

-- Move the extension
ALTER EXTENSION vector SET SCHEMA extensions;

-- Grant usage to standard Supabase roles so they can still use vector functions
GRANT USAGE ON SCHEMA extensions TO postgres, anon, authenticated, service_role;

-- Update search_path for roles so they can find 'vector' functions without prefixing 'extensions.'
ALTER ROLE authenticated SET search_path TO public, extensions;
ALTER ROLE service_role SET search_path TO public, extensions;
ALTER ROLE postgres SET search_path TO public, extensions;

-- NOTE: If you receive "permission denied" for ALTER ROLE, likely you are running as 'postgres' which is fine.
-- The critical part is 'ALTER EXTENSION' and 'GRANT USAGE'.
