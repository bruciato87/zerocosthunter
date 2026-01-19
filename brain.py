from google import genai
from google.genai import types
import os
import logging
import json
from market_data import MarketData
import time
import requests

# Configure logging
logger = logging.getLogger(__name__)

# OpenRouter Model Tier List (best quality first)
# These are FREE models on OpenRouter, ordered by capability
OPENROUTER_MODEL_TIERS = [
    "google/gemini-2.0-flash-thinking-exp:free", # Strong Reasoner (Reliable)
    "deepseek/deepseek-chat:free",           # Reliable V3
    "google/gemini-2.5-flash:free",      # Fast, 1M context
    "meta-llama/llama-3.3-70b-instruct:free", # Reliable, Good Context
]

class Brain:
    def __init__(self):
        # OpenRouter API key (primary)
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        
        # Fallback: Direct Gemini (if OpenRouter fails AND key is available)
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        self.gemini_client = genai.Client(
            api_key=self.gemini_api_key,
            http_options={'timeout': 300000}
        ) if self.gemini_api_key else None
        
        # Track current run_id for per-run API tracking
        self.current_run_id = None
        
        # Cache for best model (refreshed each run)
        self._cached_best_model = None
        self._cache_timestamp = 0
        self._cached_scored_candidates = []  # NEW: Cache full ranked list
        
        # Track last execution details for reporting
        self.last_run_details = {}
        
        # Load App Mode from DB Settings (PROD = Hybrid, PREPROD = Gemini Only)
        try:
            from db_handler import DBHandler
            db = DBHandler()
            settings = db.get_settings()
            self.app_mode = settings.get("app_mode", "PROD")
            logger.info(f"Brain Mode: {self.app_mode}")
        except Exception as e:
            logger.warning(f"Failed to load settings, defaulting to PROD: {e}")
            self.app_mode = "PROD"
        
        # Log initialization
        if self.openrouter_api_key:
            logger.info(f"Brain initialized: OpenRouter=✅, Gemini Fallback={'✅' if self.gemini_api_key else '❌'}")
        else:
            logger.warning("Brain initialized: OpenRouter=❌ (no API key), Gemini={'✅' if self.gemini_api_key else '❌'}")

    def _get_best_free_model(self, excluded_models: list = None, min_context_needed: int = 32000, task_type: str = "default") -> str:
        """
        Dynamically fetches available models from OpenRouter and selects the best FREE one.
        Uses fuzzy matching against a preference list to auto-discover new versions.
        OPTIMIZATION: Uses cached candidate list on retry to avoid repeated API calls.
        
        Args:
            min_context_needed: Minimum context window required. 
                                DeepSeek R1 (8k limit) is excluded if this is > 10000.
            task_type: Type of task calling this function. Options:
                       - "hunt": News batch analysis (medium context, reliability > power)
                       - "analyze": Deep dive single ticker (reasoning > speed)
                       - "rebalance": Portfolio analysis (medium)
                       - "sentiment": Simple classification (small, fast)
                       - "default": Balanced selection
        """
        if excluded_models is None:
            excluded_models = []

        # OPTIMIZATION: If we have a fresh cached list, use it instead of calling API
        if self._cached_scored_candidates and (time.time() - self._cache_timestamp < 300):
            # Find best non-excluded model from cached list
            for score, model_id in self._cached_scored_candidates:
                if model_id not in excluded_models:
                    logger.info(f"Using cached model: {model_id} (Score: {score})")
                    return model_id
            # All cached models excluded, will fall through to API call below

        # Return cached result if fresh (<1 hour) AND valid (not excluded)
        if self._cached_best_model and (time.time() - self._cache_timestamp < 3600):
            if self._cached_best_model not in excluded_models:
                return self._cached_best_model

        try:
            # OpenRouter requires these headers for full model visibility
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "HTTP-Referer": "https://zerocosthunter.vercel.app",
                "X-Title": "ZeroCostHunter"
            }
            
            # Fetch pricing for all models
            # Fetch pricing for all models
            url = "https://openrouter.ai/api/v1/models"
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            models_data = response.json()
            
            # Handle list response from OpenRouter
            data_list = models_data.get("data", [])
            available_models = set()
            
            for item in data_list:
                model_id = item.get("id")
                if not model_id: continue
                
                if model_id in excluded_models:
                    continue

                # Pricing check (Free only)
                pricing = item.get("pricing", {})
                prompt = float(pricing.get("prompt", "1"))
                completion = float(pricing.get("completion", "1"))
                
                # Context length check (Min 32k for large updates)
                context_length = int(item.get("context_length", 0))

                if prompt == 0 and completion == 0:
                    if context_length >= min_context_needed:
                        # WHITELIST approach: Only trust verified providers/models
                        # Added google/ for Gemma models (stable)
                        
                        trusted_providers = ['meta-llama/', 'mistralai/', 'qwen/', 'nvidia/', 'nousresearch/', 'google/']
                        
                        # DeepSeek R1 (free tier) has ~8k real limit despite claiming 163k.
                        # Only enable it if the task requires small context OR is "analyze" (reasoning priority)
                        if min_context_needed <= 10000 or task_type == "analyze":
                            trusted_providers.append('deepseek/')
                        
                        if any(tp in model_id.lower() for tp in trusted_providers):
                            available_models.add(model_id)

        except Exception as e:
             logger.warning(f"OpenRouter model discovery failed: {e}")
             # Fallback to static selection excluding bad ones
             for m in OPENROUTER_MODEL_TIERS:
                 if m not in excluded_models:
                     return m
             
             # If all failed/excluded, raise Exception to trigger Gemini fallback
             raise Exception("OpenRouter: All static models excluded.")

        # --- TASK-AWARE PREFERENCE SYSTEM ---
        # Different tasks have different optimal model profiles
        
        if task_type == "analyze":
            # Deep analysis: Prioritize REASONING models (DeepSeek R1, large Llamas)
            preferences = [
                'deepseek-r1',          # Best reasoning
                'llama-3.1-405b',       # Largest open model
                'hermes-3-llama-3.1-405b',
                'qwen3-coder',          # Strong reasoning
                'llama-3.3-70b',
            ]
        elif task_type == "hunt":
            # News batch: Prioritize RELIABILITY over raw power
            # Avoid 405B (often rate-limited), prefer stable 70B models
            preferences = [
                'gemma-3-27b',          # Google-backed, very stable
                'llama-3.3-70b',        # Good balance
                'mistral-small-3.1',    # Fast and reliable
                'qwen2.5-72b',          # Strong
                'llama-3.1-70b',
                'llama-3.1-405b',       # Only as last resort
            ]
        elif task_type == "sentiment" or task_type == "simple":
            # Simple tasks: Small, fast models
            preferences = [
                'gemma-3-27b',
                'mistral-small-3.1',
                'llama-3.1-8b',
                'gemma-2-9b',
            ]
        else:
            # Default: Balanced (original logic)
            preferences = [
                'llama-3.3-70b',
                'gemma-3-27b',
                'qwen3-coder',
                'mistral-small-3.1',
                'llama-3.1-405b',
                'deepseek-r1',
            ]

        # Score available models to find the "Most Powerful" one automatically.
        scored_candidates = []
        
        for model_id in available_models:
            if model_id in excluded_models:
                continue
                
            score = 0
            lower_id = model_id.lower()
            
            # 1. Preference List Bonus (Verified Top Tier)
            try:
                for idx, pref in enumerate(preferences):
                    if pref in model_id:
                        score += (1000 - idx * 100) # Big bonus
                        break
            except: pass
            
            # 2. "Power" & Reasoning Keywords Heuristic
            if 'r1' in lower_id: score += 200  # DeepSeek R1 reasoning
            if 'coder' in lower_id: score += 150  # Coding/reasoning focus
            if 'thinking' in lower_id: score += 150
            if 'pro' in lower_id: score += 90
            if 'ultra' in lower_id: score += 90
            if 'plus' in lower_id: score += 50
            if 'max' in lower_id: score += 50
            
            # 3. Model Size Heuristic (bigger = more capable)
            if '405b' in lower_id: score += 250  # Massive models
            if '480b' in lower_id: score += 250
            if '80b' in lower_id: score += 150
            if '70b' in lower_id: score += 120
            if '72b' in lower_id: score += 120
            if 'large' in lower_id: score += 60
            
            # 4. negative/neutral qualifiers
            if 'flash' in lower_id: score += 40
            if 'turbo' in lower_id: score += 30
            if 'distill' in lower_id: score -= 10
            if 'mini' in lower_id: score -= 20
            if '8b' in lower_id: score -= 20
            if '7b' in lower_id: score -= 20
            
            # 5. Brand Reliability Bonus (trusted providers from whitelist)
            if 'deepseek/' in lower_id: score += 120  # DeepSeek R1 is excellent (when context allows)
            if 'meta-llama/' in lower_id: score += 100  # Most reliable
            if 'qwen/' in lower_id: score += 90   # Qwen3 is very capable
            if 'mistralai/' in lower_id: score += 80
            if 'nousresearch/' in lower_id: score += 70
            if 'nvidia/' in lower_id: score += 60
            
            scored_candidates.append((score, model_id))
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        if scored_candidates:
            best_score, best_model = scored_candidates[0]
            logger.info(f"Dynamic Model Selection Winner: {best_model} (Score: {best_score}) from {len(scored_candidates)} candidates")
            # Cache both best model and full ranked list for retries
            self._cached_best_model = best_model
            self._cached_scored_candidates = scored_candidates  # NEW: Cache full list
            self._cache_timestamp = time.time()
            return best_model
        
        # Fallback (Static) if discovery yields nothing
        if available_models: # Relax context filter?
             # Try 16k fallback logic here if needed, or just return first available
             pass

        logger.warning("OpenRouter: No verified high-context free models found via API. Using static fallback.")
        return "google/gemini-2.5-flash:free"

    def _call_openrouter(self, messages: list, temperature: float = 0.3, json_mode: bool = False, model: str = None, min_context_needed: int = 32000, task_type: str = "default") -> str:
        """
        Call OpenRouter API with auto-failover and usage tracking.
        Args:
            min_context_needed: Minimum context window required.
            task_type: Type of task (hunt/analyze/rebalance/sentiment) for smart model selection.
        """
        if not self.openrouter_api_key:
            raise Exception("OPENROUTER_API_KEY not configured")

        excluded_models = []
        max_retries = 3

        for attempt in range(max_retries):
            # On first attempt, use passed model OR discover. On retries, ALWAYS rediscover.
            if attempt == 0 and model:
                selected_model = model
            else:
                selected_model = self._get_best_free_model(excluded_models=excluded_models, min_context_needed=min_context_needed, task_type=task_type)
            
            # Use selected_model for this attempt
            current_model = selected_model

            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://zerocosthunter.vercel.app",
                "X-Title": "ZeroCostHunter"
            }

            payload = {
                "model": current_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 8192,  # Increased for large analyses
                "top_p": 0.9  # Standardize top_p
            }

            # NOTE: We DO NOT send response_format={"type": "json_object"} anymore.
            # Many "Free" models (Llama 3, Mistral, etc.) on OpenRouter do not support it and throw 400 errors.
            # We rely entirely on the prompt instructions and the Regex parser below.

            try:
                response = requests.post(
                    f"{self.openrouter_base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                
                # Success checks
                if response.status_code == 200:
                    data = response.json()
                    choice = data["choices"][0]["message"]
                    content = choice.get("content", "")
                    
                    # 1. Track Usage
                    self.last_run_details = {
                        "model": current_model,
                        "usage": data.get("usage", {}),
                        "provider": "OpenRouter"
                    }
                    
                    # 2. DeepSeek Logic (Reasoning Content)
                    reasoning = choice.get("reasoning_content", "")
                    if reasoning and not content:
                        content = reasoning
                    
                    # 3. JSON Repair - Robust extraction from various LLM response formats
                    if json_mode and content:
                        import re
                        import json
                        
                        original_content = content
                        json_valid = False
                        repair_strategy = "none"  # Track which strategy worked
                        
                        # Strategy 1: Clean markdown code blocks
                        content = re.sub(r'^```(?:json)?\s*', '', content.strip())
                        content = re.sub(r'\s*```$', '', content.strip())
                        
                        # Strategy 2: Try direct parse first
                        try:
                            json.loads(content)
                            json_valid = True
                            repair_strategy = "direct" if content == original_content.strip() else "markdown_clean"
                        except:
                            pass
                        
                        # Strategy 3: Extract JSON array [...] (common for predictions)
                        if not json_valid:
                            array_match = re.search(r'(\[[\s\S]*\])', content)
                            if array_match:
                                try:
                                    json.loads(array_match.group(1))
                                    content = array_match.group(1)
                                    json_valid = True
                                    repair_strategy = "array_extract"
                                except:
                                    pass
                        
                        # Strategy 4: Extract JSON object {...}
                        if not json_valid:
                            obj_match = re.search(r'(\{[\s\S]*\})', content)
                            if obj_match:
                                try:
                                    json.loads(obj_match.group(1))
                                    content = obj_match.group(1)
                                    json_valid = True
                                    repair_strategy = "object_extract"
                                except:
                                    pass
                        
                        # Strategy 5: Remove common LLM artifacts and retry
                        if not json_valid:
                            # Remove thinking/explanation before JSON
                            cleaned = re.sub(r'^.*?(?=[\[\{])', '', original_content, flags=re.DOTALL)
                            cleaned = re.sub(r'[\]\}][^\]\}]*$', lambda m: m.group(0)[0], cleaned)
                            try:
                                json.loads(cleaned.strip())
                                content = cleaned.strip()
                                json_valid = True
                                repair_strategy = "artifact_removal"
                            except:
                                pass
                        
                        # Store repair info in last_run_details
                        self.last_run_details["json_repair_needed"] = repair_strategy != "direct"
                        self.last_run_details["repair_strategy"] = repair_strategy
                        
                        if not json_valid:
                            self.last_run_details["repair_strategy"] = "failed"
                            logger.warning(f"Invalid JSON from {current_model}, retrying...")
                            if attempt < max_retries - 1:
                                excluded_models.append(current_model)
                                continue
                    
                    
                    # Log Model Usage to Class State (Critical for Reporting)
                    self.last_run_details = {
                        "model": current_model,
                        # OpenRouter returns usage in 'usage' field usually
                        "usage": response.json().get("usage", {}),
                        "provider": "OpenRouter",
                        "repair_strategy": "none", # Will be overwritten if repair is needed later
                        "retry_count": attempt
                    }

                    # Log Success to DB
                    try:
                        from db_handler import DBHandler
                        db = DBHandler()
                        db.increment_api_counter("openrouter", run_id=self.current_run_id)
                        db.log_model_used(current_model)
                    except: pass
                    
                    return content

                
                # Failover triggers (404 Not Found, 429 Rate Limit, 400 Bad Request/Context, 5xx Server)
                elif response.status_code in [400, 404, 429, 500, 502, 503]:
                    logger.warning(f"OpenRouter Error ({response.status_code}) with {current_model}: {response.text}")
                    excluded_models.append(current_model)
                    
                    # If error was 400 (Bad Request - likely Context Length), invalidate cache to force re-selection
                    if response.status_code == 400:
                        self._cached_scored_candidates = []
                        
                    # Wait before retry
                    time.sleep(2 + attempt) 
                
                else: 
                    response.raise_for_status()
                    
            except Exception as e:
                logger.warning(f"OpenRouter Attempt {attempt+1} failed with {current_model}: {e}")
                excluded_models.append(current_model)
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2) # Brief cooldown
        
        # Track the last model attempted even on total failure
        self.last_run_details = {
            "model": excluded_models[-1] if excluded_models else "Unknown",
            "usage": {"total_tokens": "FAILED (Rate Limited)"},
            "provider": "OpenRouter (FAILED)"
        }
        raise Exception("OpenRouter: All model attempts failed.")

    def _call_gemini_fallback(self, prompt: str, json_mode: bool = False) -> str:
        """Direct Gemini API call as last-resort fallback."""
        if not self.gemini_client:
            raise Exception("No Gemini client available for fallback")
        
        config = types.GenerateContentConfig(temperature=0.3)
        if json_mode:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3
            )
        
        response = self.gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=config
        )
        
        # Track usage (Approximated as Direct API doesn't return token counts easily in this client version)
        self.last_run_details = {
            "model": "google/gemini-2.5-flash (Direct)",
            "usage": {"total_tokens": "N/A (Direct Fallback)"},
            "provider": "Google Direct"
        }
        
        content = response.text
        if json_mode:
             content = content.replace("```json", "").replace("```", "").strip()
             
        # Track fallback usage
        # Track fallback usage
        try:
            from db_handler import DBHandler
            db = DBHandler()
            db.increment_api_counter("gemini_fallback", run_id=self.current_run_id)
        except Exception:
            pass
        
        logger.info("Gemini fallback call successful")
        return content

    def _generate_with_fallback(self, prompt: str, json_mode: bool = False, model: str = None, prefer_free: bool = True, min_context_needed: int = 32000, task_type: str = "default") -> str:
        """
        Smart AI generation with automatic fallback.
        
        Flow:
        1. Try OpenRouter with best free model
        2. If rate limited, try next model in tier list
        3. If all OpenRouter fails, try direct Gemini API
        
        Args:
            task_type: Type of task (hunt/analyze/rebalance/sentiment) for smart model selection.
        Versatile generation:
        1. Try OpenRouter (if API Key exists AND mode != PREPROD)
        2. Fallback to Gemini Direct (if configured)
        """
        
        # --- MODE CHECK ---
        # If PREPROD, force Gemini (skip OpenRouter)
        force_gemini = (self.app_mode == "PREPROD")
        
        if self.openrouter_api_key and not force_gemini:
             # Regular Hybrid Flow
             try:
                 # Logic for OpenRouter Call with task-aware model selection
                 logger.info(f"OpenRouter call with task_type: {task_type}")
                 return self._call_openrouter([{"role": "user", "content": prompt}], json_mode=json_mode, model=model, min_context_needed=min_context_needed, task_type=task_type)
             except Exception as e:
                 logger.warning(f"OpenRouter failed, falling back: {e}")
        elif force_gemini:
             logger.info("🔧 PREPROD MODE: Skipping OpenRouter, forcing Gemini Direct.")

        # Fallback / Direct Gemini
        if self.gemini_api_key or self.gemini_client:
            return self._call_gemini_with_retries(prompt, json_mode)
        
        return ""

    def _call_gemini_with_retries(self, prompt: str, json_mode: bool) -> str:
        """Helper for Gemini with backoff retries."""
        max_retries = 3
        backoff = 5
        for attempt in range(max_retries + 1):
            try:
                result = self._call_gemini_fallback(prompt, json_mode=json_mode)
                return result
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries:
                        import re
                        wait_time = backoff
                        match = re.search(r"retry in (\d+(\.\d+)?)s", error_str)
                        if match:
                            wait_time = float(match.group(1)) + 2.0
                        logger.warning(f"Gemini 429. Retrying in {wait_time:.1f}s... ({attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                        backoff *= 1.5
                        continue
                # Track failure before re-raising
                self.last_run_details = {
                    "model": "gemini-2.5-flash (FAILED)",
                    "usage": {"total_tokens": "FAILED (Rate Limited)"},
                    "provider": "Google Direct (FAILED)"
                }
                logger.error(f"Gemini Fallback completely failed: {e}")
                raise e
        return ""
    
    def analyze_news_batch(self, news_list, performance_context=None, insider_context=None, portfolio_context=None, macro_context=None, whale_context=None):
        """
        [2024 UPDATE] V3.0 Hybrid Brain with Memory & Insider & Advisor Info & Macro Strategy & Whale Watcher.
        Analyze a batch of news items to find high-quality trading opportunities.
        - performance_context: Dict of {"ticker": {stats}} representing past accuracy.
        - insider_context: Dict of { "overall": "EXTREME FEAR", ... }
        - portfolio_context: Dict analysis from Advisor (sectors, tips).
        - macro_context: String summary from Economist (VIX, FED, Yields).
        - whale_context: String summary from WhaleWatcher (On-Chain Flows).
        """
        if not news_list:
            logger.info("No news to analyze.")
            return []

        # [FEEDBACK LOOP INJECTION]
        memories = ""
        if performance_context:
            memories = "\n[PAST PERFORMANCE MEMORY]\n"
            for ticker, stats in performance_context.items():
                if stats['status'] == 'NEGATIVE':
                    memories += f"WARNING: {ticker} -> High Failure Rate ({stats['win_rate']}% wins). Be extremely skeptical. Require overwhelming evidence to BUY.\n"
                elif stats['status'] == 'POSITIVE':
                    memories += f"NOTE: {ticker} -> High Success Rate ({stats['win_rate']}% wins). Your past logic works well here.\n"
                else:
                    memories += f"INFO: {ticker} -> Neutral/Insufficient history.\n"
        
        # [LESSONS LEARNED INJECTION - AI LEARNS FROM MISTAKES]
        try:
            from memory import Memory
            mem = Memory()
            lessons = mem.get_lessons_learned(limit=3)
            if lessons:
                memories += "\n[LESSONS LEARNED FROM PAST ERRORS]\n"
                for l in lessons:
                    ticker = l.get('ticker', 'N/A')
                    lesson = l.get('lessons_learned', '')
                    outcome = l.get('actual_outcome', 0)
                    memories += f"❌ {ticker} ({outcome:+.1f}%): {lesson}\n"
                memories += "**RULE: Apply these lessons to avoid repeating the same mistakes.**\n"
        except Exception as e:
            logger.warning(f"Could not load lessons: {e}")

        # [INSIDER SENTIMENT INJECTION]
        sentiment_bg = ""
        if insider_context:
            mood = insider_context.get('overall', 'NEUTRAL')
            fg_val = insider_context.get('crypto', {}).get('value', 50)
            
            # Formatting social headlines
            social_lines = ""
            headlines = insider_context.get('social', [])
            if headlines:
                social_lines = "\n\n[SOCIAL HYPE - REDDIT TRENDING]\n" + "\n".join([f"- {h}" for h in headlines])

            sentiment_bg = f"""
            [MARKET SENTIMENT CONTEXT]
            Market Mood: {mood} (Index: {fg_val}/100).
            STRATEGY: "Be Greedy when others are Fearful".
            - If "EXTREME FEAR" (<20): Look for quality assets at a discount ("Buy the Dip").
            - If "EXTREME GREED" (>80): Be cautious of tops ("Take Profit" or "Wait").
            {social_lines}
            """
        
        # [ADVISOR PORTFOLIO CONTEXT]
        advisor_bg = ""
        if portfolio_context:
            tips = portfolio_context.get('tips', [])
            sectors = portfolio_context.get('sector_percent', {})
            tips_str = "\n".join([f"- {t}" for t in tips])
            advisor_bg = f"""
            [PORTFOLIO HEALTH CONTEXT]
            Current Allocation: {sectors}
            RISK MANAGEMENT WARNINGS:
            {tips_str}
            *OBJECTIVE*: If possible, favor trades that help DIVERSIFY this portfolio.
            """

        # [MACRO ECONOMIST CONTEXT] (V4.0)
        macro_bg = ""
        if macro_context:
            macro_bg = f"""
            {macro_context}
            **MACRO RULE:**
            - If Risk Level is HIGH (FED Mtg / VIX Spike): CRITICAL CAUTION.
            - **Downgrade 'BUY' to 'WAIT'** unless the signal is overwhelmingly strong (Confidence > 95% and 'Buyout'/'Earnings Beat').
            - Do NOT recommend Buying speculative assets (Crypto/Tech) during HIGH RISK.
            """

        # [WHALE WATCHER CONTEXT] (V4.0 Phase 11)
        whale_bg = ""
        if whale_context:
            whale_bg = f"""
            {whale_context}
            **WHALE RULE:**
            - If **SELL PRESSURE (Dump Risk)** is detected (Large BTC/ETH Inflows to Exchanges):
                - Be skeptical of Bullish news on Crypto.
                - Prefer 'WAIT' or 'ACCUMULATE' with low targets.
            - If **BUY PRESSURE** (Inflow of Stablecoins):
                - Confirm BULLISH signals with higher confidence.
            """

        # [MARKET HOURS CONTEXT - ITALY TIMEZONE]
        market_hours_bg = ""
        try:
            from economist import Economist
            eco = Economist()
            market_status = eco.get_market_status()
            
            market_hours_bg = f"""
            [MARKET HOURS - ITALY ({market_status['current_time_italy']})]
            🇺🇸 US Stocks: {market_status['us_stocks']}
            🇪🇺 EU Stocks: {market_status['eu_stocks']}
            ₿ Crypto: {market_status['crypto']}
            
            **CRITICAL MARKET HOURS RULES (MANDATORY):**
            - If US market is 🔴 CLOSED: You MUST output 'SKIP' sentiment for ALL US stocks (AAPL, META, NVDA, GOOGL, MSFT, TSLA, etc.)
            - If EU market is 🔴 CLOSED: You MUST output 'SKIP' sentiment for ALL EU stocks (ETFs like EUNL.DE, RBOT.MI, AIAI.MI, etc.)
            - If WEEKEND: 'SKIP' ALL stock signals - they are NOT actionable now
            - ONLY Crypto assets (BTC, ETH, SOL, XRP, RENDER, etc.) are ALWAYS actionable 24/7
            - DO NOT suggest BUY/ACCUMULATE for stocks when their market is CLOSED
            - If you must mention a stock during closed hours, set sentiment to 'WAIT' with note 'Market closed - review at open'
            """
        except Exception as e:
            logger.warning(f"Market hours context failed: {e}")

        # [FX RATE CONTEXT]
        fx_bg = ""
        try:
            # Fetch simple EUR/USD rate
            try:
                import yfinance as yf
                t = yf.Ticker("EURUSD=X")
                data = t.history(period="1d")
                if not data.empty:
                    eur_usd = data['Close'].iloc[-1]
                else:
                    eur_usd = 1.08 # Fallback
            except:
                eur_usd = 1.08

            fx_bg = f"""
            [FX RATES - REAL TIME]
            EUR/USD Exchange Rate: {eur_usd:.4f} (1 EUR = {eur_usd:.4f} USD)
            
            **CURRENCY CONVERSION RULE (MANDATORY):**
            - News often cites USD ($) targets.
            - You MUST convert them to EUR (€) for the report.
            - FORMULA: Target_EUR = Target_USD / {eur_usd:.4f}
            - Example: Analyst says "$200". Math: 200 / {eur_usd:.4f} = €{200/eur_usd:.2f}.
            - DO NOT just change the symbol. DO THE MATH.
            """
        except Exception as e:
            logger.warning(f"FX context failed: {e}")

        # [REMOVED PATTERN RECOGNITION CONTEXT - Handled by SignalIntelligence Layer]
        pattern_bg = ""

        # Prepare the prompt
        news_text = "\n\n".join([f"Source: {item['source']}\nTitle: {item['title']}\nSummary: {item['summary']}" for item in news_list])
        
        prompt = f"""
        ROLES: [Financial Analyst, Hedge Fund Manager, Quantitative Trader]
        {memories}
        {sentiment_bg}
        {macro_bg}
        {whale_bg}
        {market_hours_bg}
        {fx_bg}
        {pattern_bg}
        
        **SYSTEM ROLE:**
        You are a Senior Investment Analyst & Quantitative Trader.
        Your goal is to validate market news with Technical Data AND produce a concrete Quantitative Prediction.

        **USER CONTEXT:**
        - The user is a **European Investor** (Currency: **EUR**).
        - If news mentions USD prices (e.g. "Apple to $200"), KEEP the target in USD ($) but ensure your reasoning considers the asset quality.
        - **Upside %** is universal, so focus heavily on that.

        **CRITICAL INSTRUCTION: TECHNICAL CHECKS (PATH C)**
        - The input news will contain technical tags (e.g. `[Technical: Price: $100, RSI: 80, Trend: BULLISH]`).
        - **YOU MUST INCORPORATE THIS DATA.**
        - **PRICE GROUNDING (CRITICAL):**
            - Extract "Price" from the technical tag. This is the **TRUE CURRENT PRICE**.
            - **Target Price Sanity Check:** Compare your extracted Target Price vs Current Price.
            - **RULE:** If Target is > 50% higher than Current Price (e.g. Current $100, Target $200), **IT IS LIKELY A HALLUCINATION OR PRE-SPLIT DATA.**
            - **ACTION:** In this case, ignore the high target. Estimate a conservative resistance (+15-20%) instead.
            - **EXCEPTION:** Only allow >50% if news explicitly mentions "Buyout", "Takeover", or "Clinical Trial Success".
        - **RSI RULE:** If RSI > 75 (Overbought), avoid "BUY" unless news is fundamental-shifting. Prefer "HOLD".
        - **TREND RULE:** If Trend is "BEARISH" (Below SMA200), be cautious. "Cheaper" is often a trap. Require STRONG positive news.

        **CRITICAL INSTRUCTION: MACRO & WHALE LOGIC (PRIORITY 1)**
        
        **A) MACRO CONTEXT CHECK**
        - You will receive a `[MACRO STRATEGIST]` context block (Risk Level, VIX, Yields).
        - **IF RISK IS 'HIGH' (e.g. FED Meeting, VIX > 30):**
            - **ACTION:** Downgrade ALL "BUY" signals to "WAIT" or "HOLD" unless the specific news is "Game Changing" (e.g. Earnings beat +20%).
            - **REASONING:** Explicitly mention "Macro Risk is High (Safety First)" in your reasoning.
        
        **B) WHALE WATCHER CHECK (Binance Flow)**
        - You will receive a `[WHALE WATCHER]` context block.
        - **IF FLOW IS 'BEARISH' (Net Selling / Dump Detected):**
            - **VETO POWER:** You CANNOT issue a "BUY" signal for BTC, ETH, SOL, or Crypto-related stocks (COIN, MSTR).
            - **ACTION:** Downgrade "BUY" to "WAIT".
            - **REASONING:** Start with "Whales are selling..."
        - **IF FLOW IS 'BULLISH' (Net Buying):**
            - **BOOST:** This is a conviction multiplier. Increase Confidence Score by +0.1.
            - **ACTION:** Validates "BUY" signals.
            - **REASONING:** Mention "Supported by Whale Accumulation."

        **LANGUAGE & FORMAT:**
        - **Reasoning**: MUST be in **ITALIAN**.
        - **Sentiment**: ONE of: ["BUY", "SELL", "ACCUMULATE", "PANIC SELL", "HOLD", "WAIT"].
        - **Required:** Your reasoning MUST cite the specific data point (Macro or Whale) if it influenced the decision.
        
        **CRITICAL FILTERS & BOOSTS:**
        1.  **Trade Republic Friendly Only:** Focus ONLY on major High-Cap Stocks and Major Cryptocurrencies.
        2.  **Ignore:** Penny stocks, low volume altcoins, obscure companies.
        3.  **OWNERSHIP RULE (CRITICAL):**
            -   **IF [Portfolio] tag is present (Asset is OWNED):**
                -   **PREFER "ACCUMULATE" over "BUY"** (since we already have exposure).
                -   Use "**BUY**" only if the opportunity is MASSIVE (e.g. Target > +30% or Confidence > 95%).
                -   Use "**HOLD**" if technicals are neutral or risk is medium.
                -   Use "**SELL**" if headwinds are strong.
            -   **IF [Portfolio] tag is MISSING (New Entry):**
                -   **MUST** use "**BUY**" if the opportunity is good.
                -   **MUST NOT** use "ACCUMULATE" (reserved for existing positions).
                -   **MUST NOT** use "SELL" or "PANIC SELL" (Cannot sell what is not owned).
                -   If news is negative, strictly **OMIT/SKIP** the signal.
        
        4.  **SENTIMENT TREND BOOST:**
            - You will see previous history tags (e.g. `[History: WIN (3/0) - Last: BUY]`).
            - **IF RECENT HISTORY IS BULLISH (e.g. Last was BUY/ACCUMULATE):**
                - **BOOST CONFIDENCE +0.05.** Momentum is real.
            - **IF RECENT HISTORY WAS LOSS:**
                - **BE EXTRA CAUTIOUS.** Penalize confidence.

        5.  **BREAKING NEWS PRIORITY:**
            - Check the `published` date. 
            - **IF NEWS < 30 MIN OLD:**
                - **BOOST CONFIDENCE +0.10.** Speed matters.
                - Mark reasoning with **"⚠️ BREAKING NEWS"**.
        
        **ADVANCED EXTRACTION RULES:**
        6.  **DYNAMIC TICKER EXTRACTION (IMPROVEMENT 3):**
            - Do not just rely on the title. Scan the `summary` or `full_text` for tickers.
            - If news mentions "Google", extract **GOOGL**. If "Facebook", **META**. If "Coca-Cola", **KO**.
            - **CRITICAL:** If news is about a whole sector (e.g. "Semiconductors rally"), extract the **LEADER** (e.g. **NVDA**) if it's explicitly named, or the Sector ETF (**SOXX/SMH**).
        
        7.  **SECTOR CORRELATION (IMPROVEMENT 8):**
            - If news is about a competitor, check if it affects the tracked asset.
            - Example: "AMD releases new chip" -> Is this BAD for **NVDA** (Competition) or GOOD (Sector Rally)?
            - **Reasoning:** Explicitly mention "Sector Correlation" if relevant.

        8.  **EARNINGS CALENDAR SAFETY (IMPROVEMENT 7):**
            - If news mentions "Earnings this week" or "Reporting tomorrow":
            - **FORBID BUY SIGNALS** unless you are 99% confident.
            - **ACTION:** Suggest "WAIT" or "HOLD" through the event volatility.
            - **REASONING:** "High risk ahead of earnings."

        9.  **NEWS QUALITY SCORE (IMPROVEMENT 9):**
            - Check the `Source` of the news.
            - **TIER 1 (High Reliability):** WSJ, Bloomberg, Reuters, CNBC, Financial Times, CoinDesk, The Block. -> **Normal Confidence.**
            - **TIER 2 (Medium):** Yahoo Finance, Decrypt, Cointelegraph. -> **Normal Confidence.**
            - **TIER 3 (Low/Blog):** Unknown blogs, "DailyCoin", "AmbCrypto". -> **PENALIZE CONFIDENCE -0.10.** treat as speculation.

        **QUANTITATIVE ANALYSIS (NEW):**
        For every signal, you MUST estimate:
        1.  **Risk Score (1-10):**
            - 1-3: Low Risk (Trend Bullish, RSI Neutral, Blue Chip)
            - 4-7: Medium Risk
            - 8-10: High Risk (Crypto, RSI > 80, Speculative, Macro High)
        2.  **Target Price (Short Term):**
            - Extract from the news and CONVERT TO EUR.
            - **VALIDATE against Current Price.**
            - If no analyst target or target invalid, estimate a logical resistance (+10-20%).
            - **Format:** String with EUR currency (e.g. "€140"). ALWAYS USE EURO €.
        3.  **Upside Percentage:**
            - Numeric value of potential gain (e.g. 15.5 for +15.5%).
            - Return 0.0 if unknown/negative.
        4.  **Risk Management (Stop Loss / Take Profit):**
            - **Stop Loss (SL):** Estimate a technical invalidation level (e.g. -5% to -8%). Return FLOAT price.
            - **Take Profit (TP):** Estimate a target level (e.g. +10% to +20%). Return FLOAT price.
            
        **OUTPUT FORMAT:** JSON list.
        
        **DATA CONTEXTS:**
        {news_text}

        **INSTRUCTIONS:**
        For each news item that contains a SIGNIFICANT signal:
        - Extract **Ticker**, **Type**, **Sentiment**.
        - Write **Reasoning** in **ITALIAN**, concise but insightful. Cite Macro/Whale if relevant.
        - **Risk Score** (1-10), **Target Price**, **Upside %**.
        - **Stop Loss** and **Take Profit** levels.
        - **Confidence Score** (0.0 to 1.0).
        
        **JSON FIELDS:** 
        ticker, asset_type, sentiment, reasoning, confidence, risk_score (int), target_price (str), upside_percentage (float), stop_loss (float), take_profit (float)

        Return strictly a JSON list.
        Example:
        [
            {{
                "ticker": "AAPL",
                "sentiment": "BUY",
                "reasoning": "Utili sopra le attese. Whales in accumulo confermano il trend rialzista nonostante il Macro incerto.",
                "confidence": 0.85,
                "risk_score": 4,
                "target_price": "$200",
                "upside_percentage": 12.5,
                "stop_loss": 185.50,
                "take_profit": 230.00
            }}
        ]
        """

        try:
            logger.info("Sending news batch to AI (Prefer FREE Gemini → DeepSeek fallback)...")
            response_text = self._generate_with_fallback(prompt, json_mode=True, prefer_free=True, min_context_needed=32000, task_type="hunt")
            
            # Parse JSON
            try:
                analysis_results = json.loads(response_text)
                logger.info(f"AI returned {len(analysis_results)} potential signals.")
                return analysis_results
            except json.JSONDecodeError:
                logger.error("Failed to parse AI response as JSON.")
                logger.debug(f"Raw response: {response_text}")
                return []
                
        except Exception as e:
            logger.error(f"Error during AI analysis: {e}")
            return []

    def parse_portfolio_from_image(self, image_path):
        """
        Uses Gemini Vision to extract structured portfolio data from a screenshot.
        """
        logger.info(f"Parsing portfolio image: {image_path}...")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Determine mime type
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = "image/png" if ext == ".png" else "image/jpeg"

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        prompt = """
        **SYSTEM ROLE:**
        You are a Financial Data Extraction Assistant.
        Your goal is to extract financial holdings from the provided SCREENSHOT of a portfolio app (e.g., Trade Republic).

        **INSTRUCTIONS:**
        1. **Identify the View Type:**
           - **List View:** Multiple assets shown.
           - **Detail View:** A single asset is shown. **CRITICAL:** The Asset Name is usually at the very top (e.g. "Bitcoin", "NVIDIA", "S&P 500").
        2. **Identify the Currency:**
           - **RULE #1:** Look at the symbol next to the **Value** or **Price** (e.g. `100 €` vs `$100`).
           - **RULE #2:** IGNORE text in the Asset Name. "Core MSCI World USD" is just a name; if the price is `100 €`, the currency is **EUR**.
           - **RULE #3:** Look for decimal format: `1.000,00` -> **EUR**. `1,000.00` -> **USD**.
           - **Default to EUR** if you see `€` anywhere. Only return "USD" if you see `$`.
        3. **Extract Ticker Symbol:**
           - **Map Names to Tickers:**
             - "Bitcoin" -> "BTC-USD"
             - "Ethereum" -> "ETH-USD"
             - "NVIDIA" -> "NVDA"
             - "Apple" -> "AAPL"
             - **ETFs (CRITICAL for Trade Republic):**
               - "Core MSCI World" / "iShares Core MSCI World" -> "EUNL.DE"
               - "S&P 500" / "iShares Core S&P 500" -> "SXR8.DE"
               - "Nasdaq 100" -> "SXRV.DE"
               - "Global Clean Energy" -> "INRG.DE"
               - "Core DAX" -> "DAXEX.DE"
           - **Look Everywhere:** If the ticker isn't explicit, infer it from the Asset Name.
           - **CRITICAL:** DO NOT RETURN NULL. If you absolutely cannot match it, return "UNKNOWN".
           - **Extract NAME:** The full name of the asset (e.g. "NVIDIA Corp").
           - **Extract TYPE:** "Crypto", "Stock", "ETF", or "Unknown".
         4. **Data Extraction:**
           - If "Quantity" is MISSING, extract **Current Value** and **PnL**.
            - **Detail View Specific:**
              - "Totale" or big number at top = **Current Value**.
              - "Guadagno" or "+/-" number = **PnL**.
              - **Avg Price**: Look for **"Prezzo d'acquisto >"** or "Prezzo d'acq". The number is usually **BELOW** or to the right.
              - **Quantity**: Look for **"Azioni"** or "Shares" or "Quote". The number is usually **BELOW** or to the right.
              - "La tua posizione" block contains the data.
              - **EXAMPLE (Layout with newlines):**
                 "Azioni
                  30,395484"      -> Quantity = 30.395484
                 "Prezzo d'acquisto >
                  107,68 €"       -> Avg Price = 107.68
            - **IMPORTANT:** Convert all numbers to standard **FLOAT format with DOTS** (e.g., "1.250,50" -> 1250.50).

        **OUTPUT FORMAT:**
        Return strictly a JSON object:
        {
            "currency": "EUR",
            "holdings": [
                {
                    "ticker": "EUNL.DE",
                    "name": "Core MSCI World",
                    "asset_type": "ETF",
                    "quantity": null,
                    "avg_price": null,
                    "current_value": 3384.23,
                    "pnl": 111.34,
                    "sector": "ETF"
                }
            ]
        }
        """

        try:
            # Vision requires Gemini (DeepSeek doesn't support images)
            logger.info("Parsing portfolio image with Gemini Vision...")
            response = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            raw_response = json.loads(response.text)
            
            # Handle both list (legacy) and new dict format
            # Handle both list (legacy) and new dict format
            if isinstance(raw_response, list):
                raw_data = raw_response
                currency = "EUR" # Default changed to EUR (User Preference)
            else:
                raw_data = raw_response.get("holdings", [])
                currency = raw_response.get("currency", "EUR")

            # Post-Processing: Fill missing links using Market Data
            final_data = []
            market = MarketData()
            
            import yfinance as yf

            # 1. Get FX Rate (EUR -> USD)
            # We need to know how much 1 EUR is in USD to convert the User's Value(EUR) to Value(USD)
            # Ticker: EURUSD=X (Returns USD per 1 EUR)
            eur_usd_rate = 1.1 # Default fallback
            try:
                fx = yf.Ticker("EURUSD=X")
                hist_fx = fx.history(period="1d")
                if not hist_fx.empty:
                    eur_usd_rate = hist_fx['Close'].iloc[-1]
            except:
                logger.warning("Could not fetch EURUSD rate, using 1.1 fallback")
            
            # Rate for USD to EUR conversion (Inverse)
            usd_eur_rate = 1 / eur_usd_rate if eur_usd_rate else 0.91

            for item in raw_data:
                ticker = item.get('ticker')
                
                # SAFETY FORCE: If ticker is from a European exchange (.DE, .MI), force currency to EUR
                # This prevents "MSCI World USD" name from tricking the model into thinking it's USD
                if ticker and (ticker.endswith('.DE') or ticker.endswith('.MI')):
                    currency = "EUR"

                
                # SAFETY CHECK: If Ticker is None or UNKNOWN
                # Update: Allow UNKNOWN if we have a valid Quantity (Split View Scenario)
                if (not ticker or ticker == "UNKNOWN") and (not item.get('quantity') and not item.get('current_value')):
                    logger.warning(f"Skipping item with invalid ticker and no value: {item}")
                    continue
                
                if not ticker:
                    ticker = "UNKNOWN"

                # Helper to clean "1.000,00" to float
                def clean_float(val):
                    if isinstance(val, str):
                        # Remove dots (thousands), replace comma with dot
                        if ',' in val and '.' in val:
                             # mixed usage
                             val = val.replace('.', '').replace(',', '.')
                        elif ',' in val:
                             # simple comma decimal
                             val = val.replace(',', '.')
                    
                    try:
                        return float(val) if val is not None else None
                    except:
                        return None

                qty = clean_float(item.get('quantity'))
                avg = clean_float(item.get('avg_price'))
                val = clean_float(item.get('current_value'))
                pnl = clean_float(item.get('pnl'))
                
                # Update item with cleaned floats
                item['quantity'] = qty
                item['avg_price'] = avg
                
                # Case 1: We have Quantity and Avg Price (Perfect)
                if qty is not None and avg is not None:
                    final_data.append(item)
                    continue
                    
                # Case 2: We have Value, PnL AND (Crucially) Avg Price from OCR
                # This is the most precise way to get Quantity without external API
                if val is not None and pnl is not None and avg is not None and avg > 0:
                    try:
                         # Logic:
                         # Cost Basis = Value - PnL
                         # Quantity = Cost Basis / Avg Price
                         cost_basis = val - pnl
                         calculated_qty = cost_basis / avg
                         
                         item['quantity'] = round(calculated_qty, 4)
                         logger.info(f"Deriving Qty from OCR AvgPrice for {ticker}: Cost={cost_basis}, Avg={avg} -> Qty={calculated_qty}")
                         final_data.append(item)
                         continue
                    except Exception as e:
                         logger.warning(f"Math error using OCR AvgPrice: {e}")

                # Case 2.5 (New): We have Quantity, Value, PnL but MISSING Avg Price
                # We can calculate Avg Price directly: (Value - PnL) / Quantity
                if qty is not None and qty > 0 and val is not None and pnl is not None and avg is None:
                    try:
                        cost_basis = val - pnl
                        calculated_avg = cost_basis / qty
                        item['avg_price'] = round(calculated_avg, 2)
                        logger.info(f"Deriving AvgPrice from Qty/Val/PnL for {ticker}: {calculated_avg}")
                        final_data.append(item)
                        continue
                    except Exception as e:
                         logger.warning(f"Math error deriving AvgPrice: {e}")
                
                # Case 3: We have Value and PnL but NO Avg Price (Back-calculate using Live Price)
                if val is not None and pnl is not None:
                    try:
                        # Strategy:
                        # 1. We assume Ticker is USD based (e.g. RNDR-USD, NVDA).
                        # 2. We have Value in EUR.
                        # Use ticker_resolver for robust resolution
                        from ticker_resolver import resolve_ticker
                        fetch_ticker = resolve_ticker(ticker)
                        
                        t = yf.Ticker(fetch_ticker)
                        hist = t.history(period="1d")
                        
                        if hist.empty:
                             # Try original ticker as fallback
                             t = yf.Ticker(ticker)
                             hist = t.history(period="1d")
                             
                        if hist.empty:
                            logger.warning(f"Could not fetch data for {ticker}")
                            item['quantity'] = 0.0
                            item['avg_price'] = 0.0
                            final_data.append(item)
                            continue
                            
                        current_price = hist['Close'].iloc[-1]
                        
                        # Determine if we need to convert the fetched price from USD to EUR logic
                        # If ticker is EUNL.DE, the price is in EUR.
                        price_is_eur = ticker.endswith('.DE') or ticker.endswith('.MI') or ticker.endswith('.PA')
                        
                        calculated_qty = 0.0
                        
                        if price_is_eur:
                             # Direct calculation: Value (EUR) / Price (EUR)
                             calculated_qty = val / current_price
                             # Cost Basis (EUR)
                             cost_basis_eur = val - pnl
                             calculated_avg_eur = cost_basis_eur / calculated_qty
                        else:
                             # USD Logic (Original)
                             # Calculate Value in USD first
                             value_usd = val
                             if currency == "EUR":
                                value_usd = val * eur_usd_rate
                                
                             calculated_qty = value_usd / current_price
                             
                             # Derive Cost Basis and Avg Price (keep in EUR)
                             cost_basis_eur = val - pnl
                             calculated_avg_eur = cost_basis_eur / calculated_qty
                        
                        item['quantity'] = round(calculated_qty, 4)
                        item['avg_price'] = round(calculated_avg_eur, 2)
                        
                        logger.info(f"Back-calculated {ticker}: Qty={calculated_qty}, Avg={calculated_avg_eur}€ (Live Price Used, Is_Eur={price_is_eur})")
                        final_data.append(item)
                        continue # Skip to next item to avoid appending twice
                    except Exception as ex:
                        logger.warning(f"Could not back-calculate for {ticker}: {ex}")
                        item['quantity'] = item.get('quantity') or 0.0
                        item['avg_price'] = item.get('avg_price') or 0.0
                        final_data.append(item)
                else:
                    final_data.append(item)

            # FINAL STEP: Normalize to EUR if source was USD
            # This ensures DB always stores EUR
            if currency == "USD":
                logger.info("Converting USD portfolio data to EUR...")
                for item in final_data:
                    # Quantity does not change
                    # Avg Price, Current Value, PnL MUST convert
                    if item.get('avg_price'):
                        item['avg_price'] = round(item['avg_price'] * usd_eur_rate, 2)
                    if item.get('current_value'):
                        item['current_value'] = round(item['current_value'] * usd_eur_rate, 2)
                    if item.get('pnl'):
                        item['pnl'] = round(item['pnl'] * usd_eur_rate, 2)
            
            return final_data

        except Exception as e:
            logger.error(f"Error parsing portfolio image: {e}")
            return []
            
    def evaluate_portfolio(self, portfolio, market_news):
        """
        Cross-reference news with existing holdings to suggest actions.
        (Simplified version: This could be expanded to specialized prompts).
        """
        # For now, we rely on the main batch analysis to catch news relevant to holdings 
        # as long as the news source covers them.
        pass

    def parse_sale_from_image(self, image_path: str) -> dict:
        """
        Uses Gemini Vision to extract sale transaction data from a Trade Republic screenshot.
        Extracts: quantity, price, imposta (tax), commissione (fee), netto ricevuto (net total).
        """
        logger.info(f"Parsing sale screenshot: {image_path}...")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        ext = os.path.splitext(image_path)[1].lower()
        mime_type = "image/png" if ext == ".png" else "image/jpeg"

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        prompt = """
        **SYSTEM ROLE:**
        You are a Financial Transaction Extraction Assistant for Trade Republic screenshots.

        **INSTRUCTIONS:**
        This is a SALE CONFIRMATION screenshot from Trade Republic (Italian version).
        Extract ALL the following data:

        1. **Strumento** (Asset name, e.g., "Render", "Bitcoin")
        2. **Transazione** (Quantity × Price, e.g., "98,7 × 2,15 €")
           - Extract: quantity (e.g., 98.7), price (e.g., 2.15)
        3. **Imposta** (Tax amount, e.g., "21,89 €")
        4. **Commissione** (Commission, usually "1,00 €")
        5. **Totale** or "Hai ricevuto" (Net amount received, e.g., "189,43 €")
        6. **Guadagno** or **Utile** (Profit amount and percentage, e.g., "+65,59 €" and "▲ 44,7 %")

        **NUMBER FORMAT:**
        - Convert European format to decimal: "1.000,50" → 1000.50
        - "21,89" → 21.89
        - "98,7" → 98.7

        **OUTPUT FORMAT:**
        Return strictly a JSON object:
        {
            "asset_name": "Render",
            "quantity": 98.7,
            "price_per_unit": 2.15,
            "gross_total": 212.21,
            "tax_amount": 21.89,
            "commission": 1.00,
            "net_received": 189.43,
            "profit_amount": 65.59,
            "profit_percent": 44.7
        }
        
        If a field is not visible, return null for that field.
        """

        try:
            # Vision requires Gemini (DeepSeek doesn't support images)
            logger.info("Parsing sale image with Gemini Vision...")
            response = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            result = json.loads(response.text)
            logger.info(f"Sale data extracted: {result}")
            return result

        except Exception as e:
            logger.error(f"Error parsing sale image: {e}")
            return {"error": str(e)}

    def generate_deep_dive(self, ticker: str, news_list: list, technical_data: str, portfolio_context: str = None, backtest_context: str = None, macro_context: str = None, whale_context: str = None, l1_context: str = None):
        """
        Generates a Strategic Analysis Report (Path B).
        Now works even without news - uses technical data!
        Enhanced with macro, whale, and L1 predictive context for comprehensive analysis.
        """
        # Build news text if available, otherwise note no news
        if news_list:
            news_text = "\n\n".join([f"Source: {item['source']}\nTitle: {item['title']}\nContent: {item['summary']}" for item in news_list])
        else:
            news_text = "Nessuna news recente trovata. Analizzare basandosi solo sui dati tecnici."
        
        prompt = f"""
        **SYSTEM ROLE:**
        You are a Senior Hedge Fund Strategist. The user has requested a DEEP DIVE analysis on **{ticker}**.
        
        **INPUT DATA:**
        - **Technical Context:** {technical_data}
        - **Portfolio Context:** {portfolio_context or "Not owned"}
        - **Historical Backtest (90d):** {backtest_context or "Not available"}
        - **Macro Context:** {macro_context or "Not available"}
        - **Whale Activity (On-Chain):** {whale_context or "Not available"}
        - **L1 Predictive Analysis:** {l1_context or "Not available"}
        - **Recent News (Full Text):**
        {news_text}


        **TASK:**
        Write a concise but professional "Battle Report" in **ITALIAN**.
        
        **FORMAT:**
        
        # 🛡️ Analisi Strategica: {ticker}
        
        ## 🐂 Bull Case (Perché potrebbe salire)
        - Bullet point 1 (cite specific earnings/news)
        - Bullet point 2
        
        ## 🐻 Bear Case (Rischi Principali)
        - Bullet point 1 (Risk factors)
        - Bullet point 2
        
        ## 🔮 The Verdict
        - **Decisione:** [BUY / HOLD / SELL / WAIT]
        - **Target Price (Est):** [Price in EUR - ALWAYS use € symbol]
        - **Risk Score:** [1-10]
        - **Catalyst:** Cosa stiamo aspettando? (e.g. Earnings date, FDA approval)

        **PORTFOLIO STRATEGY RULES (CRITICAL):**
        1. **ACCUMULATE vs BUY:**
           - **If Portfolio Context says "Not owned":** You CANNOT use "ACCUMULATE" (you can't accumulate what you don't own). Use "BUY" instead.
           - **If Portfolio Context shows ownership:** You CAN use "ACCUMULATE" to add to an existing position.
        2. **Position Sizing:** Check the "Portfolio Context".
           - **Small Position (< 5% of Total):** Do NOT suggest "Take Profit" or "Trim" for small gains. Suggest "HOLD" or "ACCUMULATE" to ride the trend.
           - **Large Position (> 20% of Total):** Be defensive. Suggest "Trim" if risk is high.
        3. **Objective:** MAXIMIZE PROFIT. Do not sell winners too early.
        4. **Contextual Verdict:** Your decision MUST account for the specific quantity and value owned.

        **TONE:** Professional, Direct, No Fluff. Use Markdown.
        """

        try:
            logger.info(f"Generating Deep Dive for {ticker} (OpenRouter auto-select)...")
            # OpenRouter auto-selects best reasoning model (DeepSeek R1 allowed if context < 10k)
            result = self._generate_with_fallback(prompt, json_mode=False, min_context_needed=8000, task_type="sentiment")
            return result
        except Exception as e:
            logger.error(f"Deep Dive failed: {e}")
            return "⚠️ Errore durante l'analisi approfondita."

if __name__ == "__main__":
    b = Brain()
    # Test stub
