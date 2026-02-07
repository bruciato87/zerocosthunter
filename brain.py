from google import genai
from google.genai import types
import os
import logging
import json
from market_data import MarketData
import time
import requests
import re
import json
from market_data import MarketData
from critic import Critic
from council import Council
from signal_intelligence import SignalIntelligence
import time

# Configure logging
logger = logging.getLogger(__name__)

# OpenRouter Model Tier List (best quality first)
# These are FREE models on OpenRouter, ordered by capability
OPENROUTER_MODEL_TIERS = [
    "google/gemini-2.0-pro-exp-02-05:free",     # Elite (if available)
    "google/gemini-2.0-flash-thinking-exp:free", # Strong Reasoner
    "google/gemini-2.0-flash-exp:free",          # Fast, Reliable
    "meta-llama/llama-3.3-70b-instruct:free",    # Good Context
    "deepseek/deepseek-chat:free",               # Reliable V3
    "qwen/qwen-2.5-72b-instruct:free",           # Powerful Backup
    "mistralai/mistral-small-24b-instruct-2501:free",
    "microsoft/phi-3-medium-128k-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "qwen/qwen-2.5-7b-instruct:free",
]

# Gemini Model Tier List (Fallback sequence)
# Priority: High performance → Reliable Fallback
GEMINI_MODEL_TIERS = [
    "gemini-2.0-flash",    # 1500 RPM (High Performance)
    "gemini-2.0-flash-lite", # Cost Effective Fallback
    "gemini-flash-latest"   # General Fallback
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
        self._cached_best_gemini = None # NEW: Cache for discovered Gemini
        self._cache_timestamp = 0
        
        # Track AI Model Usage in detail
        self.usage_history = [] 

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
            logger.info(f"Brain initialized: OpenRouter=YES, Gemini Fallback={'YES' if self.gemini_api_key else 'NO'}")
        else:
            logger.warning(f"Brain initialized: OpenRouter=❌ (no API key), Gemini={'✅' if self.gemini_api_key else '❌'}")
            
        self.critic = Critic(brain_instance=self)
        # Initialize Council (Phase C)
        self.council = Council(brain_instance=self)

    def _get_best_gemini_model(self, excluded_models: list = None) -> str:
        """
        Dynamically discovers best available Gemini model.
        Self-adapting: handles SDK API changes by trying multiple access patterns.
        """
        if excluded_models is None: excluded_models = []
        if not self.gemini_client: return None

        # Preference Order (Updated to latest Gemini models - 2026)
        PREFERENCE = [
            "gemini-2.5-pro", "gemini-2.5-flash", 
            "gemini-2.0-pro", "gemini-2.0-flash-thinking", "gemini-2.0-flash", 
            "gemini-1.5-pro", "gemini-1.5-flash"
        ]
        
        try:
            # Cache check (30 min TTL)
            if self._cached_best_gemini and (time.time() - self._cache_timestamp < 1800):
                if self._cached_best_gemini not in excluded_models:
                    return self._cached_best_gemini

            discovered = []
            for m in self.gemini_client.models.list():
                # SELF-ADAPTING: Try multiple ways to get model info
                name = None
                is_generative = False
                
                # Try different attribute patterns (SDK versions vary)
                for attr in ['name', 'model_name', 'id']:
                    name = getattr(m, attr, None)
                    if name:
                        name = str(name).replace("models/", "")
                        break
                
                if not name:
                    continue
                
                # Check if model supports generation (try multiple patterns)
                methods = None
                for methods_attr in ['supported_generation_methods', 'supported_actions', 'capabilities']:
                    methods = getattr(m, methods_attr, None)
                    if methods is not None:
                        break
                
                # If we found methods, check for generateContent support
                if methods is not None:
                    if isinstance(methods, (list, tuple)):
                        is_generative = any("generate" in str(m_).lower() for m_ in methods)
                    elif isinstance(methods, dict):
                        is_generative = methods.get("generateContent", False) or methods.get("generate_content", False)
                    else:
                        is_generative = "generate" in str(methods).lower()
                else:
                    # No methods attribute found - assume generative if name starts with gemini
                    is_generative = name.startswith("gemini")
                
                # Filter: only Gemini models that can generate text content
                if is_generative and name.startswith("gemini") and name not in excluded_models:
                    name_l = name.lower()

                    # Hard skip known non-text/specialized families for this text pipeline.
                    skip_patterns = [
                        "embedding",
                        "vision",
                        "image",
                        "aqa",
                        "tts",
                        "audio",
                        "speech",
                        "transcribe",
                        "realtime",
                        "live",
                    ]
                    if any(skip in name_l for skip in skip_patterns):
                        continue

                    # Some SDK versions expose supported output modalities.
                    # If present and TEXT is missing, skip the model.
                    modalities = None
                    for mod_attr in [
                        "supported_response_modalities",
                        "response_modalities",
                        "output_modalities",
                    ]:
                        modalities = getattr(m, mod_attr, None)
                        if modalities is not None:
                            break

                    if modalities is not None:
                        if isinstance(modalities, (list, tuple, set)):
                            normalized = {str(x).upper() for x in modalities}
                        else:
                            normalized = {str(modalities).upper()}
                        if "TEXT" not in normalized:
                            continue

                    discovered.append(name)
            
            logger.info(f"Gemini Discovery: Found {len(discovered)} models: {discovered[:5]}...")
            
            # Sort by our preference
            for pref in PREFERENCE:
                for d in discovered:
                    if pref in d:
                        self._cached_best_gemini = d
                        self._cache_timestamp = time.time()
                        logger.info(f"Selected best Gemini model: {d}")
                        return d
            
            # Fallback: use first discovered or static default
            if discovered:
                self._cached_best_gemini = discovered[0]
                self._cache_timestamp = time.time()
                logger.info(f"Using first discovered Gemini: {discovered[0]}")
                return discovered[0]
            
            return GEMINI_MODEL_TIERS[0]
            
        except Exception as e:
            logger.warning(f"Gemini discovery failed: {e}. Using static default: {GEMINI_MODEL_TIERS[0]}")
            return GEMINI_MODEL_TIERS[0]

    def _get_best_free_model(self, excluded_models: list = None, min_context_needed: int = 32000, task_type: str = "default") -> str:
        """
        Dynamically fetches available models from OpenRouter and selects the best FREE one.
        """
        if excluded_models is None:
            excluded_models = []

        # Tiered static fallback (Free models only)
        # These are used if API discovery fails or all discovered models are excluded
        STATIC_FREE_FALLBACKS = [
            "google/gemini-2.0-pro-exp-02-05:free",
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemini-2.0-flash-thinking-exp:free",
            "deepseek/deepseek-chat:free",
            "qwen/qwen-2.5-72b-instruct:free",
            "mistralai/mistral-small-24b-instruct-2501:free",
            "microsoft/phi-3-medium-128k-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free"
        ]

        # OPTIMIZATION: If we have a fresh cached list, use it instead of calling API
        if self._cached_scored_candidates and (time.time() - self._cache_timestamp < 300):
            for score, model_id in self._cached_scored_candidates:
                if model_id not in excluded_models:
                    # STRICT: Ensure it's still free even in cache
                    if ":free" in model_id:
                        logger.info(f"Using cached free model: {model_id} (Score: {score})")
                        return model_id

        # Return cached result if fresh (<1 hour) AND valid (not excluded)
        if self._cached_best_model and (time.time() - self._cache_timestamp < 3600):
            if self._cached_best_model not in excluded_models and ":free" in self._cached_best_model:
                return self._cached_best_model

        try:
            # OpenRouter requires these headers for full model visibility
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "HTTP-Referer": "https://zerocosthunter.vercel.app",
                "X-Title": "ZeroCostHunter"
            }
            
            # SELF-ADAPTING: Try multiple endpoints in case API changes
            endpoints = [
                "https://openrouter.ai/api/v1/models",
                "https://openrouter.ai/api/models",
            ]
            
            models_data = None
            for url in endpoints:
                try:
                    response = requests.get(url, headers=headers, timeout=30)
                    if response.status_code == 200:
                        models_data = response.json()
                        break
                except Exception as e:
                    logger.debug(f"OpenRouter endpoint failed ({url}): {e}")
                    continue
            
            if not models_data:
                raise Exception("All OpenRouter endpoints failed")
            
            # SELF-ADAPTING: Handle different response structures
            data_list = []
            if isinstance(models_data, list):
                data_list = models_data
            elif isinstance(models_data, dict):
                # Try common keys for the model list
                for key in ['data', 'models', 'items', 'results']:
                    if key in models_data and isinstance(models_data[key], list):
                        data_list = models_data[key]
                        break
            
            available_models = set()
            free_models_found = 0
            
            for item in data_list:
                # SELF-ADAPTING: Try multiple attribute names for model ID
                model_id = None
                for id_key in ['id', 'model_id', 'name', 'model']:
                    model_id = item.get(id_key)
                    if model_id:
                        break
                
                if not model_id or model_id in excluded_models:
                    continue

                # SELF-ADAPTING: Try multiple structures for pricing
                is_free = False
                pricing = None
                
                # Pattern 1: pricing.prompt / pricing.completion
                pricing = item.get("pricing")
                if pricing and isinstance(pricing, dict):
                    prompt = pricing.get("prompt", pricing.get("input", 1))
                    completion = pricing.get("completion", pricing.get("output", 1))
                    try:
                        is_free = float(prompt) == 0 and float(completion) == 0
                    except (TypeError, ValueError):
                        pass
                
                # Pattern 2: Direct cost fields
                if not is_free:
                    for cost_key in ['cost', 'price', 'cost_per_token']:
                        cost = item.get(cost_key)
                        if cost is not None:
                            try:
                                is_free = float(cost) == 0
                            except (TypeError, ValueError):
                                pass
                            break
                
                # Pattern 3: Explicit free flag
                if not is_free:
                    is_free = item.get("free", False) or item.get("is_free", False)
                    
                # Pattern 4: Model ID contains ":free" suffix
                if not is_free and ":free" in str(model_id).lower():
                    is_free = True
                
                if not is_free:
                    continue
                    
                free_models_found += 1
                
                # SELF-ADAPTING: Context length check with multiple keys
                context_length = 0
                for ctx_key in ['context_length', 'context_window', 'max_context', 'context_size']:
                    ctx_val = item.get(ctx_key)
                    if ctx_val:
                        try:
                            context_length = int(ctx_val)
                            break
                        except (TypeError, ValueError):
                            pass

                if context_length >= min_context_needed:
                    # Filter out VISION models (inappropriate for text hunt)
                    if "-vl-" in model_id.lower() or "vision" in model_id.lower():
                        continue

                    # [RELAXED] Accept any free model from known providers
                    # but also accept unknown providers if clearly marked :free
                    trusted_providers = ['meta-llama/', 'mistralai/', 'qwen/', 'nvidia/', 
                                        'nousresearch/', 'google/', 'microsoft/', 
                                        'anthropic/', 'openai/']
                    
                    # DeepSeek R1 context handling
                    if min_context_needed <= 10000 or task_type in ["analyze", "rebalance"]:
                        trusted_providers.append('deepseek/')
                    
                    is_trusted = any(tp in model_id.lower() for tp in trusted_providers)
                    is_explicitly_free = ":free" in model_id
                    
                    if is_trusted or is_explicitly_free:
                        available_models.add(model_id)
            
            logger.info(f"OpenRouter Discovery: {free_models_found} free models found, {len(available_models)} meet criteria (>{min_context_needed} ctx)")

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
        
        if task_type in ["analyze", "rebalance"]:
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
            # BLACKLIST: Skip unstable/buggy models (e.g. Gemma 3 returning 404s)
            if model_id in excluded_models or any(b in model_id.lower() for b in ['gemma-3', 'deepseek/deepseek-v3']):
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
        return "google/gemini-2.0-flash-exp:free"

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
        max_retries = 8  # INCREASED: More attempts before giving up
        request_timeout = 60  # INCREASED: Thinking models take time

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

            try:
                response = requests.post(
                    f"{self.openrouter_base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=request_timeout
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
                        "provider": "OpenRouter",
                        "retry_count": attempt
                    }
                    
                    # 2. DeepSeek Logic (Reasoning Content)
                    reasoning = choice.get("reasoning_content", "")
                    if reasoning and not content:
                        content = reasoning
                    
                    # 3. JSON Repair
                    if json_mode and content:
                        import re
                        import json
                        
                        original_content = content
                        json_valid = False
                        repair_strategy = "none"
                        
                        content = re.sub(r'^```(?:json)?\s*', '', content.strip())
                        content = re.sub(r'\s*```$', '', content.strip())
                        
                        try:
                            json.loads(content)
                            json_valid = True
                            repair_strategy = "direct" if content == original_content.strip() else "markdown_clean"
                        except: pass
                        
                        if not json_valid:
                            array_match = re.search(r'(\[[\s\S]*\])', content)
                            if array_match:
                                try:
                                    json.loads(array_match.group(1))
                                    content = array_match.group(1)
                                    json_valid = True
                                    repair_strategy = "array_extract"
                                except: pass
                        
                        if not json_valid:
                            obj_match = re.search(r'(\{[\s\S]*\})', content)
                            if obj_match:
                                try:
                                    json.loads(obj_match.group(1))
                                    content = obj_match.group(1)
                                    json_valid = True
                                    repair_strategy = "object_extract"
                                except: pass
                        
                        self.last_run_details["json_repair_needed"] = repair_strategy != "direct"
                        self.last_run_details["repair_strategy"] = repair_strategy
                        
                        if not json_valid:
                            logger.warning(f"Invalid JSON from {current_model}, retrying...")
                            if attempt < max_retries - 1:
                                excluded_models.append(current_model)
                                continue
                    
                    if not content or not content.strip():
                        logger.warning(f"OpenRouter model {current_model} returned empty content.")
                        if attempt < max_retries - 1:
                            excluded_models.append(current_model)
                            continue
                    
                    # Log Success to DB
                    try:
                        from db_handler import DBHandler
                        db = DBHandler()
                        db.increment_api_counter("openrouter", run_id=self.current_run_id)
                        db.log_model_used(current_model)
                    except: pass
                    
                    self._record_usage(task_type, current_model, "OpenRouter")
                    return content

                
                # Failover triggers (404 Not Found, 429 Rate Limit, 400 Bad Request/Context, 5xx Server)
                elif response.status_code in [400, 402, 404, 429, 500, 502, 503]:
                    error_text = response.text[:300] if response.text else ""
                    logger.warning(f"OpenRouter Error ({response.status_code}) with {current_model}: {error_text}")
                    excluded_models.append(current_model)
                    
                    # If error was 400 (Bad Request - likely Context Length), invalidate cache to force re-selection
                    if response.status_code == 400:
                        self._cached_scored_candidates = []
                    
                    # INTELLIGENT 429 HANDLING: Extract wait time and wait if near end of retries
                    if response.status_code == 429 and attempt >= max_retries - 3:
                        import re as regex_module
                        import time as time_module
                        match = regex_module.search(r'retry in ([\d.]+)s', error_text.lower())
                        wait_time = int(float(match.group(1))) + 2 if match else 15
                        logger.info(f"🔄 Rate limit hit at attempt {attempt+1}. Waiting {wait_time}s before continuing...")
                        time_module.sleep(wait_time)
                        # Reset exclusions to give models another chance
                        excluded_models = excluded_models[-2:]  # Keep only last 2 excluded
                
                else: 
                    response.raise_for_status()
                    
            except Exception as e:
                logger.warning(f"OpenRouter Attempt {attempt+1} failed with {current_model}: {e}")
                excluded_models.append(current_model)
                if attempt == max_retries - 1:
                    raise e
                # NO SLEEP - move to next model immediately
        
        # Track the last model attempted even on total failure
        self.last_run_details = {
            "model": excluded_models[-1] if excluded_models else "Unknown",
            "usage": {"total_tokens": "FAILED (Rate Limited)"},
            "provider": "OpenRouter (FAILED)"
        }
        raise Exception("OpenRouter: All model attempts failed.")

    def _call_gemini_fallback(self, prompt_or_contents, json_mode: bool = False, model: str = None, task_type: str = "fallback") -> str:
        """Direct Gemini API call as last-resort fallback. Supports both text string and multi-part content list."""
        if not self.gemini_client:
            raise Exception("No Gemini client available for fallback")
        
        config = types.GenerateContentConfig(temperature=0.3)
        if json_mode:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3
            )
        
        target_model = model if model else GEMINI_MODEL_TIERS[0]
        
        # Sanitize model name for Direct Gemini API
        target_model = target_model.replace("google/", "").replace(":free", "")
        
        logger.info(f"Gemini Call: Using model {target_model}")
        
        response = self.gemini_client.models.generate_content(
            model=target_model,
            contents=prompt_or_contents,
            config=config
        )
        
        if not response:
            logger.error(f"Gemini ({target_model}) returned None response.")
            raise Exception(f"Gemini ({target_model}) returned None response.")
            
        try:
            content = response.text
        except Exception as e:
             logger.error(f"Error accessing response.text for {target_model}: {e}")
             raise e

        if json_mode:
             content = content.replace("```json", "").replace("```", "").strip()
             
        # Track usage
        try:
            from db_handler import DBHandler
            db = DBHandler()
            db.increment_api_counter("gemini_fallback", run_id=self.current_run_id)
            db.log_model_used(f"google/{target_model}")
        except Exception:
            pass
        
        logger.info(f"Gemini call successful with {target_model}")
        
        # Track detail for status
        usage = {"total_tokens": "N/A (Direct)"}
        try:
            if hasattr(response, 'usage_metadata'):
                usage = {
                    "total_tokens": response.usage_metadata.total_token_count,
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "candidates_tokens": response.usage_metadata.candidates_token_count
                }
        except: pass
        
        self.last_run_details = {
            "model": f"google/{target_model} (Direct)",
            "usage": usage,
            "provider": "Google Direct"
        }


        
        self._record_usage(task_type, f"google/{target_model}", "Google Direct")
        
        return content


    def _generate_with_fallback(self, prompt: str, json_mode: bool = False, model: str = None, prefer_free: bool = True, min_context_needed: int = 32000, task_type: str = "default", prefer_direct: bool = False) -> str:
        """
        Dynamic AI generation with discovery-based fallback.
        Priority: 
        1. Discovered Gemini Free (Best)
        2. OpenRouter Free (Wide pool)
        3. Emergency Gemini Static Fallback
        """
        BACKGROUND_TASKS = ["hunt", "council_debate", "council_critique", "council_rebalance"]
        
        excluded_gemini = []

        # Step 1: Attempt Gemini Direct (Discovered)
        if self.gemini_client:
            max_gemini_retries = 5
            for attempt in range(max_gemini_retries):
                try:
                    target_gemini = self._get_best_gemini_model(excluded_models=excluded_gemini)
                    if not target_gemini:
                        break # No more models to try
                        
                    try:
                        return self._call_gemini_fallback(prompt, json_mode=json_mode, model=target_gemini, task_type=task_type)
                    except Exception as e:
                        error_msg = str(e)
                        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                            logger.warning(f"Gemini {target_gemini} exhausted (429). Trying next discovered model...")
                        else:
                            logger.error(f"Gemini {target_gemini} failed: {e}. Trying next discovered model...")
                        excluded_gemini.append(target_gemini)
                except Exception as e:
                    logger.warning(f"Gemini discovery/call failed at attempt {attempt+1}: {e}")
                    break # Critical failure in discovery logic

        # Step 2: Attempt OpenRouter Free Pool
        if self.openrouter_api_key:
            try:
                # Pass task_type down to OpenRouter call for tracking
                return self._call_openrouter([{"role": "user", "content": prompt}], json_mode=json_mode, model=model, min_context_needed=min_context_needed, task_type=task_type)
            except Exception as e:
                logger.warning(f"OpenRouter pool failed: {e}")


        # Step 3: Emergency Fallback (If everything else failed, try any remaining Gemini tier)
        if self.gemini_client:
            logger.info(f"Triggering Emergency tiered Gemini fallback for {task_type}...")
            try:
                return self._call_gemini_with_tiered_fallback(prompt, json_mode, task_type=task_type)
            except Exception as e:
                logger.error(f"Emergency fallback failed: {e}")

        raise Exception(f"AI Generation Failed: All providers (Gemini/OpenRouter) failed for {task_type}")

    def _call_gemini_with_tiered_fallback(self, prompt: str, json_mode: bool, task_type: str = "emergency_fallback") -> str:
        """
        Helper for Gemini with tiered fallback.
        If a tier is exhausted (429), moves to the next model.
        If ALL tiers exhausted, waits and retries the whole cycle.
        """
        import re as regex_module
        import time as time_module
        
        max_cycles = 3  # Maximum retry cycles
        last_error = None
        
        for cycle in range(max_cycles):
            for model_name in GEMINI_MODEL_TIERS:
                try:
                    return self._call_gemini_fallback(prompt, json_mode=json_mode, model=model_name, task_type=task_type)
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        logger.warning(f"Gemini tier exhausted for {model_name}. Trying next...")
                        last_error = e
                        # Extract wait time from error
                        match = regex_module.search(r'retry in ([\d.]+)s', error_str.lower())
                        if match:
                            last_error.wait_hint = int(float(match.group(1)))
                    else:
                        logger.warning(f"Gemini {model_name} failed with non-429 error: {e}")
                        last_error = e
                    continue
            
            # All tiers exhausted in this cycle - wait and retry if not last cycle
            if cycle < max_cycles - 1:
                wait_time = getattr(last_error, 'wait_hint', 15) + 5
                logger.info(f"🔄 All Gemini tiers exhausted. Waiting {wait_time}s before retry cycle {cycle+2}/{max_cycles}...")
                time_module.sleep(wait_time)
        
        raise last_error if last_error else Exception("All Gemini tiers failed.")
    
    def _record_usage(self, task_type: str, model_name: str, provider: str):
        """Records who did what for the final report."""
        self.usage_history.append({
            "task": task_type,
            "model": model_name,
            "provider": provider
        })

    def get_usage_summary(self) -> str:
        """
        Returns a formatted string for the Telegram footer.
        Example: 🤖 AI: Hunter (Gemini Flash) | Critic (Llama 3.3) | Council (DeepSeek)
        """
        if not self.usage_history:
            return "🤖 AI: Unknown"
            
        # Deduplicate by task (keep latest)
        usage_map = {}
        for entry in self.usage_history:
            task = entry['task']
            # Map internal task names to display names
            if "analyze" in task or "hunt" in task: display = "Hunter"
            elif "critic" in task: display = "Critic"
            elif "council" in task: display = "Council"
            else: display = task.capitalize()
            
            # Simplified model name
            model = entry['model']
            if "gemini" in model.lower(): 
                if "flash" in model.lower(): model = "Gemini Flash"
                elif "pro" in model.lower(): model = "Gemini Pro"
                else: model = "Gemini"
            elif "llama" in model.lower(): model = "Llama 3"
            elif "deepseek" in model.lower(): model = "DeepSeek"
            elif "qwen" in model.lower(): model = "Qwen"
            elif "mistral" in model.lower(): model = "Mistral"
            elif "phi" in model.lower(): model = "Phi-3"
            else: model = model.split('/')[-1] # Fallback to minimal name
            
            usage_map[display] = model
            
        # Build string
        parts = [f"{role} ({model})" for role, model in usage_map.items()]
        return "🤖 AI: " + " | ".join(parts)

    def analyze_news_batch(self, news_list, performance_context=None, insider_context=None, portfolio_context=None, macro_context=None, whale_context=None, market_regime_summary=None, social_context=None, onchain_context=None, strategic_forecast=None):
        """
        [2024 UPDATE] V3.0 Hybrid Brain with Oracle (Phase B).
        Analyze a batch of news items to find high-quality trading opportunities.
        - performance_context: Dict of {"ticker": {stats}} representing past accuracy.
        - insider_context: Dict of { "overall": "EXTREME FEAR", ... }
        - portfolio_context: Dict analysis from Advisor (sectors, tips).
        - macro_context: String summary from Economist (VIX, FED, Yields).
        - whale_context: String summary from WhaleWatcher (On-Chain Flows).
        - market_regime_summary: String summary from MarketRegimeClassifier (L2 Predictive).
        - social_context: String summary from SocialScraper (Reddit/X Hype).
        - onchain_context: String summary from OnChainWatcher (DEX Liquidity).
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
            - **Downgrade 'BUY' to 'HOLD'** unless the signal is overwhelmingly strong (Confidence > 95% and 'Buyout'/'Earnings Beat').
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
                - Prefer 'HOLD' or 'ACCUMULATE' with low targets.
            - If **BUY PRESSURE** (Inflow of Stablecoins):
                - Confirm BULLISH signals with higher confidence.
            """

        # [ORACLE CONTEXT] (V4.0 Phase B)
        oracle_bg = ""
        if social_context or onchain_context:
            oracle_bg = "\n[THE ORACLE - EARLY INTELLIGENCE]\n"
            if social_context:
                oracle_bg += f"{social_context}\n"
            if onchain_context:
                oracle_bg += f"{onchain_context}\n"
            oracle_bg += "**ORACLE RULE:** High social hype + low DEX liquidity = DANGEROUS/PUMP & DUMP. High hype + strong DEX liquidity = BULLISH TREND.\n"

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
            - If you must mention a stock during closed hours, set sentiment to 'HOLD' with note 'Market closed - review at open'
            """
        except Exception as e:
            logger.warning(f"Market hours context failed: {e}")
        
        # [SENTINEL STRATEGIC FORECAST] (V4.1)
        sentinel_bg = ""
        if strategic_forecast:
            gaps = strategic_forecast.get('gaps', [])
            warns = strategic_forecast.get('correlation_warnings', [])
            gaps_str = "\n".join([f"- {g['recommendation']}" for g in gaps])
            warns_str = "\n".join([f"- {w}" for w in warns])
            sentinel_bg = f"""
            [SENTINEL STRATEGIC FORECAST]
            REGIME: {strategic_forecast.get('regime')} ({strategic_forecast.get('risk_level')})
            STRATEGY: {strategic_forecast.get('strategy_summary')}
            
            PORTFOLIO GAPS (ACTIONABLE):
            {gaps_str if gaps_str else "Nessun gap significativo."}
            
            CORRELATION RISKS:
            {warns_str if warns_str else "Nessun rischio correlazione elevato."}
            
            **SENTINEL RULE:**
            - **Priorità ALTA** ai segnali che aiutano a colmare i GAP (es. BUY per settori UNDERWEIGHT).
            """

        # [EARNINGS CALENDAR CONTEXT] (Phase 2)
        earnings_bg = ""
        try:
            sig_intel = SignalIntelligence()
            earnings_alerts = []
            from ticker_resolver import is_probable_ticker, resolve_ticker

            # Extract unique tickers from news
            seen_tickers = set()
            for item in news_list:
                # Prefer pre-resolved tickers produced by main pipeline.
                candidates = []
                direct_ticker = item.get("ticker")
                if isinstance(direct_ticker, str) and direct_ticker.strip():
                    candidates.append(direct_ticker.strip().upper())
                else:
                    # Fallback only when ticker is missing.
                    raw_text = (item.get("title", "") + " " + item.get("summary", "")).upper()
                    candidates.extend(
                        re.findall(r"\b([A-Z0-9]{1,8}(?:[.-][A-Z0-9]{1,8})?)\b", raw_text)
                    )

                for candidate in candidates:
                    if not is_probable_ticker(candidate):
                        continue

                    checked_ticker = resolve_ticker(candidate) or candidate
                    if not checked_ticker or checked_ticker in seen_tickers:
                        continue

                    seen_tickers.add(checked_ticker)
                    earnings = sig_intel.check_earnings_risk(checked_ticker)
                    if earnings.get('has_upcoming_earnings'):
                        risk = earnings.get('risk_level', 'MEDIUM')
                        days = earnings.get('days_until', '?')
                        date = earnings.get('earnings_date', '?')
                        earnings_alerts.append(f"⚠️ {checked_ticker}: Earnings in {days} days ({date}) - Risk: {risk}")
            
            if earnings_alerts:
                earnings_bg = f"""
            [EARNINGS CALENDAR - LIVE DATA]
            The following assets have UPCOMING EARNINGS (high volatility events):
            {chr(10).join(earnings_alerts)}
            
            **EARNINGS RULE (MANDATORY):**
            - If Risk is HIGH (<=3 days): FORBID BUY signals. Use HOLD through the event.
            - If Risk is MEDIUM (4-7 days): Reduce confidence by 0.15. Warn about volatility.
            - If Risk is LOW (>7 days): Proceed normally but note the upcoming event.
            - REASONING: Always explicitly mention if earnings influenced your decision.
            """
        except Exception as e:
            logger.warning(f"Earnings calendar context failed: {e}")

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
            except Exception as e:
                logger.debug(f"FX fetch failed, fallback to static EUR/USD: {e}")
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
        {oracle_bg}
        {sentinel_bg}
        
        [L2 MARKET REGIME - CRITICAL CONTEXT]
        {market_regime_summary if market_regime_summary else "MARKET REGIME: NEUTRAL (Default)"}
        **STRATEGY RULE:** Align all confidence scores and risk assessments with this regime.
        {earnings_bg}
        
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
            - Extract `CURRENT_PRICE` from the technical tag (e.g. `CURRENT_PRICE: €100`). This is the **TRUE QUOTE**.
            - **CURRENCY CONVERSION:** The user operates in **EUR (€)**. Market news often cites USD ($). 
            - **RULE:** 1 EUR ≈ 1.05 - 1.10 USD. If you see a Target Price in USD in the news, convert it to EUR for your output.
            - **SANITY CHECK:** If Target is > 50% higher than Current Price (e.g. Current €100, Target €200), **IT IS LIKELY A HALLUCINATION OR CURRENCY MIX-UP (USD vs EUR).**
            - **ACTION:** In this case, ignore the high target. Estimate a conservative resistance (+15-20%) instead.
            - **EXCEPTION:** Only allow >50% if news explicitly mentions "Buyout", "Takeover", or "Clinical Trial Success".
        - **RSI RULE:** If RSI > 75 (Overbought), avoid "BUY" unless news is fundamental-shifting. Prefer "HOLD".
        - **TREND RULE:** If Trend is "BEARISH" (Below SMA200), be cautious. "Cheaper" is often a trap. Require STRONG positive news.

        **CRITICAL INSTRUCTION: MACRO & WHALE LOGIC (PRIORITY 1)**
        
        **A) MACRO CONTEXT CHECK**
        - You will receive a `[MACRO STRATEGIST]` context block (Risk Level, VIX, Yields).
        - **IF RISK IS 'HIGH' (e.g. FED Meeting, VIX > 30):**
            - **ACTION:** Downgrade ALL "BUY" signals to "HOLD" unless the specific news is "Game Changing" (e.g. Earnings beat +20%).
            - **REASONING:** Explicitly mention "Macro Risk is High (Safety First)" in your reasoning.
        
        **B) WHALE WATCHER CHECK (Binance Flow)**
        - You will receive a `[WHALE WATCHER]` context block.
        - **IF FLOW IS 'BEARISH' (Net Selling / Dump Detected):**
            - **VETO POWER:** You CANNOT issue a "BUY" signal for BTC, ETH, SOL, or Crypto-related stocks (COIN, MSTR).
            - **ACTION:** Downgrade "BUY" to "HOLD".
            - **REASONING:** Start with "Whales are selling..."
        - **IF FLOW IS 'BULLISH' (Net Buying):**
            - **BOOST:** This is a conviction multiplier. Increase Confidence Score by +0.1.
            - **ACTION:** Validates "BUY" signals.
            - **REASONING:** Mention "Supported by Whale Accumulation."

        **LANGUAGE & FORMAT:**
        - **Reasoning**: MUST be in **ITALIAN**.
        - **Sentiment**: ONE of: ["BUY", "SELL", "ACCUMULATE", "PANIC SELL", "HOLD", "WATCH", "AVOID"].
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
                -   **MUST NOT** use "SELL", "PANIC SELL", or "HOLD" (Cannot sell/hold what is not owned).
                -   **WATCH** for assets with high potential but currently poor entries.
                -   **AVOID** for bearish stocks or sectors to avoid.
                -   If the sentiment is not clear, **DO NOT INCLUDE THIS SIGNAL IN THE OUTPUT**. Skip it entirely.
        
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
            - **ACTION:** Suggest "HOLD" through the event volatility.
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
        - Write **Reasoning** in **ITALIAN**. Focus strictly on the **PRIMARY CATALYST** (e.g. Earnings, Product Launch, Macro shift). Be concise (max 2 sentences).
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
                
                # [PHASE A] CRITIC VALIDATION
                if analysis_results:
                    logger.info(f"Refining {len(analysis_results)} signals with The Critic...")
                    # Pass the original prompt as context (it contains news, macro, etc.)
                    analysis_results = self._verify_with_critic(analysis_results, prompt, market_regime_summary)
                    
                    # [PHASE C] COUNCIL CONSENSUS (Adversarial Debate for High-Confidence Signals)
                    analysis_results = self._run_council_consensus(analysis_results)
                
                return analysis_results
            except json.JSONDecodeError:
                logger.error("Failed to parse AI response as JSON.")
                logger.debug(f"Raw response: {response_text}")
                return []
                
        except Exception as e:
            logger.error(f"Error during AI analysis: {e}")
            return []

    def _verify_with_critic(self, signals, context_data, market_regime_summary=None):
        """
        Passes high-confidence signals to the Critic for a second opinion.
        """
        verified_signals = []
        
        critic_context = context_data
        if market_regime_summary:
            critic_context += f"\n\n[L2 MARKET REGIME - OVERRIDE]\n{market_regime_summary}\nINSTRUCTION: Did the Hunter ignore this regime? If so, REJECT."

        for sig in signals:
            try:
                # CRITIC TRIGGER POLICY:
                # 1. Any signal that is NOT a SKIP
                # 2. Confidence >= 0.40 (matches Hunter's new baseline)
                if sig.get('sentiment') != 'SKIP' and sig.get('confidence', 0) >= 0.40:
                    
                    # Prepare Context string for Critic (Extract from signal)
                    signal_context = f"""
                    News Summary: {sig.get('reasoning')}
                    Target Price: {sig.get('target_price', 'N/A')}
                    Original Confidence: {sig.get('confidence')}
                    Risk Score: {sig.get('risk_score')}
                    """
                    
                    # Call Critic
                    # We pass the FULL Prompt as context_data to give the Critic the same info the Hunter had
                    verdict_obj = self.critic.critique_signal(
                        {
                            "ticker": sig.get('ticker'),
                            "direction": sig.get('sentiment'),
                            "confidence": sig.get('confidence'),
                            "reasoning": sig.get('reasoning')
                        },
                        critic_context # Full context with Regime
                    )
                    
                    # Store Critic Data in Signal
                    sig['critic_verdict'] = verdict_obj.verdict
                    sig['critic_score'] = verdict_obj.score
                    sig['critic_reasoning'] = verdict_obj.reasoning
                    
                    if verdict_obj.verdict == "REJECT" and verdict_obj.score < 40:
                        logger.warning(f"⛔ CRITIC HARD VETO {sig['ticker']}: {verdict_obj.reasoning}")
                        sig['sentiment'] = 'HOLD'
                        sig['confidence'] = max(0.1, float(sig.get('confidence', 0.5)) - 0.5)
                    
                    elif verdict_obj.score <= 60:
                        penalty = 0.25
                        old_conf = float(sig.get('confidence', 0.5))
                        sig['confidence'] = max(0.2, old_conf * (1 - penalty))
                        logger.warning(f"⚠️ CRITIC SOFT VETO {sig['ticker']} (Score: {verdict_obj.score}): {verdict_obj.reasoning}")
                    
                    elif verdict_obj.score <= 80:
                        logger.info(f"CRITIC CAUTIOUS APPROVAL {sig['ticker']} (Score: {verdict_obj.score})")
                    
                    else:
                        logger.info(f"CRITIC STRONG APPROVAL {sig['ticker']} (Score: {verdict_obj.score})")
                        if verdict_obj.score > 90:
                            sig['confidence'] = min(0.99, float(sig.get('confidence', 0.5)) + 0.05)
                
                else:
                    # Skip Critic for low confidence
                    sig['critic_verdict'] = "SKIPPED"
                    sig['critic_score'] = None
                    sig['critic_reasoning'] = "Confidence too low for critique."
                
                verified_signals.append(sig)
                
            except Exception as e:
                logger.error(f"Critic verification failed for {sig.get('ticker')}: {e}")
                verified_signals.append(sig) # Pass through on error
                
        return verified_signals

    def _run_council_consensus(self, initial_predictions: list) -> list:
        """
        [PHASE C] Orchestrates consensus debate for high-confidence signals.
        """
        if not initial_predictions:
            return []

        # Only run if signal is strong enough (>0.75) to justify extra API calls
        final_predictions = []
        import asyncio
        
        async def run_council_parallel():
            tasks = []
            for pred in initial_predictions:
                # Council debates only high-confidence or actionable signals
                if float(pred.get('confidence', 0)) >= 0.75 and pred.get('sentiment') != 'HOLD':
                    tasks.append(self.council.get_consensus(pred['ticker'], pred))
                else:
                    final_predictions.append(pred)
            
            if tasks:
                results = await asyncio.gather(*tasks)
                final_predictions.extend(results)
        
        # Use existing loop or handle nested loops for synchronous context
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                loop.run_until_complete(run_council_parallel())
                return final_predictions
            else:
                loop.run_until_complete(run_council_parallel())
                return final_predictions
        except Exception as e:
            logger.warning(f"Council parallel execution failed: {e}. Falling back to initial predictions.")
            return initial_predictions

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
            logger.info("Parsing portfolio image with Gemini Vision (with fallback)...")
            
            contents = [
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            ]
            
            # Use improved fallback loop for Vision too
            excluded_gemini = []
            response_text = None
            max_gemini_retries = 5
            
            for attempt in range(max_gemini_retries):
                try:
                    target_gemini = self._get_best_gemini_model(excluded_models=excluded_gemini)
                    if not target_gemini:
                        break
                    
                    try:
                        response_text = self._call_gemini_fallback(contents, json_mode=True, model=target_gemini, task_type="vision_portfolio")
                        break # Success
                    except Exception as e:
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            logger.warning(f"Gemini {target_gemini} exhausted (429) during Vision. Trying next...")
                        else:
                            logger.error(f"Gemini {target_gemini} vision failure: {e}")
                        excluded_gemini.append(target_gemini)
                except Exception as e:
                    logger.warning(f"Vision discovery failure: {e}")
                    break
            
            if not response_text:
                raise Exception("Vision: All Gemini models failed.")
                
            raw_response = json.loads(response_text)
            
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
            except Exception as e:
                logger.warning(f"Could not fetch EURUSD rate, using 1.1 fallback: {e}")
            
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
                    except (TypeError, ValueError):
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

    def parse_trade_republic_pdf(self, pdf_path: str) -> dict:
        """
        Parses a Trade Republic PDF (Conferma d'ordine) using pypdf + LLM.
        Returns a dict with transaction details.
        """
        logger.info(f"Parsing Trade Republic PDF: {pdf_path}...")
        try:
            # 1. Preliminary check: Is it a valid PDF?
            with open(pdf_path, 'rb') as f:
                header = f.read(1024)
                if b'<?xml' in header or b'<!DOCTYPE' in header:
                    logger.warning(f"Detected XML/HTML header in supposed PDF: {pdf_path}")
                    return {"error": "Il file caricato è un documento XML (es. Fattura Elettronica). Carica invece la 'Conferma d'ordine' in formato PDF."}
                if not header.startswith(b'%PDF-'):
                    # Some PDFs might have small junk before header, but usually start with %PDF-
                    # If it doesn't look like PDF at all, fail early
                    if b'%PDF-' not in header:
                        return {"error": "Il file non sembra un PDF valido. Assicurati di caricare la 'Conferma d'ordine' originale di Trade Republic."}

            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            if not text.strip():
                raise Exception("Il testo estratto dal PDF è vuoto. Carica un file PDF leggibile.")

            prompt = f"""
            **SYSTEM ROLE:**
            You are a Financial Transaction Extraction Assistant specialized in Trade Republic (Italy) order confirmations.
            
            **DOCUMENT TEXT:**
            {text}
            
            **TASK:**
            Extract the following transaction details:
            - **Ticker or Asset Name**: (e.g., BTC, Apple, iShares Core MSCI World)
            - **Action**: (BUY or SELL)
            - **Quantity**: (Number of shares/units)
            - **Price per Unit**: (Price in EUR)
            - **Total Gross**: (Qty * Price)
            - **Commission**: (Usually 1.00 EUR)
            - **Taxes/Imposte**: (If applicable)
            - **Net Total**: (Final amount added or removed from account)
            
            **OUTPUT FORMAT:**
            Return strictly a JSON object:
            {{
                "ticker": "AAPL",
                "action": "BUY",
                "quantity": 10.0,
                "price": 150.25,
                "commission": 1.0,
                "tax": 0.0,
                "net_total": 1503.50,
                "asset_name": "Apple Inc"
            }}
            """
            
            # Use Brain's AI generation to parse the text
            # FORCE prefer_direct=True to get Gemini speed and avoid webhook timeouts
            response_text = self._generate_with_fallback(prompt, json_mode=True, task_type="simple", prefer_direct=True)
            import json
            result = json.loads(response_text)
            logger.info(f"PDF Parse Result: {result}")
            return result

        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            if "Stream has ended unexpectedly" in str(e) or "EOF marker not found" in str(e):
                return {"error": "Documento PDF corrotto o non valido. Assicurati di caricare la 'Conferma d'ordine' originale."}
            return {"error": str(e)}

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
            logger.info("Parsing sale image with Gemini Vision (with fallback)...")
            
            contents = [
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            ]
            
            # Use improved fallback loop
            excluded_gemini = []
            response_text = None
            max_gemini_retries = 5
            
            for attempt in range(max_gemini_retries):
                try:
                    target_gemini = self._get_best_gemini_model(excluded_models=excluded_gemini)
                    if not target_gemini:
                        break
                    
                    try:
                        response_text = self._call_gemini_fallback(contents, json_mode=True, model=target_gemini, task_type="vision_sale")
                        break # Success
                    except Exception as e:
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            logger.warning(f"Gemini {target_gemini} exhausted (429) during Vision. Trying next...")
                        else:
                            logger.error(f"Gemini {target_gemini} vision failure: {e}")
                        excluded_gemini.append(target_gemini)
                except Exception as e:
                    logger.warning(f"Vision discovery failure: {e}")
                    break

            if not response_text:
                raise Exception("Vision Sale: All Gemini models failed.")

            import json
            result = json.loads(response_text)
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

        **PRICE GROUNDING & CURRENCY (CRITICAL):**
        - All technical prices provided above are in **EUR (€)** unless explicitly marked otherwise.
        - **HALLUCINATION CHECK:** If you see a Stock (like META, NVDA, AAPL) with a price over €600, you are likely confusing USD with EUR. 
        - Current Rate: 1 EUR ≈ 1.08 USD. Cross-check your reasoning.
        - **TRUE PRICE:** Trust the `CURRENT_PRICE` from the Technical Context above. It is the real-time quote for the user's local market (Xetra/DE).


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
        - **Decisione:** [BUY / ACCUMULATE / WATCH / HOLD / SELL / AVOID]
        - **Target Price (Est):** [Price in EUR - ALWAYS use € symbol]
        - **Risk Score:** [1-10]
        - **Catalyst:** Cosa stiamo aspettando? (e.g. Earnings date, FDA approval)

        **PORTFOLIO STRATEGY RULES (CRITICAL):**
        1. **Contextual Labels:**
           - **If Portfolio Context says "Not owned":**
             - Use **BUY** for high-conviction bullish entries.
             - Use **WATCH** for neutral/interesting assets to monitor (NEVER use "HOLD").
             - Use **AVOID** for assets with high risk or bearish outlook (NEVER use "SELL").
           - **If Portfolio Context shows ownership:**
             - Use **ACCUMULATE** to increase position.
             - Use **HOLD** to stay in position.
             - Use **SELL** to exit/trim position.
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
            result = self._generate_with_fallback(prompt, json_mode=False, min_context_needed=8000, task_type="analyze", prefer_direct=True)
            
            # [PHASE C.2] COUNCIL CONSENSUS (Adversarial Critique of the Deep Dive)
            import asyncio
            try:
                # Build context for Council
                council_context = f"Macro: {macro_context}\nWhale: {whale_context}\nL1: {l1_context}"
                
                # Check for running loop (same pattern as analyze_news_batch)
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                if loop.is_running():
                    import nest_asyncio
                    nest_asyncio.apply()
                
                final_result = loop.run_until_complete(self.council.get_report_consensus(ticker, result, council_context))
                return final_result
            except Exception as e:
                logger.warning(f"Council report consensus failed: {e}. Returning initial report.")
                return result
        except Exception as e:
            logger.error(f"Deep Dive failed: {e}")
            # [FREE OPTIMIZATION] Fallback to Math-Only Mini Report instead of generic error
            return f"""
⚠️ **AI Generation Unavailable (Quota Exhausted)**

L'analisi testuale profonda non è disponibile al momento, ma ecco i dati quantitativi essenziali:

- **Ticker:** {ticker}
- **Regime:** {strategic_forecast.get('regime', 'N/A') if 'strategic_forecast' in locals() else 'N/A'}
- **Technical Verdict:** Basato su ATR/RSI, monitorare i livelli indicati nel messaggio precedente.
- **Risk Advice:** Esercitare cautela (High Risk) data l'assenza di validazione semantica.

*Riprova tra qualche ora per il report completo.*
"""

if __name__ == "__main__":
    b = Brain()
    # Test stub
