"""
Memory Module - The Neuro-Link (Phase 16)
=========================================
Gives the AI long-term memory of decisions and reasoning.
Enables semantic search via embeddings and learning from mistakes.
"""

import logging
import os
from datetime import datetime
from typing import Optional, List, Dict

logger = logging.getLogger("Memory")
DEFAULT_EMBEDDING_MODELS = ("text-embedding-004", "text-embedding-005", "gemini-embedding-001")
DEFAULT_EMBEDDING_DIM = 768

class Memory:
    """Long-term memory system for AI decisions and reasoning."""
    
    def __init__(self):
        from db_handler import DBHandler
        self.db = DBHandler()
        self._working_embedding_model = None
        self._embedding_models = []
        self._embedding_dim = DEFAULT_EMBEDDING_DIM
        self._embedding_dim_adjustment_warnings = set()
        env_model = os.getenv("GEMINI_EMBEDDING_MODEL")
        for model_name in (env_model, *DEFAULT_EMBEDDING_MODELS):
            if model_name and model_name not in self._embedding_models:
                self._embedding_models.append(model_name)
        
        # Configure Gemini client (using google-genai package)
        self.client = None
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")
        else:
            logger.warning("GEMINI_API_KEY not set, embeddings disabled")

    def _get_target_embedding_dim(self) -> int:
        """Get target embedding dimension (DB pgvector size), with safe fallback."""
        configured = getattr(self, "_embedding_dim", None)
        if isinstance(configured, int) and configured > 0:
            return configured

        raw_value = os.getenv("MEMORY_EMBEDDING_DIM", str(DEFAULT_EMBEDDING_DIM))
        try:
            parsed = int(raw_value)
            if parsed <= 0:
                raise ValueError("dimension must be > 0")
        except Exception:
            parsed = DEFAULT_EMBEDDING_DIM

        self._embedding_dim = parsed
        return parsed

    def _normalize_embedding_dimension(self, vector: List[float]) -> List[float]:
        """
        Normalize embedding vector size to match DB schema.
        Avoids hard failures like "expected 768 dimensions, not 3072".
        """
        target_dim = self._get_target_embedding_dim()
        current_dim = len(vector)
        if current_dim == target_dim:
            return vector

        warn_key = (current_dim, target_dim)
        warn_seen = getattr(self, "_embedding_dim_adjustment_warnings", None)
        if warn_seen is None:
            warn_seen = set()
            self._embedding_dim_adjustment_warnings = warn_seen

        if warn_key not in warn_seen:
            logger.warning(
                "Embedding dimension mismatch (%s -> %s). Normalizing vector before DB insert.",
                current_dim,
                target_dim,
            )
            warn_seen.add(warn_key)

        if current_dim > target_dim:
            return vector[:target_dim]
        return vector + [0.0] * (target_dim - current_dim)

    def _embedding_model_candidates(self) -> List[str]:
        """Prefer the last known-good model, then fallback models."""
        # Some tests instantiate with Memory.__new__(Memory) and skip __init__.
        working_model = getattr(self, "_working_embedding_model", None)
        embedding_models = getattr(self, "_embedding_models", None)

        if not embedding_models:
            env_model = os.getenv("GEMINI_EMBEDDING_MODEL")
            embedding_models = []
            for model_name in (env_model, *DEFAULT_EMBEDDING_MODELS):
                if model_name and model_name not in embedding_models:
                    embedding_models.append(model_name)
            self._embedding_models = embedding_models

        if working_model:
            return [working_model] + [m for m in embedding_models if m != working_model]

        return list(embedding_models)

    def _extract_embedding_values(self, response) -> Optional[List[float]]:
        """Normalize embedding responses from different SDK versions."""
        if response is None:
            return None

        # Direct vector response
        if isinstance(response, list) and response and isinstance(response[0], (int, float)):
            return [float(v) for v in response]

        candidates = []
        if isinstance(response, dict):
            if response.get("embedding") is not None:
                candidates.append(response.get("embedding"))
            if response.get("embeddings"):
                candidates.extend(response.get("embeddings"))
        else:
            single = getattr(response, "embedding", None)
            if single is not None:
                candidates.append(single)
            many = getattr(response, "embeddings", None)
            if many:
                candidates.extend(many)

        for item in candidates:
            if isinstance(item, list) and item and isinstance(item[0], (int, float)):
                return [float(v) for v in item]
            if isinstance(item, dict):
                values = item.get("values") or item.get("embedding")
                if isinstance(values, list) and values and isinstance(values[0], (int, float)):
                    return [float(v) for v in values]
            else:
                values = getattr(item, "values", None) or getattr(item, "embedding", None)
                if isinstance(values, list) and values and isinstance(values[0], (int, float)):
                    return [float(v) for v in values]

        return None
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for text using Gemini."""
        if not self.client or not text:
            return None
        
        for model_name in self._embedding_model_candidates():
            try:
                response = None

                # google-genai (newer SDK)
                if hasattr(self.client, "models") and hasattr(self.client.models, "embed_content"):
                    try:
                        response = self.client.models.embed_content(
                            model=model_name,
                            contents=text
                        )
                    except TypeError:
                        response = self.client.models.embed_content(
                            model=model_name,
                            contents=[text]
                        )

                # Backward compatibility path
                if response is None and hasattr(self.client, "embeddings") and hasattr(self.client.embeddings, "create"):
                    response = self.client.embeddings.create(
                        model=model_name,
                        input=text
                    )

                vector = self._extract_embedding_values(response)
                if vector:
                    vector = self._normalize_embedding_dimension(vector)
                    if getattr(self, "_working_embedding_model", None) != model_name:
                        logger.info(f"Memory embeddings: using model {model_name}")
                    self._working_embedding_model = model_name
                    return vector

                logger.debug(f"Embedding response format not recognized for model {model_name}")
            except Exception as e:
                err = str(e).upper()
                if "404" in err or "NOT_FOUND" in err or "NOT FOUND" in err:
                    logger.debug(f"Embedding model unavailable ({model_name}): {e}")
                    continue
                logger.warning(f"Embedding generation failed with {model_name}: {e}")
                return None

        logger.warning("Embedding generation failed: no compatible embedding model available")
        return None
    
    def save_memory(
        self,
        ticker: str,
        event_type: str,
        reasoning: str,
        sentiment: str = None,
        confidence: float = None,
        target_price: float = None,
        risk_score: int = None,
        signal_id: str = None,
        source: str = None
    ) -> bool:
        """
        Save a decision/event to memory.
        
        event_type: 'signal', 'analyze', 'trade_open', 'trade_close', 'error'
        """
        try:
            # Generate embedding for semantic search
            embed_text = f"{ticker} {event_type}: {reasoning}"
            embedding = self._generate_embedding(embed_text)
            
            data = {
                "ticker": ticker.upper(),
                "event_type": event_type,
                "reasoning": reasoning,
                "sentiment": sentiment,
                "confidence": confidence,
                "target_price": target_price,
                "risk_score": risk_score,
                "signal_id": signal_id,
                "source": source,
            }
            
            # Add embedding if available (as array string for Supabase)
            if embedding:
                data["embedding"] = embedding
            
            self.db.supabase.table("memory").insert(data).execute()
            logger.info(f"💾 Memory saved: {ticker} [{event_type}]")
            return True
            
        except Exception as e:
            logger.error(f"Memory save failed: {e}")
            return False
    
    def recall_memory(self, ticker: str, limit: int = 5) -> List[Dict]:
        """
        Recall recent memories for a specific ticker.
        Returns most recent events first.
        """
        try:
            result = self.db.supabase.table("memory") \
                .select("ticker, event_type, event_date, reasoning, sentiment, confidence, target_price, actual_outcome, lessons_learned") \
                .eq("ticker", ticker.upper()) \
                .order("event_date", desc=True) \
                .limit(limit) \
                .execute()
            
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Memory recall failed: {e}")
            return []
    
    def recall_all_recent(self, limit: int = 10) -> List[Dict]:
        """Recall most recent memories across all tickers."""
        try:
            result = self.db.supabase.table("memory") \
                .select("ticker, event_type, event_date, reasoning, sentiment, lessons_learned") \
                .order("event_date", desc=True) \
                .limit(limit) \
                .execute()
            
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Memory recall all failed: {e}")
            return []
    
    def get_lessons_learned(self, limit: int = 5) -> List[Dict]:
        """Get recent lessons learned from mistakes (negative outcomes)."""
        try:
            result = self.db.supabase.table("memory") \
                .select("ticker, event_date, reasoning, actual_outcome, lessons_learned") \
                .not_.is_("lessons_learned", "null") \
                .order("event_date", desc=True) \
                .limit(limit) \
                .execute()
            
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Lessons fetch failed: {e}")
            return []
    
    def update_outcome(
        self,
        signal_id: str,
        actual_outcome: float,
        lessons_learned: str = None
    ) -> bool:
        """
        Update a memory record with the actual trade outcome.
        Called when a trade is closed (by Signal Auditor).
        """
        try:
            update_data = {
                "actual_outcome": actual_outcome,
                "outcome_date": datetime.now().isoformat()
            }
            
            if lessons_learned:
                update_data["lessons_learned"] = lessons_learned
            
            self.db.supabase.table("memory") \
                .update(update_data) \
                .eq("signal_id", signal_id) \
                .execute()
            
            logger.info(f"📝 Memory updated: {signal_id} outcome={actual_outcome:.2f}%")
            return True
            
        except Exception as e:
            logger.error(f"Memory update failed: {e}")
            return False
    
    def search_similar(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Semantic search across all memories.
        Uses embedding similarity (cosine distance).
        
        Note: Requires pgvector RPC function in Supabase.
        """
        try:
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                logger.warning("No embedding for search query, falling back to text search")
                return self.recall_all_recent(limit)
            
            # Use Supabase RPC for vector similarity search
            # This requires a custom SQL function in Supabase
            result = self.db.supabase.rpc(
                "search_memory_by_embedding",
                {
                    "query_embedding": query_embedding,
                    "match_count": limit
                }
            ).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.warning(f"Semantic search failed, using fallback: {e}")
            return self.recall_all_recent(limit)
    
    def generate_lesson(self, ticker: str, outcome: float, original_reasoning: str) -> str:
        """
        Use AI to generate a lesson learned from a trade outcome.
        Called when a trade closes with significant P/L.
        """
        try:
            if not self.client:
                return None
            
            outcome_type = "profittevole" if outcome > 0 else "in perdita"
            prompt = f"""
            Analizza questo trade {outcome_type} e genera una breve lezione (max 2 frasi):
            
            Ticker: {ticker}
            Outcome: {outcome:+.2f}%
            Reasoning Originale: {original_reasoning}
            
            Rispondi SOLO con la lezione appresa, in italiano. Esempio:
            "Il trend macro era sfavorevole. Prossima volta aspettare conferma VIX."
            """
            
            # Use Brain's fallback system (respects APP_MODE: PREPROD/PROD)
            from brain import Brain
            brain = Brain()
            response_text = brain._generate_with_fallback(prompt, json_mode=False)
            return response_text.strip()
            
        except Exception as e:
            logger.error(f"Lesson generation failed: {e}")
            return None
    
    def get_context_for_ticker(self, ticker: str) -> str:
        """
        Get formatted memory context for a ticker to inject into AI prompts.
        Used by Brain to have historical awareness.
        """
        memories = self.recall_memory(ticker, limit=3)
        
        if not memories:
            return f"[MEMORY: Nessuna decisione storica per {ticker}]"
        
        context_lines = [f"[MEMORY: Storico decisioni {ticker}]"]
        for m in memories:
            date = m.get('event_date', '')[:10]
            sentiment = m.get('sentiment', 'N/A')
            reasoning = m.get('reasoning', '')[:150]  # Truncate
            outcome = m.get('actual_outcome')
            
            line = f"- {date}: {sentiment}"
            if outcome is not None:
                line += f" (Outcome: {outcome:+.1f}%)"
            line += f" | {reasoning}"
            
            context_lines.append(line)
        
        return "\n".join(context_lines)


if __name__ == "__main__":
    # Test
    mem = Memory()
    
    # Test save
    mem.save_memory(
        ticker="TEST",
        event_type="signal",
        reasoning="Test reasoning for memory system",
        sentiment="BUY",
        confidence=0.85
    )
    
    # Test recall
    memories = mem.recall_memory("TEST")
    print(f"Recalled {len(memories)} memories for TEST")
    for m in memories:
        print(f"  {m['event_date']}: {m['sentiment']} - {m['reasoning'][:50]}")
