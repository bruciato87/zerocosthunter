import os
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import json
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    """Parse common truthy env values."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class DBHandler:
    def __init__(self):
        # Runtime compatibility flags (cached after first DB mismatch).
        self._prediction_source_column = "auto"
        self._supports_extended_sentiment = None
        self._social_stats_enabled = True
        self.dry_run = _env_flag("DRY_RUN")

        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            logger.warning("SUPABASE_URL or SUPABASE_KEY not set. DB actions will fail.")
            self.supabase = None
        else:
            try:
                self.supabase: Client = create_client(url, key)
            except Exception as e:
                logger.error(f"Failed to create Supabase client: {e}")
                self.supabase = None

    # --- USER MANAGEMENT (V8) ---
    # PAUSED

    # --- PORTFOLIO MANAGEMENT ---

    def get_portfolio(self, chat_id: int = None):
        """Fetch current holdings from the portfolio table. Optionally filter by chat_id."""
        try:
            query = self.supabase.table("portfolio").select("*")
            if chat_id:
                # Also filter for confirmed only if we are showing the portfolio view?
                # Usually "Show Portfolio" implies confirmed items. 
                # Let's filter for is_confirmed=True if chat_id is provided, 
                # assuming the user wants to see their active portfolio.
                query = query.eq("chat_id", chat_id).eq("is_confirmed", True)
            
            response = query.execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching portfolio: {e}")
            return []

    def get_portfolio_map(self):
        """Returns portfolio as a dictionary {ticker: {qty, avg_price, sector}} for fast lookup."""
        data = self.get_portfolio()
        if not data:
            return {}
        return {item['ticker']: item for item in data}

    def add_to_portfolio(self, ticker: str, amount: float, price: float, sector: str = "Unknown", asset_name: str = None, asset_type: str = "Unknown", is_confirmed: bool = True, chat_id: int = None, stop_loss: float = 0, take_profit: float = 0):
        """Add or update a holding. Supports 'Draft' mode via is_confirmed=False."""
        try:
            data = {
                "ticker": ticker.upper(),
                "quantity": amount,
                "avg_price": price,
                "sector": sector,
                "asset_name": asset_name,
                "asset_type": asset_type,
                "is_confirmed": is_confirmed,
                "stop_loss_price": stop_loss,
                "take_profit": take_profit,
                "target_price": take_profit # Initial target equals TP
            }
            if chat_id:
                data["chat_id"] = chat_id

            # upsert=True is default behavior if ID matches, but for ticker we rely on unique constraint
            # Note: unique constraint on 'ticker' might be an issue if multiple users draft the same ticker.
            self.supabase.table("portfolio").upsert(data, on_conflict="ticker").execute()
            logger.info(f"Updated portfolio: {ticker} (Confirmed: {is_confirmed}, SL: {stop_loss}, TP: {take_profit})")
        except Exception as e:
            logger.error(f"Error updating portfolio for {ticker}: {e}")

    def update_portfolio_targets(self, ticker: str, target_price: float = None, stop_loss_price: float = None, horizon_days: int = None, entry_reason: str = None, target_type: str = 'MANUAL'):
        """Update strategic targets for a position."""
        try:
            updates = {"target_type": target_type}
            if target_price is not None: updates["target_price"] = target_price
            if stop_loss_price is not None: updates["stop_loss_price"] = stop_loss_price
            if horizon_days is not None: updates["target_horizon_days"] = horizon_days
            if entry_reason is not None: updates["entry_reason"] = entry_reason
            
            self.supabase.table("portfolio").update(updates).eq("ticker", ticker.upper()).execute()
            logger.info(f"Updated targets for {ticker}: {updates}")
            return True
        except Exception as e:
            logger.error(f"Error updating portfolio targets for {ticker}: {e}")
            return False

    def update_asset_protection(self, chat_id: int, ticker: str, stop_loss: float = None, take_profit: float = None):
        """Updates SL/TP for an existing asset."""
        try:
            updates = {}
            if stop_loss is not None:
                updates["stop_loss"] = stop_loss
            if take_profit is not None:
                updates["take_profit"] = take_profit
            
            if not updates: return False

            # Match on Ticker. ChatID is used for verification if schema allows, 
            # but currently portfolio tables typically rely on Ticker uniqueness in this basic schema.
            # We add logic to ensure we target the right row if possible.
            query = self.supabase.table("portfolio").update(updates).eq("ticker", ticker.upper())
            if chat_id:
                query = query.eq("chat_id", chat_id)
                
            query.execute()
            logger.info(f"Protected {ticker}: {updates}")
            return True
        except Exception as e:
            logger.error(f"Error updating protection for {ticker}: {e}")
            return False

    def confirm_portfolio(self, chat_id: int):
        """Mark all drafts for a user as confirmed."""
        try:
            self.supabase.table("portfolio") \
                .update({"is_confirmed": True}) \
                .eq("chat_id", chat_id) \
                .eq("is_confirmed", False) \
                .execute()
            logger.info(f"Confirmed portfolio for chat_id {chat_id}")
        except Exception as e:
            logger.error(f"Error confirming portfolio: {e}")
            raise e

    def delete_drafts(self, chat_id: int):
        """Delete unconfirmed drafts. Keep chat_id based as drafts are ephemeral per chat."""
        try:
            self.supabase.table("portfolio") \
                .delete() \
                .eq("chat_id", chat_id) \
                .eq("is_confirmed", False) \
                .execute()
            logger.info(f"Deleted drafts for chat_id {chat_id}")
        except Exception as e:
            logger.error(f"Error deleting drafts: {e}")
            raise e
    def get_drafts(self, chat_id: int):
        """Fetch unconfirmed drafts for a user."""
        try:
            response = self.supabase.table("portfolio") \
                .select("*") \
                .eq("chat_id", chat_id) \
                .eq("is_confirmed", False) \
                .execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching drafts: {e}")
            return []

    def delete_asset(self, chat_id: int, ticker: str):
        """Delete a specific asset from the portfolio."""
        try:
            self.supabase.table("portfolio") \
                .delete() \
                .eq("chat_id", chat_id) \
                .eq("ticker", ticker) \
                .execute()
            logger.info(f"Deleted asset {ticker} for chat_id {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting asset {ticker}: {e}")
            return False

    def delete_portfolio(self, chat_id: int):
        """Delete ALL assets for a user (Reset)."""
        try:
            self.supabase.table("portfolio") \
                .delete() \
                .eq("chat_id", chat_id) \
                .execute()
            logger.info(f"Deleted entire portfolio for chat_id {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting portfolio for {chat_id}: {e}")
            return False

    def update_draft_quantity(self, record_id: str, quantity: float, avg_price: float = None):
        """Update quantity (and optionally avg_price) of a specific draft record."""
        try:
            data = {"quantity": quantity}
            if avg_price is not None and avg_price > 0:
                data["avg_price"] = avg_price
                
            self.supabase.table("portfolio") \
                .update(data) \
                .eq("id", record_id) \
                .execute()
            logger.info(f"Updated quantity={quantity}, avg_price={avg_price} for draft {record_id}")
        except Exception as e:
            logger.error(f"Error updating draft quantity: {e}")

    def update_draft_ticker(self, record_id: str, ticker: str, price: float):
        """Update ticker and price of a specific draft record."""
        try:
            data = {"ticker": ticker}
            if price:
                data["avg_price"] = price
            
            self.supabase.table("portfolio") \
                .update(data) \
                .eq("id", record_id) \
                .execute()
            logger.info(f"Updated ticker for draft {record_id} to {ticker}")
        except Exception as e:
            logger.error(f"Error updating draft ticker: {e}")

    def update_asset_price(self, chat_id: int, ticker: str, new_price: float):
        """Manually update the average buy price of a confirmed asset."""
        try:
            self.supabase.table("portfolio") \
                .update({"avg_price": new_price}) \
                .eq("chat_id", chat_id) \
                .eq("ticker", ticker) \
                .execute()
            logger.info(f"Manual price update: {ticker} -> {new_price} for {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating asset price: {e}")
            return False

    def update_asset_ticker(self, chat_id: int, old_ticker: str, new_ticker: str):
        """Manually update the ticker of a confirmed asset."""
        try:
            self.supabase.table("portfolio") \
                .update({"ticker": new_ticker}) \
                .eq("chat_id", chat_id) \
                .eq("ticker", old_ticker) \
                .execute()
            logger.info(f"Manual ticker update: {old_ticker} -> {new_ticker} for {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating asset ticker: {e}")
            return False

    def update_asset_core(self, chat_id: int, ticker: str, is_core: bool):
        """Toggle core status for an asset (Phase 9)."""
        try:
            self.supabase.table("portfolio") \
                .update({"is_core": is_core}) \
                .eq("chat_id", chat_id) \
                .eq("ticker", ticker.upper()) \
                .execute()
            logger.info(f"Core status update: {ticker} -> {is_core} for {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating core status: {e}")
            return False

    def register_ticker_failure(self, user_ticker: str):
        """
        Increment the valid failure count for a ticker.
        Used to identify persistent noise/bad tickers.
        """
        if not user_ticker: return
        try:
            t_u = user_ticker.upper()
            # 1. Get current count
            current = self.get_ticker_cache(t_u)
            new_count = 1
            if current:
                new_count = (current.get("fail_count", 0) or 0) + 1
            
            # 2. Upsert
            data = {
                "user_ticker": t_u,
                "fail_count": new_count,
                "last_verified_at": datetime.now(timezone.utc).isoformat()
            }
            # Preserve existing resolved maps if any
            if current and "resolved_ticker" in current:
                 data["resolved_ticker"] = current["resolved_ticker"]
            else:
                 data["resolved_ticker"] = t_u

            self.supabase.table("ticker_cache").upsert(data, on_conflict="user_ticker").execute()
            logger.info(f"Ticker Failure Registered: {t_u} -> Count {new_count}")
        except Exception as e:
            logger.warning(f"Failed to register ticker failure for {user_ticker}: {e}")

    def get_cached_price(self, ticker: str, max_age_minutes: int = 15):
        """
        Fetch price from ticker_cache if it's fresh enough.
        Safely handles cases where columns might be missing from the schema.
        """
        try:
            t_u = ticker.upper()
            # Wrap in try-except to handle partial schema deployments
            try:
                # [V12] Standard selection - confirmed columns exist
                response = self.supabase.table("ticker_cache").select("is_crypto, currency, last_price, last_price_at").eq("user_ticker", t_u).execute()
            except Exception as schema_err:
                logger.debug(f"Schema mismatch for ticker_cache in get_cached_price: {schema_err}")
                return None

            if not response.data:
                return None
            
            data = response.data[0]
            price = data.get("last_price")
            updated_at_str = data.get("last_price_at")
            
            if price and updated_at_str:
                updated_at = self._parse_iso_timestamp(updated_at_str)
                if not updated_at:
                    return None
                if datetime.now(updated_at.tzinfo) - updated_at < timedelta(minutes=max_age_minutes):
                    return {
                        "price": float(price),
                        "is_crypto": data.get("is_crypto", False),
                        "currency": data.get("currency", "USD")
                    }
            return None
        except Exception as e:
            logger.debug(f"Cached price fetch skipped for {ticker} (schema or hit): {e}")
            return None

    @staticmethod
    def _parse_iso_timestamp(ts: str):
        """Parse ISO timestamps returned by Supabase, handling trailing Z."""
        if not ts or not isinstance(ts, str):
            return None
        try:
            parsed = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return None

    def get_cached_prices_batch(self, tickers: list, max_age_minutes: int = 15) -> dict:
        """
        Fetch fresh cached prices for multiple tickers in a reduced number of queries.
        Returns:
            {
                "AAPL": {"price": 100.0, "is_crypto": False, "currency": "USD"},
                ...
            }
        """
        if not tickers:
            return {}
        try:
            targets = []
            seen = set()
            for ticker in tickers:
                if not isinstance(ticker, str) or not ticker.strip():
                    continue
                t_u = ticker.upper().strip()
                if t_u not in seen:
                    seen.add(t_u)
                    targets.append(t_u)

            if not targets:
                return {}

            results = {}
            now_utc = datetime.now(timezone.utc)
            batch_size = 80

            for i in range(0, len(targets), batch_size):
                chunk = targets[i:i + batch_size]
                try:
                    response = self.supabase.table("ticker_cache") \
                        .select("user_ticker, is_crypto, currency, last_price, last_price_at") \
                        .in_("user_ticker", chunk) \
                        .execute()
                except Exception as schema_err:
                    logger.debug(f"Schema mismatch for ticker_cache in get_cached_prices_batch: {schema_err}")
                    continue

                for row in response.data or []:
                    price = row.get("last_price")
                    updated_at = self._parse_iso_timestamp(row.get("last_price_at"))
                    if price is None or not updated_at:
                        continue
                    reference_now = datetime.now(updated_at.tzinfo) if updated_at.tzinfo else now_utc
                    if reference_now - updated_at <= timedelta(minutes=max_age_minutes):
                        user_ticker = (row.get("user_ticker") or "").upper()
                        if not user_ticker:
                            continue
                        results[user_ticker] = {
                            "price": float(price),
                            "is_crypto": row.get("is_crypto", False),
                            "currency": row.get("currency", "USD")
                        }
            return results
        except Exception as e:
            logger.debug(f"Batch cached price fetch skipped: {e}")
            return {}

    def save_ticker_price(self, ticker: str, price: float, is_crypto: bool = False, currency: str = "USD", resolved_ticker: str = None):
        """Save discovered price to ticker_cache. Safely ignores missing columns."""
        if self.dry_run:
            logger.debug(f"DRY_RUN: skip save_ticker_price for {ticker}")
            return
        try:
            t_u = ticker.upper()
            res_t = resolved_ticker if resolved_ticker else t_u
            data = {
                "user_ticker": t_u,
                "last_price": float(price),
                "last_price_at": datetime.now(timezone.utc).isoformat(),
                "is_crypto": is_crypto,
                "currency": currency,
                "resolved_ticker": res_t 
            }
            try:
                # Use a specific select check or just try-catch the PGRST204
                self.supabase.table("ticker_cache").upsert(data, on_conflict="user_ticker").execute()
            except Exception as schema_err:
                msg = str(schema_err)
                if "column" in msg or "PGRST204" in msg or "not found" in msg:
                    # If last_price or last_price_at fail, try basic upsert without them
                    logger.debug(f"Schema mismatch for ticker_cache columns, retrying basic upsert: {msg}")
                    basic_data = {
                        "user_ticker": t_u,
                        "resolved_ticker": t_u,
                        "is_crypto": is_crypto,
                        "currency": currency
                    }
                    self.supabase.table("ticker_cache").upsert(basic_data, on_conflict="user_ticker").execute()
                else:
                    raise schema_err
        except Exception as e:
            logger.debug(f"Error saving ticker price for {ticker}: {e}")

    def save_ticker_prices_batch(self, price_rows: list):
        """
        Upsert multiple ticker prices in batched requests.
        Input row schema:
            {
                "ticker": "AAPL",
                "price": 100.0,
                "is_crypto": False,
                "currency": "USD",
                "resolved_ticker": "AAPL"
            }
        """
        if self.dry_run:
            logger.debug("DRY_RUN: skip save_ticker_prices_batch")
            return
        if not price_rows:
            return

        payload = []
        now_iso = datetime.now(timezone.utc).isoformat()
        for row in price_rows:
            ticker = str(row.get("ticker", "")).upper().strip()
            if not ticker:
                continue
            resolved = str(row.get("resolved_ticker", ticker)).upper().strip() or ticker
            try:
                price_val = float(row.get("price", 0))
            except Exception:
                continue
            payload.append({
                "user_ticker": ticker,
                "last_price": price_val,
                "last_price_at": now_iso,
                "is_crypto": bool(row.get("is_crypto", False)),
                "currency": row.get("currency", "USD"),
                "resolved_ticker": resolved,
            })

        if not payload:
            return

        batch_size = 120
        for i in range(0, len(payload), batch_size):
            chunk = payload[i:i + batch_size]
            try:
                self.supabase.table("ticker_cache").upsert(chunk, on_conflict="user_ticker").execute()
            except Exception as schema_err:
                msg = str(schema_err)
                if "column" in msg or "PGRST204" in msg or "not found" in msg:
                    logger.debug(f"Schema mismatch in save_ticker_prices_batch, falling back row-wise: {msg}")
                    for item in chunk:
                        self.save_ticker_price(
                            item["user_ticker"],
                            item["last_price"],
                            is_crypto=item.get("is_crypto", False),
                            currency=item.get("currency", "USD"),
                            resolved_ticker=item.get("resolved_ticker")
                        )
                else:
                    logger.debug(f"Batch save ticker price failed: {schema_err}")

    def update_asset_quantity(self, chat_id: int, ticker: str, new_quantity: float):
        """Manually update the quantity of a confirmed asset."""
        try:
            self.supabase.table("portfolio") \
                .update({"quantity": new_quantity}) \
                .eq("chat_id", chat_id) \
                .eq("ticker", ticker) \
                .execute()
            logger.info(f"Manual quantity update: {ticker} -> {new_quantity} for {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating asset quantity: {e}")
            return False

    def get_recent_confirmed_portfolio(self, chat_id: int, minutes: int = 5):
        """Fetch portfolio items confirmed in the last N minutes."""
        try:
            # Note: 'updated_at' is used as proxy for 'recently added' since created_at is missing
            # Using updated_at is a good proxy for 'recently added'.
            time_threshold = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
            response = self.supabase.table("portfolio") \
                .select("*") \
                .eq("chat_id", chat_id) \
                .eq("is_confirmed", True) \
                .gte("updated_at", time_threshold) \
                .execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching recent confirmed items: {e}")
            return []

    def get_audit_stats(self):
        """
        Calculates performance metrics from signal_tracking table.
        Returns: { 'win_rate': 0.0, 'total_trades': 0, 'wins': 0, 'losses': 0, 'open': 0 }
        """
        try:
            response = self.supabase.table("signal_tracking").select("status").execute()
            signals = response.data
            
            total = len(signals)
            wins = sum(1 for s in signals if s['status'] == 'WIN')
            losses = sum(1 for s in signals if s['status'] == 'LOSS')
            open_sigs = sum(1 for s in signals if s['status'] == 'OPEN')
            
            closed_trades = wins + losses
            win_rate = (wins / closed_trades * 100) if closed_trades > 0 else 0.0
            
            return {
                "win_rate": round(win_rate, 1),
                "total_trades": total,
                "wins": wins,
                "losses": losses,
                "open": open_sigs,
                "closed": closed_trades
            }
        except Exception as e:
            logger.error(f"Audit Stats Error: {e}")
            return {"win_rate": 0, "total_trades": 0, "wins": 0, "losses": 0, "open": 0, "closed": 0}

    @staticmethod
    def _is_missing_column_error(error: Exception, column_name: str) -> bool:
        """Detect Supabase/Postgres column mismatch errors for a specific column."""
        msg = str(error).lower()
        col = column_name.lower()
        if "pgrst204" in msg:
            return True
        if col not in msg:
            return False
        return (
            "schema cache" in msg
            or "unknown column" in msg
            or ("column" in msg and ("not found" in msg or "does not exist" in msg))
        )

    @staticmethod
    def _is_sentiment_constraint_error(error: Exception) -> bool:
        """Detect legacy CHECK constraint errors for predictions.sentiment."""
        msg = str(error).lower()
        return (
            "predictions_sentiment_check" in msg
            or ("check constraint" in msg and "sentiment" in msg)
            or "'23514'" in msg
            or '"23514"' in msg
        )

    @staticmethod
    def _is_missing_table_error(error: Exception, table_name: str) -> bool:
        """Detect missing table/schema errors (common in Supabase REST responses)."""
        msg = str(error).lower()
        table = table_name.lower()
        return (
            table in msg
            and (
                "404" in msg
                or "pgrst205" in msg
                or "not found" in msg
                or "relation" in msg
                or "does not exist" in msg
                or "schema cache" in msg
            )
        )

    def _prediction_source_candidates(self, has_source_url: bool) -> List[Optional[str]]:
        """Return preferred source-link column order, keeping successful choice cached."""
        if not has_source_url:
            return [None]

        preferred = getattr(self, "_prediction_source_column", "auto")
        base_order: List[Optional[str]] = ["source_url", "source_news_url", "url", None]

        if preferred is None:
            return [None]
        if preferred == "auto":
            return base_order
        if preferred in base_order:
            return [preferred] + [col for col in base_order if col != preferred]
        return base_order

    def _disable_social_stats(self, error: Exception):
        """Disable Social Oracle persistence for this process after missing-table errors."""
        if getattr(self, "_social_stats_enabled", True):
            logger.warning(f"'social_stats' table unavailable. Disabling social velocity DB writes. ({error})")
        self._social_stats_enabled = False

    def log_prediction(self, ticker: str, sentiment: str, reasoning: str, prediction_sentence: str, confidence_score: float, source_url: str, risk_score: int = 5, target_price: str = None, upside_percentage: float = 0.0, stop_loss: float = None, take_profit: float = None, critic_verdict: str = None, critic_score: int = None, critic_reasoning: str = None):
        """Log a new signal/prediction to the database."""
        try:
            source_link = source_url or None
            source_candidates = self._prediction_source_candidates(has_source_url=bool(source_link))

            sentiment_value = (sentiment or "HOLD").upper().strip()
            legacy_sentiment_map = {"WATCH": "HOLD", "AVOID": "SELL"}
            if getattr(self, "_supports_extended_sentiment", None) is False:
                sentiment_value = legacy_sentiment_map.get(sentiment_value, sentiment_value)

            base_data = {
                "ticker": ticker,
                "sentiment": sentiment_value,
                "reasoning": reasoning,
                "prediction_sentence": prediction_sentence,
                "confidence_score": confidence_score,
                "risk_score": risk_score,
                "target_price": target_price,
                "upside_percentage": upside_percentage,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "critic_verdict": critic_verdict,
                "critic_score": critic_score,
                "critic_reasoning": critic_reasoning
            }

            sentiment_retry_done = False
            while True:
                insert_error = None

                for source_col in source_candidates:
                    data = dict(base_data)
                    if source_col and source_link:
                        data[source_col] = source_link

                    try:
                        response = self.supabase.table("predictions").insert(data).execute()
                        if source_link:
                            self._prediction_source_column = source_col
                        logger.info(f"Logged prediction for {ticker}: {data['sentiment']}")
                        return response.data[0]["id"] if response.data else None
                    except Exception as e:
                        insert_error = e

                        if source_col and self._is_missing_column_error(e, source_col):
                            logger.warning(f"DB mismatch for '{source_col}', trying fallback schema...")
                            continue
                        break

                fallback_sentiment = legacy_sentiment_map.get(base_data["sentiment"])
                if (
                    insert_error
                    and not sentiment_retry_done
                    and fallback_sentiment
                    and self._is_sentiment_constraint_error(insert_error)
                ):
                    logger.warning(
                        f"Legacy sentiment constraint detected. Retrying sentiment {base_data['sentiment']} as {fallback_sentiment}."
                    )
                    base_data["sentiment"] = fallback_sentiment
                    self._supports_extended_sentiment = False
                    sentiment_retry_done = True
                    continue

                raise insert_error
        except Exception as e:
            logger.error(f"Error logging prediction for {ticker}: {e}")
            return None

    def get_settings(self):
        """Fetch general application settings."""
        try:
            response = self.supabase.table("user_settings").select("*").limit(1).execute()
            if response.data:
                settings = response.data[0]
                # Defaults if column is missing from row (not schema)
                if 'risk_profile' not in settings:
                    settings['risk_profile'] = 'BALANCED'
                return settings
            return {"min_confidence": 0.50, "only_portfolio": False, "app_mode": "PREPROD", "risk_profile": "BALANCED"}
        except Exception as e:
            logger.error(f"Error fetching settings: {e}")
            return {"min_confidence": 0.50, "only_portfolio": False, "app_mode": "PREPROD", "risk_profile": "BALANCED"}

    def update_settings(self, min_confidence=None, only_portfolio=None, app_mode=None, risk_profile=None):
        """Update application settings."""
        try:
            updates = {}
            if min_confidence is not None: updates["min_confidence"] = min_confidence
            if only_portfolio is not None: updates["only_portfolio"] = only_portfolio
            if app_mode is not None: updates["app_mode"] = app_mode.upper()
            if risk_profile is not None: updates["risk_profile"] = risk_profile.upper()
            
            if not updates: return
            
            settings = self.get_settings()
            if settings and 'id' in settings:
                self.supabase.table("user_settings").update(updates).eq("id", settings['id']).execute()
            else:
                self.supabase.table("user_settings").insert(updates).execute()
                
            logger.info(f"Settings updated: {updates}")
            return True
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return False
            return False

    def update_settings_last_run(self, timestamp: str):
        """Update the last_successful_hunt_ts setting."""
        try:
            settings = self.get_settings()
            if "id" in settings:
                self.supabase.table("user_settings") \
                    .update({"last_successful_hunt_ts": timestamp}) \
                    .eq("id", settings["id"]) \
                    .execute()
                logger.info(f"Updated last_successful_hunt_ts to {timestamp}")
                return True
            return False
        except Exception as e:
            # Handle Schema Mismatches gracefully
            msg = str(e)
            if "PGRST204" in msg or "Could not find the" in msg:
                 logger.warning("⚠️ Schema Mismatch: 'last_successful_hunt_ts' column missing in user_settings. Skipping freshness update.")
                 # No need to crash or log ERROR.
                 return False
            logger.error(f"Error updating last run time: {e}")
            return False

    # --- API USAGE TRACKING ---
    def increment_api_counter(self, provider: str, run_id: str = None) -> dict:
        """
        Increment daily API call counter for a provider.
        Optionally tracks calls per run_id.
        Returns current counters dict.
        """
        try:
            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            settings = self.get_settings()
            
            # Get or initialize counters
            counters = settings.get("api_counters") or {}
            if not isinstance(counters, dict):
                counters = {}
            
            # Reset if new day
            if counters.get("date") != today:
                counters = {"date": today, "openrouter": 0, "gemini_fallback": 0, "runs": {}, "models": {}}
            
            # Ensure dicts exist
            if "runs" not in counters:
                counters["runs"] = {}
            if "models" not in counters:
                counters["models"] = {}
            
            # Increment provider counter
            counters[provider] = counters.get(provider, 0) + 1
            
            # Track per-run if run_id provided
            if run_id:
                if run_id not in counters["runs"]:
                    counters["runs"][run_id] = {"openrouter": 0, "gemini_fallback": 0, "started_at": datetime.now(timezone.utc).isoformat(), "model_used": None}
                counters["runs"][run_id][provider] = counters["runs"][run_id].get(provider, 0) + 1
            
            # Save to DB
            if "id" in settings:
                self.supabase.table("user_settings").update({"api_counters": counters}).eq("id", settings["id"]).execute()
            
            logger.info(f"API Counter: {provider} = {counters[provider]} (today: {today})")
            return counters
        except Exception as e:
            logger.error(f"Error incrementing API counter: {e}")
            return {"date": today, "openrouter": 0, "gemini_fallback": 0}

    def log_model_used(self, model_id: str):
        """
        Log which OpenRouter model was used.
        Tracks per-model usage for visibility.
        """
        try:
            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            settings = self.get_settings()
            counters = settings.get("api_counters") or {}
            
            if not isinstance(counters, dict) or counters.get("date") != today:
                counters = {"date": today, "openrouter": 0, "gemini_fallback": 0, "runs": {}, "models": {}}
            
            if "models" not in counters:
                counters["models"] = {}
            
            # Increment per-model counter
            counters["models"][model_id] = counters["models"].get(model_id, 0) + 1
            
            # Track last model used
            counters["last_model"] = model_id
            
            # Save to DB
            if "id" in settings:
                self.supabase.table("user_settings").update({"api_counters": counters}).eq("id", settings["id"]).execute()
            
            logger.info(f"Model Used: {model_id} (count: {counters['models'][model_id]})")
        except Exception as e:
            logger.error(f"Error logging model used: {e}")

    def get_api_usage(self) -> dict:
        """
        Get current API usage stats with reset time info.
        Returns: {
            "date": "2026-01-18",
            "openrouter": 45,
            "gemini_fallback": 2,
            "runs": {...},
            "models": {"deepseek/deepseek-r1-0528:free": 40, ...},
            "last_model": "deepseek/deepseek-r1-0528:free",
            "reset_at_utc": "2026-01-19T00:00:00Z",
            "hours_until_reset": 15.5
        }
        """
        try:
            from datetime import timezone
            now_utc = datetime.now(timezone.utc)
            today = now_utc.strftime('%Y-%m-%d')
            
            settings = self.get_settings()
            counters = settings.get("api_counters") or {}
            
            if not isinstance(counters, dict) or counters.get("date") != today:
                counters = {"date": today, "openrouter": 0, "gemini_fallback": 0, "runs": {}, "models": {}}
            
            # Calculate reset time (midnight UTC next day)
            tomorrow = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            hours_until_reset = (tomorrow - now_utc).total_seconds() / 3600
            
            # OpenRouter free tier info (varies by model, using general estimate)
            # Most free models have ~50-200 RPD
            counters["limits"] = {
                "openrouter_daily": "~50-200 per model",
                "gemini_fallback": 50
            }
            
            # Add reset time info
            counters["reset_at_utc"] = tomorrow.strftime('%Y-%m-%dT%H:%M:%SZ')
            counters["reset_at_local"] = f"{tomorrow.strftime('%Y-%m-%d')} 01:00 (Italy)"  # UTC+1
            counters["hours_until_reset"] = round(hours_until_reset, 1)
            
            return counters
        except Exception as e:
            logger.error(f"Error getting API usage: {e}")
            return {"date": today, "openrouter": 0, "gemini_fallback": 0, "limits": {}}

    def check_if_analyzed_recently(self, ticker: str, new_sentiment: str, hours: int = 24) -> bool:
        """
        Check if we should SKIP this alert.
        Returns TRUE (skip) if:
        - Ticker analyzed in last N hours AND Sentiment is SAME.
        Returns FALSE (allow) if:
        - Ticker not analyzed recently.
        - OR Sentiment has CHANGED (e.g. was HOLD, now BUY).
        """
        try:
            time_threshold = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            
            # Fetch most recent prediction for this ticker
            response = self.supabase.table("predictions") \
                .select("sentiment") \
                .eq("ticker", ticker) \
                .gte("created_at", time_threshold) \
                .order("created_at", desc=True) \
                .limit(1) \
                .execute()
            
            if not response.data:
                return False # No recent analysis, allow it.

            last_sentiment = response.data[0]['sentiment']
            
            if last_sentiment == new_sentiment:
                # NOTE: Changed behavior - we LOG duplicates but DO NOT SKIP them
                # User wants to be notified of signals even if sentiment is same (new news = actionable info)
                logger.info(f"Repeat Signal: {ticker} still {last_sentiment} (previous analysis confirms trend)")
                return False  # ALLOW - user wants all signals from new news
            else:
                logger.info(f"Sentiment Shift: {ticker} changed from {last_sentiment} to {new_sentiment}. Allowing.")
                return False # ALLOW (Change)

        except Exception as e:
            logger.error(f"Error checking recent analysis for {ticker}: {e}")
            return False

    def log_system_event(self, level: str, module: str, message: str):
        """Log system events to the logs table."""
        if self.dry_run:
            logger.debug(f"DRY_RUN: skip log_system_event {level} {module}")
            return
        try:
            data = {
                "level": level,
                "module": module,
                "message": message,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            self.supabase.table("logs").insert(data).execute()
        except Exception as e:
            print(f"Failed to log system event to DB: {e}") # Fallback to print

    def save_user_state(self, chat_id: int, key: str, state_data: dict):
        """Saves a temporary state to the DB (Vercel Stateless workaround)."""
        try:
            payload = {
                "level": "INFO", # Used INFO to bypass 'level' check constraint
                "module": f"STATE:{chat_id}:{key}",
                "message": json.dumps(state_data),
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            self.supabase.table("logs").insert(payload).execute()
            logger.info(f"💾 State saved for user {chat_id}: {key}")
            return True
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return False

    def get_user_state(self, chat_id: int, key: str):
        """Retrieves the latest state from the DB."""
        try:
            response = self.supabase.table("logs") \
                .select("message") \
                .eq("level", "INFO") \
                .eq("module", f"STATE:{chat_id}:{key}") \
                .order("created_at", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                logger.info(f"📂 State retrieved for user {chat_id}: {key}")
                return json.loads(response.data[0]['message'])
            return None
        except Exception as e:
            logger.error(f"Error getting state: {e}")
            return None

    # --- DISTRIBUTED LOCK (Idempotency Key) ---
    def acquire_hunt_lock(self, request_id: str, expiry_minutes: int = 2) -> bool:
        """
        Attempts to acquire a lock for the 'hunt' process using an Idempotency Key.
        - request_id: Unique ID from Telegram (update.update_id) to identify the specific click.
        
        Logic:
        1. Fetch last log.
        2. If last log has SAME request_id (whether LOCKED or RELEASED) -> BLOCK (Duplicate/Retry).
        3. If last log is LOCKED (and not expired) -> BLOCK (Busy).
        4. Else -> ACQUIRE.
        """
        try:
            # 1. Check state
            response = self.supabase.table("logs") \
                .select("*") \
                .eq("module", "HUNTER_LOCK") \
                .order("created_at", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                last_event = response.data[0]
                last_msg = last_event.get("message", "")
                created_at = self._parse_iso_timestamp(last_event.get("created_at"))
                if not created_at:
                    created_at = datetime.now(timezone.utc)
                
                # Check Idempotency (Prevent Retries of SAME request)
                # Message format: "LOCKED|123456" or "RELEASED|123456"
                if f"|{request_id}" in last_msg:
                    logger.warning(f"Duplicate Request Detected ({request_id}). Already processed/processing. Ignoring.")
                    return False # DUPLICATE

                # Check if currently locked by ANOTHER request
                if "LOCKED" in last_msg:
                    # Check expiry
                    now = datetime.now(timezone.utc)
                    if (now - created_at).total_seconds() < (expiry_minutes * 60):
                        logger.warning(f"Hunt Locked by another process. Active since {created_at}")
                        return False # BUSY
            
            # 2. Acquire
            self.log_system_event("INFO", "HUNTER_LOCK", f"LOCKED|{request_id}")
            return True
        except Exception as e:
            logger.error(f"Lock Error: {e}")
            return True # Fail-open

    def release_hunt_lock(self, request_id: str = "unknown"):
        """Releases the hunt lock with ID reference."""
        try:
             self.log_system_event("INFO", "HUNTER_LOCK", f"RELEASED|{request_id}")
        except: pass

    # --- ALERTS (Sentinel) ---
    def add_alert(self, chat_id: int, ticker: str, condition: str, price_threshold: float):
        """Adds a new price alert."""
        try:
            data = {
                "chat_id": chat_id,
                "ticker": ticker.upper(),
                "condition": condition.upper(), # ABOVE or BELOW
                "price_threshold": price_threshold,
                "is_active": True
            }
            self.supabase.table("alerts").insert(data).execute()
            return True
        except Exception as e:
            logger.error(f"Error adding alert: {e}")
            return False

    def get_active_alerts(self):
        """Fetches all active alerts."""
        try:
            response = self.supabase.table("alerts").select("*").eq("is_active", True).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching alerts: {e}")
            return []

    def deactivate_alert(self, alert_id: str, trigger_msg: str = None):
        """Marks alert as inactive (triggered)."""
        try:
            data = {"is_active": False, "triggered_at": datetime.now(timezone.utc).isoformat()}
            if trigger_msg:
                data["trigger_message"] = trigger_msg
            
            self.supabase.table("alerts").update(data).eq("id", alert_id).execute()
            return True
        except Exception as e:
            logger.error(f"Error deactivating alert: {e}")
            return False

    def delete_alert(self, alert_id: str):
        """Deletes an alert permanently."""
        try:
            self.supabase.table("alerts").delete().eq("id", alert_id).execute()
            return True
        except Exception as e:
            logger.error(f"Error deleting alert: {e}")
            return False

    def get_user_alerts(self, chat_id: int):
        """Fetches active alerts for a specific user (for UI/Bot)."""
        try:
            response = self.supabase.table("alerts") \
                .select("*") \
                .eq("chat_id", chat_id) \
                .eq("is_active", True) \
                .execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching user alerts: {e}")
            return []

    def save_backtest_result(self, result: dict):
        """
        Saves a backtest run result to the database.
        
        Args:
            result (dict): Dictionary containing backtest metrics
                           (ticker, period_days, start_date, end_date, 
                            starting_balance, ending_balance, pnl_percent, 
                            win_rate, total_trades, strategy_version)
        """
        try:
            data, count = self.supabase.table("backtest_results").insert(result).execute()
            if count:
                logger.info(f"💾 Backtest Saved: {result.get('ticker')} | PnL: {result.get('pnl_percent')}%")
                return True
            return False
        except Exception as e:
            logger.error(f"Error saving backtest result: {e}")
            return False

    # --- TRANSACTION TRACKING ---
    def log_transaction(self, ticker: str, action: str, quantity: float, price_per_unit: float, 
                        realized_pnl: float = None, notes: str = None):
        """
        Log a BUY or SELL transaction.
        
        Args:
            ticker: Asset ticker (e.g., 'RENDER', 'BTC-USD')
            action: 'BUY' or 'SELL'
            quantity: Number of units traded
            price_per_unit: Price per unit in EUR
            realized_pnl: (Optional) Realized P&L for SELL transactions
            notes: (Optional) User notes
        """
        try:
            total_value = quantity * price_per_unit
            data = {
                "ticker": ticker.upper(),
                "action": action.upper(),
                "quantity": quantity,
                "price_per_unit": price_per_unit,
                "total_value": total_value,
                "realized_pnl": realized_pnl,
                "notes": notes,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            response = self.supabase.table("transactions").insert(data).execute()
            logger.info(f"💰 Transaction logged: {action} {quantity} {ticker} @ €{price_per_unit:.2f}")
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error logging transaction: {e}")
            return None


    def check_and_lock_command(self, chat_id: int, command_hash: str, lock_window_seconds: int = 15) -> bool:
        """
        Distributed Lock: Checks if the exact same command was processed recently.
        Returns TRUE if we can proceed (lock acquired).
        Returns FALSE if we should ignore (duplicate).
        """
        try:
            # 1. Get current lock status
            settings = self.get_settings()
            
            # If no settings exist yet, create them (should normally exist)
            if not settings:
                self.update_settings(app_mode="PROD") # Default init
                settings = {}

            last_ts_str = settings.get("last_command_ts")
            last_hash = settings.get("last_command_hash")
            
            now = datetime.now(timezone.utc)
            
            if last_ts_str and last_hash:
                last_ts = self._parse_iso_timestamp(last_ts_str)
                if not last_ts:
                    last_ts = now
                
                elapsed = (now - last_ts).total_seconds()
                
                # Check Lock
                if elapsed < lock_window_seconds and last_hash == command_hash:
                    logger.warning(f"🔒 Distributed Lock: Duplicate command {command_hash} blocked (Elapsed: {elapsed:.1f}s)")
                    return False
            
            # 2. Acquire Lock (Update DB)
            # Depending on race conditions, this might still allow small window, but for Telegram retry (10s later) it's perfect.
            self.supabase.table("user_settings").update({
                "last_command_ts": now.isoformat(),
                "last_command_hash": command_hash
            }).eq("id", settings.get("id")).execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Distributed Lock Error: {e}")
            return True # Fail open (allow execution) to avoid blocking valid commands if DB fails

    def get_transactions(self, ticker: str = None, action: str = None, limit: int = 50):
        """
        Fetch transactions, optionally filtered by ticker and/or action.
        
        Args:
            ticker: Filter by specific ticker
            action: Filter by 'BUY' or 'SELL'
            limit: Max number of results
        """
        try:
            query = self.supabase.table("transactions").select("*")
            if ticker:
                query = query.eq("ticker", ticker.upper())
            if action:
                query = query.eq("action", action.upper())
            query = query.order("created_at", desc=True).limit(limit)
            response = query.execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching transactions: {e}")
            return []

    def get_total_realized_pnl(self):
        """Calculate total realized P&L from all SELL transactions."""
        try:
            response = self.supabase.table("transactions") \
                .select("realized_pnl") \
                .eq("action", "SELL") \
                .execute()
            total = sum(t.get("realized_pnl", 0) or 0 for t in response.data)
            return round(total, 2)
        except Exception as e:
            logger.error(f"Error calculating realized P&L: {e}")
            return 0.0

    # --- TICKER CACHE (V11 - Self-Learning Resolution) ---
    
    def get_ticker_cache(self, user_ticker: str) -> dict:
        """
        Get cached ticker resolution.
        Returns: {"resolved_ticker": "0700.HK", "is_crypto": False, "currency": "HKD"} or None
        """
        try:
            response = self.supabase.table("ticker_cache") \
                .select("resolved_ticker, is_crypto, currency, last_verified_at, fail_count") \
                .eq("user_ticker", user_ticker.upper()) \
                .limit(1) \
                .execute()
            
            if response.data:
                # Return the cache record regardless of fail_count
                # Logic to ignore/reject is moved to ticker_resolver.py
                return response.data[0]
            return None
        except Exception as e:
            logger.warning(f"Ticker cache lookup failed for {user_ticker}: {e}")
            return None

    def get_ticker_cache_batch(self, user_tickers: list) -> dict:
        """
        Get cached resolutions for a list of tickers in one query.
        Returns: { "USER_TICKER": { "resolved_ticker": "...", "fail_count": 0 } }
        """
        if not user_tickers:
            return {}
        try:
            # unique and upper
            targets = list(set([t.upper() for t in user_tickers]))
            results = {}
            
            # CHUNK REQUESTS (Max 50 per call) to avoid URL limit errors
            BATCH_SIZE = 50
            for i in range(0, len(targets), BATCH_SIZE):
                chunk = targets[i:i + BATCH_SIZE]
                try:
                    response = self.supabase.table("ticker_cache") \
                        .select("user_ticker, resolved_ticker, is_crypto, currency, last_verified_at, fail_count") \
                        .in_("user_ticker", chunk) \
                        .execute()
                    
                    if response.data:
                        for item in response.data:
                            results[item['user_ticker']] = item
                except Exception as chunk_err:
                     logger.error(f"Batch chunk failed: {chunk_err}")
            
            return results
        except Exception as e:
            logger.error(f"Batch ticker lookup failed: {e}")
            return {}
    
    def save_ticker_cache(self, user_ticker: str, resolved_ticker: str, is_crypto: bool = False, currency: str = "USD"):
        """
        Save successful ticker resolution to cache.
        """
        if self.dry_run:
            logger.debug(f"DRY_RUN: skip save_ticker_cache for {user_ticker}")
            return
        try:
            self.supabase.table("ticker_cache").upsert({
                "user_ticker": user_ticker.upper(),
                "resolved_ticker": resolved_ticker,
                "is_crypto": is_crypto,
                "currency": currency,
                "last_verified_at": datetime.now().isoformat(),
                "fail_count": 0
            }, on_conflict="user_ticker").execute()
            logger.info(f"Ticker cache saved: {user_ticker} -> {resolved_ticker}")
        except Exception as e:
            logger.warning(f"Failed to save ticker cache for {user_ticker}: {e}")
    
    def increment_ticker_fail(self, user_ticker: str):
        """
        Increment fail count for a cached ticker.
        If fail_count > 3, the cache entry will be ignored and re-discovered.
        """
        if self.dry_run:
            logger.debug(f"DRY_RUN: skip increment_ticker_fail for {user_ticker}")
            return
        try:
            response = self.supabase.table("ticker_cache") \
                .select("fail_count") \
                .eq("user_ticker", user_ticker.upper()) \
                .limit(1) \
                .execute()
            
            if response.data:
                # [PROTECTION] Skip incrementing for major assets
                from ticker_resolver import PROTECTED_TICKERS
                if user_ticker.upper() in PROTECTED_TICKERS:
                    return
                
                new_count = (response.data[0].get("fail_count", 0) or 0) + 1
                self.supabase.table("ticker_cache") \
                    .update({"fail_count": new_count}) \
                    .eq("user_ticker", user_ticker.upper()) \
                    .execute()
        except Exception as e:
            logger.warning(f"Failed to increment ticker fail for {user_ticker}: {e}")

    # --- REBALANCER HISTORY (V12 - Self-Learning) ---
    
    def save_rebalancer_suggestion(self, ticker: str, action: str, amount: float = None, 
                                    confidence: float = None, reasoning: str = None,
                                    regime: str = None, sector_rotation: str = None,
                                    ticker_rsi: float = None, ticker_pnl_pct: float = None,
                                    portfolio_value: float = None, price_at_suggestion: float = None):
        """
        Save an AI-generated rebalancing suggestion for tracking and learning.
        """
        if self.dry_run:
            logger.debug(f"DRY_RUN: skip save_rebalancer_suggestion for {ticker}")
            return
        try:
            self.supabase.table("rebalancer_history").insert({
                "ticker": ticker.upper(),
                "action": action.upper(),
                "suggested_amount_eur": amount,
                "confidence": confidence,
                "reasoning": reasoning,
                "regime": regime,
                "sector_rotation": sector_rotation,
                "ticker_rsi": ticker_rsi,
                "ticker_pnl_pct": ticker_pnl_pct,
                "portfolio_value_eur": portfolio_value,
                "price_at_suggestion": price_at_suggestion,
            }).execute()
            logger.info(f"Rebalancer suggestion saved: {action} {ticker}")
        except Exception as e:
            logger.warning(f"Failed to save rebalancer suggestion: {e}")
    
    def get_rebalancer_performance(self, days: int = 30) -> dict:
        """
        Get rebalancer performance stats for the last N days.
        Used to inject learning context into AI prompt.
        """
        try:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            response = self.supabase.table("rebalancer_history") \
                .select("ticker, action, was_executed, was_good_advice, outcome_pnl_pct") \
                .gte("created_at", cutoff) \
                .execute()
            
            if not response.data:
                return {"suggestions": 0, "executed": 0, "success_rate": 0}
            
            total = len(response.data)
            executed = sum(1 for r in response.data if r.get("was_executed"))
            good = sum(1 for r in response.data if r.get("was_good_advice"))
            
            # Calculate average PnL for executed trades
            pnl_values = [r.get("outcome_pnl_pct", 0) for r in response.data 
                          if r.get("outcome_pnl_pct") is not None]
            avg_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else 0
            
            return {
                "suggestions": total,
                "executed": executed,
                "good_advice": good,
                "success_rate": (good / executed * 100) if executed > 0 else 0,
                "avg_pnl_pct": avg_pnl
            }
        except Exception as e:
            logger.warning(f"Failed to get rebalancer performance: {e}")
            return {"suggestions": 0, "executed": 0, "success_rate": 0}
    
    def mark_suggestion_executed(self, ticker: str, action: str):
        """
        Mark a recent suggestion as executed when user follows it.
        """
        try:
            # Find the most recent unexecuted suggestion for this ticker/action
            response = self.supabase.table("rebalancer_history") \
                .select("id") \
                .eq("ticker", ticker.upper()) \
                .eq("action", action.upper()) \
                .is_("was_executed", "null") \
                .order("created_at", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                self.supabase.table("rebalancer_history") \
                    .update({"was_executed": True, "executed_at": datetime.now().isoformat()}) \
                    .eq("id", response.data[0]["id"]) \
                    .execute()
                logger.info(f"Marked suggestion as executed: {action} {ticker}")
        except Exception as e:
            logger.warning(f"Failed to mark suggestion as executed: {e}")
    
    # --- DASHBOARD STATS (L13) ---
    
    def get_ticker_cache_stats(self) -> dict:
        """
        Get ticker cache statistics for dashboard display.
        """
        try:
            # Total cached tickers
            total_response = self.supabase.table("ticker_cache") \
                .select("user_ticker", count="exact") \
                .execute()
            total_cached = total_response.count if hasattr(total_response, 'count') else len(total_response.data or [])
            
            # Recent cache entries (last 7 days)
            cutoff = (datetime.now() - timedelta(days=7)).isoformat()
            recent_response = self.supabase.table("ticker_cache") \
                .select("user_ticker, resolved_ticker, is_crypto, last_verified_at") \
                .gte("last_verified_at", cutoff) \
                .order("last_verified_at", desc=True) \
                .limit(10) \
                .execute()
            
            return {
                "total_cached": total_cached,
                "recent_entries": recent_response.data or [],
                "crypto_count": sum(1 for r in (recent_response.data or []) if r.get("is_crypto"))
            }
        except Exception as e:
            logger.warning(f"Failed to get ticker cache stats: {e}")
            return {"total_cached": 0, "recent_entries": [], "crypto_count": 0}
    
    def get_rebalancer_learning_stats(self) -> dict:
        """
        Get rebalancer learning statistics for dashboard display.
        """
        try:
            # All-time stats
            all_response = self.supabase.table("rebalancer_history") \
                .select("action, was_executed, was_good_advice, outcome_pnl_pct") \
                .execute()
            
            if not all_response.data:
                return {
                    "total_suggestions": 0,
                    "executed": 0,
                    "win_rate": 0,
                    "avg_pnl": 0,
                    "by_action": {}
                }
            
            data = all_response.data
            total = len(data)
            executed = sum(1 for r in data if r.get("was_executed"))
            good = sum(1 for r in data if r.get("was_good_advice"))
            
            # PnL stats
            pnl_values = [r.get("outcome_pnl_pct", 0) for r in data 
                          if r.get("outcome_pnl_pct") is not None]
            avg_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else 0
            
            # By action breakdown
            actions = {}
            for r in data:
                action = r.get("action", "UNKNOWN")
                if action not in actions:
                    actions[action] = {"count": 0, "executed": 0, "good": 0}
                actions[action]["count"] += 1
                if r.get("was_executed"):
                    actions[action]["executed"] += 1
                if r.get("was_good_advice"):
                    actions[action]["good"] += 1
            
            return {
                "total_suggestions": total,
                "executed": executed,
                "win_rate": (good / executed * 100) if executed > 0 else 0,
                "avg_pnl": avg_pnl,
                "by_action": actions
            }
        except Exception as e:
            logger.warning(f"Failed to get rebalancer learning stats: {e}")
            return {"total_suggestions": 0, "executed": 0, "win_rate": 0, "avg_pnl": 0, "by_action": {}}

    # =========================================================================
    # PERFORMANCE METRICS TRACKING
    # =========================================================================
    
    def save_run_metrics(self, metrics: dict):
        """
        Save performance metrics for a hunt run.
        Metrics dict should contain: total_time, ai_time, news_fetch_time,
        signals_count, model_used, json_repair_needed, repair_strategy, retry_count
        """
        try:
            data = {
                "run_date": datetime.now().isoformat(),
                "total_time_seconds": metrics.get("total_time", 0),
                "ai_time_seconds": metrics.get("ai_time", 0),
                "news_fetch_time_seconds": metrics.get("news_fetch_time", 0),
                "signals_count": metrics.get("signals_count", 0),
                "model_used": metrics.get("model_used", "unknown"),
                "json_repair_needed": metrics.get("json_repair_needed", False),
                "repair_strategy": metrics.get("repair_strategy", "none"),
                "retry_count": metrics.get("retry_count", 0),
                "news_items_processed": metrics.get("news_items_processed", 0)
            }
            self.supabase.table("run_metrics").insert(data).execute()
            logger.info(f"Run Metrics Saved: {metrics.get('signals_count', 0)} signals in {metrics.get('total_time', 0):.1f}s")
        except Exception as e:
            logger.warning(f"Failed to save run metrics: {e}")

    def get_run_metrics_summary(self, days: int = 7):
        """Get summary of run metrics for the last N days."""
        try:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            result = self.supabase.table("run_metrics").select("*").gte("run_date", cutoff).execute()
            
            if not result.data:
                return {"runs": 0, "avg_time": 0, "avg_signals": 0}
            
            runs = result.data
            return {
                "runs": len(runs),
                "avg_time": sum(r.get("total_time_seconds", 0) for r in runs) / len(runs),
                "avg_signals": sum(r.get("signals_count", 0) for r in runs) / len(runs),
                "avg_ai_time": sum(r.get("ai_time_seconds", 0) for r in runs) / len(runs),
                "repair_rate": sum(1 for r in runs if r.get("json_repair_needed")) / len(runs) * 100
            }
        except Exception as e:
            logger.warning(f"Failed to get run metrics summary: {e}")
            return {"runs": 0, "avg_time": 0, "avg_signals": 0}

    # =========================================================================
    # NEWS CACHING
    # =========================================================================
    
    def get_cached_news(self, url: str, ttl_hours: int = 2):
        """
        Get cached news content if fresh enough.
        Returns (content, is_cached) or (None, False) if not cached/expired.
        """
        try:
            result = self.supabase.table("news_cache").select("content,cached_at").eq("url", url).limit(1).execute()
            
            if result.data:
                cached = result.data[0]
                cached_at = datetime.fromisoformat(cached["cached_at"].replace("Z", "+00:00"))
                age_hours = (datetime.now(cached_at.tzinfo) - cached_at).total_seconds() / 3600
                
                if age_hours < ttl_hours:
                    logger.debug(f"News cache HIT: {url[:50]}... (age: {age_hours:.1f}h)")
                    return cached["content"], True
                else:
                    logger.debug(f"News cache EXPIRED: {url[:50]}... (age: {age_hours:.1f}h)")
            return None, False
        except Exception as e:
            logger.debug(f"News cache error: {e}")
            return None, False

    def save_news_cache(self, url: str, content: str):
        """Save news content to cache with current timestamp."""
        try:
            # Upsert to update if exists
            data = {
                "url": url,
                "content": content[:10000],  # Limit content size
                "cached_at": datetime.now().isoformat()
            }
            self.supabase.table("news_cache").upsert(data, on_conflict="url").execute()
            logger.debug(f"News cached: {url[:50]}...")
        except Exception as e:
            logger.debug(f"Failed to cache news: {e}")

    # --- SOCIAL ORACLE: VELOCITY TRACKING ---
    def log_social_mentions(self, ticker: str, reddit_mentions: int, source: str = "reddit"):
        """Save current mention count for velocity calculation."""
        if not getattr(self, "_social_stats_enabled", True):
            return

        try:
            self.supabase.table("social_stats").insert({
                "ticker": ticker,
                "mentions": reddit_mentions,
                "source": source,
                "created_at": datetime.now().isoformat()
            }).execute()
        except Exception as e:
            if self._is_missing_table_error(e, "social_stats"):
                self._disable_social_stats(e)
                return
            logger.debug(f"Failed to log social mentions for {ticker}: {e}")

    def get_social_history(self, ticker: str, hours: int = 12) -> List[Dict]:
        """Fetch historical mentions for velocity calculation."""
        if not getattr(self, "_social_stats_enabled", True):
            return []

        try:
            since = (datetime.now() - timedelta(hours=hours)).isoformat()
            resp = self.supabase.table("social_stats") \
                .select("mentions, created_at") \
                .eq("ticker", ticker) \
                .gt("created_at", since) \
                .order("created_at", desc=True) \
                .execute()
            return resp.data or []
        except Exception as e:
            if self._is_missing_table_error(e, "social_stats"):
                self._disable_social_stats(e)
                return []
            logger.debug(f"Failed to fetch social history for {ticker}: {e}")
            return []

    def log_ml_health(self, log_data: Dict):
        """Log ML prediction status to ml_health_logs."""
        if self.dry_run:
            logger.debug("DRY_RUN: skip log_ml_health")
            return
        try:
            self.supabase.table("ml_health_logs").insert(log_data).execute()
        except Exception as e:
            logger.debug(f"Failed to log ML health: {e}")

    def get_ml_health_summary(self, days: int = 7) -> Dict:
        """Fetch ML health stats for the last N days."""
        try:
            since = (datetime.now() - timedelta(days=days)).isoformat()
            resp = self.supabase.table("ml_health_logs") \
                .select("model_type, status") \
                .gt("created_at", since) \
                .execute()
            
            data = resp.data or []
            summary = {}
            for item in data:
                m_type = item['model_type']
                status = item['status']
                if m_type not in summary:
                    summary[m_type] = {'SUCCESS': 0, 'FAILURE': 0}
                summary[m_type][status] += 1
            
            # Calculate rates
            for m_type in summary:
                total = summary[m_type]['SUCCESS'] + summary[m_type]['FAILURE']
                summary[m_type]['success_rate'] = (summary[m_type]['SUCCESS'] / total) if total > 0 else 0
                
            return summary
        except Exception as e:
            logger.error(f"Failed to fetch ML health summary: {e}")
            return {}

if __name__ == "__main__":
    # Test connection
    try:
        db = DBHandler()
        logger.info("Supabase connected successfully.")
    except Exception as e:
        print(f"Supabase connection failed: {e}")
