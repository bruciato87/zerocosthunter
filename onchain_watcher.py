import requests
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("OnChainWatcher")

class OnChainWatcher:
    """
    Level 15: On-Chain Intelligence Module.
    Monitors DEX liquidity, volume, and trending pairs via DexScreener.
    """
    
    BASE_URL = "https://api.dexscreener.com/latest/dex"

    def __init__(self):
        self.session = requests.Session()

    def get_token_data(self, ticker: str) -> Optional[Dict]:
        """
        Fetches on-chain data for a ticker from DexScreener.
        Searches for the ticker and returns the most liquid pair.
        """
        logger.info(f"Fetching On-Chain data for {ticker}...")
        try:
            url = f"{self.BASE_URL}/search?q={ticker}"
            resp = self.session.get(url, timeout=10)
            if resp.status_code != 200:
                logger.warning(f"⚠️ DexScreener API error: {resp.status_code}")
                return None
            
            data = resp.json()
            pairs = data.get("pairs", [])
            
            if not pairs:
                logger.info(f"No DEX pairs found for {ticker}")
                return None
            
            # Sort by liquidity descending to find the main pair
            valid_pairs = [p for p in pairs if p.get("liquidity", {}).get("usd", 0) > 1000]
            if not valid_pairs:
                return None
                
            main_pair = sorted(valid_pairs, key=lambda x: x.get("liquidity", {}).get("usd", 0), reverse=True)[0]
            
            return {
                "pair_address": main_pair.get("pairAddress"),
                "base_token": main_pair.get("baseToken", {}).get("symbol"),
                "price_usd": float(main_pair.get("priceUsd", 0)),
                "liquidity_usd": float(main_pair.get("liquidity", {}).get("usd", 0)),
                "volume_24h": float(main_pair.get("volume", {}).get("h24", 0)),
                "price_change_24h": float(main_pair.get("priceChange", {}).get("h24", 0)),
                "chain": main_pair.get("chainId")
            }
            
        except Exception as e:
            logger.error(f"Error fetching on-chain data: {e}")
            return None

    def get_onchain_context(self, ticker: str) -> str:
        """Returns a string representation of the on-chain status for a ticker."""
        data = self.get_token_data(ticker)
        if not data:
            return f"[ON-CHAIN ORACLE: {ticker} -> No significant DEX activity]"
        
        liquidity = data['liquidity_usd']
        volume = data['volume_24h']
        change = data['price_change_24h']
        
        status = "🟢 HEALTHY" if liquidity > 100_000 and volume > 50_000 else "🟡 THIN"
        if change > 20: status += " 🚀 RALLYING"
        elif change < -20: status += " ⚠️ DUMPING"
        
        return (
            f"[ON-CHAIN ORACLE: {ticker} -> {status}]\n"
            f"- Liquidity: ${liquidity:,.0f} | Vol 24h: ${volume:,.0f}\n"
            f"- 24h Change: {change:+.1f}% | Chain: {data['chain'].upper()}"
        )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    watcher = OnChainWatcher()
    print(watcher.get_onchain_context("SOL"))
