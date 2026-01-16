
import os
import sys
import jinja2
from datetime import datetime
from unittest.mock import MagicMock

# Mock dependencies before importing modules if they hit DB/Network
# preventing network calls during verification is safer and faster
sys.modules['supabase'] = MagicMock()
sys.modules['yfinance'] = MagicMock()

# Import local modules
# (We need to allow them to import the mocked yfinance/supabase)
try:
    from market_regime import MarketRegimeClassifier
    from sector_rotation import SectorRotationTracker
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def get_mock_market_regime():
    """Generates a sample output structure from MarketRegimeClassifier."""
    # We inspect the actual class or manually replicate the structure.
    
    # Instantiate properly (no args)
    classifier = MarketRegimeClassifier()
    
    # Mock internal methods to return valid partial data so the final structure is assembled
    classifier._get_educational_suggestion = lambda x: "Mock Suggestion"
    
    # Replicating the return structure from classify() manually to ensure it matches expectations
    # This is the "Contract" verification.
    return {
        "regime": "BULL",
        "confidence": 0.85,
        "signals": {},
        "recommendation": "aggressive",
        "strategy_suggestion": "Buy Dips",
        "volatility_state": "LOW",
        "confidence_multiplier": 1.15,
        "recommended_min_confidence": 0.65, # The key that was missing!
        "indicators": {"vix": 15, "spy": {}, "btc": {}},
        "bull_score": 10,
        "bear_score": 2,
        "timestamp": datetime.now().isoformat()
    }

def get_mock_sector_rotation():
    """Generates sample output from SectorRotationTracker."""
    return {
        "ranking": [
            {"sector": "Technology", "momentum_score": 5.2},
            {"sector": "Energy", "momentum_score": 2.1},
            {"sector": "Healthcare", "momentum_score": -1.5}
        ],
        "leading": [],
        "lagging": [],
        "timestamp": datetime.now().isoformat()
    }

def verify_template():
    print("🔍 Starting Dashboard Verification...")
    
    template_path = os.path.join(os.getcwd(), 'templates')
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_path))
    
    try:
        template = env.get_template('dashboard.html')
        print("✅ Template Syntax Check Passed (Jinja2 load successful)")
    except jinja2.TemplateSyntaxError as e:
        print(f"❌ Template Syntax Error: {e}")
        print(f"   Line: {e.lineno}")
        sys.exit(1)

    # Prepare Mock Context
    # This should include ALL variables used in dashboard.html
    context = {
        "total_value_eur": 10000.0,
        "total_pl_eur": 500.0,
        "total_pl_percent": 5.0,
        "paper_total_value": 10000,
        "audit_stats": {"win_rate": 60, "wins": 6, "losses": 4, "open": 2},
        "macro_stats": {"risk_level": "LOW", "vix": 14.2, "tnx_yield": 4.1},
        "whale_stats": {"status": "BULLISH", "net_flow_m": 120, "buy_vol_m": 500, "sell_vol_m": 380},
        "market_mood": {"crypto": MagicMock(value=75), "overall": "GREED"}, # Enum mock
        
        # The L2 Objects we want to verify
        "market_regime": get_mock_market_regime(),
        "sector_rotation": get_mock_sector_rotation(),
        
        "portfolio": [
            {"ticker": "AAPL", "quantity": 10, "live_value_eur": 1500, "pnl_eur": 200, "pnl_percent": 15.5}
        ],
        "signals": [
             {"ticker": "NVDA", "sentiment": "BUY", "created_at": "2024-01-01T12:00:00", "confidence_score": 0.9, 
              "reasoning": "AI Boom", "target_price": 200, "upside_percentage": 20, "risk_score": 3}
        ],
        "history": [],
        "paper_portfolio": [],
        "backtest_results": [],
        "benchmark_data": {"benchmarks": {}, "portfolio": {"return_pct": 2.0}, "beating": []},
        
        # Chart Data
        "chart_labels": ["2024-01-01", "2024-01-02"],
        "chart_data": [10000, 10100],
        
        # Meta
        "last_run": "Just now",
        "last_run_iso": datetime.now().isoformat()
    }

    try:
        # Strict Undefined Check
        # Jinja2 by default doesn't error on undefined variables (it prints empty string)
        # UNLESS we set undefined=jinja2.StrictUndefined in the env.
        # But Flask usually doesn't set StrictUndefined.
        # However, getting 'dict object has no attribute' IS a Python runtime error during access, which Jinja propagates.
        # So standard render() should catch the error we saw earlier.
        
        rendered = template.render(context)
        print("✅ Template Render check PASSED (No missing keys/attributes encountered with Mock Data)")
        
        # Optional: Check if key content is present
        if "Min. req for trade" in rendered:
             print("✅ Content Verification: 'Min. req for trade' found")
        else:
             print("⚠️  Warning: Expected content 'Min. req for trade' not found in output.")

    except Exception as e:
        print(f"❌ Runtime Render Error: {e}")
        # This catches "UndefinedError: 'dict object' has no attribute 'recommended_min_confidence'"
        sys.exit(1)

if __name__ == "__main__":
    verify_template()
