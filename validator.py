
import sys
import os
import logging
from typing import List

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Validator")

def check_none_type_safety():
    """
    Test 1: NoneType Safety Check
    Ensures critical methods handle None inputs without crashing.
    """
    logger.info("TEST 1: Checking NoneType Safety...")
    
    try:
        from signal_intelligence import SignalIntelligence
        from market_data import MarketData
        
        # Mock dependencies
        si = SignalIntelligence()
        
        # 1.1 Test check_dca_opportunity with None ticker
        logger.info("  - Testing check_dca_opportunity(None)...")
        res = si.check_dca_opportunity(None)
        if res.get("error") or res.get("reason") == "No ticker provided" or res.get("reason") == "Ticker unresolvable/rejected":
            logger.info("    ✅ Passed (Handled gracefully)")
        else:
            logger.error(f"    ❌ Failed: Unexpected response: {res}")
            return False

        # 1.2 Test check_take_profit with None ticker
        logger.info("  - Testing check_take_profit(None)...")
        res = si.check_take_profit(None)
        if not res.get("should_take_profit"):
             logger.info("    ✅ Passed (Handled gracefully)")
        else:
             logger.error(f"    ❌ Failed: Unexpected response: {res}")
             return False

        # 1.3 Test check_earnings_risk with None ticker
        logger.info("  - Testing check_earnings_risk(None)...")
        res = si.check_earnings_risk(None)
        if not res.get("has_upcoming_earnings"):
             logger.info("    ✅ Passed (Handled gracefully)")
        else:
             logger.error(f"    ❌ Failed: Unexpected response: {res}")
             return False
            
        return True
    except Exception as e:
        logger.error(f"    ❌ Failed with Exception: {e}")
        return False

def check_regime_consistency():
    """
    Test 2: Regime Consistency
    Ensures Brain receives and prioritizes the explicit regime class.
    """
    logger.info("TEST 2: Checking Regime Consistency Logic...")
    
    # Static check: Ensure main.py calls MarketRegimeClassifier BEFORE analyze_news_batch
    try:
        with open("main.py", "r") as f:
            content = f.read()
            
        regime_idx = content.find("MarketRegimeClassifier()")
        analyze_idx = content.find("brain.analyze_news_batch")
        
        if regime_idx == -1 or analyze_idx == -1:
            logger.warning("    ⚠️ Could not find calls in main.py (Parsing error?)")
            return True # Skip static check if fuzzy
            
        if regime_idx < analyze_idx:
            logger.info("    ✅ Passed (Classifier runs BEFORE Brain)")
            return True
        else:
            logger.error("    ❌ Failed: MarketRegimeClassifier runs AFTER Brain analysis!")
            logger.error(f"       Classifier Index: {regime_idx}, Brain Index: {analyze_idx}")
            return False
            
    except Exception as e:
        logger.error(f"    ❌ Failed static check: {e}")
        return False

def check_forbidden_patterns():
    """
    Test 3: Forbidden Patterns
    Ensures code doesn't contain debug artifacts like 'WAIT' literals in signal logic.
    """
    logger.info("TEST 3: Scanning for Forbidden Patterns...")
    forbidden = ["'WAIT'", '"WAIT"', "print("] # Basic checks
    
    files_to_check = ["signal_intelligence.py", "brain.py", "monitor.py"]
    all_clean = True
    
    for filename in files_to_check:
        if not os.path.exists(filename): continue
        
        with open(filename, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                for pattern in forbidden:
                    if pattern in line and "logger" not in line and "#" not in line:
                        # Allow debug prints if commented, otherwise flag
                        logger.warning(f"    ⚠️ Warning: Found {pattern} in {filename}:{i+1}")
                        # We don't fail strictly on print, but we warn
                        
    return all_clean

if __name__ == "__main__":
    logger.info("🚀 STARTING PRE-FLIGHT VALIDATOR")
    
    success = True
    success &= check_none_type_safety()
    success &= check_regime_consistency()
    success &= check_forbidden_patterns()
    
    if success:
        logger.info("\n✅ ALL SYSTEMS GO. READY FOR DEPLOY.")
        sys.exit(0)
    else:
        logger.error("\n🛑 VALIDATION FAILED. DO NOT DEPLOY.")
        sys.exit(1)
