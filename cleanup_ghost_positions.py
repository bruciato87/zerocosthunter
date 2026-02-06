import logging
from db_handler import DBHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CleanupTool")

def cleanup_paper_portfolio():
    db = DBHandler()
    logger.info("--- Cleaning up Ghost Positions ---")
    
    try:
        # 1. Fetch all paper positions
        res = db.supabase.table("paper_portfolio").select("*").execute()
        positions = res.data
        
        if not positions:
            logger.info("No paper positions found.")
            return

        logger.info(f"Found {len(positions)} paper positions.")
        
        # 2. Identify stale/ghost positions (e.g., XRP as requested)
        for pos in positions:
            ticker = pos['ticker']
            p_id = pos['id']
            
            # User specifically mentioned XRP
            if ticker == "XRP":
                logger.info(f"Removing ghost position: {ticker} (ID: {p_id})")
                db.supabase.table("paper_portfolio").delete().eq("id", p_id).execute()
        
        # 3. Handle signal_tracking cleanup (Optional but good for consistency)
        # We find WIN/LOSS signals that might be cluttering the Auditor
        res_sig = db.supabase.table("signal_tracking").select("*").in_("status", ["WIN", "LOSS", "EXPIRED"]).execute()
        stale_signals = res_sig.data
        
        if stale_signals:
            logger.info(f"Found {len(stale_signals)} terminal signals (WIN/LOSS/EXPIRED).")
            # For now, let's NOT delete these as they are good for history, 
            # but we know they exist.
            
        logger.info("Cleanup complete.")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

if __name__ == "__main__":
    cleanup_paper_portfolio()
