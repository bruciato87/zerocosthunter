import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_handler import DBHandler

def clear_signals():
    print("🧹 Clearing Signals and Predictions...")
    db = DBHandler()
    
    # 1. Clear Predictions
    try:
        # Delete all rows where id is not null (effectively all)
        db.supabase.table("predictions").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        print("✅ Predictions table cleared.")
    except Exception as e:
        print(f"⚠️ Error clearing predictions: {e}")

    # 2. Clear Signal Tracking
    try:
        db.supabase.table("signal_tracking").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        print("✅ Signal Tracking table cleared.")
    except Exception as e:
        print(f"⚠️ Error clearing signal_tracking: {e}")
        
    # 3. Release Lock (Just in case)
    try:
        db.release_hunt_lock()
        print("✅ Distributed Lock force-released.")
    except: pass

    # 4. Clear Lock History (Crucial for Debounce Reset)
    try:
        db.supabase.table("logs").delete().eq("module", "HUNTER_LOCK").execute()
        print("✅ Lock History (logs) cleared for Debounce reset.")
    except Exception as e:
        print(f"⚠️ Error clearing lock logs: {e}")

if __name__ == "__main__":
    clear_signals()
