
from db_handler import DBHandler

db = DBHandler()
# Fetch the rogue BTC signal
response = db.supabase.table("signal_tracking").select("*").eq("ticker", "BTC").execute()
for row in response.data:
    if row.get('pnl_percent', 0) > 100000:
        print(f"Deleting corrupted record {row.get('id')} with PnL {row.get('pnl_percent')}%")
        db.supabase.table("signal_tracking").delete().eq("id", row.get('id')).execute()
        print("Deleted.")
    else:
        print(f"Skipping valid record: {row.get('pnl_percent')}%")

