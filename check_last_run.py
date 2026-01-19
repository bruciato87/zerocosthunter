
import logging
from db_handler import DBHandler
import json
from datetime import datetime

def check_last_run():
    db = DBHandler()
    print("Fetching last run metrics...")
    
    try:
        # Fetch directly from Supabase to ensure we get the absolute latest
        res = db.supabase.table("run_metrics").select("*").order("run_date", desc=True).limit(1).execute()
        
        if not res.data:
            print("No run metrics found.")
            return

        run = res.data[0]
        
        # Format the output
        print("\n" + "="*40)
        print(f"🕵️‍♂️ LAST HUNT REPORT (ID: {run.get('id')})")
        print("="*40)
        print(f"🕒 Date: {run.get('run_date')}")
        print("-" * 20)
        print(f"🤖 AI Model:      {run.get('model_used')}")
        print(f"⏱️ Total Time:    {run.get('total_time_seconds', 0):.1f}s")
        print(f"   - AI Time:     {run.get('ai_time_seconds', 0):.1f}s")
        print(f"   - News Fetch:  {run.get('news_fetch_time_seconds', 0):.1f}s")
        print("-" * 20)
        print(f"🗞️ News Items:    {run.get('news_items_processed', 0)}")
        print(f"📡 Signals:       {run.get('signals_count', 0)}")
        print("-" * 20)
        print(f"🔧 JSON Repair:   {run.get('json_repair_needed', False)}")
        print(f"🛠️ Strategy:      {run.get('repair_strategy', 'none')}")
        print(f"🔄 Retries:       {run.get('retry_count', 0)}")
        print("="*40 + "\n")

    except Exception as e:
        print(f"Error fetching metrics: {e}")

if __name__ == "__main__":
    check_last_run()
