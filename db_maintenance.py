"""
Database Maintenance Module
============================
Monitors Supabase storage usage and performs automatic cleanup
when approaching the 500MB free tier limit.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger("DBMaintenance")

# Supabase Free Tier Limit
FREE_TIER_LIMIT_MB = 500
WARNING_THRESHOLD_MB = 400  # Start warning at 80%
CLEANUP_THRESHOLD_MB = 450  # Start cleanup at 90%


class DBMaintenance:
    """
    Database maintenance for storage monitoring and cleanup.
    """
    
    def __init__(self):
        from db_handler import DBHandler
        self.db = DBHandler()
        
        # Cleanup policies: (table, max_age_days, priority)
        # Priority: 1 = clean first, 5 = clean last
        self.cleanup_policies = [
            ("system_log", 7, 1),           # Logs older than 7 days - clean first
            ("memory", 90, 2),              # Memory older than 90 days
            ("backtest_results", 60, 2),    # Backtest results older than 60 days
            ("signal_tracking", 180, 3),    # Signals older than 180 days
            ("paper_trades", 90, 3),        # Paper trades older than 90 days
            ("news_seen", 30, 4),           # News tracking older than 30 days
        ]
    
    def get_database_size_mb(self) -> Optional[float]:
        """
        Get total database size in MB.
        Uses Supabase RPC or pg_database_size.
        """
        try:
            # Method 1: Try Supabase stats API (if available)
            # This may not work on all Supabase tiers
            
            # Method 2: Query individual table sizes
            tables = [
                "portfolio", "signal_tracking", "system_log", 
                "alerts", "paper_portfolio", "paper_trades",
                "backtest_results", "memory", "news_seen", "user_settings"
            ]
            
            total_rows = 0
            table_stats = {}
            
            for table in tables:
                try:
                    result = self.db.supabase.table(table).select("*", count="exact").limit(0).execute()
                    row_count = result.count if hasattr(result, 'count') else 0
                    table_stats[table] = row_count
                    total_rows += row_count
                except Exception:
                    pass  # Table might not exist
            
            # Estimate: ~1KB per row average (conservative)
            # Embeddings: ~6KB per row (768 floats * 4 bytes + overhead)
            estimated_mb = (total_rows * 1) / 1024  # 1KB per row
            
            # Adjust for memory table with embeddings
            if "memory" in table_stats:
                memory_rows = table_stats["memory"]
                estimated_mb += (memory_rows * 5) / 1024  # Extra 5KB for embeddings
            
            logger.info(f"📊 DB Size Estimate: {estimated_mb:.1f}MB ({total_rows} total rows)")
            return estimated_mb
            
        except Exception as e:
            logger.error(f"Failed to get DB size: {e}")
            return None
    
    def get_table_stats(self) -> Dict[str, int]:
        """Get row counts for all tables."""
        stats = {}
        tables = [
            "portfolio", "signal_tracking", "system_log", 
            "alerts", "paper_portfolio", "paper_trades",
            "backtest_results", "memory", "news_seen", "user_settings"
        ]
        
        for table in tables:
            try:
                result = self.db.supabase.table(table).select("*", count="exact").limit(0).execute()
                stats[table] = result.count if hasattr(result, 'count') else 0
            except Exception:
                stats[table] = -1  # Error indicator
        
        return stats
    
    def check_storage_health(self) -> Dict:
        """
        Check storage health and return status report.
        """
        size_mb = self.get_database_size_mb()
        
        if size_mb is None:
            return {
                "status": "unknown",
                "message": "Could not determine database size",
                "size_mb": 0,
                "limit_mb": FREE_TIER_LIMIT_MB,
                "usage_percent": 0
            }
        
        usage_pct = (size_mb / FREE_TIER_LIMIT_MB) * 100
        
        if size_mb >= CLEANUP_THRESHOLD_MB:
            status = "critical"
            message = f"⚠️ Storage critical! {size_mb:.1f}MB / {FREE_TIER_LIMIT_MB}MB ({usage_pct:.0f}%) - Cleanup required!"
        elif size_mb >= WARNING_THRESHOLD_MB:
            status = "warning"
            message = f"⚡ Storage warning: {size_mb:.1f}MB / {FREE_TIER_LIMIT_MB}MB ({usage_pct:.0f}%)"
        else:
            status = "healthy"
            message = f"✅ Storage healthy: {size_mb:.1f}MB / {FREE_TIER_LIMIT_MB}MB ({usage_pct:.0f}%)"
        
        return {
            "status": status,
            "message": message,
            "size_mb": size_mb,
            "limit_mb": FREE_TIER_LIMIT_MB,
            "usage_percent": usage_pct
        }
    
    def cleanup_old_records(self, force: bool = False) -> Dict[str, int]:
        """
        Delete old records based on cleanup policies.
        Only runs if storage is above threshold (or force=True).
        
        Returns dict of {table: deleted_count}
        """
        health = self.check_storage_health()
        
        if not force and health["status"] == "healthy":
            logger.info("Storage healthy, skipping cleanup")
            return {}
        
        logger.warning(f"🧹 Starting cleanup... {health['message']}")
        deleted = {}
        
        # Sort by priority (clean lower priority first for critical, higher priority tables can wait)
        sorted_policies = sorted(self.cleanup_policies, key=lambda x: x[2])
        
        for table, max_age_days, priority in sorted_policies:
            try:
                cutoff_date = (datetime.now() - timedelta(days=max_age_days)).isoformat()
                
                # Determine date column (different tables use different names)
                date_column = "created_at"
                if table == "memory":
                    date_column = "event_date"
                elif table == "signal_tracking":
                    date_column = "detected_at"
                elif table == "backtest_results":
                    date_column = "run_at"
                elif table == "paper_trades":
                    date_column = "trade_date"
                
                # Get count before delete
                before = self.db.supabase.table(table).select("*", count="exact").lt(date_column, cutoff_date).limit(0).execute()
                count_to_delete = before.count if hasattr(before, 'count') else 0
                
                if count_to_delete > 0:
                    # Delete old records
                    self.db.supabase.table(table).delete().lt(date_column, cutoff_date).execute()
                    deleted[table] = count_to_delete
                    logger.info(f"🗑️ Deleted {count_to_delete} old records from {table} (>{max_age_days} days)")
                
            except Exception as e:
                logger.warning(f"Cleanup failed for {table}: {e}")
                deleted[table] = -1
        
        total_deleted = sum(v for v in deleted.values() if v > 0)
        logger.info(f"🧹 Cleanup complete: {total_deleted} total records deleted")
        
        return deleted
    
    def run_maintenance(self) -> str:
        """
        Run full maintenance check and cleanup if needed.
        Returns a status message.
        """
        health = self.check_storage_health()
        
        if health["status"] == "critical":
            deleted = self.cleanup_old_records(force=True)
            return f"{health['message']}\n🧹 Cleaned: {sum(v for v in deleted.values() if v > 0)} records"
        elif health["status"] == "warning":
            return health["message"]
        else:
            return health["message"]


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    maint = DBMaintenance()
    
    print("="*50)
    print("Database Health Check")
    print("="*50)
    
    health = maint.check_storage_health()
    print(f"Status: {health['status']}")
    print(f"Message: {health['message']}")
    print(f"Size: {health['size_mb']:.1f}MB / {health['limit_mb']}MB")
    print(f"Usage: {health['usage_percent']:.1f}%")
    
    print("\n" + "="*50)
    print("Table Statistics")
    print("="*50)
    stats = maint.get_table_stats()
    for table, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {table}: {count} rows")
