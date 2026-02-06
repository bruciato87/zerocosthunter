import logging
import datetime
import os
from typing import Dict, Optional
from db_handler import DBHandler

logger = logging.getLogger(__name__)

class MLHealthMonitor:
    """Monitors the health and performance of ML models by tracking prediction status."""
    
    def __init__(self, db: Optional[DBHandler] = None):
        self.db = db or DBHandler()
        self.dry_run = os.environ.get("DRY_RUN", "").strip().lower() in {"1", "true", "yes", "on"}
        
    def log_prediction(self, model_type: str, status: str, error: Optional[str] = None):
        """
        Log a prediction attempt to the database.
        
        Args:
            model_type: 'classifier', 'regressor', or 'lstm'
            status: 'SUCCESS' or 'FAILURE'
            error: Optional error message if status is 'FAILURE'
        """
        if self.dry_run:
            logger.debug(f"DRY_RUN: skip ML health DB log for {model_type}:{status}")
            return
        try:
            timestamp = datetime.datetime.now().isoformat()
            
            # Log to standard logging
            if status == 'SUCCESS':
                logger.debug(f"ML Health: Prediction success for {model_type}")
            else:
                logger.error(f"ML Health: Prediction failure for {model_type} - {error}")
                
            # Log to DB (Table: ml_health_logs)
            # We assume this table exists or will be created via migration
            log_data = {
                'model_type': model_type,
                'status': status,
                'error_msg': error,
                'created_at': timestamp
            }
            
            # For now, we simulate DB logging or use a dedicated method in DBHandler
            # (I will add this method to DBHandler next)
            self.db.log_ml_health(log_data)
            
        except Exception as e:
            logger.warning(f"Failed to log ML health: {e}")

    def get_health_summary(self, days: int = 7) -> Dict:
        """Retrieve a summary of success rates for the last N days."""
        try:
            return self.db.get_ml_health_summary(days)
        except Exception as e:
            logger.error(f"Failed to get ML health summary: {e}")
            return {}
