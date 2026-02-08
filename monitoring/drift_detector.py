"""
Evidently AI Drift Detection for Fraud Detection System
=======================================================
Monitors feature, prediction, and target drift to ensure model reliability.
Triggers retraining pipelines when significant distribution shifts are detected.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple
import logging
import redis

# Evidently for drift detection
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    DatasetDriftMetric,
    DataDriftTable,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
DRIFT_THRESHOLD = float(os.getenv('DRIFT_THRESHOLD', 0.15))  # 15% drift triggers alert
REPORT_OUTPUT_PATH = os.getenv('DRIFT_REPORT_PATH', './monitoring/drift_reports')

# Feature columns to monitor
NUMERIC_FEATURES = [
    'amount', 'customer_age', 'transaction_count_1h', 
    'transaction_count_24h', 'avg_amount_7d', 'risk_score'
]
CATEGORICAL_FEATURES = [
    'merchant_category', 'location_country', 'channel', 'card_present'
]


class DriftDetector:
    """
    Drift detector using Evidently AI.
    """
    
    def __init__(self):
        self.redis_client = redis.Redis(
            host=REDIS_HOST, 
            port=REDIS_PORT,
            decode_responses=True
        )
        
        # Column mapping for Evidently
        self.column_mapping = ColumnMapping(
            target='is_fraud',
            prediction='prediction',
            numerical_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES
        )
        
        os.makedirs(REPORT_OUTPUT_PATH, exist_ok=True)
    
    def get_reference_data(self) -> pd.DataFrame:
        """
        Load reference data (training distribution).
        """
        reference_path = os.getenv(
            'REFERENCE_DATA_PATH', 
            './features/data/training_data.parquet'
        )
        
        if os.path.exists(reference_path):
            df = pd.read_parquet(reference_path)
            # Sample for efficiency
            if len(df) > 10000:
                df = df.sample(n=10000, random_state=42)
            return df
        else:
            logger.warning(f"Reference data not found at {reference_path}")
            return self._generate_synthetic_reference()
    
    def _generate_synthetic_reference(self) -> pd.DataFrame:
        """Generate synthetic reference data for testing."""
        np.random.seed(42)
        n = 5000
        
        return pd.DataFrame({
            'amount': np.random.exponential(100, n),
            'customer_age': np.random.normal(40, 15, n).clip(18, 80),
            'transaction_count_1h': np.random.poisson(2, n),
            'transaction_count_24h': np.random.poisson(10, n),
            'avg_amount_7d': np.random.exponential(80, n),
            'risk_score': np.random.beta(2, 5, n),
            'merchant_category': np.random.choice(
                ['retail', 'food', 'travel', 'entertainment', 'utilities'], n
            ),
            'location_country': np.random.choice(
                ['GB', 'US', 'DE', 'FR', 'ES'], n, p=[0.6, 0.15, 0.1, 0.1, 0.05]
            ),
            'channel': np.random.choice(['online', 'in_store', 'atm'], n),
            'card_present': np.random.choice([True, False], n, p=[0.7, 0.3]),
            'is_fraud': np.random.choice([0, 1], n, p=[0.97, 0.03]),
            'prediction': np.random.choice([0, 1], n, p=[0.96, 0.04]),
        })
    
    def get_current_data(self, hours: int = 1) -> pd.DataFrame:
        """
        Fetch current production data for analysis.
        """
        # Try to get recent predictions from Redis
        current_data = []
        
        try:
            # Get recent transaction keys
            keys = self.redis_client.keys('prediction:*')
            
            for key in keys[:5000]:  # Limit for efficiency
                data = self.redis_client.hgetall(key)
                if data:
                    current_data.append(data)
            
            if current_data:
                df = pd.DataFrame(current_data)
                # Convert types
                for col in NUMERIC_FEATURES:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
        except Exception as e:
            logger.warning(f"Could not fetch current data from Redis: {e}")
        
        # Fallback: generate synthetic current data with slight drift
        logger.info("Using synthetic current data for drift detection")
        return self._generate_drifted_data()
    
    def _generate_drifted_data(self) -> pd.DataFrame:
        """Generate synthetic current data with intentional drift for testing."""
        np.random.seed(int(datetime.now().timestamp()))
        n = 5000
        
        # Introduce drift: higher amounts, different country distribution
        return pd.DataFrame({
            'amount': np.random.exponential(120, n),  # 20% higher
            'customer_age': np.random.normal(42, 16, n).clip(18, 80),
            'transaction_count_1h': np.random.poisson(2.5, n),  # Slightly higher
            'transaction_count_24h': np.random.poisson(11, n),
            'avg_amount_7d': np.random.exponential(85, n),
            'risk_score': np.random.beta(2.5, 5, n),  # Slightly higher
            'merchant_category': np.random.choice(
                ['retail', 'food', 'travel', 'entertainment', 'utilities'], n,
                p=[0.25, 0.2, 0.25, 0.2, 0.1]  # Different distribution
            ),
            'location_country': np.random.choice(
                ['GB', 'US', 'DE', 'FR', 'ES'], n, 
                p=[0.5, 0.2, 0.12, 0.12, 0.06]  # Drift in country
            ),
            'channel': np.random.choice(['online', 'in_store', 'atm'], n),
            'card_present': np.random.choice([True, False], n, p=[0.65, 0.35]),
            'is_fraud': np.random.choice([0, 1], n, p=[0.965, 0.035]),
            'prediction': np.random.choice([0, 1], n, p=[0.955, 0.045]),
        })
    
    def detect_drift(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Run drift detection and return results.
        
        Returns:
            Tuple of (drift_detected: bool, report_dict: Dict)
        """
        logger.info("Running drift detection...")
        
        reference_data = self.get_reference_data()
        current_data = self.get_current_data()
        
        # Create Evidently report
        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])
        
        # Run drift detection
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Extract results
        result = report.as_dict()
        
        # Get dataset drift metric
        dataset_drift = result['metrics'][0]['result']
        drift_detected = dataset_drift.get('dataset_drift', False)
        drift_share = dataset_drift.get('drift_share', 0)
        
        # Detailed column drift
        column_drift = result['metrics'][1]['result']
        drifted_columns = []
        
        if 'drift_by_columns' in column_drift:
            for col, info in column_drift['drift_by_columns'].items():
                if info.get('drift_detected', False):
                    drifted_columns.append({
                        'column': col,
                        'drift_score': info.get('drift_score', 0),
                        'stattest': info.get('stattest', 'unknown')
                    })
        
        # Generate report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"{REPORT_OUTPUT_PATH}/drift_report_{timestamp}.html"
        report.save_html(report_path)
        
        # Store result in Redis for monitoring
        drift_result = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': drift_detected,
            'drift_share': drift_share,
            'drifted_columns': len(drifted_columns),
            'threshold': DRIFT_THRESHOLD,
            'report_path': report_path
        }
        
        self.redis_client.hset('drift:latest', mapping={
            k: json.dumps(v) if isinstance(v, (list, dict, bool)) else str(v)
            for k, v in drift_result.items()
        })
        
        # Log results
        if drift_detected:
            logger.warning(f"DRIFT DETECTED! {len(drifted_columns)} columns drifted")

            for col in drifted_columns:
                logger.warning(f"   - {col['column']}: score={col['drift_score']:.3f}")
        else:
            logger.info(f"No significant drift detected (share: {drift_share:.2%})")

        
        return drift_detected, {
            'drift_detected': drift_detected,
            'drift_share': drift_share,
            'drifted_columns': drifted_columns,
            'report_path': report_path,
            'reference_rows': len(reference_data),
            'current_rows': len(current_data)
        }
    
    def should_trigger_retraining(self) -> bool:
        """
        Determine if model retraining should be triggered based on drift.
        """
        drift_detected, result = self.detect_drift()
        
        # Check if drift exceeds threshold
        if result['drift_share'] > DRIFT_THRESHOLD:
            logger.warning(f"Retraining recommended: drift_share={result['drift_share']:.2%}")
            return True
        
        # Check for persistent drift (multiple consecutive detections)
        try:
            drift_history = self.redis_client.lrange('drift:history', 0, 5)
            recent_drifts = sum(1 for d in drift_history if d == 'True')
            if recent_drifts >= 3:
                logger.warning("Retraining recommended: persistent drift detected")
                return True
        except Exception:
            pass
        
        # Update history
        self.redis_client.lpush('drift:history', str(drift_detected))
        self.redis_client.ltrim('drift:history', 0, 9)  # Keep last 10
        
        return False


def main():
    """Run drift detection as standalone script."""
    print("=" * 60)
    print("Evidently AI Drift Detection")

    print("=" * 60)
    
    detector = DriftDetector()
    drift_detected, result = detector.detect_drift()
    
    print("\nResults:")

    print(f"   Drift Detected: {drift_detected}")
    print(f"   Drift Share: {result['drift_share']:.2%}")
    print(f"   Columns with Drift: {len(result['drifted_columns'])}")
    print(f"   Report: {result['report_path']}")
    
    if drift_detected:
        print("\nRECOMMENDATION: Consider retraining the model")

    
    return drift_detected


if __name__ == "__main__":
    main()
