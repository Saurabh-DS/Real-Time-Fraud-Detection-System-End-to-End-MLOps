"""
Data Drift Detection with Evidently AI
=======================================
Analyzes recent transaction data to detect distribution shifts in features and predictions.
Uses evidently for generating drift reports and identifying concept drift.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

import redis

from evidently.report import Report
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REFERENCE_DATA_PATH = os.getenv('REFERENCE_DATA_PATH', '/opt/airflow/data/features/training_data.parquet')
REPORT_OUTPUT_DIR = os.getenv('REPORT_OUTPUT_DIR', '/opt/airflow/data/drift_reports')
DRIFT_THRESHOLD = float(os.getenv('DRIFT_THRESHOLD', '0.3'))  # 30% of features drifted = alert

# Features to monitor (must match training/serving)
MONITORED_FEATURES = [
    'amount',
    'transaction_count_10m',
    'transaction_count_1h',
    'velocity_score',
    'amount_zscore',
    'hour_of_day',
    'is_high_risk_merchant',
]


class DriftMonitor:
    """
    Monitors feature distribution drift between reference (training) data
    and live production data.
    """
    
    def __init__(self):
        self.redis_client = None
        self.reference_data = None
        self.column_mapping = None
        
        self._connect_redis()
        self._load_reference_data()
        
    def _connect_redis(self):
        """Connect to Redis to fetch live prediction data."""
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info(f"Connected to Redis: {REDIS_HOST}:{REDIS_PORT}")

        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")

            self.redis_client = None
    
    def _load_reference_data(self):
        """Load reference dataset (training data) for comparison."""
        try:
            if os.path.exists(REFERENCE_DATA_PATH):
                df = pd.read_parquet(REFERENCE_DATA_PATH)
                # Select only monitored features
                available_features = [f for f in MONITORED_FEATURES if f in df.columns]
                self.reference_data = df[available_features].copy()
                
                # Sample if too large (memory efficiency)
                if len(self.reference_data) > 10000:
                    self.reference_data = self.reference_data.sample(n=10000, random_state=42)
                
                logger.info(f"Loaded reference data: {len(self.reference_data)} samples")

            else:
                logger.warning(f"Reference data not found: {REFERENCE_DATA_PATH}")

                # Generate synthetic reference data for testing
                self.reference_data = self._generate_synthetic_reference()
                
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")

            self.reference_data = self._generate_synthetic_reference()
    
    def _generate_synthetic_reference(self) -> pd.DataFrame:
        """Generate synthetic reference data for testing drift detection."""
        logger.info("Generating synthetic reference data...")

        
        np.random.seed(42)
        n_samples = 5000
        
        data = {
            'amount': np.random.lognormal(mean=3.5, sigma=1.0, size=n_samples),
            'transaction_count_10m': np.random.poisson(lam=1.5, size=n_samples),
            'transaction_count_1h': np.random.poisson(lam=5, size=n_samples),
            'velocity_score': np.random.beta(a=2, b=10, size=n_samples),
            'amount_zscore': np.random.normal(loc=0, scale=1, size=n_samples),
            'hour_of_day': np.random.randint(0, 24, size=n_samples),
            'is_high_risk_merchant': np.random.binomial(1, 0.1, size=n_samples),
        }
        
        return pd.DataFrame(data)
    
    def collect_live_data(self, time_window_hours: int = 1) -> pd.DataFrame:
        """
        Collect live prediction data from Redis for drift analysis.
        In production, this would read from a logging system (Kafka, BigQuery, etc.)
        """
        logger.info(f"Collecting live data from last {time_window_hours} hour(s)...")

        
        if not self.redis_client:
            logger.warning("Redis not connected, generating synthetic live data")
            return self._generate_synthetic_live_data()
        
        try:
            # Scan Redis for feature keys
            live_data = []
            cursor = 0
            pattern = "fraud_detection:user_realtime_features:*"
            
            while True:
                cursor, keys = self.redis_client.scan(cursor, match=pattern, count=100)
                
                for key in keys:
                    features = self.redis_client.hgetall(key)
                    if features:
                        # Convert to proper types
                        record = {}
                        for feat in MONITORED_FEATURES:
                            if feat in features:
                                try:
                                    record[feat] = float(features[feat])
                                except ValueError:
                                    record[feat] = 0
                            else:
                                record[feat] = 0
                        live_data.append(record)
                
                if cursor == 0:
                    break
            
            if live_data:
                df = pd.DataFrame(live_data)
                logger.info(f"Collected {len(df)} live samples")

                return df
            else:
                logger.warning("No live data found in Redis, using synthetic data")
                return self._generate_synthetic_live_data()
                
        except Exception as e:
            logger.error(f"Error collecting live data: {e}")
            return self._generate_synthetic_live_data()
    
    def _generate_synthetic_live_data(self, drift_amount: float = 0.2) -> pd.DataFrame:
        """
        Generate synthetic live data with intentional drift for testing.
        The drift_amount parameter controls how much drift to introduce.
        """
        logger.info(f"Generating synthetic live data with {drift_amount*100}% drift...")

        
        np.random.seed(int(datetime.now().timestamp()) % 10000)
        n_samples = 2000
        
        # Simulate drift: shift means and variances
        data = {
            # Amount: higher transactions (fraud attack)
            'amount': np.random.lognormal(mean=3.5 + drift_amount, sigma=1.2, size=n_samples),
            # Velocity: more rapid transactions
            'transaction_count_10m': np.random.poisson(lam=1.5 + drift_amount * 3, size=n_samples),
            'transaction_count_1h': np.random.poisson(lam=5 + drift_amount * 5, size=n_samples),
            # Higher velocity scores
            'velocity_score': np.random.beta(a=2 + drift_amount, b=10 - drift_amount * 2, size=n_samples),
            'amount_zscore': np.random.normal(loc=drift_amount, scale=1.2, size=n_samples),
            # Different time distribution (more night transactions)
            'hour_of_day': np.random.choice(range(24), size=n_samples, 
                                            p=[0.02]*6 + [0.03]*6 + [0.05]*6 + [0.1/1.5]*6), # Sums to (0.12+0.18+0.30+0.40) = 1.0
            # More high-risk merchants
            'is_high_risk_merchant': np.random.binomial(1, 0.1 + drift_amount, size=n_samples),
        }
        
        return pd.DataFrame(data)
    
    def run_drift_report(self, live_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate Evidently drift report comparing reference vs live data.
        Returns drift metrics and generates HTML report.
        """
        logger.info("Running drift detection analysis...")

        
        # Collect live data if not provided
        if live_data is None:
            live_data = self.collect_live_data()
        
        # Ensure we have matching columns
        available_features = [f for f in MONITORED_FEATURES 
                            if f in self.reference_data.columns and f in live_data.columns]
        
        reference = self.reference_data[available_features].copy()
        current = live_data[available_features].copy()
        
        # Build Evidently report
        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])
        
        report.run(
            reference_data=reference,
            current_data=current,
            column_mapping=None
        )
        
        # Create output directory
        os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
        
        # Generate timestamp for report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save HTML report
        html_path = os.path.join(REPORT_OUTPUT_DIR, f"drift_report_{timestamp}.html")
        report.save_html(html_path)
        logger.info(f"HTML report saved: {html_path}")

        
        # Extract metrics as JSON
        report_dict = report.as_dict()
        
        # Parse drift results
        drift_results = self._parse_drift_results(report_dict)
        
        # Save JSON metrics
        json_path = os.path.join(REPORT_OUTPUT_DIR, f"drift_metrics_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(drift_results, f, indent=2)
        logger.info(f"Metrics saved: {json_path}")

        
        return drift_results
    
    def _parse_drift_results(self, report_dict: Dict) -> Dict[str, Any]:
        """Parse Evidently report into structured drift metrics."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': False,
            'drift_share': 0.0,
            'drifted_features': [],
            'feature_details': {},
            'alert_triggered': False,
            'recommendation': ''
        }
        
        try:
            # Extract dataset drift info
            for metric in report_dict.get('metrics', []):
                metric_id = metric.get('metric', '')
                result = metric.get('result', {})
                
                if 'DatasetDriftMetric' in metric_id:
                    results['drift_detected'] = result.get('dataset_drift', False)
                    results['drift_share'] = result.get('share_of_drifted_columns', 0.0)
                    results['drifted_features'] = result.get('drifted_columns', [])
                    
                elif 'DataDriftTable' in metric_id:
                    drift_by_columns = result.get('drift_by_columns', {})
                    for col, details in drift_by_columns.items():
                        results['feature_details'][col] = {
                            'drift_detected': details.get('drift_detected', False),
                            'drift_score': details.get('drift_score', 0.0),
                            'stattest': details.get('stattest_name', 'unknown'),
                        }
            
            # Determine if alert should be triggered
            if results['drift_share'] >= DRIFT_THRESHOLD:
                results['alert_triggered'] = True
                results['recommendation'] = (
                    f"CRITICAL: {results['drift_share']*100:.1f}% of features have drifted. "
                    f"Drifted features: {', '.join(results['drifted_features'])}. "
                    "Consider triggering model retraining pipeline."
                )
            elif results['drift_detected']:
                results['recommendation'] = (
                    f"WARNING: Some drift detected in {len(results['drifted_features'])} features. "
                    "Monitor closely and investigate root cause."
                )
            else:
                results['recommendation'] = "OK: No significant drift detected. Model is stable."
                
        except Exception as e:
            logger.error(f"Error parsing drift results: {e}")
            
        return results
    
    def run_test_suite(self) -> bool:
        """
        Run Evidently test suite for CI/CD integration.
        Returns True if all tests pass, False otherwise.
        """
        logger.info("Running drift test suite...")

        
        live_data = self.collect_live_data()
        
        available_features = [f for f in MONITORED_FEATURES 
                            if f in self.reference_data.columns and f in live_data.columns]
        
        reference = self.reference_data[available_features]
        current = live_data[available_features]
        
        # Define test suite
        tests = TestSuite(tests=[
            TestNumberOfDriftedColumns(lt=3),  # Less than 3 columns drifted
            TestShareOfDriftedColumns(lt=DRIFT_THRESHOLD),  # Less than 30% drift
        ])
        
        tests.run(reference_data=reference, current_data=current)
        
        # Save test results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_path = os.path.join(REPORT_OUTPUT_DIR, f"drift_tests_{timestamp}.html")
        tests.save_html(test_path)
        
        # Get pass/fail result
        test_results = tests.as_dict()
        all_passed = all(
            test.get('status', '') == 'SUCCESS' 
            for test in test_results.get('tests', [])
        )
        
        if all_passed:
            logger.info("All drift tests PASSED")

        else:
            logger.warning("Some drift tests FAILED - review required")

            
        return all_passed


def main():
    """Main entry point for drift detection."""
    print("\n" + "="*60)
    print("FRAUD DETECTION MODEL DRIFT MONITOR")

    print("="*60 + "\n")
    
    monitor = DriftMonitor()
    
    # Run drift analysis
    results = monitor.run_drift_report()

    
    # Print summary
    print("\n" + "-"*40)
    print("DRIFT ANALYSIS SUMMARY")

    print("-"*40)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Overall Drift: {'YES' if results['drift_detected'] else 'NO'}")

    print(f"Drift Share: {results['drift_share']*100:.1f}%")
    print(f"Drifted Features: {', '.join(results['drifted_features']) or 'None'}")
    print(f"Alert Triggered: {'YES' if results['alert_triggered'] else 'NO'}")

    print(f"\nRecommendation: {results['recommendation']}")

    
    # Run test suite
    print("\n" + "-"*40)
    print("RUNNING TEST SUITE")

    print("-"*40)
    tests_passed = monitor.run_test_suite()
    
    print("\n" + "="*60)
    print(f"Reports saved to: {REPORT_OUTPUT_DIR}")

    print("="*60 + "\n")
    
    # Return exit code for CI/CD
    return 0 if tests_passed else 1


if __name__ == "__main__":
    exit(main())
