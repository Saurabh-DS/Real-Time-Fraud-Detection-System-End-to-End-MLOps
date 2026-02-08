"""
Drift Monitoring DAG
====================
Runs hourly to check for feature drift and model performance degradation.
Generates drift reports that trigger the retraining pipeline.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

import os
import json

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def run_drift_check(**context):
    """
    Run Evidently AI drift detection.
    """
    import sys
    sys.path.insert(0, '/opt/airflow/monitoring')
    
    try:
        # Configure paths for DriftMonitor
        os.environ['REFERENCE_DATA_PATH'] = '/opt/airflow/data/features/training_data.parquet'
        os.environ['REPORT_OUTPUT_DIR'] = '/opt/airflow/data/drift_reports'
        os.environ['REDIS_HOST'] = 'redis'
        
        from check_drift import DriftMonitor
        
        monitor = DriftMonitor()
        report = monitor.run_drift_report()
        
        # Save latest report for retraining DAG
        report_dir = '/opt/airflow/data/drift_reports'
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, 'latest_drift.json')
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'significant_drift': report.get('drift_detected', False),
                'drift_score': report.get('drift_share', 0), # Corrected key from check_drift.py
                'drifted_features': report.get('drifted_features', []),
            }, f)
        
        print(f"Drift report saved: {report_path}")
        return report
        
    except ImportError:
        print("Drift monitoring module not available, using mock data")
        # Mock drift check for development
        return {'drift_detected': False, 'share_of_drifted_features': 0.1}


def alert_on_drift(**context):
    """
    Send alert if significant drift detected.
    """
    ti = context['ti']
    report = ti.xcom_pull(task_ids='run_drift_check')
    
    if report and report.get('drift_detected', False):
        drift_score = report.get('share_of_drifted_features', 0)
        
        if drift_score > 0.3:
            msg = f"CRITICAL: Model drift detected! Score: {drift_score:.2%}"

            print(msg)
            # In production: Send to PagerDuty/Slack
        elif drift_score > 0.15:
            msg = f"WARNING: Moderate drift detected. Score: {drift_score:.2%}"

            print(msg)
            # In production: Send to Slack
    else:
        print("No significant drift detected")



with DAG(
    dag_id='fraud_detection_drift_monitoring',
    default_args=default_args,
    description='Hourly drift monitoring for fraud detection model',
    schedule_interval='*/5 * * * *',  # Every 5 minutes (DEMO MODE)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'fraud', 'monitoring', 'drift'],
    max_active_runs=1,
) as dag:
    
    start = EmptyOperator(task_id='start')
    
    check_drift = PythonOperator(
        task_id='run_drift_check',
        python_callable=run_drift_check,
    )
    
    alert = PythonOperator(
        task_id='alert_on_drift',
        python_callable=alert_on_drift,
    )
    
    end = EmptyOperator(task_id='end')
    
    start >> check_drift >> alert >> end
