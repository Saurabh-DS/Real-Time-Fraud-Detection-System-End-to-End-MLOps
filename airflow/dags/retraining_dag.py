"""
Fraud Detection Retraining DAG
==============================
Automated model retraining workflow triggered on a schedule or by drift detection.
Includes champion/challenger comparison and registration in MLflow.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

import os
import sys
import pandas as pd
from io import BytesIO
from minio import Minio

# Configuration
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')
OFFLINE_STORE_BUCKET = "fraud-offline-store"

# Default args for all tasks
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email': ['mlops-alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}


def check_drift_status(**context):
    """
    Check if model drift has been detected.
    Returns 'retrain_needed' or 'skip_retraining' branch.
    """
    import json
    
    # Check for drift report from monitoring
    drift_report_path = '/opt/airflow/data/drift_reports/latest_drift.json'
    
    try:
        if os.path.exists(drift_report_path):
            with open(drift_report_path, 'r') as f:
                report = json.load(f)
            
            # Check if significant drift detected
            drift_detected = report.get('significant_drift', False)
            drift_score = report.get('drift_score', 0)
            
            print(f"Drift check: detected={drift_detected}, score={drift_score}")
            
            if drift_detected and drift_score > 0.3:
                return 'generate_training_data'
        
        # Also check if it's been more than 7 days since last training
        last_train_path = '/opt/airflow/data/last_training.txt'
        if os.path.exists(last_train_path):
            with open(last_train_path) as f:
                last_train = datetime.fromisoformat(f.read().strip())
            
            days_since = (datetime.now() - last_train).days
            print(f"Days since last training: {days_since}")
            
            if days_since >= 7:
                return 'generate_training_data'
        else:
            # No record of last training, retrain
            return 'generate_training_data'
        
        return 'skip_retraining'
        
    except Exception as e:
        print(f"Error checking drift: {e}")
        # On error, trigger retraining to be safe
        return 'generate_training_data'
    
    # Check for drift report from monitoring
    drift_report_path = '/opt/airflow/data/drift_reports/latest_drift.json'
    
    try:
        if os.path.exists(drift_report_path):
            with open(drift_report_path, 'r') as f:
                report = json.load(f)
            
            # Check if significant drift detected
            drift_detected = report.get('significant_drift', False)
            drift_score = report.get('drift_score', 0)
            
            print(f"Drift check: detected={drift_detected}, score={drift_score}")
            
            if drift_detected and drift_score > 0.3:
                return 'generate_training_data'
        
        # Also check if it's been more than 7 days since last training
        last_train_path = '/opt/airflow/data/last_training.txt'
        if os.path.exists(last_train_path):
            with open(last_train_path) as f:
                last_train = datetime.fromisoformat(f.read().strip())
            
            days_since = (datetime.now() - last_train).days
            print(f"Days since last training: {days_since}")
            
            if days_since >= 7:
                return 'generate_training_data'
        else:
            # No record of last training, retrain
            return 'generate_training_data'
        
        return 'skip_retraining'
        
    except Exception as e:
        print(f"Error checking drift: {e}")
        # On error, trigger retraining to be safe
        return 'generate_training_data'



def generate_training_data(**context):
    """
    Download accummulated feature logs from MinIO (Offline Store).
    Consolidates them into a single training dataset.
    """
    print(f"Connecting to Offline Store: {OFFLINE_STORE_BUCKET} @ {MINIO_ENDPOINT}")

    
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        
        # List all parquet files in the bucket (recursive)
        objects = client.list_objects(OFFLINE_STORE_BUCKET, recursive=True)
        
        data_frames = []
        count = 0
        
        for obj in objects:
            if obj.object_name.endswith('.parquet'):
                print(f"   Reading: {obj.object_name}")
                response = client.get_object(OFFLINE_STORE_BUCKET, obj.object_name)
                buffer = BytesIO(response.read())
                df = pd.read_parquet(buffer)
                data_frames.append(df)
                count += 1
                response.close()
                response.release_conn()
        
        if not data_frames:
            print("No data found in Offline Store! Cannot train.")

            raise ValueError("Offline store is empty. No training data.")
            
        # Merge all batches
        full_df = pd.concat(data_frames, ignore_index=True)
        print(f"Merged {count} batches. Total rows: {len(full_df)}")

        
        # Save locally for training script (Persistent Volume)
        output_dir = '/opt/airflow/data/features'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'training_data.parquet')
        
        full_df.to_parquet(output_path)
        print(f"Training data saved to: {output_path}")

        
    except Exception as e:
        print(f"Failed to fetch training data: {e}")

        raise


def train_model(**context):
    """Run model training script."""
    sys.path.insert(0, '/opt/airflow/training')
    
    # Set absolute path for data since train.py uses relative default
    os.environ['DATA_PATH'] = '/opt/airflow/data/features/training_data.parquet'
    
    from train import FraudModelTrainer
    
    print("Starting Model Training...")

    trainer = FraudModelTrainer()
    run_id = trainer.train()
    
    # Store run_id for downstream tasks
    context['ti'].xcom_push(key='run_id', value=run_id)
    
    print(f"Training complete. Run ID: {run_id}")
    return run_id


def validate_model(**context):
    """
    Validate new model meets minimum performance requirements.
    """
    import mlflow
    
    run_id = context['ti'].xcom_pull(key='run_id', task_ids='train_model')
    
    if not run_id:
        raise ValueError("No run_id found from training task")
    
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    client = mlflow.tracking.MlflowClient()
    
    run = client.get_run(run_id)
    metrics = run.data.metrics
    
    # Minimum requirements
    MIN_PR_AUC = 0.70
    MIN_RECALL = 0.50
    
    pr_auc = metrics.get('pr_auc', 0)
    recall = metrics.get('recall', 0)
    
    print(f"Model validation: PR-AUC={pr_auc:.4f}, Recall={recall:.4f}")
    
    if pr_auc < MIN_PR_AUC:
        raise ValueError(f"PR-AUC {pr_auc:.4f} below minimum {MIN_PR_AUC}")
    
    if recall < MIN_RECALL:
        raise ValueError(f"Recall {recall:.4f} below minimum {MIN_RECALL}")
    
    print("Model validation passed!")
    return True


def promote_model(**context):
    """Promote validated model through registry stages."""
    sys.path.insert(0, '/opt/airflow/ops')
    
    from model_registry import ModelRegistryManager
    
    run_id = context['ti'].xcom_pull(key='run_id', task_ids='train_model')
    
    if not run_id:
        raise ValueError("No run_id found from training task")
    
    print(f"Promoting model from run {run_id}...")
    
    manager = ModelRegistryManager()
    result = manager.auto_promote_pipeline(run_id, 'fraud_detection_model')
    
    print(f"Promotion result: {result}")
    
    # Store result
    context['ti'].xcom_push(key='promotion_result', value=result)
    
    return result


def update_last_training(**context):
    """Record timestamp of successful training."""
    last_train_path = '/opt/airflow/data/last_training.txt'
    
    os.makedirs(os.path.dirname(last_train_path), exist_ok=True)
    
    with open(last_train_path, 'w') as f:
        f.write(datetime.now().isoformat())
    
    print("Updated last training timestamp")


def send_notification(**context):
    """Send notification about training result."""
    promotion_result = context['ti'].xcom_pull(
        key='promotion_result', 
        task_ids='promote_model'
    )
    
    if promotion_result:
        msg = f"""
        Model Retraining Complete!

        
        Version: v{promotion_result.get('registered_version', 'N/A')}
        Promoted to Production: {promotion_result.get('promoted_to_production', False)}
        Reason: {promotion_result.get('reason', 'N/A')}
        """
    else:
        msg = "Model retraining completed but promotion result not available"

    
    print(msg)
    # In production, this would send to Slack/PagerDuty/Email


# Create DAG
with DAG(
    dag_id='fraud_detection_retraining',
    default_args=default_args,
    description='Automated fraud detection model retraining pipeline',
    schedule_interval='0 3 * * *',  # Daily at 3 AM UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'fraud', 'retraining'],
    max_active_runs=1,
) as dag:
    
    # Start
    start = EmptyOperator(task_id='start')
    
    # Check if retraining is needed
    check_drift = BranchPythonOperator(
        task_id='check_drift_status',
        python_callable=check_drift_status,
    )
    
    # Branch: Skip retraining
    skip = EmptyOperator(task_id='skip_retraining')
    
    # Branch: Retrain
    generate_data = PythonOperator(
        task_id='generate_training_data',
        python_callable=generate_training_data,
    )
    
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )
    
    validate = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model,
    )
    
    promote = PythonOperator(
        task_id='promote_model',
        python_callable=promote_model,
    )
    
    update_timestamp = PythonOperator(
        task_id='update_last_training',
        python_callable=update_last_training,
    )
    
    notify = PythonOperator(
        task_id='send_notification',
        python_callable=send_notification,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )
    
    # End (joins both branches)
    end = EmptyOperator(
        task_id='end',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
    
    # Define task dependencies
    start >> check_drift
    check_drift >> skip >> end
    check_drift >> generate_data >> train >> validate >> promote >> update_timestamp >> notify >> end
