import os
import time
import pandas as pd
import numpy as np
from minio import Minio
from io import BytesIO

# Configuration
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')

BUCKETS = ["mlflow-artifacts", "fraud-offline-store"]

def setup_minio():
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

    # 1. Create Buckets
    for bucket in BUCKETS:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            print(f"Created bucket: {bucket}")

        else:
            print(f"Bucket already exists: {bucket}")


    # 2. Seed 'fraud-offline-store' with initial data
    # This allows the retraining DAG to have something to work with instantly
    print("\nSeeding initial training data...")

    
    # Generate some synthetic data if local version doesn't exist
    local_data_path = "data/features/training_data.parquet"
    if os.path.exists(local_data_path):
        print(f"   Reading from local: {local_data_path}")
        df = pd.read_parquet(local_data_path)
    else:
        print("   Generating synthetic seed data...")
        n_samples = 1000
        df = pd.DataFrame({
            'transaction_id': [f"seed_{i}" for i in range(n_samples)],
            'user_id': [f"user_{np.random.randint(1, 100)}" for i in range(n_samples)],
            'amount': np.random.exponential(100, n_samples),
            'merchant_category': np.random.choice(['grocery', 'electronics', 'gambling'], n_samples),
            'is_fraud': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
            'timestamp': [pd.Timestamp.now() for _ in range(n_samples)]
        })
        # Add basic fraud indicators to the seed data
        df.loc[df['is_fraud'] == 1, 'amount'] = df.loc[df['is_fraud'] == 1, 'amount'] * 5
        
        # Add required columns for train.py
        for col in ['transaction_count_10m', 'transaction_count_1h', 'transaction_count_24h', 'transaction_count_30d', 
                    'total_amount_10m', 'total_amount_1h', 'max_amount_1h', 'avg_amount_30d', 'std_amount_30d', 
                    'distinct_merchants_30d', 'distinct_categories_30d', 'distinct_cities_1h', 'seconds_since_last', 
                    'velocity_score', 'amount_zscore', 'hour_of_day', 'day_of_week', 'is_weekend', 'is_high_risk_merchant']:
            df[col] = np.random.random(n_samples)

    # Convert to Parquet
    buffer = BytesIO()
    df.to_parquet(buffer)
    buffer.seek(0)
    
    # Upload to MinIO
    object_name = f"ingested/{int(time.time())}_seed_data.parquet"
    client.put_object(
        "fraud-offline-store",
        object_name,
        data=buffer,
        length=buffer.getbuffer().nbytes,
        content_type="application/octet-stream"
    )
    print(f"Uploaded seed data to: fraud-offline-store/{object_name}")


if __name__ == "__main__":
    try:
        setup_minio()
        print("\nMinIO setup complete!")

    except Exception as e:
        print(f"\nSetup failed: {e}")

