"""
Real-Time Feature Processor
===========================
Reads raw transaction events from Kafka and computes near real-time aggregations.
Features are persisted to Redis for low-latency retrieval by the serving layer.
"""

import json
import os
import time
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Any
from dataclasses import dataclass
import redis
from kafka import KafkaConsumer
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO
from minio import Minio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'transaction_stream')
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
PREDICTION_API_URL = os.getenv('PREDICTION_API_URL', 'http://fraud-serving:8000/predict')
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')
OFFLINE_STORE_BUCKET = "fraud-offline-store"

# Feature computation windows
WINDOW_10M = timedelta(minutes=10)
WINDOW_1H = timedelta(hours=1)
WINDOW_24H = timedelta(hours=24)


@dataclass
class TransactionEvent:
    """Parsed transaction from Kafka."""
    transaction_id: str
    user_id: str
    amount: float
    timestamp: datetime
    city: str
    country: str
    merchant_category: str
    is_fraud: bool = False  # Ground truth if available (for training)


class SlidingWindowState:
    """
    In-memory sliding window state for a single user.
    Maintains transaction history for feature computation.
    """
    
    def __init__(self):
        self.transactions: List[TransactionEvent] = []
        self.max_history = WINDOW_24H
    
    def add_transaction(self, txn: TransactionEvent):
        """Add transaction and prune old ones."""
        self.transactions.append(txn)
        self._prune_old_transactions()
    
    def _prune_old_transactions(self):
        """Remove transactions older than 24 hours."""
        cutoff = datetime.now() - self.max_history
        self.transactions = [t for t in self.transactions if t.timestamp > cutoff]
    
    def compute_features(self, current_txn: TransactionEvent) -> Dict[str, Any]:
        """
        Compute all real-time features for the current transaction context.
        """
        now = current_txn.timestamp
        
        # Filter transactions by time window
        txns_10m = [t for t in self.transactions if now - t.timestamp <= WINDOW_10M]
        txns_1h = [t for t in self.transactions if now - t.timestamp <= WINDOW_1H]
        txns_24h = [t for t in self.transactions if now - t.timestamp <= WINDOW_24H]
        
        # Velocity features (COUNT in window)
        transaction_count_10m = len(txns_10m)
        transaction_count_1h = len(txns_1h)
        transaction_count_24h = len(txns_24h)
        
        # Amount features (SUM, MAX in window)
        total_amount_10m = sum(t.amount for t in txns_10m)
        total_amount_1h = sum(t.amount for t in txns_1h)
        max_amount_1h = max((t.amount for t in txns_1h), default=0.0)
        
        # Location features (DISTINCT in window)
        distinct_cities_1h = len(set(t.city for t in txns_1h))
        
        # Last transaction info
        sorted_txns = sorted(self.transactions, key=lambda x: x.timestamp, reverse=True)
        if len(sorted_txns) > 1:
            last_txn = sorted_txns[1]  # Previous transaction (not current)
            seconds_since_last = (current_txn.timestamp - last_txn.timestamp).total_seconds()
            # last_city and last_country unused in simplified logic
            pass
        else:
            seconds_since_last = 86400.0  # Default to 24 hours if first transaction
            seconds_since_last = 86400.0  # Default to 24 hours if first transaction
        
        # Derived risk scores
        velocity_score = min(1.0, transaction_count_10m / 10.0)
        
        if txns_24h:
            # avg_amount_24h and amount_anomaly_score unused in simplified logic
            pass
        else:
            pass
            
        # 30-day stats simulations for training alignment (using 24h as proxy in this simple implementation)
        # In production this would come from a longer-term store
        avg_amount_30d = sum(t.amount for t in txns_24h) / len(txns_24h) if txns_24h else current_txn.amount
        std_amount_30d = (sum((t.amount - avg_amount_30d) ** 2 for t in txns_24h) / len(txns_24h)) ** 0.5 if len(txns_24h) > 1 else 1.0
        
        features = {
            # Transaction Identifiers
            "transaction_id": current_txn.transaction_id,
            "user_id": current_txn.user_id,
            "merchant_category": current_txn.merchant_category,
            "amount": current_txn.amount,
            
            # Velocity
            "transaction_count_10m": transaction_count_10m,
            "transaction_count_1h": transaction_count_1h,
            "transaction_count_24h": transaction_count_24h,
            "transaction_count_30d": transaction_count_24h * 30, # Estimated
            
            # Amount
            "total_amount_10m": round(total_amount_10m, 2),
            "total_amount_1h": round(total_amount_1h, 2),
            "max_amount_1h": round(max_amount_1h, 2),
            "avg_amount_30d": round(avg_amount_30d, 2),
            "std_amount_30d": round(std_amount_30d, 2),
            
            # Location
            "distinct_cities_1h": distinct_cities_1h,
            "distinct_merchants_30d": len(set(t.merchant_category for t in txns_24h)), # Proxy
            "distinct_categories_30d": len(set(t.merchant_category for t in txns_24h)), # Proxy
            
            # Time
            "seconds_since_last": round(seconds_since_last, 2),
            "hour_of_day": now.hour,
            "day_of_week": now.weekday(),
            "is_weekend": 1 if now.weekday() >= 5 else 0,
            
            # Derived scores
            "velocity_score": round(velocity_score, 4),
            "amount_zscore": round((current_txn.amount - avg_amount_30d) / std_amount_30d, 4) if std_amount_30d > 0 else 0,
            
            # Metadata
            "event_timestamp": now.isoformat(),
            "is_fraud": 1 if current_txn.is_fraud else 0
        }
        
        # Add merchant risk (simple lookup simulation)
        high_risk = ['gambling', 'crypto_exchange', 'money_transfer']
        features['is_high_risk_merchant'] = 1 if current_txn.merchant_category in high_risk else 0
        
        return features


class OfflineStoreWriter:
    """
    Writes batches of computed features to MinIO (Simulating Data Lake).
    """
    def __init__(self):
        self.buffer = []
        self.batch_size = 1000
        
        # MinIO Client
        try:
            self.minio = Minio(
                MINIO_ENDPOINT,
                access_key=MINIO_ACCESS_KEY,
                secret_key=MINIO_SECRET_KEY,
                secure=False
            )
            if not self.minio.bucket_exists(OFFLINE_STORE_BUCKET):
                self.minio.make_bucket(OFFLINE_STORE_BUCKET)
                logger.info(f"Created bucket: {OFFLINE_STORE_BUCKET}")

        except Exception as e:
            logger.error(f"MinIO connection failed: {e}")

            self.minio = None

    def append(self, features: Dict):
        self.buffer.append(features)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.buffer or not self.minio:
            return
            
        try:
            df = pd.DataFrame(self.buffer)
            table = pa.Table.from_pandas(df)
            
            # Write to in-memory buffer
            parquet_buffer = BytesIO()
            pq.write_table(table, parquet_buffer)
            parquet_buffer.seek(0)
            data_len = parquet_buffer.getbuffer().nbytes
            
            # Filename: partitioned by date
            now = datetime.now()
            filename = f"date={now.date()}/batch_{int(now.timestamp())}.parquet"
            
            self.minio.put_object(
                OFFLINE_STORE_BUCKET,
                filename,
                parquet_buffer,
                data_len
            )
            logger.info(f"Flushed {len(self.buffer)} records to MinIO: {filename}")

            self.buffer = []
            
        except Exception as e:
            logger.error(f"Failed to write to Offline Store: {e}")



class FeatureProcessor:
    """
    Main feature processing service.
    Reads from Kafka, computes features, writes to Redis (Online) + MinIO (Offline).
    """
    
    def __init__(self):
        # Kafka consumer
        self.consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='feature-processor',
            auto_offset_reset='latest',
        )
        
        # Redis connection for feature store
        self.redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True
        )
        
        # Offline Store Writer
        self.offline_writer = OfflineStoreWriter()
        
        # In-memory state per user
        self.user_states: Dict[str, SlidingWindowState] = defaultdict(SlidingWindowState)
        
        logger.info("Feature Processor initialized")

        logger.info(f"   Kafka: {KAFKA_BOOTSTRAP_SERVERS} / {KAFKA_TOPIC}")
        logger.info(f"   Redis: {REDIS_HOST}:{REDIS_PORT}")
        logger.info(f"   Offline Store: {OFFLINE_STORE_BUCKET} @ {MINIO_ENDPOINT}")
    
    def _parse_transaction(self, data: Dict) -> TransactionEvent:
        """Parse Kafka message to TransactionEvent."""
        return TransactionEvent(
            transaction_id=data['transaction_id'],
            user_id=data['user_id'],
            amount=float(data['amount']),
            timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
            city=data.get('location_city', 'Unknown'),
            country=data.get('location_country', 'Unknown'),
            merchant_category=data.get('merchant_category', 'Unknown'),
            is_fraud=data.get('is_fraud', False)
        )
    
    def _push_to_redis(self, user_id: str, features: Dict):
        """
        Push computed features to Redis (Online Store).
        """
        key = f"fraud_detection:user_realtime_features:{user_id}"
        
        # We don't store labels or event metadata in Redis for inference
        inference_features = {k: v for k, v in features.items() if k not in ['is_fraud', 'event_timestamp']}
        
        self.redis.hset(key, mapping=inference_features)
        self.redis.expire(key, 3600)  # 1 hour TTL
    
    def run(self):
        """Main processing loop."""
        logger.info("\nStarting feature processing...")

        
        processed_count = 0
        start_time = time.time()
        
        try:
            for message in self.consumer:
                # Parse transaction
                txn = self._parse_transaction(message.value)
                
                # Get/create user state
                state = self.user_states[txn.user_id]
                
                # Add transaction to state
                state.add_transaction(txn)
                
                # Compute features
                features = state.compute_features(txn)
                
                # 1. Push to Online Store (Redis)
                self._push_to_redis(txn.user_id, features)
                
                # 2. Push to Offline Store (MinIO Buffer)
                self.offline_writer.append(features)
                
                # 3. Call prediction API (Monitoring side-effect)
                try:
                    pred_payload = {
                        "transaction_id": txn.transaction_id,
                        "user_id": txn.user_id,
                        "amount": txn.amount,
                        "merchant_id": f"m_{txn.transaction_id[-4:]}",
                        "merchant_category": txn.merchant_category
                    }
                    requests.post(PREDICTION_API_URL, json=pred_payload, timeout=0.1)
                except Exception:
                    pass
                
                processed_count += 1
                
                # Log progress
                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed
                    active_users = len(self.user_states)
                    logger.info(f"Processed: {processed_count} | "

                          f"Rate: {rate:.1f}/s | "
                          f"Active users: {active_users}")
                
        except KeyboardInterrupt:
            logger.info("\nShutting down feature processor...")

        finally:
            self.offline_writer.flush() # Flush remaining records
            self.consumer.close()
            logger.info(f"Processor stopped. Total processed: {processed_count}")



if __name__ == "__main__":
    processor = FeatureProcessor()
    processor.run()
