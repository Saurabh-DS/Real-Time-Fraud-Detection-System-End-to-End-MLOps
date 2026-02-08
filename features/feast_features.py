"""
Feast Feature Definitions for Fraud Detection
==============================================
Defines entities, sources, and feature views for the fraud detection system.
Ensures consistency between offline training and online serving.
"""

from datetime import timedelta
from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float32, Int64

# ======================
# Entity Definitions
# ======================
# Entities are the primary keys for feature lookups

# User entity - for user-level aggregated features
user_entity = Entity(
    name="user_id",
    description="Unique identifier for a user/customer",
    join_keys=["user_id"],
)

# Transaction entity - for transaction-level features
transaction_entity = Entity(
    name="transaction_id", 
    description="Unique identifier for a transaction",
    join_keys=["transaction_id"],
)

# ======================
# Feature Sources
# ======================
# Define where historical feature data is stored

user_features_source = FileSource(
    path="data/user_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

transaction_features_source = FileSource(
    path="data/transaction_features.parquet", 
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# ======================
# Feature Views
# ======================
# Feature views define how features are computed and served

# User-level aggregated features
# These are computed offline and materialized to Redis
user_spending_features = FeatureView(
    name="user_spending_features",
    entities=[user_entity],
    ttl=timedelta(days=1),  # Features expire after 1 day if not refreshed
    schema=[
        Field(name="transaction_count_1h", dtype=Int64),
        Field(name="transaction_count_24h", dtype=Int64),
        Field(name="avg_amount_7d", dtype=Float32),
        Field(name="max_amount_7d", dtype=Float32),
        Field(name="total_spend_30d", dtype=Float32),
        Field(name="unique_merchants_7d", dtype=Int64),
        Field(name="fraud_rate_30d", dtype=Float32),
    ],
    source=user_features_source,
    online=True,  # Enable online serving via Redis
    tags={"team": "fraud-detection", "priority": "high"},
)

# User velocity features (time-based patterns)
user_velocity_features = FeatureView(
    name="user_velocity_features",
    entities=[user_entity],
    ttl=timedelta(hours=1),  # Short TTL for real-time features
    schema=[
        Field(name="transactions_per_minute", dtype=Float32),
        Field(name="amount_velocity_1h", dtype=Float32),
        Field(name="location_changes_1h", dtype=Int64),
        Field(name="device_changes_24h", dtype=Int64),
        Field(name="time_since_last_tx_seconds", dtype=Float32),
    ],
    source=user_features_source,
    online=True,
    tags={"team": "fraud-detection", "feature-type": "velocity"},
)

# Transaction context features  
transaction_context_features = FeatureView(
    name="transaction_context_features",
    entities=[transaction_entity],
    ttl=timedelta(hours=24),
    schema=[
        Field(name="merchant_risk_score", dtype=Float32),
        Field(name="location_risk_score", dtype=Float32),
        Field(name="channel_risk_score", dtype=Float32),
        Field(name="time_of_day_risk", dtype=Float32),
        Field(name="weekend_flag", dtype=Int64),
    ],
    source=transaction_features_source,
    online=True,
    tags={"team": "fraud-detection", "feature-type": "context"},
)
