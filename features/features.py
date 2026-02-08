"""
Feast Feature Definitions
=========================
Defines the features and feature views used for fraud detection model training and serving.
Includes velocity, amount, diversity, and historical patterns.
"""

from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Float32, Int64, String
from feast.data_source import FileSource
from entities import user, merchant

# ============================================================================
# DATA SOURCES
# ============================================================================

# Historical user features (computed in batch, daily)
user_features_source = FileSource(
    name="user_features_source",
    path="data/user_features.parquet",
    timestamp_field="event_timestamp",
)

# Real-time transaction aggregations (pushed from stream processor)
user_realtime_source = FileSource(
    name="user_realtime_source", 
    path="data/user_realtime_features.parquet",
    timestamp_field="event_timestamp",
)

# Merchant reputation features
merchant_features_source = FileSource(
    name="merchant_features_source",
    path="data/merchant_features.parquet", 
    timestamp_field="event_timestamp",
)

# ============================================================================
# FEATURE VIEWS
# ============================================================================

# User Historical Features (Batch - computed daily)
# These establish the "normal" baseline for each user
user_historical_features = FeatureView(
    name="user_historical_features",
    entities=[user],
    ttl=timedelta(days=1),  # Refresh daily
    schema=[
        # Spending patterns (30-day rolling)
        Field(name="avg_transaction_amount_30d", dtype=Float32, 
              description="Average transaction amount over last 30 days"),
        Field(name="std_transaction_amount_30d", dtype=Float32,
              description="Standard deviation of transaction amounts"),
        Field(name="max_transaction_amount_30d", dtype=Float32,
              description="Maximum single transaction in 30 days"),
        Field(name="total_spend_30d", dtype=Float32,
              description="Total spending in last 30 days"),
        
        # Frequency patterns
        Field(name="transaction_count_30d", dtype=Int64,
              description="Number of transactions in 30 days"),
        Field(name="avg_transactions_per_day", dtype=Float32,
              description="Average daily transaction frequency"),
        
        # Diversity patterns
        Field(name="distinct_merchants_30d", dtype=Int64,
              description="Number of unique merchants used"),
        Field(name="distinct_categories_30d", dtype=Int64,
              description="Number of unique merchant categories"),
        Field(name="preferred_category", dtype=String,
              description="Most frequent merchant category"),
        
        # Geographic patterns
        Field(name="home_city", dtype=String,
              description="Most frequent transaction city"),
        Field(name="distinct_countries_30d", dtype=Int64,
              description="Number of unique countries transacted in"),
        
        # Channel patterns
        Field(name="online_transaction_ratio", dtype=Float32,
              description="Percentage of online vs in-store transactions"),
    ],
    online=True,
    source=user_features_source,
    tags={"team": "fraud", "feature_type": "batch"},
)

# User Real-Time Features (Streaming - updated in real-time)
# These detect sudden behavioral changes (fraud signals)
user_realtime_features = FeatureView(
    name="user_realtime_features",
    entities=[user],
    ttl=timedelta(hours=1),  # Keep fresh
    schema=[
        # Velocity features - KEY FRAUD SIGNALS
        Field(name="transaction_count_10m", dtype=Int64,
              description="Transactions in last 10 minutes (card testing detection)"),
        Field(name="transaction_count_1h", dtype=Int64,
              description="Transactions in last 1 hour"),
        Field(name="transaction_count_24h", dtype=Int64,
              description="Transactions in last 24 hours"),
        
        # Amount features - KEY FRAUD SIGNALS
        Field(name="total_amount_10m", dtype=Float32,
              description="Total spend in last 10 minutes"),
        Field(name="total_amount_1h", dtype=Float32,
              description="Total spend in last 1 hour"),
        Field(name="max_amount_1h", dtype=Float32,
              description="Maximum single transaction in last hour"),
        
        # Location features - IMPOSSIBLE TRAVEL DETECTION
        Field(name="distinct_cities_1h", dtype=Int64,
              description="Number of unique cities in last hour"),
        Field(name="last_transaction_city", dtype=String,
              description="City of last transaction"),
        Field(name="last_transaction_country", dtype=String,
              description="Country of last transaction"),
        
        # Time since features
        Field(name="seconds_since_last_transaction", dtype=Float32,
              description="Time gap from previous transaction"),
        
        # Derived risk scores
        Field(name="velocity_score", dtype=Float32,
              description="Normalized velocity anomaly score (0-1)"),
        Field(name="amount_anomaly_score", dtype=Float32,
              description="How unusual is current amount vs history (0-1)"),
    ],
    online=True,
    source=user_realtime_source,
    tags={"team": "fraud", "feature_type": "realtime"},
)

# Merchant Features (Batch - for merchant reputation)
merchant_features = FeatureView(
    name="merchant_features",
    entities=[merchant],
    ttl=timedelta(days=1),
    schema=[
        Field(name="merchant_fraud_rate_30d", dtype=Float32,
              description="Historical fraud rate for this merchant"),
        Field(name="merchant_avg_transaction", dtype=Float32,
              description="Average transaction size at this merchant"),
        Field(name="merchant_risk_category", dtype=String,
              description="Risk tier: low, medium, high"),
        Field(name="merchant_age_days", dtype=Int64,
              description="How long merchant has been in network"),
    ],
    online=True,
    source=merchant_features_source,
    tags={"team": "fraud", "feature_type": "batch"},
)
