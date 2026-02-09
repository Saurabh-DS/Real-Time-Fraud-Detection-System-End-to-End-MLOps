"""
Real-Time Fraud Detection API
=============================
FastAPI service for low-latency fraud detection inference.
Fetches real-time features from Redis and returns predictions with confidence scores.
"""

import os
import time
import json
import asyncio
import httpx
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import redis
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import uuid # moved from mid-file
import shap

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MODEL_NAME = os.getenv('MODEL_NAME', 'fraud_detection_model')
MODEL_STAGE = os.getenv('MODEL_STAGE', 'None')  # 'Production', 'Staging', or 'None' for latest
SHADOW_MODEL_URL = os.getenv('SHADOW_MODEL_URL')  # URL for shadow model (V2)

# A/B Testing Configuration
AB_TESTING_ENABLED = os.getenv('AB_TESTING_ENABLED', 'false').lower() == 'true'
AB_TRAFFIC_SPLIT = float(os.getenv('AB_TRAFFIC_SPLIT', '0.1'))  # 10% to variant by default
AB_VARIANT_MODEL_URL = os.getenv('AB_VARIANT_MODEL_URL')  # URL for variant model
AB_EXPERIMENT_NAME = os.getenv('AB_EXPERIMENT_NAME', 'fraud_model_ab_test')

# Default threshold (will be loaded from model artifacts)
DEFAULT_THRESHOLD = 0.5

# Feature configuration (must match training)
NUMERIC_FEATURES = [
    'amount',
    'transaction_count_10m',
    'transaction_count_1h',
    'transaction_count_24h',
    'transaction_count_30d',
    'total_amount_10m',
    'total_amount_1h',
    'max_amount_1h',
    'avg_amount_30d',
    'std_amount_30d',
    'distinct_merchants_30d',
    'distinct_categories_30d',
    'distinct_cities_1h',
    'seconds_since_last',
    'velocity_score',
    'amount_zscore',
    'hour_of_day',
    'day_of_week',
    'is_weekend',
    'is_high_risk_merchant',
    'customer_age',  # Added for demographic bias check
]

CATEGORICAL_FEATURES = ['merchant_category']

HIGH_RISK_CATEGORIES = ['gambling', 'crypto_exchange', 'money_transfer', 'electronics']

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'fraud_predictions_total',
    'Total number of fraud predictions',
    ['prediction', 'model_version']
)

INFERENCE_LATENCY = Histogram(
    'inference_latency_seconds',
    'Time spent on inference',
    buckets=[0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

FEATURE_FETCH_LATENCY = Histogram(
    'feature_fetch_latency_seconds',
    'Time spent fetching features from Redis',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
)

MODEL_LOADED = Gauge(
    'model_loaded',
    'Whether the model is loaded (1) or not (0)'
)

PREDICTION_PROBABILITY = Histogram(
    'prediction_probability',
    'Distribution of fraud probabilities',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Shadow Metrics
SHADOW_PREDICTION_COUNTER = Counter(
    'shadow_predictions_total',
    'Total number of shadow model predictions',
    ['prediction', 'model_version']
)

SHADOW_PROBABILITY = Histogram(
    'shadow_probability',
    'Shadow model probability distribution',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

SHADOW_AGREEMENT = Counter(
    'shadow_agreement_total',
    'Agreement between V1 and Shadow models',
    ['agreement'] # 'true' or 'false'
)

# A/B Testing Metrics
AB_EXPERIMENT_COUNTER = Counter(
    'ab_experiment_assignments_total',
    'Count of A/B experiment assignments',
    ['experiment', 'variant']  # variant: 'control' or 'treatment'
)

AB_VARIANT_PREDICTIONS = Counter(
    'ab_variant_predictions_total',
    'Predictions by experiment variant',
    ['experiment', 'variant', 'prediction']
)

AB_VARIANT_LATENCY = Histogram(
    'ab_variant_latency_seconds',
    'Latency by experiment variant',
    ['experiment', 'variant'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# Shadow Mode Traffic Split (for V1 vs V2 comparison)
SHADOW_MODE_TRAFFIC_SPLIT = {
    "v1": 0.5,   # 50% Shadow Mode
    "v2": 0.5   # 50% Shadow Mode
}


# Request/Response Models
class TransactionRequest(BaseModel):
    """
    Transaction payload for fraud prediction.
    Matches production payment gateway format.
    """
    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="User/cardholder identifier")
    amount: float = Field(..., description="Transaction amount", gt=0)
    currency: str = Field(default="GBP", description="Currency code")
    merchant_id: str = Field(..., description="Merchant identifier")
    merchant_name: Optional[str] = Field(default=None, description="Merchant name")
    merchant_category: str = Field(..., description="Merchant Category Code")
    timestamp: Optional[str] = Field(default=None, description="Transaction timestamp (ISO format)")
    location_city: Optional[str] = Field(default=None, description="Transaction city")
    location_country: Optional[str] = Field(default="GB", description="Transaction country")
    card_present: Optional[bool] = Field(default=True, description="Card present flag")
    channel: Optional[str] = Field(default="pos", description="Transaction channel")
    customer_age: Optional[int] = Field(default=35, description="Customer age for fairness checks")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "txn_123456789",
                "user_id": "user_00001",
                "amount": 150.00,
                "currency": "GBP",
                "merchant_id": "merchant_0001",
                "merchant_name": "Tesco",
                "merchant_category": "grocery_stores",
                "timestamp": "2026-01-30T21:30:00Z",
                "location_city": "London",
                "location_country": "GB",
                "card_present": True,
                "channel": "pos",
                "customer_age": 42
            }
        }


class PredictionResponse(BaseModel):
    """Response from fraud prediction endpoint."""
    transaction_id: str
    prediction: str  # "fraud" or "legitimate"
    probability: float  # Fraud probability [0, 1]
    confidence: str  # "high", "medium", "low"
    risk_factors: List[str]  # Explanation of risk signals
    latency_ms: float  # Total inference latency
    model_version: str
    features_used: Dict[str, Any]
    experiment_variant: Optional[str] = None  # A/B test variant: 'control' or 'treatment'



class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    redis_connected: bool
    model_version: str
    uptime_seconds: float


class ExplanationResponse(BaseModel):
    """SHAP-based model explanation response."""
    transaction_id: str
    prediction: str
    probability: float
    base_value: float  # Expected value (average prediction)
    feature_contributions: Dict[str, float]  # SHAP values per feature
    top_positive_factors: List[Dict[str, Any]]  # Top features pushing towards fraud
    top_negative_factors: List[Dict[str, Any]]  # Top features pushing away from fraud
    model_version: str
    latency_ms: float


# Global state
class ModelState:
    """Singleton to hold model and connections."""
    model = None
    model_version = "unknown"
    threshold = DEFAULT_THRESHOLD
    label_encoders = {}
    redis_client = None
    start_time = None
    shap_explainer = None  # SHAP TreeExplainer for XGBoost


class AuditLogger:
    """
    Immutable audit log for compliance.
    Logs every prediction to a separate secure ledger.
    """
    def __init__(self, log_file="audit_log.jsonl"):
        self.log_file = log_file
    
    def log(self, transaction_id: str, request_payload: Dict, prediction: Dict, model_version: str):
        """Log prediction event."""
        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "transaction_id": transaction_id,
            "inputs": request_payload,
            "outputs": prediction,
            "model_version": model_version,
            "hash": "sha256_placeholder"  # In prod, this would be a hash of the event
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(event) + "\n")

# State & Logger
state = ModelState()
audit_logger = AuditLogger()

class ShadowLogger:
    """
    Dedicated logger for model comparison (V1 vs V2).
    """
    def __init__(self, log_file="shadow_log.jsonl"):
        self.log_file = log_file
    
    def log(self, transaction_id: str, v1_prob: float, v2_prob: float, v1_pred: str, v2_pred: str):
        event = {
            "timestamp": datetime.now().isoformat(),
            "transaction_id": transaction_id,
            "v1_probability": v1_prob,
            "v2_probability": v2_prob,
            "v1_prediction": v1_pred,
            "v2_prediction": v2_pred,
            "diff": abs(v1_prob - v2_prob),
            "agreement": v1_pred == v2_pred
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(event) + "\n")

shadow_logger = ShadowLogger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle management for FastAPI.
    Load model and establish connections on startup.
    """
    logger.info("Starting Fraud Detection API...")
    state.start_time = time.time()
    
    # Connect to Redis
    try:
        state.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_timeout=5
        )
        state.redis_client.ping()
        logger.info(f"Redis connected: {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        state.redis_client = None
    
    # Load model from MLflow
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Get latest model version
        client = mlflow.tracking.MlflowClient()
        
        try:
            # Try to get model from registry
            if MODEL_STAGE != 'None':
                model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
            else:
                # Get latest version
                versions = client.get_latest_versions(MODEL_NAME)
                if versions:
                    latest = versions[0]
                    state.model_version = f"v{latest.version}"
                    model_uri = f"models:/{MODEL_NAME}/{latest.version}"
                else:
                    raise Exception("No model versions found in registry")
            
            state.model = mlflow.xgboost.load_model(model_uri)
            MODEL_LOADED.set(1)
            logger.info(f"Model loaded: {MODEL_NAME} ({state.model_version})")
            
            # Initialize SHAP explainer for XGBoost
            try:
                state.shap_explainer = shap.TreeExplainer(state.model)
                logger.info("SHAP explainer initialized")
            except Exception as se:
                logger.warning(f"SHAP explainer initialization failed: {se}")
            
        except Exception as e:
            logger.warning(f"Could not load from registry: {e}")
            logger.info("Trying to load from experiment runs...")
            
            # Fallback: load from latest experiment run
            try:
                experiment = client.get_experiment_by_name("fraud_detection")
                if experiment:
                    runs = client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        order_by=["start_time DESC"],
                        max_results=1
                    )
                    if runs:
                        latest_run = runs[0]
                        model_uri = f"runs:/{latest_run.info.run_id}/model"
                        state.model = mlflow.xgboost.load_model(model_uri)
                        state.model_version = f"run-{latest_run.info.run_id[:8]}"
                        MODEL_LOADED.set(1)
                        logger.info(f"Model loaded from run: {latest_run.info.run_id}")
                        
                        # Initialize SHAP explainer
                        try:
                            state.shap_explainer = shap.TreeExplainer(state.model)
                            logger.info("SHAP explainer initialized from run")
                        except Exception as se:
                            logger.warning(f"SHAP explainer failed: {se}")
                    else:
                        raise Exception("No runs found in experiment")
                else:
                    raise Exception("Experiment 'fraud_detection' not found")
            except Exception as e2:
                logger.warning(f"Could not load from runs: {e2}")
                # Use rule-based fallback for demo purposes
                logger.info("Using rule-based fallback model for demo")
                state.model = "rule-based"
                state.model_version = os.getenv("MODEL_VERSION", "fallback-v1")
                MODEL_LOADED.set(1)
            
    except Exception as e:
        logger.error(f"MLflow connection failed: {e}")
        MODEL_LOADED.set(0)
    
    # Allow overriding model version for metrics (e.g. "shadow")
    if os.getenv("METRICS_MODEL_VERSION"):
        state.model_version = os.getenv("METRICS_MODEL_VERSION")
        logger.info(f"Overriding model version for metrics: {state.model_version}")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Fraud Detection API...")
    if state.redis_client:
        state.redis_client.close()


# Create FastAPI app
app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="ML-powered fraud detection service with sub-100ms latency",
    version="1.0.0",
    lifespan=lifespan
)


def get_features_from_redis(user_id: str) -> Dict[str, Any]:
    """
    Fetch real-time features from Redis (Feast online store).
    Returns empty dict if user not found (new user).
    """
    if not state.redis_client:
        return {}
    
    start = time.time()
    try:
        key = f"fraud_detection:user_realtime_features:{user_id}"
        features = state.redis_client.hgetall(key)
        
        FEATURE_FETCH_LATENCY.observe(time.time() - start)
        
        # Convert string values to appropriate types
        typed_features = {}
        for k, v in features.items():
            try:
                # Try float first
                typed_features[k] = float(v)
            except ValueError:
                typed_features[k] = v
        
        return typed_features
        
    except Exception as e:
        logger.warning(f"Redis fetch error for {user_id}: {e}")
        FEATURE_FETCH_LATENCY.observe(time.time() - start)
        return {}


def prepare_features(transaction: TransactionRequest, redis_features: Dict) -> pd.DataFrame:
    """
    Combine transaction data with real-time features for inference.
    """
    now = datetime.now()
    
    # Start with real-time features from Redis
    features = {
        # Velocity features (from Redis or defaults for new users)
        'transaction_count_10m': redis_features.get('transaction_count_10m', 0),
        'transaction_count_1h': redis_features.get('transaction_count_1h', 0),
        'transaction_count_24h': redis_features.get('transaction_count_24h', 0),
        'transaction_count_30d': redis_features.get('transaction_count_30d', 0),
        
        # Amount features
        'total_amount_10m': redis_features.get('total_amount_10m', 0),
        'total_amount_1h': redis_features.get('total_amount_1h', 0),
        'max_amount_1h': redis_features.get('max_amount_1h', 0),
        'avg_amount_30d': redis_features.get('avg_amount_30d', transaction.amount),
        'std_amount_30d': redis_features.get('std_amount_30d', 0),
        
        # Diversity features
        'distinct_merchants_30d': redis_features.get('distinct_merchants_30d', 1),
        'distinct_categories_30d': redis_features.get('distinct_categories_30d', 1),
        'distinct_cities_1h': redis_features.get('distinct_cities_1h', 1),
        
        # Time features
        'seconds_since_last': redis_features.get('seconds_since_last_transaction', 3600),
        
        # Derived scores
        'velocity_score': redis_features.get('velocity_score', 0),
        
        # Current transaction features
        'amount': transaction.amount,
        'hour_of_day': now.hour,
        'day_of_week': now.weekday(),
        'is_weekend': 1 if now.weekday() >= 5 else 0,
        'is_high_risk_merchant': 1 if transaction.merchant_category in HIGH_RISK_CATEGORIES else 0,
        
        # Demographic features (from request)
        'customer_age': transaction.customer_age if transaction.customer_age is not None else 35,
        
        # Merchant category (will be encoded)
        'merchant_category': transaction.merchant_category,
    }
    
    # Calculate amount z-score
    avg_30d = features['avg_amount_30d']
    std_30d = features['std_amount_30d']
    if std_30d > 0:
        features['amount_zscore'] = (transaction.amount - avg_30d) / std_30d
    else:
        features['amount_zscore'] = 0
    
    # Create DataFrame
    df = pd.DataFrame([features])
    
    # Encode categorical features
    # Simple label encoding for merchant category
    category_map = {
        'grocery_stores': 0, 'restaurants': 1, 'gas_stations': 2, 'hotels': 3,
        'airlines': 4, 'electronics': 5, 'clothing': 6, 'pharmacy': 7,
        'entertainment': 8, 'utilities': 9, 'professional_services': 10,
        'gambling': 11, 'crypto_exchange': 12, 'money_transfer': 13
    }
    df['merchant_category'] = df['merchant_category'].map(
        lambda x: category_map.get(x, 14)  # 14 for unknown
    )
    
    # Ensure column order matches training
    feature_cols = NUMERIC_FEATURES + ['merchant_category']
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    return df[feature_cols]


def get_risk_factors(transaction: TransactionRequest, features: Dict, probability: float) -> List[str]:
    """
    Generate human-readable risk factors for explainability.
    # Explainability for fraud operations and regulatory compliance.

    """
    risk_factors = []
    
    # High velocity
    txn_10m = features.get('transaction_count_10m', 0)
    if txn_10m >= 5:
        risk_factors.append(f"High transaction velocity: {txn_10m} transactions in last 10 minutes")
    
    # Unusual amount
    amount_zscore = features.get('amount_zscore', 0)
    if amount_zscore > 3:
        risk_factors.append(f"Unusual amount: {amount_zscore:.1f} standard deviations above average")
    
    # High-risk merchant
    if transaction.merchant_category in HIGH_RISK_CATEGORIES:
        risk_factors.append(f"High-risk merchant category: {transaction.merchant_category}")
    
    # Multiple cities in short time
    distinct_cities = features.get('distinct_cities_1h', 1)
    if distinct_cities > 2:
        risk_factors.append(f"Geographic anomaly: {distinct_cities} cities in last hour")
    
    # New user (limited history)
    if features.get('transaction_count_30d', 0) < 5:
        risk_factors.append("Limited transaction history (new or inactive user)")
    
    # Late night transaction
    hour = features.get('hour_of_day', 12)
    if 2 <= hour <= 5:
        risk_factors.append(f"Unusual transaction time: {hour}:00")
    
    if not risk_factors:
        risk_factors.append("No significant risk factors detected")
    
    return risk_factors


async def send_to_shadow(transaction: TransactionRequest, v1_response: PredictionResponse):
    """
    Fire-and-forget shadow request to V2 model.
    Logs comparison without affecting V1 latency.
    """
    try:
        async with httpx.AsyncClient() as client:
            # Send payload to shadow service
            payload = transaction.dict()
            
            response = await client.post(SHADOW_MODEL_URL, json=payload, timeout=2.0)
            
            if response.status_code == 200:
                v2_result = response.json()
                v1_prob = v1_response.probability
                v2_prob = v2_result.get('probability', 0.0)
                
                # Metrics
                v2_pred = "fraud" if v2_prob >= 0.5 else "legitimate"
                agreement = str(v1_response.prediction == v2_pred).lower()
                
                SHADOW_PREDICTION_COUNTER.labels(prediction=v2_pred, model_version="shadow_v2").inc()
                SHADOW_PROBABILITY.observe(v2_prob)
                SHADOW_AGREEMENT.labels(agreement=agreement).inc()

                logger.info(f"SHADOW_COMPARE | TxID: {transaction.transaction_id} | "
                            f"V1: {v1_prob:.4f} | V2: {v2_prob:.4f} | "
                            f"Diff: {abs(v1_prob - v2_prob):.4f}")
                
                # Log to structured file for analysis
                v2_pred = "fraud" if v2_prob >= 0.5 else "legitimate"  # Assuming same threshold
                shadow_logger.log(
                    transaction_id=transaction.transaction_id,
                    v1_prob=v1_prob,
                    v2_prob=v2_prob,
                    v1_pred=v1_response.prediction,
                    v2_pred=v2_pred
                )
            else:
                logger.warning(f"Shadow model returned {response.status_code}")
                
    except Exception as e:
        logger.error(f"Shadow request failed (Ignored): {e}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionRequest):
    """
    Real-time fraud prediction endpoint.
    
    Flow:
    1. Fetch real-time features from Redis
    2. Combine with transaction features
    3. Run model inference
    4. Return prediction with explanation
    """
    start_time = time.time()
    
    # Check if model is loaded
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for initialization."
        )
    
    try:
        # 1. Fetch features from Redis
        redis_features = get_features_from_redis(transaction.user_id)
        
        # 2. Prepare feature vector
        feature_df = prepare_features(transaction, redis_features)
        
        # 3. Model inference
        inference_start = time.time()
        
        # Handle rule-based fallback model
        if state.model == "rule-based":
            # Simple rule-based fraud scoring
            score = 0.1  # Base probability
            if transaction.amount > 1000:
                score += 0.3
            if transaction.merchant_category in HIGH_RISK_CATEGORIES:
                score += 0.2
            if redis_features.get('transaction_count_10m', 0) > 3:
                score += 0.2
            if redis_features.get('amount_zscore', 0) > 2:
                score += 0.1
            probability = min(score, 0.99)
        else:
            # Use .values to avoid 'DataFrame object has no attribute dtype' error in some XGB/Pandas versions
            probability = float(state.model.predict_proba(feature_df.values)[0, 1])
        
        inference_time = time.time() - inference_start
        INFERENCE_LATENCY.observe(inference_time)
        
        # 4. Apply threshold
        is_fraud = probability >= state.threshold
        prediction = "fraud" if is_fraud else "legitimate"
        
        # 5. Determine confidence
        if probability >= 0.9 or probability <= 0.1:
            confidence = "high"
        elif probability >= 0.7 or probability <= 0.3:
            confidence = "medium"
        else:
            confidence = "low"
        
        # 6. Get risk factors
        combined_features = {**redis_features, **feature_df.iloc[0].to_dict()}
        risk_factors = get_risk_factors(transaction, combined_features, probability)
        
        # Calculate total latency
        total_latency = (time.time() - start_time) * 1000  # Convert to ms
        
        # Update Prometheus metrics
        PREDICTION_COUNTER.labels(prediction=prediction, model_version=state.model_version).inc()
        PREDICTION_PROBABILITY.observe(probability)
        
        # A/B Testing: Determine variant
        experiment_variant = None
        if AB_TESTING_ENABLED:
            # Use transaction_id hash for consistent assignment
            variant_hash = hash(transaction.transaction_id) % 100
            if variant_hash < (AB_TRAFFIC_SPLIT * 100):
                experiment_variant = 'treatment'
            else:
                experiment_variant = 'control'
            
            # Track assignment
            AB_EXPERIMENT_COUNTER.labels(
                experiment=AB_EXPERIMENT_NAME,
                variant=experiment_variant
            ).inc()
        
        # Build response
        response = PredictionResponse(
            transaction_id=transaction.transaction_id,
            prediction=prediction,
            probability=round(probability, 4),
            confidence=confidence,
            risk_factors=risk_factors,
            latency_ms=round(total_latency, 2),
            model_version=state.model_version,
            features_used={
                'velocity_score': combined_features.get('velocity_score', 0),
                'amount_zscore': combined_features.get('amount_zscore', 0),
                'transaction_count_10m': combined_features.get('transaction_count_10m', 0),
                'is_high_risk_merchant': combined_features.get('is_high_risk_merchant', 0),
            },
            experiment_variant=experiment_variant
        )
        
        # Track A/B variant metrics
        if experiment_variant:
            AB_VARIANT_PREDICTIONS.labels(
                experiment=AB_EXPERIMENT_NAME,
                variant=experiment_variant,
                prediction=prediction
            ).inc()
            AB_VARIANT_LATENCY.labels(
                experiment=AB_EXPERIMENT_NAME,
                variant=experiment_variant
            ).observe(total_latency / 1000)  # Convert ms to seconds

        
        # Log high-risk predictions
        if is_fraud:
            logger.warning(f"FRAUD DETECTED: {transaction.transaction_id} "
                          f"(user={transaction.user_id}, amount={transaction.amount}, "
                          f"prob={probability:.3f})")
        

        
        # Compliance Audit Log
        audit_logger.log(
            transaction_id=transaction.transaction_id,
            request_payload=transaction.dict(),
            prediction=response.dict(),
            model_version=state.model_version
        )
        
        # Enterprise Shadow Mode (Fire-and-Forget)
        if SHADOW_MODEL_URL:
            # This runs in background, strictly non-blocking
            asyncio.create_task(send_to_shadow(transaction, response))
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for load balancers and Kubernetes."""
    redis_ok = False
    if state.redis_client:
        try:
            state.redis_client.ping()
            redis_ok = True
        except Exception:
            pass
    
    uptime = time.time() - state.start_time if state.start_time else 0
    
    return HealthResponse(
        status="healthy" if state.model is not None else "degraded",
        model_loaded=state.model is not None,
        redis_connected=redis_ok,
        model_version=state.model_version,
        uptime_seconds=round(uptime, 2)
    )


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    Scraped by Prometheus for monitoring dashboards.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/explain", response_model=ExplanationResponse)
async def explain(transaction: TransactionRequest):
    """
    SHAP-based model explanation endpoint.
    
    Returns feature contributions showing how each feature
    pushes the prediction towards or away from fraud.
    
    Senior MLOps insight: Explainability is crucial for:
    - Regulatory compliance (GDPR right to explanation)
    - Fraud operations (analysts need to understand decisions)
    - Model debugging (identify feature drift impacts)
    """
    start_time = time.time()
    
    # Check if model and explainer are loaded
    if state.model is None or state.model == "rule-based":
        raise HTTPException(
            status_code=503,
            detail="SHAP explanations require ML model. Rule-based fallback doesn't support explanations."
        )
    
    if state.shap_explainer is None:
        raise HTTPException(
            status_code=503,
            detail="SHAP explainer not initialized. Please wait for model loading."
        )
    
    try:
        # 1. Fetch features from Redis
        redis_features = get_features_from_redis(transaction.user_id)
        
        # 2. Prepare feature vector
        feature_df = prepare_features(transaction, redis_features)
        
        # 3. Get model prediction
        probability = float(state.model.predict_proba(feature_df)[0, 1])
        is_fraud = probability >= state.threshold
        prediction = "fraud" if is_fraud else "legitimate"
        
        # 4. Compute SHAP values
        shap_values = state.shap_explainer.shap_values(feature_df)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # For binary classification, take positive class
            shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            shap_vals = shap_values[0]
        
        # Get base value (expected value)
        base_value = float(state.shap_explainer.expected_value)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])
        
        # 5. Build feature contribution dict
        feature_names = feature_df.columns.tolist()
        feature_contributions = {
            name: round(float(val), 6) 
            for name, val in zip(feature_names, shap_vals)
        }
        
        # 6. Sort by absolute contribution
        sorted_contribs = sorted(
            feature_contributions.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        # Top features pushing towards fraud (positive SHAP)
        top_positive = [
            {"feature": name, "contribution": val, "impact": "increases fraud risk"}
            for name, val in sorted_contribs if val > 0
        ][:5]
        
        # Top features pushing away from fraud (negative SHAP)
        top_negative = [
            {"feature": name, "contribution": val, "impact": "decreases fraud risk"}
            for name, val in sorted_contribs if val < 0
        ][:5]
        
        latency_ms = (time.time() - start_time) * 1000
        
        return ExplanationResponse(
            transaction_id=transaction.transaction_id,
            prediction=prediction,
            probability=round(probability, 4),
            base_value=round(base_value, 4),
            feature_contributions=feature_contributions,
            top_positive_factors=top_positive,
            top_negative_factors=top_negative,
            model_version=state.model_version,
            latency_ms=round(latency_ms, 2)
        )
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """API documentation redirect."""
    return {
        "name": "Real-Time Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "explain": "/explain (POST)"
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv('SERVING_PORT', '8000'))
    host = os.getenv('SERVING_HOST', '0.0.0.0')  # nosec B104
    
    uvicorn.run(app, host=host, port=port)
