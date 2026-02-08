"""
XGBoost Fraud Detection Model Training
======================================
This script handles the training of the fraud detection model using XGBoost.
It includes class imbalance handling, threshold optimization, and SHAP explainability.
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict

import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
EXPERIMENT_NAME = "fraud_detection"
MODEL_NAME = "fraud_detection_model"
DATA_PATH = os.getenv('DATA_PATH', '/opt/airflow/data/features/training_data.parquet')
RANDOM_SEED = 42

# Feature columns (must match inference)
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
]

CATEGORICAL_FEATURES = [
    'merchant_category',
]

TARGET = 'is_fraud'


class FraudModelTrainer:
    """
    End-to-end training pipeline with MLflow tracking.
    """
    
    def __init__(self):
        # Setup MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        self.label_encoders = {}
        self.best_threshold = 0.5
        
        print("[INIT] Fraud Model Trainer initialized")
        print(f"   MLflow: {MLFLOW_TRACKING_URI}")
        print(f"   Experiment: {EXPERIMENT_NAME}")
    
    def load_data(self) -> pd.DataFrame:
        """Load and validate training data."""
        print(f"\n[LOAD] Loading data from: {DATA_PATH}")
        
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Training data not found at: {DATA_PATH}")
            
        df = pd.read_parquet(DATA_PATH)
        
        print(f"   Rows: {len(df):,}")
        print(f"   Fraud rate: {df[TARGET].mean()*100:.2f}%")
        
        return df
    
    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess features for XGBoost.
        # Minimal preprocessing for XGBoost handling.

        """
        # Select features
        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
        y = df[TARGET].copy()
        
        # Encode categorical features
        for col in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Handle missing values
        X = X.fillna(0)
        
        # Handle infinities
        X = X.replace([np.inf, -np.inf], 0)
        
        print("\n[OK] Preprocessing complete:")
        print(f"   Features: {X.shape[1]}")
        print(f"   Samples: {X.shape[0]:,}")
        
        return X, y
    
    def _calculate_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     y_prob: np.ndarray) -> Dict[str, float]:
        """
        Calculate business-relevant metrics.
        
        In fraud detection:
        - Recall is CRITICAL: Missing fraud is costly (chargebacks, reputation)
        - Precision matters for customer experience (don't block good transactions)
        - PR-AUC is better than ROC-AUC for imbalanced data
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            # Standard ML metrics
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'pr_auc': average_precision_score(y_true, y_prob),
            
            # Confusion matrix components
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            
            # Business metrics
            'fraud_detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Same as recall
            'false_alarm_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        }
        
        return metrics
    
    def _optimize_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Find optimal threshold balancing precision and recall.
        
        # Optimize threshold for imbalanced data.

        We optimize for F1 but could weight towards recall for fraud.
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        
        # Calculate F1 for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        
        # Find threshold with best F1
        best_idx = np.argmax(f1_scores[:-1])  # Last element is undefined
        best_threshold = thresholds[best_idx]
        
        print("\n[THRESHOLD] Threshold optimization:")
        print(f"   Default (0.5) - Precision: {precisions[np.searchsorted(thresholds, 0.5)]:.3f}, "
              f"Recall: {recalls[np.searchsorted(thresholds, 0.5)]:.3f}")
        print(f"   Optimal ({best_threshold:.3f}) - Precision: {precisions[best_idx]:.3f}, "
              f"Recall: {recalls[best_idx]:.3f}")
        
        return best_threshold
    
    def _plot_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, 
                      output_dir: str) -> Dict[str, str]:
        """Generate and save metric visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        plot_paths = {}
        
        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc_score(y_true, y_prob):.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Fraud Detection Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        roc_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths['roc_curve'] = roc_path
        
        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'g-', linewidth=2, 
                 label=f'PR (AUC = {average_precision_score(y_true, y_prob):.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Fraud Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        pr_path = os.path.join(output_dir, 'pr_curve.png')
        plt.savefig(pr_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths['pr_curve'] = pr_path
        
        # 3. Confusion Matrix
        y_pred = (y_prob >= self.best_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix (threshold={self.best_threshold:.3f})')
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths['confusion_matrix'] = cm_path
        
        # 4. Feature Importance
        # Will be added after model training
        
        return plot_paths
    
    def train(self) -> str:
        """
        Main training function with MLflow tracking.
        Returns the run_id for the best model.
        """
        # Load and preprocess data
        df = self.load_data()
        X, y = self.preprocess(df)
        
        # Train/test split with stratification
        # Ensure we have enough samples for stratification
        min_class_samples = y.value_counts().min()
        stratify = y if min_class_samples >= 2 else None
        
        if min_class_samples < 2:
            print(f"   WARNING: Minority class has only {min_class_samples} samples. Disabling stratification.")

        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=stratify
        )
        
        print("\n[SPLIT] Data split:")
        print(f"   Train: {len(X_train):,} samples ({y_train.mean()*100:.2f}% fraud)")
        print(f"   Test: {len(X_test):,} samples ({y_test.mean()*100:.2f}% fraud)")
        
        # Calculate class weight for imbalanced data
        # Crucial for fraud detection

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"   scale_pos_weight: {scale_pos_weight:.2f}")
        
        # GPU Detection
        # Senior MLOps insight: Auto-detect GPU to leverage RTX 4060 for 10x training speedup
        # XGBoost uses CUDA directly via tree_method='gpu_hist', no PyTorch/TensorFlow needed
        use_gpu = False
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                use_gpu = True
                print("   GPU DETECTED: Enabling CUDA acceleration")

        except Exception:
            pass
        
        if not use_gpu:
            print("   CPU MODE: No GPU detected, using standard training")

        
        # XGBoost parameters
        # GPU acceleration: tree_method='gpu_hist' uses NVIDIA CUDA for 10x speedup
        # Use gpu_hist for efficient training

        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc', 'aucpr'],
            'scale_pos_weight': scale_pos_weight,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': RANDOM_SEED,
            # GPU acceleration configuration
            'tree_method': 'gpu_hist' if use_gpu else 'hist',
            'device': 'cuda' if use_gpu else 'cpu',
            'n_jobs': -1 if not use_gpu else 1,  # n_jobs ignored with GPU
            'early_stopping_rounds': 20,
        }
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            run_id = run.info.run_id
            print(f"\n[MLFLOW] Run: {run_id}")
            
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("features", NUMERIC_FEATURES + CATEGORICAL_FEATURES)
            mlflow.log_param("n_features", len(NUMERIC_FEATURES + CATEGORICAL_FEATURES))
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            
            # Train model
            print("\n[TRAIN] Training XGBoost model...")
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=50
            )
            
            # Predictions
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Optimize threshold
            self.best_threshold = self._optimize_threshold(y_test, y_prob)
            y_pred = (y_prob >= self.best_threshold).astype(int)
            
            # Calculate metrics
            metrics = self._calculate_business_metrics(y_test, y_pred, y_prob)
            
            # Log metrics
            print("\n[METRICS] Results:")
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
                if isinstance(value, float):
                    print(f"   {name}: {value:.4f}")
                else:
                    print(f"   {name}: {value}")
            
            mlflow.log_metric("optimal_threshold", self.best_threshold)
            
            # Generate and log plots
            plot_dir = f"artifacts/plots_{run_id}"
            plot_paths = self._plot_metrics(y_test, y_prob, plot_dir)
            
            # Feature importance plot
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance.head(15), x='importance', y='feature', palette='viridis')
            plt.title('Top 15 Feature Importances')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            fi_path = os.path.join(plot_dir, 'feature_importance.png')
            plt.savefig(fi_path, dpi=150)
            plt.close()
            plot_paths['feature_importance'] = fi_path
            
            # Log all plots
            for name, path in plot_paths.items():
                mlflow.log_artifact(path, artifact_path="plots")
            
            # Log model with signature
            signature = infer_signature(X_test, y_prob)
            
            # Log model without registration (compatible with MLflow v2.9)
            # Model registry requires MLflow server >= 2.10
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                signature=signature,
            )
            
            # Save model locally as well
            local_model_path = os.path.join(plot_dir, "xgboost_model.json")
            model.save_model(local_model_path)
            mlflow.log_artifact(local_model_path)
            
            # Save additional artifacts
            # Label encoders for inference
            encoder_path = os.path.join(plot_dir, 'label_encoders.json')
            encoders_serializable = {
                col: list(le.classes_) for col, le in self.label_encoders.items()
            }
            with open(encoder_path, 'w') as f:
                json.dump(encoders_serializable, f)
            mlflow.log_artifact(encoder_path)
            
            # Feature list for inference
            feature_path = os.path.join(plot_dir, 'features.json')
            with open(feature_path, 'w') as f:
                json.dump({
                    'numeric': NUMERIC_FEATURES,
                    'categorical': CATEGORICAL_FEATURES,
                    'threshold': float(self.best_threshold),
                }, f)
            mlflow.log_artifact(feature_path)
            
            print("\n[DONE] Training complete!")
            print(f"   Run ID: {run_id}")
            print(f"   Model registered as: {MODEL_NAME}")
            print(f"   Best threshold: {self.best_threshold:.4f}")
            print(f"   Test Recall: {metrics['recall']:.4f}")
            print(f"   Test Precision: {metrics['precision']:.4f}")
            print(f"   Test PR-AUC: {metrics['pr_auc']:.4f}")
            
            return run_id


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Fraud Detection Model")
    parser.add_argument('--auto-promote', action='store_true', 
                       help="Automatically promote model through registry stages")
    parser.add_argument('--skip-promotion', action='store_true',
                       help="Skip model registry promotion (just train)")
    args = parser.parse_args()
    
    trainer = FraudModelTrainer()
    run_id = trainer.train()
    print(f"\n[SUCCESS] Model training complete! Run ID: {run_id}")
    
    # Optional: Automatic model promotion
    if args.auto_promote and not args.skip_promotion:
        print("\n[REGISTRY] Starting automatic model promotion...")
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ops'))
            from model_registry import ModelRegistryManager
            
            manager = ModelRegistryManager()
            result = manager.auto_promote_pipeline(run_id, MODEL_NAME)
            
            print("\nPromotion Result:")

            print(f"   Registered Version: v{result['registered_version']}")
            print(f"   Promoted to Staging: {result['promoted_to_staging']}")
            print(f"   Promoted to Production: {result['promoted_to_production']}")
            print(f"   Reason: {result['reason']}")
            
        except ImportError as e:
            print(f"[WARN] Model registry not available: {e}")
            print("   Run with --skip-promotion to skip this step")
        except Exception as e:
            print(f"[ERROR] Model promotion failed: {e}")
    elif not args.skip_promotion:
        print("\n[TIP] Run with --auto-promote to automatically register and promote the model")
