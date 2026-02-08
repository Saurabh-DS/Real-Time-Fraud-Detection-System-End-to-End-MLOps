"""
Model Registry Management
=========================
Handles the lifecycle of fraud detection models, including registration,
promotion to staging/production, and rollback capabilities.
"""

import os
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MODEL_NAME = os.getenv('MODEL_NAME', 'fraud_detection_model')

# Promotion thresholds - model must beat current production by this margin
PROMOTION_THRESHOLDS = {
    'pr_auc': 0.01,      # 1% improvement in PR-AUC required
    'recall': 0.02,      # 2% improvement in recall
    'precision': -0.05,  # Allow up to 5% precision drop (favor recall)
}

# Minimum metrics for any model to be considered
MINIMUM_REQUIREMENTS = {
    'pr_auc': 0.70,
    'recall': 0.50,
    'precision': 0.80,
}


class ModelRegistryManager:
    """
    Manages model lifecycle in MLflow Model Registry.
    
    Supports:
    - Automatic promotion from None -> Staging -> Production
    - Champion/Challenger comparison
    - Rollback to previous version
    """
    
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.client = MlflowClient()
        
    def register_model(self, run_id: str, model_name: str = MODEL_NAME) -> ModelVersion:
        """
        Register a model from an MLflow run to the Model Registry.
        
        Args:
            run_id: MLflow run ID containing the model
            model_name: Name in the registry
            
        Returns:
            ModelVersion object
        """
        model_uri = f"runs:/{run_id}/model"
        
        try:
            # Register the model
            mv = mlflow.register_model(model_uri, model_name)
            logger.info(f"Registered model {model_name} version {mv.version}")

            
            # Add description
            self.client.update_model_version(
                name=model_name,
                version=mv.version,
                description=f"Registered from run {run_id} at {datetime.now().isoformat()}"
            )
            
            return mv
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")

            raise
    
    def get_production_model(self, model_name: str = MODEL_NAME) -> Optional[ModelVersion]:
        """Get current production model version."""
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            return versions[0] if versions else None
        except Exception as e:
            logger.warning(f"No production model found: {e}")
            return None
    
    def get_staging_model(self, model_name: str = MODEL_NAME) -> Optional[ModelVersion]:
        """Get current staging model version."""
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Staging"])
            return versions[0] if versions else None
        except Exception as e:
            logger.warning(f"No staging model found: {e}")
            return None
    
    def get_model_metrics(self, version: ModelVersion) -> Dict[str, float]:
        """Fetch metrics from the training run of a model version."""
        run_id = version.run_id
        run = self.client.get_run(run_id)
        metrics = run.data.metrics
        
        return {
            'pr_auc': metrics.get('pr_auc', 0),
            'recall': metrics.get('recall', 0),
            'precision': metrics.get('precision', 0),
            'f1': metrics.get('f1', 0),
            'roc_auc': metrics.get('roc_auc', 0),
        }
    
    def meets_minimum_requirements(self, metrics: Dict[str, float]) -> Tuple[bool, str]:
        """Check if model meets minimum deployment requirements."""
        for metric, threshold in MINIMUM_REQUIREMENTS.items():
            if metrics.get(metric, 0) < threshold:
                return False, f"{metric}={metrics.get(metric, 0):.3f} < {threshold}"
        return True, "All requirements met"
    
    def should_promote(
        self, 
        challenger_metrics: Dict[str, float], 
        champion_metrics: Dict[str, float]
    ) -> Tuple[bool, str]:
        """
        Determine if challenger should replace champion.
        
        Uses weighted scoring based on business priorities.
        """
        # Check minimum requirements first
        meets_min, reason = self.meets_minimum_requirements(challenger_metrics)
        if not meets_min:
            return False, f"Challenger fails minimum: {reason}"
        
        # Compare against champion
        improvements = []
        regressions = []
        
        for metric, required_improvement in PROMOTION_THRESHOLDS.items():
            challenger_val = challenger_metrics.get(metric, 0)
            champion_val = champion_metrics.get(metric, 0)
            diff = challenger_val - champion_val
            
            if diff >= required_improvement:
                improvements.append(f"{metric}: +{diff:.3f}")
            elif diff < 0:
                regressions.append(f"{metric}: {diff:.3f}")
        
        # Promote if PR-AUC improved (primary metric)
        pr_auc_improved = (
            challenger_metrics.get('pr_auc', 0) - champion_metrics.get('pr_auc', 0)
        ) >= PROMOTION_THRESHOLDS['pr_auc']
        
        if pr_auc_improved:
            return True, f"Improvements: {', '.join(improvements)}"
        
        return False, f"No significant improvement. Regressions: {', '.join(regressions)}"
    
    def promote_to_staging(self, version: str, model_name: str = MODEL_NAME):
        """Move model version to Staging."""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=True
        )
        logger.info(f"Promoted {model_name} v{version} to Staging")
    
    def promote_to_production(self, version: str, model_name: str = MODEL_NAME):
        """Move model version to Production (archives previous production)."""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        logger.info(f"Promoted {model_name} v{version} to Production")

    
    def rollback_production(self, model_name: str = MODEL_NAME) -> bool:
        """
        Rollback to previous production version.
        
        Returns:
            True if rollback successful, False if no previous version available
        """
        try:
            # Get all versions
            all_versions = self.client.search_model_versions(f"name='{model_name}'")
            
            # Find archived production versions
            archived_prod = [
                v for v in all_versions 
                if v.current_stage == "Archived" and "Production" in str(v.description)
            ]
            
            if not archived_prod:
                logger.error("No previous production version to rollback to")
                return False
            
            # Get most recent archived production
            prev_version = max(archived_prod, key=lambda v: int(v.version))
            
            # Promote back to production
            self.promote_to_production(prev_version.version, model_name)
            logger.info(f"Rolled back to {model_name} v{prev_version.version}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def auto_promote_pipeline(self, run_id: str, model_name: str = MODEL_NAME) -> Dict:
        """
        Full automatic promotion pipeline.
        
        1. Register model from run
        2. Promote to Staging
        3. Compare with Production champion
        4. If better, promote to Production
        
        Returns:
            Dict with promotion results
        """
        result = {
            'run_id': run_id,
            'registered_version': None,
            'promoted_to_staging': False,
            'promoted_to_production': False,
            'reason': ''
        }
        
        try:
            # 1. Register
            mv = self.register_model(run_id, model_name)
            result['registered_version'] = mv.version
            
            # 2. Get challenger metrics
            challenger_metrics = self.get_model_metrics(mv)
            
            # 3. Check minimum requirements
            meets_min, reason = self.meets_minimum_requirements(challenger_metrics)
            if not meets_min:
                result['reason'] = f"Failed minimum requirements: {reason}"
                return result
            
            # 4. Promote to Staging
            self.promote_to_staging(mv.version, model_name)
            result['promoted_to_staging'] = True
            
            # 5. Compare with production
            prod_model = self.get_production_model(model_name)
            
            if prod_model is None:
                # No production model, promote directly
                self.promote_to_production(mv.version, model_name)
                result['promoted_to_production'] = True
                result['reason'] = "First production model"
            else:
                champion_metrics = self.get_model_metrics(prod_model)
                should_promote, reason = self.should_promote(challenger_metrics, champion_metrics)
                
                if should_promote:
                    self.promote_to_production(mv.version, model_name)
                    result['promoted_to_production'] = True
                    result['reason'] = reason
                else:
                    result['reason'] = f"Staying with champion: {reason}"
            
            return result
            
        except Exception as e:
            result['reason'] = f"Pipeline error: {str(e)}"
            logger.error(result['reason'])
            return result


def main():
    """CLI for model registry operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Registry Manager")
    parser.add_argument('action', choices=['promote', 'rollback', 'status'])
    parser.add_argument('--run-id', help="MLflow run ID for promotion")
    parser.add_argument('--model-name', default=MODEL_NAME)
    
    args = parser.parse_args()
    
    manager = ModelRegistryManager()
    
    if args.action == 'promote':
        if not args.run_id:
            print("Error: --run-id required for promote action")
            return
        result = manager.auto_promote_pipeline(args.run_id, args.model_name)
        print("\nPromotion Result:")
        for k, v in result.items():
            print(f"  {k}: {v}")
            
    elif args.action == 'rollback':
        success = manager.rollback_production(args.model_name)
        print(f"Rollback {'successful' if success else 'failed'}")
        
    elif args.action == 'status':
        prod = manager.get_production_model(args.model_name)
        staging = manager.get_staging_model(args.model_name)
        
        print(f"\nModel Registry Status: {args.model_name}")
        print(f"  Production: v{prod.version if prod else 'None'}")
        print(f"  Staging: v{staging.version if staging else 'None'}")
        
        if prod:
            metrics = manager.get_model_metrics(prod)
            print("\n  Production Metrics:")
            for k, v in metrics.items():
                print(f"    {k}: {v:.4f}")


if __name__ == "__main__":
    main()
