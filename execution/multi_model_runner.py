from run_experiment import run_experiment
from typing import Any, Dict, List, Tuple, Optional
import numpy as np


def run_all_models(
    cfg: Any,
    tuning_results: Optional[Dict[str, Dict[str, Any]]] = None
) -> List[Tuple[str, Optional[Dict]]]:
    """
    Run all models with optional pre-tuned parameters.
    
    Args:
        cfg: Configuration object
        tuning_results: Results from tune_all_models() 
                       Format: {model_name: {'best_params': {...}, 'best_score': float}}
    
    Returns:
        List of (model_name, results) tuples
    """
    
    model_names = ["ridge", "lasso", "random_forest", "gbm"]
    results = []
    
    # Determine if we're using tuned params
    using_tuned = tuning_results is not None
    
    for model_name in model_names:
        print(f"\n🧪 Running model: {model_name}")
        
        # Get tuned params if available
        best_params = None
        if using_tuned and model_name in tuning_results:
            best_params = tuning_results[model_name].get('best_params')
            print(f"   Using tuned params: {best_params}")
        
        try:
            result = run_experiment(cfg, model_name=model_name, best_params=best_params)
            results.append((model_name, result))
        except Exception as e:
            print(f"❌ Failed on model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((model_name, None))

    # Print summary table
    print_summary(results, tuned=using_tuned)
    
    return results


def print_summary(results: List[Tuple[str, Optional[Dict]]], tuned: bool = False):
    """Print summary table."""
    
    title = "All Models Complete"
    if tuned:
        title += " (TUNED)"
    
    print(f"\n=== ✅ {title} ===")
    print(f"{'Model':<15} {'R2':>8} {'MAE':>8} {'RMSE':>8} {'RankIC':>8} {'L/S':>10} {'#Feat':>8}")
    print("-" * 75)
    
    for model_name, output in results:
        if output is None:
            print(f"{model_name:<15} FAILED")
            continue
        
        metrics = output.get('metrics', {})
        portfolio = output.get('portfolio_metrics', output.get('validation', {}))
        
        r2 = metrics.get('r2', float('nan'))
        mae = metrics.get('mae', float('nan'))
        rmse = metrics.get('rmse', float('nan'))
        ric = output.get('rank_ic_mean', float('nan'))
        ls = portfolio.get('long_short_spread', float('nan'))
        n_feat = len(output.get('used_features', []))
        
        # Format values
        def fmt(val):
            if isinstance(val, float) and np.isnan(val):
                return "     N/A"
            return f"{val:8.4f}"
        
        print(f"{model_name:<15} {fmt(r2)} {fmt(mae)} {fmt(rmse)} {fmt(ric)} {fmt(ls)} {n_feat:>8}")
    
    print("-" * 75)
    
    # Find best model
    valid = [(n, o) for n, o in results if o is not None]
    if valid:
        best_name, best_out = max(valid, key=lambda x: x[1].get('rank_ic_mean', 0))
        print(f"\n🏆 Best Model: {best_name.upper()}")
        print(f"   Rank IC: {best_out.get('rank_ic_mean', 0):.4f}")
        print(f"   L/S Spread: {best_out.get('portfolio_metrics', best_out.get('validation', {})).get('long_short_spread', 0):+.4f}")
        
        if best_out.get('best_params'):
            print(f"   Tuned Params: {best_out['best_params']}")
