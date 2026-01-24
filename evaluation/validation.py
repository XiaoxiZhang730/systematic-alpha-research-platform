import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, Any


def validate_predictions(y_test, y_pred, model_name: str = "model") -> Dict[str, Any]:
    """
    Validate model predictions using Spearman correlation and quintile analysis.
    
    Args:
        y_test: Actual target values
        y_pred: Predicted values
        model_name: Name of the model for display
    
    Returns:
        Dictionary with validation metrics
    """
    y_test_arr = np.asarray(y_test).flatten()
    y_pred_arr = np.asarray(y_pred).flatten()
    
    results = {}
    
    # ===== SPEARMAN CORRELATION =====
    spearman_corr, spearman_pvalue = spearmanr(y_test_arr, y_pred_arr)
    results['spearman_corr'] = float(spearman_corr)
    results['spearman_pvalue'] = float(spearman_pvalue)
    
    # ===== QUINTILE ANALYSIS =====
    try:
        pred_quintile = pd.qcut(
            y_pred_arr, 
            5, 
            labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], 
            duplicates='drop'
        )
        
        quintile_df = pd.DataFrame({
            'quintile': pred_quintile,
            'actual_return': y_test_arr
        })
        
        # IMPORTANT: Use .mean() only, NOT .agg(['mean', 'std', 'count'])
        # This returns a simple Series that we convert to dict
        quintile_means = quintile_df.groupby('quintile', observed=True)['actual_return'].mean()
        
        # Convert to simple dict: {'Q1': float, 'Q2': float, ...}
        results['quintile_returns'] = {str(k): float(v) for k, v in quintile_means.items()}
        
        # Long-Short Spread (Q5 - Q1)
        if len(quintile_means) >= 2:
            results['long_short_spread'] = float(quintile_means.iloc[-1] - quintile_means.iloc[0])
        else:
            results['long_short_spread'] = float('nan')
        
        # Monotonicity check
        results['is_monotonic'] = bool(quintile_means.is_monotonic_increasing)
        
    except Exception as e:
        results['quintile_returns'] = {}
        results['long_short_spread'] = float('nan')
        results['is_monotonic'] = False
        results['quintile_error'] = str(e)
    
    return results
