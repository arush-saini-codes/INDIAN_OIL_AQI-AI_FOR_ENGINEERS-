"""src/metrics.py
Shared Evaluation Metrics Module for UCS321 AQI Forecasting.

Provides standard regression metrics (RMSE, MAE, MAPE, R2),
CPCB qualitative AQI category mapping, categorical accuracy metrics
(Accuracy, Macro-F1, Quadratic Weighted Kappa, Confusion Matrix),
and standard predictions formatting across Phase 5 (ARIMA) and Phase 6 (LSTM).
All evaluations are strictly performed on original, unscaled AQI values.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
)

# Standard CPCB air quality index categories and ordinal ranking
CPCB_CATEGORIES = [
    'Good',          # 0-50
    'Satisfactory',  # 51-100
    'Moderate',      # 101-200
    'Poor',          # 201-300
    'Very Poor',     # 301-400
    'Severe'         # 401-500
]


def aqi_to_cpcb_category(aqi):
    """Maps continuous AQI scalar, array, or Series into CPCB discrete health categories."""
    if isinstance(aqi, (pd.Series, np.ndarray, list)):
        arr = np.asarray(aqi)
        cats = np.empty(arr.shape, dtype=object)
        cats[arr <= 50] = 'Good'
        cats[(arr > 50) & (arr <= 100)] = 'Satisfactory'
        cats[(arr > 100) & (arr <= 200)] = 'Moderate'
        cats[(arr > 200) & (arr <= 300)] = 'Poor'
        cats[(arr > 300) & (arr <= 400)] = 'Very Poor'
        cats[arr > 400] = 'Severe'
        return cats
    else:
        if aqi <= 50:
            return 'Good'
        elif aqi <= 100:
            return 'Satisfactory'
        elif aqi <= 200:
            return 'Moderate'
        elif aqi <= 300:
            return 'Poor'
        elif aqi <= 400:
            return 'Very Poor'
        else:
            return 'Severe'


def compute_regression_metrics(y_true, y_pred):
    """Computes RMSE, MAE, MAPE (%), and R2 on unscaled AQI series."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Filter any NaNs
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[valid]
    y_p = y_pred[valid]
    
    if len(y_t) == 0:
        return {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan, 'R2': np.nan}
    
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)
    # MAPE with protective epsilon to avoid division by zero
    mape = np.mean(np.abs((y_t - y_p) / np.maximum(y_t, 1e-6))) * 100.0
    r2 = r2_score(y_t, y_p)
    
    return {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'MAPE': float(mape),
        'R2': float(r2)
    }


def compute_category_metrics(y_true, y_pred, labels=None):
    """Computes multi-class category Accuracy, Macro-F1, and Quadratic Weighted Kappa.
    If inputs are numeric AQI, they are mapped to CPCB categories automatically.
    """
    if labels is None:
        labels = CPCB_CATEGORIES
        
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    
    # Check if inputs are numeric
    if np.issubdtype(y_true_arr.dtype, np.number):
        cat_true = aqi_to_cpcb_category(y_true_arr)
    else:
        cat_true = y_true_arr
        
    if np.issubdtype(y_pred_arr.dtype, np.number):
        cat_pred = aqi_to_cpcb_category(y_pred_arr)
    else:
        cat_pred = y_pred_arr
        
    acc = accuracy_score(cat_true, cat_pred)
    macro_f1 = f1_score(cat_true, cat_pred, labels=labels, average='macro', zero_division=0)
    qwk = cohen_kappa_score(cat_true, cat_pred, labels=labels, weights='quadratic')
    cm = confusion_matrix(cat_true, cat_pred, labels=labels)
    
    return {
        'Accuracy': float(acc),
        'Macro_F1': float(macro_f1),
        'QWK': float(qwk),
        'Confusion_Matrix': cm
    }


def evaluate_forecast(y_true, y_pred, model_name=None, split_name=None, horizon=1):
    """Returns a consolidated dictionary containing all regression and category metrics."""
    reg = compute_regression_metrics(y_true, y_pred)
    cat = compute_category_metrics(y_true, y_pred)
    
    result = {
        'model': model_name,
        'split': split_name,
        'horizon': horizon,
        'RMSE': round(reg['RMSE'], 3),
        'MAE': round(reg['MAE'], 3),
        'MAPE': round(reg['MAPE'], 2),
        'R2': round(reg['R2'], 4),
        'Accuracy': round(cat['Accuracy'], 4),
        'Macro_F1': round(cat['Macro_F1'], 4),
        'QWK': round(cat['QWK'], 4)
    }
    return result


def create_predictions_dataframe(dates, y_true, y_pred, horizon=1, model="Model", split="test"):
    """Creates a standard predictions dataframe matching project standards:
    Columns: date, actual, predicted, horizon, model, split
    """
    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'actual': np.asarray(y_true, dtype=float),
        'predicted': np.asarray(y_pred, dtype=float),
        'horizon': int(horizon),
        'model': str(model),
        'split': str(split)
    })
    return df


def diebold_mariano_test(y_true, y_pred1, y_pred2, h=1, criterion='MSE'):
    """Diebold-Mariano test for predictive accuracy comparison with 
    Harvey-Leybourne-Newbold (HLN 1997) small-sample correction.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth actual values.
    y_pred1 : array-like
        Predictions from benchmark / model 1.
    y_pred2 : array-like
        Predictions from challenger / model 2.
    h : int, default=1
        Forecast horizon (steps ahead).
    criterion : str, default='MSE'
        Loss function: 'MSE' (squared error) or 'MAE' (absolute error).
        
    Returns
    -------
    dict:
        DM_stat : float, HLN-adjusted DM test statistic.
        p_value : float, two-sided p-value from Student's t distribution with T-1 df.
        mean_loss_diff : float, mean loss differential (model1_loss - model2_loss).
                         Positive indicates model 2 has smaller loss.
    """
    from scipy import stats

    y_t = np.asarray(y_true, dtype=float)
    y_p1 = np.asarray(y_pred1, dtype=float)
    y_p2 = np.asarray(y_pred2, dtype=float)

    valid = ~np.isnan(y_t) & ~np.isnan(y_p1) & ~np.isnan(y_p2)
    y_t, y_p1, y_p2 = y_t[valid], y_p1[valid], y_p2[valid]

    T = len(y_t)
    if T < 3:
        return {'DM_stat': np.nan, 'p_value': np.nan, 'mean_loss_diff': np.nan}

    e1 = y_t - y_p1
    e2 = y_t - y_p2

    if criterion.upper() == 'MAE':
        d = np.abs(e1) - np.abs(e2)
    else:
        d = e1**2 - e2**2

    d_mean = np.mean(d)

    # Compute autocovariances of d up to lag h-1
    gamma = np.zeros(h)
    d_centered = d - d_mean
    for k in range(h):
        gamma[k] = np.mean(d_centered[k:] * d_centered[:T-k]) if k > 0 else np.var(d, ddof=0)

    # Long-run variance estimator
    var_d = gamma[0] + 2.0 * np.sum(gamma[1:]) if h > 1 else gamma[0]
    var_d = max(var_d, 1e-12) / T

    # Standard DM statistic
    dm_raw = d_mean / np.sqrt(var_d)

    # Harvey, Leybourne, Newbold (1997) small-sample correction
    hln_factor = np.sqrt((T + 1 - 2 * h + (h * (h - 1)) / T) / T)
    dm_stat = dm_raw * hln_factor

    # Two-sided p-value from Student's t distribution with T-1 degrees of freedom
    df_t = T - 1
    p_value = 2.0 * stats.t.sf(np.abs(dm_stat), df=df_t)

    return {
        'DM_stat': float(dm_stat),
        'p_value': float(p_value),
        'mean_loss_diff': float(d_mean)
    }


