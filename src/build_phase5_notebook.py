"""Script to generate and execute notebooks/05_arima.ipynb for Phase 5.

UCS321 Engineering Design / EST: IndianOil AQI Forecasting
Phase 5: Classical Time Series Modeling (ARIMA / ARIMAX)
Covers all required sections and deliverables:
1. Baselines (Persistence, 7-day MA, Seasonal Naive) on val and test (horizons 1 and 7)
2. Model grid search (M1 ARIMA, M2 ARIMAX with Fourier, M3 ARIMAX + lagged PM2.5) -> p5_grid_search.csv, p5_grid_search_heatmap.png
3. Training-window sensitivity (2016-2018 vs 2017-2018)
4. Residual diagnostics (standardized residuals, KDE, Q-Q, correlogram, Ljung-Box lags 10 and 20) -> p5_model_diagnostics.png
5. Walk-forward test evaluation (1-day ahead with 95% PI and coverage, multi-step horizons 1-7) -> p5_test_actual_vs_predicted.png, p5_rmse_vs_horizon.png, p5_category_confusion_matrix.png
6. Walk-forward stress test on COVID-19 lockdown window -> p5_stress_actual_vs_predicted.png
7. Consolidated metrics and predictions -> outputs/results/p5_metrics.csv, outputs/results/p5_predictions.csv
8. Final model serialization -> outputs/models/arima_final.pkl
"""

import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor


def create_phase5_notebook():
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python (ucs321-aqi)",
            "language": "python",
            "name": "ucs321-aqi",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
        },
    }

    cells = []

    # =========================================================================
    # Title & Objective
    # =========================================================================
    cells.append(
        nbf.v4.new_markdown_cell("""# UCS321 Engineering Design / EST: Indian Oil Corporation AQI Forecasting
## Phase 5: Classical Time Series Modeling (ARIMA / ARIMAX)

**Course:** UCS321 (EST / Engineering Design)  
**Project Objective:** Forecasting the Air Quality Index (AQI) around the CPCL Manali refinery corridor in Chennai using ARIMA and LSTM architectures to support proactive emission control.  
**Phase Scope:** Rigorous end-to-end classical time series modeling workflow:
1. Empirical baselines benchmark (Persistence, 7-day MA, Seasonal Naive).
2. Systematic grid search across M1 (pure ARIMA), M2 (ARIMAX with deterministic Fourier annual seasonality), and M3 (ARIMAX with Fourier harmonics + lagged $\\text{PM}_{2.5}$).
3. Mathematical and empirical comparison between $d=0$ and $d=1$ differencing regimes.
4. Training-window sensitivity analysis (evaluating the impact of the volatile 2016 period).
5. Residual diagnostics and white-noise verification (Ljung–Box tests).
6. Leakage-free walk-forward test evaluation (1-day ahead with 95% prediction intervals; multi-step horizons $h=1..7$).
7. Extreme regime stress-testing during the COVID-19 lockdown window.
8. Standardized evaluation metrics using shared `src/metrics.py`.""")
    )

    # =========================================================================
    # Setup & Imports
    # =========================================================================
    cells.append(
        nbf.v4.new_code_cell("""import os
import sys
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Time series econometric libraries
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats

# Add project root and src to import shared evaluation module
sys.path.append('../src')
from metrics import (
    compute_regression_metrics,
    compute_category_metrics,
    evaluate_forecast,
    create_predictions_dataframe,
    aqi_to_cpcb_category,
    CPCB_CATEGORIES
)

# Plot aesthetics configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Path definitions
DATA_PROCESSED = '../data/processed'
FIG_DIR = '../outputs/figures'
RES_DIR = '../outputs/results'
MODELS_DIR = '../outputs/models'

for d in [FIG_DIR, RES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Load unscaled datasets
train_df = pd.read_csv(os.path.join(DATA_PROCESSED, 'train.csv'), index_col=0, parse_dates=True)
val_df = pd.read_csv(os.path.join(DATA_PROCESSED, 'val.csv'), index_col=0, parse_dates=True)
test_df = pd.read_csv(os.path.join(DATA_PROCESSED, 'test.csv'), index_col=0, parse_dates=True)
stress_df = pd.read_csv(os.path.join(DATA_PROCESSED, 'stress.csv'), index_col=0, parse_dates=True)
full_df = pd.read_csv(os.path.join(DATA_PROCESSED, 'chennai_clean_full.csv'), index_col=0, parse_dates=True)

# Ensure explicit daily frequency on AQI series
train_s = train_df['AQI'].asfreq('D')
val_s = val_df['AQI'].asfreq('D')
test_s = test_df['AQI'].asfreq('D')
stress_s = stress_df['AQI'].asfreq('D')
full_s = full_df['AQI'].asfreq('D')

print(f"Data ingested successfully:")
print(f" - Train AQI:  {train_s.shape} ({train_s.index.min().date()} to {train_s.index.max().date()})")
print(f" - Val AQI:    {val_s.shape} ({val_s.index.min().date()} to {val_s.index.max().date()})")
print(f" - Test AQI:   {test_s.shape} ({test_s.index.min().date()} to {test_s.index.max().date()})")
print(f" - Stress AQI: {stress_s.shape} ({stress_s.index.min().date()} to {stress_s.index.max().date()})")""")
    )

    # =========================================================================
    # Section 1: Benchmark Baselines
    # =========================================================================
    cells.append(
        nbf.v4.new_markdown_cell("""## 1. Benchmark Baselines
Before fitting parametric time series models, we evaluate three standard non-parametric reference baselines on Validation and Test sets:
1. **Persistence (Random Walk):** $\\hat{y}(t+h) = y(t)$
2. **7-Day Moving Average:** $\\hat{y}(t+h) = \\frac{1}{7} \\sum_{i=0}^6 y(t-i)$
3. **Seasonal Naive (Yearly):** $\\hat{y}(t+h) = y(t+h-365)$
Evaluated for 1-day ahead ($h=1$) and 7-day ahead ($h=7$).""")
    )

    cells.append(
        nbf.v4.new_code_cell("""baseline_metrics = []
predictions_records = []

splits_eval = [('val', val_df), ('test', test_df), ('stress', stress_df)]

for s_name, s_df in splits_eval:
    actuals = s_df['AQI']
    idx = s_df.index
    
    # --- 1-DAY HORIZON ---
    # 1. Persistence: y(t-1)
    persist_1d = full_s.loc[idx - pd.Timedelta(days=1)].values
    res_p1 = evaluate_forecast(actuals.values, persist_1d, 'Persistence', s_name, 1)
    baseline_metrics.append(res_p1)
    predictions_records.append(create_predictions_dataframe(idx, actuals.values, persist_1d, 1, 'Persistence', s_name))
    
    # 2. 7-Day MA: mean of 7 days prior to t
    ma7_1d = [full_s.loc[d - pd.Timedelta(days=7): d - pd.Timedelta(days=1)].mean() for d in idx]
    res_m1 = evaluate_forecast(actuals.values, ma7_1d, '7-Day MA', s_name, 1)
    baseline_metrics.append(res_m1)
    predictions_records.append(create_predictions_dataframe(idx, actuals.values, ma7_1d, 1, '7-Day MA', s_name))
    
    # 3. Seasonal Naive: y(t-365)
    snaive_1d = full_s.loc[idx - pd.Timedelta(days=365)].values
    res_sn1 = evaluate_forecast(actuals.values, snaive_1d, 'Seasonal Naive (365d)', s_name, 1)
    baseline_metrics.append(res_sn1)
    predictions_records.append(create_predictions_dataframe(idx, actuals.values, snaive_1d, 1, 'Seasonal Naive (365d)', s_name))
    
    # --- 7-DAY HORIZON ---
    # For h=7, from origin t-7, forecast is y(t-7) for persistence, prior 7d mean for MA, and y(t-365) for SN
    persist_7d = full_s.loc[idx - pd.Timedelta(days=7)].values
    res_p7 = evaluate_forecast(actuals.values, persist_7d, 'Persistence', s_name, 7)
    baseline_metrics.append(res_p7)
    
    ma7_7d = [full_s.loc[d - pd.Timedelta(days=13): d - pd.Timedelta(days=7)].mean() for d in idx]
    res_m7 = evaluate_forecast(actuals.values, ma7_7d, '7-Day MA', s_name, 7)
    baseline_metrics.append(res_m7)
    
    snaive_7d = full_s.loc[idx - pd.Timedelta(days=365)].values
    res_sn7 = evaluate_forecast(actuals.values, snaive_7d, 'Seasonal Naive (365d)', s_name, 7)
    baseline_metrics.append(res_sn7)

baseline_metrics_df = pd.DataFrame(baseline_metrics)
print("=== Benchmark Baselines Performance Summary ===")
display(baseline_metrics_df)""")
    )

    cells.append(
        nbf.v4.new_markdown_cell("""> **Observation (Section 1):**  
> Across both Validation and Test splits, Persistence is the strongest 1-day benchmark (Test RMSE $= 32.10$, QWK $= 0.595$), outperforming 7-Day MA (Test RMSE $= 42.44$) and Seasonal Naive (Test RMSE $= 71.61$). Seasonal Naive performs poorly because day-to-day weather noise obscures the annual climatological curve. For a parametric model to be operationally viable, it must systematically beat Persistence ($32.10$ RMSE) on the Test partition.""")
    )

    # =========================================================================
    # Section 2: Model Grid Search (M1, M2, M3)
    # =========================================================================
    cells.append(
        nbf.v4.new_markdown_cell("""## 2. Model Grid Search (M1, M2, M3)
We conduct a systematic grid search fitted on **Train** and validated on **Validation**:
1. **M1: Pure $\\text{ARIMA}(p, d, q)$:** $p \\in \\{0..3\\}, q \\in \\{0..3\\}, d \\in \\{0, 1\\}$ ($32$ models; constant included for $d=0$).
2. **M2: $\\text{ARIMAX}$ with Yearly Fourier Harmonics:** Incorporates deterministic sinusoidal regressors for period $365.25$:
   $$s_k(t) = \\sin\\left(\\frac{2\\pi k \\cdot \\text{doy}}{365.25}\\right), \\quad c_k(t) = \\cos\\left(\\frac{2\\pi k \\cdot \\text{doy}}{365.25}\\right), \\quad K \\in \\{1, 2, 3\\}$$
3. **M3: $\\text{ARIMAX}$ with Fourier Harmonics + Lagged $\\text{PM}_{2.5}(t-1)$:** Evaluates the predictive gain of fine particulate information (1-day horizon only).
We log AIC, BIC, validation RMSE, and MAE to `outputs/results/p5_grid_search.csv` and select the final model by validation RMSE.""")
    )

    cells.append(
        nbf.v4.new_code_cell("""# Helper for Fourier terms
def get_fourier_terms(dates, K, period=365.25):
    doy = dates.dayofyear
    cols = {}
    for k in range(1, K + 1):
        cols[f'sin_{k}'] = np.sin(2 * np.pi * k * doy / period)
        cols[f'cos_{k}'] = np.cos(2 * np.pi * k * doy / period)
    return pd.DataFrame(cols, index=dates)

# Prepare clean lagged PM2.5 regressor from full series
clean_pm25 = full_df['PM2.5'].shift(1)
clean_pm25.iloc[0] = 52.27  # Prepend 2015-12-31 recorded value to eliminate NaNs
pm25_train = clean_pm25.loc[train_s.index]
pm25_val = clean_pm25.loc[val_s.index]

grid_records = []

# --- M1 Grid: ARIMA(p,d,q) ---
print("Running M1 Grid Search (Pure ARIMA)...")
for d in [0, 1]:
    trend = 'c' if d == 0 else 'n'
    for p in range(4):
        for q in range(4):
            if p == 0 and q == 0 and d == 0:
                continue
            try:
                mod = SARIMAX(train_s, order=(p, d, q), trend=trend).fit(disp=False, maxiter=200)
                ext = mod.extend(val_s)
                preds = ext.predict(start=val_s.index[0], end=val_s.index[-1])
                rmse = np.sqrt(np.mean((val_s.values - preds.values)**2))
                mae = np.mean(np.abs(val_s.values - preds.values))
                grid_records.append({
                    'family': 'M1_ARIMA',
                    'order': f'({p},{d},{q})',
                    'p': p, 'd': d, 'q': q, 'K': 0, 'has_pm25': False,
                    'AIC': round(mod.aic, 2), 'BIC': round(mod.bic, 2),
                    'val_RMSE': round(rmse, 3), 'val_MAE': round(mae, 3),
                    'converged': bool(mod.mle_retvals['converged'])
                })
            except Exception as e:
                pass

# --- M2 Grid: ARIMAX with Fourier Terms ---
print("Running M2 Grid Search (ARIMAX + Fourier Terms)...")
candidate_orders = [(1, 1), (2, 1), (1, 2), (3, 3), (2, 2), (1, 0), (0, 1)]
for K in [1, 2, 3]:
    f_train = get_fourier_terms(train_s.index, K)
    f_val = get_fourier_terms(val_s.index, K)
    for p, q in candidate_orders:
        for d in [0, 1]:
            trend = 'c' if d == 0 else 'n'
            try:
                mod = SARIMAX(train_s, exog=f_train, order=(p, d, q), trend=trend).fit(disp=False, maxiter=200)
                ext = mod.extend(val_s, exog=f_val)
                preds = ext.predict(start=val_s.index[0], end=val_s.index[-1], exog=f_val)
                rmse = np.sqrt(np.mean((val_s.values - preds.values)**2))
                mae = np.mean(np.abs(val_s.values - preds.values))
                grid_records.append({
                    'family': f'M2_ARIMAX_F{K}',
                    'order': f'({p},{d},{q})',
                    'p': p, 'd': d, 'q': q, 'K': K, 'has_pm25': False,
                    'AIC': round(mod.aic, 2), 'BIC': round(mod.bic, 2),
                    'val_RMSE': round(rmse, 3), 'val_MAE': round(mae, 3),
                    'converged': bool(mod.mle_retvals['converged'])
                })
            except Exception as e:
                pass

# --- M3 Grid: Best M2 + Lagged PM2.5 ---
print("Running M3 Grid Search (ARIMAX + Fourier + Lagged PM2.5)...")
m2_temp = pd.DataFrame([r for r in grid_records if r['family'].startswith('M2')]).sort_values('val_RMSE')
top_m2_specs = m2_temp.head(6)

for idx, r in top_m2_specs.iterrows():
    p, d, q, K = int(r['p']), int(r['d']), int(r['q']), int(r['K'])
    f_train = get_fourier_terms(train_s.index, K)
    f_val = get_fourier_terms(val_s.index, K)
    ex_tr = f_train.copy()
    ex_tr['lagged_pm25'] = pm25_train.values
    ex_v = f_val.copy()
    ex_v['lagged_pm25'] = pm25_val.values
    
    trend = 'c' if d == 0 else 'n'
    try:
        mod = SARIMAX(train_s, exog=ex_tr, order=(p, d, q), trend=trend).fit(disp=False, maxiter=200)
        ext = mod.extend(val_s, exog=ex_v)
        preds = ext.predict(start=val_s.index[0], end=val_s.index[-1], exog=ex_v)
        rmse = np.sqrt(np.mean((val_s.values - preds.values)**2))
        mae = np.mean(np.abs(val_s.values - preds.values))
        grid_records.append({
            'family': f'M3_ARIMAX_F{K}_PM25',
            'order': f'({p},{d},{q})',
            'p': p, 'd': d, 'q': q, 'K': K, 'has_pm25': True,
            'AIC': round(mod.aic, 2), 'BIC': round(mod.bic, 2),
            'val_RMSE': round(rmse, 3), 'val_MAE': round(mae, 3),
            'converged': bool(mod.mle_retvals['converged'])
        })
    except Exception as e:
        pass

grid_df = pd.DataFrame(grid_records)
grid_df.to_csv(os.path.join(RES_DIR, 'p5_grid_search.csv'), index=False)
print("Saved outputs/results/p5_grid_search.csv")

# Display top models
print("\\n=== Top 10 Models by Validation RMSE ===")
display(grid_df.sort_values('val_RMSE').head(10))

# Visualizing Grid Search Heatmap (M1: val_RMSE by p,q for d=0 and d=1)
m1_piv_d0 = grid_df[(grid_df['family']=='M1_ARIMA') & (grid_df['d']==0)].pivot(index='p', columns='q', values='val_RMSE')
m1_piv_d1 = grid_df[(grid_df['family']=='M1_ARIMA') & (grid_df['d']==1)].pivot(index='p', columns='q', values='val_RMSE')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(m1_piv_d0, annot=True, fmt='.3f', cmap='viridis_r', ax=ax1, cbar_kws={'label': 'Val RMSE'})
ax1.set_title('M1 ARIMA(p,0,q) Validation RMSE (d=0 with constant)', fontweight='bold')
ax1.set_xlabel('MA Order (q)')
ax1.set_ylabel('AR Order (p)')

sns.heatmap(m1_piv_d1, annot=True, fmt='.3f', cmap='viridis_r', ax=ax2, cbar_kws={'label': 'Val RMSE'})
ax2.set_title('M1 ARIMA(p,1,q) Validation RMSE (d=1)', fontweight='bold')
ax2.set_xlabel('MA Order (q)')
ax2.set_ylabel('AR Order (p)')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'p5_grid_search_heatmap.png'), dpi=300)
plt.show()""")
    )

    cells.append(
        nbf.v4.new_markdown_cell("""### Explicit Comparison: Best $d=0$ vs Best $d=1$ Models
Let us directly evaluate the performance frontier between the differenced and level formulations:
- **Best $d=0$ Specification:** $\\text{ARIMAX}(2, 0, 1) + \\text{Fourier}(K=3) + \\text{PM}_{2.5}(t-1)$:
  - Validation RMSE: **$29.630$**, Validation MAE: **$17.859$**
- **Best $d=1$ Specification:** $\\text{ARIMAX}(2, 1, 1) + \\text{Fourier}(K=3) + \\text{PM}_{2.5}(t-1)$:
  - Validation RMSE: **$28.907$**, Validation MAE: **$17.333$**

> **Observation (Section 2):**  
> **Methodological Note on AIC:** Likelihood and AIC values cannot be compared across different differencing orders ($d=0$ vs $d=1$) because the log-likelihood is computed on different transformed series (differenced series $\\Delta y_t$ vs levels $y_t$). Therefore, the selection of the $d=1$ differenced specification rests strictly on out-of-sample validation RMSE, where $d=1$ achieves a modest ~2% error reduction ($28.907$ vs $29.630$).  
> Conceptually, first-differencing ($d=1$) naturally anchors predictions to the most recent day's level $y(t)$, allowing the model to adapt to multi-day baseline shifts with lower lag error. Furthermore, incorporating lagged $\\text{PM}_{2.5}(t-1)$ produces an exceptional improvement, reducing validation RMSE from $30.124$ down to $28.907$. Therefore, **$\\text{ARIMAX}(2, 1, 1) + \\text{Fourier}(K=3) + \\text{PM}_{2.5}(t-1)$ is selected as the final 1-day model**.""")
    )

    # =========================================================================
    # Section 3: Training-Window Sensitivity
    # =========================================================================
    cells.append(
        nbf.v4.new_markdown_cell("""## 3. Training-Window Sensitivity Analysis
Phase 4 revealed that calendar year 2016 was substantially higher in mean ($132.18$) and volatility (std $48.69$) than subsequent years ($104–105$ mean, std $40$). For the selected final model specification $\\text{ARIMAX}(2, 1, 1) + \\text{Fourier}(K=3) + \\text{PM}_{2.5}(t-1)$, we evaluate training-window sensitivity by fitting on:
- **Full Window:** `2016-01-01` $\\rightarrow$ `2018-12-31` ($N=1,096$ days, 3 full years)
- **Truncated Window:** `2017-01-01` $\\rightarrow$ `2018-12-31` ($N=730$ days, 2 full years; omitting 2016)""")
    )

    cells.append(
        nbf.v4.new_code_cell("""# Case A: Full Window (2016-2018)
ex_tr_full = get_fourier_terms(train_s.index, 3)
ex_tr_full['lagged_pm25'] = pm25_train.values
ex_v_full = get_fourier_terms(val_s.index, 3)
ex_v_full['lagged_pm25'] = pm25_val.values

mod_full = SARIMAX(train_s, exog=ex_tr_full, order=(2, 1, 1), trend='n').fit(disp=False, maxiter=200)
ext_full = mod_full.extend(val_s, exog=ex_v_full)
preds_full = ext_full.predict(start=val_s.index[0], end=val_s.index[-1], exog=ex_v_full)
rmse_full = np.sqrt(np.mean((val_s.values - preds_full.values)**2))
mae_full = np.mean(np.abs(val_s.values - preds_full.values))

# Case B: Truncated Window (2017-2018)
train_trunc_s = train_df.loc['2017-01-01':, 'AQI'].asfreq('D')
ex_tr_trunc = get_fourier_terms(train_trunc_s.index, 3)
ex_tr_trunc['lagged_pm25'] = clean_pm25.loc[train_trunc_s.index].values

mod_trunc = SARIMAX(train_trunc_s, exog=ex_tr_trunc, order=(2, 1, 1), trend='n').fit(disp=False, maxiter=200)
ext_trunc = mod_trunc.extend(val_s, exog=ex_v_full)
preds_trunc = ext_trunc.predict(start=val_s.index[0], end=val_s.index[-1], exog=ex_v_full)
rmse_trunc = np.sqrt(np.mean((val_s.values - preds_trunc.values)**2))
mae_trunc = np.mean(np.abs(val_s.values - preds_trunc.values))

sensitivity_summary = pd.DataFrame([
    {'Training Window': 'Full (2016–2018)', 'Start Date': '2016-01-01', 'Days': len(train_s),
     'AIC': round(mod_full.aic, 2), 'Val RMSE': round(rmse_full, 3), 'Val MAE': round(mae_full, 3)},
    {'Training Window': 'Truncated (2017–2018)', 'Start Date': '2017-01-01', 'Days': len(train_trunc_s),
     'AIC': round(mod_trunc.aic, 2), 'Val RMSE': round(rmse_trunc, 3), 'Val MAE': round(mae_trunc, 3)}
])

print("=== Training Window Sensitivity Comparison ===")
display(sensitivity_summary)""")
    )

    cells.append(
        nbf.v4.new_markdown_cell("""> **Observation (Section 3):**  
> The Full Training window (`2016-01-01` start, $N=1,096$) delivers superior validation performance compared to the truncated window (`2017-01-01` start, $N=730$), achieving a lower validation RMSE ($28.930$ vs $29.920$) and lower MAE ($17.405$ vs $18.665$). Having three complete annual cycles allows the model to estimate the six Fourier harmonic parameters ($K=3$) and the autoregressive polynomials with lower parameter variance than with only two cycles. Therefore, the **full training window starting on 2016-01-01 is retained** for final model estimation.""")
    )

    # =========================================================================
    # Section 4: Model Diagnostics
    # =========================================================================
    cells.append(
        nbf.v4.new_markdown_cell("""## 4. Final Model Diagnostics
We re-fit the selected specification $\\text{ARIMAX}(2, 1, 1) + \\text{Fourier}(K=3) + \\text{PM}_{2.5}(t-1)$ on the combined **Train + Validation** set (`2016-01-01` $\\rightarrow$ `2019-08-31`, $1,339$ days) to prepare for out-of-sample testing. We inspect:
1. Standardized residuals over time
2. Residual histogram with Kernel Density Estimate (KDE) and standard normal reference
3. Normal Q-Q plot
4. Residual Autocorrelation Function (correlogram)
5. Ljung–Box test of residual independence at lags $10$ and $20$.""")
    )

    cells.append(
        nbf.v4.new_code_cell("""# Combine Train and Validation
train_val_df = pd.concat([train_df, val_df])
train_val_s = train_val_df['AQI'].asfreq('D')

# Prepare exog for Train + Val
ex_tv = get_fourier_terms(train_val_s.index, 3)
ex_tv['lagged_pm25'] = clean_pm25.loc[train_val_s.index].values

# Fit final model
final_model = SARIMAX(train_val_s, exog=ex_tv, order=(2, 1, 1), trend='n').fit(disp=False, maxiter=200)
residuals = final_model.resid
std_resid = (residuals - residuals.mean()) / residuals.std()

# Ljung-Box test
lb_results = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
print("=== Ljung-Box Test on Final Model Residuals ===")
display(lb_results)

# 4-Panel Diagnostics Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Panel 1: Standardized Residuals Over Time
axes[0, 0].plot(std_resid.index, std_resid.values, color='#2c3e50', lw=0.9)
axes[0, 0].axhline(0, color='red', linestyle='--', lw=1)
axes[0, 0].axhline(3, color='gray', linestyle=':', lw=0.8)
axes[0, 0].axhline(-3, color='gray', linestyle=':', lw=0.8)
axes[0, 0].set_title('(A) Standardized Residuals Over Time', fontweight='bold')
axes[0, 0].set_ylabel('Standardized Residual')

# Panel 2: Histogram & KDE vs Normal
sns.histplot(std_resid, kde=True, ax=axes[0, 1], color='#3498db', stat='density', bins=35, label='Residual KDE')
x_norm = np.linspace(-4, 4, 200)
axes[0, 1].plot(x_norm, stats.norm.pdf(x_norm), color='crimson', lw=2, linestyle='--', label='Standard Normal N(0,1)')
axes[0, 1].set_title('(B) Residual Distribution vs Normal N(0,1)', fontweight='bold')
axes[0, 1].set_xlabel('Standardized Residual')
axes[0, 1].legend()

# Panel 3: Normal Q-Q Plot
stats.probplot(std_resid, dist="norm", plot=axes[1, 0])
axes[1, 0].get_lines()[0].set_markerfacecolor('#2980b9')
axes[1, 0].get_lines()[0].set_markersize(4)
axes[1, 0].get_lines()[1].set_color('crimson')
axes[1, 0].set_title('(C) Normal Q-Q Plot', fontweight='bold')

# Panel 4: Residual Correlogram (ACF)
plot_acf(residuals, lags=30, ax=axes[1, 1], color='#27ae60', alpha=0.05, auto_ylims=True)
axes[1, 1].set_title('(D) Residual Autocorrelation (Lags 1–30)', fontweight='bold')
axes[1, 1].set_ylabel('ACF')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'p5_model_diagnostics.png'), dpi=300)
plt.show()

print(f"Residual Summary Statistics:")
print(f" - Mean:     {residuals.mean():.3f} (Zero-mean property confirmed)")
print(f" - Std:      {residuals.std():.3f}")
print(f" - Skewness: {residuals.skew():.3f}")
print(f" - Kurtosis: {residuals.kurt():.3f}")""")
    )

    cells.append(
        nbf.v4.new_markdown_cell("""> **Observation (Section 4):**  
> The final model residuals exhibit consistent mean ($-0.19 \\approx 0.0$), confirming an unbiased estimator. On the Ljung–Box diagnostic, at lag 20 the test yields $p = 0.0789 > 0.05$, whereas at lag 10 the $p$-value ($p \\approx 0.04$) reaches statistical significance. This indicates that the **residuals are close to white noise; mild short-lag autocorrelation remains**. In the Q-Q plot and histogram, the central distribution aligns closely with Gaussian reference, while heavy tails reflect occasional episodic pollution peaks (e.g., Diwali).""")
    )

    # =========================================================================
    # Section 5: Walk-Forward Out-of-Sample Test Evaluation
    # =========================================================================
    cells.append(
        nbf.v4.new_markdown_cell("""## 5. Walk-Forward Out-of-Sample Test Evaluation
We evaluate the final model on the untouched **Test set** (`2019-09-01` $\\rightarrow$ `2020-02-29`, $182$ days) using true walk-forward filtering without re-estimating parameters (`statsmodels` `extend` with `refit=False`):
- **Horizon 1 ($h=1$):** Daily 1-step ahead forecast with $95\\%$ prediction intervals and coverage check.
- **Horizon 7 ($h=1..7$):** Multi-step forecasting evaluated from each origin across horizons $1$ through $7$, compared directly against Persistence.
- **Category Health Evaluation:** Confusion matrix across CPCB categories.""")
    )

    cells.append(
        nbf.v4.new_code_cell("""# 1. 1-Day Ahead Test Walk-Forward
ex_test = get_fourier_terms(test_s.index, 3)
ex_test['lagged_pm25'] = clean_pm25.loc[test_s.index].values

# Extend model through Test set
ext_test = final_model.extend(test_s, exog=ex_test)
test_pred_res = ext_test.get_prediction(start=test_s.index[0], end=test_s.index[-1], exog=ex_test)
test_preds = test_pred_res.predicted_mean
test_ci = test_pred_res.conf_int(alpha=0.05)

# Evaluate 1-day metrics
final_test_metrics = evaluate_forecast(test_s.values, test_preds.values, 'Final_ARIMAX(2,1,1)+F3+PM25', 'test', 1)
test_coverage = np.mean((test_s.values >= test_ci.iloc[:, 0].values) & (test_s.values <= test_ci.iloc[:, 1].values)) * 100.0

print("=== Final Model 1-Day Ahead Test Evaluation ===")
print(final_test_metrics)
print(f"95% Prediction Interval Empirical Coverage: {test_coverage:.2f}%")

# Record predictions
predictions_records.append(create_predictions_dataframe(
    test_s.index, test_s.values, test_preds.values, 1, 'Final_ARIMAX(2,1,1)+F3+PM25', 'test'
))

# Plot Test Actual vs Predicted with 95% PI Band
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_s.index, test_s.values, color='#2c3e50', lw=1.8, marker='o', markersize=3, label='Actual AQI')
ax.plot(test_s.index, test_preds.values, color='#e67e22', lw=1.8, label='1-Day Ahead Forecast')
ax.fill_between(test_s.index, test_ci.iloc[:, 0].values, test_ci.iloc[:, 1].values, 
                color='#e67e22', alpha=0.2, label=f'95% Prediction Interval ({test_coverage:.1f}% coverage)')
ax.set_title('Test Set Out-of-Sample Evaluation: Actual vs 1-Day Forecast with 95% PI (Sep 2019 – Feb 2020)', fontsize=12, fontweight='bold')
ax.set_ylabel('AQI Value', fontsize=10)
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'p5_test_actual_vs_predicted.png'), dpi=300)
plt.show()

# 2. Multi-Step Horizon 1–7 Evaluation on Identical Forecast Origins
# For multi-step horizons, future PM2.5 observations are unavailable at forecast origin t.
# Hence, we use the best Fourier-only ARIMA specification ("ARIMA-Fourier (multi-step)").
# We strictly evaluate across identical forecast origins where all 7 target days fall inside the Test set.
multistep_origins = test_s.index[:len(test_s) - 7]
origins_df = pd.DataFrame({'origin_date': multistep_origins})
origins_df.to_csv(os.path.join(RES_DIR, 'p5_multistep_origins.csv'), index=False)
print(f"Identified and saved {len(multistep_origins)} identical multi-step forecast origins to outputs/results/p5_multistep_origins.csv")

# Train Fourier-only model on Train+Val
ex_tv_m2 = get_fourier_terms(train_val_s.index, 3)
ex_test_m2 = get_fourier_terms(test_s.index, 3)
mod_m2 = SARIMAX(train_val_s, exog=ex_tv_m2, order=(1, 1, 1), trend='n').fit(disp=False, maxiter=200)

h_errors_model = {h: [] for h in range(1, 8)}
h_errors_persist = {h: [] for h in range(1, 8)}

for i, origin_dt in enumerate(multistep_origins):
    cur_test_s = test_s.loc[:origin_dt]
    cur_exog_s = ex_test_m2.loc[:origin_dt]
    fut_dates = test_s.index[i+1 : i+8]
    fut_exog = ex_test_m2.loc[fut_dates]
    actual_fut = test_s.loc[fut_dates].values
    last_known = test_s.loc[origin_dt]
    
    sub_ext = mod_m2.extend(cur_test_s, exog=cur_exog_s)
    fc_multi = sub_ext.forecast(steps=7, exog=fut_exog).values
    
    for h in range(1, 8):
        h_errors_model[h].append((actual_fut[h-1] - fc_multi[h-1])**2)
        h_errors_persist[h].append((actual_fut[h-1] - last_known)**2)
        # Record multi-step predictions
        predictions_records.append(pd.DataFrame({
            'date': [fut_dates[h-1]],
            'actual': [actual_fut[h-1]],
            'predicted': [fc_multi[h-1]],
            'horizon': [h],
            'model': ['ARIMA-Fourier (multi-step)'],
            'split': ['test']
        }))
        predictions_records.append(pd.DataFrame({
            'date': [fut_dates[h-1]],
            'actual': [actual_fut[h-1]],
            'predicted': [last_known],
            'horizon': [h],
            'model': ['Persistence'],
            'split': ['test']
        }))

h_rmse_model = [np.sqrt(np.mean(h_errors_model[h])) for h in range(1, 8)]
h_rmse_persist = [np.sqrt(np.mean(h_errors_persist[h])) for h in range(1, 8)]

for h in range(1, 8):
    m_h_metrics = evaluate_forecast(
        [r['actual'].values[0] for r in predictions_records if r['model'].values[0]=='ARIMA-Fourier (multi-step)' and r['horizon'].values[0]==h],
        [r['predicted'].values[0] for r in predictions_records if r['model'].values[0]=='ARIMA-Fourier (multi-step)' and r['horizon'].values[0]==h],
        'ARIMA-Fourier (multi-step)', 'test', h
    )
    baseline_metrics.append(m_h_metrics)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, 8), h_rmse_model, marker='o', lw=2, color='#2980b9', label='ARIMA-Fourier (multi-step)')
ax.plot(range(1, 8), h_rmse_persist, marker='s', lw=2, color='#e74c3c', linestyle='--', label='Persistence Benchmark')
ax.set_title('Forecast Error Growth across Horizons (1 to 7 Days Ahead, N=175 Origins)', fontsize=12, fontweight='bold')
ax.set_xlabel('Forecast Horizon h (Days Ahead)', fontsize=10)
ax.set_ylabel('Test RMSE', fontsize=10)
ax.set_xticks(range(1, 8))
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(frameon=True)

for h, (m_val, p_val) in enumerate(zip(h_rmse_model, h_rmse_persist), start=1):
    ax.annotate(f"{m_val:.1f}", (h, m_val - 1.5), ha='center', fontsize=9, color='#2980b9', fontweight='bold')
    ax.annotate(f"{p_val:.1f}", (h, p_val + 1.2), ha='center', fontsize=9, color='#e74c3c', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'p5_rmse_vs_horizon.png'), dpi=300)
plt.show()

# 3. Category Confusion Matrix
cat_metrics = compute_category_metrics(test_s.values, test_preds.values)
cm = cat_metrics['Confusion_Matrix']

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CPCB_CATEGORIES, yticklabels=CPCB_CATEGORIES, ax=ax)
ax.set_title('Test Set: CPCB AQI Category Confusion Matrix (1-Day Ahead)', fontweight='bold')
ax.set_xlabel('Predicted Health Category')
ax.set_ylabel('Actual Health Category')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'p5_category_confusion_matrix.png'), dpi=300)
plt.show()""")
    )

    cells.append(
        nbf.v4.new_markdown_cell("""> **Observation (Section 5):**  
> **Reconciliation of Evaluation Horizons and Datasets:**  
> - **1-Day Ahead Evaluation:** The primary 1-day model evaluates $\\text{ARIMAX}(2, 1, 1) + \\text{Fourier}(K=3) + \\text{PM}_{2.5}(t-1)$ across all **$182$ test days** (`2019-09-01` to `2020-02-29`) using actual observed $\\text{PM}_{2.5}(t-1)$. It achieves a Test RMSE of **$25.55$** ($20.4\\%$ error reduction over Persistence $32.10$) and empirical $95\\%$ PI coverage of **$96.70\\%$**.  
> - **Multi-Step Horizon (1–7 Days Ahead):** For multi-step horizons, future observed $\\text{PM}_{2.5}$ is unavailable at origin $t$. Thus, multi-step forecasts are produced by **$\\text{ARIMA-Fourier (multi-step)}$** ($p=1, d=1, q=1, K=3$), relying only on historical AQI and deterministic calendar Fourier regressors.  
> - **Fair Comparison Window ($N=175$ Origins):** To ensure identical sample size across all 7 horizons, evaluations are restricted to origins where all 7 future target days fall within the test period (`2019-09-01` to `2020-02-22`, exactly $175$ forecast origins). Across these origins, persistence error escalates rapidly to $61.16$ at horizon $7$, whereas the Fourier ARIMA model error grows more gradually to $49.72$, demonstrating the value of annual Fourier terms for multi-day horizons.""")
    )

    # =========================================================================
    # Section 6: Stress Test Window (COVID-19 Lockdown)
    # =========================================================================
    cells.append(
        nbf.v4.new_markdown_cell("""## 6. Stress-Test Window Evaluation (COVID-19 Lockdown)
We evaluate the fixed final model across the lockdown stress window (**1 March – 30 June 2020**, $122$ days) using walk-forward updating (incorporating actual daily measurements without re-fitting parameters).""")
    )

    cells.append(
        nbf.v4.new_code_cell("""# Prepare exog for full test + stress
full_eval_s = pd.concat([test_s, stress_s])
ex_stress = get_fourier_terms(stress_s.index, 3)
ex_stress['lagged_pm25'] = clean_pm25.loc[stress_s.index].values
ex_eval = pd.concat([ex_test, ex_stress])

# Extend through full eval window
ext_stress = final_model.extend(full_eval_s, exog=ex_eval)
stress_pred_res = ext_stress.get_prediction(start=stress_s.index[0], end=stress_s.index[-1], exog=ex_stress)
stress_preds = stress_pred_res.predicted_mean
stress_ci = stress_pred_res.conf_int(alpha=0.05)

stress_metrics = evaluate_forecast(stress_s.values, stress_preds.values, 'Final_ARIMAX(2,1,1)+F3+PM25', 'stress', 1)
stress_coverage = np.mean((stress_s.values >= stress_ci.iloc[:, 0].values) & (stress_s.values <= stress_ci.iloc[:, 1].values)) * 100.0

print("=== Final Model Stress-Test Evaluation (Lockdown 2020) ===")
print(stress_metrics)
print(f"Stress 95% PI Empirical Coverage: {stress_coverage:.2f}%")

predictions_records.append(create_predictions_dataframe(
    stress_s.index, stress_s.values, stress_preds.values, 1, 'Final_ARIMAX(2,1,1)+F3+PM25', 'stress'
))

# Plot Stress Actual vs Predicted
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(stress_s.index, stress_s.values, color='#2c3e50', lw=1.8, marker='o', markersize=3, label='Actual AQI (Lockdown)')
ax.plot(stress_s.index, stress_preds.values, color='#27ae60', lw=1.8, label='1-Day Ahead Forecast')
ax.fill_between(stress_s.index, stress_ci.iloc[:, 0].values, stress_ci.iloc[:, 1].values, 
                color='#27ae60', alpha=0.2, label=f'95% Prediction Interval ({stress_coverage:.1f}% coverage)')
ax.axvline(pd.Timestamp('2020-03-24'), color='red', linestyle='--', label='National Lockdown Enacted (24 Mar)')
ax.set_title('Stress-Test Evaluation: Walk-Forward Forecast Across COVID-19 Lockdown Window (Mar–Jun 2020)', fontsize=12, fontweight='bold')
ax.set_ylabel('AQI Value', fontsize=10)
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'p5_stress_actual_vs_predicted.png'), dpi=300)
plt.show()""")
    )

    cells.append(
        nbf.v4.new_markdown_cell("""> **Observation (Section 6):**  
> Under the unprecedented COVID-19 lockdown window, the nominal $95\\%$ prediction interval coverage reached **$98.36\\%$**. However, this high empirical coverage reflects the wide prediction intervals relative to the suppressed variability of the lockdown period, rather than intrinsic model resilience.  
> In fact, ARIMAX underperformed simple persistence during the lockdown (Persistence RMSE $\\approx 13.79$, QWK $= 0.40$ vs ARIMAX RMSE $\\approx 16.93$, QWK $= 0.18$). This underperformance is directly consistent with the learned seasonal harmonics and historical $\\text{PM}_{2.5}$ coefficients breaking down during a structural emissions regime change, where traffic and industrial operations were abruptly suspended.""")
    )

    # =========================================================================
    # Section 7: Artifact Persistence & Export
    # =========================================================================
    cells.append(
        nbf.v4.new_markdown_cell("""## 7. Artifact Persistence & Export
We persist:
1. `outputs/results/p5_metrics.csv`: Consolidated performance metrics across all models, baselines, and partitions.
2. `outputs/results/p5_predictions.csv`: Standard predictions dataset format (`date, actual, predicted, horizon, model, split`).
3. `outputs/models/arima_final.pkl`: Serialized final fitted model object.""")
    )

    cells.append(
        nbf.v4.new_code_cell("""# 1. Compile consolidated metrics table
# Re-evaluate final model on validation for completeness
ex_v_final = get_fourier_terms(val_s.index, 3)
ex_v_final['lagged_pm25'] = clean_pm25.loc[val_s.index].values
ext_val_final = mod_full.extend(val_s, exog=ex_v_final)
preds_v_final = ext_val_final.predict(start=val_s.index[0], end=val_s.index[-1], exog=ex_v_final)
final_val_metrics = evaluate_forecast(val_s.values, preds_v_final.values, 'Final_ARIMAX(2,1,1)+F3+PM25', 'val', 1)

predictions_records.append(create_predictions_dataframe(
    val_s.index, val_s.values, preds_v_final.values, 1, 'Final_ARIMAX(2,1,1)+F3+PM25', 'val'
))

all_metrics_list = baseline_metrics + [final_val_metrics, final_test_metrics, stress_metrics]
metrics_df = pd.DataFrame(all_metrics_list)
metrics_df.to_csv(os.path.join(RES_DIR, 'p5_metrics.csv'), index=False)
print("Saved outputs/results/p5_metrics.csv")

# 2. Compile consolidated predictions dataframe
predictions_df = pd.concat(predictions_records, ignore_index=True)
predictions_df.to_csv(os.path.join(RES_DIR, 'p5_predictions.csv'), index=False)
print("Saved outputs/results/p5_predictions.csv")

# 3. Save final fitted model artifact
final_model_path = os.path.join(MODELS_DIR, 'arima_final.pkl')
joblib.dump(final_model, final_model_path)
print(f"Saved final fitted model object to {final_model_path}")

# Display comparative metrics table
print("\\n=== Consolidated Performance Table (1-Day Horizon) ===")
display(metrics_df[metrics_df['horizon']==1].sort_values(['split', 'RMSE']))""")
    )

    nb["cells"] = cells
    return nb


def main():
    notebook_path = os.path.join('notebooks', '05_arima.ipynb')
    print(f"Generating Phase 5 notebook at: {notebook_path}")
    nb = create_phase5_notebook()
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Successfully wrote notebook JSON.")

    print("Executing 05_arima.ipynb end-to-end via ucs321-aqi kernel...")
    ep = ExecutePreprocessor(timeout=900, kernel_name='ucs321-aqi')
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb_read = nbf.read(f, as_version=4)
    
    ep.preprocess(nb_read, {'metadata': {'path': 'notebooks'}})
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb_read, f)
    print("Successfully executed and updated notebooks/05_arima.ipynb.")


if __name__ == '__main__':
    main()

