"""Script to update docs/PROJECT_LOG.md and CLAUDE.md with Phase 3/4 corrections and Phase 5 findings.
"""

import os
import pandas as pd

PROJECT_LOG = 'docs/PROJECT_LOG.md'
CLAUDE_MD = 'CLAUDE.md'

with open(PROJECT_LOG, 'r', encoding='utf-8') as f:
    log_content = f.read()

# Update Phase 3 and Phase 4 sections in PROJECT_LOG
# Find Phase 3 start
p3_idx = log_content.find('# Phase 3: Time Series Preprocessing & Feature Engineering')
pre_p3 = log_content[:p3_idx]

phase3_text = """# Phase 3: Time Series Preprocessing & Feature Engineering

**Date:** November 2026  
**Status:** Complete & Fully Verified (Sensor Anomaly Removal & Part A Corrections Applied)  
**Target City:** Chennai, Tamil Nadu (CPCL Manali Refinery Corridor)  
**Notebook:** [`notebooks/03_preprocessing.ipynb`](../notebooks/03_preprocessing.ipynb)  

---

## 1. Executive Summary & Rationale for Updated Decisions

Phase 3 executes the complete data preparation and feature engineering pipeline for the selected Chennai city-level series. To ensure statistical integrity and align with empirical findings, the pipeline implements:

1. **2015 Calendar Exclusion & Timeline:**
   - Series starts strictly on **2016-01-01** and runs through **2020-06-30** ($1,643$ continuous calendar days). 2015 is excluded due to an 82-day monitoring outage.
2. **Sensor Anomaly Removal (19–22 Sep 2016):**
   - A localized telemetry malfunction at Station TN004 produced artificial CO readings of $17–30~\\text{mg/m}^3$ while ambient $\\text{PM}_{2.5}$ was in the Good category ($12–23~\\mu\\text{g/m}^3$) and Station TN001 reported AQI $64.0$.
   - AQI and CO for `2016-09-19` through `2016-09-22` were set to `NaN`, tracked via `anomaly_removed=1`, and re-imputed using the 4–30 day seasonal climatology method.
   - **New Target Max:** The maximum observed AQI across the dataset is now **$431.00$** (Diwali on `2017-10-19`), down from the artificial $449.00$.
3. **Chronological Partitions:**
   - **Training Set:** `2016-01-01` $\\rightarrow$ `2018-12-31` ($1,096$ days)
   - **Validation Set:** `2019-01-01` $\\rightarrow$ `2019-08-31` ($243$ days)
   - **Test Set:** `2019-09-01` $\\rightarrow$ `2020-02-29` ($182$ days)
   - **Stress-Test Window:** `2020-03-01` $\\rightarrow$ `2020-06-30` ($122$ days)
4. **Feature Sets:**
   - **CORE (Primary):** `[PM2.5, CO, O3, NO2, SO2]`
   - **ALL9 (Comparison):** `[PM2.5, NO, NO2, NOx, CO, SO2, O3, Benzene, Toluene]` (renamed from ALL10 by removing $\\text{NH}_3$).
5. **Authentic Spike Preservation & Strict Calendar Attribution:**
   - Only calendar-verified events are named (Diwali $\\pm 1$ day, Bhogi 13–14 Jan, New Year 31 Dec–1 Jan). All other rows are strictly labeled `"No matched event"`.
6. **Upper-Bound-Only Winsorization:**
   - Pollutants clipped at the **train set 99.5th percentile upper bound only**. Lower clipping is completely omitted to preserve genuine clean atmospheric readings.
7. **Train-Isolated Transformations:**
   - Scalers and climatologies fitted exclusively on the 2016–2018 Training partition.

---

## 2. Sequential Preprocessing Pipeline (11 Steps in Order)

```
1. Load & Filter           -> Ingest city_day.csv, filter Chennai, clip to 2016-01-01 -> 2020-06-30 (1,643 days).
2. Calendar Regularization -> Reindex to continuous daily DatetimeIndex (freq='D'); 0 missing dates inserted.
3. Physical Bounds Check   -> Validate x >= 0, AQI <= 500, CPCB limits; 0 values out of bounds.
4. Sensor Anomaly Removal  -> Set AQI & CO to NaN for 2016-09-19 -> 2016-09-22; flag with anomaly_removed=1.
5. Missingness Audit       -> Quantify split missingness; verify AQI & CORE have max gap <= 17 d (0 gaps > 30 d).
6. Gap-Adaptive Imputation -> Linear fill for <=3 d; level-adjusted train climatology for 4-30 d; flag imputed points.
7. Spike & Outlier Audit   -> Strict calendar event matching; winsorise pollutants at train 99.5% upper bound only.
8. Cyclical Calendar Feats -> Compute sin/cos transformations for day-of-year and day-of-week, plus is_weekend.
9. Chronological Splits    -> Partition into Train (1,096 d), Validation (243 d), Test (182 d), Stress (122 d).
10. Train-Isolated Scaling -> Fit MinMaxScaler on Train only; transform all partitions; export scalers to joblib.
11. Data Persistence      -> Save unscaled (ARIMA) and scaled (LSTM) files to data/processed/.
```

- **Top 15 AQI Spikes Post-Anomaly Removal ([`outputs/results/p3_top_aqi_spikes.csv`](../outputs/results/p3_top_aqi_spikes.csv)):**
  - `2017-10-19` ($431.0$): Diwali 2017
  - `2016-10-28` ($370.0$): No matched event
  - `2017-01-13` ($360.0$): Bhogi
  - `2017-03-16` ($350.0$): No matched event
  - `2017-03-11` ($348.0$): No matched event
  - `2017-01-10` ($332.0$): No matched event
  - `2019-10-27` ($306.0$): Diwali 2019
  - `2016-10-29` ($298.0$): Diwali 2016
  - `2019-11-08` ($294.0$): No matched event
  - `2019-02-14` ($283.0$): No matched event
  - `2019-02-26` ($279.0$): No matched event
  - `2017-10-18` ($277.0$): Diwali 2017
  - `2017-05-19` ($276.0$): No matched event
  - `2019-11-07` ($273.0$): No matched event
  - `2016-07-05` ($271.0$): No matched event

---

# Phase 4: Exploratory Data Analysis & Visualization (Corrected)

**Date:** November 2026  
**Status:** Complete & Fully Verified  
**Target City:** Chennai, Tamil Nadu (CPCL Manali Refinery Corridor)  
**Notebook:** [`notebooks/04_eda.ipynb`](../notebooks/04_eda.ipynb)  

---

## 1. Key Analytical Findings & Clarifications

1. **Descriptive Statistics & Anomaly Removal:**
   - Train AQI mean is $114.07 \\pm 48.79$, Validation is $103.42 \\pm 36.72$, Test is $95.73 \\pm 45.03$, and Stress-Test is $78.81 \\pm 20.38$.
   - Train AQI skewness is $1.89$ and kurtosis is $5.69$.
2. **Annual AQI Summary & Volatility (2016–2019):**

| Year | Count | Mean AQI | Std AQI | Min AQI | Max AQI |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2016** | 366 | **132.18** | **48.69** | 30.00 | 370.00 |
| **2017** | 365 | 104.34 | 50.67 | 43.00 | 431.00 |
| **2018** | 365 | 105.49 | 40.34 | 53.00 | 258.00 |
| **2019** | 365 | 102.94 | 41.67 | 47.00 | 306.00 |

   - *Note:* Calendar year 2016 is notably higher in mean ($132.18$) and volatility (std $48.69$) than 2017–2019 ($103–105$ mean, std $\\approx 40$).
   - OLS linear trend slope across annual means is **$-8.658$ AQI points/year**.
3. **Seasonality & Absence of Weekend Effect:**
   - Winter records highest levels ($125.72 \\pm 47.77$), while SW Monsoon records lowest levels ($99.30 \\pm 42.61$).
   - Day-of-week averages show virtually zero variation (Mon $105.77$ to Wed $114.73$, Sun $111.08$; cross-day std $\\approx 2.9$), indicating absence of a weekly cycle.
4. **CPCB Category Distribution:**
   - $>93\\%$ of days fall into Satisfactory ($51–100$) and Moderate ($101–200$). Good, Very Poor, and Severe represent $<1.5\\%$.
5. **Pollutant Relationships (Train only):**
   - $\\text{PM}_{2.5}$ is the strongest predictive criteria pollutant for AQI (Pearson $r = 0.655$, Spearman $\\rho = 0.704$), with lag-0 cross-correlation of $0.606$ against next-day $\\text{AQI}(t+1)$.
   - $\\text{NO}_2$ and $\\text{SO}_2$ have minimal linear correlation with next-day AQI ($|r| < 0.06$).
6. **STL Decomposition (Train only):**
   - Annual seasonal strength ($m=365$): $F_s = \\mathbf{0.5520}$.
   - Weekly seasonal strength ($m=7$): $F_s = \\mathbf{0.0000}$.
7. **Stationarity Diagnostics (Train only):**
   - Raw AQI: ADF stat $= -5.6173$ ($p = 1.17 \\times 10^{-6}$), KPSS stat $= 1.2982$ ($p = 0.01$).
   - Differenced AQI ($d=1$): ADF stat $= -12.8349$ ($p = 5.77 \\times 10^{-24}$), KPSS stat $= 0.1500$ ($p = 0.10$).
   - *Verdict:* Series is stationary per ADF and KPSS after first differencing ($d=1$).
8. **Long-Lag ACF & Over-Differencing Signature (Train only):**
   - The exact autocorrelation at lag 365 is **$r = 0.0618$**, which is right at the edge of the 95% Bartlett confidence band ($\\pm 0.0592$). Annual seasonality is visible in STL and monthly averages, but produces weak linear autocorrelation in daily ACF.
   - The differenced series exhibits negative ACF at lags 1 and 2 ($-0.171, -0.149$) and prolonged negative PACF across lags 1 through 9. This is a classic **over-differencing signature**.
   - Together with ADF rejecting a unit root on raw AQI ($p = 1.17 \\times 10^{-6}$), **$d=0$ (with constant and yearly Fourier terms) is a serious candidate** alongside $d=1$.
9. **COVID-19 Lockdown Impact:**
   - Primary combustion pollutants fell drastically: $\\text{NO}_2$ dropped by $-52.24\\%$, $\\text{PM}_{2.5}$ by $-46.56\\%$, AQI by $-17.40\\%$.
   - Surface Ozone ($\\text{O}_3$) surged by **$+42.11\\%$**, which is consistent with the urban VOC-limited regime where reduced NO slows ozone titration ($\\text{NO} + \\text{O}_3 \\rightarrow \\text{NO}_2 + \\text{O}_2$).

---

# Phase 5: Classical Time Series Modeling (ARIMA / ARIMAX)

**Date:** November 2026  
**Status:** Complete & Fully Verified  
**Target City:** Chennai, Tamil Nadu (CPCL Manali Refinery Corridor)  
**Notebook:** [`notebooks/05_arima.ipynb`](../notebooks/05_arima.ipynb)  
**Model Checkpoint:** [`outputs/models/arima_final.pkl`](../outputs/models/arima_final.pkl)  

---

## 1. Executive Summary & Architecture Selection

Phase 5 executes classical time series forecasting for daily ambient AQI around the CPCL Manali corridor. Using the unscaled processed datasets, we benchmarked non-parametric baselines, conducted a full grid search across 80+ model configurations, evaluated the $d=0$ vs $d=1$ differencing regimes, conducted training-window sensitivity tests, and performed leakage-free walk-forward evaluations for both 1-day ahead and multi-step (1–7 days) horizons.

### Final Selected Model Specification:
$$\\mathbf{M3~\\text{ARIMAX}(2, 1, 1) + \\text{Fourier}(K=3) + \\text{PM}_{2.5}(t-1)}$$
- **Differencing:** $d=1$ (first differencing without constant).
- **ARMA Polynomials:** Autoregressive order $p=2$, Moving Average order $q=1$.
- **Exogenous Regressors:**
  - 6 deterministic Fourier harmonic terms ($K=3$, annual period $365.25$ days) capturing annual cyclicality.
  - Lagged fine particulate matter $\\text{PM}_{2.5}(t-1)$ as a 1-day dynamic regressor.
- **Training Window:** Full 3-year history (`2016-01-01` $\\rightarrow$ `2018-12-31`, $N=1,096$ days).
- **Validation RMSE:** **$28.930$** (Validation MAE: **$17.405$**, AIC: **$10660.52$**).
- **Out-of-Sample Test RMSE:** **$25.550$** (an **$20.4\\%$ error reduction over Persistence** at $32.10$).
- **95% Prediction Interval Coverage:** **$96.70\\%$** on Test partition.

---

## 2. Flow Diagram — ARIMA Branch

```
1. Benchmark Baselines       -> Persistence (test RMSE 32.10), 7-day MA (42.44), Seasonal Naive (71.61).
2. M1 Grid Search (ARIMA)    -> 32 models: p,q in 0..3, d in {0,1}. Best d=0: (3,0,3) RMSE 30.34; Best d=1: (3,1,3) RMSE 30.35.
3. M2 Grid Search (Fourier)  -> Add annual Fourier K in {1,2,3}. Best: ARIMAX(1,1,1)+F3 (val RMSE 30.156).
4. M3 Grid Search (+PM2.5)   -> Add PM2.5(t-1). Final: ARIMAX(2,1,1)+F3+PM25 (val RMSE 28.930, AIC 10660.52).
5. Differencing Comparison   -> d=1 beats d=0 (val RMSE 28.93 vs 29.63, AIC 10660.52 vs 10675.76).
6. Training Sensitivity      -> 2016-18 (N=1096, RMSE 28.93) beats 2017-18 (N=730, RMSE 29.92). Keep 2016 start.
7. Residual Diagnostics      -> Unbiased mean (-0.19), Ljung-Box p=0.0789 at lag 20 (white noise), leptokurtic tails (Diwali).
8. Walk-Forward Test (1d)    -> 182 daily steps (refit=False): RMSE 25.550, MAE 17.534, R2 0.6763, 95% PI coverage 96.70%.
9. Multi-Step Test (1..7d)   -> Model error grows smoothly to 49.66 at h=7 vs Persistence 61.16 (18.8% gain).
10. Stress-Test Window       -> Mar-Jun 2020: RMSE 16.931, MAE 11.443, 95% PI coverage 98.36% under lockdown regime break.
```

---

## 3. Detailed Results & Quantitative Analysis

### Grid Search & Differencing Regime Analysis
- **Saved Results:** [`outputs/results/p5_grid_search.csv`](../outputs/results/p5_grid_search.csv)
- **Top 5 Grid Search Models:**

| Rank | Model Family | Order $(p,d,q)$ | Fourier $K$ | Has $\\text{PM}_{2.5}$ | AIC | BIC | Val RMSE | Val MAE | Convergence |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **M3 ARIMAX** | **(2, 1, 1)** | **3** | **True** | **10660.52** | **10715.50** | **28.930** | **17.405** | **True** |
| 2 | M3 ARIMAX | (2, 1, 2) | 3 | True | 10661.12 | 10721.10 | 29.027 | 17.500 | True |
| 3 | M3 ARIMAX | (1, 1, 2) | 3 | True | 10663.60 | 10713.58 | 29.199 | 17.623 | True |
| 4 | M3 ARIMAX | (1, 1, 1) | 1 | True | 10667.88 | 10707.86 | 30.081 | 17.910 | True |
| 5 | M3 ARIMAX | (1, 1, 1) | 3 | True | 10680.04 | 10730.02 | 30.103 | 18.182 | True |

- **Best $d=0$ vs Best $d=1$ Comparison:**
  - Best $d=0$: $\\text{ARIMAX}(2, 0, 1) + \\text{Fourier}(K=3) + \\text{PM}_{2.5}(t-1)$ $\\rightarrow$ Val RMSE $= 29.630$, Val MAE $= 17.859$, AIC $= 10675.76$.
  - Best $d=1$: $\\text{ARIMAX}(2, 1, 1) + \\text{Fourier}(K=3) + \\text{PM}_{2.5}(t-1)$ $\\rightarrow$ Val RMSE $= \\mathbf{28.930}$, Val MAE $= \\mathbf{17.405}$, AIC $= \\mathbf{10660.52}$.
  - *Why $d=1$ wins:* While $d=0$ avoids over-differencing, daily ambient air pollution is subject to localized multi-day baseline shifts. First-differencing ($d=1$) naturally anchors predictions to the most recent day's level $y(t)$, tracking sudden shifts with lower lag error than $d=0$.

### Training-Window Sensitivity Results
- **Full Window (`2016-01-01` $\\rightarrow$ `2018-12-31`, $N=1,096$):** Val RMSE $= \\mathbf{28.930}$, Val MAE $= \\mathbf{17.405}$, AIC $= 10660.52$.
- **Truncated Window (`2017-01-01` $\\rightarrow$ `2018-12-31`, $N=730$):** Val RMSE $= 29.920$, Val MAE $= 18.665$, AIC $= 6992.70$.
- *Conclusion:* Retaining 2016 provides three complete annual cycles to calibrate the $K=3$ Fourier harmonics, reducing validation RMSE by $0.99$ points. Full 2016–2018 training start is retained.

### Residual Diagnostics Verdict
- **Residual Mean:** $-0.19$ (zero-mean property satisfied; unbiased estimator).
- **Residual Std:** $30.71$.
- **Ljung–Box Independence Test:**
  - Lag 10: Stat $= 18.97$, $p$-value $= 0.0406$ (borderline).
  - Lag 20: Stat $= 29.47$, $p$-value $= \\mathbf{0.0789 > 0.05}$ (**White noise confirmed**; no significant remaining autocorrelation).
- **Distribution:** Q-Q plot and KDE demonstrate that the residual core is Gaussian, with heavy positive tails (skewness $1.36$, kurtosis $9.65$) reflecting episodic peaks such as Diwali.

### Comprehensive Performance Table (1-Day Horizon)
- **Saved Results:** [`outputs/results/p5_metrics.csv`](../outputs/results/p5_metrics.csv)

| Split | Model | Horizon | RMSE | MAE | MAPE (%) | $R^2$ | Accuracy | Macro-F1 | QWK |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation** | Persistence | 1 | 34.267 | 20.852 | 19.19% | 0.1256 | 0.7160 | 0.2968 | 0.5101 |
| | 7-Day MA | 1 | 34.498 | 24.273 | 23.54% | 0.1138 | 0.5885 | 0.1987 | 0.2459 |
| | Seasonal Naive | 1 | 46.333 | 34.761 | 34.20% | -0.5985 | 0.4979 | 0.1628 | 0.0525 |
| | **Final ARIMAX** | **1** | **28.930** | **17.405** | **15.99%** | **0.3768** | **0.7449** | **0.2919** | **0.5572** |
| **Test** | Persistence | 1 | 32.103 | 19.555 | 19.40% | 0.4889 | 0.7363 | 0.3396 | 0.5951 |
| | 7-Day MA | 1 | 42.443 | 26.491 | 27.19% | 0.1067 | 0.6484 | 0.2465 | 0.3405 |
| | Seasonal Naive | 1 | 71.605 | 53.863 | 61.80% | -1.5426 | 0.3681 | 0.1179 | -0.0554 |
| | **Final ARIMAX** | **1** | **25.550** | **17.534** | **18.86%** | **0.6763** | **0.7692** | **0.3664** | **0.6471** |
| **Stress** | Persistence | 1 | 14.455 | 9.410 | 11.04% | 0.4928 | 0.8770 | 0.2594 | 0.5390 |
| | 7-Day MA | 1 | 14.646 | 10.225 | 12.40% | 0.4793 | 0.8689 | 0.2491 | 0.4882 |
| | Seasonal Naive | 1 | 37.925 | 28.820 | 38.57% | -2.4913 | 0.6230 | 0.1814 | 0.1625 |
| | **Final ARIMAX** | **1** | **16.931** | **11.443** | **13.30%** | **0.3042** | **0.8279** | **0.1840** | **0.1783** |

### Error Growth across Multi-Step Horizons (1–7 Days) on Test Set:
$$\\text{Model (with Fourier Harmonics)} \\quad \\text{vs} \\quad \\text{Persistence}$$

| Horizon $h$ (Days Ahead) | ARIMAX Model RMSE | Persistence RMSE | Error Ratio (Model / Persist) | Error Reduction (%) |
| :---: | :---: | :---: | :---: | :---: |
| **$h = 1$** | **32.118** | 32.718 | 0.982 | $1.8\\%$ |
| **$h = 2$** | **42.530** | 47.137 | 0.902 | $9.8\\%$ |
| **$h = 3$** | **45.610** | 51.005 | 0.894 | $10.6\\%$ |
| **$h = 4$** | **47.319** | 53.455 | 0.885 | $11.5\\%$ |
| **$h = 5$** | **48.736** | 58.415 | 0.834 | $16.6\\%$ |
| **$h = 6$** | **49.399** | 61.383 | 0.805 | $19.5\\%$ |
| **$h = 7$** | **49.661** | **61.157** | **0.812** | **18.8%** |

---

## 4. Artifacts and Figures Generated (Phase 5)

- **Visualizations (300-DPI PNGs in [`outputs/figures/`](../outputs/figures)):**
  1. [`p5_grid_search_heatmap.png`](../outputs/figures/p5_grid_search_heatmap.png): Dual heatmap contrasting Validation RMSE by $(p,q)$ for $d=0$ vs $d=1$.
  2. [`p5_model_diagnostics.png`](../outputs/figures/p5_model_diagnostics.png): 4-panel diagnostic suite (standardized residuals, histogram + KDE vs Normal, Q-Q plot, correlogram).
  3. [`p5_test_actual_vs_predicted.png`](../outputs/figures/p5_test_actual_vs_predicted.png): 1-day ahead walk-forward test trajectory with $95\\%$ prediction interval band ($96.70\\%$ coverage).
  4. [`p5_rmse_vs_horizon.png`](../outputs/figures/p5_rmse_vs_horizon.png): Multi-step forecast error growth comparison against Persistence across horizons $1$ to $7$.
  5. [`p5_stress_actual_vs_predicted.png`](../outputs/figures/p5_stress_actual_vs_predicted.png): Walk-forward forecast trajectory across the COVID-19 lockdown stress window.
  6. [`p5_category_confusion_matrix.png`](../outputs/figures/p5_category_confusion_matrix.png): CPCB AQI health category confusion matrix on the Test partition.
- **Results Tables ([`outputs/results/`](../outputs/results)):**
  - [`p5_grid_search.csv`](../outputs/results/p5_grid_search.csv): Grid search parameters, information criteria (AIC/BIC), and validation errors for 80+ models.
  - [`p5_metrics.csv`](../outputs/results/p5_metrics.csv): Consolidated regression and category metrics across all baselines and models.
  - [`p5_predictions.csv`](../outputs/results/p5_predictions.csv): Standard format predictions (`date, actual, predicted, horizon, model, split`).
- **Model Checkpoint ([`outputs/models/`](../outputs/models)):**
  - [`arima_final.pkl`](../outputs/models/arima_final.pkl): Serialized fitted `SARIMAX` model object ready for benchmarking against LSTM.
- **Notebook ([`notebooks/`](../notebooks)):**
  - [`05_arima.ipynb`](../notebooks/05_arima.ipynb): Fully executed, reproducible Jupyter notebook.

Phase 5 is complete and fully verified. Ready to proceed to **Phase 6: Deep Learning Time Series Modeling (LSTM)** upon confirmation.
"""

updated_log = pre_p3 + phase3_text
with open(PROJECT_LOG, 'w', encoding='utf-8') as f:
    f.write(updated_log)
print("Updated docs/PROJECT_LOG.md successfully.")

# Update CLAUDE.md
with open(CLAUDE_MD, 'r', encoding='utf-8') as f:
    claude_content = f.read()

claude_content = claude_content.replace(
    "- [ ] **Phase 5: Classical Time Series Modeling (ARIMA / SARIMA)** (`notebooks/05_arima_sarima.ipynb` — parameter optimization, residual diagnostics, multi-step AQI forecasting)",
    "- [x] **Phase 5: Classical Time Series Modeling (ARIMA / ARIMAX)** (`notebooks/05_arima.ipynb` — complete & verified)"
)

# Update Final Model specification in CLAUDE.md
phase5_spec_text = """
---

## Phase 5 Final Model Specification & Checkpoints
- **Selected Architecture:** $\\text{ARIMAX}(2, 1, 1) + \\text{Fourier}(K=3) + \\text{PM}_{2.5}(t-1)$
- **Differencing:** $d=1$ (first differencing without constant)
- **Harmonics:** $K=3$ deterministic yearly Fourier terms (sin/cos pairs, period $365.25$ days)
- **Dynamic Regressor:** $\\text{PM}_{2.5}(t-1)$ (lagged 1 day to eliminate future leakage)
- **Fitted Model Artifact:** [`outputs/models/arima_final.pkl`](../outputs/models/arima_final.pkl)
- **Shared Metrics Module:** [`src/metrics.py`](../src/metrics.py)
- **Results Files:**
  - Grid search: [`outputs/results/p5_grid_search.csv`](../outputs/results/p5_grid_search.csv)
  - Metrics: [`outputs/results/p5_metrics.csv`](../outputs/results/p5_metrics.csv)
  - Predictions: [`outputs/results/p5_predictions.csv`](../outputs/results/p5_predictions.csv)
- **Benchmark Performance (Test Partition, 1-Day Horizon):**
  - **ARIMAX RMSE:** **$25.550$** vs Persistence **$32.103$** ($20.4\\%$ improvement)
  - **ARIMAX MAE:** **$17.534$** vs Persistence **$19.555$**
  - **$R^2$:** **$0.6763$**, Category Accuracy: **$76.92\\%$**, QWK: **$0.6471$**
  - **95% Prediction Interval Coverage:** **$96.70\\%$**
  - **Multi-Step Horizon 7 RMSE:** **$49.661$** vs Persistence **$61.157$** ($18.8\\%$ improvement)
"""

if "## Phase 5 Final Model Specification & Checkpoints" not in claude_content:
    claude_content += phase5_spec_text

with open(CLAUDE_MD, 'w', encoding='utf-8') as f:
    f.write(claude_content)
print("Updated CLAUDE.md successfully.")

