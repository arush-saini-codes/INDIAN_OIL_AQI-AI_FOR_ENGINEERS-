# Final Authoritative Project Numbers Reference
**UCS321 Engineering Design / EST Course Project**  
**Sponsor & Problem Statement:** Indian Oil Corporation Limited (CPCL Manali Refinery, Chennai) — Air Quality Index (AQI) Forecasting.  
**Freeze Date:** March 2026  
**Status:** Frozen single source of truth for technical report writing.

---

## 1. Dataset & Chronological Partition Boundaries

| Split Name | Start Date | End Date | Calendar Days ($N$) | Mean AQI | Std AQI | Primary Function | Source File |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Training** | 2016-01-01 | 2018-12-31 | 1096 | 112.86 | 46.91 | Model fitting, parameter estimation, scaler calibration | `data/processed/train.csv` |
| **Validation** | 2019-01-01 | 2019-08-31 | 243 | 103.42 | 36.72 | HPO, early stopping, conformal quantile calibration | `data/processed/val.csv` |
| **Test** | 2019-09-01 | 2020-02-29 | 182 | 95.73 | 45.03 | Out-of-sample benchmarking & statistical testing | `data/processed/test.csv` |
| **Stress (Lockdown)** | 2020-03-01 | 2020-06-30 | 122 | 78.81 | 20.38 | Out-of-distribution structural break evaluation | `data/processed/stress.csv` |
| **Full Series** | 2016-01-01 | 2020-06-30 | 1643 | 107.04 | 44.92 | Entire continuous audited modelling series | `data/processed/chennai_clean_full.csv` |

*Exclusion Note:* 2015 data is excluded due to an 82-day monitoring outage (Sep–Nov 2015). Sensor anomaly from 2016-09-19 to 2016-09-22 (TN004 CO failure) was removed and re-imputed.

---

## 2. Day-Ahead ($h=1$) Test Benchmark (182 Days: 2019-09-01 to 2020-02-29)

Source: `outputs/results/p7_master_results.csv`

| Model Family | Specific Model Name | RMSE | MAE | MAPE (%) | $R^2$ | CPCB Acc (%) | Macro F1 | QWK |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Champion** | **ARIMAX(2,1,1) + Fourier(K=3) + PM2.5** | **25.97** | **18.45** | **20.25** | **0.6657** | **76.92** | **0.3664** | **0.6471** |
| **Ensemble Hybrid** | Hybrid ARIMAX + Residual LSTM | 26.81 | 19.91 | 22.58 | 0.6435 | 74.73 | 0.3644 | 0.6296 |
| **DL Champion** | CNN-LSTM (Lookback=7, Core features) | 27.63 | 21.21 | 23.92 | 0.6214 | 68.68 | 0.3167 | 0.5384 |
| **Classical** | Pure ARIMA(2,1,1) | 30.58 | 19.23 | 19.33 | 0.5362 | 74.18 | 0.3473 | 0.5587 |
| **Deep Learning** | Univariate Stacked LSTM (Lookback=7) | 31.73 | 22.87 | 25.43 | 0.5008 | 70.88 | 0.3205 | 0.5226 |
| **Baseline** | Naive Daily Persistence ($AQI_t$, 182 days) | 32.10 | 19.56 | 19.40 | 0.4889 | 73.63 | 0.3396 | 0.5951 |
| **Baseline** | 7-Day Moving Average | 42.44 | 26.49 | 27.19 | 0.1067 | 64.84 | 0.2465 | 0.3405 |
| **Baseline** | Seasonal Naive (365-Day Lag) | 71.61 | 53.86 | 61.80 | -1.5426 | 36.81 | 0.1179 | -0.0554 |

---

## 3. Validation ($N=243$) and Stress ($N=122$) Split Benchmarks

Source: `outputs/results/p7_master_results.csv`, `outputs/results/p5_predictions.csv`, `outputs/results/p6_predictions.csv`

| Model Name | Val RMSE | Val MAE | Val $R^2$ | Val Acc (%) | Stress RMSE | Stress MAE | Stress $R^2$ | Stress Acc (%) | Stress Mean Bias ($\bar{e}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure ARIMA(2,1,1)** | 34.02 | 20.48 | 0.1383 | 72.84 | **13.51** | **9.06** | **0.5567** | 86.89 | **-1.15** |
| **Naive Persistence** | 34.27 | 20.85 | 0.1256 | 71.60 | 14.45 | 9.41 | 0.4928 | **87.70** | **-0.31** |
| **7-Day MA** | 34.50 | 24.27 | 0.1138 | 58.85 | 14.65 | 10.22 | 0.4793 | 86.89 | **-0.82** |
| **Hybrid ARIMAX+LSTM** | — | — | — | — | 15.62 | 9.46 | 0.4080 | 86.89 | **-3.19** |
| **Univariate LSTM** | 30.65 | 20.24 | 0.3001 | 69.96 | 15.98 | 12.77 | 0.3801 | 86.07 | **+7.84** |
| **ARIMAX(2,1,1)+F3+PM2.5** | 28.91 | 17.33 | 0.3778 | 74.90 | 17.74 | 12.37 | 0.2361 | 80.33 | **-8.32** |
| **CNN-LSTM (Multivariate)** | **25.63** | **17.03** | **0.5107** | **75.72** | 20.28 | 15.40 | 0.0019 | 81.97 | **+10.55** |
| **Seasonal Naive (365d)** | 46.33 | 34.76 | -0.5985 | 49.79 | 37.93 | 28.82 | -2.4913 | 62.30 | **+22.70** |

*Note on Stress Mean Forecast Bias:* Defined as $\bar{e} = \frac{1}{N}\sum_{t=1}^{N}(\hat{y}_t - y_t)$ on the 122 stress days (2020-03-01 to 2020-06-30). Pure ARIMA achieved the lowest stress RMSE (13.51), followed by persistence (14.45); models using exogenous or seasonal terms (ARIMAX 17.74, multivariate CNN-LSTM 20.28) performed worst, with the LSTMs over-predicting (mean bias +7.84 / +10.55) and ARIMAX under-predicting (−8.32).

---

## 4. Multi-Horizon Forecasting Benchmark ($h=1$ to $7$ Days)

Evaluated strictly across **175 identical forecast origins** (2019-09-01 to 2020-02-22).  
Sources: `outputs/results/p5_multistep_origins.csv`, `outputs/results/p5_metrics.csv`, `outputs/results/p6_metrics.csv`.

*Direct Multi-Step LSTM Retraining Note:* Retrained across 5 seeds on Train+Val for **2 epochs**, which was selected strictly on the validation set without test leakage (minimizing mean validation RMSE across all 7 horizons at 33.88).

| Horizon ($h$) | Persistence RMSE | ARIMA-Fourier RMSE | LSTM-7day RMSE | Persistence MAE | ARIMA-Fourier MAE | LSTM-7day MAE | LSTM beats Persistence? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$h = 1$** | 32.72 | **32.25** | 40.39 | **20.15** | 22.37 | 31.96 | No |
| **$h = 2$** | 47.14 | **42.83** | 47.30 | **28.99** | 31.64 | 37.45 | No |
| **$h = 3$** | 51.01 | **46.05** | 51.46 | **32.01** | 35.01 | 41.85 | No |
| **$h = 4$** | 53.46 | **47.83** | 54.42 | **33.79** | 36.72 | 44.91 | No |
| **$h = 5$** | 58.42 | **49.33** | 56.08 | **36.80** | 38.07 | 46.92 | **Yes** (56.08 < 58.42) |
| **$h = 6$** | 61.38 | **50.06** | 54.48 | **38.05** | 38.87 | 44.46 | **Yes** (54.48 < 61.38) |
| **$h = 7$** | 61.16 | **50.35** | 53.04 | **37.53** | 39.34 | 42.76 | **Yes** (53.04 < 61.16) |

*Hypothesis Note:* Recursive ARIMA-Fourier achieves the lowest RMSE across intermediate horizons $h=2\dots 6$, consistent with the hypothesis that deterministic seasonal harmonics constrain multi-day trajectory dispersion.

---

## 5. Statistical Hypothesis Testing: Diebold–Mariano Matrix ($h=1$, $T=182$)

Small-sample Harvey–Leybourne–Newbold (HLN 1997) corrected test statistics and two-sided p-values under squared-error loss.  
Sources: `outputs/results/p7_diebold_mariano_stat.csv`, `outputs/results/p7_diebold_mariano_pvalues.csv`.

| Comparison ($M_1$ vs $M_2$) | HLN Test Stat ($DM$) | Two-Sided $p$-Value | Significant at Unadjusted $\alpha=0.05$? | Significant at Bonferroni $\alpha_{\text{adj}}=0.005$? |
| :--- | :---: | :---: | :---: | :---: |
| **ARIMAX vs Persistence** | -2.502 | **0.0132** | Yes (ARIMAX beats Persistence) | No |
| **ARIMAX vs Pure ARIMA** | -2.190 | **0.0297** | Yes (ARIMAX beats Pure ARIMA) | No |
| **ARIMAX vs Final LSTM** | -2.516 | **0.0127** | Yes (ARIMAX beats LSTM) | No |
| **ARIMAX vs Hybrid** | -4.015 | **0.0001** | Yes (ARIMAX beats Hybrid) | **Yes** ($p < 0.005$) |
| **Final LSTM vs Persistence** | -1.785 | **0.0758** | No | No |
| **Pure ARIMA vs Persistence** | -1.828 | **0.0691** | No | No |
| **Hybrid vs Persistence** | -2.484 | **0.0139** | Yes (Hybrid beats Persistence) | No |
| **Hybrid vs Final LSTM** | -1.543 | **0.1246** | No | No |
| **Hybrid vs Pure ARIMA** | -2.317 | **0.0216** | Yes (Hybrid beats Pure ARIMA) | No |
| **Final LSTM vs Pure ARIMA** | -1.493 | **0.1373** | No | No |

---

## 6. Operational Alert Decision Metrics (Test Split, $N=182$)

Source: `outputs/results/p7_alert_evaluation.csv`

| Metric / Threshold | ARIMAX(2,1,1)+F3+PM2.5 | Hybrid ARIMAX+LSTM | Final CNN-LSTM | Pure ARIMA(2,1,1) | Naive Persistence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Actual Alert Days (AQI > 100)** | 52 | 52 | 52 | 52 | 52 |
| Precision (AQI > 100) | **64.52%** | 62.12% | 57.38% | 62.50% | 56.90% |
| Recall (AQI > 100) | 76.92% | **78.85%** | 67.31% | 67.31% | 63.46% |
| F1 Score (AQI > 100) | **0.7018** | 0.6949 | 0.6195 | 0.6481 | 0.6000 |
| **Actual Alert Days (AQI > 200)** | 10 | 10 | 10 | 10 | 10 |
| Precision (AQI > 200) | **85.71%** | **85.71%** | 80.00% | 75.00% | 66.67% |
| Recall (AQI > 200) | **60.00%** | **60.00%** | 40.00% | **60.00%** | **60.00%** |
| F1 Score (AQI > 200) | **0.7059** | **0.7059** | 0.5333 | 0.6667 | 0.6316 |

*Sample Size Note:* AQI > 200 days occurred on 10 days in the Test split (5.5% frequency), creating severe sample size limits for assessing high-threshold alert reliability.

---

## 7. Uncertainty Quantification: Prediction Intervals

Sources: `outputs/results/p7_conformal_quantiles.json`, `outputs/results/p7_master_results.csv`.

| Interval Construction | Type | Calibration Quantile ($q$) | Mean Width (AQI) | Test Empirical Coverage | Stress Empirical Coverage |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ARIMAX Split-Conformal** | Distribution-free | **63.87** | 127.74 | **96.70%** | **98.36%** |
| **LSTM Split-Conformal** | Distribution-free | **57.71** | 115.43 | **96.15%** | **98.36%** |
| **ARIMAX Parametric** | Gaussian ($\pm 1.96 \sigma_t$) | N/A | 93.73 | **96.70%** | 93.44% |
| **LSTM MC-Dropout** | Epistemic parameter | N/A | 31.69 | **37.91%** | 27.87% |

*Caveat on Coverage:* Split-conformal theoretical finite-sample marginal coverage guarantees require exchangeability, which time-series data with drift and seasonal shifts do not strictly satisfy.

---

## 8. Worst-Forecast Days Audit & Festival Identification

Source: `outputs/results/p7_worst_days.csv`

| Date | Actual AQI | ARIMAX Error | LSTM Error | Calendar Event Match | Description / Root Cause |
| :---: | :---: | :---: | :---: | :--- | :--- |
| **2019-10-27** | 306.0 | **-140.78** | **-144.73** | **Diwali** | Verified festival firecracker smoke; single largest spike |
| **2020-01-14** | 217.0 | **-92.20** | **-82.60** | **Bhogi** | Verified harvest bonfire burning festival; extreme morning smoke |
| **2019-10-28** | 239.0 | -81.25 | -84.99 | No matched event | Post-Diwali lingering stagnation |
| **2019-11-20** | 236.0 | -76.88 | -83.17 | No matched event | Unmodeled meteorological stagnation |
| **2019-11-22** | 240.0 | -76.08 | -81.56 | No matched event | Regional atmospheric inversion |

*Audit Summary:* Exactly **2 of the 5 worst days** correspond to calendar-verified festival events. The remaining 3 days carry "No matched event".
