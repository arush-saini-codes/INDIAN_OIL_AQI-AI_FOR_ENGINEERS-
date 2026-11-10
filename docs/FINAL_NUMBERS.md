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

*Exclusion Note:* 2015 data is excluded due to an 82-day monitoring outage (Sep–Nov 2015). Sensor anomalies across 15 days in Aug–Sep 2016 (including 2016-09-19 to 2016-09-22 TN004 CO failure) were removed and re-imputed.

*EDA Diagnostics Summary (Train $N=1,096$):* Annual OLS slope $-7.616$ AQI/year; Winter $126.67 \pm 49.83$, SW Monsoon $96.30 \pm 32.47$; $\text{PM}_{2.5}$ contemporaneous $r = 0.766$ ($\rho = 0.731$), next-day lag-1 $r = 0.729$; STL annual strength $F_s = 0.5672$, weekly $F_s = 0.0079$; raw PACF lag-1 $= 0.678$.

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
| **Pure ARIMA(2,1,1)** | 30.49 | 20.45 | 0.3076 | 72.43 | **13.51** | **9.06** | **0.5567** | 86.89 | **-1.15** |
| **Naive Persistence** | 34.27 | 20.85 | 0.1256 | 71.60 | 14.45 | 9.41 | 0.4928 | **87.70** | **-0.31** |
| **7-Day MA** | 34.50 | 24.27 | 0.1138 | 58.85 | 14.65 | 10.22 | 0.4793 | 86.89 | **-0.82** |
| **Hybrid ARIMAX+LSTM** | — | — | — | — | 15.62 | 9.46 | 0.4080 | 86.89 | **-3.19** |
| **Univariate LSTM** | 29.61 | 20.03 | 0.3473 | 72.02 | 15.98 | 12.77 | 0.3801 | 86.07 | **+7.84** |
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
| **ARIMAX vs Persistence** | -2.504 | **0.0132** | Yes (ARIMAX beats Persistence) | No |
| **ARIMAX vs Pure ARIMA** | -2.192 | **0.0297** | Yes (ARIMAX beats Pure ARIMA) | No |
| **ARIMAX vs Final LSTM** | -2.519 | **0.0127** | Yes (ARIMAX beats LSTM) | No |
| **ARIMAX vs Hybrid** | -3.935 | **0.0001** | Yes (ARIMAX beats Hybrid) | **Yes** ($p < 0.005$) |
| **Final LSTM vs Persistence** | -1.786 | **0.0758** | No | No |
| **Pure ARIMA vs Persistence** | -0.861 | **0.3903** | No | No |
| **Hybrid vs Persistence** | -2.126 | **0.0349** | Yes (Hybrid beats Persistence) | No |
| **Hybrid vs Final LSTM** | -1.284 | **0.2010** | No | No |
| **Hybrid vs Pure ARIMA** | -1.751 | **0.0818** | No | No |
| **Final LSTM vs Pure ARIMA** | -1.488 | **0.1385** | No | No |

---

## 6. Operational Alert Decision Metrics (Test Split, $N=182$)

Source: `outputs/results/p7_alert_evaluation.csv`

| Metric / Threshold | ARIMAX(2,1,1)+F3+PM2.5 | Hybrid ARIMAX+LSTM | Final CNN-LSTM | Pure ARIMA(2,1,1) | Naive Persistence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Actual Alert Days (AQI > 100)** | 52 | 52 | 52 | 52 | 52 |
| Precision (AQI > 100) | **64.52%** | 59.21% | 53.01% | 61.11% | 64.15% |
| Recall (AQI > 100) | 76.92% | **86.54%** | 84.62% | 63.46% | 65.38% |
| F1 Score (AQI > 100) | 0.7018 | **0.7031** | 0.6519 | 0.6226 | 0.6476 |
| **Actual Alert Days (AQI > 200)** | 10 | 10 | 10 | 10 | 10 |
| Precision (AQI > 200) | 85.71% | 85.71% | 80.00% | **100.00%** | 70.00% |
| Recall (AQI > 200) | 60.00% | 60.00% | 40.00% | 50.00% | **70.00%** |
| F1 Score (AQI > 200) | **0.7059** | **0.7059** | 0.5333 | 0.6667 | 0.7000 |

*Sample Size Note:* AQI > 200 days occurred on 10 days in the Test split (5.5% frequency), creating severe sample size limits for assessing high-threshold alert reliability.

---

## 7. Uncertainty Quantification: Prediction Intervals

Sources: `outputs/results/p5_predictions.csv`, `outputs/results/p6_predictions.csv`, `outputs/models/arima_final.pkl`, `outputs/results/p7_master_results.csv`.

| Interval Construction | Type | Calibration Quantile ($q$) | Mean Width (AQI) | Test Empirical Coverage | Stress Empirical Coverage |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ARIMAX Split-Conformal** | Distribution-free | **63.87** | 127.74 | **96.70%** | **98.36%** |
| **LSTM Split-Conformal** | Distribution-free | **57.71** | 115.43 | **96.15%** | **98.36%** |
| **ARIMAX Parametric** | Gaussian ($\pm 1.96 \sigma_t$) | N/A | 114.39 | **96.15%** | 97.54% |
| **LSTM MC-Dropout** | Epistemic parameter | N/A | 31.69 | **37.91%** | N/A (not saved for Stress) |

*Caveat on Coverage:* Split-conformal theoretical finite-sample marginal coverage guarantees require exchangeability, which time-series data with drift and seasonal shifts do not strictly satisfy.

---

## 8. Worst-Forecast Days Audit & Spike Analysis

Source: `outputs/results/p7_worst_days.csv`, `outputs/results/p7_spike_vs_normal.csv`

*Spike vs Normal Split (Test $N=182$):* Top 10% highest Test days defined by $\text{AQI} \ge 134.90$ (`q90 = 134.90`, $N=19$ Spike days; $N=163$ Normal days). All models show negative spike bias ($-27.58$ to $-55.54$ AQI; ARIMAX $-36.83$, LSTM $-39.09$, Hybrid $-35.74$).

| Date | Actual AQI | ARIMAX Error | LSTM Error | Calendar Event Match | Description / Root Cause |
| :---: | :---: | :---: | :---: | :--- | :--- |
| **2019-10-27** | 306.0 | **-140.81** | **-144.74** | **Diwali 2019** | Verified festival firecracker smoke; single largest spike |
| **2020-01-14** | 230.0 | **-92.22** | **-85.63** | **Bhogi 2020** | Verified harvest bonfire burning festival; morning smoke |
| **2019-11-04** | 222.0 | -90.43 | -83.72 | No matched event | Sudden post-monsoon jump from 91 to 222 AQI |
| **2019-11-12** | 77.0 | +72.90 | +66.91 | No matched event | Over-prediction following rapid ambient drop after multi-day peak |
| **2019-11-07** | 273.0 | -67.96 | -75.09 | No matched event | Unmodeled severe pollution buildup |

*Audit Summary:* Exactly **2 of the 5 worst days** (Rank 1 & Rank 2 for both ARIMAX and LSTM) correspond to calendar-verified festival events (Diwali 2019-10-27, Bhogi 2020-01-14). The remaining 3 days carry "No matched event".
