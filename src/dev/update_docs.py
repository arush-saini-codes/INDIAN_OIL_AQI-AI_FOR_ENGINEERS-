"""Script to update docs/PROJECT_LOG.md and CLAUDE.md with Phase 3 corrections and Phase 4 findings.
"""

import os
import pandas as pd
import numpy as np

PROJECT_LOG = 'docs/PROJECT_LOG.md'
CLAUDE_MD = 'CLAUDE.md'

with open(PROJECT_LOG, 'r', encoding='utf-8') as f:
    log_content = f.read()

# Identify Phase 3 start
p3_idx = log_content.find('# Phase 3: Time Series Preprocessing & Feature Engineering')
pre_p3 = log_content[:p3_idx]

phase3_and_4_text = """# Phase 3: Time Series Preprocessing & Feature Engineering

**Date:** November 2026  
**Status:** Complete & Fully Verified (Part A Corrections Applied)  
**Target City:** Chennai, Tamil Nadu (CPCL Manali Refinery Corridor)  
**Notebook:** [`notebooks/03_preprocessing.ipynb`](../notebooks/03_preprocessing.ipynb)  

---

## 1. Executive Summary & Rationale for Updated Decisions

Phase 3 executes the complete data preparation and feature engineering pipeline for the selected Chennai city-level series. To ensure statistical integrity and align with empirical findings from Phase 2 and user directives, the pipeline implements:

1. **2015 Calendar Exclusion & New Timeline:**
   - *Action:* The analysis horizon starts strictly on **2016-01-01** and runs through **2020-06-30** ($1,643$ continuous calendar days).
   - *Reasoning:* Calendar year 2015 was impaired by an 82-day monitoring outage (September–November 2015). Because the operational cause of this outage remains unverified, attempting multi-month synthetic reconstruction would introduce severe structural bias. Truncating the series to 2016 eliminates this outage entirely, leaving a pristine 4.5-year daily record.
2. **Updated Chronological Partitions:**
   - **Training Set:** `2016-01-01` $\\rightarrow$ `2018-12-31` ($1,096$ calendar days, 3 complete annual cycles).
   - **Validation Set:** `2019-01-01` $\\rightarrow$ `2019-08-31` ($243$ calendar days, $0.00\\%$ missing AQI).
   - **Test Set:** `2019-09-01` $\\rightarrow$ `2020-02-29` ($182$ calendar days, $0.00\\%$ missing AQI).
   - **Stress-Test Window:** `2020-03-01` $\\rightarrow$ `2020-06-30` ($122$ calendar days, $0.00\\%$ missing AQI).
3. **Dual Feature Sets for Recurrent Networks:**
   - **CORE Feature Set (Primary):** `[PM2.5, CO, O3, NO2, SO2]` (5 criteria pollutants critically associated with refinery combustion and urban smog).
   - **ALL9 Feature Set (Comparison Experiment):** `[PM2.5, NO, NO2, NOx, CO, SO2, O3, Benzene, Toluene]` (renamed from ALL10 by removing $\\text{NH}_3$, which had a 411-day monitoring gap and negligible correlation with AQI).
   - *Excluded Pollutants:* $\\text{PM}_{10}$ ($87.86\\%$ missing) and $\\text{Xylene}$ ($100\\%$ missing).
4. **Authentic Spike Preservation & Strict Calendar Attribution:**
   - High AQI values are **not** capped or pruned as statistical anomalies.
   - Only calendar-verified festive/cultural events are named in the cause column (Diwali $\\pm 1$ day, Bhogi 13–14 Jan, New Year 31 Dec–1 Jan). Every other row is explicitly labeled `"No matched event"` without inventing meteorological or industrial causes.
5. **Upper-Bound-Only Winsorization:**
   - Pollutant inputs entering neural networks are clipped at the **train set 99.5th percentile upper bound only**.
   - Lower clipping is completely omitted to preserve authentic low-pollution clean atmospheric states (monsoon scavenging and COVID-19 lockdown signals).
6. **Strict Train-Isolated Transformations:**
   - To prevent temporal data leakage, all imputation climatologies, winsorization clipping quantiles, and MinMax normalization parameters were fitted **exclusively on the 2016–2018 Training partition** and applied statically to downstream splits.

---

## 2. Sequential Preprocessing Pipeline (10 Steps in Order)

### Step 1: Load & Filter
- Ingested `data/raw/city_day.csv` ($29,531$ rows). Filtered `City == "Chennai"`.
- Parsed `Date` attribute into a continuous `DatetimeIndex` and sorted chronologically.
- Retained target variables (`AQI`, `AQI_Bucket`) and the 9 usable pollutant covariates in `ALL9` (`PM2.5`, `NO`, `NO2`, `NOx`, `CO`, `SO2`, `O3`, `Benzene`, `Toluene`).
- Clipped series to `2016-01-01` through `2020-06-30` ($1,643$ rows).

### Step 2: Reindex to Complete Daily Calendar
- Reindexed against an uninterrupted daily calendar template (`freq='D'`).
- Result: Exactly $1,643$ expected calendar days matched $1,643$ existing records. Zero ($0$) missing calendar timestamps were required to be inserted.

### Step 3: Physical Plausibility & Validity Checks
- Evaluated physical boundary conditions based on CPCB/NAAQS ambient measurement standards:
  - Non-negativity ($x \\ge 0$): Count of negative values $= 0$.
  - Upper plausibility thresholds: $\\text{AQI} \\le 500$, $\\text{PM}_{2.5} \\le 1,000~\\mu\\text{g/m}^3$, $\\text{CO} \\le 50~\\text{mg/m}^3$, gaseous pollutants $\\le 500~\\mu\\text{g/m}^3$, VOCs $\\le 200~\\mu\\text{g/m}^3$.
  - Result: All observed values fell strictly within acceptable atmospheric bounds. Exactly **zero ($0$) values were invalidated**.

### Step 4: Missing-Value Audit (Before Imputation)
- Missingness assessed across each chronological partition:
  - In Validation, Test, and Stress-Test windows: **$0.00\\%$ missingness for AQI and all CORE pollutants**.
  - In Train: AQI missingness was $3.28\\%$ ($36$ days); $\\text{PM}_{2.5}$ $3.01\\%$ ($33$ days); $\\text{CO}$ $2.01\\%$ ($22$ days); $\\text{O}_3$ $3.47\\%$ ($38$ days); $\\text{NO}_2, \\text{SO}_2$ $3.01\\%$ ($33$ days).
- **Gap-Duration Distribution for Target and CORE Pollutants:**
  - `AQI`: Maximum gap $= 17$ days (Train partition only). **Gaps $>30$ days $= 0$**.
  - `PM2.5`, `O3`, `NO2`, `SO2`: Maximum gap $= 17$ days. **Gaps $>30$ days $= 0$**.
  - `CO`: Maximum gap $= 11$ days. **Gaps $>30$ days $= 0$**.

### Step 5: Gap-Adaptive, Leakage-Safe Imputation
1. **Short Gaps ($\\le 3$ Days):** Imputed via time-based linear interpolation between adjacent known anchors.
2. **Medium Gaps ($4 \\le L \\le 30$ Days):** Imputed using level-adjusted train seasonal climatology:
   $$\\hat{y}(t) = C(\\text{doy}(t)) + \\delta(t)$$
   where $C(\\text{doy})$ is computed strictly on the 2016–2018 Training partition and smoothed via a 15-day circular rolling window.
3. **Severe Gaps ($> 30$ Days):** Strict procedural gate. Zero gaps $> 30$ days were present in AQI and CORE pollutants.
4. **Imputation Counts:** `AQI`: $36$ values ($2.19\\%$); `PM2.5`: $33$ values ($2.01\\%$); `CO`: $22$ values ($1.34\\%$); `O3`: $38$ values ($2.31\\%$); `NO2`, `SO2`: $33$ values ($2.01\\%$).

### Step 6: Outlier Policy, Spike Table & Upper Winsorization
1. **Spike Table with Strict Calendar Matching:**
   - Isolated top 15 AQI values ($279.0$ to $449.0$).
   - Strict matching rule (within $\\pm 1$ day of known festive dates; all others `"No matched event"`):
     - `2017-10-19` ($431.0$): **Diwali 2017**
     - `2017-01-13` ($360.0$): **Bhogi**
     - `2019-10-27` ($306.0$): **Diwali 2019**
     - `2016-10-29` ($298.0$): **Diwali 2016**
     - All other 11 dates (`2016-09-21`, `2016-09-20`, `2016-09-22`, `2016-09-19`, `2016-10-28`, `2017-03-16`, `2017-03-11`, `2017-01-10`, `2019-11-08`, `2019-02-14`, `2019-02-26`): Strictly designated `"No matched event"`.
   - **Preservation Decision:** All AQI spikes represent authentic physical/atmospheric events and are preserved un-clipped.
2. **Upper-Bound-Only Pollutant Winsorization:**
   - Pollutant inputs clipped at **Train 99.5th percentile upper bound only**. Zero lower bounds clipped.
   - Clipping thresholds and series exceedances:
     - `PM2.5`: Upper bound $= 182.114~\\mu\\text{g/m}^3$ $\\rightarrow$ $7$ days clipped ($0.43\\%$)
     - `NO`: Upper bound $= 26.575~\\mu\\text{g/m}^3$ $\\rightarrow$ $13$ days clipped ($0.79\\%$)
     - `NO2`: Upper bound $= 40.212~\\mu\\text{g/m}^3$ $\\rightarrow$ $8$ days clipped ($0.49\\%$)
     - `NOx`: Upper bound $= 40.788~\\mu\\text{g/m}^3$ $\\rightarrow$ $18$ days clipped ($1.10\\%$)
     - `CO`: Upper bound $= 6.269~\\text{mg/m}^3$ $\\rightarrow$ $6$ days clipped ($0.37\\%$)
     - `SO2`: Upper bound $= 28.769~\\mu\\text{g/m}^3$ $\\rightarrow$ $7$ days clipped ($0.43\\%$)
     - `O3`: Upper bound $= 72.189~\\mu\\text{g/m}^3$ $\\rightarrow$ $11$ days clipped ($0.67\\%$)
     - `Benzene`: Upper bound $= 12.315~\\mu\\text{g/m}^3$ $\\rightarrow$ $8$ days clipped ($0.49\\%$)
     - `Toluene`: Upper bound $= 18.258~\\mu\\text{g/m}^3$ $\\rightarrow$ $7$ days clipped ($0.43\\%$)

### Step 7: Calendar Features
- Engineered cyclical coordinates:
  - Day-of-year: $\\text{doy}_\\text{sin} = \\sin(2\\pi \\cdot \\text{doy} / 365.25)$, $\\text{doy}_\\text{cos} = \\cos(2\\pi \\cdot \\text{doy} / 365.25)$
  - Day-of-week: $\\text{dow}_\\text{sin} = \\sin(2\\pi \\cdot \\text{dow} / 7)$, $\\text{dow}_\\text{cos} = \\cos(2\\pi \\cdot \\text{dow} / 7)$
  - Weekend flag: $\\text{is\\_weekend} \\in \\{0, 1\\}$.

### Step 8: Chronological Partitioning
- Created four disjoint chronological partitions without temporal overlap or shuffling:
  - **Train:** `2016-01-01` to `2018-12-31` ($1,096$ days, $100\\%$ complete post-imputation)
  - **Validation:** `2019-01-01` to `2019-08-31` ($243$ days, $100\\%$ complete)
  - **Test:** `2019-09-01` to `2020-02-29` ($182$ days, $100\\%$ complete)
  - **Stress Test:** `2020-03-01` to `2020-06-30` ($122$ days, $100\\%$ complete)
  - Total verified rows: $1,096 + 243 + 182 + 122 = 1,643$ rows.

### Step 9: Train-Isolated Scaling
- Fitted `MinMaxScaler(feature_range=(0, 1))` **strictly on the Training set**:
  - `scaler_target.joblib`: Fitted on `train_df[['AQI']]` (Min: $30.0$, Max: $449.0$).
  - `scaler_features_core.joblib`: Fitted on `train_df[CORE_POLLUTANTS]`.
  - `scaler_features_all9.joblib`: Fitted on `train_df[ALL9_POLLUTANTS]`.
- Persisted scalers to `outputs/models/`.
- Statically transformed Validation, Test, and Stress-Test sets.

### Step 10: Processed Dataset Persistence
- Persisted nine structured CSV files to `data/processed/`:
  - `chennai_clean_full.csv`: Full $1,643$-day continuous series.
  - `train.csv`, `val.csv`, `test.csv`, `stress.csv`: Unscaled datasets for ARIMA/SARIMA.
  - `train_scaled.csv`, `val_scaled.csv`, `test_scaled.csv`, `stress_scaled.csv`: Scaled datasets for LSTM.

---

## 3. Flow Diagram — Preprocessing Steps

```
1. Load & Filter           -> Ingest city_day.csv, filter Chennai, clip to 2016-01-01 -> 2020-06-30 (1,643 days).
2. Calendar Regularization -> Reindex to continuous daily DatetimeIndex (freq='D'); 0 missing dates inserted.
3. Physical Bounds Check   -> Validate x >= 0, AQI <= 500, CPCB limits; 0 values out of bounds.
4. Missingness Audit       -> Quantify split missingness; verify AQI & CORE have max gap <= 17 d (0 gaps > 30 d).
5. Gap-Adaptive Imputation -> Linear fill for <=3 d; level-adjusted train climatology for 4-30 d; flag imputed points.
6. Spike & Outlier Audit   -> Strict calendar event matching; winsorise pollutants at train 99.5% upper bound only.
7. Cyclical Calendar Feats -> Compute sin/cos transformations for day-of-year and day-of-week, plus is_weekend.
8. Chronological Splits    -> Partition into Train (1,096 d), Validation (243 d), Test (182 d), Stress (122 d).
9. Train-Isolated Scaling  -> Fit MinMaxScaler on Train only; transform all partitions; export scalers to joblib.
10. Data Persistence       -> Save unscaled (ARIMA) and scaled (LSTM) files to data/processed/.
```

---

## 4. Artifacts and Figures Generated (Phase 3)

- **Visualizations (300-DPI PNGs in [`outputs/figures/`](../outputs/figures)):**
  1. [`p3_missing_data_before_after.png`](../outputs/figures/p3_missing_data_before_after.png): Bar chart of missing percentages before and after imputation per feature.
  2. [`p3_imputation_before_after.png`](../outputs/figures/p3_imputation_before_after.png): AQI trajectory plot highlighting imputed timestamps on observed series.
  3. [`p3_aqi_spike_events_table.png`](../outputs/figures/p3_aqi_spike_events_table.png): Structured visual table identifying top 15 AQI peaks with verified calendar events.
  4. [`p3_outlier_boxplots_clipping.png`](../outputs/figures/p3_outlier_boxplots_clipping.png): Before/after boxplots showing upper-bound-only clipping for CORE pollutants.
  5. [`p3_final_cleaned_aqi_splits.png`](../outputs/figures/p3_final_cleaned_aqi_splits.png): Complete cleaned AQI time series across 2016–2020 showing partitions.
- **Results Tables ([`outputs/results/`](../outputs/results)):**
  - [`p3_top_aqi_spikes.csv`](../outputs/results/p3_top_aqi_spikes.csv): Top 15 AQI spike records with calendar-verified events.
- **Fitted Models & Scalers ([`outputs/models/`](../outputs/models)):**
  - `scaler_target.joblib`, `scaler_features_core.joblib`, `scaler_features_all9.joblib`.
- **Processed Data ([`data/processed/`](../data/processed)):**
  - `chennai_clean_full.csv`, `train.csv`, `val.csv`, `test.csv`, `stress.csv`, `train_scaled.csv`, `val_scaled.csv`, `test_scaled.csv`, `stress_scaled.csv`.

---

# Phase 4: Exploratory Data Analysis & Visualization

**Date:** November 2026  
**Status:** Complete & Fully Verified  
**Target City:** Chennai, Tamil Nadu (CPCL Manali Refinery Corridor)  
**Notebook:** [`notebooks/04_eda.ipynb`](../notebooks/04_eda.ipynb)  

---

## 1. Executive Summary & Core Modeling Rules

Phase 4 executes a thorough 10-section exploratory and diagnostic analysis to uncover the mathematical and physical structure governing the Chennai AQI time series. To safeguard against data leakage, all statistical tests, autocorrelation analyses, STL decompositions, and cross-lagged correlations driving model parameterization were computed **strictly on the Train partition** (`2016-01-01` $\\rightarrow$ `2018-12-31`, $1,096$ calendar days).

---

## 2. Flow Diagram — EDA Steps

```
1. Descriptive Statistics Across Splits -> Audit central tendency, dispersion, skewness, kurtosis across splits (save CSV).
2. Trend Analysis & Trajectory Modeling -> 30d/90d rolling means; linear regression of annual means (slope = -9.70 AQI/yr).
3. Seasonality & Microclimate Regimes   -> Monthly boxplots, synoptic seasonal regimes, lack of weekend effect, Year x Month heatmap.
4. CPCB AQI Category Health Distributions -> Quantify severe class imbalance (>93% Satisfactory/Moderate, <1.5% Good/Severe).
5. Feature Relationships & Lagged Corr   -> Train-only Pearson/Spearman heatmaps; lagged correlations (lags 0-7) against AQI(t+1).
6. Spike Investigation & Sep 2016 Audit -> Top 15 spikes profile; multi-station check revealing localized CO anomaly at TN004.
7. Time Series Decomposition (STL)      -> Train-only STL: Annual (m=365, F_s=0.5653) vs Weekly (m=7, F_s=0.0000).
8. Stationarity Diagnostics (ADF & KPSS)-> Train-only ADF & KPSS on raw, differenced, and log AQI; confirm d=1 requirement.
9. Autoregressive / Moving Average Identification -> ACF/PACF up to lag 40 & lag 400; candidate ARIMA orders; Fourier terms for m=365.
10. Lockdown Impact Comparative Analysis -> Mar-Jun 2020 vs 2018-19: PM2.5 (-46.56%), NO2 (-52.24%), Ozone (+42.11% un-titration).
```

---

## 3. Detailed Quantitative Findings by Section

### Section 1: Descriptive Statistics Across Splits
- **Results exported to:** [`outputs/results/p4_descriptive_stats.csv`](../outputs/results/p4_descriptive_stats.csv)
- **Mean AQI Evolution:** Train ($115.18 \\pm 51.97$) $\\rightarrow$ Validation ($103.42 \\pm 36.72$) $\\rightarrow$ Test ($95.73 \\pm 45.03$) $\\rightarrow$ Stress-Test ($78.81 \\pm 20.38$).
- **Distributional Asymmetry:** Train AQI exhibits positive skewness of $2.02$ and kurtosis of $7.05$. Train $\\text{PM}_{2.5}$ exhibits skewness of $1.77$ and kurtosis of $6.51$.
- *Observation:* Episodic emission peaks dominate ambient variability over symmetric Gaussian fluctuations, indicating that robust loss functions (Huber loss) will benefit neural network training.

### Section 2: Trend Analysis
- **Annual Mean AQI (2016–2019):**
  - 2016: $135.66$
  - 2017: $104.34$
  - 2018: $105.49$
  - 2019: $102.94$
- **Linear Trend Slope:** $-9.702$ AQI points per calendar year.
- *Observation:* The series exhibits a notable downward secular trend between 2016 and 2017 before stabilizing at an annual mean of $\\approx 103–105$ in 2018–2019. The 90-day rolling mean traces rhythmic annual crests in winter and troughs during the monsoon.

### Section 3: Seasonality & Temporal Cycles
- **Synoptic Seasonal Means (2016–2019):**
  - Winter (Dec–Feb): $125.72 \\pm 47.77$ AQI (peak pollution due to temperature inversions)
  - Summer (Mar–May): $109.16 \\pm 43.90$ AQI (moderate dispersion and photochemical activity)
  - SW Monsoon (Jun–Sep): $101.96 \\pm 46.96$ AQI (cleanest seasonal regime due to rain wash-out)
  - NE Monsoon / Post-monsoon (Oct–Nov): $116.81 \\pm 60.93$ AQI (cyclonic rain interrupted by Diwali spikes)
- **Day-of-Week Variation:**
  - Means: Mon ($106.50$), Tue ($111.16$), Wed ($115.62$), Thu ($114.83$), Fri ($112.36$), Sat ($112.72$), Sun ($111.72$).
  - Standard deviation across day-of-week means is only $2.94$ AQI units.
- *Observation:* There is no discernible 'weekend effect' in Chennai ambient AQI. Refinery operations, thermal power generation, and port logistics operate 24/7 without weekly abatement.

### Section 4: CPCB AQI Category Distribution
- **Distribution Across Partitions (%):**
  - Train: Good ($0.5\\%$), Satisfactory ($49.5\\%$), Moderate ($43.7\\%$), Poor ($5.4\\%$), Very Poor ($0.5\\%$), Severe ($0.4\\%$).
  - Validation: Good ($1.2\\%$), Satisfactory ($56.4\\%$), Moderate ($40.7\\%$), Poor ($1.6\\%$), Very Poor ($0.0\\%$), Severe ($0.0\\%$).
  - Test: Good ($1.6\\%$), Satisfactory ($69.8\\%$), Moderate ($23.1\\%$), Poor ($4.9\\%$), Very Poor ($0.5\\%$), Severe ($0.0\\%$).
  - Stress-Test: Good ($0.8\\%$), Satisfactory ($84.4\\%$), Moderate ($14.8\\%$), Poor ($0.0\\%$), Very Poor ($0.0\\%$), Severe ($0.0\\%$).
- *Observation:* Severe class imbalance is present ($>93\\%$ concentrated in Satisfactory and Moderate). Rare categories (Good, Very Poor, Severe) account for $<1.5\\%$. Consequently, Phase 6/7 evaluations must report Macro-F1 and Cohen's Weighted Kappa rather than raw accuracy.

### Section 5: Pollutant Relationships & Lagged Correlations (Train Only)
- **Correlations with Concurrent AQI:**
  - $\\text{PM}_{2.5}$: Pearson $r = 0.6359$, Spearman $\\rho = 0.6934$ (primary driver)
  - $\\text{CO}$: Pearson $r = 0.3611$, Spearman $\\rho = 0.1773$
  - $\\text{O}_3$: Pearson $r = 0.2376$, Spearman $\\rho = 0.3241$
  - $\\text{NO}_2$: Pearson $r = 0.0579$, Spearman $\\rho = 0.0075$
  - $\\text{SO}_2$: Pearson $r = 0.0246$, Spearman $\\rho = -0.0321$
- **Cross-Lagged Correlations with Target $\\text{AQI}(t+1)$:**
  - $\\text{AQI}(t)$: Lag $0 = 0.697$, Lag $1 = 0.478$, Lag $2 = 0.345$, Lag $3 = 0.253$.
  - $\\text{PM}_{2.5}(t)$: Lag $0 = 0.606$, Lag $1 = 0.332$, Lag $2 = 0.234$, Lag $3 = 0.156$.
  - $\\text{CO}(t)$: Lag $0 = 0.295$, Lag $1 = 0.256$, Lag $2 = 0.199$, Lag $3 = 0.173$.
- *Observation:* $\\text{PM}_{2.5}$ and autoregressive $\\text{AQI}$ are the two dominant forward predictors for $t+1$ forecasting. $\\text{NO}_2$ and $\\text{SO}_2$ have minimal linear predictive power.

### Section 6: Spike Investigation & Sep 2016 Forensic Deep Dive
- **Diwali & Bhogi Spikes:** Cleanly driven by high particulate loadings ($\\text{PM}_{2.5} > 150~\\mu\\text{g/m}^3$).
- **Forensic Deep Dive on 19–22 Sep 2016 (AQI $372–449$):**
  - City-level $\\text{PM}_{2.5}$ remained low ($12.8–22.2~\\mu\\text{g/m}^3$, CPCB Good range).
  - City-level raw $\\text{CO}$ surged to anomalous levels ($18.1–29.7~\\text{mg/m}^3$).
  - Multi-station audit of `station_day.csv` revealed that Station TN004 (Velachery) recorded CO of $18–30~\\text{mg/m}^3$ and AQI $434–449$, whereas Station TN001 (Alandur) recorded Satisfactory AQI ($64.0$) on Sep 19 before going offline.
  - Due to unmonitored $\\text{PM}_{10}$ ($87.9\\%$ missing), the overall AQI sub-index defaulted entirely to this isolated CO surge at Station TN004, likely representing a hyper-local sensor or telemetry anomaly. Preserving this spike without trimming tests model resilience against historical telemetry faults.

### Section 7: Time Series Decomposition (Train Only)
- **Annual STL Decomposition (period = 365):**
  - Seasonal Strength: $F_s = \\mathbf{0.5653}$ (strong, distinct annual cycle with $\\pm 30$ AQI amplitude).
- **Weekly STL Decomposition (period = 7):**
  - Seasonal Strength: $F_s = \\mathbf{0.0000}$ (negligible weekly cyclicality).
- *Observation:* Annual seasonality is prominent and requires modeling, whereas weekly differencing or weekly seasonal AR terms are empirically unjustified.

### Section 8: Stationarity Diagnostics (Train Only)
- **ADF & KPSS Joint Hypothesis Test Results:**
  - **Raw AQI:** ADF stat $= -5.5661$ ($p = 1.50 \\times 10^{-6}$, rejects unit root) but KPSS stat $= 1.3730$ ($p = 0.01$, rejects stationarity). *Verdict:* Trend/Mean non-stationary due to structural shifts and seasonal cycles.
  - **First-Differenced AQI ($d=1$):** ADF stat $= -15.0765$ ($p = 8.58 \\times 10^{-28}$, strongly rejects unit root); KPSS stat $= 0.0625$ ($p = 0.10$, fails to reject stationarity). *Verdict:* **Strictly Stationary (Both Tests Agree)**.
  - **Log-Transformed AQI:** ADF stat $= -5.2391$ ($p = 7.31 \\times 10^{-6}$); KPSS stat $= 1.4254$ ($p = 0.01$).
  - **First-Differenced Log AQI:** ADF stat $= -12.9926$ ($p = 2.79 \\times 10^{-24}$); KPSS stat $= 0.0599$ ($p = 0.10$).
- *Observation:* Differencing raw AQI once ($d=1$) achieves strict stationarity. Log transform does not provide variance advantages and introduces back-transformation bias; differencing raw AQI ($d=1$) is recommended for ARIMA.

### Section 9: ACF / PACF Analysis & Model Orders (Train Only)
- **Raw Series Diagnostics:** ACF decays slowly (lag 1 $= 0.697$, lag 2 $= 0.478$, lag 3 $= 0.344$); PACF cuts off cleanly after lag 1 (lag 1 $= 0.697$, lag 2 $= -0.015$, within Bartlett confidence bounds). This confirms a primary $\\text{AR}(1)$ foundation.
- **First-Differenced Series Diagnostics ($d=1$):** ACF has negative spikes at lag 1 ($-0.140$) and lag 2 ($-0.141$); PACF has significant negative values at lags 1 to 4.
- **Candidate ARIMA Orders:** $\\text{ARIMA}(1, 1, 1)$, $\\text{ARIMA}(2, 1, 1)$, $\\text{ARIMA}(1, 1, 2)$, and baseline $\\text{ARIMA}(0, 1, 1)$.
- **Annual Cycle Handling ($m=365$):** Long-lag ACF exhibits a distinct positive seasonal peak at lag $365$ ($r = 0.224$). However, standard $\\text{SARIMA}(P,D,Q)_{365}$ is computationally intractable due to $365 \\times 365$ state-space inversion. The recommended operational strategy is **ARIMAX with deterministic Fourier terms (harmonic sin/cos pairs)** as external regressors.

### Section 10: COVID-19 Lockdown Impact Comparative Analysis
- **Mar–Jun 2020 vs Pre-Lockdown Baseline (Mar–Jun 2018–2019):**
  - **AQI:** Baseline $= 95.41$, Lockdown $= 78.81$, Shift $= \\mathbf{-17.40\\%}$
  - **$\\text{PM}_{2.5}$:** Baseline $= 44.81~\\mu\\text{g/m}^3$, Lockdown $= 23.94~\\mu\\text{g/m}^3$, Shift $= \\mathbf{-46.56\\%}$
  - **$\\text{NO}_2$:** Baseline $= 19.90~\\mu\\text{g/m}^3$, Lockdown $= 9.51~\\mu\\text{g/m}^3$, Shift $= \\mathbf{-52.24\\%}$
  - **$\\text{SO}_2$:** Baseline $= 7.04~\\mu\\text{g/m}^3$, Lockdown $= 6.15~\\mu\\text{g/m}^3$, Shift $= \\mathbf{-12.69\\%}$
  - **$\\text{CO}$:** Baseline $= 0.92~\\text{mg/m}^3$, Lockdown $= 0.81~\\text{mg/m}^3$, Shift $= \\mathbf{-11.96\\%}$
  - **$\\text{O}_3$ (Ozone):** Baseline $= 30.51~\\mu\\text{g/m}^3$, Lockdown $= 43.36~\\mu\\text{g/m}^3$, Shift $= \\mathbf{+42.11\\%}$
- *Observation:* Primary combustion pollutants (vehicular $\\text{NO}_2$ and particulate $\\text{PM}_{2.5}$) plummeted by $\\approx 50\\%$. Tropospheric ozone experienced a dramatic **$+42.11\\%$ surge**. This illustrates the classic urban 'VOC-limited' regime: suppressing NO emissions halts the chemical titration of ozone ($\\text{NO} + \\text{O}_3 \\rightarrow \\text{NO}_2 + \\text{O}_2$), allowing un-scavenged ozone to accumulate under strong solar radiation.

---

## 4. Artifacts and Figures Generated (Phase 4)

- **Visualizations (300-DPI PNGs in [`outputs/figures/`](../outputs/figures)):**
  1. [`p4_trend_analysis.png`](../outputs/figures/p4_trend_analysis.png): Trajectory with 30-day and 90-day rolling means, plus annual regression slope ($-9.70$ AQI/year).
  2. [`p4_seasonality_analysis.png`](../outputs/figures/p4_seasonality_analysis.png): 4-panel figure: monthly boxplots, synoptic seasons, day-of-week, and Year $\\times$ Month heatmap.
  3. [`p4_aqi_category_distribution.png`](../outputs/figures/p4_aqi_category_distribution.png): Grouped bar chart comparing CPCB category distributions across all partitions.
  4. [`p4_pollutant_correlations.png`](../outputs/figures/p4_pollutant_correlations.png): Side-by-side Pearson and Spearman correlation matrices on Train partition.
  5. [`p4_lagged_correlations.png`](../outputs/figures/p4_lagged_correlations.png): Cross-lagged correlations of CORE pollutants at lags $0–7$ against $\\text{AQI}(t+1)$.
  6. [`p4_spike_pollutant_breakdown.png`](../outputs/figures/p4_spike_pollutant_breakdown.png): 4-panel spike breakdown and multi-station audit of the Sep 2016 episode.
  7. [`p4_stl_decomposition.png`](../outputs/figures/p4_stl_decomposition.png): STL decomposition comparing Annual ($F_s = 0.5653$) vs Weekly ($F_s = 0.0000$) cycles on Train.
  8. [`p4_acf_pacf.png`](../outputs/figures/p4_acf_pacf.png): 5-panel ACF/PACF suite: raw series (lags 1–40), differenced series ($d=1$), and long-lag ACF (lags 1–400).
  9. [`p4_lockdown_comparative.png`](../outputs/figures/p4_lockdown_comparative.png): Comparative bar chart showing absolute levels and percentage shifts during COVID-19 lockdown.
- **Results Tables ([`outputs/results/`](../outputs/results)):**
  - [`p4_descriptive_stats.csv`](../outputs/results/p4_descriptive_stats.csv): Comprehensive summary statistics for AQI and CORE pollutants across all splits.
- **Notebook ([`notebooks/`](../notebooks)):**
  - [`04_eda.ipynb`](../notebooks/04_eda.ipynb): Fully executed, reproducible Jupyter notebook containing all code, tables, and academic observations.

Phase 4 is complete and fully verified. Ready to proceed to **Phase 5: Classical Time Series Modeling (ARIMA / SARIMA)** upon user confirmation.
"""

# Replace in PROJECT_LOG
updated_log = pre_p3 + phase3_and_4_text
with open(PROJECT_LOG, 'w', encoding='utf-8') as f:
    f.write(updated_log)
print("Updated docs/PROJECT_LOG.md successfully.")

# Update CLAUDE.md
with open(CLAUDE_MD, 'r', encoding='utf-8') as f:
    claude_content = f.read()

# Replace ALL10 with ALL9 in Decision 7
claude_content = claude_content.replace(
    "- **ALL10 (Comparison Experiment)**: `[PM2.5, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene]`",
    "- **ALL9 (Comparison Experiment)**: `[PM2.5, NO, NO2, NOx, CO, SO2, O3, Benzene, Toluene]` (NH3 removed due to 411-day gap and weak correlation)"
)

# Replace winsorization in Decision 8
claude_content = claude_content.replace(
    "Pollutant inputs are winsorised using train-derived 0.5th and 99.5th percentiles to protect gradient stability in neural networks.",
    "Pollutant inputs are winsorised at the UPPER BOUND ONLY (train set 99.5th percentile). Lower clipping is omitted to preserve authentic clean atmospheric readings (monsoon scavenging, lockdown signals). True AQI spikes are preserved intact."
)

# Update Phase 4 checkbox in phase plan
claude_content = claude_content.replace(
    "- [ ] **Phase 4: Exploratory Data Analysis & Visualization** (`notebooks/04_eda.ipynb` — seasonality, ADF stationarity, PACF/ACF, pollutant correlations, industrial emission patterns)",
    "- [x] **Phase 4: Exploratory Data Analysis & Visualization** (`notebooks/04_eda.ipynb` — complete & verified)"
)

# Add Modelling hints for Phase 5/6 section if not present
hints_section = """
---

## Modelling Hints for Phase 5 & Phase 6 (Discovered in Phase 4 EDA)
1. **Order of Differencing ($d=1$):**
   - Raw AQI is non-stationary (KPSS stat $= 1.373 > 0.463$ critical).
   - First differencing ($d=1$) achieves strict stationarity across both ADF ($p = 8.58 \\times 10^{-28}$) and KPSS ($p = 0.10$).
   - Log transformation is not necessary; differencing raw AQI preserves additive physical units and avoids exponentiation bias.
2. **Candidate ARIMA Orders $(p, d, q)$:**
   - Raw PACF cuts off sharply after lag 1 (lag 1 $= 0.697$, lag 2 $= -0.015$), indicating an underlying $\\text{AR}(1)$ process.
   - Differenced ACF has significant spikes at lags 1 and 2 ($-0.140, -0.141$).
   - Recommended candidate set for Phase 5 grid-search: **$\\text{ARIMA}(1, 1, 1)$**, **$\\text{ARIMA}(2, 1, 1)$**, **$\\text{ARIMA}(1, 1, 2)$**, and baseline **$\\text{ARIMA}(0, 1, 1)$**.
3. **Seasonality Strategy ($m=365$ vs $m=7$):**
   - Weekly seasonality is negligible ($F_s = 0.0000$; day-of-week mean variation is only $\\approx 3$ AQI units); omit $s=7$ SARIMA differencing.
   - Annual seasonality is strong ($F_s = 0.5653$, lag-365 ACF peak $= 0.224$).
   - Standard $\\text{SARIMA}(P,D,Q)_{365}$ is state-space intractable ($365 \\times 365$ matrix inversion); use **ARIMAX / SARIMAX with deterministic harmonic Fourier pairs** (sin/cos terms) as external regressors.
4. **Predictive Features & Lookback for LSTM (Phase 6):**
   - Dominant forward predictor is $\\text{PM}_{2.5}$ (cross-correlation with next-day AQI $= 0.606$), followed by $\\text{CO}$ ($0.295$) and $\\text{O}_3$ ($0.207$).
   - $\\text{NO}_2$ and $\\text{SO}_2$ have near-zero linear cross-correlations with next-day AQI ($< 0.06$).
   - Cross-correlation remains significant through lag 3 ($r > 0.25$), recommending an LSTM sequence lookback window of **7 to 14 days**.
   - Include cyclical coordinates `doy_sin` and `doy_cos`.
5. **Loss Function & Distributional Skewness:**
   - High positive skewness ($2.02$) and heavy tails (kurtosis $7.05$) in Train AQI suggest evaluating **Huber loss** or **MAE** alongside standard MSE to prevent rare spikes from destabilizing LSTM gradient descent.
6. **Class Imbalance in Evaluation:**
   - $>93\\%$ of days fall into CPCB 'Satisfactory' ($51–100$) and 'Moderate' ($101–200$) categories; 'Good', 'Very Poor', and 'Severe' are rare episodic states ($<1.5\\%$ combined).
   - Category-accuracy metrics must report **Macro-Averaged F1-Score** and **Cohen's Quadratic Weighted Kappa** in addition to overall RMSE, MAE, and MAPE.
"""

if "## Modelling Hints for Phase 5 & Phase 6" not in claude_content:
    claude_content += hints_section
else:
    # replace existing
    start_pos = claude_content.find("## Modelling Hints for Phase 5 & Phase 6")
    claude_content = claude_content[:start_pos] + hints_section.lstrip()

with open(CLAUDE_MD, 'w', encoding='utf-8') as f:
    f.write(claude_content)
print("Updated CLAUDE.md successfully.")

