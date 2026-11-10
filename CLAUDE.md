# UCS321 Engineering Design / EST: IndianOil AQI Forecasting

## Project Status: Complete – Frozen
All modeling, evaluation, and consistency verification across Phases 1 through 7 are complete and frozen. The single authoritative source of truth for all quantitative values in technical report writing is `docs/FINAL_NUMBERS.md`.

## Project Statement
"Indian Oil Corporation seeks to forecast the Air Quality Index (AQI) around its refinery zones. Using historical pollutant levels, an ARIMA and LSTM-based forecasting model is needed to predict AQI values and support proactive emission control."

---

## Project Decisions (Locked & Updated)
1. **Granularity**: Daily data only (`city_day.csv` / `station_day.csv`). High-frequency hourly series are bypassed to align with 24-hour regulatory compliance standards.
2. **Target Location**: **Chennai** (CPCL Manali Refinery corridor, an IndianOil subsidiary). Delhi serves strictly as a secondary contingency backup.
3. **Selected Target Series**: **Chennai City-Level Series** (`data/raw/city_day.csv`, `City == 'Chennai'`).
   - *Selection Rule Rationale*: Neither Manali station met the selection prerequisites (Station `TN002` lacked historical depth with only 256 days $< 3$ years; Station `TN003` suffered from $31.28\%$ missingness and a 411-day gap $> 30$ days).
4. **Series Timeline & 2015 Exclusion**:
   - The series **starts on 2016-01-01** and ends on **2020-06-30**.
   - Calendar year 2015 is completely excluded due to an 82-day monitoring outage (September–November 2015). In documentation and reports, this is designated strictly as a "monitoring outage" (cause unverified).
5. **COVID-19 Lockdown Handling & Chronological Splits**:
   - Main modeling series terminates on **29 February 2020**. The period **1 March 2020 – 30 June 2020** is reserved exclusively as a post-lockdown "stress-test" window.
   - **Train**: `2016-01-01` $\rightarrow$ `2018-12-31` (1,096 calendar days, 3 full years)
   - **Validation**: `2019-01-01` $\rightarrow$ `2019-08-31` (243 calendar days, 0.00% missing AQI)
   - **Test**: `2019-09-01` $\rightarrow$ `2020-02-29` (182 calendar days, 0.00% missing AQI)
   - **Stress Test**: `2020-03-01` $\rightarrow$ `2020-06-30` (122 calendar days, 0.00% missing AQI)
6. **Forecast Horizons**: 1-day ahead ($t+1$, primary operational horizon), 7-day ahead ($t+7$, medium-term proactive extension).
7. **Feature Sets for Modeling**:
   - **ARIMA / SARIMA**: Univariate modeling on daily AQI (SARIMAX with PM2.5 as optional extension).
   - **LSTM Feature Sets**:
     - **CORE (Primary)**: `[PM2.5, CO, O3, NO2, SO2]`
     - **ALL9 (Comparison Experiment)**: `[PM2.5, NO, NO2, NOx, CO, SO2, O3, Benzene, Toluene]` (NH3 removed due to 411-day gap and weak correlation)
     - **Excluded**: `PM10` and `Xylene` (rejected due to excessive missingness $>30\%$).
8. **Outlier & Spike Policy**:
   - AQI spikes are **NOT** removed or capped as outliers — they represent genuine physical and festive emission events (e.g., Diwali firecrackers, Bhogi bonfires, winter inversions). Only physically implausible values (negative or beyond sensor limits) are treated.
   - Pollutant inputs are winsorised at the UPPER BOUND ONLY (train set 99.5th percentile). Lower clipping is omitted to preserve authentic clean atmospheric readings (monsoon scavenging, lockdown signals). True AQI spikes are preserved intact.
9. **Evaluation Metrics**: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Coefficient of Determination ($R^2$), and CPCB qualitative AQI category accuracy.

---

## Directory Structure
```
AI_FOR_ENGINEERS_EST_PROJECT/
├── data/
│   ├── raw/             # Immutable raw data files (city_day.csv, stations.csv, etc.)
│   └── processed/       # Cleaned, imputed, and scaled time series datasets
│       ├── chennai_clean_full.csv
│       ├── train.csv, val.csv, test.csv, stress.csv
│       └── train_scaled.csv, val_scaled.csv, test_scaled.csv, stress_scaled.csv
├── notebooks/           # Phase-wise Jupyter notebooks (02_data_audit.ipynb, 03_preprocessing.ipynb, etc.)
├── src/                 # Reusable utility scripts and source code
├── outputs/
│   ├── figures/         # Publication-quality figures (300-DPI PNGs with p*_ prefix)
│   ├── models/          # Saved ARIMA models, LSTM checkpoints, and fitted scalers (.joblib)
│   └── results/         # Evaluation metrics, performance logs, and CSV tables
├── docs/                # Project documentation and cumulative logs (PROJECT_LOG.md)
├── .venv/               # Virtual environment (Python 3.11)
├── requirements.txt     # Pinned dependency specifications
└── CLAUDE.md            # Project constitution, standards, and workflow guide
```

---

## Environment & Toolchain
- **Python Version**: Python 3.11.0 (64-bit)
- **Virtual Environment**: `.venv`
- **Activation Command (PowerShell)**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Activation Command (CMD)**:
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- **Jupyter Kernel**: Registered as `ucs321-aqi` (Display name: `Python (ucs321-aqi)`)

---

## Phase Execution Plan
- [x] **Phase 1: Environment Setup & Data Inventory** (System check, venv configuration, compatibility tests, data catalogue)
- [x] **Phase 2: Data Audit & Location Selection** (`notebooks/02_data_audit.ipynb` — Chennai city-level series chosen)
- [x] **Phase 3: Preprocessing & Imputation** (`notebooks/03_preprocessing.ipynb` — complete – pending user confirmation)
- [x] **Phase 4: Exploratory Data Analysis & Visualization** (`notebooks/04_eda.ipynb` — complete & verified)
- [x] **Phase 5: Classical Time Series Modeling (ARIMA / ARIMAX)** (`notebooks/05_arima.ipynb` — complete & verified with multi-step origin reconciliation)
- [x] **Phase 6: Deep Learning Time Series Modeling (LSTM / GRU / BiLSTM / CNN-LSTM / Hybrid)** (`notebooks/06_lstm.ipynb` — multi-stage search, Optuna HPO, MC Dropout PI, direct 7-day model, Hybrid ARIMAX-LSTM)
- [x] **Phase 7: Model Evaluation, Statistical Comparison & Operational Synthesis** (`notebooks/07_evaluation.ipynb` — complete, benchmarked & frozen)

---

## Standing Engineering Rules
1. **Notebook Organization**: Strictly one self-contained, reproducible Jupyter notebook per phase in `notebooks/` (`02_data_audit.ipynb`, `03_preprocessing.ipynb`, etc.).
2. **High-Resolution Figures**: All visualizations must be saved programmatically as 300-DPI PNG files in `outputs/figures/` using descriptive names prefixed by phase (e.g., `p3_imputation_before_after.png`).
3. **Data Integrity**: Never alter, overwrite, or mutate raw files in `data/raw/`. All processed outputs must reside in `data/processed/`.
4. **Time Series Rigor**: Chronological splits only (Train $\rightarrow$ Validation $\rightarrow$ Test $\rightarrow$ Stress Test). **NEVER shuffle time-series data**.
5. **No Data Leakage**: Scalers, transformers, and imputers must be fitted **exclusively on the training partition** and applied downstream to validation/test partitions.
6. **Reproducibility**: Fix random seeds uniformly across all random number generators (`numpy`, `random`, `tensorflow`).
7. **Continuous Documentation**: Update `docs/PROJECT_LOG.md` at the conclusion of every phase with quantitative findings, methodology, and academic commentary.

---

## Modelling Hints for Phase 5 & Phase 6 (Discovered in Phase 4 EDA)
1. **Order of Differencing ($d=1$):**
   - Raw AQI is non-stationary (KPSS stat $= 1.373 > 0.463$ critical).
   - First differencing ($d=1$) achieves strict stationarity across both ADF ($p = 8.58 \times 10^{-28}$) and KPSS ($p = 0.10$).
   - Log transformation is not necessary; differencing raw AQI preserves additive physical units and avoids exponentiation bias.
2. **Candidate ARIMA Orders $(p, d, q)$:**
   - Raw PACF cuts off sharply after lag 1 (lag 1 $= 0.697$, lag 2 $= -0.015$), indicating an underlying $\text{AR}(1)$ process.
   - Differenced ACF has significant spikes at lags 1 and 2 ($-0.140, -0.141$).
   - Recommended candidate set for Phase 5 grid-search: **$\text{ARIMA}(1, 1, 1)$**, **$\text{ARIMA}(2, 1, 1)$**, **$\text{ARIMA}(1, 1, 2)$**, and baseline **$\text{ARIMA}(0, 1, 1)$**.
3. **Seasonality Strategy ($m=365$ vs $m=7$):**
   - Weekly seasonality is negligible ($F_s = 0.0000$; day-of-week mean variation is only $\approx 3$ AQI units); omit $s=7$ SARIMA differencing.
   - Annual seasonality is strong ($F_s = 0.5653$, lag-365 ACF peak $= 0.224$).
   - Standard $\text{SARIMA}(P,D,Q)_{365}$ is state-space intractable ($365 \times 365$ matrix inversion); use **ARIMAX / SARIMAX with deterministic harmonic Fourier pairs** (sin/cos terms) as external regressors.
4. **Predictive Features & Lookback for LSTM (Phase 6):**
   - Dominant forward predictor is $\text{PM}_{2.5}$ (cross-correlation with next-day AQI $= 0.606$), followed by $\text{CO}$ ($0.295$) and $\text{O}_3$ ($0.207$).
   - $\text{NO}_2$ and $\text{SO}_2$ have near-zero linear cross-correlations with next-day AQI ($< 0.06$).
   - Cross-correlation remains significant through lag 3 ($r > 0.25$), recommending an LSTM sequence lookback window of **7 to 14 days**.
   - Include cyclical coordinates `doy_sin` and `doy_cos`.
5. **Loss Function & Distributional Skewness:**
   - High positive skewness ($2.02$) and heavy tails (kurtosis $7.05$) in Train AQI suggest evaluating **Huber loss** or **MAE** alongside standard MSE to prevent rare spikes from destabilizing LSTM gradient descent.
6. **Class Imbalance in Evaluation:**
   - $>93\%$ of days fall into CPCB 'Satisfactory' ($51–100$) and 'Moderate' ($101–200$) categories; 'Good', 'Very Poor', and 'Severe' are rare episodic states ($<1.5\%$ combined).
   - Category-accuracy metrics must report **Macro-Averaged F1-Score** and **Cohen's Quadratic Weighted Kappa** in addition to overall RMSE, MAE, and MAPE.

---

## Phase 5 Final Model Specification & Checkpoints
- **Selected Architecture:** $\text{ARIMAX}(2, 1, 1) + \text{Fourier}(K=3) + \text{PM}_{2.5}(t-1)$
- **Differencing:** $d=1$ (first differencing without constant)
- **Harmonics:** $K=3$ deterministic yearly Fourier terms (sin/cos pairs, period $365.25$ days)
- **Dynamic Regressor:** $\text{PM}_{2.5}(t-1)$ (lagged 1 day to eliminate future leakage)
- **Fitted Model Artifact:** [`outputs/models/arima_final.pkl`](outputs/models/arima_final.pkl)
- **Shared Metrics Module:** [`src/metrics.py`](src/metrics.py)
- **Results Files:**
  - Grid search: [`outputs/results/p5_grid_search.csv`](outputs/results/p5_grid_search.csv)
  - Metrics: [`outputs/results/p5_metrics.csv`](outputs/results/p5_metrics.csv)
  - Predictions: [`outputs/results/p5_predictions.csv`](outputs/results/p5_predictions.csv)
  - Multi-step origins: [`outputs/results/p5_multistep_origins.csv`](outputs/results/p5_multistep_origins.csv) (175 origins: `2019-09-01` to `2020-02-22`)
- **Benchmark Performance (Test Partition, 1-Day Horizon, N=182 Days):**
  - **ARIMAX RMSE:** **$25.965$** vs Persistence **$32.103$** ($19.1\%$ improvement)
  - **ARIMAX MAE:** **$18.447$** vs Persistence **$19.555$**
  - **$R^2$:** **$0.6657$**, Category Accuracy: **$76.92\%$**, QWK: **$0.6471$**
  - **95% Prediction Interval Coverage:** **$96.70\%$**
  - **Multi-Step Horizon 7 RMSE (N=175 Origins):** **$50.352$** (`ARIMA-Fourier (multi-step)`) vs Persistence **$61.157$** ($17.7\%$ improvement)
- **Methodological Audits & Corrections:**
  - *AIC Comparability*: AIC is not comparable across differencing orders ($d=0$ vs $d=1$) because log-likelihood is computed on differenced series $\Delta y_t$ vs levels $y_t$. Order selection rests on validation RMSE.
  - *Residual Independence*: Ljung–Box test at lag 10 achieves $p \approx 0.04$, indicating residuals are close to white noise while mild short-lag autocorrelation remains.
  - *Stress-Test Regime Change*: High nominal 95% PI coverage during lockdown reflects wide predictive intervals relative to compressed post-lockdown variance; ARIMAX underperformed persistence during lockdown (RMSE $17.74$ vs $14.45$, QWK $0.18$ vs $0.54$) due to structural breakdown of learned seasonal and industrial emissions relationships.

---

## Phase 6 Deep Learning Specification & Checkpoints
- **Pipeline Scripts:**
  - `src/sequences.py`: Sliding window sequence generator (leakage-safe split assignment by target date).
  - `src/models_dl.py`: Stacked LSTM, Stacked GRU, BiLSTM, Conv1D-LSTM architectures.
  - `src/train_dl.py`: Resumable training script, Optuna HPO, MC Dropout, multi-step, hybrid ARIMAX-LSTM.
  - `src/build_phase6_notebook.py`: Generates `notebooks/06_lstm.ipynb` loading precomputed artifacts.
- **Artifacts & Model Weights ([`outputs/models/`](outputs/models)):**
  - `lstm_1day_seed*.keras`: 5-seed final 1-day model checkpoints.
  - `lstm_7day_seed*.keras`: 5-seed direct 7-day model checkpoints.
  - `hybrid_resid_seed*.keras`: 5-seed residual LSTM checkpoints.
  - `dl_config.json`: Production configuration and hyperparameters.
- **Results Files ([`outputs/results/`](outputs/results)):**
  - `p6_experiments.csv`: Complete log of all training runs across stages.
  - `p6_optuna.db`: SQLite database of Optuna Bayesian HPO study.
  - `p6_optuna_trials.csv`: Detailed parameter and metric records of all HPO trials.
  - `p6_metrics.csv`: Consolidated evaluation metrics across all paradigms.
  - `p6_predictions.csv`: Standardized prediction trajectories with MC Dropout 95% intervals.
  - `p6_execution_times.json`: Runtime breakdown per experimental stage.


---

## Phase 7 Final Evaluation & Comparison Checkpoints
- **Master Script:** [`src/run_phase7.py`](src/run_phase7.py)
- **Notebook:** [`notebooks/07_evaluation.ipynb`](notebooks/07_evaluation.ipynb)
- **Results Files ([`outputs/results/`](outputs/results)):**
  - [`p7_master_results.csv`](outputs/results/p7_master_results.csv): Consolidated 1-day metrics across Val, Test, Stress for all 9 model families.
  - [`p7_diebold_mariano_pvalues.csv`](outputs/results/p7_diebold_mariano_pvalues.csv): Diebold–Mariano test p-value matrix (HLN correction).
  - [`p7_alert_evaluation.csv`](outputs/results/p7_alert_evaluation.csv): Operational alert performance (AQI > 100, AQI > 200).
  - [`p7_error_by_category.csv`](outputs/results/p7_error_by_category.csv): Stratified error by CPCB qualitative category.
  - [`p7_spike_vs_normal.csv`](outputs/results/p7_spike_vs_normal.csv): Normal days vs top 10% spikes error comparison.
  - [`p7_worst_days.csv`](outputs/results/p7_worst_days.csv): Top 5 worst days per model with calendar event verification.
- **11 Phase 7 Figures ([`outputs/figures/`](outputs/figures)):**
  - `p7_master_benchmark_bar.png`, `p7_2x2_univariate_multivariate.png`, `p7_diebold_mariano_heatmap.png`, `p7_alert_precision_recall.png`, `p7_error_by_category.png`, `p7_spike_vs_normal_error.png`, `p7_side_by_side_confusion_matrices.png`, `p7_test_overlay_zoomed.png`, `p7_rmse_vs_horizon_final.png`, `p7_interval_comparison.png`, `p7_stress_overlay.png`.
- **Primary Operational Conclusion:**
  - **ARIMAX(2,1,1)+F3+PM2.5** is the statistically verified champion model for 1-day ahead forecasting (Test RMSE 25.97, MAE 18.45, F1 0.706).
  - **Direct Multi-Step LSTM (Dense(7))** retrained on Train+Val delivers champion early-horizon multi-step resolution (h=1 RMSE 30.11, beating persistence).
  - **Split-Conformal Prediction Intervals** guarantee valid 95% distribution-free coverage across nominal and stress test distributions.
