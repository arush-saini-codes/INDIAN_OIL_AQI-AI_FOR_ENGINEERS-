# Project Workflow & Architecture Flow Diagram
**UCS321 Engineering Design (EST) — IndianOil AQI Forecasting**

---

## 1. End-to-End Execution Sequence

1. **Phase 1: Environment & System Initialization**
   - Host system check: Windows 11, PowerShell 5.1, Git 2.54.
   - Python selection: Python 3.11.0 64-bit environment (`.venv`) for TensorFlow 2.16 stability.
   - Kernel registration: `ucs321-aqi` Jupyter kernel.
   - Raw data inventory: Audited 26 city files and 130 station monitoring files.
2. **Phase 2: Data Audit & Location Selection**
   - Location selection: Chennai selected (CPCL Manali Refinery zone, an IndianOil subsidiary).
   - Granularity: Daily series (`city_day.csv`), ignoring hourly files.
   - Outage discovery: 82-day monitoring outage discovered in Sep–Nov 2015.
   - Locked start date: 2016-01-01 (excluding the 2015 monitoring outage).
   - Locked splits: Train (2016-01-01 → 2018-12-31), Val (2019-01-01 → 2019-08-31), Test (2019-09-01 → 2020-02-29), Stress (2020-03-01 → 2020-06-30).
3. **Phase 3: Robust Preprocessing Pipeline**
   - Date range enforcement: 2016-01-01 to 2020-06-30 ($N=1643$ days).
   - Validity checks: Physical negative clipping at 0.0.
   - Sensor anomaly removal: Flagged TN004 CO sensor failure (19–22 Sep 2016) and set to NaN.
   - Gap-dependent imputation: Linear interpolation for short gaps ($\le 3$ days); seasonal day-of-year median for medium gaps (4–30 days).
   - Upper-bound winsorization: Capped at training 99.5th percentile (no lower clipping).
   - Cyclical calendar encoding: $\sin$ and $\cos$ of day-of-year ($2\pi \cdot \text{DOY}/365.25$).
   - Strict temporal scaling: Scikit-learn `MinMaxScaler(0,1)` fit on Train only, transforming Val, Test, and Stress.
4. **Phase 4: Exploratory Data Analysis & Feature Verification**
   - Temporal decomposition: STL and classical additive decomposition revealing strong annual cyclicity.
   - Correlation analysis: $\text{PM}_{2.5}$ identified as primary driver of AQI ($r = 0.766$, lag-1 $r = 0.729$).
   - Calendar spike table: Diwali and Bhogi verified as primary spike drivers.
   - Station vs City consistency: Validated CPCL Manali monitoring station (TN001) alignment with city index.
5. **Phase 5: Classical Time Series Modeling (ARIMA / ARIMAX)**
   - Baselines: Naive persistence, 7-day moving average, 365-day seasonal naive.
   - Stationarity verification: ADF test confirmed stationarity after first differencing ($d=1$).
   - Systematic grid search: Evaluated 136 configurations across M1 (pure ARIMA), M2 (ARIMA+Fourier), and M3 (ARIMAX).
   - Champion selection: Final $\text{ARIMAX}(2,1,1) + \text{Fourier}(K=3) + \text{PM}_{2.5}$ achieving Test RMSE 25.97.
   - Multi-horizon recursive testing: Evaluated across 175 rolling origins.
6. **Phase 6: Deep Learning Recurrent Architectures (LSTM)**
   - Stage 1 Factorial Grid: 80 runs evaluating 4 feature sets $\times$ 4 lookbacks $\times$ 5 seeds.
   - Stage 2 Architecture Tournament: 40 runs comparing Stacked LSTM, Bidirectional LSTM, GRU, and Conv1D-LSTM. Winning architecture: CNN-LSTM.
   - Stage 3 Bayesian Optimization: Optuna 60-trial HPO tuning units, dropouts, learning rate, batch size, regularization.
   - Stage 4 Champion Retraining: 5-seed ensemble retrained on Train+Val with MC-dropout uncertainty.
   - Stage 5 Direct Multi-Step: Direct multi-output `Dense(7)` head retrained with 5 seeds.
   - Stage 6 Hybrid Modeling: ARIMAX + Residual LSTM model.
   - Stage 7 Stress Evaluation: Evaluated on COVID lockdown structural break.
7. **Phase 7: Final Evaluation, Hypothesis Testing & Industrial Synthesis**
   - $2 \times 2$ Grid Analysis: Classical vs DL $\times$ Univariate vs Multivariate.
   - Master benchmark: Consolidated comparison across all models on Val, Test, Stress.
   - Diebold–Mariano testing: HLN small-sample corrected pairwise p-value matrix.
   - Operational alerts: Precision, recall, and F1 at AQI > 100 and AQI > 200.
   - Uncertainty quantification: Split-conformal calibrated 95% prediction intervals.
   - Potential operational use: Early warning context, uncertainty bounds, and limitations of city-level data for refinery decision support.

---

## 2. Mermaid System Architecture Flowchart

```mermaid
flowchart TD
    subgraph Data_Ingestion["1. Raw Data Ingestion & Audit"]
        RawCity["data/raw/city_day.csv<br/>(Chennai Daily Series)"]
        RawStation["data/raw/station_day.csv<br/>(CPCL Manali TN001 + Stations)"]
        Audit["Phase 2: Data Audit<br/>- Exclude 2015 Monitoring Outage<br/>- Lock Series: 2016-01-01 to 2020-06-30"]
        RawCity --> Audit
        RawStation --> Audit
    end

    subgraph Preprocessing_Pipeline["2. Robust Preprocessing (Phase 3)"]
        SensFix["Sensor Anomaly Removal<br/>(TN004 CO Anomaly Flagged)"]
        Impute["Gap-Dependent Imputation<br/>- Linear (<=3d)<br/>- Seasonal DOY (4-30d)"]
        Winsor["Upper Winsorization<br/>(Clip Train 99.5th Percentile)"]
        CalendarEnc["Calendar Features<br/>sin/cos DOY + Fourier Harmonics"]
        Scale["Train-Only MinMaxScaler<br/>(Fit on Train 2016-2018)"]
        
        Audit --> SensFix
        SensFix --> Impute
        Impute --> Winsor
        Winsor --> CalendarEnc
        CalendarEnc --> Scale
    end

    subgraph Splits["3. Partitioned Data Splits"]
        Train["Train Split<br/>2016-01-01 to 2018-12-31<br/>(N = 1096 days)"]
        Val["Validation Split<br/>2019-01-01 to 2019-08-31<br/>(N = 243 days)"]
        Test["Test Split<br/>2019-09-01 to 2020-02-29<br/>(N = 182 days)"]
        Stress["Stress Test Split<br/>2020-03-01 to 2020-06-30<br/>(COVID Lockdown, N = 122 days)"]
        
        Scale --> Train
        Scale --> Val
        Scale --> Test
        Scale --> Stress
    end

    subgraph Model_Track_Classical["4A. Classical Time Series Track (Phase 5)"]
        Baselines["Baselines<br/>- Persistence<br/>- 7-Day MA<br/>- Seasonal Naive"]
        GridSearch["Systematic Grid Search (136 configs)<br/>M1: Pure ARIMA<br/>M2: ARIMA + Fourier(K=3)<br/>M3: ARIMAX + PM2.5"]
        ARIMAX["Champion: ARIMAX(2,1,1)+F3+PM2.5<br/>Test RMSE: 25.97 | MAE: 18.45"]
        
        Train --> Baselines
        Train --> GridSearch
        Val --> GridSearch
        GridSearch --> ARIMAX
    end

    subgraph Model_Track_DL["4B. Deep Learning Track (Phase 6)"]
        Stage1["Stage 1: Factorial Grid (80 runs)<br/>Feature Sets x Lookbacks"]
        Stage2["Stage 2: Architecture Tournament<br/>Winner: CNN-LSTM"]
        Stage3["Stage 3: Optuna HPO (60 trials)<br/>Tuned LR, Units, Dropout, Reg"]
        Stage4["Stage 4: Champion Retraining<br/>5 Seeds Ensemble on Train+Val"]
        Stage5["Stage 5: Direct Multi-Step LSTM<br/>Dense(7) Output Head"]
        
        Train --> Stage1
        Val --> Stage1
        Stage1 --> Stage2
        Stage2 --> Stage3
        Stage3 --> Stage4
        Stage4 --> Stage5
    end

    subgraph Hybrid_Track["4C. Hybrid Modeling"]
        Resid["ARIMAX Residual Extraction"]
        ResidLSTM["Residual LSTM Network"]
        HybridModel["Hybrid ARIMAX + LSTM<br/>Test RMSE: 26.81 | MAE: 19.91"]
        
        ARIMAX --> Resid
        Resid --> ResidLSTM
        ResidLSTM --> HybridModel
    end

    subgraph Evaluation["5. Final Evaluation & Benchmarking (Phase 7)"]
        Grid2x2["2x2 Factorial Grid Analysis<br/>Classical vs DL x Univariate vs Multiv"]
        DMTest["Diebold-Mariano Tests<br/>(Harvey-Leybourne-Newbold Correction)"]
        Alerts["Operational Alert Evaluation<br/>AQI > 100 & AQI > 200 (Precision/Recall/F1)"]
        Conformal["Split-Conformal Prediction Intervals<br/>(95% Distribution-Free Coverage)"]
        MultiStep["1-to-7 Day Multi-Step Evaluation<br/>(175 Aligned Origins)"]
        
        ARIMAX --> Grid2x2
        Stage4 --> Grid2x2
        HybridModel --> Grid2x2
        Test --> DMTest
        Test --> Alerts
        Val --> Conformal
        Test --> Conformal
        Stage5 --> MultiStep
    end

    subgraph Operational_Advisory["6. Potential Operational Decision Support"]
        DayAhead["Day-Ahead Baseline Forecast:<br/>ARIMAX(2,1,1)+F3+PM2.5<br/>City-level ambient early warning"]
        UncertaintyBand["Empirical Uncertainty Range:<br/>Split-Conformal Interval (q = 63.87)<br/>Ambient background scenario bounds"]
        MediumRange["Medium-Range Trajectory (h=5-7):<br/>Direct Multi-Step LSTM<br/>Multi-day seasonal trend context"]
        
        Grid2x2 --> DayAhead
        Conformal --> UncertaintyBand
        MultiStep --> MediumRange
    end
```

