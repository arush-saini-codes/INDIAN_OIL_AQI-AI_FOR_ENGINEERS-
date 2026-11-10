# Executive Results Summary: IndianOil AQI Forecasting
**University Course Project — UCS321 (EST) Engineering Design**  
**Sponsor Problem:** Indian Oil Corporation Limited (CPCL Manali Refinery, Chennai)  
**Target:** 1-day ahead and 7-day ahead Air Quality Index (AQI) forecasting for proactive environmental management  
**Evaluation Period:** Test Split (182 days: 2019-09-01 to 2020-02-29); COVID Lockdown Stress Split (122 days: 2020-03-01 to 2020-06-30, Mean AQI = 78.81)

---

## 8 Core Factual Findings

1. **Classical ARIMAX Outperformed All Models for 1-Day Forecasting**:
   - The **ARIMAX(2,1,1) + Fourier(K=3) + PM2.5** model achieved the lowest Test RMSE of **25.97** and lowest MAE of **18.45**, with an $R^2$ of **0.6657** and CPCB category accuracy of **76.92%**.
2. **Deep Learning Underperformed Classical ARIMAX on Test Despite Extensive Tuning**:
   - The champion Deep Learning model (**CNN-LSTM**, Lookback=7, Core features, 80/48 units) achieved a Test RMSE of **27.63** and MAE of **21.21**. While it had lower RMSE than ARIMAX on the Validation set (25.63 vs 28.91), it did not maintain this advantage on the Test set.
3. **Statistical Hypothesis Testing (Diebold–Mariano with HLN Correction)**:
   - At the standard unadjusted significance level ($\alpha = 0.05$), pairwise Diebold–Mariano tests with Harvey–Leybourne–Newbold (HLN) small-sample correction indicate that ARIMAX significantly outperformed Persistence ($p = 0.0132$), Pure ARIMA ($p = 0.0297$), LSTM ($p = 0.0127$), and Hybrid ($p = 0.0001$). The difference between LSTM and Persistence was not statistically significant ($p = 0.0758$).
   - *Multiple-Comparison Note:* When applying a Bonferroni correction for 10 pairwise comparisons ($\alpha_{\text{adj}} = 0.005$), only the performance difference between ARIMAX and Hybrid remains statistically significant ($p = 0.0001$).
4. **Exogenous Chemical Information is Indispensable Across Both Paradigms**:
   - Adding particulate matter ($\text{PM}_{2.5}$) to classical ARIMA improved Test RMSE from 30.58 to 25.97 (-15.1%). Adding core criteria pollutants to univariate LSTM improved Test RMSE from 31.73 to 27.63 (-12.9%).
5. **Multi-Horizon 7-Day Performance & Validation-Selected Epochs**:
   - Training epochs for the Direct Multi-Step LSTM (Dense(7)) were selected strictly on the validation set without test leakage: mean validation RMSE across all 7 horizons reached its minimum at **Epoch 2** (33.88).
   - Retrained on Train+Val across 5 seeds, the **Direct Multi-Step LSTM beats Persistence only at horizons 5, 6, and 7** ($h=5$: 56.08 vs 58.42; $h=6$: 54.48 vs 61.38; $h=7$: 53.04 vs 61.16). At early horizons $h \in [1, 2, 3, 4]$, both Persistence and recursive ARIMA-Fourier outperform the Direct Multi-Step LSTM ($h=1$: Persistence 32.72, Fourier 32.25 vs LSTM 40.39).
   - Recursive ARIMA-Fourier achieved the lowest RMSE across intermediate horizons $h=2\dots 6$, consistent with the hypothesis that deterministic Fourier harmonics constrain multi-day trajectory dispersion.
6. **Error Analysis on Spikes and Worst-Forecast Days**:
   - On the top 10% highest AQI days (AQI $\ge 148$), every model under-predicted actual levels with mean negative bias ranging from -35.7 to -55.5 AQI units.
   - In an audit of the top 5 worst-forecast days for each model, **2 of the 5 worst days correspond to calendar-verified festival events** (Diwali on 2019-10-27, where ARIMAX under-predicted by 140.8 AQI and LSTM by 144.7 AQI; and Bhogi on 2020-01-14, error -92.2 AQI). The remaining 3 days carry "No matched event".
7. **COVID Lockdown Stress Test (1 Mar – 30 Jun 2020)**:
   - Pure ARIMA achieved the lowest stress RMSE (13.51), followed by persistence (14.45); models using exogenous or seasonal terms (ARIMAX 17.74, multivariate CNN-LSTM 20.28) performed worst, with the LSTMs over-predicting (mean bias +7.84 / +10.55) and ARIMAX under-predicting (−8.32).
   - Across the 122-day national lockdown (mean AQI 78.81 vs pre-lockdown mean 109.30, and test split mean 95.73), model performance and mean forecast bias ($\bar{e} = \frac{1}{N}\sum (\hat{y}_t - y_t)$) were: Pure ARIMA(2,1,1) achieved RMSE 13.51, MAE 9.06, and Bias -1.15 AQI; Naive Persistence achieved RMSE 14.45, MAE 9.41, and Bias -0.31 AQI; 7-Day MA achieved RMSE 14.65, MAE 10.22, and Bias -0.82 AQI; Hybrid ARIMAX+LSTM achieved RMSE 15.62, MAE 9.46, and Bias -3.19 AQI; Univariate LSTM achieved RMSE 15.98, MAE 12.77, and Bias +7.84 AQI; ARIMAX(2,1,1)+F3+PM2.5 achieved RMSE 17.74, MAE 12.37, and Bias -8.32 AQI; and Standalone Multivariate CNN-LSTM achieved RMSE 20.28, MAE 15.40, and Bias +10.55 AQI.
8. **Prediction Interval Calibration**:
   - Split-conformal prediction intervals ($q = 63.87$ for ARIMAX, $q = 57.71$ for LSTM) achieved **96.70%** and **96.15%** empirical coverage on the test set, and **98.36%** during the stress period. Note that the theoretical finite-sample guarantee assumes data exchangeability, which time series with drift do not strictly satisfy. In contrast, Monte Carlo dropout undercovered at **37.91%** because it captures only epistemic parameter uncertainty.

---

## Potential Operational Use

Day-ahead ARIMAX(2,1,1)+F3+PM2.5 forecasts, coupled with split-conformal prediction intervals, can serve as an informational early-warning input for industrial environmental monitoring. Operational decision thresholds demonstrate measured utility on the test set: at AQI > 100 (Moderate+), the model achieved 64.5% precision and 76.9% recall (F1 = 0.702 across 52 actual alert days); at AQI > 200 (Poor+), it achieved 85.7% precision and 60.0% recall (F1 = 0.706 across 10 actual alert days, though rare-event sample size is limited). Festival days represent a known failure mode where all models exhibit large negative forecast bias; addressing this requires explicit calendar and event indicator features in future work. Finally, because the underlying CPCB monitoring represents city-level ambient air quality rather than fence-line perimeter monitoring, these forecasts reflect regional atmospheric conditions and do not isolate the CPCL Manali refinery's specific operational emissions contribution.

---

## Limitations

1. **City-Level vs Fence-Line Resolution**:
   - Data represents CPCB city-level aggregated monitoring for Chennai rather than dedicated fence-line perimeter sensors directly enclosing the CPCL Manali refinery battery limit.
2. **Meteorological Feature Absence**:
   - Planetary boundary layer height, wind velocity and direction, relative humidity, and thermal inversion strength were unavailable in the CPCB historical series, constraining models from predicting meteorological stagnation.
3. **Severe Spike Under-Sampling**:
   - Days with AQI > 200 (Poor+) occurred only 10 times in the test split (5.5% frequency), creating severe sample size limits for assessing high-threshold alert reliability.

---

## Future Scope

1. **Dedicated On-Site Refinery Sensor Integration**:
   - Ingest Continuous Emission Monitoring Systems (CEMS) from CPCL stack exhausts combined with fence-line micro-sensor networks.
2. **Coupling with Numerical Weather Prediction (NWP)**:
   - Integrate meteorological forecasts (wind fields, planetary boundary layer height) to capture atmospheric dispersion conditions.
3. **Calendar Event Indicator Features**:
   - Incorporate explicit binary calendar indicator variables for known episodic events (Diwali, Bhogi) to allow models to learn festival-specific emission increments.
