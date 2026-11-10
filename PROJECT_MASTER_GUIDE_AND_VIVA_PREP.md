# Project Master Guide & Viva Preparation
**UCS321 – EST Project | IndianOil (CPCL Manali, Chennai) AQI Forecasting**

---

## 1. The Project in 5 Sentences

We built a forecasting pipeline to predict daily Air Quality Index (AQI) around Indian Oil's CPCL Manali refinery zone in Chennai using historical CPCB pollutant data from 2016 to 2020. Because the two local Manali stations failed our data-completeness rules, we used the Chennai city-level daily series and carefully cleaned 15 faulty CO sensor days without removing real pollution spikes like Diwali. We trained and compared classical statistical models (ARIMA and ARIMAX with annual Fourier terms and lagged PM2.5) against staged deep learning networks (CNN-LSTM, GRU, Stacked LSTM, a 7-day direct model, and an ARIMAX+LSTM hybrid) under strict chronological splitting with zero data leakage. On the 182-day held-out Test period, classical ARIMAX(2,1,1)+Fourier(K=3)+PM2.5 achieved the lowest 1-day error (RMSE 25.97, MAE 18.45), beating CNN-LSTM (27.63), Hybrid (26.81), and Persistence (32.10), though only ARIMAX vs Hybrid remained significant after a 10-test Bonferroni correction. During the 2020 COVID-19 lockdown structural break (mean AQI 78.81), Pure ARIMA (RMSE 13.51) and naive persistence (14.45) stayed most accurate, while exogenous ARIMAX and the LSTMs degraded under the sudden shift.

---

## 2. Key-Numbers Cheat Sheet

All numbers match `docs/FINAL_NUMBERS.md` (single source of truth).

| Metric / Item | Exact Value | Context / Notes |
| :--- | :---: | :--- |
| **Series Window & Splits** | `2016-01-01` to `2020-06-30` ($N=1,643$) | Train 1,096d (mean 112.86) \| Val 243d (103.42) \| Test 182d (95.73) \| Stress 122d (78.81) |
| **2015 Excluded** | 82-day outage | Sep–Nov 2015 monitoring outage; series starts 2016-01-01 |
| **Manali Station Rejection** | TN002: 256d \| TN003: 31.28% missing | TN002 < 3 yrs; TN003 max gap 411 days > 30 days |
| **TN004 CO Sensor Anomaly** | 15 days removed (Aug–Sep 2016) | Target max dropped from faulty 449.00 to Diwali 2017 (431.00) |
| **Champion 1-Day Test (ARIMAX)** | **RMSE 25.97, MAE 18.45, $R^2$ 0.6657** | MAPE 20.25%, CPCB Acc 76.92%, Macro-F1 0.3664, QWK 0.6471 |
| **Hybrid ARIMAX+LSTM Test** | RMSE 26.81, MAE 19.91, $R^2$ 0.6435 | CPCB Acc 74.73%, QWK 0.6296 |
| **CNN-LSTM (L=7, CORE) Test** | RMSE 27.63, MAE 21.21, $R^2$ 0.6214 | Val RMSE 25.63; Test Acc 68.68%, QWK 0.5384 (5 seeds, 18 epochs) |
| **Pure ARIMA(2,1,1) Test** | RMSE 30.58, MAE 19.23, $R^2$ 0.5362 | CPCB Acc 74.18%, QWK 0.5587 |
| **Univariate LSTM (L=7) Test** | RMSE 31.73, MAE 22.87, $R^2$ 0.5008 | CPCB Acc 70.88%, QWK 0.5226 |
| **Persistence Test ($N=182$)** | RMSE 32.10, MAE 19.56, $R^2$ 0.4889 | 7-Day MA: 42.44 / 26.49 \| Seasonal Naive 365d: 71.61 / 53.86 |
| **2x2 Exogenous PM2.5 / CORE Gain**| Classical: $-15.1\%$ \| DL: $-12.9\%$ | Pure ARIMA 30.58 $\to$ ARIMAX 25.97; Uni LSTM 31.73 $\to$ CNN-LSTM 27.63 |
| **Diebold–Mariano (Test, $T=182$)** | ARIMAX vs Hybrid: $p=0.0001$ | Unadjusted $p<0.05$ vs Persist (0.0132), Pure ARIMA (0.0297), LSTM (0.0127); Bonferroni $\alpha=0.005$ keeps only ARIMAX vs Hybrid |
| **7-Day Multi-Step (175 Origins)** | Val Epoch **2** (min Val RMSE 33.88) | LSTM-7d beats Persistence **only at $h=5,6,7$** (56.08 vs 58.42; 54.48 vs 61.38; 53.04 vs 61.16); loses at $h=1\dots 4$ (40.39 vs 32.72 at $h=1$). ARIMA-Fourier leads all $h$ (32.25 to 50.35). |
| **Stress Test (Lockdown, $N=122$)** | **Pure ARIMA RMSE 13.51** (bias -1.15) | Persist 14.45 (-0.31) \| Hybrid 15.62 (-3.19) \| Uni LSTM 15.98 (+7.84) \| ARIMAX 17.74 (-8.32) \| CNN-LSTM 20.28 (+10.55) |
| **Operational Alerts (ARIMAX)** | AQI >100: F1 0.7018 \| >200: F1 0.7059 | >100 (52d): Prec 64.52%, Rec 76.92% \| >200 (10d): Prec 85.71%, Rec 60.00% |
| **Prediction Intervals (95%)** | Split-conformal: **96.70%** (ARIMAX) | $q=63.87$ (width 127.74) \| LSTM conformal 96.15% ($q=57.71$) \| MC-Dropout: **37.91%** (width 31.69) |
| **Worst-Forecast Days (Top 5)** | **2 of 5** matched festivals | Diwali `2019-10-27` (actual 306, err -140.78) & Bhogi `2020-01-14` (217, err -92.20); 3 unmatched |

---

## 3. Every Major Decision & 1–2 Line Justification

1. **Daily data instead of hourly**: CPCB AQI and NAAQS compliance standards are defined on 24-hour averages, and daily aggregation smooths rush-hour noise while keeping ARIMA and LSTM training stable.
2. **Chennai city-level series instead of Manali stations**: Neither Manali station passed our pre-registered quality rule—TN002 had only 256 days (< 3 years) and TN003 had 31.28% missingness with a 411-day gap (> 30 days)—so Chennai city-level (`city_day.csv`) was selected.
3. **2016-01-01 start date (excluding 2015)**: Calendar year 2015 contained an 82-day monitoring outage (Sep–Nov 2015), whereas starting on 2016-01-01 gives 1,643 near-continuous days.
4. **COVID-19 lockdown exclusion from training (Stress split)**: The national lockdown starting March 2020 caused an unprecedented drop in emissions (mean AQI 78.81), so keeping Mar–Jun 2020 untouched prevents distorting normal-regime parameters and lets us test structural breaks honestly.
5. **Sensor anomaly removal (15 days in Aug–Sep 2016)**: Cross-checking station data showed station TN004 logged corrupted CO spikes pushing city AQI up to 449 while PM2.5 and station TN001 stayed normal (~64); flagging and re-imputing those 15 days stopped models from fitting sensor hardware faults.
6. **Upper-bound-only clipping (99.5th percentile on pollutants)**: Upper clipping tames extreme sensor outliers in input pollutants, while omitting lower clipping preserves clean monsoon readings and the 2020 lockdown drop; target AQI spikes were never clipped.
7. **Feature sets CORE vs ALL9**: CORE (`PM2.5, CO, O3, NO2, SO2`) focuses on primary criteria pollutants with low missingness and refinery relevance, while ALL9 tests whether adding NO, NOx, Benzene, and Toluene helps (PM10, Xylene, and NH3 were excluded due to high gaps).
8. **First-order differencing ($d=1$)**: Although raw ADF and KPSS gave conflicting signals, both agreed on strict stationarity at $d=1$, and $d=1$ beat $d=0$ on Validation RMSE (28.907 vs 30.128) by tracking multi-year drift locally.
9. **Fourier terms ($K=3$) instead of SARIMA $m=365$**: Seasonal SARIMA with period 365 requires inverting a 365-dimensional state matrix and is intractable on daily data, whereas $K=3$ Fourier sine/cosine pairs model annual cycles cleanly with just 6 exogenous columns.
10. **Walk-forward evaluation**: Step-by-step evaluation across the 182 Test days mimics real daily deployment where only past data up to day $t$ is available to forecast $t+1$.
11. **5 random seeds for deep learning**: Neural network weight initialization and mini-batch shuffling create variance on small datasets, so averaging 5 fixed seeds (`42, 123, 456, 789, 1011`) reduces run-to-run luck and yields stable ensembles.
12. **Retraining on Train+Val**: After locking all hyperparameters and epochs strictly on Validation, retraining on combined Train+Val (1,339 days) uses the most recent 2019 regime data right before forecasting Test.
13. **Split-conformal prediction intervals**: Unlike Gaussian parametric bands or MC dropout, split-conformal intervals calibrate empirical error quantiles from Validation residuals without assuming normality (with the caveat that time-series drift relaxes exchangeability).
14. **Diebold–Mariano test with Bonferroni adjustment**: DM with HLN correction tests whether Test forecast error differences could just be sample noise, and Bonferroni ($\alpha_{\text{adj}} = 0.005$ across 10 pairs) guards against false discoveries from multiple comparisons.

---

## 4. Concepts Explained Simply (3–5 Lines Each)

- **AQI & CPCB Categories**: India's AQI compresses multiple pollutant readings into a single 0–500 health score using the highest sub-index among monitored pollutants. CPCB divides this into six bands: Good (0–50), Satisfactory (51–100), Moderate (101–200), Poor (201–300), Very Poor (301–400), and Severe (401–500).
- **Stationarity**: A time series is stationary when its mean, variance, and autocorrelation structure stay roughly constant over time. Classical ARIMA models need stationarity so parameters learned in past years stay meaningful later.
- **ADF vs KPSS**: Both test for stationarity, but with opposite null hypotheses. ADF checks whether a unit root exists (rejecting $H_0$ supports stationarity), whereas KPSS assumes the series is stationary (rejecting $H_0$ indicates non-stationarity). Using both catches trend-stationary edge cases.
- **Differencing ($d=1$)**: Differencing transforms the series to day-to-day changes, $y'_t = y_t - y_{t-1}$. It removes slow level shifts so the model focuses on daily fluctuations relative to yesterday's level.
- **ACF / PACF**: The Autocorrelation Function (ACF) measures total correlation between today and $k$ days ago, helping choose moving-average order $q$. Partial ACF (PACF) measures direct correlation at lag $k$ after removing intermediate lags, helping choose autoregressive order $p$.
- **ARIMA($p,d,q$)**: Combines $p$ autoregressive lags of past values, $d$ differences to stabilize the level, and $q$ moving-average lags of past forecast errors into one linear equation.
- **ARIMAX**: An extension of ARIMA that adds external (exogenous) variables—in our case, 6 annual Fourier terms and yesterday's PM2.5—alongside AQI's own history.
- **Fourier Terms**: Pairs of sine and cosine waves ($\sin(2\pi k \cdot \text{DOY}/365.25), \cos(\dots)$) that represent smooth repeating annual seasons without needing 365 seasonal lags.
- **LSTM Gates**: Long Short-Term Memory cells use three gates—forget, input, and output—to decide what past information to erase, store, or pass forward, preventing gradients from vanishing over multi-day sequences.
- **GRU**: A Gated Recurrent Unit simplifies LSTM by merging the forget and input gates into an update gate and combining cell and hidden states, using fewer parameters.
- **CNN-LSTM**: Places a 1-D convolutional layer ahead of the LSTM to scan short local patterns across multiple pollutants over a few consecutive days before the LSTM tracks the sequence.
- **Dropout**: Randomly zeroes out a fraction of neuron connections during training so the network cannot rely too heavily on single weights, reducing overfitting.
- **Early Stopping**: Watches Validation loss after every epoch and stops training when Validation stops improving, preventing the network from memorizing training noise.
- **Huber Loss**: Acts like squared error (MSE) for small residuals and linear error (MAE) for large residuals, keeping sudden festival spikes from distorting gradient updates.
- **Optuna / TPE**: Optuna searches hyperparameters intelligently rather than brute-forcing every combination; its Tree-structured Parzen Estimator (TPE) builds probability models of good vs bad trials to focus on promising regions.
- **MC Dropout**: Keeps dropout turned on at test time and runs 100 forward passes per seed to measure how uncertain the network's weights are (epistemic uncertainty), though it ignores sensor noise.
- **Conformal Prediction**: Computes absolute errors on a held-out calibration set (Validation) and takes the 95th-percentile quantile $q$ to wrap future point forecasts in intervals $[\hat{y}-q, \hat{y}+q]$.
- **RMSE vs MAE**: MAE averages absolute errors equally, whereas RMSE squares errors before averaging, penalizing big misses (like festival spikes) much more heavily.
- **QWK (Quadratic Weighted Kappa)**: Measures classification agreement between predicted and actual CPCB health bands from -1 to +1, penalizing multi-category misses (e.g., predicting Satisfactory when actual is Poor) quadratically.
- **Diebold–Mariano Test**: Compares the daily loss series of two models to test whether one model is statistically more accurate than another on the same test set; the HLN correction adjusts for finite sample size.
- **Data Leakage**: Occurs whenever information from Validation or Test sneaks into training, imputation, scaling, or hyperparameter choices, making evaluation scores artificially optimistic.

---

## 5. 50 Likely Viva Questions & Model Answers

### Group 1: Data & Scope (Q1–Q7)
1. **Why not use the hourly data?**  
   CPCB AQI is defined on 24-hour averages, so daily data matches regulatory reporting standards. Hourly files also contain heavy sub-daily traffic noise and huge missingness, whereas daily series stabilize ARIMA estimation and keep sequence lengths practical.
2. **Why did you use Chennai city-level data instead of the two Manali refinery stations?**  
   We applied an objective pre-modeling rule: Manali station TN002 only had 256 days (< 3 years), and TN003 had 31.28% missingness with a 411-day outage. Following the rule, Chennai city-level (`city_day.csv`) was selected cleanly.
3. **Why did you exclude 2015 from the dataset?**  
   Calendar year 2015 had an 82-day monitoring outage between September and November. Starting on 2016-01-01 gives 1,643 near-continuous daily observations without a multi-month gap.
4. **Why is March–June 2020 kept separate as a stress test?**  
   India's COVID-19 lockdown sharply cut traffic and industrial activity, dropping Chennai's mean AQI to 78.81. Training on lockdown data would corrupt normal-regime patterns, while isolating it tests robustness under a real structural break.
5. **What is the refinery's actual contribution to the AQI?**  
   Our dataset comes from city-wide CPCB ambient monitors, not refinery fence-line sensors, so we cannot isolate CPCL Manali's exact share of emissions. Our forecasts reflect regional ambient air quality around Chennai, combining industrial, port, and urban traffic sources.
6. **Why were PM10, Xylene, and NH3 excluded from LSTM feature sets?**  
   PM10 was missing 87.86% of the time and Xylene 100%, failing our 30% missingness threshold. NH3 had a 411-day early gap and near-zero correlation with AQI ($r=0.0075$), so we kept the 9 clean features (ALL9) and 5 primary ones (CORE).
7. **Why choose Chennai as primary location over Delhi?**  
   The project focuses on Indian Oil's refinery zones, and Chennai hosts IOCL's subsidiary CPCL Manali refinery. Delhi was kept strictly as a backup in case Chennai data proved unusable.

### Group 2: Preprocessing (Q8–Q14)
8. **Isn't removing data cheating? (Regarding the 15 anomaly days in 2016)**  
   No—we verified from `station_day.csv` that station TN004 had a CO sensor fault (station CO up to 17–30 mg/m³) pushing city AQI to 449 while PM2.5 was low and station TN001 read ~64. Removing confirmed hardware faults is standard data hygiene, whereas all real festival spikes like Diwali 2017 (431) were preserved intact.
9. **Why did you remove lower-bound clipping and use upper-only winsorization?**  
   Lower clipping was artificially raising clean low-pollution days from monsoon washouts and the 2020 lockdown. Removing lower clipping preserved clean atmospheric signals while upper 99.5th-percentile clipping on Train tamed rare input sensor spikes.
10. **How did you handle missing values without leaking future data?**  
    All missingness occurred in the Training split (Validation, Test, Stress were 0.00% missing). Short gaps ($\le 3$ days) used linear interpolation, and medium gaps (4–30 days) used day-of-year medians fitted strictly on Training data.
11. **Why wasn't target AQI itself winsorized?**  
    High AQI spikes like Diwali (431) and Bhogi (360) are the exact severe pollution episodes an operational warning system cares about most. Clipping the target would hide real public-health risk.
12. **Why add cyclical `sin`/`cos` day-of-year features instead of integer month 1–12?**  
    Integer months create an artificial jump between December (12) and January (1). Sine and cosine encoding keeps December 31 and January 1 smoothly adjacent on a circle.
13. **How do you know there's no leakage in scaling and windowing?**  
    `MinMaxScaler` and winsorization quantiles were fitted strictly on Train (2016–2018) and saved before transforming Val, Test, and Stress. In `src/sequences.py`, windows are partitioned strictly by target timestamp so no future features ever cross split boundaries.
14. **Why keep unscaled CSVs separately from scaled CSVs?**  
    ARIMA and ARIMAX work directly in interpretable AQI units (and $\mu\text{g/m}^3$ for PM2.5), whereas neural networks need `[0, 1]` scaled inputs for stable gradient descent. Keeping both guarantees identical splits across paradigms.

### Group 3: Exploratory Data Analysis (Q15–Q20)
15. **What is the strongest predictor of next-day AQI in Chennai?**  
    Fine particulate matter PM2.5 has the highest Spearman correlation ($\rho = 0.704$) and strongest next-day lag-1 cross-correlation ($r = 0.606$) with AQI, followed by CO ($0.295$) and O3 ($0.207$).
16. **Why did raw ADF and KPSS disagree on stationarity?**  
    Raw AQI showed a gentle downward drift across 2016–2019 (132.18 in 2016 down to ~103–105 later). ADF rejected a unit root ($p=1.17\times 10^{-6}$), while KPSS rejected level stationarity ($p=0.01$) because of that mean shift.
17. **What does STL decomposition tell us about weekly vs annual seasonality?**  
    STL measured annual seasonal strength at $F_s = 0.5520$ (winter highs vs monsoon lows) and weekly seasonal strength at $F_s = 0.0000$. Day-of-week mean AQI varies by only ~3 points, so weekly seasonal terms were unnecessary.
18. **Why is CPCB category balance a problem for simple accuracy?**  
    Over 93% of days fall in just two bands (Satisfactory and Moderate), while Good, Very Poor, and Severe together represent under 1.5%. High raw accuracy can hide poor spike detection, which is why we also report Macro-F1 and QWK.
19. **What happened to pollutants during the 2020 lockdown?**  
    Combustion pollutants fell sharply (NO2 $-52.2\%$, PM2.5 $-46.6\%$), pulling AQI down to mean 78.81. Meanwhile surface Ozone (O3) rose $+42.1\%$, consistent with reduced NO titration in urban VOC-limited chemistry.
20. **Why did you keep 2016 in training if its mean AQI (132.18) was higher than 2017–2019?**  
    Our sensitivity check showed training on full 2016–2018 ($N=1,096$, Val RMSE 28.907) beat truncating to 2017–2018 ($N=730$, Val RMSE 29.920). Three complete annual cycles gave better Fourier harmonic calibration.

### Group 4: ARIMA & ARIMAX (Q21–Q27)
21. **Why use Fourier harmonics instead of seasonal SARIMA with $m=365$?**  
    SARIMA($P,D,Q)_{365}$ builds a $365\times 365$ state-space matrix that is computationally prohibitive on daily data. Three Fourier sine/cosine pairs ($K=3$) capture annual seasonality cleanly using just 6 exogenous columns.
22. **Why can't you compare AIC between $d=0$ and $d=1$ ARIMA models?**  
    First differencing changes the effective sample size ($N-1$) and the target variable itself ($y_t$ vs $\Delta y_t$), so log-likelihoods and AIC values are on different scales. Model choice between $d=0$ and $d=1$ must rely on out-of-sample Validation RMSE and MAE.
23. **Why did $d=1$ beat $d=0$ even though differenced ACF had a slight over-differencing signature?**  
    Because daily AQI has local multi-year and seasonal baseline shifts, $d=1$ anchors tomorrow's prediction to today's realized level, whereas $d=0$ pulls predictions back toward the 2016–2018 training mean (112.86). On Validation, $d=1$ beat $d=0$ (RMSE 28.907 vs 30.128).
24. **How did adding PM2.5 help ARIMAX?**  
    Adding 1-day-lagged PM2.5($t-1$) dropped Validation RMSE from 30.124 (ARIMA+F3) to 28.907, and cut Test RMSE by 15.1% compared to Pure ARIMA (30.58 down to 25.97).
25. **Did ARIMAX pass residual diagnostics?**  
    Yes: residual mean was near zero (-0.273) and Ljung–Box tests did not reject white noise at lag 10 ($p=0.389$) or lag 20 ($p=0.607$). Heavy positive tails (kurtosis 10.87) remain due to unmodeled Diwali-scale spikes.
26. **How did you run 7-day-ahead classical forecasts without cheating with future PM2.5?**  
    Future PM2.5 is unknown at day $t$, so multi-step 1–7 day classical forecasts used `ARIMA-Fourier` recursively (unrolling AQI predictions using only known future calendar Fourier terms) across the 175 Test origins.
27. **Why didn't you refit ARIMAX daily during the Test walk-forward?**  
    Using Kalman filtering (`apply` without refitting parameters) keeps 2016–2018 learned weights strictly frozen while updating the state with each new day's observation—fast, leakage-free, and directly comparable to frozen LSTM weights.

### Group 5: LSTM & Deep Learning (Q28–Q34)
28. **Why did a short 7-day lookback beat 14, 30, and 60 days in Stage 1?**  
    Cross-correlations between pollutants and AQI decay beyond 3–7 days. Longer windows (30–60 days) add parameters and noise on a ~1,000-day dataset without adding predictive signal.
29. **Why did CNN-LSTM win Stage 2 over Stacked LSTM and GRU?**  
    A 1-D convolution extracts short multi-day local patterns across pollutants (e.g., stagnation build-up over 3 days) before feeding the recurrent layer, achieving Val RMSE 26.241 vs 27.240 (GRU) and 27.878 (Stacked LSTM).
30. **Why did Optuna choose Huber loss and batch size 16?**  
    Huber loss dampens extreme gradients from rare Diwali-level spikes so weights fit typical dynamics cleanly, while batch size 16 introduces helpful gradient noise for regularization on a ~1,000-sample dataset. Learning rate dominated Optuna importance (67.9%).
31. **Why did the 7-day LSTM need only 2 epochs?**  
    When we checked Validation RMSE epoch-by-epoch across all 7 horizons without touching the Test set, Validation error reached its minimum right at Epoch 2 (mean Val RMSE 33.88) and rose steadily thereafter. This happens because longer training overfits the higher 2016–2018 training baseline (mean 112.86 vs 2019 Validation mean 103.42), hurting multi-day trajectories.
32. **Why retrain the final LSTM on Train+Val across 5 seeds?**  
    After locking hyperparameters and epoch counts on Validation, adding the 2019 Validation months gives the final model recent seasonal dynamics right up to the Test start, and averaging 5 seeds lowers prediction variance.
33. **How does the Hybrid ARIMAX+LSTM work?**  
    ARIMAX makes the primary linear forecast, and a residual CNN-LSTM is trained to predict the remaining non-linear ARIMAX error from the past 7 days of residuals and pollutants. Combining them improved standalone DL Test RMSE from 27.63 to 26.81.
34. **Why did MC Dropout undercover so badly (37.91% vs 95% nominal)?**  
    MC Dropout only samples variation across neural weights (epistemic uncertainty), which is relatively small when an ensemble has converged, while ignoring irreducible daily sensor and weather noise (aleatoric uncertainty).

### Group 6: Evaluation & Statistical Tests (Q35–Q40)
35. **Why does the LSTM lose on MAE to persistence (21.21 vs 19.56) even though it wins on RMSE (27.63 vs 32.10)?**  
    RMSE squares errors, so persistence gets penalized heavily on sudden spikes and reversals, whereas CNN-LSTM smooths predictions toward moderate values. However, on calm consecutive days, persistence copies yesterday's exact level with tiny error, giving persistence a lower median/MAE while CNN-LSTM retains slight positive bias on calm days.
36. **What did the Diebold–Mariano test show, and why mention Bonferroni?**  
    At standard unadjusted $\alpha = 0.05$, ARIMAX outperformed Persistence ($p=0.0132$), Pure ARIMA ($p=0.0297$), and CNN-LSTM ($p=0.0127$). Because we ran 10 pairwise tests, Bonferroni correction lowers the threshold to $\alpha = 0.005$, leaving only ARIMAX vs Hybrid ($p=0.0001$) statistically significant—reminding us not to overclaim small gaps on $T=182$ days.
37. **What are split-conformal intervals, and what is their main caveat here?**  
    They take the finite-sample 95th-percentile absolute residual from Validation ($q=63.87$ for ARIMAX) to build $[\hat{y}-q, \hat{y}+q]$, achieving 96.70% empirical Test coverage. The caveat is that theoretical marginal coverage assumes data exchangeability, which time series with multi-year drift do not strictly satisfy.
38. **Why use 175 origins for multi-step comparisons instead of all 182 Test days?**  
    Forecasting 7 days forward from late February 2020 would spill into the March lockdown window. Restricting origins to `2019-09-01` through `2020-02-22` keeps all 7 horizons inside Test on identical dates across all models.
39. **How reliable are the AQI > 200 operational alert numbers?**  
    ARIMAX achieved 85.71% precision and 60.00% recall (F1 = 0.7059), which looks promising, but only 10 days out of 182 exceeded AQI 200 (5.5%). That small sample size means extreme-alert reliability remains uncertain.
40. **Why report QWK in addition to category accuracy?**  
    Accuracy treats predicting Good instead of Poor the same as predicting Good instead of Satisfactory. Quadratic Weighted Kappa penalizes large category jumps quadratically, better reflecting public-health severity.

### Group 7: Results Synthesis (Q41–Q46)
41. **Why did ARIMA beat the LSTM?**  
    Three considerations are consistent with this: first, ~1,096 training days is a small dataset where parsimonious linear ARIMAX with ~10 parameters generalizes with lower variance than deep neural networks with thousands of weights. Second, daily AQI behaves largely as a differenced linear process tied to lagged PM2.5. Third, tuning 180+ DL runs on a 243-day Validation set likely introduced validation selection bias (CNN-LSTM beat ARIMAX on Val 25.63 vs 28.91, but reversed on Test 27.63 vs 25.97).
42. **Why did Pure ARIMA do best during the lockdown (RMSE 13.51)?**  
    Pure ARIMA(2,1,1) uses only first-differenced AQI without seasonal Fourier curves or pre-lockdown multivariate pollutant weights. When emissions collapsed to mean 78.81, Pure ARIMA (bias -1.15) and persistence (14.45, bias -0.31) adapted immediately to yesterday's lower baseline, whereas LSTMs over-predicted (bias +7.84 / +10.55) and ARIMAX under-predicted (bias -8.32) from shifted PM2.5-to-AQI relationships.
43. **How well did the Direct 7-Day LSTM perform overall?**  
    After fixing epoch selection to Epoch 2 on Validation, the 7-day LSTM beats naive persistence only at horizons 5, 6, and 7 (e.g., 53.04 vs 61.16 at $h=7$). At horizons 1–4 it loses to persistence, and recursive ARIMA-Fourier achieves the lowest RMSE across every horizon.
44. **What caused the worst forecast errors on the Test set?**  
    Auditing the top 5 worst Test days showed 2 of the 5 correspond to calendar-verified cultural festivals: Diwali (`2019-10-27`, AQI 306, miss $-140.78$) and Bhogi (`2020-01-14`, AQI 217, miss $-92.20$). The other 3 had no matched calendar event and reflect unmodeled weather stagnation.
45. **Did adding exogenous pollutants help both model families equally?**  
    Very consistently: Test RMSE dropped 15.1% in Classical (30.58 $\to$ 25.97) and 12.9% in Deep Learning (31.73 $\to$ 27.63), showing chemical inputs matter across paradigms.
46. **Why did the Hybrid ARIMAX+LSTM fall between ARIMAX and CNN-LSTM?**  
    On linear daily series, ARIMAX already extracts nearly all predictable signal (white-noise residuals); fitting a second neural stage on noisy residuals slightly increased Test variance (RMSE 26.81 vs 25.97), though it still beat standalone CNN-LSTM (27.63).

### Group 8: Limitations & Operational Use (Q47–Q50)
47. **Can Indian Oil deploy this right now to adjust refinery unit throughput?**  
    No—and we explicitly avoid prescribing refinery unit actions. Because our data is city-level ambient monitoring rather than CPCL Manali fence-line telemetry, forecasts serve only as an informational ambient early-warning input, not an isolated measure of refinery stack emissions.
48. **What is the biggest missing input in the dataset?**  
    Meteorological variables—wind speed, wind direction, relative humidity, and planetary boundary layer height—were absent from the CPCB archive, preventing models from anticipating atmospheric inversions.
49. **Could we fix the Diwali under-prediction error easily?**  
    Yes: adding explicit binary/lead-time calendar indicator features for major pyrotechnic and bonfire festivals (Diwali, Bhogi) in future work would let ARIMAX or neural models learn festival increments directly.
50. **Why is our codebase easy to reproduce and audit?**  
    Every phase has a self-contained notebook (`02` through `07`), core logic lives in modular `src/` scripts, `requirements.txt` pins every dependency, `docs/FINAL_NUMBERS.md` freezes all metrics, and `src/check_consistency.py` programmatically checks predictions against documentation.

---

## 6. How to Demo (Step-by-Step Walkthrough Order)

1. **Start with the System Architecture**: Open [`outputs/figures/flow_diagram_compact.png`](outputs/figures/flow_diagram_compact.png) (or `flow_diagram.png`) to walk through the 6-stage leakage-free pipeline in 60 seconds.
2. **Show Data Audit & Anomaly Cleanup**: Open [`notebooks/02_data_audit.ipynb`](notebooks/02_data_audit.ipynb) (station comparison table showing TN002/TN003 failures) and [`notebooks/03_preprocessing.ipynb`](notebooks/03_preprocessing.ipynb) (`p3_final_cleaned_aqi_splits.png` and TN004 CO anomaly cells).
3. **Show EDA Insights**: Open [`outputs/figures/p4_stl_decomposition.png`](outputs/figures/p4_stl_decomposition.png) and [`outputs/figures/p4_acf_pacf.png`](outputs/figures/p4_acf_pacf.png) to explain annual seasonality and $d=0$ vs $d=1$.
4. **Show Modeling Highlights**: Open [`notebooks/05_arima.ipynb`](notebooks/05_arima.ipynb) (top-10 grid search & diagnostics) and [`outputs/figures/p6_stage2_architecture_comparison.png`](outputs/figures/p6_stage2_architecture_comparison.png).
5. **Show Final Benchmarking & Verification**: Open [`notebooks/07_evaluation.ipynb`](notebooks/07_evaluation.ipynb) alongside key figures [`p7_master_benchmark_bar.png`](outputs/figures/p7_master_benchmark_bar.png), [`p7_test_overlay_zoomed.png`](outputs/figures/p7_test_overlay_zoomed.png), [`p7_rmse_vs_horizon_final.png`](outputs/figures/p7_rmse_vs_horizon_final.png), and [`p7_stress_overlay.png`](outputs/figures/p7_stress_overlay.png), and run `python src/check_consistency.py` live in terminal to verify 0 mismatches.

---

## 7. What We Would Do Differently / Future Work

- **Fence-Line & CEMS Integration**: Pair CPCB ambient monitors with refinery perimeter micro-sensors and Continuous Emission Monitoring System (CEMS) stack logs to isolate refinery plumes from urban traffic.
- **Incorporate Meteorology**: Couple forecast models with Numerical Weather Prediction (NWP) wind fields, humidity, and boundary-layer height so non-festival stagnation episodes are predictable.
- **Festival Calendar Indicators & Online Adaptation**: Include explicit Diwali/Bhogi event indicators and implement periodic rolling window recalibration (or online state updating) so models adapt smoothly across regime breaks like lockdowns without over-anchoring to past baselines.
