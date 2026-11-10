# Comprehensive Project Figure Index
**UCS321 Engineering Design (EST) — IndianOil AQI Forecasting**  
All figures are stored at 300 DPI in [`outputs/figures/`](../outputs/figures).  
Figures designated with **[REPORT]** (~15 figures) are the primary visual assets recommended for the final technical report.

---

## Complete Catalog of Project Figures

| File Name | Phase | Figure Caption / Technical Description | Report Section | Designation |
| :--- | :---: | :--- | :--- | :---: |
| `p2_candidate_series_comparison.png` | Phase 2 | Comparison of candidate monitoring series across Chennai stations | Data Audit | Appendix |
| `p2_missing_value_heatmap.png` | Phase 2 | Heatmap of missingness across stations and pollutants (2015-2020) | Data Audit | Appendix |
| `p2_monthly_seasonality.png` | Phase 2 | Monthly AQI seasonality boxplots for Chennai | Data Audit | Appendix |
| `p2_yearly_mean_aqi.png` | Phase 2 | Annual mean AQI trend across Chennai stations | Data Audit | Appendix |
| `p2_covid_lockdown_impact.png` | Phase 2 | Structural drop in AQI across Chennai during March–June 2020 | Data Audit | Appendix |
| `p3_missing_data_before_after.png` | Phase 3 | Missingness profile before and after interpolation | Preprocessing | Appendix |
| `p3_imputation_before_after.png` | Phase 3 | Time series overlay of raw vs imputed values across 82-day gap | Preprocessing | Appendix |
| `p3_outlier_boxplots_clipping.png` | Phase 3 | Pollutant boxplots showing upper-bound winsorization at 99.5th percentile | Preprocessing | **[REPORT]** |
| `p3_aqi_spike_events_table.png` | Phase 3 | Calendar-matched AQI spike events table (Diwali, Bhogi, New Year) | Preprocessing | Appendix |
| `p3_final_cleaned_aqi_splits.png` | Phase 3 | Final cleaned AQI trajectory partitioned into Train, Val, Test, Stress | Preprocessing | **[REPORT]** |
| `p4_trend_analysis.png` | Phase 4 | Long-term trend analysis with 30-day and 365-day rolling averages | EDA | Appendix |
| `p4_seasonality_analysis.png` | Phase 4 | Day-of-week, monthly, and seasonal boxplots | EDA | Appendix |
| `p4_stl_decomposition.png` | Phase 4 | STL decomposition of Chennai daily AQI (Trend, Seasonal, Residual) | EDA | **[REPORT]** |
| `p4_acf_pacf.png` | Phase 4 | Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots | EDA | **[REPORT]** |
| `p4_pollutant_correlations.png` | Phase 4 | Spearman correlation matrix across AQI and criteria pollutants | EDA | **[REPORT]** |
| `p4_lagged_correlations.png` | Phase 4 | Cross-correlation between lagged pollutants (lags 1-7) and AQI | EDA | Appendix |
| `p4_aqi_category_distribution.png` | Phase 4 | Frequency distribution of CPCB qualitative health categories | EDA | Appendix |
| `p4_spike_pollutant_breakdown.png` | Phase 4 | Chemical fingerprint and pollutant contribution during Diwali spikes | EDA | Appendix |
| `p4_lockdown_comparative.png` | Phase 4 | Pre-lockdown vs lockdown pollutant level comparison | EDA | Appendix |
| `p5_grid_search_heatmap.png` | Phase 5 | ARIMA parameter grid search validation RMSE heatmap | Classical Modeling | Appendix |
| `p5_model_diagnostics.png` | Phase 5 | 4-in-1 residual diagnostic plot for ARIMAX(2,1,1)+F3+PM2.5 | Classical Modeling | **[REPORT]** |
| `p5_test_actual_vs_predicted.png` | Phase 5 | ARIMAX 1-day forecast vs actual on Test split with 95% parametric PI | Classical Modeling | **[REPORT]** |
| `p5_category_confusion_matrix.png` | Phase 5 | Confusion matrix for ARIMAX discrete CPCB category predictions | Classical Modeling | Appendix |
| `p5_rmse_vs_horizon.png` | Phase 5 | Recursive multi-step RMSE trajectory (h=1 to 7) for ARIMA-Fourier | Classical Modeling | Appendix |
| `p5_stress_actual_vs_predicted.png` | Phase 5 | ARIMAX walk-forward performance on COVID lockdown stress window | Classical Modeling | Appendix |
| `p6_stage1_heatmap.png` | Phase 6 | Stage 1 validation RMSE heatmap (Feature Sets vs Lookbacks) | Deep Learning | Appendix |
| `p6_stage2_architecture_comparison.png`| Phase 6 | Architecture tournament bar chart with 5-seed error bars | Deep Learning | **[REPORT]** |
| `p6_optuna_history.png` | Phase 6 | Optuna Bayesian optimization convergence history across 60 trials | Deep Learning | Appendix |
| `p6_optuna_importance.png` | Phase 6 | Hyperparameter importance breakdown (Learning Rate, Units, Reg) | Deep Learning | Appendix |
| `p6_learning_curves.png` | Phase 6 | Training vs Validation loss curves for winning CNN-LSTM | Deep Learning | Appendix |
| `p6_test_actual_vs_predicted.png` | Phase 6 | Final 1-day LSTM ensemble forecast vs actual with MC-dropout band | Deep Learning | Appendix |
| `p6_confusion_matrix.png` | Phase 6 | CPCB category confusion matrix for the LSTM ensemble | Deep Learning | Appendix |
| `p6_rmse_vs_horizon.png` | Phase 6 | Retrained 7-day LSTM vs ARIMA-Fourier vs Persistence | Deep Learning | Appendix |
| `p6_stress_actual_vs_predicted.png` | Phase 6 | Walk-forward stress test comparison across all model paradigms | Deep Learning | Appendix |
| `p7_2x2_univariate_multivariate.png` | Phase 7 | 2x2 factorial grid evaluation: Test RMSE by Paradigm & Scope | Final Evaluation | **[REPORT]** |
| `p7_master_benchmark_bar.png` | Phase 7 | Consolidated Master Benchmark Bar Chart (Test RMSE & MAE) | Final Evaluation | **[REPORT]** |
| `p7_diebold_mariano_heatmap.png` | Phase 7 | Diebold-Mariano test p-value matrix with HLN correction | Final Evaluation | **[REPORT]** |
| `p7_alert_precision_recall.png` | Phase 7 | Operational alert precision, recall, and F1 at AQI > 100 & AQI > 200 | Final Evaluation | **[REPORT]** |
| `p7_side_by_side_confusion_matrices.png`| Phase 7| Side-by-side confusion matrices for ARIMAX, LSTM, and Hybrid | Final Evaluation | **[REPORT]** |
| `p7_error_by_category.png` | Phase 7 | Stratified RMSE grouped by actual qualitative CPCB category | Final Evaluation | Appendix |
| `p7_spike_vs_normal_error.png` | Phase 7 | Forecast degradation on normal days vs top 10% extreme spikes | Final Evaluation | **[REPORT]** |
| `p7_test_overlay_zoomed.png` | Phase 7 | Test forecast overlay with zoomed Diwali 2019 inset | Final Evaluation | **[REPORT]** |
| `p7_rmse_vs_horizon_final.png` | Phase 7 | Final multi-horizon RMSE vs horizon (1 to 7 days) on 175 origins | Final Evaluation | **[REPORT]** |
| `p7_interval_comparison.png` | Phase 7 | Split-Conformal PI vs Parametric PI vs MC-Dropout PI on 30-day window | Final Evaluation | **[REPORT]** |
| `p7_stress_overlay.png` | Phase 7 | Structural break evaluation during COVID lockdown window | Final Evaluation | **[REPORT]** |

---

## Summary of 15 Primary Figures for Report (`[REPORT]`)

1. [`p3_final_cleaned_aqi_splits.png`](../outputs/figures/p3_final_cleaned_aqi_splits.png) — *Cleaned AQI Series & Chronological Partitions*
2. [`p3_outlier_boxplots_clipping.png`](../outputs/figures/p3_outlier_boxplots_clipping.png) — *Upper-Bound Winsorization Distribution*
3. [`p4_stl_decomposition.png`](../outputs/figures/p4_stl_decomposition.png) — *STL Trend-Seasonal Decomposition*
4. [`p4_acf_pacf.png`](../outputs/figures/p4_acf_pacf.png) — *Autocorrelation & Stationarity Profile*
5. [`p4_pollutant_correlations.png`](../outputs/figures/p4_pollutant_correlations.png) — *Spearman Inter-Pollutant Correlation Matrix*
6. [`p5_model_diagnostics.png`](../outputs/figures/p5_model_diagnostics.png) — *ARIMAX Residual Diagnostic Suite*
7. [`p5_test_actual_vs_predicted.png`](../outputs/figures/p5_test_actual_vs_predicted.png) — *ARIMAX One-Step Test Forecast Trajectory*
8. [`p6_stage2_architecture_comparison.png`](../outputs/figures/p6_stage2_architecture_comparison.png) — *Deep Learning Architecture Tournament*
9. [`p7_2x2_univariate_multivariate.png`](../outputs/figures/p7_2x2_univariate_multivariate.png) — *2x2 Experimental Paradigm Grid*
10. [`p7_master_benchmark_bar.png`](../outputs/figures/p7_master_benchmark_bar.png) — *Master Test Benchmark (RMSE & MAE)*
11. [`p7_diebold_mariano_heatmap.png`](../outputs/figures/p7_diebold_mariano_heatmap.png) — *Diebold-Mariano Statistical Significance Matrix*
12. [`p7_alert_precision_recall.png`](../outputs/figures/p7_alert_precision_recall.png) — *Operational Alert Precision, Recall, and F1*
13. [`p7_spike_vs_normal_error.png`](../outputs/figures/p7_spike_vs_normal_error.png) — *Error Degradation on Extreme Spikes*
14. [`p7_rmse_vs_horizon_final.png`](../outputs/figures/p7_rmse_vs_horizon_final.png) — *Multi-Horizon Forecast Accuracy (1 to 7 Days)*
15. [`p7_interval_comparison.png`](../outputs/figures/p7_interval_comparison.png) — *Split-Conformal vs MC-Dropout Prediction Intervals*

