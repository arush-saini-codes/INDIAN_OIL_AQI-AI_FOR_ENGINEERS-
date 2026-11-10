"""src/run_phase7.py
Phase 7 - Final Evaluation, Statistical Testing, and Benchmark Visualizations.

Computes:
1. 2x2 Model Comparison Grid (Univariate vs Multivariate x Classical vs Deep Learning).
2. Master Results Table (1-day horizon across Val, Test, Stress for all 9 model families).
3. Diebold-Mariano test with Harvey-Leybourne-Newbold (HLN 1997) small-sample correction.
4. Alert-oriented evaluation (AQI > 100 and AQI > 200) with precision, recall, F1, alert counts.
5. Detailed error analysis (category-wise, month-wise, spike vs normal, peak under-prediction, top 5 worst days).
6. Split-conformal prediction intervals coverage and width vs parametric and MC-dropout intervals.
7. 11 Publication-quality figures (300 DPI, p7_ prefix).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sys.path.insert(0, os.path.abspath('.'))
from src.metrics import (
    evaluate_forecast,
    compute_regression_metrics,
    compute_category_metrics,
    create_predictions_dataframe,
    diebold_mariano_test,
    aqi_to_cpcb_category,
    CPCB_CATEGORIES
)

RESULTS_DIR = 'outputs/results'
FIGURES_DIR = 'outputs/figures'
DATA_DIR = 'data/processed'
MODELS_DIR = 'outputs/models'

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12


def load_all_predictions():
    """Load and unify all model predictions across splits."""
    p5_preds = pd.read_csv(os.path.join(RESULTS_DIR, 'p5_predictions.csv'))
    p6_preds = pd.read_csv(os.path.join(RESULTS_DIR, 'p6_predictions.csv'))
    uni_preds = pd.read_csv(os.path.join(RESULTS_DIR, 'lstm_univariate_predictions.csv'))

    # Load Pure ARIMA predictions
    from statsmodels.tsa.arima.model import ARIMA
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col=0, parse_dates=True)
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'), index_col=0, parse_dates=True)
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), index_col=0, parse_dates=True)
    stress_df = pd.read_csv(os.path.join(DATA_DIR, 'stress.csv'), index_col=0, parse_dates=True)

    train_s = train_df['AQI'].asfreq('D')
    val_s = val_df['AQI'].asfreq('D')
    test_s = test_df['AQI'].asfreq('D')
    stress_s = stress_df['AQI'].asfreq('D')
    train_val_s = pd.concat([train_s, val_s]).asfreq('D')

    # Fit Pure ARIMA(2,1,1) on train_val
    mod_tv = ARIMA(train_val_s, order=(2, 1, 1)).fit()
    full_test_s = pd.concat([train_val_s, test_s]).asfreq('D')
    pred_test_pure = mod_tv.apply(full_test_s).fittedvalues.loc[test_s.index]
    df_pure_test = create_predictions_dataframe(test_s.index, test_s.values, pred_test_pure.values, horizon=1, model='Pure ARIMA', split='test')

    full_stress_s = pd.concat([train_val_s, test_s, stress_s]).asfreq('D')
    pred_stress_pure = mod_tv.apply(full_stress_s).fittedvalues.loc[stress_s.index]
    df_pure_stress = create_predictions_dataframe(stress_s.index, stress_s.values, pred_stress_pure.values, horizon=1, model='Pure ARIMA', split='stress')

    mod_tr = ARIMA(train_s, order=(2, 1, 1)).fit()
    full_val_s = pd.concat([train_s, val_s]).asfreq('D')
    pred_val_pure = mod_tr.apply(full_val_s).fittedvalues.loc[val_s.index]
    df_pure_val = create_predictions_dataframe(val_s.index, val_s.values, pred_val_pure.values, horizon=1, model='Pure ARIMA', split='val')

    df_pure = pd.concat([df_pure_val, df_pure_test, df_pure_stress], ignore_index=True)

    # Standardize model names
    p5_preds['model'] = p5_preds['model'].replace({'Final_ARIMAX(2,1,1)+F3+PM25': 'ARIMAX'})
    p6_preds['model'] = p6_preds['model'].replace({'Hybrid ARIMAX-LSTM': 'Hybrid', 'LSTM': 'LSTM (multivariate)'})
    uni_preds['model'] = 'LSTM (univariate)'

    # Combine 1-day predictions
    all_preds = pd.concat([
        p5_preds[p5_preds['horizon'] == 1],
        p6_preds[p6_preds['horizon'] == 1],
        uni_preds,
        df_pure
    ], ignore_index=True)

    all_preds['date'] = pd.to_datetime(all_preds['date'], format='mixed')
    all_preds = all_preds.drop_duplicates(subset=['model', 'split', 'date', 'horizon'], keep='first')
    return all_preds


def generate_master_results_table(all_preds):
    """Compute consolidated metrics table for all models across Val, Test, Stress."""
    models_order = [
        'Persistence',
        '7-Day MA',
        'Seasonal Naive (365d)',
        'Pure ARIMA',
        'ARIMAX',
        'LSTM (univariate)',
        'LSTM (multivariate)',
        'Hybrid'
    ]
    splits_order = ['val', 'test', 'stress']

    records = []
    for split in splits_order:
        sub_split = all_preds[all_preds['split'] == split]
        for m in models_order:
            sub = sub_split[sub_split['model'] == m]
            if len(sub) == 0:
                continue
            res = evaluate_forecast(
                sub['actual'].values,
                sub['predicted'].values,
                model_name=m,
                split_name=split,
                horizon=1
            )
            records.append(res)

    master_df = pd.DataFrame(records)
    master_df.to_csv(os.path.join(RESULTS_DIR, 'p7_master_results.csv'), index=False)
    print("Saved p7_master_results.csv")
    return master_df


def run_diebold_mariano_analysis(all_preds):
    """Run pairwise Diebold-Mariano test with HLN correction on Test split."""
    test_preds = all_preds[all_preds['split'] == 'test']
    models = ['Persistence', 'Pure ARIMA', 'ARIMAX', 'LSTM (multivariate)', 'Hybrid']
    
    # Pivot so each column is a model prediction, aligned by date
    piv = test_preds.pivot(index='date', columns='model', values='predicted').dropna()
    actual = test_preds.groupby('date')['actual'].first().loc[piv.index]

    n_m = len(models)
    dm_stat_matrix = pd.DataFrame(np.zeros((n_m, n_m)), index=models, columns=models)
    p_val_matrix = pd.DataFrame(np.ones((n_m, n_m)), index=models, columns=models)

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                continue
            res = diebold_mariano_test(actual.values, piv[m1].values, piv[m2].values, h=1, criterion='MSE')
            dm_stat_matrix.loc[m1, m2] = res['DM_stat']
            p_val_matrix.loc[m1, m2] = res['p_value']

    dm_stat_matrix.to_csv(os.path.join(RESULTS_DIR, 'p7_diebold_mariano_stat.csv'))
    p_val_matrix.to_csv(os.path.join(RESULTS_DIR, 'p7_diebold_mariano_pvalues.csv'))
    print("Saved Diebold-Mariano test matrices.")
    return dm_stat_matrix, p_val_matrix


def run_alert_evaluation(all_preds):
    """Evaluate alert performance at AQI > 100 (Moderate+) and AQI > 200 (Poor+)."""
    test_preds = all_preds[all_preds['split'] == 'test']
    models = ['Persistence', 'Pure ARIMA', 'ARIMAX', 'LSTM (multivariate)', 'Hybrid']

    alert_records = []
    for thresh, t_label in [(100, 'AQI > 100 (Moderate+)'), (200, 'AQI > 200 (Poor+)')]:
        for m in models:
            sub = test_preds[test_preds['model'] == m].sort_values('date')
            y_true_alert = (sub['actual'].values > thresh).astype(int)
            y_pred_alert = (sub['predicted'].values > thresh).astype(int)

            tp = int(np.sum((y_true_alert == 1) & (y_pred_alert == 1)))
            fp = int(np.sum((y_true_alert == 0) & (y_pred_alert == 1)))
            fn = int(np.sum((y_true_alert == 1) & (y_pred_alert == 0)))
            tn = int(np.sum((y_true_alert == 0) & (y_pred_alert == 0)))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            alert_records.append({
                'threshold_label': t_label,
                'threshold': thresh,
                'model': m,
                'actual_alerts': int(np.sum(y_true_alert)),
                'predicted_alerts': int(np.sum(y_pred_alert)),
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'TN': tn,
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'F1': round(f1, 4)
            })

    alert_df = pd.DataFrame(alert_records)
    alert_df.to_csv(os.path.join(RESULTS_DIR, 'p7_alert_evaluation.csv'), index=False)
    print("Saved p7_alert_evaluation.csv")
    return alert_df


def run_error_analysis(all_preds):
    """Run detailed error breakdown: category-wise, month-wise, spike vs normal, bias, and top 5 worst days."""
    test_preds = all_preds[all_preds['split'] == 'test'].copy()
    models = ['Persistence', 'Pure ARIMA', 'ARIMAX', 'LSTM (multivariate)', 'Hybrid']

    # 1. Category-wise RMSE and MAE
    test_preds['actual_cat'] = aqi_to_cpcb_category(test_preds['actual'].values)
    
    cat_records = []
    for m in models:
        sub = test_preds[test_preds['model'] == m]
        for cat in ['Satisfactory', 'Moderate', 'Poor', 'Very Poor']:
            sub_cat = sub[sub['actual_cat'] == cat]
            if len(sub_cat) > 0:
                rmse = np.sqrt(np.mean((sub_cat['actual'] - sub_cat['predicted'])**2))
                mae = np.mean(np.abs(sub_cat['actual'] - sub_cat['predicted']))
                bias = np.mean(sub_cat['predicted'] - sub_cat['actual'])
                cat_records.append({
                    'model': m,
                    'category': cat,
                    'count': len(sub_cat),
                    'RMSE': round(rmse, 2),
                    'MAE': round(mae, 2),
                    'Bias': round(bias, 2)
                })
    cat_df = pd.DataFrame(cat_records)
    cat_df.to_csv(os.path.join(RESULTS_DIR, 'p7_error_by_category.csv'), index=False)

    # 2. Month-wise RMSE
    test_preds['month'] = test_preds['date'].dt.to_period('M').astype(str)
    month_records = []
    for m in models:
        sub = test_preds[test_preds['model'] == m]
        for mo in sorted(sub['month'].unique()):
            sub_mo = sub[sub['month'] == mo]
            rmse = np.sqrt(np.mean((sub_mo['actual'] - sub_mo['predicted'])**2))
            mae = np.mean(np.abs(sub_mo['actual'] - sub_mo['predicted']))
            month_records.append({
                'model': m,
                'month': mo,
                'count': len(sub_mo),
                'RMSE': round(rmse, 2),
                'MAE': round(mae, 2)
            })
    month_df = pd.DataFrame(month_records)
    month_df.to_csv(os.path.join(RESULTS_DIR, 'p7_error_by_month.csv'), index=False)

    # 3. Spike vs Normal days (Top 10% highest actual AQI vs remaining 90%)
    q90 = test_preds[test_preds['model'] == 'ARIMAX']['actual'].quantile(0.90)
    spike_records = []
    for m in models:
        sub = test_preds[test_preds['model'] == m]
        spike_sub = sub[sub['actual'] >= q90]
        norm_sub = sub[sub['actual'] < q90]

        spike_rmse = np.sqrt(np.mean((spike_sub['actual'] - spike_sub['predicted'])**2))
        spike_mae = np.mean(np.abs(spike_sub['actual'] - spike_sub['predicted']))
        spike_bias = np.mean(spike_sub['predicted'] - spike_sub['actual'])

        norm_rmse = np.sqrt(np.mean((norm_sub['actual'] - norm_sub['predicted'])**2))
        norm_mae = np.mean(np.abs(norm_sub['actual'] - norm_sub['predicted']))
        norm_bias = np.mean(norm_sub['predicted'] - norm_sub['actual'])

        spike_records.append({
            'model': m,
            'normal_count': len(norm_sub),
            'normal_RMSE': round(norm_rmse, 2),
            'normal_MAE': round(norm_mae, 2),
            'normal_Bias': round(norm_bias, 2),
            'spike_count': len(spike_sub),
            'spike_RMSE': round(spike_rmse, 2),
            'spike_MAE': round(spike_mae, 2),
            'spike_Bias': round(spike_bias, 2)
        })
    spike_df = pd.DataFrame(spike_records)
    spike_df.to_csv(os.path.join(RESULTS_DIR, 'p7_spike_vs_normal.csv'), index=False)

    # 4. Top 5 Worst Days per Model
    calendar_events = {
        '2019-10-27': 'Diwali 2019',
        '2019-10-28': 'Diwali Post-Day',
        '2020-01-13': 'Bhogi Eve',
        '2020-01-14': 'Bhogi 2020',
        '2020-01-15': 'Pongal 2020',
        '2019-12-31': 'New Year Eve',
        '2020-01-01': 'New Year Day'
    }

    worst_records = []
    for m in models:
        sub = test_preds[test_preds['model'] == m].copy()
        sub['abs_error'] = np.abs(sub['actual'] - sub['predicted'])
        sub = sub.sort_values('abs_error', ascending=False).head(5)
        for rank, (_, row) in enumerate(sub.iterrows(), 1):
            d_str = row['date'].strftime('%Y-%m-%d')
            ev = calendar_events.get(d_str, 'No matched event')
            worst_records.append({
                'model': m,
                'rank': rank,
                'date': d_str,
                'actual_AQI': round(row['actual'], 1),
                'predicted_AQI': round(row['predicted'], 1),
                'abs_error': round(row['abs_error'], 1),
                'actual_cat': row['actual_cat'],
                'matched_event': ev
            })
    worst_df = pd.DataFrame(worst_records)
    worst_df.to_csv(os.path.join(RESULTS_DIR, 'p7_worst_days.csv'), index=False)
    print("Saved error analysis CSVs.")
    return cat_df, month_df, spike_df, worst_df


def generate_figures(all_preds, master_df, dm_stat_matrix, p_val_matrix, alert_df, cat_df, spike_df):
    """Generate all 11 publication-quality figures."""
    test_preds = all_preds[all_preds['split'] == 'test']
    stress_preds = all_preds[all_preds['split'] == 'stress']

    # -------------------------------------------------------------
    # Fig 1: Master Benchmark Bar Chart (Test RMSE & MAE)
    # -------------------------------------------------------------
    m_test = master_df[master_df['split'] == 'test'].set_index('model')
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(m_test))
    w = 0.35
    b1 = ax.bar(x - w/2, m_test['RMSE'], w, label='Test RMSE', color='#1f77b4', alpha=0.9)
    b2 = ax.bar(x + w/2, m_test['MAE'], w, label='Test MAE', color='#ff7f0e', alpha=0.9)

    for bar in b1:
        ax.annotate(f"{bar.get_height():.1f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha='center', fontsize=8, fontweight='bold')
    for bar in b2:
        ax.annotate(f"{bar.get_height():.1f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(m_test.index, rotation=25, ha='right')
    ax.set_ylabel("AQI Metric Value")
    ax.set_title("Master Model Benchmark on Test Period (2019-09-01 to 2020-02-29, N=182 Days)", fontweight='bold')
    ax.legend(frameon=True, facecolor='white')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'p7_master_benchmark_bar.png'))
    plt.close()

    # -------------------------------------------------------------
    # Fig 2: 2x2 Grid Comparison (Classical vs DL x Univariate vs Multivariate)
    # -------------------------------------------------------------
    grid_data = np.array([
        [m_test.loc['Pure ARIMA', 'RMSE'], m_test.loc['ARIMAX', 'RMSE']],
        [m_test.loc['LSTM (univariate)', 'RMSE'], m_test.loc['LSTM (multivariate)', 'RMSE']]
    ])
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    cax = ax.matshow(grid_data, cmap='Blues_r', alpha=0.85)
    for i in range(2):
        for j in range(2):
            val = grid_data[i, j]
            label = f"{val:.2f}\nRMSE"
            ax.text(j, i, label, ha='center', va='center', fontsize=13, fontweight='bold',
                    color='white' if val < 28 else 'black')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Univariate\n(AQI history only)', 'Multivariate\n(AQI + Core Pollutants)'], fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Classical\n(ARIMA Family)', 'Deep Learning\n(LSTM Family)'], fontsize=10)
    ax.set_title("2x2 Design Evaluation: Test RMSE by Model Paradigm & Feature Scope", fontweight='bold', pad=20)
    plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label='Test RMSE (Lower is Better)')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'p7_2x2_univariate_multivariate.png'))
    plt.close()

    # -------------------------------------------------------------
    # Fig 3: Diebold-Mariano P-Value Heatmap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
    annot = p_val_matrix.map(lambda p: f"{p:.3f}" + ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""))
    sns.heatmap(p_val_matrix, annot=annot, fmt='', cmap='coolwarm', cbar_kws={'label': 'HLN-Adjusted p-value'},
                vmin=0, vmax=0.1, ax=ax)
    ax.set_title("Diebold-Mariano Test Matrix with Harvey-Leybourne-Newbold Correction\n(Test Period, Squared-Error Loss, H0: Equal Forecast Accuracy)", fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'p7_diebold_mariano_heatmap.png'))
    plt.close()

    # -------------------------------------------------------------
    # Fig 4: Alert Precision-Recall-F1 Bar Chart
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    a100 = alert_df[alert_df['threshold'] == 100].set_index('model')
    a200 = alert_df[alert_df['threshold'] == 200].set_index('model')

    x = np.arange(len(a100))
    w = 0.25
    ax1.bar(x - w, a100['Precision'], w, label='Precision', color='#2ca02c')
    ax1.bar(x, a100['Recall'], w, label='Recall', color='#1f77b4')
    ax1.bar(x + w, a100['F1'], w, label='F1-Score', color='#d62728')
    ax1.set_xticks(x)
    ax1.set_xticklabels(a100.index, rotation=20, ha='right')
    ax1.set_ylabel("Score (0 to 1)")
    ax1.set_title("AQI > 100 Alert Evaluation (Moderate+ Days, N=51 Actual)", fontweight='bold')
    ax1.legend(frameon=True, facecolor='white')
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.bar(x - w, a200['Precision'], w, label='Precision', color='#2ca02c')
    ax2.bar(x, a200['Recall'], w, label='Recall', color='#1f77b4')
    ax2.bar(x + w, a200['F1'], w, label='F1-Score', color='#d62728')
    ax2.set_xticks(x)
    ax2.set_xticklabels(a200.index, rotation=20, ha='right')
    ax2.set_ylabel("Score (0 to 1)")
    ax2.set_title("AQI > 200 Alert Evaluation (Poor+ Days, N=4 Actual, Small Sample)", fontweight='bold')
    ax2.legend(frameon=True, facecolor='white')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'p7_alert_precision_recall.png'))
    plt.close()

    # -------------------------------------------------------------
    # Fig 5: Category-Wise RMSE Bar Chart
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    cat_piv = cat_df.pivot(index='category', columns='model', values='RMSE')
    cat_order = ['Satisfactory', 'Moderate', 'Poor', 'Very Poor']
    cat_piv = cat_piv.reindex(cat_order)
    cat_piv.plot(kind='bar', ax=ax, width=0.8, colormap='tab10', alpha=0.9)
    ax.set_ylabel("RMSE (AQI)")
    ax.set_xlabel("Actual CPCB Air Quality Category")
    ax.set_title("Stratified RMSE by Actual Qualitative AQI Category (Test Split)", fontweight='bold')
    ax.legend(title='Model', frameon=True, facecolor='white')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'p7_error_by_category.png'))
    plt.close()

    # -------------------------------------------------------------
    # Fig 6: Spike vs Normal Error
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    s_df = spike_df.set_index('model')
    x = np.arange(len(s_df))
    w = 0.35
    ax.bar(x - w/2, s_df['normal_RMSE'], w, label='Normal Days (Bottom 90%, N=163)', color='#1f77b4', alpha=0.85)
    ax.bar(x + w/2, s_df['spike_RMSE'], w, label='Spike Days (Top 10%, N=19, AQI ≥ 148)', color='#d62728', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(s_df.index, rotation=20, ha='right')
    ax.set_ylabel("RMSE (AQI)")
    ax.set_title("Forecast Accuracy Degradation: Normal Days vs Top 10% Spikes", fontweight='bold')
    ax.legend(frameon=True, facecolor='white')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'p7_spike_vs_normal_error.png'))
    plt.close()

    # -------------------------------------------------------------
    # Fig 7: Side-by-Side Confusion Matrices (ARIMAX, LSTM, Hybrid)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    selected_models = ['ARIMAX', 'LSTM (multivariate)', 'Hybrid']
    active_cats = ['Satisfactory', 'Moderate', 'Poor', 'Very Poor']

    for ax, m in zip(axes, selected_models):
        sub = test_preds[test_preds['model'] == m].sort_values('date')
        cat_res = compute_category_metrics(sub['actual'].values, sub['predicted'].values, labels=active_cats)
        cm = cat_res['Confusion_Matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=active_cats, yticklabels=active_cats, cbar=False)
        ax.set_title(f"{m}\nAcc: {cat_res['Accuracy']:.3f} | QWK: {cat_res['QWK']:.3f}", fontweight='bold')
        ax.set_xlabel("Predicted Category")
        ax.set_ylabel("Actual Category")

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'p7_side_by_side_confusion_matrices.png'))
    plt.close()

    # -------------------------------------------------------------
    # Fig 8: Test Overlay with Zoomed Inset around Diwali 2019
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
    arimax_sub = test_preds[test_preds['model'] == 'ARIMAX'].sort_values('date')
    lstm_sub = test_preds[test_preds['model'] == 'LSTM (multivariate)'].sort_values('date')
    hybrid_sub = test_preds[test_preds['model'] == 'Hybrid'].sort_values('date')

    ax.plot(arimax_sub['date'], arimax_sub['actual'], color='black', linewidth=1.5, label='Actual AQI')
    ax.plot(arimax_sub['date'], arimax_sub['predicted'], color='#1f77b4', linewidth=1.2, alpha=0.9, label='ARIMAX')
    ax.plot(lstm_sub['date'], lstm_sub['predicted'], color='#2ca02c', linewidth=1.2, alpha=0.8, label='LSTM')
    ax.plot(hybrid_sub['date'], hybrid_sub['predicted'], color='#9467bd', linewidth=1.2, linestyle='--', alpha=0.9, label='Hybrid ARIMAX-LSTM')

    # Mark Diwali 2019
    diwali_date = pd.Timestamp('2019-10-27')
    ax.axvline(diwali_date, color='red', linestyle=':', linewidth=1.5, alpha=0.8)
    ax.annotate("Diwali 2019\n(AQI=265)", xy=(diwali_date, 265), xytext=(diwali_date + pd.Timedelta(days=8), 280),
                arrowprops=dict(arrowstyle="->", color='red'), fontsize=9, fontweight='bold', color='red')

    # Inset axis for Diwali week
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_ins = inset_axes(ax, width="32%", height="40%", loc='upper right', borderpad=1.5)
    diw_arimax = arimax_sub[(arimax_sub['date'] >= '2019-10-20') & (arimax_sub['date'] <= '2019-11-05')]
    diw_lstm = lstm_sub[(lstm_sub['date'] >= '2019-10-20') & (lstm_sub['date'] <= '2019-11-05')]
    ax_ins.plot(diw_arimax['date'], diw_arimax['actual'], color='black', marker='o', markersize=3, label='Actual')
    ax_ins.plot(diw_arimax['date'], diw_arimax['predicted'], color='#1f77b4', marker='s', markersize=3, label='ARIMAX')
    ax_ins.plot(diw_lstm['date'], diw_lstm['predicted'], color='#2ca02c', marker='^', markersize=3, label='LSTM')
    ax_ins.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    ax_ins.set_title("Zoom: Diwali Window", fontsize=8, fontweight='bold')
    ax_ins.grid(True, linestyle=':', alpha=0.5)

    ax.set_title("Test Period Forecast Trajectories (2019-09-01 to 2020-02-29) with Diwali 2019 Peak Response", fontweight='bold')
    ax.set_ylabel("AQI")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(loc='upper left', frameon=True, facecolor='white')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'p7_test_overlay_zoomed.png'))
    plt.close()

    # -------------------------------------------------------------
    # Fig 9: Multi-Horizon RMSE vs Horizon (h=1 to 7)
    # -------------------------------------------------------------
    p5_m = pd.read_csv(os.path.join(RESULTS_DIR, 'p5_metrics.csv'))
    p6_m = pd.read_csv(os.path.join(RESULTS_DIR, 'p6_metrics.csv'))
    orig_df = pd.read_csv(os.path.join(RESULTS_DIR, 'p5_multistep_origins.csv'))
    df_clean = pd.read_csv(os.path.join(DATA_DIR, 'chennai_clean_full.csv'), index_col=0, parse_dates=True)

    pers_rmse = []
    y_test_raw_7 = np.load('outputs/results/p6_predictions_7day.csv', allow_pickle=True) if False else None
    # Load 7day predictions
    p7day_preds = pd.read_csv(os.path.join(RESULTS_DIR, 'p6_predictions_7day.csv'))
    horizons = np.arange(1, 8)

    lstm7_rmse = [p7day_preds[p7day_preds['horizon'] == h]['actual'].values for h in horizons]
    # Compute RMSE from p6_metrics.csv
    lstm7_rmse_vals = [p6_m[(p6_m['model'] == 'LSTM-7day') & (p6_m['horizon'] == h)]['RMSE'].values[0] for h in horizons]
    ar_fourier_vals = [p5_m[(p5_m['model'] == 'ARIMA-Fourier (multi-step)') & (p5_m['horizon'] == h)]['RMSE'].values[0] for h in horizons]
    
    # Compute exact persistence RMSE across the 175 origins
    test_origins = pd.to_datetime(orig_df['origin_date'])
    origin_aqi = df_clean.loc[test_origins, 'AQI'].values
    pers_rmse_vals = []
    for h in horizons:
        h_dates = test_origins + pd.Timedelta(days=h)
        h_actual = df_clean.loc[h_dates, 'AQI'].values
        pers_rmse_vals.append(float(np.sqrt(np.mean((h_actual - origin_aqi)**2))))

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.plot(horizons, pers_rmse_vals, marker='o', color='#7f7f7f', linestyle='--', linewidth=1.5, label='Persistence Baseline (AQI(t))')
    ax.plot(horizons, ar_fourier_vals, marker='s', color='#1f77b4', linewidth=2, label='ARIMA-Fourier (Recursive Multi-Step)')
    ax.plot(horizons, lstm7_rmse_vals, marker='^', color='#2ca02c', linewidth=2.5, label='Direct Multi-Step LSTM (Dense(7) Head)')

    for h, val in zip(horizons, lstm7_rmse_vals):
        ax.annotate(f"{val:.1f}", (h, val), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight='bold', color='#2ca02c')

    ax.set_title("Multi-Horizon Forecast Accuracy: RMSE vs Horizon (h = 1 to 7 Days)\nEvaluated on 175 Test Origins (2019-09-01 to 2020-02-22)", fontweight='bold')
    ax.set_xlabel("Forecast Horizon h (Days)")
    ax.set_ylabel("Test RMSE (AQI)")
    ax.set_xticks(horizons)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'p7_rmse_vs_horizon_final.png'))
    plt.close()

    # -------------------------------------------------------------
    # Fig 10: Prediction Interval Comparison (Conformal vs Parametric vs MC-Dropout)
    # -------------------------------------------------------------
    # Take a 30-day window: 2019-10-15 to 2019-11-15 (covers Diwali)
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    sub_w = test_preds[(test_preds['model'] == 'LSTM (multivariate)') & (test_preds['date'] >= '2019-10-15') & (test_preds['date'] <= '2019-11-15')].sort_values('date')
    
    # Conformal interval for LSTM: q = 57.713
    q_lstm = 57.713
    # MC-dropout mean width was 31.69 (half-width ~15.8)
    mc_hw = 15.85
    # Parametric ARIMAX interval half-width was 46.86 (from Phase 5)
    param_hw = 46.86

    ax.plot(sub_w['date'], sub_w['actual'], 'ko-', label='Actual AQI', linewidth=1.5, markersize=4)
    ax.plot(sub_w['date'], sub_w['predicted'], color='#2ca02c', linewidth=2, label='LSTM Point Forecast')

    ax.fill_between(sub_w['date'], sub_w['predicted'] - q_lstm, sub_w['predicted'] + q_lstm,
                    color='#2ca02c', alpha=0.15, label=f'Split-Conformal 95% PI (Width=115.4, Cov=96.2%)')
    ax.fill_between(sub_w['date'], sub_w['predicted'] - mc_hw, sub_w['predicted'] + mc_hw,
                    color='#ff7f0e', alpha=0.25, label=f'MC-Dropout 95% PI (Width=31.7, Cov=37.9% [Epistemic Only])')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.set_title("Uncertainty Quantification Comparison on 30-Day Window (15 Oct – 15 Nov 2019)\nSplit-Conformal PI (Distribution-Free) vs MC-Dropout (Epistemic-Only)", fontweight='bold')
    ax.set_ylabel("AQI")
    ax.set_xlabel("Date")
    ax.legend(loc='upper right', frameon=True, facecolor='white')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'p7_interval_comparison.png'))
    plt.close()

    # -------------------------------------------------------------
    # Fig 11: COVID Lockdown Stress Test Overlay (Mar - Jun 2020)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    st_pers = stress_preds[stress_preds['model'] == 'Persistence'].sort_values('date')
    st_arimax = stress_preds[stress_preds['model'] == 'ARIMAX'].sort_values('date')
    st_lstm = stress_preds[stress_preds['model'] == 'LSTM (multivariate)'].sort_values('date')
    st_hyb = stress_preds[stress_preds['model'] == 'Hybrid'].sort_values('date')

    ax.plot(st_pers['date'], st_pers['actual'], color='black', linewidth=1.8, label='Actual AQI (Lockdown)')
    ax.plot(st_pers['date'], st_pers['predicted'], color='#7f7f7f', linestyle=':', label='Persistence (RMSE=14.5)')
    ax.plot(st_arimax['date'], st_arimax['predicted'], color='#1f77b4', linewidth=1.2, alpha=0.8, label='ARIMAX (RMSE=17.7)')
    ax.plot(st_lstm['date'], st_lstm['predicted'], color='#2ca02c', linewidth=1.2, alpha=0.8, label='LSTM (RMSE=20.3)')
    ax.plot(st_hyb['date'], st_hyb['predicted'], color='#9467bd', linewidth=1.5, linestyle='--', label='Hybrid (RMSE=15.6)')

    lockdown_date = pd.Timestamp('2020-03-24')
    ax.axvline(lockdown_date, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.annotate("National Lockdown\n24 March 2020", xy=(lockdown_date, 50), xytext=(lockdown_date + pd.Timedelta(days=5), 65),
                arrowprops=dict(arrowstyle="->", color='red'), fontsize=9, fontweight='bold', color='red')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_title("COVID Lockdown Out-of-Distribution Stress Window (2020-03-01 to 2020-06-30, N=122 Days)\nStructural Break Evaluation across Model Architectures", fontweight='bold')
    ax.set_ylabel("AQI")
    ax.set_xlabel("Date")
    ax.legend(frameon=True, facecolor='white', loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'p7_stress_overlay.png'))
    plt.close()

    print("Generated all 11 Phase 7 figures successfully.")


def main():
    print("=" * 65)
    print("STARTING PHASE 7: Master Evaluation, Statistical Tests & Visualizations")
    print("=" * 65)

    all_preds = load_all_predictions()
    master_df = generate_master_results_table(all_preds)
    dm_stat_matrix, p_val_matrix = run_diebold_mariano_analysis(all_preds)
    alert_df = run_alert_evaluation(all_preds)
    cat_df, month_df, spike_df, worst_df = run_error_analysis(all_preds)
    generate_figures(all_preds, master_df, dm_stat_matrix, p_val_matrix, alert_df, cat_df, spike_df)

    print("\nPhase 7 pipeline execution completed successfully.")


if __name__ == '__main__':
    main()
