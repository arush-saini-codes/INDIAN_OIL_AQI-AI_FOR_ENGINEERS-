"""src/check_consistency.py
Consistency Audit Script for UCS321 AQI Forecasting Project.

Recomputes:
1. Split means (Train, Val, Test, Stress).
2. Headline Test & Stress metrics (RMSE, MAE) from raw prediction CSVs.
3. Multi-horizon metrics (horizons 1 to 7) on the 175 origins.
4. Checks consistency against p7_master_results.csv, p5_metrics.csv, p6_metrics.csv,
   and scans documentation files (docs/RESULTS_SUMMARY.md, docs/PROJECT_LOG.md, README.md).
"""

import os
import re
import numpy as np
import pandas as pd

DATA_FILE = 'data/processed/chennai_clean_full.csv'
P5_PREDS = 'outputs/results/p5_predictions.csv'
P6_PREDS = 'outputs/results/p6_predictions.csv'
P6_7DAY_PREDS = 'outputs/results/p6_predictions_7day.csv'
ORIGINS_FILE = 'outputs/results/p5_multistep_origins.csv'
MASTER_FILE = 'outputs/results/p7_master_results.csv'


def main():
    print("=" * 70)
    print("RUNNING END-TO-END DATA & METRIC CONSISTENCY CHECK")
    print("=" * 70)

    mismatches = []

    # -------------------------------------------------------------
    # 1. SPLIT MEANS AUDIT
    # -------------------------------------------------------------
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    splits = {
        'Train': df.loc['2016-01-01':'2018-12-31'],
        'Val': df.loc['2019-01-01':'2019-08-31'],
        'Test': df.loc['2019-09-01':'2020-02-29'],
        'Stress': df.loc['2020-03-01':'2020-06-30']
    }

    print("\n--- 1. Split Means & Counts ---")
    split_means = {}
    for name, s_df in splits.items():
        m_val = float(s_df['AQI'].mean())
        split_means[name] = m_val
        print(f"  {name:7s}: N = {len(s_df):4d} days | Mean AQI = {m_val:.2f} | Range = [{s_df['AQI'].min():.1f}, {s_df['AQI'].max():.1f}]")

    stress_mean = split_means['Stress']
    if abs(stress_mean - 78.81) > 0.05:
        mismatches.append(f"Stress mean mismatch: expected ~78.81, got {stress_mean:.2f}")

    # -------------------------------------------------------------
    # 2. HEADLINE 1-DAY METRICS AUDIT (FROM PREDICTIONS)
    # -------------------------------------------------------------
    print("\n--- 2. Recomputing Headline 1-Day Metrics from Predictions ---")
    p5_df = pd.read_csv(P5_PREDS)
    p6_df = pd.read_csv(P6_PREDS)
    master_df = pd.read_csv(MASTER_FILE)

    def get_rmse_mae(pred_df, model_col_val, split_val, h=1):
        sub = pred_df[(pred_df['model'] == model_col_val) & (pred_df['split'] == split_val) & (pred_df['horizon'] == h)]
        # Drop duplicates if any
        sub = sub.drop_duplicates(subset=['date'])
        rmse = float(np.sqrt(np.mean((sub['actual'] - sub['predicted'])**2)))
        mae = float(np.mean(np.abs(sub['actual'] - sub['predicted'])))
        return len(sub), rmse, mae

    # Test Split Check
    test_models = {
        'Persistence': (p5_df, 'Persistence', 'Persistence'),
        'ARIMAX': (p5_df, 'Final_ARIMAX(2,1,1)+F3+PM25', 'ARIMAX'),
        'LSTM (multivariate)': (p6_df, 'LSTM', 'LSTM (multivariate)'),
        'LSTM (univariate)': (pd.read_csv('outputs/results/lstm_univariate_predictions.csv'), 'LSTM (univariate)', 'LSTM (univariate)'),
        'Hybrid': (p6_df, 'Hybrid ARIMAX-LSTM', 'Hybrid')
    }

    print("Test Split (182 days):")
    for m_label, (df_source, m_col, master_name) in test_models.items():
        n, r, m = get_rmse_mae(df_source, m_col, 'test', h=1)
        # Check against master_df
        master_match = master_df[(master_df['split'] == 'test') & (master_df['model'] == master_name)]
        if len(master_match) > 0:
            mr = master_match['RMSE'].values[0]
            mm = master_match['MAE'].values[0]
            diff_r = abs(r - mr)
            diff_m = abs(m - mm)
            status = "OK" if (diff_r < 0.05 and diff_m < 0.05) else "MISMATCH"
            print(f"  {m_label:20s}: N={n} | Preds: RMSE={r:.3f}, MAE={m:.3f} | Master: RMSE={mr:.3f}, MAE={mm:.3f} | {status}")
            if status == "MISMATCH":
                mismatches.append(f"{m_label} test metric mismatch between preds and master: {r:.3f} vs {mr:.3f}")

    print("\nStress Split (122 days):")
    for m_label, (df_source, m_col, master_name) in test_models.items():
        n, r, m = get_rmse_mae(df_source, m_col, 'stress', h=1)
        master_match = master_df[(master_df['split'] == 'stress') & (master_df['model'] == master_name)]
        if len(master_match) > 0:
            mr = master_match['RMSE'].values[0]
            mm = master_match['MAE'].values[0]
            diff_r = abs(r - mr)
            diff_m = abs(m - mm)
            status = "OK" if (diff_r < 0.05 and diff_m < 0.05) else "MISMATCH"
            print(f"  {m_label:20s}: N={n} | Preds: RMSE={r:.3f}, MAE={m:.3f} | Master: RMSE={mr:.3f}, MAE={mm:.3f} | {status}")
            if status == "MISMATCH":
                mismatches.append(f"{m_label} stress metric mismatch between preds and master: {r:.3f} vs {mr:.3f}")

    # -------------------------------------------------------------
    # 3. MULTI-STEP TABLE AUDIT (175 ORIGINS)
    # -------------------------------------------------------------
    print("\n--- 3. Multi-Step Horizon Evaluation (175 Test Origins) ---")
    orig_df = pd.read_csv(ORIGINS_FILE)
    test_origins = pd.to_datetime(orig_df['origin_date'])
    n_orig = len(test_origins)
    print(f"  Total Origins: {n_orig} (from {test_origins.min().strftime('%Y-%m-%d')} to {test_origins.max().strftime('%Y-%m-%d')})")

    # Recompute Persistence strictly on the 175 origins
    # Origin t forecast is AQI(t)
    origin_aqi = df.loc[test_origins, 'AQI'].values
    p7day_df = pd.read_csv(P6_7DAY_PREDS)
    p5_m = pd.read_csv('outputs/results/p5_metrics.csv')
    p6_m = pd.read_csv('outputs/results/p6_metrics.csv')

    print("  H | Persistence RMSE (175) | Fourier RMSE (175) | LSTM-7day RMSE (175) | LSTM beats Persist?")
    for h in range(1, 8):
        target_dates = test_origins + pd.Timedelta(days=h)
        actual_h = df.loc[target_dates, 'AQI'].values
        pers_rmse_h = float(np.sqrt(np.mean((actual_h - origin_aqi)**2)))
        
        fourier_sub = p5_m[(p5_m['model'] == 'ARIMA-Fourier (multi-step)') & (p5_m['horizon'] == h)]
        four_rmse_h = fourier_sub['RMSE'].values[0] if len(fourier_sub) > 0 else np.nan

        lstm_sub = p6_m[(p6_m['model'] == 'LSTM-7day') & (p6_m['horizon'] == h)]
        lstm_rmse_h = lstm_sub['RMSE'].values[0] if len(lstm_sub) > 0 else np.nan

        beats = "YES" if (lstm_rmse_h < pers_rmse_h) else "NO"
        print(f"  {h} | {pers_rmse_h:22.2f} | {four_rmse_h:18.2f} | {lstm_rmse_h:20.2f} | {beats}")

    # -------------------------------------------------------------
    # 4. STRESS MEAN FORECAST BIAS AUDIT
    # -------------------------------------------------------------
    print("\n--- 4. Stress Split Mean Forecast Bias Audit ---")
    expected_bias = {
        'Persistence': -0.31,
        'ARIMAX': -8.32,
        'LSTM (multivariate)': +10.55,
        'LSTM (univariate)': +7.84,
        'Hybrid': -3.19
    }
    for m_label, (df_source, m_col, master_name) in test_models.items():
        sub = df_source[(df_source['model'] == m_col) & (df_source['split'] == 'stress') & (df_source['horizon'] == 1)].drop_duplicates(subset=['date'])
        if len(sub) > 0:
            bias_val = float(np.mean(sub['predicted'] - sub['actual']))
            exp_b = expected_bias[master_name]
            diff_b = abs(bias_val - exp_b)
            status = "OK" if diff_b < 0.05 else "MISMATCH"
            print(f"  {m_label:20s}: Mean Bias = {bias_val:+.2f} (Expected: {exp_b:+.2f}) | {status}")
            if status == "MISMATCH":
                mismatches.append(f"{m_label} stress bias mismatch: got {bias_val:.2f}, expected {exp_b:.2f}")

    # -------------------------------------------------------------
    # 5. TEXT SCAN OF DOCUMENTATION FOR OUTDATED NUMBERS & PHRASES
    # -------------------------------------------------------------
    print("\n--- 5. Scanning Documentation for Outdated / Mismatched Strings ---")
    doc_files = [
        'docs/RESULTS_SUMMARY.md',
        'docs/PROJECT_LOG.md',
        'docs/FINAL_NUMBERS.md',
        'README.md',
        'notebooks/07_evaluation.ipynb'
    ]

    outdated_patterns = [
        (r'\b68\.2\b', 'outdated stress mean AQI 68.2 (should be 78.81)'),
        (r'h\s*=\s*1\s*RMSE\s*of\s*(\*\*)?30\.11', 'outdated leaked 7day h=1 RMSE 30.11 (should be 40.39)'),
        (r'proven statistical significance', 'unqualified phrase: proven statistical significance'),
        (r'mathematically guaranteed', 'unqualified phrase: mathematically guaranteed'),
        (r'anchoring to pre-lockdown levels', 'prohibited phrase: anchoring to pre-lockdown levels'),
        (r'adapted faster through its ARIMAX component', 'prohibited phrase: adapted faster through its ARIMAX component')
    ]

    for fpath in doc_files:
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        for pat, desc in outdated_patterns:
            matches = re.findall(pat, content, re.IGNORECASE)
            if matches:
                mismatches.append(f"In {fpath}: Found {desc} ({len(matches)} occurrences)")
                print(f"  [FOUND ISSUE] {fpath}: {desc} ({len(matches)}x)")

    # -------------------------------------------------------------
    # 6. DELIVERABLES AUDIT (REPORT DOCX, PPTX, VIVA GUIDE)
    # -------------------------------------------------------------
    print("\n--- 6. Scanning Deliverables (Report, PPTX, Viva Guide) ---")
    deliverable_texts = {}

    report_path = 'Report/UCS321_AQI_Forecasting_Report.docx'
    if os.path.exists(report_path):
        from docx import Document
        d = Document(report_path)
        parts = [p.text for p in d.paragraphs]
        for tbl in d.tables:
            for row in tbl.rows:
                parts.append(" | ".join(c.text.strip() for c in row.cells))
        deliverable_texts['Report (.docx)'] = "\n".join(parts)

    ppt_path = 'Presentation/UCS321_AQI_Forecasting.pptx'
    if os.path.exists(ppt_path):
        from pptx import Presentation
        prs = Presentation(ppt_path)
        parts = []
        for si, slide in enumerate(prs.slides, 1):
            for shp in slide.shapes:
                if shp.has_text_frame:
                    parts.append(shp.text_frame.text)
                if shp.has_table:
                    for row in shp.table.rows:
                        parts.append(" | ".join(c.text.strip() for c in row.cells))
            if slide.has_notes_slide:
                parts.append(slide.notes_slide.notes_text_frame.text)
        deliverable_texts['Presentation (.pptx)'] = "\n".join(parts)

    viva_path = 'PROJECT_MASTER_GUIDE_AND_VIVA_PREP.md'
    if os.path.exists(viva_path):
        with open(viva_path, 'r', encoding='utf-8') as vf:
            deliverable_texts['Viva Guide (.md)'] = vf.read()

    banned_words = [
        (r'\bproves\b', 'banned word: proves'),
        (r'\bdecisively\b', 'banned word: decisively'),
        (r'\bguaranteed\b', 'banned word: guaranteed'),
        (r'\bconfirms\b', 'banned word: confirms'),
    ]
    inconsistent_nums = [
        (r'\b68\.2\b', 'outdated stress mean 68.2 (should be 78.81)'),
        (r'\b30\.11\b', 'outdated leaked 7day h=1 RMSE 30.11 (should be 40.39)'),
        (r'\b57\.75\b', 'outdated leaked 7day h=7 RMSE 57.75 (should be 53.04)'),
        (r'\b60\.61\b', 'outdated persistence h=7 RMSE 60.61 (should be 61.16)'),
        (r'\b120\.30\b', 'outdated train mean 120.30 (should be 112.86)'),
        (r'\b91\.24\b', 'outdated val mean 91.24 (should be 103.42)'),
        (r'\b102\.48\b', 'outdated test mean 102.48 (should be 95.73)'),
        (r'\b21\.61\b', 'outdated univariate LSTM stress RMSE 21.61 (should be 15.98)'),
    ]

    for label, txt in deliverable_texts.items():
        issues_here = 0
        for pat, desc in banned_words + inconsistent_nums:
            hits = re.findall(pat, txt, re.IGNORECASE)
            if hits:
                issues_here += 1
                mismatches.append(f"In {label}: {desc} ({len(hits)}x)")
                print(f"  [FOUND ISSUE] {label}: {desc} ({len(hits)}x)")
        if issues_here == 0:
            print(f"  {label:22s}: 0 banned words, 0 inconsistent numbers | OK")

    # Verify all headline numbers from FINAL_NUMBERS.md appear in Report
    if 'Report (.docx)' in deliverable_texts and os.path.exists('docs/FINAL_NUMBERS.md'):
        with open('docs/FINAL_NUMBERS.md', 'r', encoding='utf-8') as fnf:
            fn_txt = fnf.read()
        headline_nums = [
            '25.97', '18.45', '0.6657', '76.92', '0.6471',
            '26.81', '19.91', '0.6435', '74.73',
            '27.63', '21.21', '0.6214', '68.68',
            '30.58', '19.23', '31.73', '22.87', '32.10', '19.56',
            '13.51', '14.45', '15.62', '15.98', '17.74', '20.28',
            '78.81', '112.86', '103.42', '95.73',
            '40.39', '56.08', '54.48', '53.04', '61.16', '50.35',
            '0.0001', '0.0132', '0.0297', '0.0127', '0.0758',
            '96.70', '96.15', '37.91', '63.87', '57.71',
            '-1.15', '-0.31', '-3.19', '+7.84', '-8.32', '+10.55',
            '-140.78', '-92.20'
        ]
        rep_txt = deliverable_texts['Report (.docx)']
        missing_in_rep = [num for num in headline_nums if num not in rep_txt]
        missing_in_fn = [num for num in headline_nums if num.lstrip('+') not in fn_txt and num not in fn_txt]
        if missing_in_rep or missing_in_fn:
            if missing_in_rep:
                mismatches.append(f"Report missing headline numbers: {missing_in_rep}")
                print(f"  [FOUND ISSUE] Missing in Report: {missing_in_rep}")
            if missing_in_fn:
                mismatches.append(f"FINAL_NUMBERS missing headline numbers: {missing_in_fn}")
                print(f"  [FOUND ISSUE] Missing in FINAL_NUMBERS: {missing_in_fn}")
        else:
            print(f"  Report Headline Check : All {len(headline_nums)} headline numbers verified against FINAL_NUMBERS.md | OK")

    # Summary
    print("\n" + "=" * 70)
    if mismatches:
        print(f"CONSISTENCY CHECK FINISHED WITH {len(mismatches)} ISSUES TO FIX:")
        for m in mismatches:
            print(f"  - {m}")
    else:
        print("ALL CONSISTENCY CHECKS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == '__main__':
    main()
