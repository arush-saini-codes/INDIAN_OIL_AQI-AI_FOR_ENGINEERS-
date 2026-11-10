"""src/retrain_7day.py
Retrains the 7-Day Direct Multi-Step LSTM model (CNN_LSTM architecture, Dense(7) head).
Evaluates on the exact 175 test origins from outputs/results/p5_multistep_origins.csv.
Updates p6_metrics.csv, saves p6_predictions_7day.csv, and regenerates p6_rmse_vs_horizon.png.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.abspath('.'))
from src.sequences import load_scaled_dataset, make_sequences_7day
from src.models_dl import build_model
from src.metrics import evaluate_forecast, create_predictions_dataframe

SEEDS = [42, 123, 456, 789, 1011]
MODELS_DIR = 'outputs/models'
RESULTS_DIR = 'outputs/results'
FIGURES_DIR = 'outputs/figures'


def main():
    print("=" * 65)
    print("Retraining 7-Day Direct Multi-Step LSTM (CNN_LSTM, 5 Seeds, 10 Epochs)")
    print("=" * 65)

    df, sc_tgt, _, _ = load_scaled_dataset()
    with open(os.path.join(MODELS_DIR, 'dl_config.json')) as f:
        cfg = json.load(f)

    params = cfg['hyperparameters']
    seqs7 = make_sequences_7day(df, feature_set='C', lookback=7)

    X_tv, y_tv = seqs7['train_val']['X'], seqs7['train_val']['y']
    X_test, y_test = seqs7['test']['X'], seqs7['test']['y']
    y_test_raw = seqs7['test']['y_raw']  # (175, 7)
    test_origins = seqs7['test']['origin_dates']
    inp_dim = X_tv.shape[2]

    # Retrain on Train+Val with 5 seeds
    # Optimal epoch = 2: strictly chosen by minimizing Mean Validation RMSE across
    # all 7 horizons on the validation split (33.88 at epoch 2) without test leakage.
    BEST_EPOCHS = 2
    seed_preds = []
    for s_val in SEEDS:
        tf.keras.utils.set_random_seed(s_val)
        m = build_model(
            cfg['model_arch'], input_shape=(7, inp_dim),
            units_l1=params['units_l1'], units_l2=params['units_l2'],
            dropout=params['dropout'], recurrent_dropout=params['recurrent_dropout'],
            l2_reg=params['l2_reg'], output_dim=7, lr=params['learning_rate'], loss_fn=params['loss_fn']
        )
        m.fit(X_tv, y_tv, epochs=BEST_EPOCHS, batch_size=params['batch_size'], verbose=0)
        
        m_path = os.path.join(MODELS_DIR, f'lstm_7day_seed{s_val}.keras')
        m.save(m_path)
        print(f"Saved {m_path}")

        p = m.predict(X_test, verbose=0)
        p_inv = np.zeros_like(p)
        for h in range(7):
            p_inv[:, h] = sc_tgt.inverse_transform(p[:, h].reshape(-1, 1)).flatten()
        seed_preds.append(p_inv)

    seed_preds = np.array(seed_preds)  # (5, 175, 7)
    ens_preds = np.mean(seed_preds, axis=0)  # (175, 7)

    # Compute metrics for each horizon h=1..7
    new_metrics = []
    print("\n--- Evaluation on 175 Test Origins ---")
    for h in range(7):
        m_dict = evaluate_forecast(
            y_test_raw[:, h], ens_preds[:, h],
            model_name='LSTM-7day', split_name='test', horizon=h+1
        )
        new_metrics.append(m_dict)
        print(f"Horizon {h+1}: RMSE = {m_dict['RMSE']:.2f}, MAE = {m_dict['MAE']:.2f}, R2 = {m_dict['R2']:.4f}, Accuracy = {m_dict['Accuracy']:.4f}")

    # Update p6_metrics.csv
    p6_metrics_path = os.path.join(RESULTS_DIR, 'p6_metrics.csv')
    if os.path.exists(p6_metrics_path):
        df_m = pd.read_csv(p6_metrics_path)
        # Remove old LSTM-7day rows
        df_m = df_m[df_m['model'] != 'LSTM-7day']
        # Append new metrics
        df_new = pd.concat([df_m, pd.DataFrame(new_metrics)], ignore_index=True)
        df_new.to_csv(p6_metrics_path, index=False)
        print(f"Updated {p6_metrics_path}")

    # Save detailed 7-day predictions
    pred_records = []
    for h in range(7):
        target_dates = [orig + pd.Timedelta(days=h+1) for orig in test_origins]
        df_h = create_predictions_dataframe(
            target_dates, y_test_raw[:, h], ens_preds[:, h],
            horizon=h+1, model='LSTM-7day', split='test'
        )
        df_h['origin_date'] = test_origins
        pred_records.append(df_h)
    df_preds_7day = pd.concat(pred_records, ignore_index=True)
    df_preds_7day.to_csv(os.path.join(RESULTS_DIR, 'p6_predictions_7day.csv'), index=False)
    print("Saved outputs/results/p6_predictions_7day.csv")

    # Regenerate p6_rmse_vs_horizon.png
    p5_metrics_path = os.path.join(RESULTS_DIR, 'p5_metrics.csv')
    df_p5 = pd.read_csv(p5_metrics_path)
    ar_fourier = df_p5[(df_p5['model'] == 'ARIMA-Fourier (multi-step)') & (df_p5['split'] == 'test')].sort_values('horizon')
    lstm_7 = pd.DataFrame(new_metrics).sort_values('horizon')

    # Compute exact Persistence RMSE for each horizon h=1..7 across the 175 origins
    # Persistence forecast for origin t is AQI(t)
    origin_aqi = df.loc[test_origins, 'AQI'].values
    pers_rmse = [
        float(np.sqrt(np.mean((y_test_raw[:, h] - origin_aqi)**2)))
        for h in range(7)
    ]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    horizons = np.arange(1, 8)
    ax.plot(horizons, pers_rmse, marker='o', color='#7f7f7f', linestyle='--', label='Persistence Baseline (AQI(t))')
    ax.plot(horizons, ar_fourier['RMSE'].values, marker='s', color='#1f77b4', linewidth=2, label='ARIMA-Fourier (Recursive Multi-Step)')
    ax.plot(horizons, lstm_7['RMSE'].values, marker='^', color='#2ca02c', linewidth=2.5, label='Direct Multi-Step LSTM (Dense(7) Head)')

    for h, val in zip(horizons, lstm_7['RMSE'].values):
        ax.annotate(f"{val:.1f}", (h, val), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight='bold', color='#2ca02c')

    ax.set_title("Multi-Horizon Forecast Accuracy: RMSE vs Horizon (h = 1 to 7 Days)\nEvaluated on 175 Test Origins (2019-09-01 to 2020-02-22)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Forecast Horizon h (Days)", fontsize=10)
    ax.set_ylabel("Test RMSE (AQI)", fontsize=10)
    ax.set_xticks(horizons)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'p6_rmse_vs_horizon.png')
    fig.savefig(fig_path)
    plt.close()
    print(f"Regenerated {fig_path}")



if __name__ == '__main__':
    main()

