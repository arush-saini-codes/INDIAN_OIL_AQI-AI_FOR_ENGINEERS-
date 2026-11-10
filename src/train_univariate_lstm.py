"""src/train_univariate_lstm.py
Train Univariate LSTM Baseline for the 2x2 Model Comparison Grid.
Evaluates on Validation, Test, and Stress splits across 5 seeds.
Outputs predictions and metrics to outputs/results/
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.abspath('.'))
from src.sequences import load_scaled_dataset, make_sequences_1day
from src.models_dl import build_model
from src.metrics import evaluate_forecast, create_predictions_dataframe

SEEDS = [42, 123, 456, 789, 1011]
RESULTS_DIR = 'outputs/results'
MODELS_DIR = 'outputs/models'


def main():
    print("=" * 60)
    print("Training Univariate LSTM Baseline (U feature set, Lookback=7)")
    print("=" * 60)

    df, sc_tgt, _, _ = load_scaled_dataset()
    seqs = make_sequences_1day(df, feature_set='U', lookback=7)

    X_tr, y_tr = seqs['train']['X'], seqs['train']['y']
    X_val, y_val = seqs['val']['X'], seqs['val']['y']
    X_tv, y_tv = seqs['train_val']['X'], seqs['train_val']['y']
    X_test, y_test = seqs['test']['X'], seqs['test']['y']
    X_stress, y_stress = seqs['stress']['X'], seqs['stress']['y']

    y_val_raw = seqs['val']['y_raw']
    y_test_raw = seqs['test']['y_raw']
    y_stress_raw = seqs['stress']['y_raw']

    val_dates = seqs['val']['target_dates']
    test_dates = seqs['test']['target_dates']
    stress_dates = seqs['stress']['target_dates']

    inp_shape = (7, X_tr.shape[2])

    # Step 1: Early stopping on validation to find optimal epochs
    val_epochs = []
    for s in SEEDS:
        tf.keras.utils.set_random_seed(s)
        m = build_model(
            'Stacked_LSTM', input_shape=inp_shape,
            units_l1=64, units_l2=32, dropout=0.2, recurrent_dropout=0.0,
            output_dim=1, lr=1e-3, loss_fn='mse'
        )
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        hist = m.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[es], verbose=0)
        best_ep = int(np.argmin(hist.history['val_loss']) + 1)
        val_epochs.append(best_ep)
        print(f"Seed {s}: Best Val Epoch = {best_ep}")

    med_epochs = int(np.median(val_epochs))
    print(f"Median Best Epoch: {med_epochs}")

    # Step 2: Validation predictions (models trained on Train only)
    val_seed_preds = []
    for s in SEEDS:
        tf.keras.utils.set_random_seed(s)
        m = build_model(
            'Stacked_LSTM', input_shape=inp_shape,
            units_l1=64, units_l2=32, dropout=0.2, recurrent_dropout=0.0,
            output_dim=1, lr=1e-3, loss_fn='mse'
        )
        m.fit(X_tr, y_tr, epochs=med_epochs, batch_size=32, verbose=0)
        p = m.predict(X_val, verbose=0)
        p_inv = sc_tgt.inverse_transform(p).flatten()
        val_seed_preds.append(p_inv)

    val_pred_ens = np.mean(val_seed_preds, axis=0)
    val_metrics = evaluate_forecast(y_val_raw, val_pred_ens, model_name='LSTM (univariate)', split_name='val', horizon=1)
    print("Validation Metrics:", val_metrics)

    # Step 3: Train on Train+Val for Test and Stress evaluations
    test_seed_preds = []
    stress_seed_preds = []
    for s in SEEDS:
        tf.keras.utils.set_random_seed(s)
        m = build_model(
            'Stacked_LSTM', input_shape=inp_shape,
            units_l1=64, units_l2=32, dropout=0.2, recurrent_dropout=0.0,
            output_dim=1, lr=1e-3, loss_fn='mse'
        )
        m.fit(X_tv, y_tv, epochs=med_epochs, batch_size=32, verbose=0)
        
        # Save model
        m.save(os.path.join(MODELS_DIR, f'lstm_univariate_seed{s}.keras'))

        p_test = sc_tgt.inverse_transform(m.predict(X_test, verbose=0)).flatten()
        p_stress = sc_tgt.inverse_transform(m.predict(X_stress, verbose=0)).flatten()

        test_seed_preds.append(p_test)
        stress_seed_preds.append(p_stress)

    test_pred_ens = np.mean(test_seed_preds, axis=0)
    stress_pred_ens = np.mean(stress_seed_preds, axis=0)

    test_metrics = evaluate_forecast(y_test_raw, test_pred_ens, model_name='LSTM (univariate)', split_name='test', horizon=1)
    stress_metrics = evaluate_forecast(y_stress_raw, stress_pred_ens, model_name='LSTM (univariate)', split_name='stress', horizon=1)

    print("Test Metrics:", test_metrics)
    print("Stress Metrics:", stress_metrics)

    # Save metrics and predictions
    df_metrics = pd.DataFrame([val_metrics, test_metrics, stress_metrics])
    df_metrics.to_csv(os.path.join(RESULTS_DIR, 'lstm_univariate_metrics.csv'), index=False)

    df_val_p = create_predictions_dataframe(val_dates, y_val_raw, val_pred_ens, horizon=1, model='LSTM (univariate)', split='val')
    df_test_p = create_predictions_dataframe(test_dates, y_test_raw, test_pred_ens, horizon=1, model='LSTM (univariate)', split='test')
    df_stress_p = create_predictions_dataframe(stress_dates, y_stress_raw, stress_pred_ens, horizon=1, model='LSTM (univariate)', split='stress')

    all_preds = pd.concat([df_val_p, df_test_p, df_stress_p], ignore_index=True)
    all_preds.to_csv(os.path.join(RESULTS_DIR, 'lstm_univariate_predictions.csv'), index=False)
    print("Saved Univariate LSTM metrics and predictions successfully.")


if __name__ == '__main__':
    main()

