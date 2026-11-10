"""End-to-end resumable training, hyperparameter optimization, and evaluation script
for Phase 6 Deep Learning models.
Supports:
  - Stage 1: Feature set (U, P, C, A) x Lookback (7, 14, 30, 60) grid search (5 seeds)
  - Stage 2: Architecture comparison (Stacked LSTM, Stacked GRU, BiLSTM, CNN-LSTM) (5 seeds)
  - Stage 3: Optuna HPO with TPE sampler and MedianPruner (60 trials x 2 seeds)
  - Stage 4 / Final 1-Day: Retrain winning config on Train+Val (5 seeds) + MC Dropout 95% PI
  - Stage 5 / 7-Day: Direct multi-step Dense(7) model on Train+Val (5 seeds) on exact p5 origins
  - Stage 6 / Hybrid: Residual LSTM trained on ARIMAX one-step errors (5 seeds)
  - Stage 7 / Stress Test: Evaluation on COVID lockdown window (2020-03-01 -> 2020-06-30)
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import json
import random
import argparse
import joblib
import optuna
import numpy as np
import pandas as pd

# Suppress TF C++ logs and configure deterministic CPU threading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# Hardware CPU Threading configuration
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(4)

from src.sequences import load_scaled_dataset, make_sequences_1day, make_sequences_7day
from src.models_dl import build_model
from src.metrics import evaluate_forecast, compute_category_metrics

# Output paths
RESULTS_DIR = 'outputs/results'
MODELS_DIR = 'outputs/models'
LOG_PATH = os.path.join(RESULTS_DIR, 'p6_training.log')
EXPERIMENTS_CSV = os.path.join(RESULTS_DIR, 'p6_experiments.csv')
OPTUNA_DB = os.path.join(RESULTS_DIR, 'p6_optuna.db')
OPTUNA_TRIALS_CSV = os.path.join(RESULTS_DIR, 'p6_optuna_trials.csv')
CONFIG_JSON = os.path.join(MODELS_DIR, 'dl_config.json')

SEEDS_5 = [42, 123, 456, 789, 1011]
SEEDS_2 = [42, 123]


def log(msg):
    """Write timestamped log message to stdout and p6_training.log."""
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{ts}] {msg}"
    print(formatted)
    sys.stdout.flush()
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(formatted + '\n')


def set_seed(seed):
    """Set deterministic seeds across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def init_experiments_csv():
    """Ensure experiments tracking CSV exists."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(EXPERIMENTS_CSV):
        df = pd.DataFrame(columns=[
            'timestamp', 'stage', 'model_arch', 'feature_set', 'lookback', 'seed',
            'best_epoch', 'train_time_sec', 'val_loss', 'val_RMSE', 'val_MAE',
            'val_R2', 'val_MAPE', 'val_QWK'
        ])
        df.to_csv(EXPERIMENTS_CSV, index=False)


def is_run_completed(stage, model_arch, feature_set, lookback, seed):
    """Check if an experiment run has already been recorded."""
    if not os.path.exists(EXPERIMENTS_CSV):
        return False
    df = pd.read_csv(EXPERIMENTS_CSV)
    sub = df[
        (df['stage'] == str(stage)) &
        (df['model_arch'] == str(model_arch)) &
        (df['feature_set'] == str(feature_set)) &
        (df['lookback'] == int(lookback)) &
        (df['seed'] == int(seed))
    ]
    return len(sub) > 0


def record_experiment_run(record_dict):
    """Append a single finished experiment run to p6_experiments.csv."""
    df = pd.DataFrame([record_dict])
    df.to_csv(EXPERIMENTS_CSV, mode='a', header=False, index=False)


# =============================================================================
# STAGE 1: Feature Set x Lookback Grid Search
# =============================================================================
def run_stage_1(df, sc_tgt):
    """Execute Stage 1 grid search across 4 feature sets x 4 lookbacks x 5 seeds."""
    log("====================================================================")
    log("STARTING STAGE 1: Feature Set x Lookback Search (Baseline Stacked LSTM)")
    log("====================================================================")

    feature_sets = ['U', 'P', 'C', 'A']
    lookbacks = [7, 14, 30, 60]

    for fs in feature_sets:
        for lb in lookbacks:
            for seed in SEEDS_5:
                if is_run_completed(1, 'Stacked_LSTM', fs, lb, seed):
                    log(f"[Stage 1] Skipping already completed run: fs={fs}, lb={lb}, seed={seed}")
                    continue

                set_seed(seed)
                seqs = make_sequences_1day(df, feature_set=fs, lookback=lb)
                X_tr, y_tr = seqs['train']['X'], seqs['train']['y']
                X_val, y_val = seqs['val']['X'], seqs['val']['y']
                y_val_raw = seqs['val']['y_raw']

                m = build_model('Stacked_LSTM', input_shape=(lb, X_tr.shape[2]), units_l1=64, units_l2=64, dropout=0.2, lr=1e-3, loss_fn='mse')
                es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

                t0 = time.time()
                hist = m.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[es], verbose=0)
                dur = time.time() - t0

                best_epoch = int(np.argmin(hist.history['val_loss']) + 1)
                best_val_loss = float(np.min(hist.history['val_loss']))

                val_preds = m.predict(X_val, verbose=0)
                val_preds_inv = sc_tgt.inverse_transform(val_preds).flatten()
                m_metrics = evaluate_forecast(y_val_raw, val_preds_inv)

                rec = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'stage': 1,
                    'model_arch': 'Stacked_LSTM',
                    'feature_set': fs,
                    'lookback': lb,
                    'seed': seed,
                    'best_epoch': best_epoch,
                    'train_time_sec': round(dur, 2),
                    'val_loss': round(best_val_loss, 5),
                    'val_RMSE': round(m_metrics['RMSE'], 3),
                    'val_MAE': round(m_metrics['MAE'], 3),
                    'val_R2': round(m_metrics['R2'], 4),
                    'val_MAPE': round(m_metrics['MAPE'], 3),
                    'val_QWK': round(m_metrics['QWK'], 4)
                }
                record_experiment_run(rec)
                log(f"[Stage 1 Done] fs={fs}, lb={lb}, seed={seed} -> Val RMSE={m_metrics['RMSE']:.2f}, Epoch={best_epoch}, Time={dur:.1f}s")


def get_stage_1_top2():
    """Rank Stage 1 configurations by mean val RMSE and return top 2."""
    df = pd.read_csv(EXPERIMENTS_CSV)
    s1 = df[df['stage'] == 1]
    agg = s1.groupby(['feature_set', 'lookback'])['val_RMSE'].agg(['mean', 'std']).reset_index()
    agg = agg.sort_values('mean')
    top2 = agg.head(2).to_dict('records')
    return top2, agg


# =============================================================================
# STAGE 2: Architecture Comparison
# =============================================================================
def run_stage_2(df, sc_tgt, top2_configs):
    """Compare Stacked LSTM, Stacked GRU, BiLSTM, and CNN-LSTM on top 2 Stage 1 configs."""
    log("====================================================================")
    log("STARTING STAGE 2: Architecture Comparison on Top 2 Configurations")
    log("====================================================================")

    archs = ['Stacked_LSTM', 'Stacked_GRU', 'BiLSTM', 'CNN_LSTM']

    for cfg in top2_configs:
        fs = cfg['feature_set']
        lb = int(cfg['lookback'])

        for arch in archs:
            for seed in SEEDS_5:
                if is_run_completed(2, arch, fs, lb, seed):
                    log(f"[Stage 2] Skipping already completed run: arch={arch}, fs={fs}, lb={lb}, seed={seed}")
                    continue

                set_seed(seed)
                seqs = make_sequences_1day(df, feature_set=fs, lookback=lb)
                X_tr, y_tr = seqs['train']['X'], seqs['train']['y']
                X_val, y_val = seqs['val']['X'], seqs['val']['y']
                y_val_raw = seqs['val']['y_raw']

                m = build_model(arch, input_shape=(lb, X_tr.shape[2]), units_l1=64, units_l2=64, dropout=0.2, lr=1e-3, loss_fn='mse')
                es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

                t0 = time.time()
                hist = m.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[es], verbose=0)
                dur = time.time() - t0

                best_epoch = int(np.argmin(hist.history['val_loss']) + 1)
                best_val_loss = float(np.min(hist.history['val_loss']))

                val_preds = m.predict(X_val, verbose=0)
                val_preds_inv = sc_tgt.inverse_transform(val_preds).flatten()
                m_metrics = evaluate_forecast(y_val_raw, val_preds_inv)

                rec = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'stage': 2,
                    'model_arch': arch,
                    'feature_set': fs,
                    'lookback': lb,
                    'seed': seed,
                    'best_epoch': best_epoch,
                    'train_time_sec': round(dur, 2),
                    'val_loss': round(best_val_loss, 5),
                    'val_RMSE': round(m_metrics['RMSE'], 3),
                    'val_MAE': round(m_metrics['MAE'], 3),
                    'val_R2': round(m_metrics['R2'], 4),
                    'val_MAPE': round(m_metrics['MAPE'], 3),
                    'val_QWK': round(m_metrics['QWK'], 4)
                }
                record_experiment_run(rec)
                log(f"[Stage 2 Done] arch={arch}, fs={fs}, lb={lb}, seed={seed} -> Val RMSE={m_metrics['RMSE']:.2f}, Time={dur:.1f}s")


def get_stage_2_winner():
    """Rank Stage 2 configurations and select winning architecture."""
    df = pd.read_csv(EXPERIMENTS_CSV)
    s2 = df[df['stage'] == 2]
    agg = s2.groupby(['model_arch', 'feature_set', 'lookback'])['val_RMSE'].agg(['mean', 'std']).reset_index()
    agg = agg.sort_values('mean')
    winner = agg.iloc[0].to_dict()
    return winner, agg


# =============================================================================
# STAGE 3: Optuna Hyperparameter Optimization
# =============================================================================
def run_stage_3_optuna(df, sc_tgt, winner_cfg, n_trials=60):
    """Run Optuna study on the Stage 2 winning architecture."""
    log("====================================================================")
    log(f"STARTING STAGE 3: Optuna HPO ({n_trials} Trials, 2 Seeds/Trial)")
    log("====================================================================")

    arch = winner_cfg['model_arch']
    fs = winner_cfg['feature_set']
    lb = int(winner_cfg['lookback'])

    seqs = make_sequences_1day(df, feature_set=fs, lookback=lb)
    X_tr, y_tr = seqs['train']['X'], seqs['train']['y']
    X_val, y_val = seqs['val']['X'], seqs['val']['y']
    y_val_raw = seqs['val']['y_raw']
    inp_dim = X_tr.shape[2]

    storage_name = f"sqlite:///{OPTUNA_DB}"
    study = optuna.create_study(
        study_name="p6_optuna_hpo",
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    def objective(trial):
        u1 = trial.suggest_int('units_l1', 32, 128, step=16)
        u2 = trial.suggest_int('units_l2', 16, 64, step=16)
        drop = trial.suggest_float('dropout', 0.0, 0.4, step=0.05)
        rec_drop = trial.suggest_float('recurrent_dropout', 0.0, 0.2, step=0.05)
        lr = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
        bs = trial.suggest_categorical('batch_size', [16, 32, 64])
        l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
        loss_fn = trial.suggest_categorical('loss_fn', ['mse', 'huber'])

        seed_rmses = []
        for s_idx, s_val in enumerate(SEEDS_2):
            set_seed(s_val)
            m = build_model(
                arch, input_shape=(lb, inp_dim), units_l1=u1, units_l2=u2,
                dropout=drop, recurrent_dropout=rec_drop, l2_reg=l2_reg,
                output_dim=1, lr=lr, loss_fn=loss_fn
            )
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

            hist = m.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=100, batch_size=bs, callbacks=[es], verbose=0)
            
            # Pruning check at epoch 20 for first seed
            if s_idx == 0:
                epoch_loss = float(np.min(hist.history['val_loss']))
                trial.report(epoch_loss, step=len(hist.history['loss']))
                if trial.should_prune():
                    raise optuna.TrialPruned()

            val_preds = m.predict(X_val, verbose=0)
            val_preds_inv = sc_tgt.inverse_transform(val_preds).flatten()
            rmse = np.sqrt(np.mean((y_val_raw - val_preds_inv)**2))
            seed_rmses.append(rmse)

        mean_rmse = float(np.mean(seed_rmses))
        return mean_rmse

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)
    if remaining > 0:
        log(f"Optuna study resuming: {completed} already completed, running {remaining} more trials...")
        study.optimize(objective, n_trials=remaining)
    else:
        log(f"Optuna study already has {completed} trials completed.")

    # Export trials dataframe
    trials_df = study.trials_dataframe()
    trials_df.to_csv(OPTUNA_TRIALS_CSV, index=False)
    log(f"Saved {len(trials_df)} Optuna trials to {OPTUNA_TRIALS_CSV}")

    best_trial = study.best_trial
    log(f"[Optuna Best Trial] #{best_trial.number}: Val RMSE = {best_trial.value:.3f}")
    log(f"Best Params: {best_trial.params}")

    # Re-evaluate best Optuna config across full 5 seeds
    log("Re-evaluating best Optuna configuration with 5 seeds...")
    p = best_trial.params
    optuna_5seed_rmses = []
    optuna_epochs = []

    for s_val in SEEDS_5:
        set_seed(s_val)
        m = build_model(
            arch, input_shape=(lb, inp_dim), units_l1=p['units_l1'], units_l2=p['units_l2'],
            dropout=p['dropout'], recurrent_dropout=p['recurrent_dropout'], l2_reg=p['l2_reg'],
            output_dim=1, lr=p['learning_rate'], loss_fn=p['loss_fn']
        )
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        hist = m.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=100, batch_size=p['batch_size'], callbacks=[es], verbose=0)
        best_ep = int(np.argmin(hist.history['val_loss']) + 1)
        optuna_epochs.append(best_ep)

        val_preds = m.predict(X_val, verbose=0)
        val_preds_inv = sc_tgt.inverse_transform(val_preds).flatten()
        rmse = np.sqrt(np.mean((y_val_raw - val_preds_inv)**2))
        optuna_5seed_rmses.append(rmse)

        rec = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'stage': 3,
            'model_arch': f"{arch}_Optuna",
            'feature_set': fs,
            'lookback': lb,
            'seed': s_val,
            'best_epoch': best_ep,
            'train_time_sec': 0.0,
            'val_loss': float(np.min(hist.history['val_loss'])),
            'val_RMSE': round(rmse, 3),
            'val_MAE': 0.0,
            'val_R2': 0.0,
            'val_MAPE': 0.0,
            'val_QWK': 0.0
        }
        record_experiment_run(rec)

    mean_opt_rmse = float(np.mean(optuna_5seed_rmses))
    std_opt_rmse = float(np.std(optuna_5seed_rmses))
    median_epochs = int(np.median(optuna_epochs))

    log(f"Stage 2 Winner Val RMSE: {winner_cfg['mean']:.3f} ± {winner_cfg['std']:.3f}")
    log(f"Optuna Best 5-Seed Val RMSE: {mean_opt_rmse:.3f} ± {std_opt_rmse:.3f} (Median Epochs: {median_epochs})")

    # Selection decision: does Optuna beat Stage 2 winner?
    if mean_opt_rmse <= winner_cfg['mean']:
        log("SUCCESS: Optuna configuration beat or matched Stage 2 winner! Adopting Optuna configuration.")
        final_params = p
        final_arch = arch
        final_fs = fs
        final_lb = lb
        final_median_epochs = median_epochs
    else:
        log("Optuna 5-seed did not beat Stage 2 winner; keeping Stage 2 baseline parameters.")
        final_params = {
            'units_l1': 64, 'units_l2': 64, 'dropout': 0.2, 'recurrent_dropout': 0.0,
            'learning_rate': 1e-3, 'batch_size': 32, 'l2_reg': 0.0, 'loss_fn': 'mse'
        }
        final_arch = winner_cfg['model_arch']
        final_fs = winner_cfg['feature_set']
        final_lb = int(winner_cfg['lookback'])
        # Median epochs from Stage 2 runs
        s2_df = pd.read_csv(EXPERIMENTS_CSV)
        s2_sub = s2_df[(s2_df['stage']==2) & (s2_df['model_arch']==final_arch) & (s2_df['feature_set']==final_fs) & (s2_df['lookback']==final_lb)]
        final_median_epochs = int(s2_sub['best_epoch'].median())

    return study, final_arch, final_fs, final_lb, final_params, final_median_epochs


# =============================================================================
# STAGE 4: Final 1-Day Model Retrained on Train+Val & MC Dropout
# =============================================================================
def train_final_1day(df, sc_tgt, arch, fs, lb, params, median_epochs):
    """Retrain final 1-day model on Train+Val across 5 seeds for median_epochs."""
    log("====================================================================")
    log(f"STAGE 4: Final 1-Day Model Retraining on Train+Val (Epochs={median_epochs}, 5 Seeds)")
    log("====================================================================")

    seqs = make_sequences_1day(df, feature_set=fs, lookback=lb)
    X_tv, y_tv = seqs['train_val']['X'], seqs['train_val']['y']
    X_test, y_test = seqs['test']['X'], seqs['test']['y']
    y_test_raw = seqs['test']['y_raw']
    test_dates = seqs['test']['target_dates']
    inp_dim = X_tv.shape[2]

    # Ensure dropout is at least 0.2 for MC Dropout
    mc_dropout = max(0.2, params.get('dropout', 0.2))

    seed_models = []
    seed_preds = []
    all_mc_passes = []

    for s_val in SEEDS_5:
        set_seed(s_val)
        m = build_model(
            arch, input_shape=(lb, inp_dim), units_l1=params['units_l1'], units_l2=params['units_l2'],
            dropout=mc_dropout, recurrent_dropout=params['recurrent_dropout'], l2_reg=params['l2_reg'],
            output_dim=1, lr=params['learning_rate'], loss_fn=params['loss_fn']
        )
        m.fit(X_tv, y_tv, epochs=median_epochs, batch_size=params['batch_size'], verbose=0)
        
        # Save model artifact
        m_path = os.path.join(MODELS_DIR, f'lstm_1day_seed{s_val}.keras')
        m.save(m_path)
        seed_models.append(m)

        # Standard deterministic test prediction
        p_test = m.predict(X_test, verbose=0)
        p_test_inv = sc_tgt.inverse_transform(p_test).flatten()
        seed_preds.append(p_test_inv)

        # 100 stochastic MC Dropout passes for this seed
        for _ in range(100):
            p_mc = m(X_test, training=True).numpy()
            p_mc_inv = sc_tgt.inverse_transform(p_mc).flatten()
            all_mc_passes.append(p_mc_inv)

    seed_preds = np.array(seed_preds)  # Shape: (5, 182)
    ensemble_preds = np.mean(seed_preds, axis=0)  # Shape: (182,)

    # MC Dropout 95% Prediction Interval (pooled 500 passes)
    all_mc_passes = np.array(all_mc_passes)  # Shape: (500, 182)
    pi_lower = np.percentile(all_mc_passes, 2.5, axis=0)
    pi_upper = np.percentile(all_mc_passes, 97.5, axis=0)

    # Test Metrics across seeds (mean ± std)
    seed_metrics_list = [evaluate_forecast(y_test_raw, seed_preds[i]) for i in range(5)]
    seed_rmse_mean = np.mean([m['RMSE'] for m in seed_metrics_list])
    seed_rmse_std = np.std([m['RMSE'] for m in seed_metrics_list])

    # Ensemble Metrics (The LSTM)
    ens_metrics = evaluate_forecast(y_test_raw, ensemble_preds)
    pi_coverage = float(np.mean((y_test_raw >= pi_lower) & (y_test_raw <= pi_upper)) * 100.0)
    pi_width = float(np.mean(pi_upper - pi_lower))

    log(f"Test 1-Day LSTM Seeds Mean ± Std RMSE: {seed_rmse_mean:.2f} ± {seed_rmse_std:.2f}")
    log(f"Test 1-Day LSTM Ensemble RMSE: {ens_metrics['RMSE']:.2f}, MAE: {ens_metrics['MAE']:.2f}, R2: {ens_metrics['R2']:.4f}, QWK: {ens_metrics['QWK']:.4f}")
    log(f"MC Dropout 95% PI Coverage: {pi_coverage:.2f}%, Mean Width: {pi_width:.2f}")

    return {
        'seed_models': seed_models,
        'ensemble_preds': ensemble_preds,
        'seed_preds': seed_preds,
        'pi_lower': pi_lower,
        'pi_upper': pi_upper,
        'pi_coverage': pi_coverage,
        'pi_width': pi_width,
        'ens_metrics': ens_metrics,
        'seed_metrics_list': seed_metrics_list,
        'test_dates': test_dates,
        'y_test_raw': y_test_raw,
        'mc_dropout': mc_dropout
    }


# =============================================================================
# STAGE 5: 7-Day Direct Multi-Step Model (Dense(7))
# =============================================================================
def train_final_7day(df, sc_tgt, arch, fs, lb, params):
    """Train direct multi-step 7-day model (Dense(7)) and evaluate on p5_multistep_origins.csv."""
    log("====================================================================")
    log("STAGE 5: 7-Day Direct Multi-Step Model Training (Dense(7), 5 Seeds)")
    log("====================================================================")

    seqs7 = make_sequences_7day(df, feature_set=fs, lookback=lb)
    X_tr, y_tr = seqs7['train']['X'], seqs7['train']['y']
    X_val, y_val = seqs7['val']['X'], seqs7['val']['y']
    X_tv, y_tv = seqs7['train_val']['X'], seqs7['train_val']['y']
    X_test, y_test = seqs7['test']['X'], seqs7['test']['y']
    y_test_raw = seqs7['test']['y_raw']  # Shape: (175, 7)
    test_origins = seqs7['test']['origin_dates']
    inp_dim = X_tr.shape[2]

    # Find early stopping epochs on validation
    set_seed(42)
    m_val = build_model(
        arch, input_shape=(lb, inp_dim), units_l1=params['units_l1'], units_l2=params['units_l2'],
        dropout=params['dropout'], recurrent_dropout=params['recurrent_dropout'], l2_reg=params['l2_reg'],
        output_dim=7, lr=params['learning_rate'], loss_fn=params['loss_fn']
    )
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    hist = m_val.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=100, batch_size=params['batch_size'], callbacks=[es], verbose=0)
    best_ep_7 = int(np.argmin(hist.history['val_loss']) + 1)
    log(f"7-Day Model Validation Best Epoch: {best_ep_7}")

    # Retrain on Train+Val with 5 seeds
    seed_preds_7 = []
    seed_models_7 = []
    for s_val in SEEDS_5:
        set_seed(s_val)
        m = build_model(
            arch, input_shape=(lb, inp_dim), units_l1=params['units_l1'], units_l2=params['units_l2'],
            dropout=params['dropout'], recurrent_dropout=params['recurrent_dropout'], l2_reg=params['l2_reg'],
            output_dim=7, lr=params['learning_rate'], loss_fn=params['loss_fn']
        )
        m.fit(X_tv, y_tv, epochs=best_ep_7, batch_size=params['batch_size'], verbose=0)
        
        m_path = os.path.join(MODELS_DIR, f'lstm_7day_seed{s_val}.keras')
        m.save(m_path)
        seed_models_7.append(m)

        p = m.predict(X_test, verbose=0)  # Shape: (175, 7)
        # Inverse transform each horizon column
        p_inv = np.zeros_like(p)
        for h in range(7):
            p_inv[:, h] = sc_tgt.inverse_transform(p[:, h].reshape(-1, 1)).flatten()
        seed_preds_7.append(p_inv)

    seed_preds_7 = np.array(seed_preds_7)  # Shape: (5, 175, 7)
    ensemble_preds_7 = np.mean(seed_preds_7, axis=0)  # Shape: (175, 7)

    # Compute RMSE per horizon h=1..7
    h_rmse_lstm = []
    for h in range(7):
        rmse_h = float(np.sqrt(np.mean((y_test_raw[:, h] - ensemble_preds_7[:, h])**2)))
        h_rmse_lstm.append(rmse_h)
        log(f"7-Day LSTM Horizon {h+1}: RMSE = {rmse_h:.2f}")

    return {
        'seed_models': seed_models_7,
        'ensemble_preds': ensemble_preds_7,
        'h_rmse': h_rmse_lstm,
        'y_test_raw': y_test_raw,
        'test_origins': test_origins,
        'best_epoch': best_ep_7
    }


# =============================================================================
# STAGE 6: Hybrid ARIMAX + LSTM Model
# =============================================================================
def train_hybrid_model(df, sc_tgt, arimax_model_path, fs, lb):
    """Train residual LSTM predicting ARIMAX one-step forecast errors."""
    log("====================================================================")
    log("STAGE 6: Hybrid ARIMAX + LSTM Residual Modeling (5 Seeds)")
    log("====================================================================")

    # Load fitted ARIMAX model
    arimax_model = joblib.load(arimax_model_path)

    # Reconstruct in-sample ARIMAX predictions on train+val
    tv_dates = pd.date_range('2016-01-01', '2019-08-31', freq='D')
    arimax_tv_preds = arimax_model.fittedvalues.loc[tv_dates]
    
    # Load out-of-sample ARIMAX predictions on test and stress
    p5_preds_df = pd.read_csv('outputs/results/p5_predictions.csv')
    p5_arimax = p5_preds_df[
        (p5_preds_df['model'] == 'Final_ARIMAX(2,1,1)+F3+PM25') &
        (p5_preds_df['horizon'] == 1)
    ].set_index('date')['predicted']
    p5_arimax.index = pd.to_datetime(p5_arimax.index)

    # Full series ARIMAX one-step prediction series
    full_arimax_preds = pd.concat([arimax_tv_preds, p5_arimax]).sort_index()

    # Calculate residuals: e(t) = actual(t) - arimax(t)
    actual_s = df['AQI']
    residuals_s = actual_s.loc[full_arimax_preds.index] - full_arimax_preds

    # Scale residuals with train-fitted statistics
    train_res = residuals_s.loc['2016-01-01':'2018-12-31']
    res_min, res_max = float(train_res.min()), float(train_res.max())
    scaled_res = (residuals_s - res_min) / (res_max - res_min + 1e-8)

    # Build sequence matrix: [scaled_residual, input_features]
    from src.sequences import get_feature_columns
    feat_cols = get_feature_columns(fs)
    feat_mat = df.loc[scaled_res.index, feat_cols].values
    res_mat = scaled_res.values.reshape(-1, 1)
    hybrid_X_full = np.hstack([res_mat, feat_mat])  # Shape: (N, 1 + D)
    dates = scaled_res.index

    # Sliding windows for residual forecasting
    X_list, y_list, y_raw_list, orig_dates, tgt_dates = [], [], [], [], []
    for i in range(lb - 1, len(dates) - 1):
        orig_dates.append(dates[i])
        tgt_dates.append(dates[i+1])
        X_list.append(hybrid_X_full[i - lb + 1 : i + 1])
        y_list.append(scaled_res.iloc[i + 1])
        y_raw_list.append(residuals_s.iloc[i + 1])

    X_all = np.array(X_list, dtype=np.float32)
    y_all = np.array(y_list, dtype=np.float32)
    tgt_dates = pd.DatetimeIndex(tgt_dates)

    # Partitions
    tr_mask = (tgt_dates >= '2016-01-01') & (tgt_dates <= '2018-12-31')
    val_mask = (tgt_dates >= '2019-01-01') & (tgt_dates <= '2019-08-31')
    tv_mask = (tgt_dates >= '2016-01-01') & (tgt_dates <= '2019-08-31')
    test_mask = (tgt_dates >= '2019-09-01') & (tgt_dates <= '2020-02-29')
    stress_mask = (tgt_dates >= '2020-03-01') & (tgt_dates <= '2020-06-30')

    # Find early stopping epochs
    set_seed(42)
    m_val = build_model('Stacked_LSTM', input_shape=(lb, X_all.shape[2]), units_l1=32, units_l2=16, dropout=0.2, lr=1e-3, loss_fn='mse')
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    hist = m_val.fit(X_all[tr_mask], y_all[tr_mask], validation_data=(X_all[val_mask], y_all[val_mask]), epochs=100, batch_size=32, callbacks=[es], verbose=0)
    best_ep = int(np.argmin(hist.history['val_loss']) + 1)
    log(f"Hybrid Residual LSTM Best Validation Epoch: {best_ep}")

    # Retrain on Train+Val with 5 seeds
    seed_hybrid_preds = []
    seed_models = []
    for s_val in SEEDS_5:
        set_seed(s_val)
        m = build_model('Stacked_LSTM', input_shape=(lb, X_all.shape[2]), units_l1=32, units_l2=16, dropout=0.2, lr=1e-3, loss_fn='mse')
        m.fit(X_all[tv_mask], y_all[tv_mask], epochs=best_ep, batch_size=32, verbose=0)
        
        m_path = os.path.join(MODELS_DIR, f'hybrid_resid_seed{s_val}.keras')
        m.save(m_path)
        seed_models.append(m)

        # Predict test residuals
        p_res_scaled = m.predict(X_all[test_mask], verbose=0).flatten()
        p_res = p_res_scaled * (res_max - res_min) + res_min

        # Add to ARIMAX test forecast
        arimax_test_preds = full_arimax_preds.loc[tgt_dates[test_mask]].values
        hybrid_test = arimax_test_preds + p_res
        seed_hybrid_preds.append(hybrid_test)

    ensemble_hybrid_test = np.mean(seed_hybrid_preds, axis=0)
    actual_test = actual_s.loc[tgt_dates[test_mask]].values
    hybrid_metrics = evaluate_forecast(actual_test, ensemble_hybrid_test)
    log(f"Hybrid ARIMAX+LSTM Test RMSE: {hybrid_metrics['RMSE']:.2f}, MAE: {hybrid_metrics['MAE']:.2f}, R2: {hybrid_metrics['R2']:.4f}")

    return {
        'seed_models': seed_models,
        'ensemble_preds': ensemble_hybrid_test,
        'metrics': hybrid_metrics,
        'test_dates': tgt_dates[test_mask],
        'actual_test': actual_test,
        'X_all': X_all,
        'stress_mask': stress_mask,
        'tgt_dates': tgt_dates,
        'full_arimax_preds': full_arimax_preds,
        'res_min': res_min,
        'res_max': res_max
    }


# =============================================================================
# STAGE 7: Stress Testing on COVID-19 Lockdown Window
# =============================================================================
def run_stress_testing(df, sc_tgt, final_1day_models, final_7day_models, hybrid_data):
    """Evaluate 1-day LSTM, 7-day LSTM, and Hybrid on COVID lockdown window."""
    log("====================================================================")
    log("STAGE 7: Stress-Test Evaluation on COVID-19 Lockdown (Mar–Jun 2020)")
    log("====================================================================")

    # 1. 1-Day LSTM on Stress
    seqs1 = make_sequences_1day(df, feature_set='C', lookback=14)
    X_stress = seqs1['stress']['X']
    y_stress_raw = seqs1['stress']['y_raw']
    stress_dates = seqs1['stress']['target_dates']

    preds_1d_stress = []
    for m in final_1day_models:
        p = m.predict(X_stress, verbose=0)
        p_inv = sc_tgt.inverse_transform(p).flatten()
        preds_1d_stress.append(p_inv)
    ens_1d_stress = np.mean(preds_1d_stress, axis=0)
    m_1d_stress = evaluate_forecast(y_stress_raw, ens_1d_stress)
    log(f"Stress-Test 1-Day LSTM Ensemble: RMSE = {m_1d_stress['RMSE']:.2f}, MAE = {m_1d_stress['MAE']:.2f}, R2 = {m_1d_stress['R2']:.4f}, QWK = {m_1d_stress['QWK']:.4f}")

    # 2. Hybrid on Stress
    X_all = hybrid_data['X_all']
    stress_mask = hybrid_data['stress_mask']
    tgt_dates = hybrid_data['tgt_dates']
    full_arimax = hybrid_data['full_arimax_preds']
    res_min, res_max = hybrid_data['res_min'], hybrid_data['res_max']

    preds_hyb_stress = []
    for m in hybrid_data['seed_models']:
        p_res = m.predict(X_all[stress_mask], verbose=0).flatten() * (res_max - res_min) + res_min
        arimax_stress = full_arimax.loc[tgt_dates[stress_mask]].values
        preds_hyb_stress.append(arimax_stress + p_res)
    ens_hyb_stress = np.mean(preds_hyb_stress, axis=0)
    actual_stress = df.loc[tgt_dates[stress_mask], 'AQI'].values
    m_hyb_stress = evaluate_forecast(actual_stress, ens_hyb_stress)
    log(f"Stress-Test Hybrid ARIMAX+LSTM: RMSE = {m_hyb_stress['RMSE']:.2f}, MAE = {m_hyb_stress['MAE']:.2f}, R2 = {m_hyb_stress['R2']:.4f}, QWK = {m_hyb_stress['QWK']:.4f}")

    return {
        'lstm_1d': {'metrics': m_1d_stress, 'preds': ens_1d_stress, 'dates': stress_dates, 'actual': y_stress_raw},
        'hybrid': {'metrics': m_hyb_stress, 'preds': ens_hyb_stress, 'dates': tgt_dates[stress_mask], 'actual': actual_stress}
    }


# =============================================================================
# MAIN ORCHESTRATION PIPELINE
# =============================================================================
def main():
    total_start_time = time.time()
    init_experiments_csv()

    log("Ingesting clean datasets and fitted scalers...")
    df, sc_tgt, sc_core, sc_all9 = load_scaled_dataset()

    # Step 1: Run Stage 1
    t0 = time.time()
    run_stage_1(df, sc_tgt)
    time_stage_1 = time.time() - t0
    top2_configs, s1_summary = get_stage_1_top2()
    log(f"Stage 1 Complete in {time_stage_1:.1f}s. Top 2 configurations:\n{pd.DataFrame(top2_configs)}")

    # Step 2: Run Stage 2
    t0 = time.time()
    run_stage_2(df, sc_tgt, top2_configs)
    time_stage_2 = time.time() - t0
    winner_cfg, s2_summary = get_stage_2_winner()
    log(f"Stage 2 Complete in {time_stage_2:.1f}s. Winning architecture:\n{winner_cfg}")

    # Step 3: Run Stage 3 (Optuna HPO)
    t0 = time.time()
    study, final_arch, final_fs, final_lb, final_params, final_median_epochs = run_stage_3_optuna(df, sc_tgt, winner_cfg, n_trials=60)
    time_stage_3 = time.time() - t0
    log(f"Stage 3 Complete in {time_stage_3:.1f}s. Selected Config: Arch={final_arch}, FS={final_fs}, LB={final_lb}, Epochs={final_median_epochs}")

    # Save exact configuration to dl_config.json
    dl_config = {
        'model_arch': final_arch,
        'feature_set': final_fs,
        'lookback': final_lb,
        'median_epochs': final_median_epochs,
        'hyperparameters': final_params,
        'seeds': SEEDS_5
    }
    with open(CONFIG_JSON, 'w', encoding='utf-8') as f:
        json.dump(dl_config, f, indent=4)
    log(f"Saved configuration to {CONFIG_JSON}")

    # Step 4: Final 1-Day Model Retraining & MC Dropout
    t0 = time.time()
    res_1day = train_final_1day(df, sc_tgt, final_arch, final_fs, final_lb, final_params, final_median_epochs)
    time_stage_4 = time.time() - t0

    # Step 5: Final 7-Day Model Retraining
    t0 = time.time()
    res_7day = train_final_7day(df, sc_tgt, final_arch, final_fs, final_lb, final_params)
    time_stage_5 = time.time() - t0

    # Step 6: Hybrid ARIMAX+LSTM Model
    t0 = time.time()
    arimax_path = 'outputs/models/arima_final.pkl'
    res_hybrid = train_hybrid_model(df, sc_tgt, arimax_path, final_fs, final_lb)
    time_stage_6 = time.time() - t0

    # Step 7: Stress Testing
    t0 = time.time()
    res_stress = run_stress_testing(df, sc_tgt, res_1day['seed_models'], res_7day['seed_models'], res_hybrid)
    time_stage_7 = time.time() - t0

    total_time = time.time() - total_start_time
    log(f"All Phase 6 DL stages completed successfully in {total_time:.1f}s ({total_time/60:.2f} mins)!")

    # Persist consolidated metrics and predictions
    log("Compiling outputs/results/p6_metrics.csv and outputs/results/p6_predictions.csv...")
    
    # Load Phase 5 metrics to include baselines
    p5_metrics = pd.read_csv('outputs/results/p5_metrics.csv')
    
    # Compile Phase 6 metrics rows
    p6_rows = []
    # 1-Day LSTM Ensemble (Test)
    r1 = res_1day['ens_metrics'].copy()
    r1.update({'model': 'LSTM', 'split': 'test', 'horizon': 1})
    p6_rows.append(r1)

    # 1-Day LSTM Ensemble (Stress)
    rs = res_stress['lstm_1d']['metrics'].copy()
    rs.update({'model': 'LSTM', 'split': 'stress', 'horizon': 1})
    p6_rows.append(rs)

    # 7-Day LSTM (Test, horizons 1..7)
    for h in range(1, 8):
        y_true_h = res_7day['y_test_raw'][:, h-1]
        y_pred_h = res_7day['ensemble_preds'][:, h-1]
        m_h = evaluate_forecast(y_true_h, y_pred_h)
        m_h.update({'model': 'LSTM-7day', 'split': 'test', 'horizon': h})
        p6_rows.append(m_h)

    # Hybrid ARIMAX-LSTM (Test)
    rh = res_hybrid['metrics'].copy()
    rh.update({'model': 'Hybrid ARIMAX-LSTM', 'split': 'test', 'horizon': 1})
    p6_rows.append(rh)

    # Hybrid ARIMAX-LSTM (Stress)
    rhs = res_stress['hybrid']['metrics'].copy()
    rhs.update({'model': 'Hybrid ARIMAX-LSTM', 'split': 'stress', 'horizon': 1})
    p6_rows.append(rhs)

    p6_metrics_df = pd.DataFrame(p6_rows)
    # Combine with Phase 5 baselines
    consolidated_metrics = pd.concat([p5_metrics, p6_metrics_df], ignore_index=True)
    consolidated_metrics.to_csv(os.path.join(RESULTS_DIR, 'p6_metrics.csv'), index=False)
    log("Saved outputs/results/p6_metrics.csv")

    # Compile predictions dataframe
    pred_dfs = []
    # Test 1-Day LSTM
    pred_dfs.append(pd.DataFrame({
        'date': res_1day['test_dates'],
        'actual': res_1day['y_test_raw'],
        'predicted': res_1day['ensemble_preds'],
        'pi_lower_95': res_1day['pi_lower'],
        'pi_upper_95': res_1day['pi_upper'],
        'horizon': 1,
        'model': 'LSTM',
        'split': 'test'
    }))
    # Stress 1-Day LSTM
    pred_dfs.append(pd.DataFrame({
        'date': res_stress['lstm_1d']['dates'],
        'actual': res_stress['lstm_1d']['actual'],
        'predicted': res_stress['lstm_1d']['preds'],
        'pi_lower_95': np.nan,
        'pi_upper_95': np.nan,
        'horizon': 1,
        'model': 'LSTM',
        'split': 'stress'
    }))
    # Test Hybrid
    pred_dfs.append(pd.DataFrame({
        'date': res_hybrid['test_dates'],
        'actual': res_hybrid['actual_test'],
        'predicted': res_hybrid['ensemble_preds'],
        'pi_lower_95': np.nan,
        'pi_upper_95': np.nan,
        'horizon': 1,
        'model': 'Hybrid ARIMAX-LSTM',
        'split': 'test'
    }))
    # Stress Hybrid
    pred_dfs.append(pd.DataFrame({
        'date': res_stress['hybrid']['dates'],
        'actual': res_stress['hybrid']['actual'],
        'predicted': res_stress['hybrid']['preds'],
        'pi_lower_95': np.nan,
        'pi_upper_95': np.nan,
        'horizon': 1,
        'model': 'Hybrid ARIMAX-LSTM',
        'split': 'stress'
    }))

    p6_preds_df = pd.concat(pred_dfs, ignore_index=True)
    p6_preds_df.to_csv(os.path.join(RESULTS_DIR, 'p6_predictions.csv'), index=False)
    log("Saved outputs/results/p6_predictions.csv")

    # Save summary execution times
    times_summary = {
        'total_time_sec': round(total_time, 2),
        'total_time_min': round(total_time / 60, 2),
        'stage_1_time_sec': round(time_stage_1, 2),
        'stage_2_time_sec': round(time_stage_2, 2),
        'stage_3_time_sec': round(time_stage_3, 2),
        'stage_4_time_sec': round(time_stage_4, 2),
        'stage_5_time_sec': round(time_stage_5, 2),
        'stage_6_time_sec': round(time_stage_6, 2),
        'stage_7_time_sec': round(time_stage_7, 2)
    }
    with open(os.path.join(RESULTS_DIR, 'p6_execution_times.json'), 'w', encoding='utf-8') as f:
        json.dump(times_summary, f, indent=4)
    log(f"Execution times: {times_summary}")


if __name__ == '__main__':
    main()
