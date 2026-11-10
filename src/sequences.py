"""Data sequence and sliding window generator for Phase 6 Deep Learning.
Strictly leakage-free: scalers fitted on train only, sliding windows look back
into past information only, and samples are partitioned by target timestamp.
"""

import os
import joblib
import numpy as np
import pandas as pd

# Core pollutant mappings
CORE_POLLUTANTS = ['PM2.5', 'CO', 'O3', 'NO2', 'SO2']
ALL9_POLLUTANTS = ['PM2.5', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']
PARSIMONIOUS_POLLUTANTS = ['PM2.5', 'CO', 'O3']

SPLIT_RANGES = {
    'train': ('2016-01-01', '2018-12-31'),
    'val': ('2019-01-01', '2019-08-31'),
    'test': ('2019-09-01', '2020-02-29'),
    'stress': ('2020-03-01', '2020-06-30'),
    'train_val': ('2016-01-01', '2019-08-31')
}


def load_scaled_dataset(data_path='data/processed/chennai_clean_full.csv', models_dir='outputs/models'):
    """Load full series and create scaled feature matrices using saved train scalers."""
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df = df.sort_index()

    scaler_target = joblib.load(os.path.join(models_dir, 'scaler_target.joblib'))
    scaler_core = joblib.load(os.path.join(models_dir, 'scaler_features_core.joblib'))
    scaler_all9 = joblib.load(os.path.join(models_dir, 'scaler_features_all9.joblib'))

    # Target scaling
    aqi_scaled = scaler_target.transform(df[['AQI']]).flatten()
    df['AQI_scaled'] = aqi_scaled

    # Feature scaling
    core_scaled = scaler_core.transform(df[CORE_POLLUTANTS])
    for idx, col in enumerate(CORE_POLLUTANTS):
        df[f'{col}_scaled_core'] = core_scaled[:, idx]

    all9_scaled = scaler_all9.transform(df[ALL9_POLLUTANTS])
    for idx, col in enumerate(ALL9_POLLUTANTS):
        df[f'{col}_scaled_all9'] = all9_scaled[:, idx]

    return df, scaler_target, scaler_core, scaler_all9


def get_feature_columns(feature_set='C'):
    """Return feature column list for requested set: U, P, C, or A."""
    cyclical = ['doy_sin', 'doy_cos']
    if feature_set == 'U':
        return ['AQI_scaled'] + cyclical
    elif feature_set == 'P':
        return ['AQI_scaled', 'PM2.5_scaled_core', 'CO_scaled_core', 'O3_scaled_core'] + cyclical
    elif feature_set == 'C':
        return ['AQI_scaled', 'PM2.5_scaled_core', 'CO_scaled_core', 'O3_scaled_core', 
                'NO2_scaled_core', 'SO2_scaled_core'] + cyclical
    elif feature_set == 'A':
        return ['AQI_scaled'] + [f'{c}_scaled_all9' for c in ALL9_POLLUTANTS] + cyclical
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")


def make_sequences_1day(df, feature_set='C', lookback=14):
    """Generate 1-day ahead sliding window sequences.
    
    Inputs at origin t: features from t-L+1 ... t.
    Target: AQI_scaled at t+1.
    Splits partitioned by target date t+1.
    """
    feature_cols = get_feature_columns(feature_set)
    X_mat = df[feature_cols].values
    y_mat = df['AQI_scaled'].values
    y_raw = df['AQI'].values
    dates = df.index

    N = len(df)
    X_list, y_list, y_raw_list, target_dates, origin_dates = [], [], [], [], []

    for i in range(lookback - 1, N - 1):
        origin_t = dates[i]
        target_t = dates[i + 1]
        x_window = X_mat[i - lookback + 1 : i + 1]
        y_val = y_mat[i + 1]
        y_raw_val = y_raw[i + 1]

        X_list.append(x_window)
        y_list.append(y_val)
        y_raw_list.append(y_raw_val)
        target_dates.append(target_t)
        origin_dates.append(origin_t)

    X_all = np.array(X_list, dtype=np.float32)
    y_all = np.array(y_list, dtype=np.float32)
    y_raw_all = np.array(y_raw_list, dtype=np.float32)
    target_dates = pd.DatetimeIndex(target_dates)
    origin_dates = pd.DatetimeIndex(origin_dates)

    splits = {}
    for split_name, (st, en) in SPLIT_RANGES.items():
        mask = (target_dates >= pd.Timestamp(st)) & (target_dates <= pd.Timestamp(en))
        splits[split_name] = {
            'X': X_all[mask],
            'y': y_all[mask],
            'y_raw': y_raw_all[mask],
            'target_dates': target_dates[mask],
            'origin_dates': origin_dates[mask]
        }

    return splits


def make_sequences_7day(df, feature_set='C', lookback=14, origins_csv='outputs/results/p5_multistep_origins.csv'):
    """Generate 7-day direct multi-step sliding window sequences.
    
    Inputs at origin t: features from t-L+1 ... t.
    Targets: AQI_scaled at [t+1, ..., t+7].
    For test partition, strictly aligns with origins_csv.
    """
    feature_cols = get_feature_columns(feature_set)
    X_mat = df[feature_cols].values
    y_mat = df['AQI_scaled'].values
    y_raw = df['AQI'].values
    dates = df.index

    N = len(df)
    X_list, y_list, y_raw_list, origin_dates, target_dates_list = [], [], [], [], []

    for i in range(lookback - 1, N - 7):
        origin_t = dates[i]
        fut_dates = dates[i + 1 : i + 8]
        x_window = X_mat[i - lookback + 1 : i + 1]
        y_vals = y_mat[i + 1 : i + 8]
        y_raw_vals = y_raw[i + 1 : i + 8]

        X_list.append(x_window)
        y_list.append(y_vals)
        y_raw_list.append(y_raw_vals)
        origin_dates.append(origin_t)
        target_dates_list.append(fut_dates)

    X_all = np.array(X_list, dtype=np.float32)
    y_all = np.array(y_list, dtype=np.float32)
    y_raw_all = np.array(y_raw_list, dtype=np.float32)
    origin_dates = pd.DatetimeIndex(origin_dates)

    splits = {}
    for split_name in ['train', 'val', 'train_val', 'stress']:
        st, en = SPLIT_RANGES[split_name]
        # All 7 future target days must fall inside split
        mask = [
            (f[0] >= pd.Timestamp(st)) and (f[-1] <= pd.Timestamp(en))
            for f in target_dates_list
        ]
        mask = np.array(mask, dtype=bool)
        splits[split_name] = {
            'X': X_all[mask],
            'y': y_all[mask],
            'y_raw': y_raw_all[mask],
            'origin_dates': origin_dates[mask],
            'target_dates': [target_dates_list[idx] for idx in np.where(mask)[0]]
        }

    # For test split: align strictly with exact saved origins from Phase 5
    if os.path.exists(origins_csv):
        orig_df = pd.read_csv(origins_csv)
        req_origins = pd.to_datetime(orig_df['origin_date'])
        test_mask = origin_dates.isin(req_origins)
        splits['test'] = {
            'X': X_all[test_mask],
            'y': y_all[test_mask],
            'y_raw': y_raw_all[test_mask],
            'origin_dates': origin_dates[test_mask],
            'target_dates': [target_dates_list[idx] for idx in np.where(test_mask)[0]]
        }
    else:
        st, en = SPLIT_RANGES['test']
        mask = np.array([(f[0] >= pd.Timestamp(st)) and (f[-1] <= pd.Timestamp(en)) for f in target_dates_list], dtype=bool)
        splits['test'] = {
            'X': X_all[mask],
            'y': y_all[mask],
            'y_raw': y_raw_all[mask],
            'origin_dates': origin_dates[mask],
            'target_dates': [target_dates_list[idx] for idx in np.where(mask)[0]]
        }

    return splits

