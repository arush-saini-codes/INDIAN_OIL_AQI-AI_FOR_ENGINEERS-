"""Environment verification script for UCS321 EST AQI Forecasting Project.

Checks all core dependencies, fits a sample auto_arima model,
and trains a minimal Keras LSTM model for 1 epoch.
"""

import sys
import os

# Suppress excessive TensorFlow logging for clean verification output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    print("=" * 60)
    print("UCS321 EST Project: Environment Verification")
    print("=" * 60)
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version   : {sys.version.split()[0]}")
    print("-" * 60)

    # 1. Package Imports and Version Reporting
    packages = {}

    import numpy as np

    packages["numpy"] = np.__version__

    import pandas as pd

    packages["pandas"] = pd.__version__

    import scipy

    packages["scipy"] = scipy.__version__

    import matplotlib

    packages["matplotlib"] = matplotlib.__version__

    import seaborn as sns

    packages["seaborn"] = sns.__version__

    import sklearn

    packages["scikit-learn"] = sklearn.__version__

    import statsmodels

    packages["statsmodels"] = statsmodels.__version__

    import pmdarima

    packages["pmdarima"] = pmdarima.__version__

    import tensorflow as tf

    packages["tensorflow"] = tf.__version__

    import openpyxl

    packages["openpyxl"] = openpyxl.__version__

    import ipykernel

    packages["ipykernel"] = ipykernel.__version__

    print("Installed Package Versions:")
    for pkg, ver in packages.items():
        print(f"  - {pkg:<15}: {ver}")

    print("-" * 60)

    # 2. Hardware / GPU Detection
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow GPU Detection:")
    if gpus:
        print(f"  GPU(s) Available: {len(gpus)}")
        for gpu in gpus:
            print(f"    - {gpu}")
    else:
        print(
            "  No GPU detected. Running in standard CPU mode (expected on standard Windows setups)."
        )

    print("-" * 60)

    # 3. Test pmdarima auto_arima fit on synthetic series
    print("Testing pmdarima auto_arima...")
    np.random.seed(42)
    dummy_series = (
        np.sin(np.linspace(0, 20, 60)) + np.random.normal(0, 0.1, 60) + 50
    )
    try:
        model_arima = pmdarima.auto_arima(
            dummy_series,
            start_p=1,
            start_q=1,
            max_p=2,
            max_q=2,
            seasonal=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )
        print(
            f"  auto_arima fit SUCCESS! Selected order: ARIMA{model_arima.order}"
        )
    except Exception as e:
        print(f"  auto_arima fit FAILED: {e}")
        sys.exit(1)

    print("-" * 60)

    # 4. Test Keras LSTM Model on synthetic time-series data
    print("Testing TensorFlow/Keras LSTM...")
    try:
        # 32 samples, sequence length 5, 1 feature
        X_dummy = np.random.randn(32, 5, 1).astype(np.float32)
        y_dummy = np.random.randn(32, 1).astype(np.float32)

        lstm_model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(5, 1)),
                tf.keras.layers.LSTM(8),
                tf.keras.layers.Dense(1),
            ]
        )
        lstm_model.compile(optimizer="adam", loss="mse")
        history = lstm_model.fit(
            X_dummy, y_dummy, epochs=1, batch_size=16, verbose=0
        )
        loss = history.history["loss"][0]
        print(f"  Keras LSTM 1-epoch fit SUCCESS! Initial loss: {loss:.4f}")
    except Exception as e:
        print(f"  Keras LSTM fit FAILED: {e}")
        sys.exit(1)

    print("=" * 60)
    print("ALL ENVIRONMENT AND COMPATIBILITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()

