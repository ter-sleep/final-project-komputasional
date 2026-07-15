"""Preprocessing: load data inflasi, buat lag features, normalisasi, split."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

RANDOM_SEED = 42
N_LAGS = 12          # 12 bulan = satu siklus musiman tahunan
TRAIN_RATIO = 0.8    # split kronologis 80:20


def load_data(path='../data/inflasi.csv'):
    df = pd.read_csv(path)
    df['periode'] = pd.to_datetime(df['periode'])
    return df.sort_values('periode').reset_index(drop=True)


def create_lag_features(series, n_lags=N_LAGS):
    """Ubah time series menjadi supervised learning:
    fitur = inflasi t-1..t-12, target = inflasi bulan t."""
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i - n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def prepare(path='../data/inflasi.csv'):
    """Return X_train, X_test, y_train, y_test, scaler, df, split index."""
    df = load_data(path)
    values = df['inflasi'].values.reshape(-1, 1)

    scaler = MinMaxScaler()                       # normalisasi wajib untuk SVR
    scaled = scaler.fit_transform(values).flatten()

    X, y = create_lag_features(scaled)
    split = int(len(X) * TRAIN_RATIO)             # JANGAN shuffle time series
    return (X[:split], X[split:], y[:split], y[split:], scaler, df, split)
