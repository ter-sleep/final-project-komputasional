"""
preprocessing.py
----------------
Modul untuk memuat dan memproses data inflasi Indonesia.

Tahapan:
1. Load data CSV
2. Membuat fitur lag (sliding window 12 lag)
3. Normalisasi dengan MinMaxScaler
4. Split data secara kronologis 80:20 (tanpa shuffle)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# Seed untuk reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Jumlah lag yang digunakan sebagai fitur
JUMLAH_LAG = 12


def load_data(path_csv: str) -> pd.DataFrame:
    """
    Memuat data inflasi dari file CSV.

    Parameters
    ----------
    path_csv : str
        Path ke file CSV yang berisi kolom 'periode' dan 'inflasi'.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan kolom 'periode' (datetime) dan 'inflasi' (float).
    """
    df = pd.read_csv(path_csv)
    # Konversi kolom periode ke format datetime
    df["periode"] = pd.to_datetime(df["periode"])
    df = df.sort_values("periode").reset_index(drop=True)
    print(f"[INFO] Data berhasil dimuat: {len(df)} baris")
    return df


def buat_fitur_lag(series: np.ndarray, jumlah_lag: int = JUMLAH_LAG):
    """
    Membuat matriks fitur menggunakan sliding window (lag features).

    Setiap baris pada X berisi nilai inflasi t-1, t-2, ..., t-lag_max,
    dan y berisi nilai inflasi pada waktu t.

    Parameters
    ----------
    series : np.ndarray
        Array nilai inflasi satu dimensi.
    jumlah_lag : int
        Jumlah lag yang digunakan sebagai fitur.

    Returns
    -------
    X : np.ndarray, shape (n_sampel, jumlah_lag)
    y : np.ndarray, shape (n_sampel,)
    """
    X, y = [], []
    for i in range(jumlah_lag, len(series)):
        # Fitur: nilai inflasi dari t-lag_max hingga t-1
        X.append(series[i - jumlah_lag: i])
        # Target: nilai inflasi pada waktu t
        y.append(series[i])
    return np.array(X), np.array(y)


def preprocessing(path_csv: str, jumlah_lag: int = JUMLAH_LAG):
    """
    Pipeline preprocessing lengkap: load data, buat lag features,
    normalisasi, dan split kronologis 80:20.

    Catatan penting: scaler hanya di-fit pada data training untuk menghindari
    kebocoran data (data leakage) dari test set ke proses training.

    Parameters
    ----------
    path_csv : str
        Path ke file CSV inflasi.
    jumlah_lag : int
        Jumlah lag features (default 12).

    Returns
    -------
    X_train : np.ndarray
    X_test  : np.ndarray
    y_train : np.ndarray
    y_test  : np.ndarray
    scaler  : MinMaxScaler  (di-fit hanya pada data train)
    df      : pd.DataFrame  (dataframe asli)
    """
    # 1. Load data
    df = load_data(path_csv)
    nilai_inflasi = df["inflasi"].values.astype(float)

    # 2. Tentukan titik split berdasarkan panjang data raw (sebelum lag)
    #    Indeks split pada raw series: data dari 0 s.d. n_split_raw-1 untuk training.
    #    Dengan jumlah_lag lag features, total sampel = len(nilai_inflasi) - jumlah_lag.
    n_total = len(nilai_inflasi) - jumlah_lag
    n_train_raw = int(len(nilai_inflasi) * 0.8)  # Titik split pada raw series

    # 3. Fit scaler HANYA pada porsi training raw series (tanpa data leakage)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(nilai_inflasi[:n_train_raw].reshape(-1, 1))

    # 4. Transform seluruh series menggunakan scaler yang di-fit pada data train
    nilai_dinormalisasi = scaler.transform(nilai_inflasi.reshape(-1, 1)).flatten()

    # 5. Buat fitur lag (sliding window) dari seluruh series ternormalisasi
    X, y = buat_fitur_lag(nilai_dinormalisasi, jumlah_lag)
    print(f"[INFO] Total sampel setelah lag features: {len(X)}")

    # 6. Split kronologis 80:20 (tanpa shuffle untuk menjaga urutan waktu)
    n_train = int(len(X) * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print(f"[INFO] Ukuran data train : {X_train.shape[0]} sampel")
    print(f"[INFO] Ukuran data test  : {X_test.shape[0]} sampel")

    return X_train, X_test, y_train, y_test, scaler, df


if __name__ == "__main__":
    # Tentukan path data relatif terhadap lokasi skrip
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_csv = os.path.join(base_dir, "data", "inflasi.csv")

    X_train, X_test, y_train, y_test, scaler, df = preprocessing(path_csv)

    print("\n--- Ringkasan Preprocessing ---")
    print(f"Fitur (lag)    : {X_train.shape[1]}")
    print(f"Train samples  : {X_train.shape[0]}")
    print(f"Test samples   : {X_test.shape[0]}")
    print(f"y_train range  : [{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"y_test range   : [{y_test.min():.4f}, {y_test.max():.4f}]")
