"""
baseline.py
-----------
Modul untuk melatih dan mengevaluasi model SVR baseline menggunakan
hyperparameter default scikit-learn (kernel RBF, C=1.0, gamma='scale',
epsilon=0.1).

Metrik evaluasi: RMSE, MAE, MAPE pada test set.
"""

import numpy as np
import os
import sys
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Tambahkan direktori src ke path agar bisa mengimpor preprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocessing

# Seed untuk reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def hitung_mape(y_aktual: np.ndarray, y_prediksi: np.ndarray) -> float:
    """
    Menghitung Mean Absolute Percentage Error (MAPE).

    Parameters
    ----------
    y_aktual   : np.ndarray  Nilai aktual
    y_prediksi : np.ndarray  Nilai hasil prediksi

    Returns
    -------
    float : MAPE dalam persen (%)
    """
    # Hindari pembagian dengan nol
    mask = y_aktual != 0
    return np.mean(np.abs((y_aktual[mask] - y_prediksi[mask]) / y_aktual[mask])) * 100


def latih_svr_baseline(X_train: np.ndarray, y_train: np.ndarray) -> SVR:
    """
    Melatih model SVR dengan hyperparameter default (kernel RBF).

    Parameters
    ----------
    X_train : np.ndarray  Fitur data latih
    y_train : np.ndarray  Target data latih

    Returns
    -------
    SVR : Model SVR yang sudah dilatih
    """
    model = SVR(kernel="rbf")  # Hyperparameter default: C=1.0, gamma='scale', epsilon=0.1
    model.fit(X_train, y_train)
    print("[INFO] Model SVR baseline berhasil dilatih.")
    return model


def evaluasi_model(
    model: SVR,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler,
    nama_model: str = "Baseline SVR"
) -> dict:
    """
    Mengevaluasi model pada test set dan mengembalikan metrik.

    Parameters
    ----------
    model      : SVR     Model yang sudah dilatih
    X_test     : np.ndarray
    y_test     : np.ndarray  (nilai ternormalisasi)
    scaler     : MinMaxScaler  Scaler untuk inverse transform
    nama_model : str     Label model untuk display

    Returns
    -------
    dict : {'rmse': float, 'mae': float, 'mape': float, 'y_pred': np.ndarray}
    """
    # Prediksi pada test set
    y_pred_norm = model.predict(X_test)

    # Inverse transform ke skala asli
    y_test_asli = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_asli = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()

    # Hitung metrik
    rmse = np.sqrt(mean_squared_error(y_test_asli, y_pred_asli))
    mae = mean_absolute_error(y_test_asli, y_pred_asli)
    mape = hitung_mape(y_test_asli, y_pred_asli)

    print(f"\n--- Evaluasi {nama_model} ---")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MAPE : {mape:.4f}%")

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "y_pred": y_pred_asli,
        "y_test": y_test_asli,
    }


if __name__ == "__main__":
    # Tentukan path data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_csv = os.path.join(base_dir, "data", "inflasi.csv")

    # Preprocessing data
    X_train, X_test, y_train, y_test, scaler, df = preprocessing(path_csv)

    # Latih model baseline
    model_baseline = latih_svr_baseline(X_train, y_train)

    # Evaluasi model
    hasil = evaluasi_model(model_baseline, X_test, y_test, scaler, "Baseline SVR")

    print("\n[INFO] Model SVR baseline selesai dievaluasi.")
    print(f"[INFO] Hyperparameter: C={model_baseline.C}, "
          f"gamma={model_baseline.gamma}, epsilon={model_baseline.epsilon}")
