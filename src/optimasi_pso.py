"""
optimasi_pso.py
---------------
Modul untuk mengoptimasi hyperparameter SVR menggunakan
Particle Swarm Optimization (PSO) dari library pyswarms.

Konfigurasi PSO:
- Optimizer  : GlobalBestPSO
- Partikel   : 20
- Iterasi    : 30
- c1 = c2    : 1.5 (koefisien akselerasi kognitif & sosial)
- w          : 0.7 (bobot inersia)
- Ruang cari : C [0.1, 1000], gamma [0.0001, 10], epsilon [0.0001, 0.5]

Fungsi fitness: rata-rata RMSE TimeSeriesSplit 3-fold pada training set.
Kurva konvergensi disimpan ke hasil/grafik/konvergensi_pso.png.
"""

import numpy as np
import os
import sys
import matplotlib
matplotlib.use("Agg")  # Gunakan backend non-interaktif untuk menyimpan grafik
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

import pyswarms as ps
from pyswarms.single import GlobalBestPSO

# Tambahkan direktori src ke path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocessing
from baseline import hitung_mape, evaluasi_model

# Seed untuk reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---- Konfigurasi PSO ----
JUMLAH_PARTIKEL = 20
JUMLAH_ITERASI = 30
KOEF_C1 = 1.5      # Koefisien akselerasi kognitif
KOEF_C2 = 1.5      # Koefisien akselerasi sosial
BOBOT_W = 0.7      # Bobot inersia

# Batas bawah dan atas ruang pencarian hyperparameter
# Urutan: [C, gamma, epsilon]
BATAS_BAWAH = [0.1,    0.0001, 0.0001]
BATAS_ATAS  = [1000.0, 10.0,   0.5   ]

# Jumlah fold untuk validasi silang time series
JUMLAH_FOLD = 3


def fungsi_fitness(params: np.ndarray, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """
    Fungsi fitness untuk PSO: rata-rata RMSE dari TimeSeriesSplit.

    Parameter PSO di setiap partikel adalah [C, gamma, epsilon].
    Nilai fitness yang lebih kecil berarti performa lebih baik.

    Parameters
    ----------
    params   : np.ndarray, shape (n_partikel, 3)
               Kolom 0: C, Kolom 1: gamma, Kolom 2: epsilon
    X_train  : np.ndarray  Data fitur training
    y_train  : np.ndarray  Data target training

    Returns
    -------
    np.ndarray, shape (n_partikel,) : Nilai RMSE rata-rata tiap partikel
    """
    n_partikel = params.shape[0]
    fitness_values = np.zeros(n_partikel)

    # Inisialisasi TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=JUMLAH_FOLD)

    for i in range(n_partikel):
        C_val      = params[i, 0]
        gamma_val  = params[i, 1]
        eps_val    = params[i, 2]

        rmse_tiap_fold = []
        for idx_train, idx_val in tscv.split(X_train):
            X_tr, X_val = X_train[idx_train], X_train[idx_val]
            y_tr, y_val = y_train[idx_train], y_train[idx_val]

            # Latih SVR dengan hyperparameter dari partikel
            model = SVR(kernel="rbf", C=C_val, gamma=gamma_val, epsilon=eps_val)
            model.fit(X_tr, y_tr)

            # Prediksi dan hitung RMSE
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            rmse_tiap_fold.append(rmse)

        # Rata-rata RMSE di semua fold sebagai nilai fitness
        fitness_values[i] = np.mean(rmse_tiap_fold)

    return fitness_values


def jalankan_pso(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Menjalankan PSO untuk mencari hyperparameter SVR terbaik.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray

    Returns
    -------
    best_cost  : float        Nilai fitness terbaik (RMSE terkecil)
    best_pos   : np.ndarray   Hyperparameter terbaik [C, gamma, epsilon]
    cost_history : list       Riwayat biaya terbaik per iterasi
    """
    print("[INFO] Memulai optimasi PSO...")
    print(f"       Partikel: {JUMLAH_PARTIKEL}, Iterasi: {JUMLAH_ITERASI}")
    print(f"       c1={KOEF_C1}, c2={KOEF_C2}, w={BOBOT_W}")

    # Opsi PSO
    opsi_pso = {"c1": KOEF_C1, "c2": KOEF_C2, "w": BOBOT_W}

    # Batas ruang pencarian
    bounds = (np.array(BATAS_BAWAH), np.array(BATAS_ATAS))

    # Inisialisasi GlobalBestPSO
    optimizer = GlobalBestPSO(
        n_particles=JUMLAH_PARTIKEL,
        dimensions=3,          # C, gamma, epsilon
        options=opsi_pso,
        bounds=bounds,
    )

    # Jalankan optimasi menggunakan fungsi fitness kustom
    best_cost, best_pos = optimizer.optimize(
        fungsi_fitness,
        iters=JUMLAH_ITERASI,
        X_train=X_train,
        y_train=y_train,
        verbose=True,
    )

    # Ambil riwayat konvergensi
    cost_history = optimizer.cost_history

    print(f"\n[INFO] Hyperparameter terbaik ditemukan:")
    print(f"       C       = {best_pos[0]:.4f}")
    print(f"       gamma   = {best_pos[1]:.6f}")
    print(f"       epsilon = {best_pos[2]:.6f}")
    print(f"       RMSE CV = {best_cost:.6f}")

    return best_cost, best_pos, cost_history


def simpan_kurva_konvergensi(cost_history: list, output_dir: str):
    """
    Menyimpan grafik kurva konvergensi PSO ke file PNG.

    Parameters
    ----------
    cost_history : list   Riwayat nilai fitness terbaik per iterasi
    output_dir   : str    Direktori output untuk grafik
    """
    os.makedirs(output_dir, exist_ok=True)
    path_output = os.path.join(output_dir, "konvergensi_pso.png")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cost_history) + 1), cost_history,
             color="steelblue", linewidth=2, marker="o", markersize=4)
    plt.title("Kurva Konvergensi PSO\n(Optimasi Hyperparameter SVR)", fontsize=14)
    plt.xlabel("Iterasi", fontsize=12)
    plt.ylabel("Nilai Fitness Terbaik (RMSE CV)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path_output, dpi=150)
    plt.close()
    print(f"[INFO] Kurva konvergensi disimpan ke: {path_output}")


if __name__ == "__main__":
    # Tentukan path data dan output
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_csv  = os.path.join(base_dir, "data", "inflasi.csv")
    grafik_dir = os.path.join(base_dir, "hasil", "grafik")

    # Preprocessing data
    X_train, X_test, y_train, y_test, scaler, df = preprocessing(path_csv)

    # Jalankan PSO
    best_cost, best_pos, cost_history = jalankan_pso(X_train, y_train)

    # Simpan kurva konvergensi
    simpan_kurva_konvergensi(cost_history, grafik_dir)

    # Latih ulang SVR dengan hyperparameter terbaik pada seluruh data training
    C_opt      = best_pos[0]
    gamma_opt  = best_pos[1]
    eps_opt    = best_pos[2]

    model_pso = SVR(kernel="rbf", C=C_opt, gamma=gamma_opt, epsilon=eps_opt)
    model_pso.fit(X_train, y_train)
    print("\n[INFO] SVR-PSO berhasil dilatih ulang dengan hyperparameter terbaik.")

    # Evaluasi pada test set
    hasil_pso = evaluasi_model(model_pso, X_test, y_test, scaler, "SVR-PSO")

    print("\n[INFO] Optimasi dan evaluasi SVR-PSO selesai.")
