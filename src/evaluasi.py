"""
evaluasi.py
-----------
Modul untuk membandingkan performa model Baseline SVR vs SVR-PSO.

Output:
1. Tabel perbandingan metrik (RMSE, MAE, MAPE, % perbaikan)
   disimpan ke hasil/perbandingan.csv
2. Grafik batang perbandingan metrik
   disimpan ke hasil/grafik/perbandingan_metrik.png
"""

import numpy as np
import pandas as pd
import os
import sys
import matplotlib
matplotlib.use("Agg")  # Gunakan backend non-interaktif untuk menyimpan grafik
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Tambahkan direktori src ke path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocessing
from baseline import latih_svr_baseline, evaluasi_model, hitung_mape
from optimasi_pso import jalankan_pso, simpan_kurva_konvergensi

# Seed untuk reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def hitung_persen_perbaikan(nilai_baseline: float, nilai_baru: float):
    """
    Menghitung persentase perbaikan dari baseline ke model baru.

    Nilai positif berarti model baru lebih baik (metrik lebih kecil).

    Parameters
    ----------
    nilai_baseline : float  Metrik model baseline
    nilai_baru     : float  Metrik model baru

    Returns
    -------
    float : Persentase perbaikan (%), atau None jika baseline bernilai 0
    """
    if nilai_baseline == 0:
        # Persentase perbaikan tidak terdefinisi jika baseline = 0
        return None
    return ((nilai_baseline - nilai_baru) / nilai_baseline) * 100


def buat_tabel_perbandingan(hasil_baseline: dict, hasil_pso: dict) -> pd.DataFrame:
    """
    Membuat tabel perbandingan metrik antara Baseline SVR dan SVR-PSO.

    Parameters
    ----------
    hasil_baseline : dict  {'rmse', 'mae', 'mape'}
    hasil_pso      : dict  {'rmse', 'mae', 'mape'}

    Returns
    -------
    pd.DataFrame : Tabel perbandingan
    """
    perbaikan_rmse = hitung_persen_perbaikan(hasil_baseline["rmse"], hasil_pso["rmse"])
    perbaikan_mae  = hitung_persen_perbaikan(hasil_baseline["mae"],  hasil_pso["mae"])
    perbaikan_mape = hitung_persen_perbaikan(hasil_baseline["mape"], hasil_pso["mape"])

    # Bulatkan nilai perbaikan hanya jika tidak None
    def _bulatkan(val, desimal=2):
        return round(val, desimal) if val is not None else None

    tabel = pd.DataFrame({
        "Metrik": ["RMSE", "MAE", "MAPE (%)"],
        "Baseline SVR": [
            round(hasil_baseline["rmse"], 4),
            round(hasil_baseline["mae"],  4),
            round(hasil_baseline["mape"], 4),
        ],
        "SVR-PSO": [
            round(hasil_pso["rmse"], 4),
            round(hasil_pso["mae"],  4),
            round(hasil_pso["mape"], 4),
        ],
        "Perbaikan (%)": [
            _bulatkan(perbaikan_rmse),
            _bulatkan(perbaikan_mae),
            _bulatkan(perbaikan_mape),
        ],
    })
    return tabel


def simpan_tabel_csv(tabel: pd.DataFrame, output_dir: str):
    """
    Menyimpan tabel perbandingan ke file CSV.

    Parameters
    ----------
    tabel      : pd.DataFrame  Tabel perbandingan
    output_dir : str           Direktori output
    """
    os.makedirs(output_dir, exist_ok=True)
    path_output = os.path.join(output_dir, "perbandingan.csv")
    tabel.to_csv(path_output, index=False)
    print(f"[INFO] Tabel perbandingan disimpan ke: {path_output}")


def buat_grafik_perbandingan(tabel: pd.DataFrame, output_dir: str):
    """
    Membuat dan menyimpan grafik batang perbandingan metrik.

    Parameters
    ----------
    tabel      : pd.DataFrame  Tabel perbandingan
    output_dir : str           Direktori output grafik
    """
    os.makedirs(output_dir, exist_ok=True)
    path_output = os.path.join(output_dir, "perbandingan_metrik.png")

    metrik_list    = tabel["Metrik"].tolist()
    nilai_baseline = tabel["Baseline SVR"].tolist()
    nilai_pso      = tabel["SVR-PSO"].tolist()

    x = np.arange(len(metrik_list))
    lebar_batang = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Batang untuk Baseline SVR
    batang1 = ax.bar(x - lebar_batang / 2, nilai_baseline, lebar_batang,
                     label="Baseline SVR", color="steelblue", alpha=0.8, edgecolor="black")
    # Batang untuk SVR-PSO
    batang2 = ax.bar(x + lebar_batang / 2, nilai_pso, lebar_batang,
                     label="SVR-PSO", color="darkorange", alpha=0.8, edgecolor="black")

    # Tambahkan label nilai di atas batang
    for batang in [batang1, batang2]:
        for rect in batang:
            tinggi = rect.get_height()
            ax.annotate(
                f"{tinggi:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, tinggi),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9
            )

    ax.set_title("Perbandingan Metrik: Baseline SVR vs SVR-PSO\n"
                 "(Peramalan Inflasi Indonesia)", fontsize=14)
    ax.set_xlabel("Metrik", fontsize=12)
    ax.set_ylabel("Nilai Metrik", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrik_list, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path_output, dpi=150)
    plt.close()
    print(f"[INFO] Grafik perbandingan disimpan ke: {path_output}")


def buat_grafik_prediksi(hasil_baseline: dict, hasil_pso: dict,
                         df: pd.DataFrame, jumlah_lag: int, output_dir: str):
    """
    Membuat dan menyimpan grafik perbandingan nilai aktual vs prediksi
    pada test set untuk kedua model.

    Parameters
    ----------
    hasil_baseline : dict   Berisi 'y_test' dan 'y_pred'
    hasil_pso      : dict   Berisi 'y_pred'
    df             : pd.DataFrame  DataFrame asli dengan kolom 'periode'
    jumlah_lag     : int    Jumlah lag yang digunakan
    output_dir     : str    Direktori output
    """
    os.makedirs(output_dir, exist_ok=True)
    path_output = os.path.join(output_dir, "prediksi_test.png")

    y_test   = hasil_baseline["y_test"]
    y_pred_b = hasil_baseline["y_pred"]
    y_pred_p = hasil_pso["y_pred"]

    # Hitung indeks test set berdasarkan jumlah sampel total dan split 80:20
    n_total  = len(df) - jumlah_lag
    n_train  = int(n_total * 0.8)
    idx_test = range(n_train, n_total)

    # Ambil label periode untuk test set
    periode_test = df["periode"].iloc[
        [i + jumlah_lag for i in idx_test]
    ].dt.strftime("%Y-%m").tolist()

    plt.figure(figsize=(14, 6))
    plt.plot(periode_test, y_test,   label="Aktual",       color="black",
             linewidth=2, marker="o", markersize=4)
    plt.plot(periode_test, y_pred_b, label="Baseline SVR", color="steelblue",
             linewidth=1.5, linestyle="--")
    plt.plot(periode_test, y_pred_p, label="SVR-PSO",      color="darkorange",
             linewidth=1.5, linestyle="-.")

    plt.title("Perbandingan Prediksi Inflasi Indonesia (Test Set)", fontsize=14)
    plt.xlabel("Periode", fontsize=12)
    plt.ylabel("Inflasi (%)", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path_output, dpi=150)
    plt.close()
    print(f"[INFO] Grafik prediksi disimpan ke: {path_output}")


if __name__ == "__main__":
    # Tentukan path data dan output
    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_csv   = os.path.join(base_dir, "data", "inflasi.csv")
    hasil_dir  = os.path.join(base_dir, "hasil")
    grafik_dir = os.path.join(base_dir, "hasil", "grafik")

    # ---- 1. Preprocessing ----
    from preprocessing import JUMLAH_LAG
    X_train, X_test, y_train, y_test, scaler, df = preprocessing(path_csv)

    # ---- 2. Baseline SVR ----
    model_baseline = latih_svr_baseline(X_train, y_train)
    hasil_baseline = evaluasi_model(model_baseline, X_test, y_test, scaler, "Baseline SVR")

    # ---- 3. Optimasi PSO ----
    best_cost, best_pos, cost_history = jalankan_pso(X_train, y_train)

    # Simpan kurva konvergensi
    simpan_kurva_konvergensi(cost_history, grafik_dir)

    # Latih ulang SVR dengan hyperparameter terbaik
    model_pso = SVR(
        kernel="rbf",
        C=best_pos[0],
        gamma=best_pos[1],
        epsilon=best_pos[2]
    )
    model_pso.fit(X_train, y_train)
    hasil_pso = evaluasi_model(model_pso, X_test, y_test, scaler, "SVR-PSO")

    # ---- 4. Buat & simpan tabel perbandingan ----
    tabel = buat_tabel_perbandingan(hasil_baseline, hasil_pso)
    print("\n--- Tabel Perbandingan ---")
    print(tabel.to_string(index=False))

    simpan_tabel_csv(tabel, hasil_dir)
    buat_grafik_perbandingan(tabel, grafik_dir)
    buat_grafik_prediksi(hasil_baseline, hasil_pso, df, JUMLAH_LAG, grafik_dir)

    print("\n[INFO] Evaluasi lengkap selesai.")
