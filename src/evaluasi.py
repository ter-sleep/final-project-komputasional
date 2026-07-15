"""Fase 4: metrik evaluasi, plot, dan tabel perbandingan sebelum vs sesudah."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

N_LAGS = 12


def evaluasi(y_true, y_pred, label, simpan=None):
    """Hitung RMSE, MAE, MAPE dan simpan ke CSV (opsional)."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f'{label} -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}%')
    if simpan:
        pd.DataFrame([{'model': label, 'RMSE': rmse, 'MAE': mae,
                       'MAPE': mape}]).to_csv(simpan, index=False)
    return rmse, mae, mape


def plot_prediksi(df, split, y_true, y_pred, judul, path):
    """Plot aktual vs prediksi pada periode test."""
    tanggal = df['periode'].iloc[N_LAGS + split:]
    plt.figure(figsize=(12, 4))
    plt.plot(tanggal, y_true, label='Aktual', color='steelblue')
    plt.plot(tanggal, y_pred, label='Prediksi', color='tomato', ls='--')
    plt.title(f'Aktual vs Prediksi — {judul}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == '__main__':
    # Gabungkan hasil baseline & PSO menjadi tabel perbandingan
    base = pd.read_csv('../hasil/baseline_metrik.csv')
    pso = pd.read_csv('../hasil/pso_metrik.csv')
    tabel = pd.concat([base, pso], ignore_index=True)
    for m in ['RMSE', 'MAE', 'MAPE']:
        tabel[f'Perbaikan {m} (%)'] = (
            (tabel[m].iloc[0] - tabel[m]) / tabel[m].iloc[0] * 100).round(2)
    tabel.to_csv('../hasil/perbandingan.csv', index=False)
    print(tabel.to_string(index=False))

    # Grafik batang perbandingan metrik
    ax = tabel.set_index('model')[['RMSE', 'MAE', 'MAPE']].plot(
        kind='bar', figsize=(8, 4), rot=0)
    ax.set_title('Perbandingan Kinerja Sebelum vs Sesudah Optimasi')
    plt.tight_layout()
    plt.savefig('../hasil/grafik/perbandingan_metrik.png', dpi=150)
