"""Fase 3: Optimasi hyperparameter SVR (C, gamma, epsilon) dengan PSO."""
import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from preprocessing import prepare
from evaluasi import evaluasi, plot_prediksi

np.random.seed(42)  # reproducibility

X_train, X_test, y_train, y_test, scaler, df, split = prepare()

# Batas ruang pencarian: [C, gamma, epsilon]
LB = np.array([0.1, 0.0001, 0.0001])   # lower bound
UB = np.array([1000., 10., 0.5])       # upper bound


def fitness(particles):
    """Fitness = rata-rata RMSE cross-validation (TimeSeriesSplit).
    PSO meminimalkan nilai ini."""
    scores = []
    tscv = TimeSeriesSplit(n_splits=3)   # CV khusus time series (tanpa data leakage)
    for p in particles:
        C, gamma, eps = p
        rmses = []
        for tr_idx, va_idx in tscv.split(X_train):
            m = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=eps)
            m.fit(X_train[tr_idx], y_train[tr_idx])
            pred = m.predict(X_train[va_idx])
            rmses.append(np.sqrt(mean_squared_error(y_train[va_idx], pred)))
        scores.append(np.mean(rmses))
    return np.array(scores)


if __name__ == '__main__':
    # Parameter PSO standar (Clerc & Kennedy): c1=c2=1.5, w=0.7
    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
    optimizer = ps.single.GlobalBestPSO(
        n_particles=20, dimensions=3, options=options, bounds=(LB, UB))

    best_cost, best_pos = optimizer.optimize(fitness, iters=30)
    C_opt, gamma_opt, eps_opt = best_pos
    print(f'\nHyperparameter optimal: C={C_opt:.4f}, '
          f'gamma={gamma_opt:.4f}, epsilon={eps_opt:.4f}')

    # Latih ulang dengan hyperparameter optimal, evaluasi di test set
    model = SVR(kernel='rbf', C=C_opt, gamma=gamma_opt, epsilon=eps_opt)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_test_asli = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_asli = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    evaluasi(y_test_asli, y_pred_asli, 'SVR-PSO',
             simpan='../hasil/pso_metrik.csv')
    plot_prediksi(df, split, y_test_asli, y_pred_asli,
                  'SVR-PSO (Sesudah Optimasi)',
                  '../hasil/grafik/pso_prediksi.png')

    # Kurva konvergensi PSO (bukti optimasi berjalan — wajib masuk laporan)
    plt.figure(figsize=(8, 4))
    plt.plot(optimizer.cost_history)
    plt.title('Kurva Konvergensi PSO')
    plt.xlabel('Iterasi')
    plt.ylabel('Fitness (RMSE CV)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../hasil/grafik/pso_konvergensi.png', dpi=150)
