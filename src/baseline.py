"""Fase 2: SVR baseline (hyperparameter default) — kinerja SEBELUM optimasi."""
import numpy as np
from sklearn.svm import SVR
from preprocessing import prepare
from evaluasi import evaluasi, plot_prediksi

np.random.seed(42)  # reproducibility

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, scaler, df, split = prepare()

    # SVR default: C=1.0, gamma='scale', epsilon=0.1
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Kembalikan ke skala asli untuk interpretasi
    y_test_asli = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_asli = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    evaluasi(y_test_asli, y_pred_asli, 'SVR Baseline',
             simpan='../hasil/baseline_metrik.csv')
    plot_prediksi(df, split, y_test_asli, y_pred_asli,
                  'SVR Baseline (Sebelum Optimasi)',
                  '../hasil/grafik/baseline_prediksi.png')
