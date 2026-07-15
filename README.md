# Optimasi SVR Menggunakan PSO untuk Peramalan Inflasi Indonesia

Final Project Rekayasa Komputasional — Semester 4

## Anggota Kelompok
1. [Nama 1] — [NIM]
2. [Nama 2] — [NIM]
3. [Nama 3] — [NIM]

## Deskripsi
Membandingkan kinerja Support Vector Regression (SVR) default (baseline)
dengan SVR yang hyperparameter-nya (C, gamma, epsilon) dioptimasi
menggunakan Particle Swarm Optimization (PSO) untuk peramalan
inflasi bulanan Indonesia (2010–2026, sumber: Bank Indonesia).

## Metode
- **Metode dasar:** Forecasting — SVR (kernel RBF)
- **Metode optimasi:** Particle Swarm Optimization (pyswarms)
- **Metrik evaluasi:** RMSE, MAE, MAPE

## Struktur Project
```
├── data/inflasi.csv          # data inflasi bulanan BI (ganti dummy dengan data asli)
├── src/preprocessing.py      # load data, lag features, scaling, split
├── src/konversi_data_bi.py   # konversi file unduhan BI ke CSV
├── src/baseline.py           # SVR default (sebelum optimasi)
├── src/optimasi_pso.py       # SVR + PSO (sesudah optimasi)
├── src/evaluasi.py           # metrik & visualisasi perbandingan
├── notebooks/main.ipynb      # notebook Google Colab (semua fase)
└── hasil/                    # output metrik & grafik
```

## Cara Menjalankan
```bash
pip install -r requirements.txt
cd src
python baseline.py       # kinerja sebelum optimasi
python optimasi_pso.py   # kinerja sesudah optimasi
python evaluasi.py       # tabel & grafik perbandingan
```

> **Catatan:** File `data/inflasi.csv` saat ini berisi data dummy (2010–2011).
> Ganti dengan data asli dari [Bank Indonesia](https://www.bi.go.id/id/statistik/indikator/data-inflasi.aspx)
> menggunakan skrip `src/konversi_data_bi.py`.
