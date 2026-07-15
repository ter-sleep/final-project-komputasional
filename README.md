# Optimasi SVR Menggunakan PSO untuk Peramalan Inflasi Indonesia

## Deskripsi Project

Project ini merupakan Final Project mata kuliah **Rekayasa Komputasional** yang bertujuan untuk meramalkan tingkat inflasi bulanan Indonesia menggunakan **Support Vector Regression (SVR)** yang hyperparameternya dioptimasi dengan **Particle Swarm Optimization (PSO)**.

Model SVR-PSO dibandingkan dengan model baseline SVR menggunakan hyperparameter default untuk mengukur peningkatan akurasi peramalan.

---

## Anggota Kelompok

| No | Nama | NIM |
|----|------|-----|
| 1  | [Nama Anggota 1] | [NIM 1] |
| 2  | [Nama Anggota 2] | [NIM 2] |
| 3  | [Nama Anggota 3] | [NIM 3] |

---

## Metode

### 1. Data
- Data inflasi bulanan Indonesia dari Bank Indonesia, periode Januari 2010 – Maret 2026.
- Sumber: [Bank Indonesia – Data Inflasi](https://www.bi.go.id/id/statistik/indikator/data-inflasi.aspx)

### 2. Preprocessing
- Sliding window dengan 12 lag features (lag-1 hingga lag-12).
- Normalisasi menggunakan **MinMaxScaler** (skala 0–1).
- Split kronologis **80 : 20** tanpa shuffle untuk menjaga urutan waktu.

### 3. Baseline SVR
- Kernel: **RBF**
- Hyperparameter: nilai default scikit-learn (`C=1.0`, `gamma='scale'`, `epsilon=0.1`).
- Evaluasi: RMSE, MAE, MAPE pada test set.

### 4. Optimasi PSO
- Library: **pyswarms** (`GlobalBestPSO`)
- Jumlah partikel: 20, iterasi: 30
- Koefisien akselerasi: c1 = c2 = 1.5, inersia: w = 0.7
- Ruang pencarian:
  - `C` : [0.1 – 1000]
  - `gamma` : [0.0001 – 10]
  - `epsilon` : [0.0001 – 0.5]
- Fungsi fitness: rata-rata RMSE **TimeSeriesSplit 3-fold** pada training set.
- Kurva konvergensi disimpan di `hasil/grafik/konvergensi_pso.png`.

### 5. Evaluasi
- Tabel perbandingan Baseline vs SVR-PSO (RMSE, MAE, MAPE, % perbaikan) disimpan ke `hasil/perbandingan.csv`.
- Grafik batang perbandingan disimpan di `hasil/grafik/perbandingan_metrik.png`.

---

## Struktur Project

```
final-project-komputasional/
├── README.md
├── requirements.txt
├── data/
│   └── inflasi.csv          # Data inflasi bulanan (periode, inflasi)
├── src/
│   ├── preprocessing.py     # Load data, lag features, scaling, split
│   ├── baseline.py          # SVR baseline dengan hyperparameter default
│   ├── optimasi_pso.py      # SVR + PSO hyperparameter tuning
│   └── evaluasi.py          # Perbandingan metrik & visualisasi
├── notebooks/
│   └── main.ipynb           # Notebook Google Colab
└── hasil/
    ├── perbandingan.csv      # Tabel perbandingan metrik (auto-generated)
    └── grafik/              # Output grafik PNG (auto-generated)
```

---

## Cara Menjalankan

### Persyaratan
- Python 3.8+
- Instal dependensi:

```bash
pip install -r requirements.txt
```

### Menjalankan Script

```bash
# 1. Preprocessing data
python src/preprocessing.py

# 2. Baseline SVR
python src/baseline.py

# 3. Optimasi PSO
python src/optimasi_pso.py

# 4. Evaluasi & perbandingan
python src/evaluasi.py
```

### Menjalankan di Google Colab

1. Buka `notebooks/main.ipynb` di Google Colab.
2. Ikuti instruksi pada sel pertama untuk mengupload file data.
3. Jalankan semua sel secara berurutan.

---

## Hasil

Hasil evaluasi tersimpan di:
- `hasil/perbandingan.csv` – tabel metrik perbandingan
- `hasil/grafik/` – grafik konvergensi PSO dan perbandingan metrik

---

## Lisensi

Project ini dibuat untuk keperluan akademis.