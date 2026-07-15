# LAPORAN FINAL PROJECT
# Optimasi Support Vector Regression (SVR) Menggunakan Particle Swarm Optimization (PSO) untuk Peramalan Inflasi Indonesia

**Mata Kuliah:** Rekayasa Komputasional
**Tahun:** 2026

**Anggota Kelompok:**
1. [Nama 1] — [NIM]
2. [Nama 2] — [NIM]
3. [Nama 3] — [NIM]

**Repository:** https://github.com/ter-sleep/final-project-komputasional

---

## BAB 1 — PENDAHULUAN

### 1.1 Latar Belakang
Inflasi merupakan salah satu indikator makroekonomi terpenting yang memengaruhi daya beli masyarakat, suku bunga, dan arah kebijakan moneter Bank Indonesia. Peramalan inflasi yang akurat sangat dibutuhkan oleh pemerintah, pelaku usaha, dan masyarakat dalam pengambilan keputusan ekonomi.

Metode machine learning seperti Support Vector Regression (SVR) telah terbukti unggul dalam memodelkan data time series yang bersifat non-linear seperti inflasi. Namun, kinerja SVR sangat sensitif terhadap pemilihan hyperparameter, yaitu parameter regularisasi (C), parameter kernel (gamma), dan lebar epsilon-tube (epsilon). Pemilihan hyperparameter secara manual (trial-and-error) tidak efisien dan tidak menjamin hasil optimal.

Particle Swarm Optimization (PSO) adalah algoritma optimasi metaheuristik berbasis perilaku kawanan (swarm intelligence) yang mampu mencari kombinasi hyperparameter optimal secara otomatis dan efisien. Oleh karena itu, penelitian ini mengombinasikan SVR dengan PSO untuk meningkatkan akurasi peramalan inflasi Indonesia.

### 1.2 Rumusan Masalah
1. Bagaimana kinerja SVR dengan hyperparameter default dalam meramal inflasi bulanan Indonesia?
2. Bagaimana pengaruh optimasi hyperparameter menggunakan PSO terhadap kinerja peramalan SVR?

### 1.3 Tujuan
1. Membangun model peramalan inflasi bulanan Indonesia berbasis Support Vector Regression.
2. Membandingkan kinerja SVR sebelum dan sesudah optimasi hyperparameter dengan PSO menggunakan metrik RMSE, MAE, dan MAPE.

### 1.4 Batasan Masalah
1. Data yang digunakan adalah data inflasi year-on-year (YoY) bulanan Indonesia dari Bank Indonesia, periode Januari 2010 – Juni 2026 (198 observasi).
2. Kernel SVR yang digunakan adalah Radial Basis Function (RBF).
3. Optimasi terbatas pada tiga hyperparameter: C, gamma, dan epsilon.

---

## BAB 2 — LANDASAN TEORI

### 2.1 Inflasi dan Peramalan Time Series
Inflasi adalah kenaikan harga barang dan jasa secara umum dan terus-menerus dalam jangka waktu tertentu. Peramalan time series memanfaatkan pola historis untuk memprediksi nilai masa depan. Salah satu teknik umum adalah sliding window (lag features), yaitu menggunakan n nilai sebelumnya sebagai fitur untuk memprediksi nilai berikutnya. Penelitian ini menggunakan 12 lag (12 bulan) untuk menangkap pola musiman tahunan.

### 2.2 Support Vector Regression (SVR)
SVR adalah pengembangan Support Vector Machine untuk kasus regresi. SVR mencari fungsi f(x) yang menyimpang maksimal sebesar epsilon dari target aktual (konsep epsilon-tube), sekaligus sedatar mungkin. Tiga hyperparameter utama SVR kernel RBF:
- **C** — parameter regularisasi; mengontrol trade-off antara kompleksitas model dan toleransi kesalahan. C kecil → model sederhana (cenderung underfitting); C besar → model kompleks (risiko overfitting).
- **gamma** — parameter kernel RBF; menentukan jangkauan pengaruh satu sampel data.
- **epsilon** — lebar tabung toleransi; kesalahan di dalam tabung tidak dipenalti.

### 2.3 Particle Swarm Optimization (PSO)
PSO (Kennedy & Eberhart, 1995) meniru perilaku kawanan burung dalam mencari makanan. Setiap partikel merepresentasikan satu kandidat solusi (kombinasi C, gamma, epsilon) yang bergerak di ruang pencarian berdasarkan:
- **pbest** — posisi terbaik yang pernah dicapai partikel itu sendiri;
- **gbest** — posisi terbaik yang pernah dicapai seluruh kawanan.

Pembaruan kecepatan dan posisi:

```
v(t+1) = w·v(t) + c1·r1·(pbest − x(t)) + c2·r2·(gbest − x(t))
x(t+1) = x(t) + v(t+1)
```

dengan w = inertia weight, c1 = koefisien kognitif, c2 = koefisien sosial, r1, r2 = bilangan acak [0,1].

### 2.4 Metrik Evaluasi
- **RMSE** = √( (1/n) Σ(yᵢ − ŷᵢ)² ) — menghukum kesalahan besar lebih berat.
- **MAE** = (1/n) Σ|yᵢ − ŷᵢ| — rata-rata kesalahan absolut.
- **MAPE** = (100/n) Σ|(yᵢ − ŷᵢ)/yᵢ| — kesalahan relatif dalam persen; sensitif terhadap nilai aktual mendekati nol.

---

## BAB 3 — METODOLOGI

### 3.1 Data
- **Sumber:** Bank Indonesia (https://www.bi.go.id/id/statistik/indikator/data-inflasi.aspx)
- **Jumlah:** 198 observasi bulanan (Januari 2010 – Juni 2026)
- **Praproses:** pembersihan format nilai ("3,34 %" → 3.34), konversi nama bulan Indonesia ke format tanggal, pengurutan kronologis.

### 3.2 Tahapan Penelitian

```
Data Inflasi BI (198 bulan)
        │
        ▼
Praproses & pembersihan data
        │
        ▼
Sliding window 12 lag → 186 sampel (fitur 12 bulan, target bulan ke-13)
        │
        ▼
Normalisasi MinMax [0,1]
        │
        ▼
Split kronologis 80:20 → Train 148 sampel, Test 38 sampel
        │
        ├────────────────────────┐
        ▼                        ▼
[A] SVR default          [B] SVR + tuning PSO
(C=1, gamma='scale',     (fitness: RMSE CV 3-fold
 epsilon=0.1)             TimeSeriesSplit)
        │                        │
        └───────────┬────────────┘
                    ▼
    Evaluasi & perbandingan pada data test
         (RMSE, MAE, MAPE)
```

Catatan penting: split dilakukan secara kronologis tanpa pengacakan (shuffle) untuk menghindari kebocoran data (data leakage) pada time series. Validasi silang di dalam fitness PSO menggunakan TimeSeriesSplit dengan alasan yang sama.

### 3.3 Konfigurasi Eksperimen

| Komponen | Nilai |
|---|---|
| SVR baseline | kernel RBF, C=1.0, gamma='scale', epsilon=0.1 |
| Ruang pencarian PSO | C ∈ [0.1, 1000], gamma ∈ [0.0001, 10], epsilon ∈ [0.0001, 0.5] |
| Parameter PSO | 20 partikel, 30 iterasi, c1 = c2 = 1.5, w = 0.7 |
| Fungsi fitness | rata-rata RMSE TimeSeriesSplit 3-fold pada data train |
| Lingkungan | Python 3 (Google Colab), scikit-learn, pyswarms, random seed = 42 |

---

## BAB 4 — HASIL DAN PEMBAHASAN

### 4.1 Eksplorasi Data
Grafik time series inflasi 2010–2026 menunjukkan pola yang fluktuatif: puncak sekitar 8,4% pada periode 2013–2014 (dampak kenaikan harga BBM), tren menurun hingga di bawah 2% pada 2020–2021 (pandemi), kenaikan kembali ke ±6% pada 2022–2023, lalu titik terendah mendekati 0% pada awal 2025 sebelum naik kembali ke ±3% pada 2026.

*(Lampirkan: grafik time series inflasi — hasil/grafik/)*

### 4.2 Hasil Optimasi PSO
Kurva konvergensi menunjukkan fitness turun tajam dari 0,143 menjadi ±0,108 hanya dalam 2 iterasi pertama, kemudian stabil (konvergen) di sekitar 0,107 hingga iterasi ke-30. Hal ini membuktikan PSO bekerja efektif menemukan area optimal dengan cepat.

Hyperparameter optimal hasil PSO:

| Hyperparameter | Default | Hasil PSO |
|---|---|---|
| C | 1.0 | **201.78** |
| gamma | 'scale' | **0.2295** |
| epsilon | 0.1 | **0.1448** |

*(Lampirkan: kurva konvergensi PSO)*

### 4.3 Perbandingan Kinerja

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| SVR Baseline | 0.6897 | 0.5247 | 87.01% |
| SVR-PSO | 0.7009 | 0.5539 | **43.36%** |
| Perubahan | −1.62% | −5.55% | **+50.17%** |

*(Lampirkan: grafik prediksi aktual vs baseline vs SVR-PSO, dan grafik batang perbandingan metrik)*

### 4.4 Pembahasan

**1. Perbaikan MAPE yang drastis (50,17%).** MAPE baseline sangat tinggi (87,01%) karena pada periode uji terdapat nilai inflasi aktual yang mendekati nol (awal 2025). Karena MAPE membagi kesalahan dengan nilai aktual, nilai aktual yang sangat kecil membuat kesalahan relatif membengkak. SVR-PSO berhasil memangkas MAPE hingga setengahnya (43,36%) karena prediksinya jauh lebih dekat mengikuti pergerakan aktual, termasuk pada periode inflasi rendah.

**2. Analisis visual: baseline underfitting.** Grafik prediksi memperlihatkan SVR baseline menghasilkan garis yang cenderung datar dan gagal menangkap dinamika inflasi — indikasi underfitting akibat nilai C=1 yang terlalu kecil (model terlalu "kaku"). Sebaliknya, SVR-PSO dengan C≈202 mengikuti pola turun-naik data aktual: penurunan menuju titik terendah awal 2025 dan kenaikan kembali pada 2025–2026.

**3. Trade-off RMSE/MAE.** RMSE dan MAE SVR-PSO sedikit lebih tinggi dibanding baseline. Ini terjadi karena model responsif berani mengikuti fluktuasi sehingga kadang overshoot pada titik balik, sedangkan model datar "aman" secara rata-rata namun tidak informatif untuk peramalan. Fenomena ini menunjukkan pentingnya tidak menilai model dari satu metrik saja.

**4. Implikasi.** Secara praktis, model SVR-PSO lebih bermakna untuk peramalan inflasi karena mampu mengantisipasi arah pergerakan (naik/turun), yang merupakan informasi utama bagi pengambil kebijakan — meskipun tidak seluruh metrik membaik. Pemilihan metrik evaluasi (dan fungsi fitness) memengaruhi kesimpulan dan arah optimasi.

---

## BAB 5 — PENUTUP

### 5.1 Kesimpulan
1. SVR dengan hyperparameter default menghasilkan RMSE 0.6897, MAE 0.5247, dan MAPE 87.01%, namun prediksinya cenderung datar dan gagal menangkap dinamika inflasi (underfitting).
2. Optimasi PSO berhasil menemukan hyperparameter optimal (C=201.78, gamma=0.2295, epsilon=0.1448) dan menurunkan MAPE sebesar 50.17% (dari 87.01% menjadi 43.36%), menghasilkan model yang mampu mengikuti pola pergerakan inflasi aktual.
3. Terdapat trade-off antara model konservatif (baseline, unggul tipis pada RMSE/MAE) dan model responsif (SVR-PSO, unggul jauh pada MAPE dan analisis visual).

### 5.2 Saran
1. Membandingkan PSO dengan algoritma metaheuristik lain seperti Genetic Algorithm (GA) atau Grey Wolf Optimizer (GWO).
2. Menambahkan variabel eksogen seperti nilai tukar rupiah, BI rate, dan harga pangan sebagai fitur tambahan.
3. Menguji fungsi fitness berbasis MAPE atau pendekatan multi-objective agar seluruh metrik membaik secara konsisten.

---

## DAFTAR PUSTAKA
1. Vapnik, V. (1995). *The Nature of Statistical Learning Theory*. Springer.
2. Kennedy, J., & Eberhart, R. (1995). Particle Swarm Optimization. *Proceedings of IEEE International Conference on Neural Networks*, 1942–1948.
3. Smola, A. J., & Schölkopf, B. (2004). A tutorial on support vector regression. *Statistics and Computing*, 14(3), 199–222.
4. Bank Indonesia. (2026). *Data Inflasi*. https://www.bi.go.id/id/statistik/indikator/data-inflasi.aspx
5. [Tambahkan 2–3 jurnal SVR-PSO forecasting berbahasa Indonesia/Inggris]

---

## LAMPIRAN
- **Lampiran A:** Repository GitHub — https://github.com/ter-sleep/final-project-komputasional
- **Lampiran B:** Kode inti (src/preprocessing.py, src/baseline.py, src/optimasi_pso.py, src/evaluasi.py)
- **Lampiran C:** Grafik lengkap (hasil/grafik/): time series inflasi, kurva konvergensi PSO, prediksi aktual vs model, perbandingan metrik
- **Lampiran D:** Notebook eksperimen (notebooks/main.ipynb)
