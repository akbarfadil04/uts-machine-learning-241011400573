# UTS Machine Learning: Linear Regression & Logistic Regression

| | |
|---|---|
| **Nama** | Akbar Fadil |
| **NIM** | 241011400573 |
| **Kelas** | Reg B |
| **Random state** | 73 (dua digit terakhir NIM) |

## Tujuan
Menerapkan Linear Regression (regresi) dan Logistic Regression (klasifikasi) pada dua dataset baru. Model, pembagian data, dan metrik evaluasi diimplementasikan sendiri dengan **NumPy** (library lain hanya Pandas dan Matplotlib). **scikit-learn hanya dipakai pada bagian bonus** untuk perbandingan.

## Dataset
| File | Kolom | Keterangan |
|---|---|---|
| `Ads_Data.csv` | `BiayaIklan`, `Penjualan` | 40 data. Biaya iklan per bulan (rentang 0,8–9,9 juta Rp) dan penjualan per bulan (33,2–121,1 juta Rp). |
| `Email_Data.csv` | `JumlahLink`, `KataPromo`, `Spam` | 60 data (35 bukan spam, 25 spam). Jumlah link dan kata promosi per email. |

## Bagian A: Linear Regression
Data dibagi 80:20 (32 training, 8 testing). Model dilatih dengan *normal equation* dan menghasilkan persamaan **ŷ = 29,0444 + 8,9912·X**. Artinya, setiap tambahan biaya iklan 1 juta Rp diikuti kenaikan penjualan sekitar 8,99 juta Rp.

| Metrik | Training | Testing |
|---|---|---|
| MAE | 4,9225 | 2,9905 |
| MSE | 41,5429 | 14,7567 |
| RMSE | 6,4454 | 3,8414 |
| R² | 0,9261 | 0,9785 |

Biaya iklan hampir linear terhadap penjualan, sehingga model cukup baik. Error testing yang lebih kecil dari training kemungkinan karena data testing hanya 8 titik, bukan karena model lebih bagus pada data baru.

**Prediksi:** 4,5 juta → 69,50 juta; 7 juta → 91,98 juta; 12 juta → 136,94 juta. Prediksi 12 juta adalah ekstrapolasi (di luar rentang 0,8–9,9 juta dan di atas penjualan tertinggi pada data), sehingga kurang bisa dipercaya.

### Eksperimen Gradient Descent (5000 iterasi)
| Learning rate | Intercept | Koefisien | MSE training akhir | Hasil |
|---|---|---|---|---|
| Normal equation | 29,0444 | 8,9912 | 41,5429 | referensi |
| 0,001 | 17,8372 | 10,6970 | 64,1871 | terlalu lambat, belum konvergen |
| 0,02 | 29,0444 | 8,9912 | 41,5429 | pas, sama dengan normal equation |
| 0,1 | NaN | NaN | NaN | divergen |

Pada learning rate 0,1 loss terus membesar dan menjadi tak hingga mulai iterasi ke-358 (muncul `RuntimeWarning: overflow`) karena langkah update terlalu besar sehingga melompati titik minimum. Hasil ini sengaja dibiarkan apa adanya di notebook sebagai bagian eksperimen.

## Bagian B: Logistic Regression
Data dibagi 75:25 (45 training, 15 testing) lalu dilatih dengan gradient descent (`learning_rate=0.1`, `n_iterations=10000`). Persamaan model: **P(Spam) = σ(−8,7729 + 0,7336·JumlahLink + 0,7335·KataPromo)**. Log loss turun dari 0,6931 menjadi 0,3154 dan hampir stabil. Decision boundary kira-kira x₁ + x₂ ≈ 12.

Hasil pada data testing (threshold 0,5): **accuracy 1,0000, precision 1,0000, recall 1,0000, F1 1,0000**.

| | Prediksi Bukan Spam | Prediksi Spam |
|---|---|---|
| **Aktual Bukan Spam** | 6 | 0 |
| **Aktual Spam** | 0 | 9 |

Hasil sempurna ini berasal dari data testing yang kecil (15 email) dan berisi 60% spam, sedangkan training sekitar 36% spam. Dua email testing punya probabilitas 0,507, sangat dekat dengan batas keputusan, jadi hasil ini tidak boleh dianggap jaminan performa pada email baru.

### Eksperimen threshold
| Threshold | FP | FN | Precision | Recall |
|---|---|---|---|---|
| 0,3 | 1 | 0 | 0,9000 | 1,0000 |
| 0,5 | 0 | 0 | 1,0000 | 1,0000 |
| 0,7 | 0 | 3 | 1,0000 | 0,6667 |

Threshold yang dipilih adalah **0,5**. Menurunkan threshold menaikkan FP (email penting bisa masuk spam), menaikkan threshold membuat spam lolos.

### Prediksi email baru (threshold 0,5)
| Email | JumlahLink | KataPromo | P(Spam) | Prediksi |
|---|---|---|---|---|
| E1 | 1 | 2 | 0,0014 | Bukan Spam |
| E2 | 5 | 6 | 0,3309 | Bukan Spam |
| E3 | 9 | 10 | 0,9943 | Spam |
| E4 | 2 | 11 | 0,6820 | Spam |

## Bonus: Perbandingan dengan scikit-learn (versi 1.8.0)
- **Linear Regression:** koefisien, intercept, dan semua metrik testing sama dengan implementasi sendiri (selisih sekitar 10⁻¹³, hanya galat pembulatan).
- **Logistic Regression:** hasil **berbeda** pada pengaturan default (L2, `C=1.0`), yaitu intercept −8,1802, m₁ = 0,6779, m₂ = 0,6833. Pada data testing, accuracy 0,8667, recall 0,7778, precision 1,0, F1 0,875 (2 email spam lolos). Penyebabnya adalah regularisasi L2 yang tidak ada pada implementasi sendiri.
- Dengan `C=1e10` (regularisasi hampir nol), semua metrik testing sklearn sama dengan milik sendiri (1,0), dan koefisien mendekati (−8,9231; 0,7467; 0,7460). Selisih kecil dengan implementasi sendiri (−8,7729; 0,7336; 0,7335) karena gradient descent 10.000 iterasi belum konvergen penuh. Dengan 100.000 iterasi hasilnya menjadi (−8,9232; 0,7467; 0,7460).

## Kesimpulan
Untuk data ini, hubungan biaya iklan dan penjualan cukup linear (R² testing 0,9785) dan filter spam sederhana dengan dua fitur dapat memisahkan kelas dengan baik. Meski begitu, kedua dataset kecil (8 dan 15 data testing) sehingga angka evaluasi sangat bergantung pada pembagian data dan belum tentu berlaku pada data baru. Eksperimen juga menunjukkan bahwa learning rate dan regularisasi sangat memengaruhi hasil.

## Struktur Repository
```
uts-machine-learning-241011400573/
├── README.md
├── tugas_UTS.ipynb   # notebook lengkap dengan output
├── Ads_Data.csv
└── Email_Data.csv
```

## Cara Menjalankan
Butuh Python 3 dengan `numpy`, `pandas`, `matplotlib`, dan `scikit-learn` (hanya bonus), lalu buka `tugas_UTS.ipynb` di VS Code atau Jupyter dan pilih *Run All*. Semua angka di README diambil dari output notebook ini.
