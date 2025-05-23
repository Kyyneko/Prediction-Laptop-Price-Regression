# Laporan Proyek Machine Learning - Prediksi Harga Laptop

## Domain Proyek

Laptop merupakan perangkat penting dalam mendukung produktivitas manusia di era digital, mulai dari keperluan pendidikan, pekerjaan profesional, hingga hiburan. Di tengah banyaknya variasi spesifikasi dan merek, konsumen kerap kesulitan menilai harga yang wajar untuk sebuah laptop. Dengan demikian, proyek ini bertujuan membangun sistem prediksi harga laptop berbasis machine learning yang dapat membantu pengguna dan pelaku industri teknologi dalam pengambilan keputusan. Sistem ini menganalisis berbagai fitur laptop untuk memberikan estimasi harga yang akurat.

**Mengapa masalah ini penting?**

- Konsumen dapat mengevaluasi _value-for-money_ dari produk secara lebih objektif.
- Produsen dapat menyusun strategi penetapan harga yang kompetitif dan berdasarkan data.
- Retailer dapat mengoptimalkan inventaris dan menetapkan harga berdasarkan spesifikasi dan tren pasar terkini.

**Referensi:**

- Statista (2023). _Global Laptop Market Value_. (Catatan: Sebaiknya gunakan referensi aktual dan relevan dengan tahun proyek)
- A. Géron, _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_, O'Reilly, 2019.
- Dokumentasi Scikit-learn: https://scikit-learn.org

## Business Understanding

### Problem Statements

1.  Bagaimana cara mengembangkan model yang dapat memprediksi harga laptop secara akurat berdasarkan spesifikasi teknis (seperti CPU, RAM, penyimpanan, GPU) dan fitur lainnya (seperti merek, ukuran layar, garansi)?
2.  Algoritma machine learning mana yang paling optimal untuk memprediksi harga laptop, dengan mempertimbangkan akurasi, stabilitas, dan kemampuan generalisasi pada data baru?

### Goals

1.  Mengembangkan dan melatih beberapa model prediksi harga laptop menggunakan teknik _supervised learning_ (regresi) pada dataset yang telah diproses.
2.  Menentukan model terbaik berdasarkan evaluasi performa metrik standar (R², RMSE, MAE) pada data uji dan hasil validasi silang untuk memastikan kestabilan prediksi.
3.  Memahami faktor-faktor atau fitur laptop yang paling signifikan mempengaruhi harganya.

### Solution Statements

- Membangun dan membandingkan beberapa model regresi: Linear Regression, Ridge Regression, Lasso Regression, Decision Tree Regressor, Random Forest Regressor, dan Gradient Boosting Regressor.
- Melakukan evaluasi komprehensif terhadap setiap model menggunakan metrik R², RMSE, dan MAE pada data latih dan data uji.
- Memilih model terbaik tidak hanya berdasarkan performa puncak pada data uji, tetapi juga memperhatikan kestabilan yang ditunjukkan melalui skor validasi silang (cross-validation) dan kemampuan generalisasi (menghindari overfitting).
- Menyediakan fungsi prediksi yang siap digunakan, yang menerima input spesifikasi laptop dari pengguna dan mengeluarkan estimasi harga berdasarkan model terbaik yang telah dilatih.

## Data Understanding

Dataset yang digunakan adalah `laptop_price_dataset.csv`, berisi kumpulan informasi detail mengenai spesifikasi dan harga berbagai laptop.

- Jumlah data: 893 baris/observasi laptop.
- Fitur awal: Terdiri dari beberapa kolom yang kemudian diproses menjadi 13 fitur (gabungan numerik dan kategorikal) yang digunakan untuk pemodelan.
- Target variabel: `price` (harga laptop, diasumsikan dalam satuan mata uang seperti Dolar AS ($) berdasarkan output, namun perlu dikonfirmasi skala dan mata uangnya).

**Kondisi Data Awal:**

- Dataset awalnya diperiksa untuk kolom yang tidak relevan (seperti sisa indeks 'Unnamed: 0'), nilai yang hilang (_missing values_), dan data duplikat.
- Setelah pembersihan awal, tidak ditemukan nilai hilang yang signifikan pada fitur-fitur yang akan digunakan, dan tidak ada data duplikat yang teridentifikasi.

**Variabel Penting yang Digunakan dalam Pemodelan:**

- **Fitur Kategorikal:** `brand`, `processor`, `Ram_type`, `ROM_type`, `GPU`, `OS`.
- **Fitur Numerik:** `spec_rating`, `display_size`, `ram_size` (hasil _engineering_), `rom_size` (hasil _engineering_), `resolution_width`, `resolution_height`, `warranty`.

**Visualisasi dan Exploratory Data Analysis (EDA):**
Analisis eksploratif data dilakukan untuk memahami distribusi data, hubungan antar variabel, dan mendapatkan wawasan awal.
_(Untuk visualisasi detail, lihat Gambar 1 dan 2 di bagian Lampiran)_

- **Distribusi harga laptop:** Histogram harga menunjukkan distribusi yang cenderung _right-skewed_, artinya mayoritas laptop berada pada rentang harga lebih rendah hingga menengah, dengan beberapa laptop memiliki harga yang sangat tinggi.
- **Rata-rata harga per brand:** Terdapat variasi harga yang signifikan antar merek. Merek seperti Razer dan Apple menunjukkan rata-rata harga yang jauh lebih tinggi dibandingkan merek lain, mengindikasikan segmentasi pasar premium.
- **Scatterplot harga dengan fitur lain:**
  - `spec_rating` vs `price`: Menunjukkan tren positif, di mana laptop dengan rating spesifikasi lebih tinggi umumnya memiliki harga lebih tinggi.
  - `display_size` vs `price`: Hubungannya kurang jelas dibandingkan `spec_rating`, menandakan ukuran layar saja mungkin bukan penentu utama harga atau memiliki interaksi kompleks dengan fitur lain.
- **Matriks Korelasi antar fitur numerik:** Fitur seperti `ram_size`, `total_pixels` (gabungan `resolution_width` dan `resolution_height`), dan `rom_size` menunjukkan korelasi positif yang cukup kuat dengan `price`. Fitur `price_per_spec` (hasil _engineering_) memiliki korelasi sangat tinggi (0.987) dengan `price`, namun fitur ini tidak digunakan dalam pemodelan untuk menghindari _data leakage_.

## Data Preparation

**Langkah-langkah yang dilakukan:**

1.  **Pembersihan Data:**
    - Menghapus kolom yang tidak relevan (misalnya, 'Unnamed: 0', 'Unnamed: 0.1') yang merupakan artefak dari penyimpanan data.
    - Memeriksa dan memastikan tidak ada data duplikat (ditemukan 0 duplikat setelah pengecekan).
    - Memeriksa _missing values_; tidak ada _missing values_ signifikan yang ditemukan pada fitur-fitur yang relevan setelah pembersihan awal.
2.  **Feature Engineering:** Membuat fitur baru dari fitur yang sudah ada untuk meningkatkan representasi data dan potensi performa model:
    - `ram_size`: Mengekstrak nilai numerik ukuran RAM (dalam GB) dari format teks.
    - `rom_size`: Mengekstrak nilai numerik ukuran ROM (dalam GB), dengan konversi dari TB ke GB jika perlu.
    - `total_pixels`: Menghitung total piksel layar dari `resolution_width` dan `resolution_height`.
    - `price_category`: Mengkategorikan laptop berdasarkan rentang harga (Budget, Mid-range, High-end, Premium). Fitur ini utamanya digunakan untuk stratifikasi saat pembagian data. _Observasi: Ambang batas fungsi ini perlu ditinjau karena semua laptop dalam dataset terkategorikan 'Premium', menandakan skala harga aktual mungkin berbeda dari asumsi awal fungsi._
3.  **Seleksi Fitur:** Memilih fitur-fitur yang akan digunakan untuk pemodelan (`numerical_features` dan `categorical_features`).
4.  **Pembagian Data:** Membagi dataset menjadi data latih (80%) dan data uji (20%) menggunakan `train_test_split`. Stratifikasi berdasarkan `price_category` dilakukan untuk memastikan distribusi harga yang representatif di kedua set.
5.  **Preprocessing Fitur:**
    - **One-Hot Encoding:** Menerapkan pada fitur kategorikal untuk mengubahnya menjadi format numerik yang dapat diproses model, dengan `drop='first'` untuk menghindari multikolinieritas.
    - **Standard Scaling:** Menerapkan `StandardScaler` pada fitur numerik untuk menstandarisasi skala fitur (mean 0, standar deviasi 1), penting untuk algoritma yang sensitif terhadap skala.

**Alasan dilakukannya Data Preparation:**

- Memastikan data bersih dari error, inkonsistensi, dan informasi yang tidak relevan.
- Mengubah data ke dalam format numerik yang sesuai untuk input sebagian besar algoritma machine learning.
- Potensi untuk meningkatkan informasi yang dapat diekstrak oleh model melalui pembuatan fitur baru yang lebih informatif.
- Menstandarisasi skala fitur untuk meningkatkan stabilitas dan konvergensi beberapa algoritma.
- Membagi data secara proporsional untuk pelatihan dan evaluasi yang objektif.

## Modeling

**Algoritma yang digunakan dan dievaluasi:**
Proses pemodelan melibatkan pelatihan dan evaluasi enam algoritma regresi yang berbeda untuk menemukan yang paling cocok untuk dataset ini:

- Linear Regression
- Ridge Regression (dengan regularisasi L2)
- Lasso Regression (dengan regularisasi L1)
- Decision Tree Regressor
- Random Forest Regressor (ensemble)
- Gradient Boosting Regressor (ensemble)

**Parameter dan Pipeline:**

- Setiap model dilatih sebagai bagian dari `Pipeline` Scikit-learn yang mengintegrasikan langkah-langkah preprocessing (Standard Scaling untuk fitur numerik dan One-Hot Encoding untuk fitur kategorikal) dengan algoritma regresi itu sendiri. Ini memastikan konsistensi dalam penerapan transformasi data.
- Parameter `random_state=42` digunakan pada semua model untuk memastikan reproduktifitas hasil. Untuk model ensemble (Random Forest dan Gradient Boosting), `n_estimators=100` digunakan.

**Pemilihan Model Terbaik dan Hasilnya:**
Setelah pelatihan dan evaluasi pada data uji, **Ridge Regression** menunjukkan performa terbaik secara keseluruhan:

- **Test R² (R-squared): 0.815**
- **Test RMSE (Root Mean Squared Error): 26,166** (dalam satuan mata uang target)
- **Test MAE (Mean Absolute Error): 15,280** (dalam satuan mata uang target)

**Kelebihan & Kekurangan Algoritma (Contoh):**

- **Ridge Regression:** Stabil dan efektif dalam menangani multikolinearitas (korelasi tinggi antar fitur prediktor) karena adanya penalti L2. Cenderung memberikan solusi yang lebih general.
- **Decision Tree Regressor:** Mudah diinterpretasikan dan dapat menangkap hubungan non-linier, namun sangat rentan terhadap _overfitting_ jika tidak diatur kedalamannya (pruning) atau digunakan dalam ensemble.
- **Random Forest dan Gradient Boosting:** Umumnya memberikan akurasi yang tinggi karena merupakan metode ensemble yang menggabungkan banyak model lemah. Namun, bisa lebih kompleks untuk diinterpretasikan (_black box_) dan memerlukan lebih banyak sumber daya komputasi.

_(Untuk perbandingan performa semua model, lihat Gambar 3 di bagian Lampiran)_

## Evaluation

**Metrik Evaluasi yang Digunakan:**
Kinerja model regresi dievaluasi menggunakan metrik standar berikut pada data uji:

- **R² (Coefficient of Determination):** Mengukur seberapa baik model menjelaskan variabilitas dalam data harga laptop. Nilai R² 0.815 berarti sekitar 81.5% variasi harga laptop dapat dijelaskan oleh fitur-fitur dalam model Ridge Regression.
- **RMSE (Root Mean Squared Error):** Merupakan akar dari rata-rata kuadrat kesalahan. RMSE sebesar 26,166 berarti, rata-rata, prediksi model memiliki kesalahan sekitar $26,166 (jika satuan adalah $) dari harga aktual. Memberikan bobot lebih pada kesalahan besar.
- **MAE (Mean Absolute Error):** Adalah rata-rata dari nilai absolut kesalahan. MAE sebesar 15,280 berarti, rata-rata, prediksi model menyimpang sebesar $15,280 dari harga aktual, tanpa memperhatikan arah kesalahan. Kurang sensitif terhadap outlier dibandingkan RMSE.

**Hasil Evaluasi Model Terbaik (Ridge Regression):**

- **Final Test R²:** 0.815
- **Final Test RMSE:** $26,166
- **Final Test MAE:** $15,280
- Model juga dievaluasi menggunakan **validasi silang 5-fold (5-fold cross-validation)** pada data latih, di mana Ridge Regression menunjukkan skor R² rata-rata yang baik (0.786) dengan standar deviasi yang rendah (0.039), mengindikasikan stabilitas dan kemampuan generalisasi yang baik.

**Analisis Visual Hasil Model Terbaik:**
_(Untuk visualisasi detail, lihat Gambar 4 di bagian Lampiran)_

- **Actual vs Predicted Prices Plot:** Sebagian besar titik data prediksi harga oleh model Ridge Regression berkumpul di sekitar garis diagonal, menunjukkan kesesuaian yang baik antara harga aktual dan harga prediksi.
- **Residuals Plot:** Plot residual (selisih antara aktual dan prediksi) menunjukkan sebaran titik yang relatif acak di sekitar garis nol, tanpa pola yang jelas. Ini mengindikasikan bahwa model tidak memiliki bias sistematis. Meskipun demikian, terdapat beberapa _outlier_ yang menunjukkan adanya kasus-kasus di mana model kesulitan memprediksi harga secara akurat.

---

## Kesimpulan

Model prediksi harga laptop berhasil dikembangkan dengan menggunakan algoritma **Ridge Regression**, yang mencapai akurasi tinggi dengan nilai **R² sebesar 0.815** pada data uji. Ini menunjukkan bahwa model mampu menjelaskan sebagian besar variabilitas harga laptop berdasarkan fitur-fitur yang diberikan. Proyek ini memberikan dasar yang solid dan alat yang berguna untuk:

- Estimasi harga laptop bagi konsumen dan penjual.
- Analisis pasar dan penentuan strategi _pricing_ bagi produsen dan retailer.
- _Benchmarking_ kompetitif antar produk laptop di pasar.

**Saran untuk Pengembangan Lebih Lanjut:**

- **Penambahan Fitur:** Mempertimbangkan fitur tambahan yang mungkin relevan seperti daya tahan baterai, bobot laptop, material bodi, tahun rilis model, atau bahkan sentimen pasar dari ulasan produk.
- **Hyperparameter Tuning:** Melakukan _tuning_ parameter yang lebih ekstensif pada model Ridge Regression (atau model lain yang menjanjikan) menggunakan teknik seperti `GridSearchCV` atau `RandomizedSearchCV` untuk potensi peningkatan performa lebih lanjut.
- **Perbaikan Fungsi Prediksi:** Fungsi prediksi contoh menghasilkan harga negatif, yang menandakan perlunya investigasi dan perbaikan, terutama dalam penanganan fitur input dan validasi data baru sebelum prediksi.
- **Analisis Fitur Mendalam:** Menganalisis koefisien dari model Ridge Regression untuk lebih memahami dampak kuantitatif dari setiap fitur terhadap harga (setelah transformasi).
- **Penyesuaian `price_category`:** Merevisi ambang batas pada fungsi `categorize_price` agar lebih sesuai dengan distribusi harga aktual dataset untuk stratifikasi yang lebih bermakna atau analisis segmen harga.
- **Deployment:** Mengembangkan model menjadi sebuah API (_Application Programming Interface_) atau aplikasi web interaktif sederhana agar mudah diakses dan digunakan oleh pengguna akhir.

---

## Lampiran ( Referensi Gambar)

**Gambar 1: Analisis Harga Laptop**
_([Histogram dan Bar Chart yang menunjukkan sebaran harga laptop dari dataset](img/analisisharga.png))_

**Gambar 2: Matriks Korelasi Fitur Numerik**
_([Heatmap yang menunjukkan korelasi antar fitur numerik dan harga](img/matrikskorelasi.png))_

**Gambar 3: Perbandingan Performa Model**
_([Grafik bar yang membandingkan skor R², RMSE, atau MAE antar model yang diuji](img/perbandinganmodel.png))_

**Gambar 4: Evaluasi Model Terbaik (Ridge Regression) - Actual vs Predicted & Residuals Plot**
_([Scatter plot Actual vs Predicted dan plot Residuals untuk model terpilih](img/apvsprp.png))_
