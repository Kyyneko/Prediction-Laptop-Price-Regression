# Laporan Proyek Machine Learning - Prediksi Harga Laptop

## Domain Proyek

Laptop merupakan perangkat penting dalam mendukung produktivitas manusia di era digital, mulai dari keperluan pendidikan, pekerjaan profesional, hingga hiburan. Di tengah banyaknya variasi spesifikasi dan merek, konsumen kerap kesulitan menilai harga yang wajar untuk sebuah laptop. Dengan demikian, proyek ini bertujuan membangun sistem prediksi harga laptop berbasis machine learning yang dapat membantu pengguna dan pelaku industri teknologi dalam pengambilan keputusan. Sistem ini menganalisis berbagai fitur laptop untuk memberikan estimasi harga yang akurat.

**Mengapa masalah ini penting?**

- Konsumen dapat mengevaluasi _value-for-money_ dari produk secara lebih objektif.
- Produsen dapat menyusun strategi penetapan harga yang kompetitif dan berdasarkan data.
- Retailer dapat mengoptimalkan inventaris dan menetapkan harga berdasarkan spesifikasi dan tren pasar terkini.

**Referensi:**

- _Sebagai contoh, Anda bisa mencari artikel atau laporan pasar terbaru:_ "Mordor Intelligence (2024). _Laptop Market Size & Share Analysis - Growth Trends & Forecasts (2024 - 2029)_." atau sumber data konkret yang Anda gunakan.
- A. Géron, _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_, O'Reilly, 2019.
- Dokumentasi Scikit-learn: https://scikit-learn.org

## Business Understanding

#### Problem Statements

1.  Bagaimana cara mengembangkan model yang dapat memprediksi harga laptop secara akurat berdasarkan spesifikasi teknis (seperti CPU, RAM, penyimpanan, GPU) dan fitur lainnya (seperti merek, ukuran layar, garansi)?
2.  Algoritma machine learning mana yang paling optimal untuk memprediksi harga laptop, dengan mempertimbangkan akurasi, stabilitas, dan kemampuan generalisasi pada data baru?

#### Goals

1.  Mengembangkan model machine learning yang mampu memprediksi harga laptop dengan tingkat akurasi tinggi, berdasarkan analisis fitur-fitur teknis dan non-teknis laptop yang relevan.
2.  Mengidentifikasi dan memilih algoritma machine learning regresi yang paling optimal dan stabil dari beberapa kandidat algoritma, berdasarkan metrik evaluasi standar (seperti R², RMSE, MAE) dan hasil validasi silang, untuk tugas prediksi harga laptop.

### Solution Statements

- Membangun dan membandingkan beberapa model regresi: Linear Regression, Ridge Regression, Lasso Regression, Decision Tree Regressor, Random Forest Regressor, dan Gradient Boosting Regressor.
- Melakukan evaluasi komprehensif terhadap setiap model menggunakan metrik R², RMSE, dan MAE pada data latih dan data uji.
- Memilih model terbaik tidak hanya berdasarkan performa puncak pada data uji, tetapi juga memperhatikan kestabilan yang ditunjukkan melalui skor validasi silang (cross-validation) dan kemampuan generalisasi (menghindari overfitting).
- Menyediakan fungsi prediksi yang siap digunakan, yang menerima input spesifikasi laptop dari pengguna dan mengeluarkan estimasi harga berdasarkan model terbaik yang telah dilatih.

## Data Understanding

Dataset yang digunakan adalah `laptop_price_dataset.csv`, berisi kumpulan informasi detail mengenai spesifikasi dan harga berbagai laptop.
**Tautan Sumber Data:** Dataset diperoleh dari file lokal `laptop_price_dataset.csv`. (Jika dari sumber publik seperti Kaggle, sebutkan di sini).

**Jumlah Data Awal:**
Dataset awal terdiri dari **893 baris** (observasi laptop) dan **17 kolom** (fitur termasuk target).

**Kondisi Data Awal:**

- **Missing Values:** Setelah pemeriksaan awal, tidak ditemukan nilai hilang yang signifikan pada fitur-fitur utama yang akan digunakan. Beberapa kolom seperti 'Unnamed: 0.1' dan 'Unnamed: 0' (yang merupakan sisa indeks) akan dihapus.
- **Duplikat:** Tidak ditemukan data duplikat setelah proses pembersihan awal (0 duplikat teridentifikasi).

**Tautan Sumber Data:**
https://www.kaggle.com/code/ahmedessamsaber/laptop-price-prediction-dataset

**Uraian Seluruh Fitur pada Data Awal:**
Berikut adalah deskripsi untuk setiap fitur dalam dataset awal:

1.  `Unnamed: 0.1`: Indeks tambahan, kemungkinan dari proses penyimpanan sebelumnya.
2.  `Unnamed: 0`: Indeks, kemungkinan dari proses penyimpanan sebelumnya.
3.  `brand`: Merek atau pabrikan laptop (misalnya, ASUS, HP, Apple).
4.  `name`: Nama model spesifik dari laptop.
5.  `price`: Harga laptop (variabel target). Diasumsikan dalam Dolar AS ($) berdasarkan analisis output, namun memerlukan konfirmasi.
6.  `spec_rating`: Peringkat spesifikasi keseluruhan laptop yang diberikan (skala numerik).
7.  `processor`: Jenis dan model prosesor yang digunakan (misalnya, Intel Core i7, AMD Ryzen 5).
8.  `CPU`: Informasi lebih detail mengenai CPU (seringkali duplikasi atau bagian dari 'processor').
9.  `Ram`: Informasi mengenai RAM laptop, biasanya dalam format teks (misalnya, "8GB DDR4").
10. `Ram_type`: Jenis teknologi RAM yang digunakan (misalnya, DDR4, DDR5).
11. `ROM`: Informasi mengenai penyimpanan internal laptop, dalam format teks (misalnya, "512GB SSD", "1TB HDD").
12. `ROM_type`: Jenis teknologi penyimpanan internal (misalnya, SSD, HDD).
13. `GPU`: Jenis dan model kartu grafis yang digunakan (misalnya, NVIDIA GeForce RTX 3060, Intel Iris Xe).
14. `display_size`: Ukuran diagonal layar laptop dalam inci.
15. `resolution_width`: Lebar resolusi layar dalam piksel.
16. `resolution_height`: Tinggi resolusi layar dalam piksel.
17. `OS`: Sistem operasi yang terpasang pada laptop (misalnya, Windows, macOS, Linux).
18. `warranty`: Informasi mengenai masa garansi laptop (misalnya, 1 tahun, 2 tahun).

**Variabel Penting yang Digunakan dalam Pemodelan (setelah seleksi dan _engineering_):**

- **Fitur Kategorikal:** `brand`, `processor`, `Ram_type`, `ROM_type`, `GPU`, `OS`.
- **Fitur Numerik:** `spec_rating`, `display_size`, `ram_size` (hasil _engineering_), `rom_size` (hasil _engineering_), `resolution_width`, `resolution_height`, `warranty`.
- **Target variabel:** `price`.

**Visualisasi dan Exploratory Data Analysis (EDA):**
Analisis eksploratif data dilakukan untuk memahami distribusi data, hubungan antar variabel, dan mendapatkan wawasan awal.
_(Untuk visualisasi detail, lihat Gambar 1 dan 2 di bagian Lampiran)_

- **Distribusi harga laptop:** Histogram harga menunjukkan distribusi yang cenderung _right-skewed_, artinya mayoritas laptop berada pada rentang harga lebih rendah hingga menengah, dengan beberapa laptop memiliki harga yang sangat tinggi.
- **Rata-rata harga per brand:** Terdapat variasi harga yang signifikan antar merek. Merek seperti Razer dan Apple menunjukkan rata-rata harga yang jauh lebih tinggi dibandingkan merek lain, mengindikasikan segmentasi pasar premium.
- **Scatterplot harga dengan fitur lain:**
  - `spec_rating` vs `price`: Menunjukkan tren positif, di mana laptop dengan rating spesifikasi lebih tinggi umumnya memiliki harga lebih tinggi.
  - `display_size` vs `price`: Hubungannya kurang jelas dibandingkan `spec_rating`, menandakan ukuran layar saja mungkin bukan penentu utama harga atau memiliki interaksi kompleks dengan fitur lain.
- **Matriks Korelasi antar fitur numerik:** Fitur seperti `ram_size`, `total_pixels` (gabungan `resolution_width` dan `resolution_height`), dan `rom_size` menunjukkan korelasi positif yang cukup kuat dengan `price`.

## Data Preparation

1. Pembersihan Data Awal

- Menghapus kolom yang tidak relevan seperti 'Unnamed: 0.1' dan 'Unnamed: 0' yang merupakan artefak dari penyimpanan data.
- Memeriksa dan menangani data duplikat (ditemukan 0 duplikat).
- Memeriksa _missing values_; tidak ada _missing values_ yang ditemukan pada dataset yang digunakan.

2. Feature Engineering

Membuat fitur baru dari fitur yang sudah ada untuk meningkatkan representasi data dan potensi performa model:

- **`ram_size`**: Mengekstrak nilai numerik ukuran RAM (dalam GB) dari kolom 'Ram' yang berformat teks menggunakan regex. Nilai default 8GB diberikan jika ekstraksi gagal atau kolom tidak tersedia.
- **`rom_size`**: Mengekstrak nilai numerik ukuran ROM (dalam GB) dari kolom 'ROM' yang berformat teks, dengan konversi otomatis dari TB ke GB jika diperlukan. Nilai default 256GB diberikan jika ekstraksi gagal atau kolom tidak tersedia.
- **`total_pixels`**: Menghitung total piksel layar dari `resolution_width` dan `resolution_height` (`resolution_width * resolution_height`). Nilai default Full HD (1920x1080 = 2,073,600 piksel) digunakan jika kolom resolusi tidak tersedia.
- **`price_per_spec`**: Menghitung rasio harga terhadap `spec_rating`. Fitur ini dibuat untuk analisis eksploratif namun **tidak digunakan** sebagai fitur input dalam pemodelan akhir untuk menghindari _data leakage_.
- **`price_category`**: Mengkategorikan laptop berdasarkan rentang harga (Budget: <$500, Mid-range: $500-$999, High-end: $1000-$1999, Premium: ≥$2000). Fitur ini digunakan untuk stratifikasi saat pembagian data.

3. Seleksi Fitur

Memilih fitur-fitur yang akan digunakan untuk pemodelan berdasarkan ketersediaan di dataset:

- **Fitur Numerik**: `spec_rating`, `display_size`, `warranty`, `ram_size`, `rom_size`, dan jika tersedia: `resolution_width`, `resolution_height`
- **Fitur Kategorikal**: Dipilih dari `brand`, `processor`, `Ram_type`, `ROM_type`, `GPU`, `OS` (hanya yang tersedia di dataset yang digunakan)

4. Pembagian Data

Membagi dataset menjadi data latih (80%) dan data uji (20%) menggunakan `train_test_split` dengan `random_state=42`. Stratifikasi berdasarkan `price_category` dilakukan untuk memastikan distribusi harga yang representatif di kedua set.

5. Preprocessing Fitur

Menggunakan `ColumnTransformer` dalam `Pipeline`:

- **Standard Scaling**: Menerapkan `StandardScaler` pada semua fitur numerik untuk menstandarisasi skala (mean=0, std=1), penting untuk algoritma yang sensitif terhadap skala seperti Linear Regression dan algoritma berbasis jarak.
- **One-Hot Encoding**: Menerapkan pada semua fitur kategorikal untuk mengubahnya menjadi format numerik yang dapat diproses model, dengan parameter:
  - `drop='first'`: Menghindari multikolinieritas dengan menghapus satu kolom dummy
  - `handle_unknown='ignore'`: Menangani kategori baru yang mungkin muncul pada data uji

**Catatan Penting tentang Missing Values**

Setelah tahap pembersihan data awal dan feature engineering, semua fitur yang dipilih untuk pemodelan (baik numerik maupun kategorikal) tidak memiliki missing values. Hal ini karena:

- Dataset asli tidak memiliki missing values pada kolom-kolom kunci
- Fitur baru hasil feature engineering dibuat dengan nilai default yang sudah ditentukan
- Oleh karena itu, **tidak diperlukan langkah imputasi missing values** pada tahap preprocessing

**Alasan dilakukannya Data Preparation**

- Memastikan data bersih dari error, inkonsistensi, dan informasi yang tidak relevan
- Menciptakan fitur baru yang lebih informatif melalui ekstraksi dan transformasi data
- Mengubah data ke dalam format numerik yang sesuai untuk input algoritma machine learning
- Menstandarisasi skala fitur untuk meningkatkan stabilitas dan konvergensi algoritma
- Membagi data secara proporsional dan stratified untuk pelatihan dan evaluasi yang objektif
- Memastikan pipeline preprocessing dapat menangani data baru dengan konsisten

## Modeling

**Algoritma yang digunakan dan dievaluasi:**
Proses pemodelan melibatkan pelatihan dan evaluasi enam algoritma regresi yang berbeda untuk menemukan yang paling cocok untuk dataset ini:

1.  **Linear Regression:**
    - **Cara Kerja:** Model ini mencoba menemukan hubungan linear terbaik antara fitur input dan variabel target dengan menyesuaikan garis (atau hyperplane dalam dimensi lebih tinggi) yang meminimalkan jumlah kuadrat selisih antara nilai prediksi dan nilai aktual (Sum of Squared Errors).
2.  **Ridge Regression (dengan regularisasi L2):**
    - **Cara Kerja:** Mirip dengan Linear Regression, namun menambahkan _penalty term_ (L2-norm atau kuadrat dari besaran koefisien) ke fungsi biaya. Tujuannya adalah untuk mengecilkan koefisien fitur, sehingga mengurangi kompleksitas model dan membantu mengatasi multikolinearitas.
3.  **Lasso Regression (dengan regularisasi L1):**
    - **Cara Kerja:** Juga mirip dengan Linear Regression, tetapi menggunakan _penalty term_ L1-norm (nilai absolut dari besaran koefisien). Selain mengecilkan koefisien, Lasso dapat membuat beberapa koefisien menjadi nol, sehingga efektif untuk melakukan seleksi fitur secara otomatis.
4.  **Decision Tree Regressor:**
    - **Cara Kerja:** Membangun model prediksi dalam bentuk struktur pohon. Pohon ini dibuat dengan mempartisi dataset secara rekursif menjadi subset yang lebih kecil berdasarkan nilai fitur tertentu. Setiap _leaf node_ pada pohon merepresentasikan nilai prediksi (rata-rata dari target di _leaf_ tersebut).
5.  **Random Forest Regressor (ensemble):**
    - **Cara Kerja:** Merupakan metode _ensemble learning_ yang membangun banyak Decision Tree secara independen selama proses pelatihan (menggunakan teknik _bagging_ dan pemilihan fitur acak). Prediksi akhir adalah rata-rata dari prediksi semua pohon individu, yang membantu mengurangi varians dan _overfitting_.
6.  **Gradient Boosting Regressor (ensemble):**
    - **Cara Kerja:** Juga metode _ensemble learning_ yang membangun model (biasanya Decision Tree) secara sekuensial. Setiap model baru dilatih untuk mengoreksi kesalahan (_residuals_) dari model sebelumnya. Proses ini secara bertahap meningkatkan akurasi model.

**Parameter dan Pipeline:**

- Setiap model dilatih sebagai bagian dari `Pipeline` Scikit-learn yang mengintegrasikan langkah-langkah preprocessing (Standard Scaling untuk fitur numerik dan One-Hot Encoding untuk fitur kategorikal) dengan algoritma regresi itu sendiri. Ini memastikan konsistensi dalam penerapan transformasi data.
- Parameter `random_state=42` digunakan pada semua model dan `train_test_split` untuk memastikan reproduktifitas hasil. Untuk model ensemble (Random Forest dan Gradient Boosting), `n_estimators=100` (jumlah pohon) digunakan sebagai parameter awal.

**Pemilihan Model Terbaik dan Hasilnya:**
Setelah pelatihan, evaluasi pada data uji, dan perbandingan menggunakan cross-validation, **Ridge Regression** menunjukkan performa terbaik secara keseluruhan dari keenam model yang diuji:

- **Test R² (R-squared): 0.815**
- **Test RMSE (Root Mean Squared Error): 26,161** (dalam satuan mata uang target)
- **Test MAE (Mean Absolute Error): 15,269** (dalam satuan mata uang target)

_(Untuk perbandingan performa semua model, lihat Gambar 3 di bagian Lampiran)_

## Evaluation

**Metrik Evaluasi yang Digunakan:**
Kinerja model regresi dievaluasi menggunakan metrik standar berikut pada data uji:

- **R² (Coefficient of Determination):** Mengukur seberapa baik model menjelaskan variabilitas dalam data harga laptop. Nilai R² 0.815 berarti sekitar 81.5% variasi harga laptop dapat dijelaskan oleh fitur-fitur dalam model Ridge Regression.
- **RMSE (Root Mean Squared Error):** Merupakan akar dari rata-rata kuadrat kesalahan. RMSE sebesar 26,166 berarti, rata-rata, prediksi model memiliki kesalahan sekitar $26,161 (jika satuan adalah $) dari harga aktual. Memberikan bobot lebih pada kesalahan besar.
- **MAE (Mean Absolute Error):** Adalah rata-rata dari nilai absolut kesalahan. MAE sebesar 15,280 berarti, rata-rata, prediksi model menyimpang sebesar $15,269 dari harga aktual, tanpa memperhatikan arah kesalahan. Kurang sensitif terhadap outlier dibandingkan RMSE.

**Hasil Evaluasi Model-Model dan Pemilihan Model Terbaik:**
Semua enam model (Linear Regression, Ridge Regression, Lasso Regression, Decision Tree Regressor, Random Forest Regressor, dan Gradient Boosting Regressor) dilatih dan dievaluasi. Hasil metrik (Test R², Test RMSE, Test MAE, dan CV R² Mean) untuk setiap model dicatat dan dibandingkan (dirangkum dalam Tabel Hasil atau Gambar 3 di Lampiran).

Berdasarkan perbandingan komprehensif:

- **Ridge Regression** terpilih sebagai model terbaik dengan skor **Test R² = 0.815**, **Test RMSE = $26,166**, dan **Test MAE = $15,280**.
- Model ini juga menunjukkan stabilitas yang baik melalui **validasi silang 5-fold** pada data latih, dengan skor **CV R² Mean = 0.786** dan standar deviasi yang relatif rendah (**CV R² Std = 0.039**). Performa pada data uji (Test R²) sedikit lebih tinggi daripada CV R² Mean, namun perbedaan ini tidak signifikan dan tidak menunjukkan overfitting yang parah, terutama karena model Ridge dengan regularisasi cenderung lebih general. Random Forest dan Gradient Boosting juga menunjukkan performa yang baik, namun Ridge Regression dipilih karena kombinasi performa yang kuat, stabilitas, dan interpretasi yang relatif lebih sederhana dibandingkan model ensemble yang lebih kompleks.

**Analisis Visual Hasil Model Terbaik (Ridge Regression):**
_(Untuk visualisasi detail, lihat Gambar 4 di bagian Lampiran)_

- **Actual vs Predicted Prices Plot:** Sebagian besar titik data prediksi harga oleh model Ridge Regression berkumpul di sekitar garis diagonal, menunjukkan kesesuaian yang baik antara harga aktual dan harga prediksi.
- **Residuals Plot:** Plot residual (selisih antara aktual dan prediksi) menunjukkan sebaran titik yang relatif acak di sekitar garis nol, tanpa pola yang jelas. Ini mengindikasikan bahwa model tidak memiliki bias sistematis. Meskipun demikian, terdapat beberapa _outlier_ yang menunjukkan adanya kasus-kasus di mana model kesulitan memprediksi harga secara akurat.

**Hubungan dengan Business Understanding:**
Hasil evaluasi model secara langsung menjawab pertanyaan dan tujuan bisnis yang telah ditetapkan:

1.  **Menjawab Problem Statement 1 (Pengembangan Model Akurat):**
    Model Ridge Regression berhasil dikembangkan dan menunjukkan kemampuan untuk memprediksi harga laptop dengan akurasi yang baik (R² = 0.815). Ini berarti model dapat menjelaskan sekitar 81.5% variasi harga berdasarkan fitur yang diberikan.

2.  **Menjawab Problem Statement 2 (Algoritma Optimal):**
    Dari enam algoritma yang dievaluasi, Ridge Regression teridentifikasi sebagai yang paling optimal untuk dataset dan masalah ini, dengan mempertimbangkan keseimbangan antara akurasi (R², RMSE, MAE) dan stabilitas (hasil cross-validation).

3.  **Mencapai Goal 1 (Pengembangan Model Akurat):**
    Tercapai dengan pengembangan model Ridge Regression yang memberikan prediksi harga dengan R² 0.815.

4.  **Mencapai Goal 2 (Identifikasi Algoritma Optimal):**
    Tercapai. Ridge Regression dipilih sebagai algoritma terbaik setelah perbandingan komprehensif.

5.  **Mencapai Goal 3 (Memahami Faktor Signifikan):**
    Meskipun laporan ini berfokus pada pemilihan model, analisis feature importance dari model tree-based (Random Forest, Gradient Boosting) yang dieksplorasi dalam notebook, atau analisis koefisien dari Ridge Regression (setelah invers transformasi scaling), dapat memberikan wawasan tentang fitur mana (misalnya, `spec_rating`, `ram_size`, `rom_size`, `GPU` tertentu) yang paling berpengaruh terhadap harga. Bagian ini dapat diperdalam pada pengembangan selanjutnya.

6.  **Dampak Solution Statements:**
    - **Pembangunan dan perbandingan beberapa model regresi:** Ini sangat berdampak karena memungkinkan identifikasi Ridge Regression sebagai model terbaik dari beberapa alternatif, daripada hanya mengandalkan satu model.
    - **Evaluasi komprehensif (R², RMSE, MAE):** Memberikan dasar kuantitatif yang kuat untuk menilai dan membandingkan kinerja model, memastikan keputusan pemilihan model berbasis data.
    - **Pemilihan model terbaik berdasarkan performa dan stabilitas (CV):** Memastikan model yang dipilih tidak hanya berkinerja baik pada satu split data uji tetapi juga general dan stabil, mengurangi risiko overfitting.
    - **Penyediaan fungsi prediksi:** Fungsi `predict_laptop_price` yang dikembangkan (meskipun memerlukan perbaikan untuk contoh input tertentu) merupakan solusi konkret yang memungkinkan model digunakan secara praktis untuk estimasi harga.

---

## Kesimpulan

Model prediksi harga laptop berhasil dikembangkan dengan menggunakan algoritma **Ridge Regression**, yang mencapai akurasi tinggi dengan nilai **R² sebesar 0.815** pada data uji. Ini menunjukkan bahwa model mampu menjelaskan sebagian besar variabilitas harga laptop berdasarkan fitur-fitur yang diberikan. Proyek ini memberikan dasar yang solid dan alat yang berguna untuk:

- Estimasi harga laptop bagi konsumen dan penjual.
- Analisis pasar dan penentuan strategi _pricing_ bagi produsen dan retailer.
- _Benchmarking_ kompetitif antar produk laptop di pasar.

**Saran untuk Pengembangan Lebih Lanjut:**

- **Penambahan Fitur:** Mempertimbangkan fitur tambahan yang mungkin relevan seperti daya tahan baterai, bobot laptop, material bodi, tahun rilis model, atau bahkan sentimen pasar dari ulasan produk.
- **Hyperparameter Tuning:** Melakukan _tuning_ parameter yang lebih ekstensif pada model Ridge Regression (atau model lain yang menjanjikan) menggunakan teknik seperti `GridSearchCV` atau `RandomizedSearchCV` untuk potensi peningkatan performa lebih lanjut.
- **Perbaikan Fungsi Prediksi:** Fungsi prediksi contoh dalam notebook menghasilkan harga negatif untuk input tertentu. Ini menandakan perlunya investigasi dan perbaikan, terutama dalam penanganan fitur input baru (termasuk validasi dan memastikan semua fitur yang dibutuhkan model ada dengan nilai yang wajar) sebelum prediksi.
- **Analisis Fitur Mendalam (Goal 3):** Menganalisis koefisien dari model Ridge Regression (setelah data ditransformasi kembali ke skala asli jika memungkinkan atau dengan interpretasi pada data standar) atau feature importance dari model ensemble untuk lebih memahami dampak kuantitatif dari setiap fitur terhadap harga.
- **Penyesuaian `price_category`:** Merevisi ambang batas pada fungsi `categorize_price` agar lebih sesuai dengan distribusi harga aktual dataset untuk stratifikasi yang lebih bermakna atau analisis segmen harga yang lebih baik jika fitur ini ingin digunakan lebih lanjut.
- **Deployment:** Mengembangkan model menjadi sebuah API (_Application Programming Interface_) atau aplikasi web interaktif sederhana agar mudah diakses dan digunakan oleh pengguna akhir.

---

## Lampiran (Referensi Gambar)

**Gambar 1: Analisis Harga Laptop**
_([Histogram dan Bar Chart yang menunjukkan sebaran harga laptop dari dataset](img/analisisharga.png))_

**Gambar 2: Matriks Korelasi Fitur Numerik**
_([Heatmap yang menunjukkan korelasi antar fitur numerik dan harga](img/matrikskorelasi.png))_

**Gambar 3: Perbandingan Performa Model**
_([Grafik bar atau tabel yang membandingkan skor R², RMSE, atau MAE antar model yang diuji](img/perbandinganmodel.png))_

**Gambar 4: Evaluasi Model Terbaik (Ridge Regression) - Actual vs Predicted & Residuals Plot**
_([Scatter plot Actual vs Predicted dan plot Residuals untuk model terpilih](img/apvsprp.png))_

---

## Struktur Laporan

Laporan ini disusun secara sistematis untuk memudahkan pembaca memahami seluruh proses proyek, dengan struktur sebagai berikut:

1.  **Domain Proyek**
    Menjelaskan latar belakang, alasan pemilihan masalah, dan urgensi penyelesaiannya beserta referensi pendukung.

2.  **Business Understanding**
    Mendefinisikan pernyataan masalah, tujuan proyek, dan solusi yang diajukan lengkap dengan metrik evaluasi yang akan digunakan.

3.  **Data Understanding**
    Menguraikan sumber data, karakteristik dataset, variabel yang digunakan, serta hasil eksplorasi awal data dan visualisasi.

4.  **Data Preparation**
    Memaparkan proses pembersihan data, teknik feature engineering, seleksi fitur, pembagian dataset, dan preprocessing data.

5.  **Modeling**
    Menjelaskan algoritma yang digunakan, parameter yang diterapkan, pipeline yang dibangun, serta kelebihan dan kekurangan masing-masing model.

6.  **Evaluation**
    Menyajikan metrik evaluasi, hasil eksperimen, analisis performa model terbaik, dan visualisasi pendukung.

7.  **Kesimpulan dan Saran**
    Merangkum temuan utama, manfaat model, dan rekomendasi untuk pengembangan lebih lanjut.

8.  **Lampiran**
    Menyertakan gambar, grafik, dan dokumen pendukung yang memperkaya laporan.
