# **Proyek Pertama** : Membuat Model Predictive Analytics
# - Nama         : Mahendra Kirana M.B
# - Email        : mahendrakirana284@gmail.com
# - ID Dicoding  : MC208D5Y1158
# ## **Prediksi Harga Laptop Berdasarkan Spesifikasinya dengan Menggunakan Metode Regresi**

# ############################################################################
# CELL 1: IMPORT LIBRARIES DAN SETUP
# ############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder # LabelEncoder tidak digunakan
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
import re # Import re yang digunakan di fungsi feature engineering

warnings.filterwarnings('ignore')

# Set style untuk visualisasi
plt.style.use('default')
sns.set_palette("husl")

print("✅ Libraries berhasil diimport")

# ############################################################################
# CELL 2: LOAD DAN EKSPLORASI DATA
# ############################################################################

# Load dataset
# Ganti 'laptop_price_dataset.csv' dengan path file Anda jika berbeda
try:
    df = pd.read_csv('laptop_price_dataset.csv')
except FileNotFoundError:
    print("❌ Error: File 'laptop_price_dataset.csv' tidak ditemukan. Pastikan file berada di direktori yang sama atau berikan path yang benar.")
    exit()


print("\n📊 INFORMASI DATASET")
print("="*50)
print(f"Jumlah baris: {df.shape[0]}")
print(f"Jumlah kolom: {df.shape[1]}")
print(f"\nKolom-kolom dalam dataset:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

# Tampilkan preview data
print("\n📋 PREVIEW DATA (5 baris pertama):")
print(df.head())

# ############################################################################
# CELL 3: DATA INFO DAN STATISTIK
# ############################################################################

# Info dataset
print("\n🔍 INFORMASI DETAIL DATASET:")
print("="*50)
df.info()

print("\n📈 STATISTIK DESKRIPTIF:")
print("="*50)
print(df.describe())

# ############################################################################
# CELL 4: DATA CLEANING
# ############################################################################

print("\n🧹 DATA CLEANING")
print("="*50)

# Drop kolom yang tidak diperlukan (jika ada)
columns_to_drop = []
if 'Unnamed: 0.1' in df.columns:
    columns_to_drop.append('Unnamed: 0.1')
if 'Unnamed: 0' in df.columns:
    columns_to_drop.append('Unnamed: 0')

if columns_to_drop:
    df_clean = df.drop(columns=columns_to_drop, axis=1)
    print(f"Dropped columns: {columns_to_drop}")
else:
    df_clean = df.copy()
    print("No unnamed columns to drop")

# Cek missing values
print("\nMissing values per kolom:")
print(df_clean.isnull().sum())

# Cek duplikasi
print(f"\nJumlah data duplikat: {df_clean.duplicated().sum()}")

# Hapus duplikasi jika ada
if df_clean.duplicated().sum() > 0:
    df_clean.drop_duplicates(inplace=True)
    print(f"Data duplikat telah dihapus. Jumlah data duplikat sekarang: {df_clean.duplicated().sum()}")


# Analisis target variable (price)
# Pastikan kolom 'price' ada sebelum mengaksesnya
if 'price' in df_clean.columns:
    print(f"\nAnalisis Target Variable (price):")
    print(f"Min price: ${df_clean['price'].min():,}")
    print(f"Max price: ${df_clean['price'].max():,}")
    print(f"Mean price: ${df_clean['price'].mean():.2f}")
    print(f"Median price: ${df_clean['price'].median():.2f}")
else:
    print("❌ Error: Kolom 'price' tidak ditemukan dalam dataset setelah cleaning.")
    # Anda mungkin ingin menghentikan eksekusi jika target variabel hilang
    # exit()

# ############################################################################
# CELL 5: EXPLORATORY DATA ANALYSIS - VISUALISASI
# ############################################################################
print("\n🎨 EXPLORATORY DATA ANALYSIS - VISUALISASI")
print("="*50)

# Pastikan kolom yang diperlukan untuk visualisasi ada
required_eda_cols = ['price', 'brand', 'spec_rating', 'display_size']
missing_eda_cols = [col for col in required_eda_cols if col not in df_clean.columns]

if not missing_eda_cols:
    # Set up plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Laptop Price Analysis - EDA', fontsize=16, fontweight='bold')

    # 1. Distribusi harga
    axes[0,0].hist(df_clean['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Distribusi Harga Laptop')
    axes[0,0].set_xlabel('Harga ($)')
    axes[0,0].set_ylabel('Frekuensi')
    axes[0,0].grid(True, alpha=0.3)

    # 2. Harga berdasarkan brand
    brand_price = df_clean.groupby('brand')['price'].mean().sort_values(ascending=False)
    axes[0,1].bar(range(len(brand_price)), brand_price.values, color='lightcoral')
    axes[0,1].set_title('Rata-rata Harga per Brand')
    axes[0,1].set_xlabel('Brand')
    axes[0,1].set_ylabel('Rata-rata Harga ($)')
    axes[0,1].set_xticks(range(len(brand_price)))
    axes[0,1].set_xticklabels(brand_price.index, rotation=45, ha='right')
    axes[0,1].grid(True, alpha=0.3)

    # 3. Korelasi spec_rating vs price
    axes[1,0].scatter(df_clean['spec_rating'], df_clean['price'], alpha=0.6, color='green')
    axes[1,0].set_title('Spec Rating vs Price')
    axes[1,0].set_xlabel('Spec Rating')
    axes[1,0].set_ylabel('Harga ($)')
    axes[1,0].grid(True, alpha=0.3)

    # 4. Display size vs price
    axes[1,1].scatter(df_clean['display_size'], df_clean['price'], alpha=0.6, color='orange')
    axes[1,1].set_title('Display Size vs Price')
    axes[1,1].set_xlabel('Display Size (inch)')
    axes[1,1].set_ylabel('Harga ($)')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to prevent suptitle overlap
    plt.show()
else:
    print(f"❌ Peringatan: Kolom berikut untuk EDA tidak ditemukan: {', '.join(missing_eda_cols)}. Visualisasi dilewati.")


# ############################################################################
# CELL 6: ANALISIS KATEGORIKAL
# ############################################################################

print("\n📋 ANALISIS VARIABEL KATEGORIKAL:")
print("="*50)

categorical_cols_analysis = ['brand', 'processor', 'Ram_type', 'ROM_type', 'GPU', 'OS']
for col in categorical_cols_analysis:
    if col in df_clean.columns:
        print(f"\n{col.upper()}:")
        print(f"Jumlah kategori unik: {df_clean[col].nunique()}")
        print("Top 5 kategori:")
        print(df_clean[col].value_counts().head())
    else:
        print(f"\nKolom '{col}' tidak ditemukan untuk analisis kategorikal.")

# ############################################################################
# CELL 7: FEATURE ENGINEERING
# ############################################################################

print("\n🔧 FEATURE ENGINEERING")
print("="*50)

# Buat copy dataset untuk feature engineering
df_features = df_clean.copy()

# 1. Extract RAM size
def extract_ram_size(ram_str):
    try:
        numbers = re.findall(r'\d+', str(ram_str))
        if numbers:
            return int(numbers[0])
        return 8  # default value
    except:
        return 8

if 'Ram' in df_features.columns:
    df_features['ram_size'] = df_features['Ram'].apply(extract_ram_size)
    print("- ram_size: Ukuran RAM dalam GB (diekstrak dari 'Ram')")
else:
    # Jika tidak ada kolom Ram, buat default jika kolom 'ram_size' belum ada
    if 'ram_size' not in df_features.columns:
        df_features['ram_size'] = 8
        print("- ram_size: Dibuat dengan nilai default 8 GB (kolom 'Ram' tidak ditemukan)")

# 2. Extract ROM size
def extract_rom_size(rom_str):
    try:
        rom_str_upper = str(rom_str).upper()
        numbers = re.findall(r'\d+', str(rom_str))
        if not numbers:
            return 256 # default jika tidak ada angka

        size = int(numbers[0])
        if 'TB' in rom_str_upper:
            return size * 1000  # Convert TB to GB
        return size # Asumsi GB jika tidak ada TB
    except:
        return 256

if 'ROM' in df_features.columns:
    df_features['rom_size'] = df_features['ROM'].apply(extract_rom_size)
    print("- rom_size: Ukuran ROM dalam GB (diekstrak dari 'ROM')")
else:
    if 'rom_size' not in df_features.columns:
        df_features['rom_size'] = 256
        print("- rom_size: Dibuat dengan nilai default 256 GB (kolom 'ROM' tidak ditemukan)")


# 3. Create resolution feature
if 'resolution_width' in df_features.columns and 'resolution_height' in df_features.columns:
    df_features['total_pixels'] = df_features['resolution_width'] * df_features['resolution_height']
    print("- total_pixels: Total resolusi (width × height)")
else:
    df_features['total_pixels'] = 1920 * 1080  # default FHD
    print("- total_pixels: Dibuat dengan nilai default FHD (1920x1080) (kolom resolusi tidak lengkap)")


# 4. Create price per spec ratio
# Pastikan 'price' dan 'spec_rating' ada
if 'price' in df_features.columns and 'spec_rating' in df_features.columns:
    # Hindari pembagian dengan nol jika spec_rating bisa 0
    df_features['price_per_spec'] = df_features.apply(
        lambda row: row['price'] / row['spec_rating'] if row['spec_rating'] != 0 else 0, axis=1
    )
    print("- price_per_spec: Rasio harga terhadap spec rating")
else:
    print("- Peringatan: 'price_per_spec' tidak dapat dibuat karena 'price' atau 'spec_rating' tidak ditemukan.")


# 5. Categorize price ranges
def categorize_price(price):
    if price < 500:
        return 'Budget'
    elif price < 1000:
        return 'Mid-range'
    elif price < 2000:
        return 'High-end'
    else:
        return 'Premium'

if 'price' in df_features.columns:
    df_features['price_category'] = df_features['price'].apply(categorize_price)
    print("- price_category: Kategori harga")
else:
    print("- Peringatan: 'price_category' tidak dapat dibuat karena 'price' tidak ditemukan.")


print("✅ Feature engineering selesai.")

# ############################################################################
# CELL 8: CORRELATION ANALYSIS
# ############################################################################

print("\n🔗 CORRELATION ANALYSIS")
print("="*50)

# Pilih kolom numerik untuk analisis korelasi
numeric_cols_corr = ['price', 'spec_rating', 'display_size', 'warranty',
                     'ram_size', 'rom_size', 'price_per_spec']

# Tambahkan kolom resolution jika ada dan belum ada di list
if 'resolution_width' in df_features.columns and 'resolution_width' not in numeric_cols_corr:
    numeric_cols_corr.append('resolution_width')
if 'resolution_height' in df_features.columns and 'resolution_height' not in numeric_cols_corr:
    numeric_cols_corr.append('resolution_height')
if 'total_pixels' in df_features.columns and 'total_pixels' not in numeric_cols_corr:
    numeric_cols_corr.append('total_pixels')


# Pastikan semua kolom ada
available_numeric_cols_corr = [col for col in numeric_cols_corr if col in df_features.columns]

if 'price' in available_numeric_cols_corr and len(available_numeric_cols_corr) > 1:
    correlation_matrix = df_features[available_numeric_cols_corr].corr()

    # Visualisasi correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Tampilkan korelasi dengan price
    if 'price' in correlation_matrix:
        price_corr = correlation_matrix['price'].sort_values(ascending=False)
        print("\nKorelasi dengan Price (diurutkan):")
        for feature, corr_val in price_corr.items():
            if feature != 'price':
                print(f"{feature:20}: {corr_val:.3f}")
    else:
        print("Kolom 'price' tidak ada dalam matriks korelasi.")
else:
    print("❌ Peringatan: Tidak cukup kolom numerik yang tersedia (termasuk 'price') untuk analisis korelasi.")

# ############################################################################
# CELL 9: PREPARE DATA UNTUK MODELING
# ############################################################################

print("\n🎯 PREPARE DATA UNTUK MODELING")
print("="*50)

if 'price' not in df_features.columns:
    print("❌ Error: Target variable 'price' tidak ditemukan. Tidak dapat melanjutkan ke pemodelan.")
    exit()

target = 'price'

# Numerical features - hanya gunakan yang tersedia
base_numerical = ['spec_rating', 'display_size', 'warranty', 'ram_size', 'rom_size']
numerical_features = [col for col in base_numerical if col in df_features.columns]

# Tambahkan resolution features jika ada
if 'resolution_width' in df_features.columns and 'resolution_width' not in numerical_features:
    numerical_features.append('resolution_width')
if 'resolution_height' in df_features.columns and 'resolution_height' not in numerical_features:
    numerical_features.append('resolution_height')
if 'total_pixels' in df_features.columns and 'total_pixels' not in numerical_features: # Tambahkan total_pixels jika dibuat
    numerical_features.append('total_pixels')


# Categorical features - hanya gunakan yang tersedia
base_categorical = ['brand', 'processor', 'Ram_type', 'ROM_type', 'GPU', 'OS']
categorical_features = [col for col in base_categorical if col in df_features.columns]

print(f"Numerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")

# Cek apakah ada fitur yang dipilih
if not numerical_features and not categorical_features:
    print("❌ Error: Tidak ada fitur numerik atau kategorikal yang tersedia untuk pemodelan.")
    exit()

X = df_features[numerical_features + categorical_features].copy() # Gunakan .copy() untuk menghindari SettingWithCopyWarning
y = df_features[target].copy()

# Handle missing values in features before splitting
# For numerical features, fill with median
for col in numerical_features:
    if X[col].isnull().any():
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        print(f"Missing values in numerical feature '{col}' filled with median ({median_val}).")

# For categorical features, fill with mode or a constant string 'Unknown'
for col in categorical_features:
    if X[col].isnull().any():
        mode_val = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
        X[col].fillna(mode_val, inplace=True)
        print(f"Missing values in categorical feature '{col}' filled with mode ('{mode_val}').")


print(f"\nShape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Split data
# Pastikan 'price_category' ada untuk stratifikasi
if 'price_category' in df_features.columns and df_features['price_category'].notnull().all():
    # Periksa apakah ada cukup sampel di setiap strata
    value_counts = df_features['price_category'].value_counts()
    if all(count >= 2 for count in value_counts): # Minimal 2 sampel per kelas untuk stratifikasi
        stratify_col = df_features['price_category']
        print("Menggunakan stratifikasi berdasarkan 'price_category'.")
    else:
        stratify_col = None
        print("Peringatan: Tidak cukup sampel di beberapa 'price_category' untuk stratifikasi. Stratifikasi dinonaktifkan.")
else:
    stratify_col = None
    print("Peringatan: Kolom 'price_category' tidak tersedia atau mengandung NaN. Stratifikasi dinonaktifkan.")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_col
)

print(f"\nData split:")
print(f"Training set: {X_train.shape[0]} samples, X_test: {X_test.shape[0]} samples")

# ############################################################################
# CELL 10: PREPROCESSING PIPELINE
# ############################################################################

print("\n⚙️ SETUP PREPROCESSING PIPELINE")
print("="*50)

# Buat preprocessor
# Hanya buat transformer jika ada fitur yang sesuai
transformers_list = []
if numerical_features:
    transformers_list.append(('num', StandardScaler(), numerical_features))
if categorical_features:
    transformers_list.append(('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features)) # sparse_output=False untuk kemudahan inspeksi

if not transformers_list:
    print("❌ Error: Tidak ada fitur numerik atau kategorikal untuk preprocessing. Pipeline tidak dapat dibuat.")
    # exit() # Matikan exit agar skrip tetap berjalan jika tidak ada fitur, namun model tidak akan dilatih
    preprocessor = None # Set preprocessor ke None jika tidak ada fitur
else:
    preprocessor = ColumnTransformer(transformers=transformers_list, remainder='passthrough') # remainder='passthrough' jika ada kolom tak terduga
    print("✅ Preprocessing pipeline telah dibuat")


# ############################################################################
# CELL 11: MODEL BUILDING DAN TRAINING
# ############################################################################

print("\n🤖 MODEL BUILDING DAN TRAINING")
print("="*50)

# Definisikan models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(random_state=42),
    'Lasso Regression': Lasso(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100)
}

# Training dan evaluasi
results = {}
trained_models = {}

if preprocessor and (numerical_features or categorical_features): # Hanya lanjut jika ada preprocessor dan fitur
    print("Training models...")
    print("-" * 30)

    for name, model in models.items():
        print(f"Training {name}...")

        # Buat pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        try:
            # Train model
            pipeline.fit(X_train, y_train)

            # Predict
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)

            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)

            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=min(5, X_train.shape[0]), scoring='r2') # Pastikan cv tidak lebih besar dari jumlah sampel

            # Store results
            results[name] = {
                'Train R²': train_r2,
                'Test R²': test_r2,
                'Train RMSE': train_rmse,
                'Test RMSE': test_rmse,
                'Test MAE': test_mae,
                'CV R² Mean': cv_scores.mean(),
                'CV R² Std': cv_scores.std()
            }

            trained_models[name] = pipeline
            print(f"  ✅ {name} - Test R²: {test_r2:.3f}, Test RMSE: {test_rmse:.0f}")

        except Exception as e:
            print(f"  ❌ Error training {name}: {e}")
            results[name] = {key: np.nan for key in ['Train R²', 'Test R²', 'Train RMSE', 'Test RMSE', 'Test MAE', 'CV R² Mean', 'CV R² Std']}
            trained_models[name] = None
else:
    print("❌ Peringatan: Tidak ada preprocessor atau fitur yang valid. Pelatihan model dilewati.")


# ############################################################################
# CELL 12: MODEL EVALUATION DAN COMPARISON
# ############################################################################

print("\n📊 MODEL EVALUATION DAN COMPARISON")
print("="*50)

if results: # Hanya jika ada hasil model
    # Buat DataFrame hasil
    results_df = pd.DataFrame(results).T.dropna(how='all') # Hapus model yang gagal total
    print("PERFORMA SEMUA MODEL:")
    print("="*80)
    print(results_df.round(3))

    if not results_df.empty:
        # Visualisasi perbandingan model
        fig, axes = plt.subplots(2, 2, figsize=(17, 14)) # Ukuran disesuaikan
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

        # Filter model yang berhasil dilatih (Test R² bukan NaN)
        valid_models_names = results_df.index.tolist()
        valid_models_count = len(valid_models_names)

        if valid_models_count > 0:
            # 1. R² Score comparison
            test_r2_scores = results_df['Test R²'].values
            axes[0,0].bar(range(valid_models_count), test_r2_scores, color='lightblue', edgecolor='navy')
            axes[0,0].set_title('Test R² Score Comparison')
            axes[0,0].set_xlabel('Models')
            axes[0,0].set_ylabel('R² Score')
            axes[0,0].set_xticks(range(valid_models_count))
            axes[0,0].set_xticklabels(valid_models_names, rotation=45, ha='right')
            axes[0,0].grid(True, alpha=0.3)

            # 2. RMSE comparison
            test_rmse_scores = results_df['Test RMSE'].values
            axes[0,1].bar(range(valid_models_count), test_rmse_scores, color='lightcoral', edgecolor='darkred')
            axes[0,1].set_title('Test RMSE Comparison')
            axes[0,1].set_xlabel('Models')
            axes[0,1].set_ylabel('RMSE')
            axes[0,1].set_xticks(range(valid_models_count))
            axes[0,1].set_xticklabels(valid_models_names, rotation=45, ha='right')
            axes[0,1].grid(True, alpha=0.3)

            # 3. Cross-validation comparison
            cv_r2_means = results_df['CV R² Mean'].values
            cv_r2_stds = results_df['CV R² Std'].values
            axes[1,0].bar(range(valid_models_count), cv_r2_means, yerr=cv_r2_stds,
                          color='lightgreen', edgecolor='darkgreen', capsize=5)
            axes[1,0].set_title('Cross-Validation R² Score')
            axes[1,0].set_xlabel('Models')
            axes[1,0].set_ylabel('CV R² Score')
            axes[1,0].set_xticks(range(valid_models_count))
            axes[1,0].set_xticklabels(valid_models_names, rotation=45, ha='right')
            axes[1,0].grid(True, alpha=0.3)

            # 4. Train vs Test R²
            train_r2_scores = results_df['Train R²'].values
            x_pos = np.arange(valid_models_count)
            width = 0.35
            axes[1,1].bar(x_pos - width/2, train_r2_scores, width, label='Train R²', color='orange', alpha=0.7)
            axes[1,1].bar(x_pos + width/2, test_r2_scores, width, label='Test R²', color='blue', alpha=0.7)
            axes[1,1].set_title('Train vs Test R² (Overfitting Check)')
            axes[1,1].set_xlabel('Models')
            axes[1,1].set_ylabel('R² Score')
            axes[1,1].set_xticks(x_pos)
            axes[1,1].set_xticklabels(valid_models_names, rotation=45, ha='right')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

            # Identifikasi best model
            if not results_df['Test R²'].isnull().all(): # Pastikan ada nilai Test R² yang tidak NaN
                best_model_name = results_df['Test R²'].idxmax()
                best_model = trained_models.get(best_model_name) # Gunakan .get() untuk keamanan
                if best_model:
                    print(f"\n🏆 BEST MODEL: {best_model_name}")
                    print(f"Test R² Score: {results[best_model_name]['Test R²']:.3f}")
                    print(f"Test RMSE: {results[best_model_name]['Test RMSE']:.0f}")
                    print(f"Test MAE: {results[best_model_name]['Test MAE']:.0f}")
                else:
                    print("❌ Tidak ada model terbaik yang valid ditemukan.")
                    best_model_name = None # Reset best_model_name
            else:
                print("❌ Tidak ada nilai Test R² yang valid untuk menentukan model terbaik.")
                best_model_name = None
                best_model = None
        else:
            print("Tidak ada model yang berhasil dilatih untuk divisualisasikan.")
            best_model_name = None
            best_model = None
    else:
        print("Tidak ada hasil model yang valid untuk dievaluasi.")
        best_model_name = None
        best_model = None # Inisialisasi best_model jika results_df kosong
else:
    print("Tidak ada hasil model untuk dievaluasi.")
    best_model_name = None
    best_model = None # Inisialisasi best_model jika results kosong

# ############################################################################
# CELL 13: FEATURE IMPORTANCE (untuk tree-based models)
# ############################################################################

print("\n🎯 FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Pastikan best_model dan best_model_name ada dan valid
if best_model_name and best_model and \
   ('Random Forest' in best_model_name or 'Decision Tree' in best_model_name or 'Gradient Boosting' in best_model_name):

    try:
        # Get feature names setelah preprocessing
        preprocessor_fitted = best_model.named_steps['preprocessor']
        regressor_step = best_model.named_steps['regressor']

        # Get feature names from ColumnTransformer
        feature_names_out = []
        for name, transformer, columns in preprocessor_fitted.transformers_:
            if transformer == 'drop' or transformer == 'passthrough':
                continue
            if hasattr(transformer, 'get_feature_names_out'):
                if name == 'cat': # Untuk OneHotEncoder
                    feature_names_out.extend(transformer.get_feature_names_out(columns))
                else: # Untuk StandardScaler atau lainnya
                    feature_names_out.extend(columns) # StandardScaler tidak mengubah nama fitur
            else: # Jika tidak ada get_feature_names_out, gunakan nama kolom asli
                feature_names_out.extend(columns)

        # Get feature importance
        if hasattr(regressor_step, 'feature_importances_'):
            importance_scores = regressor_step.feature_importances_

            if len(feature_names_out) == len(importance_scores):
                # Buat DataFrame feature importance
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names_out,
                    'importance': importance_scores
                }).sort_values('importance', ascending=False)

                print("TOP 15 MOST IMPORTANT FEATURES:")
                print("-" * 40)
                for i, (_, row) in enumerate(feature_importance_df.head(15).iterrows(), 1):
                    print(f"{i:2d}. {row['feature']:<35} : {row['importance']:.3f}")

                # Visualisasi feature importance
                plt.figure(figsize=(12, 8))
                top_features = feature_importance_df.head(15)
                plt.barh(range(len(top_features)), top_features['importance'], color='skyblue', edgecolor='navy')
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Importance Score')
                plt.title(f'Top 15 Feature Importance - {best_model_name}', fontweight='bold')
                plt.gca().invert_yaxis()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            else:
                print("❌ Peringatan: Jumlah nama fitur dan skor kepentingan tidak cocok.")
                print(f"  Nama fitur ({len(feature_names_out)}): {feature_names_out}")
                print(f"  Skor kepentingan ({len(importance_scores)}): {importance_scores}")

        else:
            print(f"Model {best_model_name} tidak memiliki atribut 'feature_importances_'.")

    except Exception as e:
        print(f"❌ Error saat mengambil feature importance: {e}")
elif best_model_name:
    print(f"Feature importance tidak ditampilkan untuk model '{best_model_name}' (bukan tree-based atau tidak valid).")
else:
    print("Tidak ada model terbaik yang dipilih untuk analisis feature importance.")


# ############################################################################
# CELL 14: FINAL MODEL EVALUATION
# ############################################################################

print("\n🎉 FINAL MODEL EVALUATION")
print("="*50)

if best_model and best_model_name and results and best_model_name in results:
    # Final predictions
    try:
        y_pred_final = best_model.predict(X_test)
        final_r2 = results[best_model_name]['Test R²']
        final_rmse = results[best_model_name]['Test RMSE']
        final_mae = results[best_model_name]['Test MAE']

        print(f"FINAL MODEL: {best_model_name}")
        print(f"Final Test R²: {final_r2:.3f}")
        print(f"Final Test RMSE: ${final_rmse:.0f}")
        print(f"Final Test MAE: ${final_mae:.0f}")

        # Prediction vs Actual plot
        plt.figure(figsize=(12, 5))

        # 1. Prediction vs Actual scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_pred_final, alpha=0.6, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Garis y=x
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.title(f'Actual vs Predicted Prices ({best_model_name})')
        plt.grid(True, alpha=0.3)
        plt.gca().ticklabel_format(style='plain', axis='both') # Format angka biasa

        # 2. Residuals plot
        plt.subplot(1, 2, 2)
        residuals = y_test - y_pred_final
        plt.scatter(y_pred_final, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price ($)')
        plt.ylabel('Residuals ($)')
        plt.title(f'Residuals Plot ({best_model_name})')
        plt.grid(True, alpha=0.3)
        plt.gca().ticklabel_format(style='plain', axis='both') # Format angka biasa

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"❌ Error saat evaluasi final model: {e}")
else:
    print("Tidak ada model terbaik yang valid untuk evaluasi final.")

# ############################################################################
# CELL 15: PREDICTION EXAMPLES
# ############################################################################

print("\n🔮 CONTOH PREDIKSI")
print("="*50)

if 'best_model' in locals() and best_model is not None and 'y_pred_final' in locals() and len(y_test) > 0:
    # Contoh prediksi
    sample_indices = X_test.index[:min(5, len(y_test))] # Ambil indeks dari X_test untuk konsistensi
    print("CONTOH PREDIKSI HARGA LAPTOP:")
    print("-" * 70)
    print(f"{'No':<3} {'Actual':<10} {'Predicted':<10} {'Error':<10} {'Error %':<10}")
    print("-" * 70)

    for i, idx in enumerate(sample_indices, 1):
        actual = y_test.loc[idx]
        # Cari prediksi yang sesuai dengan idx dari X_test
        # Ini mengasumsikan y_pred_final memiliki urutan yang sama dengan y_test hasil split
        # dan X_test.index[i-1] adalah cara untuk mendapatkan indeks asli jika y_pred_final adalah array numpy
        # Jika y_pred_final dibuat dari X_test yang indexnya terjaga, maka kita bisa pakai X_test.index juga
        # Kita akan gunakan y_pred_final[X_test.index.get_loc(idx)] jika y_pred_final adalah numpy array yang tidak berindeks.
        # Namun, karena y_pred_final = best_model.predict(X_test), urutannya harusnya sama dengan X_test.
        # Jadi kita bisa gunakan X_test.index.get_loc(idx) untuk mendapatkan posisi numerik dari idx di X_test.

        # Untuk lebih aman, prediksi ulang untuk sampel spesifik
        sample_data_for_pred = X_test.loc[[idx]]
        predicted = best_model.predict(sample_data_for_pred)[0]

        error = abs(actual - predicted)
        error_pct = (error / actual) * 100 if actual != 0 else float('inf')

        print(f"{i:<3} ${actual:<9,.0f} ${predicted:<9,.0f} ${error:<9,.0f} {error_pct:<9.1f}%")
else:
    print("Tidak dapat menampilkan contoh prediksi. Model terbaik tidak tersedia atau data uji kosong.")

# ############################################################################
# CELL 16: PREDICTION FUNCTION
# ############################################################################

# Pastikan numerical_features dan categorical_features terdefinisi global atau dilewatkan
# Kita akan mengasumsikan mereka terdefinisi secara global dari CELL 9

def predict_laptop_price(model_pipeline, feature_dict):
    """
    Function untuk memprediksi harga laptop

    Parameters:
    - model_pipeline: trained model pipeline (termasuk preprocessor)
    - feature_dict: dictionary berisi feature values

    Returns:
    - predicted_price: prediksi harga
    """
    if not model_pipeline:
        print("❌ Error: Model tidak tersedia untuk prediksi.")
        return None
    if not (numerical_features or categorical_features): # Cek apakah ada fitur yang didefinisikan
        print("❌ Error: Daftar fitur (numerical/categorical) tidak terdefinisi.")
        return None

    # Buat DataFrame dari input
    input_df = pd.DataFrame([feature_dict])

    # Pastikan semua required columns ada dan isi default jika perlu
    all_model_features = numerical_features + categorical_features

    for col in all_model_features:
        if col not in input_df.columns:
            print(f"Fitur '{col}' tidak ada di input, menggunakan nilai default.")
            # Set default values (ini bisa lebih canggih, misal dari training set means/modes)
            if col in numerical_features:
                if col == 'spec_rating': input_df[col] = X[col].median() if col in X else 3.0
                elif col == 'display_size': input_df[col] = X[col].median() if col in X else 15.6
                elif col == 'warranty': input_df[col] = X[col].median() if col in X else 1
                elif col == 'ram_size': input_df[col] = X[col].median() if col in X else 8
                elif col == 'rom_size': input_df[col] = X[col].median() if col in X else 256
                elif col == 'resolution_width': input_df[col] = X[col].median() if col in X else 1920
                elif col == 'resolution_height': input_df[col] = X[col].median() if col in X else 1080
                elif col == 'total_pixels': input_df[col] = X[col].median() if col in X else 1920*1080
                else: input_df[col] = 0
            elif col in categorical_features:
                input_df[col] = X[col].mode()[0] if col in X and not X[col].mode().empty else 'Unknown'
        # Konversi tipe data jika perlu, terutama untuk numerik
        if col in numerical_features and input_df[col].dtype == 'object':
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except ValueError:
                print(f"Peringatan: Tidak dapat mengkonversi '{col}' ke numerik, menggunakan 0.")
                input_df[col] = 0


    # Reorder columns sesuai dengan urutan saat training
    # Preprocessor ColumnTransformer akan menangani pemilihan kolom yang benar
    # Jadi, pastikan input_df hanya memiliki kolom yang diharapkan oleh preprocessor
    try:
        input_df_ordered = input_df[all_model_features]
    except KeyError as e:
        print(f"❌ Error: Kolom yang dibutuhkan model tidak ditemukan di input setelah default: {e}")
        print(f"Kolom di input_df: {input_df.columns.tolist()}")
        print(f"Kolom yang diharapkan: {all_model_features}")
        return None

    # Predict
    try:
        prediction = model_pipeline.predict(input_df_ordered)[0]
        return prediction
    except Exception as e:
        print(f"❌ Error saat prediksi dengan model: {e}")
        print(f"Data yang coba diprediksi:\n{input_df_ordered.head()}")
        # Coba untuk melihat transformasi data jika ada masalah
        try:
            transformed_data = model_pipeline.named_steps['preprocessor'].transform(input_df_ordered)
            print(f"Data setelah transformasi oleh preprocessor:\n{transformed_data}")
        except Exception as te:
            print(f"❌ Error saat mencoba transform data input: {te}")
        return None


print("\n🔧 FUNCTION UNTUK PREDIKSI BARU TELAH DIBUAT!")
print("="*50)

# Example prediction
# Pastikan 'best_model' sudah terdefinisi dan valid
if 'best_model' in locals() and best_model is not None:
    example_laptop = {
        'spec_rating': 4.5,
        'display_size': 15.6,
        'warranty': 2,
        'ram_size': 16,
        'rom_size': 512,
        'resolution_width': 1920, # Tambahkan fitur yang mungkin digunakan model
        'resolution_height': 1080,
        'total_pixels': 1920*1080
    }

    # Tambahkan categorical features jika ada dan digunakan oleh model
    for cat_col in categorical_features: # Gunakan categorical_features yang didefinisikan di CELL 9
        if cat_col == 'brand': example_laptop[cat_col] = 'ASUS'
        elif cat_col == 'processor': example_laptop[cat_col] = 'Intel Core i7'
        elif cat_col == 'OS': example_laptop[cat_col] = 'Windows'
        elif cat_col == 'Ram_type': example_laptop[cat_col] = 'DDR4' # Contoh
        elif cat_col == 'ROM_type': example_laptop[cat_col] = 'SSD' # Contoh
        elif cat_col == 'GPU': example_laptop[cat_col] = 'NVIDIA GeForce RTX 3060' # Contoh
        else: example_laptop[cat_col] = 'Standard' # Default untuk kolom kategori lain

    try:
        predicted_price_example = predict_laptop_price(best_model, example_laptop)
        if predicted_price_example is not None:
            print(f"Contoh prediksi harga: ${predicted_price_example:,.0f}")
        else:
            print("Prediksi contoh gagal.")
    except Exception as e:
        print(f"Error dalam contoh prediksi: {e}")
else:
    print("Model terbaik tidak tersedia untuk contoh prediksi.")


# ############################################################################
# CELL 17: MODEL SUMMARY DAN BUSINESS INSIGHTS
# ############################################################################

print("\n📋 MODEL SUMMARY DAN BUSINESS INSIGHTS")
print("="*50)

if 'df' in locals() and 'df_features' in locals() and \
   'best_model_name' in locals() and best_model_name and \
   'final_r2' in locals() and 'final_rmse' in locals() and 'final_mae' in locals() and \
   (numerical_features or categorical_features): # Pastikan variabel ada

    print(f"Dataset: {df.shape[0]} laptop records awal, {df_features.shape[0]} setelah cleaning/feature eng.")
    print(f"Features used for modeling: {len(numerical_features + categorical_features)}")
    print(f"Best model: {best_model_name}")
    print(f"Model performance:")
    print(f"  - R² Score: {final_r2:.3f} (menjelaskan {final_r2*100:.1f}% variasi harga)")
    print(f"  - RMSE: ${final_rmse:,.0f} (rata-rata error prediksi)")
    print(f"  - MAE: ${final_mae:,.0f} (rata-rata absolute error)")

    # Price category distribution
    if 'price_category' in df_features.columns:
        print(f"\nDISTRIBUSI KATEGORI HARGA (setelah feature engineering):")
        price_dist = df_features['price_category'].value_counts(normalize=True) * 100
        price_counts = df_features['price_category'].value_counts()
        for category, pct in price_dist.items():
            print(f"  {category}: {price_counts[category]} laptop ({pct:.1f}%)")
    else:
        print("\nDistribusi kategori harga tidak tersedia (kolom 'price_category' tidak ada).")


    # Brand analysis
    if 'brand' in df_features.columns and 'price' in df_features.columns:
        print(f"\nTOP 5 BRAND BERDASARKAN RATA-RATA HARGA:")
        try:
            brand_avg = df_features.groupby('brand')['price'].mean().sort_values(ascending=False).head()
            for brand, price_val in brand_avg.items():
                print(f"  {brand}: ${price_val:,.0f}")
        except Exception as e:
            print(f"  Tidak dapat melakukan analisis brand: {e}")

    else:
        print("\nAnalisis brand tidak tersedia (kolom 'brand' atau 'price' tidak ada).")


    print(f"\n💡 INTERPRETASI HASIL:")
    print("="*50)
    if final_r2 >= 0.8:
        print("✅ Model SANGAT BAIK - Prediksi sangat akurat dan dapat diandalkan.")
    elif final_r2 >= 0.6:
        print("✅ Model BAIK - Prediksi cukup akurat, berguna untuk estimasi umum.")
    elif final_r2 >= 0.4:
        print("⚠️ Model CUKUP - Ada ruang signifikan untuk improvement, gunakan dengan hati-hati.")
    else:
        print("❌ Model KURANG BAIK - Perlu perbaikan besar sebelum digunakan secara praktis.")

    print(f"\n🎯 Model ini ({best_model_name}) dapat digunakan untuk:")
    print("1. Estimasi harga laptop berdasarkan spesifikasi yang diberikan.")
    print("2. Membantu dalam strategi penetapan harga dan analisis pasar (jika akurasi memadai).")
    print("3. Evaluasi apakah harga suatu laptop sepadan dengan spesifikasinya (value-for-money).")
    print("4. Benchmarking harga kompetitif (dengan asumsi data mencakup pasar yang relevan).")

else:
    print("Beberapa variabel penting tidak terdefinisi. Ringkasan model dan insight tidak dapat ditampilkan sepenuhnya.")


print("\n✅ ANALISIS SELESAI!")
print("="*50)
print("Semua tahap analisis telah dijalankan:")
print(f"✅ Data exploration & cleaning {'berhasil' if 'df_clean' in locals() else 'tidak lengkap'}")
print(f"✅ Feature engineering {'berhasil' if 'df_features' in locals() else 'tidak lengkap'}")
print(f"✅ Model training & evaluation {'berhasil' if results else 'tidak lengkap/gagal'}")
print(f"✅ Best model selection {'berhasil' if 'best_model' in locals() and best_model else 'tidak ada'}")
print(f"✅ Prediction function {'ready' if callable(predict_laptop_price) else 'tidak siap'}")

if 'best_model' in locals() and best_model:
    print("🎉 Model siap untuk digunakan lebih lanjut atau deployment")
else:
    print("⚠️ Model belum siap sepenuhnya, periksa output sebelumnya untuk detail.")