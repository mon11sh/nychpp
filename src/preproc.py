import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
import os
warnings.filterwarnings('ignore')

# Load raw data first
data_path = r'C:\code_1\house_pred\data\raw\nycd.csv'
print("Loading raw NYC Housing dataset for preprocessing...")
df_raw = pd.read_csv(data_path)
print(f"Original shape: {df_raw.shape}")

# Create working copy
df = df_raw.copy()

# =============================================================================
# STEP 1: OUTLIER DETECTION & HANDLING
# =============================================================================
print("\n1. OUTLIER ANALYSIS & HANDLING")

# IQR method for outlier detection
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers.shape[0], lower_bound, upper_bound

# Check outliers
features_to_check = ['PRICE', 'BEDS', 'BATH', 'PROPERTYSQFT']
outlier_summary = {}
for feature in features_to_check:
    count, lb, ub = detect_outliers_iqr(df, feature)
    outlier_summary[feature] = {'outliers': count, 'lower': lb, 'upper': ub}

print("Outlier Summary:")
for feature, info in outlier_summary.items():
    pct = (info['outliers']/len(df))*100
    print(f"{feature}: {info['outliers']} outliers ({pct:.1f}%)")

# Cap extreme outliers (top 1% and bottom 1%)
price_p1 = df['PRICE'].quantile(0.01)
price_p99 = df['PRICE'].quantile(0.99)
df['PRICE_capped'] = df['PRICE'].clip(lower=price_p1, upper=price_p99)

df['BEDS_capped'] = np.clip(df['BEDS'], 1, 8)  # Reasonable max for NYC
df['BATH_capped'] = np.clip(df['BATH'], 0, 6)
df['PROPERTYSQFT_capped'] = df['PROPERTYSQFT'].clip(lower=df['PROPERTYSQFT'].quantile(0.01),
                                                   upper=df['PROPERTYSQFT'].quantile(0.99))

# =============================================================================
# STEP 2: LOG TRANSFORMATION (SKEWED DISTRIBUTIONS)
# =============================================================================
print("\n2. LOG TRANSFORMATION")

df['log_PRICE'] = np.log1p(df['PRICE_capped'])  # log(1+x) for safety
df['log_PROPERTYSQFT'] = np.log1p(df['PROPERTYSQFT_capped'])

# =============================================================================
# STEP 3: FEATURE ENGINEERING
# =============================================================================
print("\n3. FEATURE ENGINEERING")

# Location-based features
nyc_center_lat, nyc_center_lon = 40.7128, -74.0060
df['distance_to_center_km'] = np.sqrt(
    (df['LATITUDE'] - nyc_center_lat)**2 + 
    (df['LONGITUDE'] - nyc_center_lon)**2
) * 111  # Approx km conversion

# Ratios
df['price_per_sqft'] = df['PRICE_capped'] / df['PROPERTYSQFT_capped']
df['beds_per_bath'] = df['BEDS_capped'] / (df['BATH_capped'] + 1)  # +1 to avoid div0
df['sqft_per_bed'] = df['PROPERTYSQFT_capped'] / (df['BEDS_capped'] + 1)

# FIXED: Neighborhood clustering (5 bins need 6 edges, 5 labels)
df['neighborhood'] = pd.cut(df['LATITUDE'], 
                           bins=[40.49, 40.65, 40.72, 40.78, 40.85, 40.92],
                           labels=['Staten_Island', 'Brooklyn_S', 'Brooklyn_N', 'Manhattan_S', 'Manhattan_N'])

# =============================================================================
# STEP 4: CATEGORICAL ENCODING
# =============================================================================
print("\n4. CATEGORICAL ENCODING")

# TYPE encoding (most important categorical)
type_encoder = LabelEncoder()
df['TYPE_encoded'] = type_encoder.fit_transform(df['TYPE'])
print("TYPE categories:", dict(zip(type_encoder.classes_, type_encoder.transform(type_encoder.classes_))))

# BROKERTITLE - frequency encode
broker_freq = df['BROKERTITLE'].value_counts()
df['broker_freq'] = df['BROKERTITLE'].map(broker_freq)
df.drop('BROKERTITLE', axis=1, inplace=True)

# =============================================================================
# STEP 5: SELECT FINAL FEATURES
# =============================================================================
print("\n5. FEATURE SELECTION")

# Core numeric features
numeric_features = [
    'log_PRICE', 'BEDS_capped', 'BATH_capped', 'log_PROPERTYSQFT',
    'LATITUDE', 'LONGITUDE', 'distance_to_center_km',
    'price_per_sqft', 'beds_per_bath', 'sqft_per_bed', 'broker_freq'
]

# Categorical
categorical_features = ['TYPE_encoded', 'neighborhood']

# Combine
feature_cols = numeric_features + categorical_features

# Create final clean dataset
df_processed = df[feature_cols + ['PRICE']].copy()  # Keep original PRICE for reference

# Neighborhood as numeric for modeling
df_processed['neighborhood_encoded'] = LabelEncoder().fit_transform(df_processed['neighborhood'].astype(str))
feature_cols[-1] = 'neighborhood_encoded'

print(f"Final feature set: {feature_cols}")
print(f"Processed shape: {df_processed.shape}")

# =============================================================================
# STEP 6: VISUALIZATION & VALIDATION
# =============================================================================
print("\n6. VISUALIZATION")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Preprocessing Results', fontsize=16)

# 1. Price distribution before/after
axes[0,0].hist(df_raw['PRICE'], bins=50, alpha=0.5, label='Original', color='red')
axes[0,0].hist(df_processed['log_PRICE'], bins=50, alpha=0.7, label='Log Transformed', color='green')
axes[0,0].set_title('PRICE Distribution')
axes[0,0].legend()

# 2. Feature correlations
corr_matrix = df_processed[feature_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0,1], fmt='.2f')
axes[0,1].set_title('Feature Correlation Matrix')

# 3. Beds distribution
axes[0,2].hist(df_raw['BEDS'], alpha=0.5, label='Original', bins=20)
axes[0,2].hist(df['BEDS_capped'], alpha=0.7, label='Capped', bins=20)
axes[0,2].set_title('BEDS Distribution')
axes[0,2].legend()

# 4. Price vs Sqft
axes[1,0].scatter(df['log_PROPERTYSQFT'], df['log_PRICE'], alpha=0.6)
axes[1,0].set_xlabel('Log PROPERTYSQFT')
axes[1,0].set_ylabel('Log PRICE')
axes[1,0].set_title('Price vs Sqft')

# 5. Distance to center
axes[1,1].scatter(df['distance_to_center_km'], df['log_PRICE'], alpha=0.6)
axes[1,1].set_xlabel('Distance to NYC Center (km)')
axes[1,1].set_ylabel('Log PRICE')
axes[1,1].set_title('Price vs Location')

# 6. TYPE distribution
type_counts = df['TYPE'].value_counts().head(6)  # Top 6 for pie chart
axes[1,2].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
axes[1,2].set_title('Property TYPE Distribution (Top 6)')

plt.tight_layout()
plt.savefig(r'C:\code_1\house_pred\data\processed\preprocessing_results.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# STEP 7: SAVE PROCESSED DATA
# =============================================================================
print("\n7. SAVING RESULTS")

# Create directories
os.makedirs(r'C:\code_1\house_pred\data\processed', exist_ok=True)
os.makedirs(r'C:\code_1\house_pred\data\interim', exist_ok=True)

# Save processed data
df_processed.drop('neighborhood', axis=1, errors='ignore', inplace=True)  # Clean up
df_processed.to_csv(r'C:\code_1\house_pred\data\processed\nyc_housing_processed.csv', index=False)
df.to_csv(r'C:\code_1\house_pred\data\interim\nyc_housing_full_features.csv', index=False)

# Save encoders and metadata
import pickle
with open(r'C:\code_1\house_pred\data\processed\encoders.pkl', 'wb') as f:
    pickle.dump({'type_encoder': type_encoder}, f)

print("\n✅ PREPROCESSING COMPLETE!")
print(f"📊 Final dataset shape: {df_processed.shape}")
print(f"🎯 Target: 'log_PRICE' (use this for modeling)")
print(f"📈 Features ready: {len(feature_cols)} features")
print(f"💾 Files saved:")
print(f"   - processed/nyc_housing_processed.csv (FINAL DATASET)")
print(f"   - interim/nyc_housing_full_features.csv (all features)")
print(f"   - processed/preprocessing_results.png (visualizations)")
print(f"   - processed/encoders.pkl (for inference)")

print("\n🎉 Data is now READY FOR MODELING!")
print("Next step: Train-test split and model training")
