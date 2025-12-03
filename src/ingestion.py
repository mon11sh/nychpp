import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Path to dataset
data_path = r'C:\code_1\house_pred\data\raw\nycd.csv'

# Step 1: Load the dataset
print("Loading NYC Housing dataset...")
df = pd.read_csv(data_path)

print(f"Dataset loaded successfully! Shape: {df.shape}")
print("\n" + "="*60)
print("INITIAL DATA INSPECTION")
print("="*60)

# Step 2: Basic info
print("\n1. Dataset Info:")
print(df.info())
print("\n2. First 5 rows:")
print(df.head())
print("\n3. Dataset shape:", df.shape)
print("\n4. Column names:")
print(list(df.columns))

# Step 3: Missing values check
print("\n5. Missing values per column:")
missing_data = df.isnull().sum()
missing_pct = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing_data,
    'Missing_Percentage': missing_pct
}).sort_values('Missing_Percentage', ascending=False)
print(missing_df)

# Step 4: Basic statistics
print("\n6. Basic statistics for numeric columns:")
print(df.describe())

# Step 5: PRICE column specific inspection (critical for target variable)
print("\n7. PRICE column inspection:")
print("PRICE unique values sample:", df['PRICE'].dropna().unique()[:10])
print("PRICE data type:", df['PRICE'].dtype)
print("PRICE null count:", df['PRICE'].isnull().sum())
print("PRICE sample values:")
print(df['PRICE'].head(10).to_string())

# Step 6: Key feature inspections
key_features = ['BEDS', 'BATH', 'PROPERTYSQFT', 'LATITUDE', 'LONGITUDE', 'TYPE']
print("\n8. Key features quick check:")
for feature in key_features:
    if feature in df.columns:
        print(f"\n{feature}:")
        print(f"  - Nulls: {df[feature].isnull().sum()}")
        print(f"  - Data type: {df[feature].dtype}")
        print(f"  - Unique values: {df[feature].nunique()}")
        if df[feature].dtype in ['int64', 'float64']:
            print(f"  - Min/Max: {df[feature].min()}/{df[feature].max()}")

# Step 7: Save initial inspection results
inspection_path = r'C:\code_1\house_pred\data\raw\inspection_report.csv'
missing_df.to_csv(inspection_path)
print(f"\n✅ Inspection report saved to: {inspection_path}")

# Step 8: Quick visualization of target distribution
if 'PRICE' in df.columns:
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['PRICE'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    plt.title('PRICE Distribution')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.boxplot(df['PRICE'].dropna())
    plt.title('PRICE Boxplot (Outliers)')
    plt.ylabel('Price')
    
    plt.tight_layout()
    plt.savefig(r'C:\code_1\house_pred\data\raw\price_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

print("\n" + "="*60)
print("INGESTION COMPLETE! Ready for preprocessing.")
print("Key findings:")
print("- Dataset shape:", df.shape)
print("- Total missing values:", df.isnull().sum().sum())
print("- PRICE issues to check:", "Non-numeric?" if df['PRICE'].dtype == 'object' else "Looks numeric")
print("="*60)

# Keep original dataframe for reference
df_raw = df.copy()
print("\n✅ Raw data saved as 'df_raw' for reference")
