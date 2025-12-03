import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

print("🔮 NYC HOUSE PRICE PREDICTION SCRIPT")
print("="*50)

# FIXED: Complete standalone prediction script
MODEL_PATH = r'C:\code_1\house_pred\models\final_model.pkl'
PROCESSED_PATH = r'C:\code_1\house_pred\data\processed\nyc_housing_processed.csv'

# Load processed data to get correct feature order
df_processed = pd.read_csv(PROCESSED_PATH)
FEATURE_COLS = ['BEDS_capped', 'BATH_capped', 'log_PROPERTYSQFT', 'LATITUDE', 'LONGITUDE', 
                'distance_to_center_km', 'price_per_sqft', 'beds_per_bath', 'sqft_per_bed', 
                'broker_freq', 'TYPE_encoded', 'neighborhood_encoded']

print(f"✅ Features loaded: {len(FEATURE_COLS)} features")
print("📋 TYPE encoding: 0=Co-op, 2=Condo, 7=House, 12=Townhouse, etc.")

# Load model artifacts
with open(MODEL_PATH, 'rb') as f:
    artifacts = pickle.load(f)

model = artifacts['model']
scaler = artifacts.get('scaler', None)
best_model_name = artifacts.get('best_model_name', 'RandomForest')

print(f"✅ Model loaded: {best_model_name}")

def preprocess_input(beds, bath, propertysqft, lat, lon, type_encoded=2, neighborhood_encoded=0, broker_freq=10):
    """Preprocess new house data same way as training"""
    
    # Apply same capping/transformation as training
    beds_capped = min(max(beds, 1), 8)
    bath_capped = min(max(bath, 0), 6)
    propertysqft_capped = min(max(propertysqft, 230), 10000)  # Reasonable cap
    
    # Engineered features
    distance_to_center = np.sqrt((lat - 40.7128)**2 + (lon + 74.0060)**2) * 111
    log_propertysqft = np.log1p(propertysqft_capped)
    
    # Ratios (using reasonable defaults for price_per_sqft)
    price_per_sqft = 800  # NYC avg, will be predicted
    beds_per_bath = beds_capped / (bath_capped + 1)
    sqft_per_bed = propertysqft_capped / (beds_capped + 1)
    
    # Create input DataFrame with EXACT training feature order
    input_data = pd.DataFrame({
        'BEDS_capped': [beds_capped],
        'BATH_capped': [bath_capped],
        'log_PROPERTYSQFT': [log_propertysqft],
        'LATITUDE': [lat],
        'LONGITUDE': [lon],
        'distance_to_center_km': [distance_to_center],
        'price_per_sqft': [price_per_sqft],
        'beds_per_bath': [beds_per_bath],
        'sqft_per_bed': [sqft_per_bed],
        'broker_freq': [broker_freq],
        'TYPE_encoded': [type_encoded],
        'neighborhood_encoded': [neighborhood_encoded]
    })
    
    return input_data[FEATURE_COLS]

def predict_price(beds, bath, propertysqft, lat, lon, type_encoded=2, broker_freq=10, neighborhood_encoded=0):
    """Complete prediction pipeline"""
    input_data = preprocess_input(beds, bath, propertysqft, lat, lon, type_encoded, neighborhood_encoded, broker_freq)
    
    # Scale if needed (linear models)
    if scaler is not None:
        input_scaled = scaler.transform(input_data)
        pred_log = model.predict(input_scaled)[0]
    else:
        pred_log = model.predict(input_data)[0]
    
    # Convert back to dollars
    price = np.expm1(pred_log)
    return {
        'predicted_log_price': pred_log,
        'predicted_price': price,
        'formatted_price': f"${price:,.0f}",
        'input_features': input_data.iloc[0].to_dict()
    }

# 🧪 TEST PREDICTIONS - FIXED
print("\n🧪 SAMPLE PREDICTIONS")
test_cases = [
    {"name": "Manhattan Condo 3BR", "beds": 3, "bath": 2, "sqft": 1500, "lat": 40.75, "lon": -73.95, "type": 2},
    {"name": "Brooklyn House 2BR", "beds": 2, "bath": 1, "sqft": 1200, "lat": 40.68, "lon": -73.98, "type": 7},
    {"name": "Luxury Townhouse 5BR", "beds": 5, "bath": 5, "sqft": 4000, "lat": 40.78, "lon": -73.96, "type": 12},
    {"name": "Studio Condo", "beds": 1, "bath": 1, "sqft": 600, "lat": 40.73, "lon": -74.00, "type": 2}
]

for case in test_cases:
    # Extract only the parameters predict_price accepts
    result = predict_price(
        beds=case['beds'], 
        bath=case['bath'], 
        propertysqft=case['sqft'], 
        lat=case['lat'], 
        lon=case['lon'], 
        type_encoded=case['type']
    )
    print(f"{case['name']}: {result['formatted_price']}")

# Interactive test
print("\n🎯 INTERACTIVE TEST:")
beds = float(input("Enter BEDS (1-8): "))
bath = float(input("Enter BATH (0-6): "))
sqft = float(input("Enter SQFT (230+): "))
lat = float(input("Enter LATITUDE (40.49-40.92): "))
lon = float(input("Enter LONGITUDE (-74.25 to -73.70): "))
type_code = int(input("Enter TYPE (2=Condo, 7=House, 12=Townhouse): "))

result = predict_price(beds, bath, sqft, lat, lon, type_code)
print(f"\n🏠 PREDICTED PRICE: {result['formatted_price']}")

# 💾 SAVE WORKING PREDICTION FUNCTION
os.makedirs(r'C:\code_1\house_pred\models', exist_ok=True)
with open(r'C:\code_1\house_pred\models\predict_function.pkl', 'wb') as f:
    pickle.dump({
        'predict_price': predict_price,
        'preprocess_input': preprocess_input,
        'feature_cols': FEATURE_COLS,
        'type_encoder_info': "0=Co-op, 2=Condo, 7=House, 12=Townhouse"
    }, f)

print(f"\n✅ PREDICTION SCRIPT READY!")
print("📱 Next: Say 'streamlit' for web app!")
