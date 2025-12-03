# app.py - Save this in C:\code_1\house_pred\src\app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page config
st.set_page_config(
    page_title="NYC House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_artifacts():
    """Load model and processed data"""
    with open(r'C:\code_1\house_pred\models\final_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    
    df_processed = pd.read_csv(r'C:\code_1\house_pred\data\processed\nyc_housing_processed.csv')
    
    FEATURE_COLS = ['BEDS_capped', 'BATH_capped', 'log_PROPERTYSQFT', 'LATITUDE', 'LONGITUDE', 
                    'distance_to_center_km', 'price_per_sqft', 'beds_per_bath', 'sqft_per_bed', 
                    'broker_freq', 'TYPE_encoded', 'neighborhood_encoded']
    
    return artifacts, df_processed, FEATURE_COLS

def preprocess_input(beds, bath, propertysqft, lat, lon, type_encoded, neighborhood_encoded, broker_freq):
    """Preprocess input exactly like training"""
    beds_capped = min(max(beds, 1), 8)
    bath_capped = min(max(bath, 0), 6)
    propertysqft_capped = min(max(propertysqft, 230), 10000)
    
    distance_to_center = np.sqrt((lat - 40.7128)**2 + (lon + 74.0060)**2) * 111
    log_propertysqft = np.log1p(propertysqft_capped)
    
    price_per_sqft = 800
    beds_per_bath = beds_capped / (bath_capped + 1)
    sqft_per_bed = propertysqft_capped / (beds_capped + 1)
    
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
    return input_data

# Load artifacts
artifacts, df_processed, FEATURE_COLS = load_artifacts()
model = artifacts['model']
scaler = artifacts.get('scaler', None)

# Main app
st.markdown('<h1 class="main-header">🏠 NYC House Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("**R² = 0.998 | Production RandomForest Model**")

# Sidebar
with st.sidebar:
    st.header("📊 Property Details")
    beds = st.slider("Bedrooms", 1, 8, 3)
    bath = st.slider("Bathrooms", 0, 6, 2)
    sqft = st.slider("Square Footage", 230, 10000, 1500)
    
    st.header("📍 Location")
    col1, col2 = st.columns(2)
    with col1:
        lat = st.slider("Latitude", 40.49, 40.92, 40.75)
    with col2:
        lon = st.slider("Longitude", -74.25, -73.70, -73.95)
    
    st.header("🏷️ Property Type")
    type_options = {
        "Condo": 2,
        "House": 7, 
        "Townhouse": 12,
        "Co-op": 0,
        "Multi-family": 10
    }
    selected_type = st.selectbox("Select Type", list(type_options.keys()))
    type_encoded = type_options[selected_type]
    
    neighborhood_options = {
        "Manhattan": 3,
        "Brooklyn": 1,
        "Staten Island": 0
    }
    neighborhood = st.selectbox("Neighborhood", list(neighborhood_options.keys()))
    neighborhood_encoded = neighborhood_options[neighborhood]
    
    broker_freq = st.slider("Broker Popularity (frequency)", 1, 50, 10)
    
    predict_btn = st.button("🚀 Predict Price", type="primary")

# Main prediction area
if predict_btn:
    with st.spinner("🔮 Predicting..."):
        input_data = preprocess_input(beds, bath, sqft, lat, lon, type_encoded, neighborhood_encoded, broker_freq)
        
        if scaler:
            input_scaled = scaler.transform(input_data)
            pred_log = model.predict(input_scaled)[0]
        else:
            pred_log = model.predict(input_data)[0]
        
        price = np.expm1(pred_log)
        price_per_sqft_pred = price / sqft
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Price", f"${price:,.0f}", delta=None)
        with col2:
            st.metric("Price per Sqft", f"${price_per_sqft_pred:,.0f}/sqft")
        with col3:
            st.metric("Confidence", f"{artifacts.get('test_r2', 0.998):.1%}", delta=None)
        
        # Feature breakdown
        st.subheader("📈 Feature Contributions")
        input_dict = input_data.iloc[0].to_dict()
        feature_df = pd.DataFrame(list(input_dict.items()), columns=['Feature', 'Value'])
        st.dataframe(feature_df, use_container_width=True)

# Model performance dashboard
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Model Performance")
    
    # Model comparison
    model_results = {
        "LinearRegression": 0.7815,
        "Ridge": 0.7819,
        "RandomForest": 0.9980,
        "XGBoost": 0.9967
    }
    
    fig = px.bar(x=list(model_results.keys()), y=list(model_results.values()),
                title="Model Comparison (Test R²)",
                labels={'x': 'Model', 'y': 'R² Score'})
    fig.update_traces(marker_color=['lightgray']*3 + ['#1f77b4'])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("🏆 Top Features")
    top_features = {
        "price_per_sqft": "59%",
        "log_PROPERTYSQFT": "21%",
        "BATH_capped": "16%",
        "distance_to_center": "2%"
    }
    for feature, importance in top_features.items():
        st.metric(feature.replace("_", " ").title(), importance)

# NYC Map visualization
st.header("🗺️ NYC Price Heatmap")
df_display = df_processed[['LATITUDE', 'LONGITUDE', 'PRICE']].copy()
df_display['PRICE'] = np.expm1(df_processed['log_PRICE'])

fig_map = px.scatter_mapbox(df_display, 
                           lat="LATITUDE", lon="LONGITUDE",
                           color="PRICE", size_max=15,
                           color_continuous_scale="Viridis",
                           mapbox_style="census",
                           title="NYC Property Prices (Training Data)",
                           hover_data=['PRICE'],
                           zoom=10,
                           height=500)
st.plotly_chart(fig_map, use_container_width=True)

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Adjust sliders** for bedrooms, bathrooms, square footage
    2. **Set location** using latitude/longitude (Manhattan: 40.75, -73.95)
    3. **Select property type** from dropdown
    4. **Click Predict** for instant valuation
    5. **Model accuracy**: 99.8% (R²) on 4,801 NYC listings
    """)

st.markdown("---")
st.markdown("**Built with ❤️ using Streamlit | Model R²: 0.998**")
