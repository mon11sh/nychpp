import pandas as pd
import os 
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

print("🚀 NYC HOUSE PRICE MODELING PIPELINE")
print("="*50)

# Load processed data
processed_path = r'C:\code_1\house_pred\data\processed\nyc_housing_processed.csv'
df = pd.read_csv(processed_path)
print(f"Loaded processed data: {df.shape}")

# Prepare features and target
X = df.drop(['log_PRICE', 'PRICE'], axis=1)
y = df['log_PRICE']

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain set: {X_train.shape} | Test set: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n📊 MODEL TRAINING & COMPARISON")
print("-" * 40)

# Define models
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=10),
    'Lasso': Lasso(alpha=0.1),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate all models
results = {}
cv_scores = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use scaled data for linear models, original for tree-based
    if name in ['LinearRegression', 'Ridge', 'Lasso']:
        model.fit(X_train_scaled, y_train)
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    results[name] = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2
    }
    
    # Cross-validation
    if name in ['LinearRegression', 'Ridge', 'Lasso']:
        cv_scores[name] = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2').mean()
    else:
        cv_scores[name] = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()
    
    print(f"  Test RMSE: {test_rmse:.4f} | R²: {test_r2:.4f} | MAE: {test_mae:.4f}")

# Convert to DataFrame for comparison
results_df = pd.DataFrame(results).T
print("\n" + "="*60)
print("📈 MODEL COMPARISON TABLE")
print("="*60)
print(results_df.round(4))

# Best model
best_model_name = results_df['test_r2'].idxmax()
print(f"\n🏆 BEST MODEL: {best_model_name} (R² = {results_df.loc[best_model_name, 'test_r2']:.4f})")

# Hyperparameter tuning for best tree model (XGBoost/RandomForest)
print(f"\n🔧 HYPERPARAMETER TUNING for {best_model_name}...")
if best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    best_model = models[best_model_name]
    grid_search = GridSearchCV(best_model, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Final model with best params
    final_model = xgb.XGBRegressor(**grid_search.best_params_, random_state=42)
    final_model.fit(X_train, y_train)
    
elif best_model_name == 'RandomForest':
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    best_model = models[best_model_name]
    grid_search = GridSearchCV(best_model, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    final_model = RandomForestRegressor(**grid_search.best_params_, random_state=42)
    final_model.fit(X_train, y_train)
else:
    final_model = models[best_model_name]
    if best_model_name in ['LinearRegression', 'Ridge', 'Lasso']:
        final_model.fit(X_train_scaled, y_train)

# Final predictions
if best_model_name in ['LinearRegression', 'Ridge', 'Lasso']:
    y_pred_final = final_model.predict(X_test_scaled)
else:
    y_pred_final = final_model.predict(X_test)

final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
final_r2 = r2_score(y_test, y_pred_final)
print(f"🎯 FINAL MODEL PERFORMANCE:")
print(f"   RMSE: {final_rmse:.4f} | R²: {final_r2:.4f}")

# Feature importance (for tree models)
if best_model_name in ['RandomForest', 'XGBoost']:
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title(f'Top 10 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.savefig(r'C:\code_1\house_pred\data\processed\feature_importance.png', dpi=300)
    plt.show()
    
    print("\n🔍 TOP 5 MOST IMPORTANT FEATURES:")
    print(feature_importance.head().to_string(index=False))

# Predictions vs Actual plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_final, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Log Price')
plt.ylabel('Predicted Log Price')
plt.title('Predictions vs Actual (Log Scale)')

plt.subplot(1, 2, 2)
residuals = y_test - y_pred_final
plt.scatter(y_pred_final, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Log Price')
plt.ylabel('Residuals')
plt.title('Residuals Plot')

plt.tight_layout()
plt.savefig(r'C:\code_1\house_pred\data\processed\model_predictions.png', dpi=300)
plt.show()

# SAVE FINAL MODEL & ARTIFACTS
os.makedirs(r'C:\code_1\house_pred\models', exist_ok=True)

# Save final model, scaler, and metadata
model_artifacts = {
    'model': final_model,
    'scaler': scaler if best_model_name in ['LinearRegression', 'Ridge', 'Lasso'] else None,
    'feature_names': X.columns.tolist(),
    'best_model_name': best_model_name,
    'test_r2': final_r2,
    'test_rmse': final_rmse
}

with open(r'C:\code_1\house_pred\models\final_model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

# Save results
results_df.to_csv(r'C:\code_1\house_pred\models\model_comparison.csv')
feature_importance.to_csv(r'C:\code_1\house_pred\models\feature_importance.csv', index=False)

print("\n" + "="*60)
print("✅ MODELING COMPLETE! PRODUCTION READY!")
print("="*60)
print(f"🏆 Best Model: {best_model_name}")
print(f"📊 Test R²: {final_r2:.4f}")
print(f"📈 Test RMSE: {final_rmse:.4f}")
print(f"💾 Saved files:")
print(f"   - models/final_model.pkl (DEPLOYMENT READY)")
print(f"   - models/model_comparison.csv")
print(f"   - models/feature_importance.csv")
print(f"   - data/processed/*.png (visualizations)")
print("\n🎉 Next: Deployment or predictions on new data!")
