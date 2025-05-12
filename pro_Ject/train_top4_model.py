import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import joblib
import os

print("\n--- TRAINING TOP-4 FEATURES MODEL ---\n")

# Check if model already exists and remove it
if os.path.exists('svr_model_top4.pkl'):
    os.remove('svr_model_top4.pkl')
    print("Removed existing model file")

# Load the dataset
try:
    df = pd.read_csv("C:/Users/ahmed/Downloads/Train (6).csv")
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Handle missing values
df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])
print("Missing values handled")

# Select only the top 4 features and target
selected_features = ['Item_MRP', 'Outlet_Type', 'Outlet_Establishment_Year', 'Outlet_Location_Type']
X = df[selected_features]
y = df['Item_Outlet_Sales']
print(f"Selected features: {selected_features}")
print(f"Feature matrix shape: {X.shape}")  # Should be (8523, 4)

# Preprocess features
# Encode categorical features
categorical_features = ['Outlet_Type', 'Outlet_Location_Type'] 
encoder = OrdinalEncoder()
X[categorical_features] = encoder.fit_transform(X[categorical_features])
print("Categorical features encoded")

# Normalize features
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("Features scaled")
print(f"Preprocessed feature matrix shape: {X_scaled.shape}")  # Should still be (8523, 4)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
print(f"Training feature matrix shape: {X_train.shape}")  # Should be (5966, 4)

# Train SVR model
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)
print("Model trained")

# Test with a sample input to ensure it works with 4 features
sample_input = X_test.iloc[[0]]
sample_pred = svr_model.predict(sample_input)
print(f"Sample prediction test - Input shape: {sample_input.shape}, Prediction: {sample_pred[0]:.2f}")

# Save model
joblib.dump(svr_model, 'svr_model_top4.pkl')
print("Model saved as svr_model_top4.pkl")

# Evaluate model
from sklearn.metrics import mean_squared_error, r2_score
y_pred = svr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model evaluation - MSE: {mse:.4f}, R²: {r2:.4f}")

print("\n--- MODEL TRAINING COMPLETE ---\n") 