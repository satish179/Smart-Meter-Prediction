"""
Properly train and save CEEW BR04 enhanced model for production use
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

print("="*80)
print("TRAINING & SAVING CEEW BR04 ENHANCED MODEL FOR PRODUCTION")
print("="*80)

# Load enhanced CEEW data
df = pd.read_csv("enhanced_datasets/CEEW_BR04_with_weather.csv")
df['x_Timestamp'] = pd.to_datetime(df['x_Timestamp'])
df.set_index('x_Timestamp', inplace=True)
df.sort_index(inplace=True)

print(f"\nTotal records: {len(df):,}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Resample to hourly
df_hourly = df.resample('H').agg({
    't_kWh': 'sum',
    'z_Avg Voltage (Volt)': 'mean',
    'z_Avg Current (Amp)': 'mean',
    'y_Freq (Hz)': 'mean',
    'temp': 'mean',
    'humidity': 'mean',
    'pressure': 'mean',
    'wind_speed': 'mean'
})

print(f"Hourly records: {len(df_hourly):,}")

# Add temporal features
df_hourly['hour'] = df_hourly.index.hour
df_hourly['day_of_week'] = df_hourly.index.dayofweek
df_hourly['month'] = df_hourly.index.month
df_hourly['hour_sin'] = np.sin(2 * np.pi * df_hourly['hour'] / 24)
df_hourly['hour_cos'] = np.cos(2 * np.pi * df_hourly['hour'] / 24)

# Lag features
df_hourly['lag_1h'] = df_hourly['t_kWh'].shift(1)
df_hourly['lag_24h'] = df_hourly['t_kWh'].shift(24)
df_hourly['lag_168h'] = df_hourly['t_kWh'].shift(168)

# Rolling stats
df_hourly['rolling_mean_24h'] = df_hourly['t_kWh'].shift(1).rolling(24).mean()
df_hourly['rolling_std_24h'] = df_hourly['t_kWh'].shift(1).rolling(24).std()

# Clean
df_hourly = df_hourly.dropna()

# Feature set
feature_cols = [
    'lag_1h', 'lag_24h', 'lag_168h',
    'rolling_mean_24h', 'rolling_std_24h',
    'hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos',
    'z_Avg Voltage (Volt)', 'z_Avg Current (Amp)', 'y_Freq (Hz)',
    'temp', 'humidity', 'pressure', 'wind_speed'
]

print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

X = df_hourly[feature_cols].values
y = np.log1p(df_hourly['t_kWh'].values)

# Temporal split
split_idx = int(len(X) * 0.9)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model
print("\nBuilding DNN...")
model = Sequential([
    Dense(256, activation='relu', input_dim=len(feature_cols)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse'
)

# Train
print("Training...")
model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=128,
    validation_split=0.15,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ],
    verbose=0
)

# Evaluate
pred_log = model.predict(X_test_scaled, verbose=0).flatten()
pred_real = np.expm1(pred_log)
y_test_real = np.expm1(y_test)

r2 = r2_score(y_test_real, pred_real)
mae = mean_absolute_error(y_test_real, pred_real)

print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")
print(f"R² Score: {r2:.4f} ({r2*100:.2f}%)")
print(f"MAE:      {mae:.6f} kWh")

# Save model
os.makedirs('models', exist_ok=True)

model.save('models/ceew_br04_production.h5')
joblib.dump(scaler, 'models/ceew_br04_scaler.pkl')
joblib.dump(feature_cols, 'models/ceew_br04_features.pkl')

# Also save sample data for reference
sample_data = {
    'features': feature_cols,
    'r2_score': r2,
    'mae': mae,
    'training_records': len(X_train),
    'test_records': len(X_test),
    'date_range': f"{df_hourly.index.min()} to {df_hourly.index.max()}"
}
joblib.dump(sample_data, 'models/ceew_br04_metadata.pkl')

print(f"\n✅ MODEL SAVED:")
print(f"  models/ceew_br04_production.h5")
print(f"  models/ceew_br04_scaler.pkl")
print(f"  models/ceew_br04_features.pkl")
print(f"  models/ceew_br04_metadata.pkl")
print(f"\n{'='*80}")
