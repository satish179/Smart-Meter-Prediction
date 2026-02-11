"""
Train CEEW BR04 with weather features
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

print("="*80)
print("TRAINING CEEW BR04 WITH WEATHER")
print("="*80)

# Load enhanced CEEW data
df = pd.read_csv("enhanced_datasets/CEEW_BR04_with_weather.csv")
df['x_Timestamp'] = pd.to_datetime(df['x_Timestamp'])
df.set_index('x_Timestamp', inplace=True)
df.sort_index(inplace=True)

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

# Add features
df_hourly['hour'] = df_hourly.index.hour
df_hourly['day_of_week'] = df_hourly.index.dayofweek
df_hourly['month'] = df_hourly.index.month
df_hourly['hour_sin'] = np.sin(2 * np.pi * df_hourly['hour'] / 24)
df_hourly['hour_cos'] = np.cos(2 * np.pi * df_hourly['hour'] / 24)

df_hourly['lag_1h'] = df_hourly['t_kWh'].shift(1)
df_hourly['lag_24h'] = df_hourly['t_kWh'].shift(24)
df_hourly['lag_168h'] = df_hourly['t_kWh'].shift(168)

df_hourly['rolling_mean_24h'] = df_hourly['t_kWh'].shift(1).rolling(24).mean()
df_hourly['rolling_std_24h'] = df_hourly['t_kWh'].shift(1).rolling(24).std()

df_hourly = df_hourly.dropna()

feature_cols = [
    'lag_1h', 'lag_24h', 'lag_168h', 'rolling_mean_24h', 'rolling_std_24h',
    'hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos',
    'z_Avg Voltage (Volt)', 'z_Avg Current (Amp)', 'y_Freq (Hz)',
    'temp', 'humidity', 'pressure', 'wind_speed'  # Weather features
]

X = df_hourly[feature_cols].values
y = np.log1p(df_hourly['t_kWh'].values)

split_idx = int(len(X) * 0.9)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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

model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')

model.fit(X_train_scaled, y_train, epochs=100, batch_size=128,
          validation_split=0.15,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)],
          verbose=0)

pred_log = model.predict(X_test_scaled, verbose=0).flatten()
r2 = r2_score(np.expm1(y_test), np.expm1(pred_log))

print(f"\nRESULTS:")
print(f"  Baseline (no weather): 28.80%")
print(f"  Enhanced (with weather): {r2*100:.2f}%")
print(f"  Change: {(r2-0.288)*100:+.2f}%")

if r2 > 0.288:
    print("\n✅ Weather improved CEEW accuracy!")
else:
    print("\n⚠️  Weather did not help CEEW")

print(f"\n{'='*80}")
