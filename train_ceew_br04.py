"""
Train DNN on CEEW BR04 smart meter data.
Much larger dataset than Garud - should achieve even higher accuracy.
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
print("TRAINING DNN ON CEEW BR04 SMART METER DATA")
print("="*80)

# Load only BR04 meter
print("\nLoading CEEW BR04 meter data...")
df = pd.read_csv("archive_extracted/CEEW - Smart meter data Bareilly 2020.csv")
df = df[df['meter'] == 'BR04'].copy()

print(f"Total records: {len(df):,}")

# Convert timestamp
df['x_Timestamp'] = pd.to_datetime(df['x_Timestamp'])
df.set_index('x_Timestamp', inplace=True)
df.sort_index(inplace=True)

print(f"Date range: {df.index.min()} to {df.index.max()}")

# Resample to hourly (to reduce size)
df_hourly = df.resample('H').agg({
    't_kWh': 'sum',
    'z_Avg Voltage (Volt)': 'mean',
    'z_Avg Current (Amp)': 'mean',
    'y_Freq (Hz)': 'mean'
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
df_hourly['lag_168h'] = df_hourly['t_kWh'].shift(168)  # 1 week

# Rolling stats
df_hourly['rolling_mean_24h'] = df_hourly['t_kWh'].shift(1).rolling(24).mean()
df_hourly['rolling_std_24h'] = df_hourly['t_kWh'].shift(1).rolling(24).std()

# Clean
df_hourly = df_hourly.dropna()

target_col = 't_kWh'
feature_cols = ['lag_1h', 'lag_24h', 'lag_168h', 'rolling_mean_24h', 'rolling_std_24h',
                'hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos',
                'z_Avg Voltage (Volt)', 'z_Avg Current (Amp)', 'y_Freq (Hz)']

X = df_hourly[feature_cols].values
y = df_hourly[target_col].values

# Log transform
y_log = np.log1p(y)

# TEMPORAL SPLIT
split_idx = int(len(X) * 0.9)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y_log[:split_idx], y_log[split_idx:]

print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,} (temporal split)")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Deep Neural Network
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
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Train
print("\nTraining DNN...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=128,
    validation_split=0.15,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ],
    verbose=0  # Silent to avoid encoding errors
)

# Evaluate
pred_log = model.predict(X_test_scaled, verbose=0).flatten()
pred_real = np.expm1(pred_log)
y_test_real = np.expm1(y_test)

r2 = r2_score(y_test_real, pred_real)
mae = mean_absolute_error(y_test_real, pred_real)

print(f"\n{'='*80}")
print(f"RESULTS: CEEW BR04")
print(f"{'='*80}")
print(f"R² Score: {r2:.4f} ({r2*100:.2f}%)")
print(f"MAE:      {mae:.6f} kWh")

if r2 >= 0.85:
    print("\n✅ TARGET ACHIEVED!")
    # Save model
    model.save('models/dnn_CEEW_BR04.h5')
    joblib.dump(scaler, 'models/dnn_CEEW_BR04_scaler.pkl')
    joblib.dump(feature_cols, 'models/dnn_CEEW_BR04_features.pkl')
    print(f"Model saved as dnn_CEEW_BR04.h5")
    
print(f"={'='*80}")
