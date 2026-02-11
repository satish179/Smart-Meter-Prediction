import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    """
    Loads raw data, performs cleaning, and adds features.
    Returns:
        df (pd.DataFrame): Processed dataframe ready for training/inference
        scalers (dict): Dictionary of fitted scaler objects
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # 1. Parse Dates
    try:
        df['Datetime'] = pd.to_datetime(df['date'], dayfirst=True)
    except Exception as e:
        print(f"Date parsing error: {e}")
        # Try inference if specific format fails
        df['Datetime'] = pd.to_datetime(df['date'])
        
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)

    # 2. Basic Cleaning
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Resample to 1 Hour to ensure stability and predictability
    # 1-minute data is too noisy for non-leakage regression
    df = df.resample('1H').mean(numeric_only=True)
    df.interpolate(method='time', inplace=True)

    # 3. Units Conversion
    # Original 'active_power' is likely in Watts given values ~200-1000 for a home
    df['active_power_kw'] = df['active_power'] / 1000.0
    
    # Target: Energy (kWh) in this minute
    # Power (kW) * Time (hours) = kW * (1/60)
    df['energy_kwh'] = df['active_power_kw'] / 60.0

    # 4. Feature Engineering
    print("Engineering features...")
    
    # A. Time Features (Cyclic)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    df['is_weekend'] = (df.index.weekday >= 5).astype(int)
    
    # B. Weather Rename
    rename_map = {
        'temp': 'Temperature',
        'humidity': 'Humidity',
        'pressure': 'Pressure', 
        'speed': 'WindSpeed'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # C. Lag Features (Strictly Past Information)
    # We want to predict energy at time T.
    # We can use Energy at T-1h, T-24h.
    target = 'energy_kwh'
    
    # Lag 1 Hour (1 step)
    df['lag_1h'] = df[target].shift(1)
    df['lag_2h'] = df[target].shift(2)
    df['lag_3h'] = df[target].shift(3)
    # Lag 24 Hours (Daily)
    df['lag_24h'] = df[target].shift(24)
    # Lag 48 Hours (2 Days ago)
    df['lag_48h'] = df[target].shift(48)
    # Lag 1 Week (168 Hours)
    df['lag_168h'] = df[target].shift(168)
    
    # D. Rolling Features
    # Rolling mean of last 3 hours
    df['rolling_mean_3h'] = df[target].shift(1).rolling(window=3).mean()
    df['rolling_mean_6h'] = df[target].shift(1).rolling(window=6).mean()
    df['rolling_mean_12h'] = df[target].shift(1).rolling(window=12).mean()
    df['rolling_mean_24h'] = df[target].shift(1).rolling(window=24).mean()
    df['rolling_std_24h'] = df[target].shift(1).rolling(window=24).std()
    df['rolling_max_24h'] = df[target].shift(1).rolling(window=24).max()
    
    # E. Interaction Features (Critical for Boosting)
    # Effect of previous hour usage differs by temperature (AC load)
    df['interact_lag1_temp'] = df['lag_1h'] * df['Temperature']
    # Effect of time of day on usage momentum
    df['interact_lag1_hour'] = df['lag_1h'] * df['hour_sin']
    
    # Drop NaNs created by lags
    df.dropna(inplace=True)
    
    return df

def save_preprocessors(df, target_col='energy_kwh', save_dir='models'):
    """Fits and saves scalers for features and target."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    features = [c for c in df.columns if c != target_col]
    
    # We don't necessarily need to scale for XGBoost, but it helps for LSTM/Linear models.
    # Let's save a generic feature list too.
    joblib.dump(features, os.path.join(save_dir, 'feature_names.pkl'))
    print(f"Saved {len(features)} feature names.")
    
    return features
