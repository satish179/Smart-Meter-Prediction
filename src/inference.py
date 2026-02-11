import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

class EnergyPredictor:
    def __init__(self, models_dir='models'):
        self.rf_path = os.path.join(models_dir, 'xgboost_model.pkl') # Keep name for compat, actually RF now
        self.features_path = os.path.join(models_dir, 'feature_names.pkl')
        self.lstm_path = os.path.join(models_dir, 'lstm_model.h5')
        self.scaler_X_path = os.path.join(models_dir, 'scaler_X.pkl')
        self.scaler_y_path = os.path.join(models_dir, 'scaler_y.pkl')
        
        self.rf_model = None
        self.lstm_model = None
        self.feature_names = None
        self.scaler_X = None
        self.scaler_y = None
        
        self._load_artifacts()
        
    def _load_artifacts(self):
        # Load RF
        try:
            self.rf_model = joblib.load(self.rf_path)
            self.feature_names = joblib.load(self.features_path)
            print("RF Model loaded.")
        except:
            print("RF Model not found.")

        # Load LSTM
        try:
            if os.path.exists(self.lstm_path):
                self.lstm_model = tf.keras.models.load_model(self.lstm_path)
                self.scaler_X = joblib.load(self.scaler_X_path)
                self.scaler_y = joblib.load(self.scaler_y_path)
                print("LSTM Model loaded.")
        except Exception as e:
            print(f"LSTM Load Error: {e}")

    def predict_one(self, target_date, current_weather, historical_data, model_type='RF'):
        """
        model_type: 'RF' or 'LSTM'
        """
        # 1. Feature Engineering
        features = {}
        
        # Ensure target_date is a pandas Timestamp for easy attribute access
        target_date = pd.Timestamp(target_date)
        
        # A. Time Features
        features['hour_sin'] = np.sin(2 * np.pi * target_date.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * target_date.hour / 24)
        features['day_sin'] = np.sin(2 * np.pi * target_date.dayofyear / 365)
        features['day_cos'] = np.cos(2 * np.pi * target_date.dayofyear / 365)
        features['month'] = target_date.month
        features['hour'] = target_date.hour
        features['is_weekend'] = 1 if target_date.weekday() >= 5 else 0
        
        # B. Weather
        features['Temperature'] = current_weather.get('Temperature', 25.0)
        features['Humidity'] = current_weather.get('Humidity', 60.0)
        features['Pressure'] = current_weather.get('Pressure', 1013.0)
        features['WindSpeed'] = current_weather.get('WindSpeed', 5.0)
        
        # C. Lags & Rolling (Shared logic)
        target_col = 'energy_kwh'
        
        # Lag 1h
        t_minus_1h = target_date - pd.Timedelta(hours=1)
        try:
            idx = historical_data.index.get_indexer([t_minus_1h], method='nearest')[0]
            features['lag_1h'] = historical_data.iloc[idx][target_col]
        except:
            features['lag_1h'] = 0.5

        # Lag 2h
        t_minus_2h = target_date - pd.Timedelta(hours=2)
        try:
            idx = historical_data.index.get_indexer([t_minus_2h], method='nearest')[0]
            features['lag_2h'] = historical_data.iloc[idx][target_col]
        except:
            features['lag_2h'] = 0.5

        # Lag 3h
        t_minus_3h = target_date - pd.Timedelta(hours=3)
        try:
            idx = historical_data.index.get_indexer([t_minus_3h], method='nearest')[0]
            features['lag_3h'] = historical_data.iloc[idx][target_col]
        except:
            features['lag_3h'] = 0.5
            
        # Lag 24h
        t_minus_24h = target_date - pd.Timedelta(hours=24)
        try:
            idx = historical_data.index.get_indexer([t_minus_24h], method='nearest')[0]
            features['lag_24h'] = historical_data.iloc[idx][target_col]
        except:
            features['lag_24h'] = 0.5

        # Lag 48h
        t_minus_48h = target_date - pd.Timedelta(hours=48)
        try:
            idx = historical_data.index.get_indexer([t_minus_48h], method='nearest')[0]
            features['lag_48h'] = historical_data.iloc[idx][target_col]
        except:
            features['lag_48h'] = 0.5
            
        # Lag 168h
        t_minus_168 = target_date - pd.Timedelta(hours=168)
        try:
            idx = historical_data.index.get_indexer([t_minus_168], method='nearest')[0]
            features['lag_168h'] = historical_data.iloc[idx][target_col]
        except:
            features['lag_168h'] = 0.5
            
        # Rolling Means
        try:
            features['rolling_mean_3h'] = historical_data[target_col].tail(3).mean()
            features['rolling_mean_6h'] = historical_data[target_col].tail(6).mean()
            features['rolling_mean_12h'] = historical_data[target_col].tail(12).mean()
            features['rolling_mean_24h'] = historical_data[target_col].tail(24).mean()
            features['rolling_std_24h'] = historical_data[target_col].tail(24).std()
            features['rolling_max_24h'] = historical_data[target_col].tail(24).max()
        except:
            features['rolling_mean_3h'] = 0.5
            features['rolling_mean_6h'] = 0.5
            features['rolling_mean_12h'] = 0.5
            features['rolling_mean_24h'] = 0.5
            features['rolling_std_24h'] = 0.0
            features['rolling_max_24h'] = 1.0
            
        # Interaction
        features['interact_lag1_temp'] = features.get('lag_1h', 0) * features.get('Temperature', 0)
        features['interact_lag1_hour'] = features.get('lag_1h', 0) * features.get('hour_sin', 0)
            
        # D. Assemble DataFrame
        input_df = pd.DataFrame([features])
        
        # Ensure order matches training
        if self.feature_names is None:
             return 0.0 # Not loaded
             
        final_input = pd.DataFrame(0, index=[0], columns=self.feature_names)
        for col in features:
            if col in self.feature_names:
                final_input[col] = input_df[col]
        
        # PREDICT
        if model_type == 'LSTM' and self.lstm_model:
            # Scale
            X_scaled = self.scaler_X.transform(final_input)
            # Reshape [1, 1, features]
            X_reshaped = X_scaled.reshape((1, 1, X_scaled.shape[1]))
            # Predict
            pred_scaled = self.lstm_model.predict(X_reshaped, verbose=0)
            # Inverse
            pred_kwh = self.scaler_y.inverse_transform(pred_scaled)[0][0]
            
        else:
            # RF (Default)
            if self.rf_model:
                pred_log = self.rf_model.predict(final_input)[0]
                pred_kwh = np.expm1(pred_log)
            else:
                return 0.0
        
        return max(0.0, float(pred_kwh))

    def predict_forecast(self, start_date, weather_forecasts, recent_history, model_type='RF'):
        predictions = []
        history = recent_history.copy()
        current_time = start_date
        
        for weather in weather_forecasts:
            pred = self.predict_one(current_time, weather, history, model_type)
            predictions.append({'time': current_time, 'kwh': pred})
            
            # Update History
            new_row = pd.DataFrame({'energy_kwh': [pred]}, index=[current_time])
            history = pd.concat([history, new_row])
            current_time += pd.Timedelta(hours=1)
            
        return predictions
