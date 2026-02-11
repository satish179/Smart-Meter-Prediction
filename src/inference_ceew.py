"""
Inference module for CEEW BR04 production model
"""
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


class CEEWEnergyPredictor:
    """Predictor using CEEW BR04 model (92% R2, year-long training)."""

    def __init__(self, models_dir='models'):
        self.model_path = os.path.join(models_dir, 'ceew_br04_production.h5')
        self.scaler_path = os.path.join(models_dir, 'ceew_br04_scaler.pkl')
        self.features_path = os.path.join(models_dir, 'ceew_br04_features.pkl')

        self.model = None
        self.scaler = None
        self.feature_names = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Load model, scaler, and feature names."""
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            self.model.compile(optimizer='adam', loss='mse')
            self.scaler = joblib.load(self.scaler_path)
            self.feature_names = joblib.load(self.features_path)
            print('CEEW BR04 model loaded successfully')
            print(f'Features: {len(self.feature_names)}')
        except Exception as e:
            print(f'Error loading CEEW model: {e}')

    def predict_one(self, target_date, current_weather, historical_data, model_type='CEEW'):
        """
        Predict energy for a single timestamp.

        Args:
            target_date: datetime to predict for
            current_weather: dict with Temperature, Humidity, Pressure, WindSpeed
            historical_data: DataFrame with historical consumption
            model_type: ignored (always uses CEEW model)

        Returns:
            Predicted energy in kWh
        """
        if self.model is None:
            return 0.0

        # Feature engineering
        hour = target_date.hour
        day_of_week = target_date.weekday()
        month = target_date.month
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # Get lags from historical data (use last known values as fallback)
        try:
            if 'energy_kwh' not in historical_data.columns or historical_data.empty:
                raise ValueError('historical_data must include non-empty energy_kwh')

            lag_1h = historical_data['energy_kwh'].iloc[-1]
            lag_24h = historical_data['energy_kwh'].iloc[-24] if len(historical_data) >= 24 else lag_1h
            lag_168h = historical_data['energy_kwh'].iloc[-168] if len(historical_data) >= 168 else lag_1h
            rolling_mean_24h = historical_data['energy_kwh'].iloc[-24:].mean() if len(historical_data) >= 24 else lag_1h
            rolling_std_24h = historical_data['energy_kwh'].iloc[-24:].std() if len(historical_data) >= 24 else 0.1
        except Exception:
            lag_1h = 0.5
            lag_24h = 0.5
            lag_168h = 0.5
            rolling_mean_24h = 0.5
            rolling_std_24h = 0.1

        voltage = current_weather.get('Voltage', 230.0)
        current = current_weather.get('Current', 2.0)
        frequency = current_weather.get('Frequency', 50.0)

        features = {
            'lag_1h': lag_1h,
            'lag_24h': lag_24h,
            'lag_168h': lag_168h,
            'rolling_mean_24h': rolling_mean_24h,
            'rolling_std_24h': rolling_std_24h,
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'z_Avg Voltage (Volt)': voltage,
            'z_Avg Current (Amp)': current,
            'y_Freq (Hz)': frequency,
            'temp': current_weather.get('Temperature', 25.0),
            'humidity': current_weather.get('Humidity', 60.0),
            'pressure': current_weather.get('Pressure', 1013.0),
            'wind_speed': current_weather.get('WindSpeed', 2.0),
        }

        X = np.array([[features[f] for f in self.feature_names]])
        X_scaled = self.scaler.transform(X)
        pred_log = self.model.predict(X_scaled, verbose=0)[0][0]
        pred_kwh = np.expm1(pred_log)
        return max(0.0, float(pred_kwh))

    def predict_forecast(self, start_time, forecast_weather, historical_data, model_type='CEEW'):
        """
        Generate forecast by rolling predictions into lag history.

        Args:
            start_time: starting datetime
            forecast_weather: list of weather dicts
            historical_data: DataFrame with historical data
            model_type: ignored (always CEEW)

        Returns:
            List of dicts with 'time' and 'kwh' keys
        """
        predictions = []
        current_time = pd.Timestamp(start_time)

        history = historical_data.copy()
        if not isinstance(history.index, pd.DatetimeIndex):
            history.index = pd.to_datetime(history.index)
        if 'energy_kwh' not in history.columns:
            history['energy_kwh'] = 0.5
        history = history[['energy_kwh']].sort_index()

        for weather in forecast_weather:
            pred = self.predict_one(current_time, weather, history, model_type=model_type)
            predictions.append({'time': current_time, 'kwh': pred})
            new_row = pd.DataFrame({'energy_kwh': [pred]}, index=[current_time])
            history = pd.concat([history, new_row]).sort_index()
            current_time += pd.Timedelta(hours=1)

        return predictions


# For backward compatibility
EnergyPredictor = CEEWEnergyPredictor
