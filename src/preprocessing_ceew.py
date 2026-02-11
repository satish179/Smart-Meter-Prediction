"""
Preprocessing for CEEW BR04 dataset
"""
import pandas as pd
import numpy as np

def load_and_preprocess_ceew_data(filepath='enhanced_datasets/CEEW_BR04_with_weather.csv'):
    """
    Load and preprocess CEEW BR04 data for visualization and prediction
    """
    # Load data
    df = pd.read_csv(filepath)
    df['x_Timestamp'] = pd.to_datetime(df['x_Timestamp'])
    df.set_index('x_Timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # Resample to hourly for better performance
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
    
    # Rename columns for compatibility with app
    df_hourly = df_hourly.rename(columns={
        't_kWh': 'energy_kwh',
        'z_Avg Voltage (Volt)': 'Voltage',
        'z_Avg Current (Amp)': 'Current',
        'y_Freq (Hz)': 'Frequency',
        'temp': 'Temperature',
        'humidity': 'Humidity',
        'pressure': 'Pressure',
        'wind_speed': 'WindSpeed'
    })
    
    # Add active_power_kw for display (approximate from energy)
    df_hourly['active_power_kw'] = df_hourly['energy_kwh']  # For hourly data, they're equivalent
    
    return df_hourly

# For backward compatibility
def load_and_preprocess_data(filepath='enhanced_datasets/CEEW_BR04_with_weather.csv'):
    """Wrapper for CEEW data"""
    return load_and_preprocess_ceew_data(filepath)
