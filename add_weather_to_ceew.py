"""
Add weather to CEEW (archive) datasets as well
Using same hybrid pattern approach for 2020 data
"""
import pandas as pd
import numpy as np
import pickle
import os

print("="*80)
print("ADDING WEATHER TO CEEW DATASETS (ARCHIVE)")
print("="*80)

# Load weather patterns (already extracted)
print("\nStep 1: Loading weather patterns...")
df_orig = pd.read_csv("energy_weather_raw_data.csv")
df_orig['date'] = pd.to_datetime(df_orig['date'], errors='coerce')
df_orig.dropna(subset=['date'], inplace=True)

df_orig['month'] = df_orig['date'].dt.month
df_orig['hour'] = df_orig['date'].dt.hour

# Create patterns
patterns = {}
for month in range(1, 13):
    month_data = df_orig[df_orig['month'] == month]
    patterns[month] = {
        'temp_mean': month_data['temp'].mean(),
        'temp_std': month_data['temp'].std(),
        'humidity_mean': month_data['humidity'].mean(),
        'humidity_std': month_data['humidity'].std(),
        'pressure_mean': month_data['pressure'].mean(),
        'pressure_std': month_data['pressure'].std(),
        'wind_mean': month_data['speed'].mean(),
        'wind_std': month_data['speed'].std()
    }

print("✓ Patterns loaded")

def generate_weather(dt, patterns):
    """Generate realistic weather"""
    month = dt.month
    hour = dt.hour
    
    pattern = patterns[month]
    hour_factor = np.sin((hour - 6) * np.pi / 12)
    
    temp = pattern['temp_mean'] + hour_factor * 3 + np.random.normal(0, pattern['temp_std'] * 0.3)
    humidity = np.clip(pattern['humidity_mean'] + np.random.normal(0, pattern['humidity_std'] * 0.3), 0, 100)
    pressure = pattern['pressure_mean'] + np.random.normal(0, pattern['pressure_std'] * 0.3)
    wind = np.clip(pattern['wind_mean'] + np.random.normal(0, pattern['wind_std'] * 0.3), 0, 20)
    
    return {
        'temp': round(temp, 1),
        'humidity': round(humidity, 1),
        'pressure': round(pressure, 1),
        'wind_speed': round(wind, 2)
    }

# Process CEEW BR04 (our test dataset)
print("\nStep 2: Processing CEEW BR04...")

df = pd.read_csv("archive_extracted/CEEW - Smart meter data Bareilly 2020.csv")
df = df[df['meter'] == 'BR04'].copy()

print(f"  Total records: {len(df):,}")

df['x_Timestamp'] = pd.to_datetime(df['x_Timestamp'])
df.sort_values('x_Timestamp', inplace=True)

print(f"  Date range: {df['x_Timestamp'].min()} to {df['x_Timestamp'].max()}")

# Generate weather
print(f"  Generating weather...", end='')
weather_data = []
for dt in df['x_Timestamp']:
    weather = generate_weather(dt, patterns)
    weather_data.append(weather)

weather_df = pd.DataFrame(weather_data)
df = pd.concat([df.reset_index(drop=True), weather_df], axis=1)

# Save
os.makedirs('enhanced_datasets', exist_ok=True)
output_file = 'enhanced_datasets/CEEW_BR04_with_weather.csv'
df.to_csv(output_file, index=False)

print(f" ✓")
print(f"  Saved: {output_file}")
print(f"  Columns: {list(df.columns)}")

print(f"\n{'='*80}")
print("CEEW ENHANCEMENT COMPLETE")
print(f"{'='*80}")
print("\nNext: Train model on enhanced CEEW to compare with baseline 28.80%")
