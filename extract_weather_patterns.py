"""
Step 1: Extract weather patterns from original dataset
Creates a pattern library based on month and hour
"""
import pandas as pd
import numpy as np
import pickle

print("="*80)
print("EXTRACTING WEATHER PATTERNS FROM ORIGINAL DATASET")
print("="*80)

# Load original dataset
df = pd.read_csv("energy_weather_raw_data.csv")
df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)

print(f"Original dataset: {len(df):,} records")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Extract temporal features
df['month'] = df['date'].dt.month
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek

# Weather columns
weather_cols = ['temp', 'pressure', 'humidity', 'speed']  # speed = wind speed

print(f"\nWeather columns: {weather_cols}")

# Calculate patterns by month and hour
patterns = {}

for month in range(1, 13):
    patterns[month] = {}
    for hour in range(24):
        subset = df[(df['month'] == month) & (df['hour'] == hour)]
        
        if len(subset) > 0:
            patterns[month][hour] = {
                'temp_mean': subset['temp'].mean(),
                'temp_std': subset['temp'].std(),
                'pressure_mean': subset['pressure'].mean(),
                'pressure_std': subset['pressure'].std(),
                'humidity_mean': subset['humidity'].mean(),
                'humidity_std': subset['humidity'].std(),
                'speed_mean': subset['speed'].mean(),
                'speed_std': subset['speed'].std(),
                'count': len(subset)
            }
        else:
            # Fallback to month average if no data for specific hour
            month_data = df[df['month'] == month]
            patterns[month][hour] = {
                'temp_mean': month_data['temp'].mean() if len(month_data) > 0 else 20.0,
                'temp_std': month_data['temp'].std() if len(month_data) > 0 else 5.0,
                'pressure_mean': month_data['pressure'].mean() if len(month_data) > 0 else 1013.0,
                'pressure_std': month_data['pressure'].std() if len(month_data) > 0 else 10.0,
                'humidity_mean': month_data['humidity'].mean() if len(month_data) > 0 else 60.0,
                'humidity_std': month_data['humidity'].std() if len(month_data) > 0 else 15.0,
                'speed_mean': month_data['speed'].mean() if len(month_data) > 0 else 2.0,
                'speed_std': month_data['speed'].std() if len(month_data) > 0 else 1.0,
                'count': 0
            }

# Save patterns
with open('weather_patterns.pkl', 'wb') as f:
    pickle.dump(patterns, f)

print(f"\n✅ Weather patterns extracted and saved to weather_patterns.pkl")

# Display sample pattern
print(f"\nSample pattern (May, 12:00):")
sample = patterns[5][12]
for key, val in sample.items():
    if key != 'count':
        print(f"  {key}: {val:.2f}")

print(f"\n{'='*80}")
print("PATTERN EXTRACTION COMPLETE")
print(f"{'='*80}")
