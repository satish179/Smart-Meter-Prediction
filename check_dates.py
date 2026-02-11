
import pandas as pd
import os

files = ['energy_weather_raw_data.csv', 'live_data_log.csv']

for f in files:
    if os.path.exists(f):
        try:
            df = pd.read_csv(f)
            # Try to infer time column
            time_col = None
            for col in df.columns:
                if 'time' in col.lower() or 'date' in col.lower():
                    time_col = col
                    break
            
            if time_col:
                df[time_col] = pd.to_datetime(df[time_col])
                print(f"File: {f}")
                print(f"  Min Date: {df[time_col].min()}")
                print(f"  Max Date: {df[time_col].max()}")
                print(f"  Rows: {len(df)}")
            else:
                print(f"File: {f} - No time column found")
        except Exception as e:
            print(f"Error reading {f}: {e}")
    else:
        print(f"File not found: {f}")
