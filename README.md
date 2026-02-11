# Lumina Energy Analytics

Streamlit application for smart-home energy analytics with:
- Historical consumption dashboard
- Virtual smart meter live telemetry
- 24-hour load forecast using a trained CEEW model
- Live weather integration
- Solar production and ROI estimation

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure WeatherAPI key (pick one):
- `.streamlit/secrets.toml`:
```toml
[secrets]
WEATHER_API_KEY = "your_key_here"
```
- or environment variable:
```bash
set WEATHER_API_KEY=your_key_here
```

4. Run app:

```bash
streamlit run app.py
```

## Data/Model Paths

- Main dataset: `enhanced_datasets/CEEW_BR04_with_weather.csv`
- Model artifacts: `models/ceew_br04_production.h5`, `models/ceew_br04_scaler.pkl`, `models/ceew_br04_features.pkl`
- Live telemetry log: `live_data_log.csv`

## Notes

- Forecast now rolls predictions recursively across the horizon (lag features update at each step).
- Live monitor appends generated telemetry to `live_data_log.csv`.
- Do not commit real API keys.
