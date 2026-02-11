import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import datetime
import time
import os
import csv

# Custom Modules
from src.inference_ceew import CEEWEnergyPredictor as EnergyPredictor
from src.preprocessing_ceew import load_and_preprocess_ceew_data as load_and_preprocess_data
from src.weather import get_live_weather, get_weather_forecast, detect_user_location, get_major_city
from streamlit_js_eval import get_geolocation
from src.anomaly import AnomalyDetector
from src.virtual_meter import VirtualSmartMeter
from src.solar import SolarSimulator
from src.chatbot import EnergyChatbot

# --- 1. Page Configuration & CSS ---
st.set_page_config(
    page_title="Lumina | Energy Analytics",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional/Clean Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    :root {
        --bg: #0b1220;
        --panel: #121b2b;
        --panel-soft: #18253a;
        --stroke: #2a3b57;
        --text: #e7edf7;
        --muted: #9db0ca;
        --accent: #38bdf8;
        --accent-2: #f59e0b;
        --ok: #10b981;
        --warn: #f59e0b;
        --danger: #ef4444;
    }
    
    /* Global Background */
    .stApp {
        background:
          radial-gradient(1200px 500px at 82% -5%, rgba(56, 189, 248, 0.12), transparent 60%),
          radial-gradient(900px 420px at -10% 0%, rgba(245, 158, 11, 0.08), transparent 60%),
          var(--bg);
    }
    
    /* Styles for new components */
    .device-card {
        background-color: var(--panel);
        border: 1px solid var(--stroke);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: all 0.2s ease;
        height: 100%;
    }
    
    .device-active {
        border-color: var(--ok);
        background-color: rgba(16, 185, 129, 0.1);
    }
    
    .device-inactive {
        border-color: var(--stroke);
    }
    
    .status-dot {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .dot-green { background-color: #10b981; box-shadow: 0 0 8px rgba(16,185,129,0.5); }
    .dot-gray { background-color: #6b7280; }
    
    /* Metric Cards - Professional */
    .metric-card {
        background: linear-gradient(180deg, var(--panel-soft), var(--panel));
        border: 1px solid var(--stroke);
        border-radius: 12px;
        padding: 20px 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: #3e5a82;
        transform: translateY(-2px);
    }
    
    .metric-title {
        color: var(--muted);
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    
    .metric-value {
        color: var(--text);
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
    }
    
    .metric-suffix {
        font-size: 0.95rem;
        color: var(--muted);
        font-weight: 600;
        margin-left: 4px;
    }

    .hero-title {
        color: var(--text);
        font-size: clamp(2rem, 2.8vw, 3rem);
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.03em;
        margin-bottom: 4px;
    }
    .hero-sub {
        color: var(--muted);
        font-size: 1.05rem;
        margin-bottom: 6px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f1a2c;
        border-right: 1px solid var(--stroke);
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 10px;
        font-weight: 700;
        transition: background-color 0.2s;
        width: 100%;
    }

    .stTextInput input, .stDateInput input {
        border-radius: 10px !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 14px;
        border-bottom: 1px solid var(--stroke);
        padding-bottom: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 14px;
        font-weight: 700;
        color: var(--muted);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #fff;
        border-bottom-color: var(--accent-2);
    }

</style>
""", unsafe_allow_html=True)

# --- 2. System Initialization ---
LIVE_LOG_PATH = 'live_data_log.csv'
LIVE_LOG_COLUMNS = [
    'timestamp',
    'voltage_v',
    'current_a',
    'power_kw',
    'power_factor',
    'frequency_hz',
]

def dataframe_to_csv_bytes(df, include_index=False):
    return df.to_csv(index=include_index).encode('utf-8')

def hour_label(hour_value):
    return f"{int(hour_value):02d}:00"

def seed_demo_live_history(num_points=120):
    """Create lightweight synthetic live telemetry history for instant demo readiness."""
    now = pd.Timestamp.now()
    times = pd.date_range(end=now, periods=num_points, freq='s')
    baseline = 0.35 + 0.08 * np.sin(np.linspace(0, 6 * np.pi, num_points))
    noise = np.random.normal(0, 0.02, size=num_points)
    power = np.clip(baseline + noise, 0.05, None)
    return pd.DataFrame({'Time': times, 'Power': np.round(power, 3)})

def apply_quick_range(frame, quick_range, custom_range=None):
    if frame.empty:
        return frame.copy()
    now = pd.Timestamp.now()
    if quick_range == "24H":
        return frame.loc[frame.index >= (now - pd.Timedelta(hours=24))]
    if quick_range == "7D":
        return frame.loc[frame.index >= (now - pd.Timedelta(days=7))]
    if quick_range == "30D":
        return frame.loc[frame.index >= (now - pd.Timedelta(days=30))]
    if quick_range == "Custom" and custom_range and len(custom_range) == 2:
        start_date, end_date = custom_range
        mask = (frame.index.date >= start_date) & (frame.index.date <= end_date)
        return frame.loc[mask]
    return frame.copy()

def build_forecast_dataframe(raw_predictions, history_df):
    pred_df = pd.DataFrame(raw_predictions).copy()
    if pred_df.empty:
        return pred_df
    pred_df['time'] = pd.to_datetime(pred_df['time'], errors='coerce')
    pred_df = pred_df.dropna(subset=['time', 'kwh'])
    pred_df['kwh'] = pd.to_numeric(pred_df['kwh'], errors='coerce').clip(lower=0)
    pred_df = pred_df.dropna(subset=['kwh']).sort_values('time').reset_index(drop=True)
    baseline_std = float(history_df['energy_kwh'].tail(24 * 14).std()) if not history_df.empty else 0.05
    sigma = max(0.05, baseline_std * 0.25)
    horizon_scale = np.sqrt(np.arange(1, len(pred_df) + 1) / max(1, len(pred_df)))
    band = sigma * (0.7 + 0.6 * horizon_scale)
    pred_df['min_kwh'] = np.maximum(0, pred_df['kwh'] - band)
    pred_df['max_kwh'] = pred_df['kwh'] + band
    width = np.maximum(1e-6, pred_df['max_kwh'] - pred_df['min_kwh'])
    pred_df['confidence_pct'] = np.clip(100 - ((width / np.maximum(pred_df['kwh'], 0.05)) * 100), 35, 98)
    return pred_df

def generate_action_recommendations(current_kwh, forecast_df):
    actions = []
    if current_kwh > 0.65:
        actions.append("Current load is high. Shift heavy appliances to late morning or afternoon.")
    elif current_kwh > 0.35:
        actions.append("Usage is moderate. Avoid stacking HVAC and water heating in the same hour.")
    else:
        actions.append("Usage is efficient right now. This is a good window for discretionary loads.")
    if forecast_df is not None and not forecast_df.empty:
        peak_row = forecast_df.loc[forecast_df['kwh'].idxmax()]
        actions.append(
            f"Peak demand expected around {peak_row['time'].strftime('%I:%M %p')}. "
            "Pre-cool or pre-heat before this period."
        )
    actions.append("Enable anomaly alerts and review the History tab daily for recurring spikes.")
    return actions

def build_alert_table(history_df, forecast_df, live_alerts):
    rows = []
    if not history_df.empty:
        rolling = history_df['energy_kwh'].rolling(24, min_periods=12)
        baseline = rolling.mean().fillna(history_df['energy_kwh'].mean())
        std = rolling.std().fillna(history_df['energy_kwh'].std()).replace(0, np.nan).fillna(0.05)
        z = (history_df['energy_kwh'] - baseline) / std
        abnormal = history_df[z > 2.5].tail(10)
        for ts, val in abnormal['energy_kwh'].items():
            rows.append({
                'time': pd.Timestamp(ts),
                'severity': 'High' if val > history_df['energy_kwh'].quantile(0.95) else 'Medium',
                'issue': 'Unexpected usage spike',
                'impact': f"{val:.2f} kWh in one interval",
                'action': 'Check concurrent HVAC/heater/EV load and reschedule non-critical usage.'
            })
    if forecast_df is not None and not forecast_df.empty:
        peak = forecast_df['kwh'].max()
        avg = forecast_df['kwh'].mean()
        if peak > avg * 1.35:
            peak_row = forecast_df.loc[forecast_df['kwh'].idxmax()]
            rows.append({
                'time': peak_row['time'],
                'severity': 'Medium',
                'issue': 'Forecasted peak period',
                'impact': f"Expected peak {peak:.2f} kWh",
                'action': 'Move high-power activities to lower-demand hours.'
            })
    for raw in live_alerts[:8]:
        rows.append({
            'time': pd.Timestamp.now(),
            'severity': 'High',
            'issue': raw,
            'impact': 'Real-time anomaly trigger',
            'action': 'Inspect active devices and reduce load immediately if not intentional.'
        })
    if not rows:
        return pd.DataFrame(columns=['time', 'severity', 'issue', 'impact', 'action', 'status'])
    alert_df = pd.DataFrame(rows).drop_duplicates(subset=['issue', 'impact']).sort_values('time', ascending=False)
    alert_df['status'] = 'Open'
    return alert_df.reset_index(drop=True)

@st.cache_resource
def init_system():
    predictor = EnergyPredictor()
    try:
        df = load_and_preprocess_data('enhanced_datasets/CEEW_BR04_with_weather.csv')
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None, None
    return predictor, df

predictor, df = init_system()
if df is None: st.stop()

# --- LOAD PERSISTED LIVE DATA ---
try:
    if os.path.exists(LIVE_LOG_PATH):
        try:
            live_log = pd.read_csv(
                LIVE_LOG_PATH,
                on_bad_lines='skip',
                engine='python',
                usecols=['timestamp', 'power_kw']
            )
        except Exception:
            # Auto-repair malformed legacy logs (mixed 6/7+ fields) and retry.
            repaired_rows = []
            with open(LIVE_LOG_PATH, 'r', encoding='utf-8', errors='ignore', newline='') as f_in:
                reader = csv.reader(f_in)
                for row in reader:
                    if not row:
                        continue
                    if row[0] == 'timestamp':
                        repaired_rows.append(LIVE_LOG_COLUMNS)
                        continue
                    if len(row) < len(LIVE_LOG_COLUMNS):
                        continue
                    repaired_rows.append(row[:len(LIVE_LOG_COLUMNS)])

            if repaired_rows:
                with open(LIVE_LOG_PATH, 'w', encoding='utf-8', newline='') as f_out:
                    writer = csv.writer(f_out)
                    writer.writerows(repaired_rows)

            live_log = pd.read_csv(
                LIVE_LOG_PATH,
                on_bad_lines='skip',
                engine='python',
                usecols=['timestamp', 'power_kw']
            )
        if not live_log.empty:
            if 'timestamp' in live_log.columns and 'power_kw' in live_log.columns:
                live_log['Timestamp'] = pd.to_datetime(live_log['timestamp'], errors='coerce')
                live_log['power_kw'] = pd.to_numeric(live_log['power_kw'], errors='coerce')
                live_log = live_log.dropna(subset=['Timestamp', 'power_kw'])
                # Convert Power(kW) to Energy(kWh) assuming 1s interval
                live_log['energy_kwh'] = live_log['power_kw'] / 3600.0
                live_log.set_index('Timestamp', inplace=True)

                # Keep only necessary columns matching main df
                if 'energy_kwh' in live_log.columns:
                    df = pd.concat([df, live_log[['energy_kwh']]])
                    df = df.sort_index()
except Exception as e:
    st.error(f"Warning: Could not load past live logs: {e}")

# --- 3. Sidebar UI ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=60)
    st.title("Lumina")
    st.caption("Simple Energy Monitoring")
    st.divider()

    st.subheader("System Status")
    col_stat1, col_stat2 = st.columns(2)
    col_stat1.success("ONLINE")
    col_stat2.info("v2.5 Pro")
    st.divider()
    st.subheader("Setup")
    
    if 'location' not in st.session_state:
        st.session_state.location = ""
    if 'quick_demo_mode' not in st.session_state:
        st.session_state.quick_demo_mode = False
    if 'demo_bootstrapped' not in st.session_state:
        st.session_state.demo_bootstrapped = False

    # Legacy IP callback removed to enforce GPS
    def update_location_callback():
        pass 

    st.session_state.quick_demo_mode = st.toggle(
        "Demo Mode",
        value=st.session_state.quick_demo_mode,
        help="One-click setup with sample location and seeded visuals."
    )
    if st.session_state.quick_demo_mode and not st.session_state.location:
        st.session_state.location = "Bengaluru"
    if not st.session_state.quick_demo_mode:
        st.session_state.demo_bootstrapped = False

    st.markdown("**Location**")


    # Enforce Precise Location (GPS)
    geo_data = get_geolocation()
    if geo_data and 'coords' in geo_data:
        lat = geo_data['coords']['latitude']
        lon = geo_data['coords']['longitude']
        coord_string = f"{lat},{lon}"

        if 'last_geo_coords' not in st.session_state:
            st.session_state.last_geo_coords = None
            
        if st.session_state.last_geo_coords != coord_string:
            st.session_state.last_geo_coords = coord_string
            # Resolve using geopy to get Major City
            with st.spinner("Finding nearest major city..."):
                city = get_major_city(lat, lon)
                if city:
                    st.session_state.location = city
                else:
                    st.session_state.location = coord_string
            st.rerun()

    col_loc1, col_loc2 = st.columns([3, 1])
    with col_loc1:
        city_name = st.text_input("Region Name", key="location", label_visibility="collapsed")
    
    with col_loc2:
        # Button merely triggers rerun, allowing get_geolocation to fire again if needed
        st.button("Refresh", help="Refresh GPS Location")
    


    with st.expander("🌤️ Live Conditions", expanded=True):
        live_box = st.empty()
        with live_box.container():
            if city_name:
                live_weather = get_live_weather(city_name)
                resolved_name = live_weather.get("LocationName", city_name)
            else:
                live_weather = {
                    "Temperature": 0,
                    "Humidity": 0,
                    "Pressure": 0,
                    "WindSpeed": 0,
                    "LocationName": "Waiting...",
                }
                resolved_name = "Waiting..."
                st.info("Waiting for location details...")

            c1, c2, c3 = st.columns(3)
            c1.metric("Location", resolved_name)
            c2.metric("Temp", f"{float(live_weather.get('Temperature', 0)):.1f}°")
            c3.metric("Humidity", f"{float(live_weather.get('Humidity', 0)):.1f}%")
            st.caption(f"Wind: {float(live_weather.get('WindSpeed', 0)):.2f} m/s")

    st.divider()
    st.subheader("Preferences")
    cost_per_kwh = st.slider("Tariff ($/kWh)", 0.05, 0.50, 0.15, 0.01)
    compact_mode = st.toggle("Compact Layout", value=False, help="Use tighter spacing.")
    show_explanations = st.toggle("Show Smart Explanations", value=False, help="Show lightweight plain-language summaries across tabs.")
    
    st.divider()
    
    with st.expander("Assistant", expanded=False):
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_query = st.text_input("Question", placeholder="Why is my bill high?", key="chat_input")
        if st.button("Ask", key="chat_btn") and user_query:
            bot = EnergyChatbot(df, current_cost_kwh=cost_per_kwh)
            response = bot.get_response(user_query)
            st.session_state.chat_history.insert(0, {"user": user_query, "bot": response})

        if st.session_state.chat_history:
            for chat in st.session_state.chat_history[:2]:
                st.markdown(f"**You:** {chat['user']}")
                st.markdown(f"**Lumina:** {chat['bot']}")
                st.divider()
    
if compact_mode:
    st.markdown("""
    <style>
        .metric-card {
            padding: 14px 12px !important;
        }
        .metric-title {
            font-size: 0.72rem !important;
            margin-bottom: 6px !important;
        }
        .metric-value {
            font-size: 1.65rem !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 13px !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
@media (max-width: 900px) {
    .metric-card {
        padding: 12px 10px !important;
    }
    .metric-title {
        font-size: 0.68rem !important;
    }
    .metric-value {
        font-size: 1.4rem !important;
    }
    .hero-title {
        font-size: 1.7rem !important;
    }
    .hero-sub {
        font-size: 0.95rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

if st.session_state.get("quick_demo_mode", False) and not st.session_state.get("demo_bootstrapped", False):
    if 'live_history' not in st.session_state or st.session_state.live_history.empty:
        st.session_state.live_history = seed_demo_live_history(120)
    if st.session_state.get("location"):
        try:
            demo_weather = get_weather_forecast(st.session_state.location, hours=24)
            demo_preds = predictor.predict_forecast(pd.Timestamp.now().ceil('H'), demo_weather, df)
            st.session_state.forecast_df = pd.DataFrame(demo_preds)
        except Exception:
            pass
    st.session_state.demo_bootstrapped = True

if st.session_state.get("quick_demo_mode", False):
    st.info("Quick Demo Mode is ON: sample location, live telemetry history, and forecast are preloaded.")


# --- 4. Main Dashboard Header ---
st.markdown(
    f"""
    <div class='hero-title'>Energy Dashboard</div>
    <div class='hero-sub'>Overview for <b>{datetime.datetime.now().strftime('%B %Y')}</b> • AI Forecast + Live Telemetry</div>
    """,
    unsafe_allow_html=True
)


# Date logic
# Allow selecting up to TODAY even if data is old, so users can see "Live" or "Forecast" context
min_date = df.index.min().date()
max_date = max(df.index.max().date(), datetime.date.today())

# Date Filter
col_date1, col_date2 = st.columns([1, 3])
with col_date1:
    date_range = st.date_input(
        "Select Timeframe",
        value=(min_date, min(max_date, min_date + datetime.timedelta(days=365))), # Default to First Year (e.g. 2020)
        min_value=min_date,
        max_value=max_date
    )

# Filter
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df.index.date >= start_date) & (df.index.date <= end_date)
    filtered_df = df.loc[mask]
    
    # If filtered range includes Today and df has no data for today, 
    # we should ideally append session_state.live_history if we want it on the chart.
    # For now, just ensuring the filter works is Step 1.
    if 'live_history' in st.session_state and not st.session_state.live_history.empty:
        # Convert Live History to match Main DF structure for visualization
        live_data = st.session_state.live_history.copy()
        live_data = live_data.rename(columns={'Time': 'Timestamp', 'Power': 'energy_kwh'}) # Map Power to Energy for visual continuity
        live_data.set_index('Timestamp', inplace=True)
        
        # Filter live data to match selected range if needed (though usually it IS today)
        live_mask = (live_data.index.date >= start_date) & (live_data.index.date <= end_date)
        live_data_filtered = live_data.loc[live_mask]
        
        if not live_data_filtered.empty:
            # Convert kW (Power) to kWh (Energy)
            # Assumption: Live data is approx 1 second interval.
            # Energy (kWh) = Power (kW) * (1/3600) hours
            live_data_filtered['energy_kwh'] = live_data_filtered['energy_kwh'] / 3600.0
            
            # Concatenate for the chart
            filtered_df = pd.concat([filtered_df, live_data_filtered])

else:
    filtered_df = df

if filtered_df.empty:
    st.info("No historical data available for this range. Start the Live Monitor to see real-time data for Today.")

# Metrics Calculation
total_energy = filtered_df['energy_kwh'].sum() if not filtered_df.empty else 0
avg_daily = total_energy / (len(filtered_df)/24) if len(filtered_df) > 24 else total_energy
total_cost = total_energy * cost_per_kwh

# --- 5. Custom Metric Cards (Professional) ---
col1, col2, col3, col4 = st.columns(4)

def card(col, title, value, suffix=""):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}<span class="metric-suffix">{suffix}</span></div>
        </div>
        """, unsafe_allow_html=True)

card(col1, "Total Consumption", f"{total_energy:,.1f}", "kWh")
card(col2, "Est. Cost", f"${total_cost:,.2f}", "")
card(col3, "Carbon Footprint", f"{total_energy * 0.82:,.0f}", "kg")
card(col4, "Daily Average", f"{avg_daily:.1f}", "kWh")

st.write("") # Spacer

# --- 6. Tabs Content ---
tab_dashboard, tab_history, tab_prediction, tab_alerts, tab_live, tab_solar, tab_advanced = st.tabs(
    ["Dashboard", "History", "Forecast", "Alerts", "Live", "Solar", "Advanced"]
)

if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = pd.DataFrame()
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'alert_acknowledged' not in st.session_state:
    st.session_state.alert_acknowledged = False

forecast_overlay = st.session_state.forecast_df.copy() if not st.session_state.forecast_df.empty else pd.DataFrame()
current_kwh = float(filtered_df['energy_kwh'].iloc[-1]) if not filtered_df.empty else 0.0
current_cost_hour = current_kwh * cost_per_kwh
status_text = "Normal"
if not filtered_df.empty:
    if current_kwh > filtered_df['energy_kwh'].quantile(0.9):
        status_text = "High Usage"
    elif current_kwh > filtered_df['energy_kwh'].quantile(0.7):
        status_text = "Elevated"

with tab_dashboard:
    st.subheader("Dashboard")
    show_forecast_overlay = st.toggle("Show forecast overlay", value=True, key="dashboard_show_forecast")
    dashboard_df = apply_quick_range(df, "24H")

    today_start = pd.Timestamp.now().normalize()
    yesterday_start = today_start - pd.Timedelta(days=1)
    today_data = df[df.index >= today_start] if not df.empty else pd.DataFrame()
    yesterday_data = df[(df.index >= yesterday_start) & (df.index < today_start)] if not df.empty else pd.DataFrame()
    today_total = float(today_data['energy_kwh'].sum()) if not today_data.empty else 0.0
    yesterday_total = float(yesterday_data['energy_kwh'].sum()) if not yesterday_data.empty else 0.0
    today_delta = (today_total - yesterday_total) if yesterday_total > 0 else None
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Usage", f"{current_kwh:.3f} kWh")
    c2.metric("Today Total", f"{today_total:.2f} kWh", delta=f"{today_delta:+.2f} vs yesterday" if today_delta is not None else None)
    c3.metric("Estimated Cost Today", f"${today_total * cost_per_kwh:.2f}")

    if not dashboard_df.empty:
        plot_df = dashboard_df[['energy_kwh']].copy().sort_index().reset_index().rename(columns={'index': 'Timestamp'})
        actual = alt.Chart(plot_df).mark_area(
            line={'color': '#3b82f6'},
            color=alt.Gradient(
                gradient='linear',
                stops=[
                    alt.GradientStop(color='#3b82f6', offset=0),
                    alt.GradientStop(color='rgba(59, 130, 246, 0.12)', offset=1)
                ],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X('Timestamp:T', title='Time'),
            y=alt.Y('energy_kwh:Q', title='Energy (kWh)'),
            tooltip=[alt.Tooltip('Timestamp:T', format='%d %b %H:%M'), alt.Tooltip('energy_kwh:Q', format='.3f')]
        )
        layers = actual
        if show_forecast_overlay and not forecast_overlay.empty:
            fc = forecast_overlay.copy()
            fc['time'] = pd.to_datetime(fc['time'], errors='coerce')
            fc = fc.dropna(subset=['time', 'kwh'])
            forecast_line = alt.Chart(fc).mark_line(color='#f59e0b', strokeDash=[5, 4], strokeWidth=2.2).encode(
                x=alt.X('time:T'),
                y=alt.Y('kwh:Q'),
                tooltip=[alt.Tooltip('time:T', format='%d %b %H:%M'), alt.Tooltip('kwh:Q', format='.3f')]
            )
            layers = actual + forecast_line
        st.altair_chart(layers.interactive().properties(height=340), use_container_width=True)
    else:
        st.info("No data available for the selected dashboard window.")

    st.markdown("#### Next Best Action")
    recommendations = generate_action_recommendations(
        current_kwh,
        forecast_overlay if show_forecast_overlay else pd.DataFrame()
    )
    st.write(f"- {recommendations[0]}")

with tab_history:
    st.subheader("Historical Trends")
    st.caption("Choose a range to inspect past usage patterns and hourly intensity.")
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        quick_range = st.radio("Range", ["24H", "7D", "30D", "Custom"], horizontal=True, key="history_range")
    with c2:
        compare_previous = st.toggle("Compare with previous period", value=False, key="history_compare")
    with c3:
        show_anomaly_markers = st.toggle("Show anomaly markers", value=True, key="history_anomalies")

    custom_history_range = None
    if quick_range == "Custom":
        custom_history_range = st.date_input(
            "Custom Range",
            value=(min_date, min(max_date, min_date + datetime.timedelta(days=14))),
            min_value=min_date,
            max_value=max_date,
            key="history_custom_range"
        )

    history_df = apply_quick_range(df, quick_range, custom_history_range)
    if history_df.empty:
        st.warning("No data in selected range.")
    else:
        # Keep old visual behavior: high-resolution area chart for smaller windows,
        # daily aggregate for very large windows.
        if len(history_df) < 2000:
            chart_data = history_df[['energy_kwh']].copy().sort_index().reset_index().rename(columns={'index': 'Timestamp'})
            tooltip_format = '%d %b %H:%M:%S'
            threshold = history_df['energy_kwh'].mean() + (1.5 * history_df['energy_kwh'].std())
            resample_freq = None
        else:
            chart_data = history_df['energy_kwh'].resample('D').sum().reset_index()
            chart_data.columns = ['Timestamp', 'energy_kwh']
            tooltip_format = '%d %b'
            threshold = chart_data['energy_kwh'].mean() + chart_data['energy_kwh'].std()
            resample_freq = 'D'

        base = alt.Chart(chart_data).encode(
            x=alt.X('Timestamp:T', title='Time', axis=alt.Axis(grid=False)),
            y=alt.Y('energy_kwh:Q', title='Energy (kWh)'),
            tooltip=[alt.Tooltip('Timestamp:T', format=tooltip_format), alt.Tooltip('energy_kwh:Q', format='.3f')]
        )
        area = base.mark_area(
            line={'color': '#3b82f6'},
            color=alt.Gradient(
                gradient='linear',
                stops=[
                    alt.GradientStop(color='#3b82f6', offset=0),
                    alt.GradientStop(color='rgba(59, 130, 246, 0.10)', offset=1)
                ],
                x1=1, x2=1, y1=1, y2=0
            )
        )
        layers = area

        if show_anomaly_markers:
            points = base.mark_circle(size=60, opacity=0.85).encode(
                color=alt.condition(
                    alt.datum.energy_kwh > threshold,
                    alt.value('#ef4444'),
                    alt.value('transparent')
                )
            )
            layers = layers + points

        if compare_previous:
            duration = history_df.index.max() - history_df.index.min()
            prev_start = history_df.index.min() - duration
            prev_end = history_df.index.min()
            previous = df[(df.index >= prev_start) & (df.index < prev_end)].copy()
            if not previous.empty:
                if resample_freq == 'D':
                    previous = previous['energy_kwh'].resample('D').sum().reset_index()
                    previous.columns = ['Timestamp', 'energy_kwh']
                else:
                    previous = previous[['energy_kwh']].sort_index().reset_index().rename(columns={'index': 'Timestamp'})
                previous['Timestamp'] = previous['Timestamp'] + duration
                prev_line = alt.Chart(previous).mark_line(color='#94a3b8', strokeDash=[4, 4], strokeWidth=2).encode(
                    x='Timestamp:T',
                    y='energy_kwh:Q',
                    tooltip=[alt.Tooltip('Timestamp:T', format=tooltip_format), alt.Tooltip('energy_kwh:Q', format='.3f')]
                )
                layers = layers + prev_line

        st.altair_chart(
            layers.interactive().properties(height=360).configure_view(strokeWidth=0),
            use_container_width=True
        )

        st.markdown("#### Hourly Intensity Heatmap")
        heatmap_data = history_df.copy()
        heatmap_data['Hour'] = heatmap_data.index.hour
        heatmap_source = heatmap_data.groupby('Hour', as_index=False)['energy_kwh'].mean()
        heatmap = alt.Chart(heatmap_source).mark_rect().encode(
            x=alt.X('Hour:O', title='Hour of Day'),
            y=alt.Y('energy_kwh:Q', title='Avg kWh'),
            color=alt.Color('energy_kwh:Q', scale=alt.Scale(scheme='blues'), title='kWh'),
            tooltip=[alt.Tooltip('Hour:O'), alt.Tooltip('energy_kwh:Q', format='.3f')]
        ).properties(height=210)
        st.altair_chart(heatmap, use_container_width=True)

        peak = history_df['energy_kwh'].max()
        low = history_df['energy_kwh'].min()
        avg = history_df['energy_kwh'].mean()
        h1, h2, h3 = st.columns(3)
        h1.metric("Average", f"{avg:.3f} kWh")
        h2.metric("Peak", f"{peak:.3f} kWh")
        h3.metric("Lowest", f"{low:.3f} kWh")
        st.download_button(
            "Download Filtered History CSV",
            data=dataframe_to_csv_bytes(history_df.reset_index()),
            file_name=f"history_{quick_range.lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="download_history"
        )
        if show_explanations:
            peak_time = history_df['energy_kwh'].idxmax()
            st.caption(f"Peak was {peak:.3f} kWh near {peak_time.strftime('%d %b %I:%M %p')}.")

with tab_prediction:
    st.subheader("Forecast")
    st.caption("Create a forward forecast using current weather and recent usage history.")
    p1, p2, p3 = st.columns([1, 1, 2])
    with p1:
        horizon = st.selectbox("Horizon", [24, 48, 72, 168], index=0, format_func=lambda x: f"Next {x}h")
    with p2:
        temp_offset = st.slider("What-if temp offset (°C)", -5, 5, 0, 1)
    with p3:
        behavior = st.select_slider("Behavior profile", options=["Conservative", "Normal", "Heavy"], value="Normal")

    behavior_mult = {"Conservative": 0.92, "Normal": 1.0, "Heavy": 1.12}
    if st.button("Generate Forecast", key="btn_forecast_pro", type="primary"):
        if not city_name:
            st.warning("Location not detected. Please wait for GPS or enter a location.")
        else:
            with st.spinner("Running AI forecast..."):
                try:
                    forecast_weather = get_weather_forecast(city_name, hours=int(horizon))
                    for w in forecast_weather:
                        w['Temperature'] = float(w.get('Temperature', 25.0)) + temp_offset
                    now = pd.Timestamp.now()
                    predictions = predictor.predict_forecast(now.ceil('H'), forecast_weather, df)
                    pred_df = build_forecast_dataframe(predictions, df)
                    pred_df['kwh'] = pred_df['kwh'] * behavior_mult[behavior]
                    pred_df['min_kwh'] = pred_df['min_kwh'] * behavior_mult[behavior]
                    pred_df['max_kwh'] = pred_df['max_kwh'] * behavior_mult[behavior]
                    st.session_state.forecast_df = pred_df
                except Exception as e:
                    st.error(f"Forecast Error: {str(e)}")

    if 'forecast_df' in st.session_state and not st.session_state.forecast_df.empty:
        pred_df = st.session_state.forecast_df.copy()
        pred_df['time'] = pd.to_datetime(pred_df['time'])
        peak_row = pred_df.loc[pred_df['kwh'].idxmax()]
        peak_time = peak_row['time']
        peak_load = float(pred_df['kwh'].max())
        total_load = pred_df['kwh'].sum()
        pi1, pi2, pi3 = st.columns(3)
        pi1.metric("Projected Total", f"{total_load:.1f} kWh")
        pi2.metric("Peak Hour Energy", f"{peak_load:.2f} kWh")
        pi3.metric("Peak Hour", peak_time.strftime("%d %b %I:%M %p"))

        f_chart = alt.Chart(pred_df).mark_area(
            line={'color': '#8b5cf6'},
            color=alt.Gradient(
                gradient='linear',
                stops=[
                    alt.GradientStop(color='#8b5cf6', offset=0),
                    alt.GradientStop(color='rgba(139, 92, 246, 0.1)', offset=1)
                ],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X('time:T', title='Timeline', axis=alt.Axis(format='%H:%M')),
            y=alt.Y('kwh:Q', title='Predicted Consumption (kWh)'),
            tooltip=[alt.Tooltip('time:T', format='%d %b %H:%M'), alt.Tooltip('kwh:Q', format='.3f')]
        ).properties(height=360)

        peak_point = pd.DataFrame({'time': [peak_time], 'kwh': [peak_load]})
        peak_layer = alt.Chart(peak_point).mark_point(color='#f43f5e', size=100, filled=True)
        st.altair_chart((f_chart + peak_layer).interactive(), use_container_width=True)
        st.info(
            f"Expected peak around {peak_time.strftime('%I:%M %p')}. "
            "Forecast is based on recent consumption behavior and weather conditions."
        )
        st.download_button(
            "Download Forecast CSV",
            data=dataframe_to_csv_bytes(pred_df),
            file_name=f"forecast_{int(horizon)}h_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="download_forecast"
        )
    else:
        st.info("Generate a forecast to view confidence bands and recommendations.")

with tab_alerts:
    st.subheader("Alerts and Recommended Actions")
    alert_df = build_alert_table(filtered_df, st.session_state.get('forecast_df', pd.DataFrame()), st.session_state.get('alerts', []))
    if st.button("Acknowledge all alerts", key="ack_alerts"):
        st.session_state.alert_acknowledged = True
    if alert_df.empty:
        st.success("No active alerts. Usage is within expected patterns.")
    else:
        if st.session_state.alert_acknowledged:
            alert_df['status'] = 'Acknowledged'
        summary = alert_df['severity'].value_counts().to_dict()
        a1, a2, a3 = st.columns(3)
        a1.metric("Open", int((alert_df['status'] == 'Open').sum()))
        a2.metric("High", int(summary.get('High', 0)))
        a3.metric("Medium", int(summary.get('Medium', 0)))
        st.dataframe(alert_df, use_container_width=True, hide_index=True)
        st.caption("Each alert includes impact and a next action to reduce cost or avoid peak load.")

with tab_live:
    st.subheader("Smart Home Monitor")
    
    if 'virtual_meter' not in st.session_state or 'heater_on' not in st.session_state.virtual_meter.state:
        st.session_state.virtual_meter = VirtualSmartMeter()
        
    if 'anomaly_detector' not in st.session_state:
        st.session_state.anomaly_detector = AnomalyDetector()
        st.session_state.alerts = [] # List to store string alerts
        
    # --- NEW DEVICE CONTROL GRID ---
    dev_c1, dev_c2, dev_c3 = st.columns(3)
    
    # Device State Helpers
    state = st.session_state.virtual_meter.state
    
    def device_widget(col, name, key, power, icon):
        is_on = state.get(key, False)
        status_class = "device-active" if is_on else "device-inactive"
        dot_class = "dot-green" if is_on else "dot-gray"
        status_text = "Running" if is_on else "Standby"
        
        with col:
            # Custom container styling via HTML is cleaner for 'cards'
            st.markdown(f"""
            <div class="device-card {status_class}">
                <div style="font-size: 24px; margin-bottom: 5px;">{icon}</div>
                <div style="font-weight: 600; color: #e5e7eb;">{name}</div>
                <div style="font-size: 12px; color: #9ca3af; margin: 5px 0;">
                    <span class="status-dot {dot_class}"></span>{status_text}
                </div>
                <div style="font-size: 14px; color: #fff; font-weight: bold;">{power} kW</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"{'Stop' if is_on else 'Start'} {name}", key=f"btn_{key}"):
                st.session_state.virtual_meter.toggle_device(key)
                st.session_state.live_monitor_active = True
                st.rerun()

    device_widget(dev_c1, "HVAC System", "ac_on", "1.5", "❄️")
    device_widget(dev_c2, "Water Heater", "heater_on", "2.0", "🔥")
    device_widget(dev_c3, "EV Charger", "ev_charger_on", "3.0", "🚗")
    
    st.divider()
    # --- REAL-TIME CHARTING ---
    chart_placeholder = st.empty()
    metric_placeholder = st.empty()
    perf_c1, perf_c2, perf_c3 = st.columns(3)
    with perf_c1:
        refresh_interval = st.slider(
            "Refresh Interval (s)",
            0.2,
            2.0,
            0.5,
            0.1,
            key="refresh_interval",
            help="How often each live sample is generated. Lower = faster updates and higher CPU usage."
        )
    with perf_c2:
        samples_per_burst = st.slider(
            "Samples per Burst",
            5,
            60,
            20,
            5,
            key="samples_per_burst",
            help="How many readings to process before rerendering. Higher = smoother streams, but longer blocking bursts."
        )
    with perf_c3:
        max_points = st.slider(
            "Chart Points",
            30,
            300,
            120,
            10,
            key="max_points",
            help="Maximum points kept in the live chart window. Higher = longer history, but heavier rendering."
        )
    
    if st.toggle("Enable Real-Time Telemetry", key="live_monitor_active", value=False):
        if 'live_history' not in st.session_state:
            st.session_state.live_history = pd.DataFrame(columns=['Time', 'Power'])
            
        for _ in range(int(samples_per_burst)):
            data = st.session_state.virtual_meter.generate_reading()
            try:
                row = {k: data.get(k) for k in LIVE_LOG_COLUMNS}
                pd.DataFrame([row], columns=LIVE_LOG_COLUMNS).to_csv(
                    LIVE_LOG_PATH,
                    mode='a',
                    header=not os.path.exists(LIVE_LOG_PATH),
                    index=False
                )
            except Exception:
                pass
            
            # Append to session history
            now = datetime.datetime.now()
            new_row = pd.DataFrame({'Time': [now], 'Power': [data['power_kw']]})
            st.session_state.live_history = pd.concat([st.session_state.live_history, new_row]).tail(max_points)
            
            # Update Chart every iteration is heavy, but needed for "Live" feel. 
            # Optimization: Create chart object once and update data? Streamlit makes this hard.
            # We keep it but ensure it's efficient.
            
            with chart_placeholder.container():
                # Simplified Chart for performance - faster rendering
                live_chart = alt.Chart(st.session_state.live_history).mark_area(
                    line={'color':'#10b981'},
                    color=alt.Gradient(
                        gradient='linear',
                        stops=[alt.GradientStop(color='#10b981', offset=0),
                               alt.GradientStop(color='rgba(16, 185, 129, 0.1)', offset=1)],
                        x1=1, x2=1, y1=1, y2=0
                    )
                ).encode(
                    x=alt.X('Time:T', axis=alt.Axis(format='%H:%M:%S', title=None)), # Remove title for speed/cleanliness
                    y=alt.Y('Power:Q', scale=alt.Scale(domain=[0, 8]), title=None),
                ).properties(height=250, title="Live Power (kW)")
                
                st.altair_chart(live_chart, use_container_width=True)
                
            with metric_placeholder.container():
                # Metric Strip
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Power", f"{data['power_kw']:.2f} kW")
                m2.metric("Voltage", f"{data['voltage_v']} V")
                m3.metric("Current", f"{data['current_a']} A")
                m4.metric("Freq", f"{data['frequency_hz']} Hz")
                
                # Check for Anomalies
                alert = st.session_state.anomaly_detector.check_anamoly(now, data['power_kw'])
                if alert:
                    st.toast(alert, icon="🚨")
                    st.session_state.alerts.insert(0, f"{now.strftime('%H:%M:%S')}: {alert}")
                    
                # Show Recent Alerts (Limit to 3 for performance)
                if st.session_state.alerts:
                    with st.expander("🚨 Recent Alerts", expanded=True):
                        for a in st.session_state.alerts[:3]: 
                            st.write(a)
            
            time.sleep(refresh_interval)
        st.rerun()

    st.divider()
    st.markdown("#### Export Data")
    live_export_df = pd.DataFrame()
    if os.path.exists(LIVE_LOG_PATH):
        try:
            live_export_df = pd.read_csv(LIVE_LOG_PATH, on_bad_lines='skip', engine='python')
        except Exception:
            live_export_df = pd.DataFrame()
    if not live_export_df.empty:
        st.download_button(
            "Download Live Logs CSV",
            data=dataframe_to_csv_bytes(live_export_df),
            file_name=f"live_data_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="download_live_logs"
        )
    elif 'live_history' in st.session_state and not st.session_state.live_history.empty:
        session_export = st.session_state.live_history.copy()
        st.download_button(
            "Download Session Telemetry CSV",
            data=dataframe_to_csv_bytes(session_export),
            file_name=f"live_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="download_live_session"
        )
    else:
        st.caption("No live telemetry data available yet.")
    if show_explanations:
        if 'live_history' in st.session_state and not st.session_state.live_history.empty:
            lh = st.session_state.live_history.copy()
            avg_p = float(lh['Power'].mean())
            max_p = float(lh['Power'].max())
            min_p = float(lh['Power'].min())
            st.caption(
                f"Insight: Live load averages {avg_p:.2f} kW with a range of {min_p:.2f} to {max_p:.2f} kW in the current window."
            )
        else:
            st.caption("Insight: Start telemetry to show a live trend summary here.")

with tab_advanced:
    st.subheader("Scenario Simulator")
    if 'last_sim_result' not in st.session_state:
        st.session_state.last_sim_result = None

    if city_name:
        live_sim_weather = get_live_weather(city_name)
        default_temp = float(live_sim_weather.get('Temperature', 25.0))
        default_hum = int(live_sim_weather.get('Humidity', 50))
    else:
        default_temp = 25.0
        default_hum = 50

    sim_col1, sim_col2 = st.columns([1, 1])
    with sim_col1:
        sim_temp = st.slider("Outside Temperature (°C)", 0.0, 50.0, default_temp, key="sim_temp")
        sim_hum = st.slider("Humidity (%)", 0, 100, default_hum, key="sim_hum")
    with sim_col2:
        sim_hour = st.slider("Hour of Day (0-23)", 0, 23, 18, key="sim_hour")
        sim_occupancy = st.select_slider("Occupancy Level", options=["Low", "Medium", "High"], value="Medium", key="sim_occupancy")

    if st.button("Run Scenario Analysis", key="run_scenario", type="primary"):
        sim_time = datetime.datetime.now().replace(hour=sim_hour, minute=0, second=0)
        occ_mult = {"Low": 0.8, "Medium": 1.0, "High": 1.3}
        weather = {
            'Temperature': sim_temp,
            'Humidity': sim_hum,
            'Pressure': 1013,
            'WindSpeed': 5,
            'Voltage': 230.0,
            'Current': 2.0,
            'Frequency': 50.0
        }
        base_pred = predictor.predict_one(sim_time, weather, df)
        final_pred = base_pred * occ_mult[sim_occupancy]
        st.session_state.last_sim_result = {
            "final_pred": float(final_pred),
            "sim_time": sim_time,
            "sim_temp": float(sim_temp),
            "sim_hum": int(sim_hum),
            "sim_occupancy": sim_occupancy,
        }
    sim_result = st.session_state.get("last_sim_result")
    if sim_result:
        r1, r2, r3 = st.columns(3)
        r1.metric("Expected Hourly Energy", f"{sim_result['final_pred']:.2f} kWh")
        r2.metric("Estimated Cost", f"${sim_result['final_pred'] * cost_per_kwh:.3f}")
        r3.metric("Scenario Time", sim_result['sim_time'].strftime("%I:%M %p"))
        if show_explanations:
            st.caption(
                f"Scenario predicts {sim_result['final_pred']:.2f} kWh at "
                f"{sim_result['sim_temp']:.1f}°C / {sim_result['sim_hum']}% humidity "
                f"with {sim_result['sim_occupancy']} occupancy."
            )

# --- 7. Solar Potential Tab ---
with tab_solar:
    st.subheader("☀️ Solar Power Estimator")
    
    col_solar1, col_solar2 = st.columns([1, 2])
    
    with col_solar1:
        st.info("Simulate how a Rooftop Solar System would perform right now based on current weather.")
        
        st.markdown("#### Configuration")
        system_size = st.slider("System Size (kW)", 1.0, 10.0, 5.0, 0.5, help="Typical home creates valid 3-6kW.")
        
        panel_type = st.selectbox(
            "Panel Technology", 
            ["Standard (Poly/Mono)", "Premium (High Efficiency)", "Economy (Thin Film)"],
            index=0,
            help="Premium panels work better in low light and heat."
        )
        
        eff_map = {
            "Standard (Poly/Mono)": 0.18, 
            "Premium (High Efficiency)": 0.21, 
            "Economy (Thin Film)": 0.14
        }
        
        # Instantiate Simulator
        solar_sim = SolarSimulator(system_size_kw=system_size, efficiency=eff_map[panel_type])
        
        # Real-time Calculation
        # Ensure we have live weather (might be waiting for async location)
        if 'live_weather' not in locals():
             live_weather = get_live_weather("London") # Fallback to prevent crash if not defined in scope
             
        current_solar_kw = solar_sim.calculate_instant_power(live_weather)
        
        # Gauge Visualization (Simple Metric for now)
        st.metric(
            label="Current Generation Potential",
            value=f"{current_solar_kw} kW",
            delta=f"{live_weather.get('UV', 0)} UV Index",
            delta_color="normal"
        )
        
        if current_solar_kw > 0.1:
            st.success(f"Converting Sunlight! ({live_weather.get('Cloud', 0)}% Cloud Cover)")
        else:
            st.warning("Low/No Production (Night or Heavy Cloud)")

    with col_solar2:
        st.markdown("### Daily Estimation")
        
        predicted_daily_kwh = solar_sim.estimate_daily_production(live_weather)
        
        # Compare with average daily usage
        coverage = (predicted_daily_kwh / avg_daily * 100) if avg_daily > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Est. Daily Production", f"{predicted_daily_kwh} kWh")
        c2.metric("Grid Offset", f"{min(100, coverage):.1f}%")
        c3.metric("Est. Daily Savings", f"${predicted_daily_kwh * cost_per_kwh:.2f}")
        
        st.progress(min(1.0, coverage/100))
        
        st.divider()
        st.markdown("### Net Metering Simulation (Monthly)")
        
        # Calculate ROI
        roi_data = solar_sim.calculate_roi(monthly_bill=total_cost, current_tariff_kwh=cost_per_kwh, system_cost=system_size * 1000) # Approx $1000/kW
        
        roi_c1, roi_c2, roi_c3 = st.columns(3)
        roi_c1.metric("New Monthly Bill", f"${roi_data['new_bill']:.2f}", delta=f"-${roi_data['monthly_savings']:.2f}")
        roi_c2.metric("ROI Breakeven", f"{roi_data['breakeven_years']} Years")
        roi_c3.metric("CO2 Saved/Month", f"{roi_data['monthly_production_kwh'] * 0.82:.0f} kg")

    if show_explanations:
        st.caption(
            f"Insight: Solar can generate about {current_solar_kw:.2f} kW now and {predicted_daily_kwh:.1f} kWh/day, "
            f"offsetting ~{min(100, coverage):.1f}% of usage with ~${predicted_daily_kwh * cost_per_kwh:.2f}/day savings."
        )
        with st.expander("More details", expanded=False):
            st.write(f"Estimated breakeven period is around {roi_data['breakeven_years']} years.")

    st.divider()
    
st.caption("© 2026 Lumina. All rights reserved.")

