"""
Live weather fetching with WeatherAPI.com (primary) and wttr.in (backup)
"""
import requests
import streamlit as st
from datetime import datetime
from geopy.geocoders import Nominatim
import os

def is_coordinate_query(value):
    if not isinstance(value, str) or "," not in value:
        return False
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        return False
    try:
        float(parts[0])
        float(parts[1])
        return True
    except (TypeError, ValueError):
        return False

def normalize_place_name(value):
    if value is None:
        return None
    cleaned = " ".join(str(value).strip().split()).strip(",")
    if not cleaned or is_coordinate_query(cleaned):
        return None
    return cleaned

def pick_major_name_from_address(address):
    if not isinstance(address, dict):
        return None
    # Keep this strict to avoid tiny/locality noise names.
    for key in ("city", "town", "municipality"):
        name = normalize_place_name(address.get(key))
        if name:
            return name
    return None

@st.cache_data(ttl=3600 * 12, show_spinner=False)
def reverse_geocode_bigdatacloud(lat, lon):
    """
    Reverse geocode via BigDataCloud (no API key) as fallback when Nominatim rate-limits.
    Returns a normalized city/town-style name when available.
    """
    try:
        response = requests.get(
            "https://api.bigdatacloud.net/data/reverse-geocode-client",
            params={
                "latitude": lat,
                "longitude": lon,
                "localityLanguage": "en",
            },
            timeout=6,
        )
        response.raise_for_status()
        data = response.json()
        for key in ("city", "locality", "principalSubdivision"):
            name = normalize_place_name(data.get(key))
            if name:
                return name
    except Exception:
        return None
    return None

@st.cache_data(ttl=3600 * 12, show_spinner=False)
def reverse_geocode_nominatim(lat, lon):
    """
    Reverse geocode with Nominatim. Cached to reduce external calls and avoid 509 bursts.
    """
    try:
        geolocator = Nominatim(user_agent="lumina_energy_dashboard", timeout=10)
        location = geolocator.reverse((lat, lon), zoom=10, language='en', addressdetails=True)
        if location and 'address' in location.raw:
            return pick_major_name_from_address(location.raw.get('address'))
    except Exception:
        return None
    return None

def get_major_city(lat, lon):
    """
    Resolve coordinates to the nearest major city using OpenStreetMap (Nominatim).
    Zoom=10 ensures we get 'city' level details, avoiding small villages.
    Returns: "City" or None
    """
    city = reverse_geocode_nominatim(lat, lon)
    if city:
        return city
    city = reverse_geocode_bigdatacloud(lat, lon)
    if city:
        return city
    return None

def get_weather_api_key():
    """Get WeatherAPI key from Streamlit secrets or environment."""
    try:
        if "WEATHER_API_KEY" in st.secrets:
            return st.secrets["WEATHER_API_KEY"]
        if "secrets" in st.secrets and "WEATHER_API_KEY" in st.secrets["secrets"]:
            return st.secrets["secrets"]["WEATHER_API_KEY"]
    except Exception:
        pass
    return os.getenv("WEATHER_API_KEY")

def detect_user_location():
    """
    Auto-detect user location based on IP address
    Returns: "City, Country" or None
    """
    print("DEBUG: Attempting to detect location via IP geolocation providers...")

    # Provider 1 (HTTPS): ipapi.co
    try:
        response = requests.get('https://ipapi.co/json/', timeout=5)
        print(f"DEBUG: ipapi.co status: {response.status_code}")
        data = response.json()
        city = str(data.get('city', '')).strip()
        country = str(data.get('country_name', '')).strip()
        timezone = str(data.get('timezone', 'UTC')).strip() or "UTC"
        if city and country:
            loc = f"{city}, {country}"
            print(f"DEBUG: ipapi.co detected: {loc}, timezone: {timezone}")
            return {'location': loc, 'timezone': timezone}
    except Exception as e:
        print(f"DEBUG: ipapi.co failed: {e}")

    # Provider 2 (fallback): ip-api.com (HTTP)
    try:
        response = requests.get('http://ip-api.com/json', timeout=5)
        print(f"DEBUG: ip-api.com status: {response.status_code}")
        data = response.json()
        if data.get('status') == 'success':
            city = str(data.get('city', '')).strip()
            country = str(data.get('country', '')).strip()
            timezone = str(data.get('timezone', 'UTC')).strip() or "UTC"
            if city and country:
                loc = f"{city}, {country}"
                print(f"DEBUG: ip-api.com detected: {loc}, timezone: {timezone}")
                return {'location': loc, 'timezone': timezone}
    except Exception as e:
        print(f"DEBUG: ip-api.com failed: {e}")

    return None


def fetch_weather_from_weatherapi(location):
    """
    Fetch current weather from WeatherAPI.com
    Returns dict with Temperature, Humidity, Pressure, WindSpeed, Voltage, Current, Frequency
    """
    api_key = get_weather_api_key()
    if not api_key:
        return None
    url = "https://api.weatherapi.com/v1/current.json"
    
    try:
        response = requests.get(
            url,
            params={'key': api_key, 'q': location, 'aqi': 'no'},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        
        current = data['current']

        return {
            'Temperature': current['temp_c'],
            'Humidity': current['humidity'],
            'Pressure': current['pressure_mb'],
            'WindSpeed': current['wind_kph'] / 3.6,  # Convert to m/s
            'Cloud': current.get('cloud', 0),    
            'UV': current.get('uv', 0.0),        
            'Voltage': 230.0,
            'Current': 2.0,
            'Frequency': 50.0,
            'source': 'WeatherAPI.com',
            'LocationName': data['location']['name'],
            'Timezone': data['location'].get('tz_id')
        }
    except Exception as e:
        # print(f"DEBUG: WeatherAPI Exception: {e}")
        return None

@st.cache_data(ttl=600, show_spinner=False)
def fetch_weather_from_wttr(location):
    """
    Fetch current weather from wttr.in (backup)
    """
    url = f"https://wttr.in/{location}"
    
    try:
        response = requests.get(
            url,
            params={'format': 'j1'},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        
        current = data['current_condition'][0]
        nearest_area = data.get('nearest_area', [{}])[0]
        area_name = nearest_area.get('areaName', [{}])[0].get('value', '')
        region_name = nearest_area.get('region', [{}])[0].get('value', '')
        country_name = nearest_area.get('country', [{}])[0].get('value', '')
        location_name = area_name or region_name or country_name or 'Unknown Location'
        return {
            'Temperature': float(current['temp_C']),
            'Humidity': float(current['humidity']),
            'Pressure': float(current['pressure']),
            'WindSpeed': float(current['windspeedKmph']) / 3.6,
            'Cloud': float(current.get('cloudcover', 0)),
            'UV': float(current.get('uvIndex', 0)),    
            'Voltage': 230.0,
            'Current': 2.0,
            'Frequency': 50.0,
            'source': 'wttr.in',
            'LocationName': location_name
        }
    except Exception as e:
        return None

@st.cache_data(ttl=3600*24) # Cache indefinitely/long time for locations
def resolve_coordinates(location_name):
    """
    Resolve a city name to Lat,Lon using Nominatim
    """
    try:
        geolocator = Nominatim(user_agent="smart_home_controller_resolution", timeout=10)
        loc = geolocator.geocode(location_name)
        if loc:
            return f"{loc.latitude},{loc.longitude}"
    except Exception:
        return None
    return None

def get_live_weather(location):
    """
    Get live weather with automatic fallback
    Primary: WeatherAPI.com
    Backup: wttr.in
    Fallback: Geocoding -> WeatherAPI
    """
    # 1. Try direct lookup (Primary)
    weather = fetch_weather_from_weatherapi(location)
    if weather:
        return weather
    
    # 2. Try Geocoding only when WeatherAPI key exists and location is textual.
    # Without API key this path only adds failing geocoder calls.
    if get_weather_api_key() and (not is_coordinate_query(str(location))):
        coords = resolve_coordinates(location)
        if coords:
            weather = fetch_weather_from_weatherapi(coords)
            if weather:
                return weather
    
    # 3. Try backup source (wttr.in) - often better at names
    wttr_query = location if is_coordinate_query(str(location)) else str(location).split(',')[0]
    weather = fetch_weather_from_wttr(wttr_query)
    if weather:
        return weather
    
    # 4. Return defaults if all fail
    fallback_name = str(location).strip() if location else 'Unknown'
    return {
        'Temperature': 25.0,
        'Humidity': 60.0,
        'Pressure': 1013.0,
        'WindSpeed': 2.0,
        'Cloud': 20.0, # Default
        'UV': 5.0,     # Default
        'Voltage': 230.0,
        'Current': 2.0,
        'Frequency': 50.0,
        'source': 'default',
        'LocationName': fallback_name
    }

@st.cache_data(ttl=3600) # Forecast doesn't change often
def get_weather_forecast(location, hours=24):
    """
    Get hourly weather forecast for next N hours
    Returns list of weather dicts
    """
    api_key = get_weather_api_key()
    if not api_key:
        base_weather = get_live_weather(location)
        return [{
            'Temperature': base_weather['Temperature'] + (i % 24 - 12) * 0.5,
            'Humidity': base_weather['Humidity'],
            'Pressure': base_weather['Pressure'],
            'WindSpeed': base_weather['WindSpeed'],
            'Voltage': 230.0,
            'Current': 2.0,
            'Frequency': 50.0
        } for i in range(hours)]
    url = "https://api.weatherapi.com/v1/forecast.json"
    
    try:
        # Request 2 days to ensure we cover the next 24 hours across midnight
        days = 2
        
        response = requests.get(
            url,
            params={'key': api_key, 'q': location, 'days': days, 'aqi': 'no'},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        current_epoch = datetime.now().timestamp()
        
        # Extract hourly data
        forecast = []
        for day in data['forecast']['forecastday']:
            for hour in day['hour']:
                # Skip past hours (allow 1 hour buffer for "current" hour)
                if hour['time_epoch'] < current_epoch - 3600:
                    continue
                    
                if len(forecast) >= hours:
                    break
                    
                forecast.append({
                    'Temperature': hour['temp_c'],
                    'Humidity': hour['humidity'],
                    'Pressure': hour['pressure_mb'],
                    'WindSpeed': hour['wind_kph'] / 3.6,
                    'Voltage': 230.0,
                    'Current': 2.0,
                    'Frequency': 50.0
                })
            if len(forecast) >= hours:
                break
        
        if not forecast:
            raise ValueError("No forecast data extracted from API response")
            
        return forecast[:hours]
    
    except Exception as e:
        print(f"Forecast failed: {e}")
        # Generate fallback forecast with slight variations
        base_weather = get_live_weather(location)
        forecast = []
        for i in range(hours):
            forecast.append({
                'Temperature': base_weather['Temperature'] + (i % 24 - 12) * 0.5,
                'Humidity': base_weather['Humidity'],
                'Pressure': base_weather['Pressure'],
                'WindSpeed': base_weather['WindSpeed'],
                'Voltage': 230.0,
                'Current': 2.0,
                'Frequency': 50.0
            })
        return forecast
