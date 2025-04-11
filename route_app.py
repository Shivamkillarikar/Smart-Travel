# --- Imports ---
import streamlit as st
import folium
import joblib
import pandas as pd
import numpy as np
import requests
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium
import google.generativeai as genai

# --- Streamlit Setup ---
st.set_page_config(page_title="🚗 Smart Travel", layout="centered")
st.title("🚗 Smart Route Recommender")
st.write("Get the best route based on weather and traffic conditions!")

# --- Load model and data ---
@st.cache_resource
def load_model():
    return joblib.load("final.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("india (1).csv")  # Make sure the filename matches the uploaded file

model = load_model()
df = load_data()

# --- Weather API ---
WEATHER_API = "1afdd88fb14c4b25e2e9192d7eabbba9"
WEATHER_URL = "http://api.openweathermap.org/data/2.5/weather"

# 🔐 Gemini Setup
genai.configure(api_key="AIzaSyAMty2R33y3UHeC7jVLAll4YM_GxNK0gnc")

def get_ai_recommendations(temp, wind_speed, rain):
    prompt = (
        f"Given the following weather conditions:\n"
        f"- Temperature: {temp}°C\n"
        f"- Wind Speed: {wind_speed} km/h\n"
        f"- Rainfall: {rain} mm\n\n"
        f"Give 3 short and friendly travel safety tips. Use emojis to make it engaging."
    )
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")
    response = model.generate_content(prompt)
    return response.text.strip().split("\n")

def get_weather_impact(city):
    try:
        params = {"q": city, "appid": WEATHER_API, "units": "metric"}
        res = requests.get(WEATHER_URL, params=params)
        data = res.json()
        temp = data["main"]["temp"]
        wind = data["wind"]["speed"]
        rain = data.get("rain", {}).get("1h", 0)
        impact = 1 + ((temp - 25) * 0.002) + (wind * 0.005) + (rain * 0.01)
        return max(0.8, min(1.2, impact)), temp, wind, rain
    except:
        return 1.0, None, None, None

# --- Session State ---
if "show_map" not in st.session_state:
    st.session_state.show_map = False

with st.form("route_form"):
    source_cities = sorted(df['Start'].unique())
    destination_cities = sorted(df['End'].unique())

    start = st.selectbox("Select Start Location", source_cities)
    filtered_destinations = [city for city in destination_cities if city != start]
    end = st.selectbox("Select Destination", filtered_destinations)

    traffic = st.slider("Traffic Level (1.0 - 1.5)", 1.0, 1.5, 1.2)
    base_time = st.number_input("Base Travel Time (in minutes)", min_value=1.0, value=60.0)
    submitted = st.form_submit_button("Get Route")

if submitted:
    st.session_state.show_map = True
    st.session_state.start = start
    st.session_state.end = end
    st.session_state.traffic = traffic
    st.session_state.base_time = base_time

if st.session_state.show_map:
    geolocator = Nominatim(user_agent="shivamkillarikar007@gmail.com")


    try:
        start_loc = geolocator.geocode(st.session_state.start)
        end_loc = geolocator.geocode(st.session_state.end)

        if not start_loc or not end_loc:
            st.error("❌ Could not locate one of the locations. Try different names.")
        else:
            with st.spinner("🔍 Fetching route and weather..."):
                route = df[(df['Start'] == st.session_state.start) & (df['End'] == st.session_state.end)]

                if route.empty:
                    st.warning("❌ No route found in dataset.")
                else:
                    route = route.iloc[0]
                    checkpoints = route["Checkpoints"].split(', ')
                    weather_impacts = []

                    # Display map
                    m = folium.Map(location=[(start_loc.latitude + end_loc.latitude)/2,
                                             (start_loc.longitude + end_loc.longitude)/2],
                                   zoom_start=7)

                    folium.Marker([start_loc.latitude, start_loc.longitude],
                                  tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
                    folium.Marker([end_loc.latitude, end_loc.longitude],
                                  tooltip="Destination", icon=folium.Icon(color="red")).add_to(m)
                    folium.PolyLine([[start_loc.latitude, start_loc.longitude],
                                     [end_loc.latitude, end_loc.longitude]],
                                    color="blue").add_to(m)

                    # Weather for all checkpoints
                    st.subheader("🌦️ Weather Impact + AI Travel Tips")
                    for loc in [st.session_state.start] + checkpoints + [st.session_state.end]:
                        impact, temp, wind, rain = get_weather_impact(loc)
                        weather_impacts.append(impact)

                        st.markdown(f"### 📍 {loc}")
                        st.write(f"- **Impact:** `{impact:.2f}`")
                        st.write(f"- **Temp:** {temp}°C | **Wind:** {wind} km/h | **Rain:** {rain} mm")

                        if temp is not None:
                            with st.spinner("🧠 Gemini is writing tips..."):
                                tips = get_ai_recommendations(temp, wind, rain)
                                for tip in tips:
                                    st.write(f"💡 {tip}")

                    avg_impact = np.mean(weather_impacts)
                    predicted_time = model.predict([[route['Distance_km'],
                                                     st.session_state.traffic,
                                                     st.session_state.base_time,
                                                     avg_impact]])[0]
                    predicted_time = max(st.session_state.base_time, predicted_time)

                    # Show map and results
                    st_folium(m, width=700, height=500)
                    st.success(f"🛣️ Route: {route['Checkpoints']}")
                    st.info(f"📏 Distance: {route['Distance_km']} km")
                    st.success(f"⏳ Estimated Travel Time: `{predicted_time:.2f} minutes`")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
