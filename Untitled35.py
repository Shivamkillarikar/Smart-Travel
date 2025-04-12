#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# OpenWeather API Config
API_KEY = "1afdd88fb14c4b25e2e9192d7eabbba9"  # Replace with your OpenWeather API Key
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

def get_weather_impact(city):
    """Fetch real-time weather data and compute weather impact factor"""
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        data = response.json()
        temp = data["main"]["temp"]
        wind_speed = data["wind"]["speed"]
        rain = data.get("rain", {}).get("1h", 0)  # Default to 0 if no rain data
        
        # Compute Weather Impact Factor
        weather_impact = 1 + ((temp - 25) * 0.005) + (wind_speed * 0.01) + (rain * 0.02) 
        return max(0.8, min(1.2, weather_impact)), temp, wind_speed, rain  # Keep within reasonable bounds
    
    return 1.0, None, None, None  # Default impact with None values for temp, wind, and rain

# Load Dataset
df = pd.read_csv(r"C:\Users\Srushti\Downloads\realistic_india_routes.csv")

# Prepare Data for ML Model
df['Weather_Impact'] = 1.0  # Default value, will be replaced when predicting
df['Adjusted_Time'] = df['Time_Minutes'] * df['Weather_Impact']

X = df[['Distance_km', 'Traffic', 'Base_Time', 'Weather_Impact']]
y = df['Adjusted_Time']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
print(f"Model MAE: {mean_absolute_error(y_test, y_pred):.2f} minutes")

# Save Model
joblib.dump(model, "final.pkl")

def predict_best_route(start, end, traffic, base_time):
    """Find the best route & predict travel time with real-time weather impact"""
    routes = df[(df['Start'] == start) & (df['End'] == end)]
    if routes.empty:
        print("No routes found.")
        return

    best_route = routes.loc[routes['Time_Minutes'].idxmin()]
    checkpoints = best_route['Checkpoints'].split(', ')
    
    # Get real-time weather impact for start, checkpoints, and end
    weather_impacts = []
    print("\n🌦️ Weather Impact for Locations:")
    
    for location in [start] + checkpoints + [end]:  # Include all in route
        impact, temp, wind, rain = get_weather_impact(location)
        weather_impacts.append(impact)
        print(f"📍 {location}: {impact:.2f} (Temp: {temp}°C, Wind: {wind} km/h, Rain: {rain} mm)")

    # Compute the final weather impact as an average across all locations
    avg_weather_impact = np.mean(weather_impacts)

    # Predict Travel Time
    model = joblib.load("final.pkl")
    predicted_time = model.predict([[best_route['Distance_km'], traffic, base_time, avg_weather_impact]])[0]
    # Adjust for traffic impact
    traffic_impact_factor = 1 + (traffic * 0.20)  # Increase weight if necessary
    predicted_time *= traffic_impact_factor  


    # Output
    print(f"\n🚀 Best Route from {start} to {end}:")
    print(f"🛣️ Path: {best_route['Checkpoints']}")
    print(f"📏 Distance: {best_route['Distance_km']} km")
    print(f"⏳ Predicted Travel Time: {predicted_time:.2f} minutes")

# User Input
start = input("Enter Start Location: ")
end = input("Enter Destination: ")
traffic = float(input("Enter Traffic Level (0-1): "))
base_time = float(input("Enter Base Travel Time: "))

predict_best_route(start, end, traffic, base_time)


# In[ ]:




