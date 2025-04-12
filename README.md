# Smart-Travel

# 🚗 Smart Travel Time Predictor with Weather & AI Assistant

An intelligent travel assistant that predicts the best route and travel time across Indian cities, factoring in **live weather conditions**, **traffic levels**, and providing **AI-generated safety tips**. It also visualizes the journey using interactive maps powered by **Folium**.

---

## 📸 Demo
![Screenshot (464)](https://github.com/user-attachments/assets/cb2c9a61-1c18-47c7-a792-41add3bb5ee3)
![Screenshot (465)](https://github.com/user-attachments/assets/f07a8df9-77c7-4fd2-a7ea-0811d1d8672e)
![Screenshot (466)](https://github.com/user-attachments/assets/a96ec973-0e7b-4e53-ba40-745cf1b3cd9d)
![Screenshot (467)](https://github.com/user-attachments/assets/3d6f695f-8a9a-46f4-8fe5-aefa561ad4b1)
![Screenshot (468)](https://github.com/user-attachments/assets/4f7ab456-2c75-4a43-908d-7f6cc36655aa)
![Screenshot (469)](https://github.com/user-attachments/assets/13b7f635-2be3-4f2f-bf88-6fd68924ae5f)





## 🌟 Features!



- 🧠 **AI-Powered Travel Tips** using Google Gemini API
- 🌦️ **Live Weather Integration** using OpenWeatherMap API
- 🚥 **Traffic-Aware Predictions** with machine learning
- 📊 **Random Forest Model** trained on real-world travel data
- 🗺️ **Interactive Route Map** with **Folium**
- 📈 Model Evaluation (MAE, MSE, RMSE, R²)
- 🔐 API Key-based secure configuration

---

## 📦 Tech Stack

- **Frontend**: Streamlit, Folium
- **Backend / ML**:
  - Pandas, NumPy
  - scikit-learn (Random Forest Regressor)
  - joblib (for model persistence)
- **APIs**:
  - [OpenWeatherMap](https://openweathermap.org/api)
  - [Google Gemini (Generative AI)](https://ai.google.dev/)
- **Visualization**: Folium for maps, Streamlit UI

---

## 📁 Folder Structure

```bash
.
├── india (1).csv               # Dataset
├── final.pkl                   # Trained ML model
├── app.py                      # Main Streamlit application
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
