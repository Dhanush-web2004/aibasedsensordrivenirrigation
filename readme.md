# 🌿 AquaMind – AI Smart Irrigation System

A production-ready AI-powered irrigation system combining an ESP32 soil sensor, Flask ML backend, Firebase Realtime Database, and a professional web dashboard.

---

## 📁 Project Structure

```
smart-irrigation/
├── frontend/
│   └── index.html          # Full SPA dashboard (HTML + Tailwind + JS)
├── backend/
│   ├── app.py              # Flask API server
│   └── requirements.txt    # Python dependencies
├── model/
│   ├── train_model.py      # ML training script (generates model.pkl)
│   └── artifacts/          # Generated after running train_model.py
│       ├── model.pkl
│       ├── crop_encoder.pkl
│       ├── weather_encoder.pkl
│       └── metadata.json
├── firmware/
│   └── smart_irrigation.ino  # ESP32 Arduino sketch
├── firebase.json             # Firebase Hosting config
├── render.yaml               # Render.com deployment config
└── README.md
```

---

## ⚙️ Setup Guide

### 1. Train the ML Model

```bash
cd model
pip install scikit-learn pandas numpy joblib
python train_model.py
# → Creates artifacts/ folder with model.pkl
```

Model accuracy: **88.4%** on held-out test set.

### 2. Configure Environment Variables

Create `backend/.env`:

```env
OPENWEATHER_API_KEY=your_openweathermap_api_key
FIREBASE_DB_URL=https://YOUR_PROJECT-default-rtdb.firebaseio.com
FIREBASE_CREDENTIALS_PATH=/path/to/serviceAccount.json
FLASK_DEBUG=false
PORT=5000
```

Get your free OpenWeatherMap API key at: https://openweathermap.org/api

### 3. Run the Backend Locally

```bash
cd backend
pip install -r requirements.txt
python app.py
```

API will be available at `http://localhost:5000`.

### 4. Configure the Frontend

Edit `frontend/index.html`, find the `firebaseConfig` object near the bottom of the `<script>` block, and replace with your Firebase project config:

```js
const firebaseConfig = {
  apiKey:            "YOUR_API_KEY",
  authDomain:        "YOUR_PROJECT.firebaseapp.com",
  databaseURL:       "https://YOUR_PROJECT-default-rtdb.firebaseio.com",
  projectId:         "YOUR_PROJECT_ID",
  storageBucket:     "YOUR_PROJECT.appspot.com",
  messagingSenderId: "YOUR_SENDER_ID",
  appId:             "YOUR_APP_ID",
};
```

Open the Dashboard → Settings tab and set your Backend API URL.

### 5. Flash the ESP32

1. Install [Arduino IDE](https://www.arduino.cc/en/software)
2. Add ESP32 board support
3. Install libraries: `Firebase ESP32`, `DHT sensor library`, `ArduinoJson`
4. Edit `firmware/smart_irrigation.ino` with your WiFi/Firebase credentials
5. Flash to your ESP32

---

## 🚀 Deployment

### Backend – Render.com

1. Push repo to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Set **Build Command**: `pip install -r backend/requirements.txt && cd model && python train_model.py`
4. Set **Start Command**: `cd backend && gunicorn app:app --workers 2`
5. Add environment variables in Render dashboard
6. Copy the deployed URL into Dashboard → Settings

### Frontend – Firebase Hosting

```bash
npm install -g firebase-tools
firebase login
firebase init hosting   # select "frontend" as public directory
firebase deploy
```

---

## 🔌 API Reference

### `GET /weather?city=Hyderabad`

Returns live weather data from OpenWeatherMap.

**Response:**
```json
{
  "city": "Hyderabad",
  "temperature": 32.4,
  "humidity": 58,
  "description": "Partly cloudy",
  "forecast_rain_mm": 1.2,
  "weather_condition": "partly_cloudy",
  "daily_forecast": [...]
}
```

### `POST /predict`

```json
{
  "crop": "rice",
  "soil_moisture": 40,
  "city": "Hyderabad",
  "auto_mode": false
}
```

**Response:**
```json
{
  "irrigation_needed": 1,
  "confidence": 91.4,
  "reason": "Irrigation recommended: soil moisture (40%) is below threshold for rice (70%); little to no rain forecast.",
  "weather_summary": { ... },
  "pump_updated": false
}
```

### `GET /crops`

Returns list of supported crop types.

### `GET /health`

Health check endpoint with model accuracy.

---

## 🧠 ML Model

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Training samples | 2,160 (90 days × 24 readings/day) |
| Test accuracy | **88.4%** |
| Features | crop_type, soil_moisture, humidity, temperature, rainfall_forecast, weather_condition |
| Target | irrigation_needed (0 / 1) |

### Decision Logic

- If `soil_moisture < crop_threshold` → irrigation score increases
- If `rainfall_forecast ≥ 10mm` → score decreases (skip irrigation)
- If `temperature > 35°C` → score increases (higher evaporation)
- If `humidity < 30%` → score increases (faster moisture loss)

### Crop Thresholds (Soil Moisture %)

| Crop | Threshold |
|---|---|
| Rice | 70% |
| Sugarcane | 65% |
| Vegetables | 55% |
| Maize | 45% |
| Cotton | 40% |
| Wheat | 35% |

---

## 🔒 Security Notes

- Never commit API keys or service account files to git — use environment variables
- Add `.env` and `serviceAccount.json` to `.gitignore`
- Enable Firebase Authentication rules in your Firebase Console
- Use HTTPS for all API calls in production
- Set Firebase Database rules to restrict read/write to authenticated users

---

## 📱 Dashboard Features

- **Firebase Login** with email/password
- **Demo Mode** — works without any backend (simulated data)
- **AI Prediction Panel** — crop + city + moisture → irrigation decision
- **Live Weather Card** — OpenWeatherMap integration
- **7-Day Rain Forecast** Chart
- **Pump Manual Control** + AUTO mode toggle
- **Prediction History** log
- **Analytics Page** — moisture trends, decision distribution, model info
- **Settings Page** — configure backend URL dynamically
- **Firebase Realtime** — live sensor data via WebSocket

---

## 🏗 Architecture

```
ESP32 (soil + DHT22)
    │  WebSocket / HTTPS
    ▼
Firebase Realtime Database
    │  onValue listener
    ▼
Web Dashboard (Firebase Hosting)
    │  POST /predict
    ▼
Flask Backend (Render.com)
    │  fetch weather
    ├─► OpenWeatherMap API
    │  run ML model
    ├─► RandomForestClassifier (model.pkl)
    │  update pump (if AUTO)
    └─► Firebase Admin SDK → pump/status
```