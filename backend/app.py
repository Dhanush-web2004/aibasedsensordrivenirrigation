from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
import os
import joblib
import json
from datetime import datetime, timedelta

# -------------------------------
# APP SETUP
# -------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------
# API KEY
# -------------------------------
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "your api key")
print("API KEY USED:", OPENWEATHER_API_KEY)
# -------------------------------
# BASE URL
# -------------------------------
BASE_URL = "https://api.openweathermap.org/data/2.5"

# -------------------------------
# CROP MOISTURE THRESHOLDS (from train_model.py / paper)
# -------------------------------
CROP_MOISTURE_THRESHOLDS = {
    "rice":       70,
    "sugarcane":  65,
    "vegetables": 55,
    "maize":      45,
    "cotton":     40,
    "wheat":      35,
}

# Water demand levels for crop recommendation
CROP_WATER_DEMAND = {
    "rice":       "HIGH",
    "sugarcane":  "HIGH",
    "vegetables": "MEDIUM",
    "maize":      "MEDIUM",
    "wheat":      "LOW",
    "cotton":     "LOW",
}

# Suggested low-water alternatives
CROP_ALTERNATIVES = {
    "rice":       ["maize", "wheat", "cotton"],
    "sugarcane":  ["maize", "vegetables"],
    "vegetables": ["wheat", "cotton"],
    "maize":      ["wheat", "cotton"],
    "wheat":      [],
    "cotton":     [],
}

# Water saved estimate when switching
WATER_SAVINGS = {
    "HIGH":   "40-60% reduction possible",
    "MEDIUM": "20-35% reduction possible",
    "LOW":    "Already water-efficient",
}

# -------------------------------
# ANOMALY DETECTION HISTORY
# -------------------------------
_moisture_history = []
_MAX_HISTORY      = 10

# -------------------------------
# LOAD ML MODEL (if available)
# -------------------------------
_model       = None
_crop_enc    = None
_weather_enc = None
_metadata    = None

def _load_model():
    global _model, _crop_enc, _weather_enc, _metadata
    try:
        _model       = joblib.load("artifacts/model.pkl")
        _crop_enc    = joblib.load("artifacts/crop_encoder.pkl")
        _weather_enc = joblib.load("artifacts/weather_encoder.pkl")
        with open("artifacts/metadata.json") as f:
            _metadata = json.load(f)
        print("ML model loaded from artifacts/")
    except Exception as e:
        print(f"ML model not found ({e}). Using rule-based fallback.")

_load_model()

# -------------------------------
# WEATHER FUNCTIONS
# -------------------------------
def fetch_weather(city):
    if not OPENWEATHER_API_KEY:
        raise ValueError("API key missing")

    params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}

    res = requests.get(f"{BASE_URL}/weather", params=params, timeout=8).json()
    if "main" not in res:
        raise ValueError(f"City not found or invalid API key: {res.get('message','')}")

    fc    = requests.get(f"{BASE_URL}/forecast", params=params, timeout=8).json()
    items = fc.get("list", [])

    # Next 24 h rain
    rain_24h = sum(item.get("rain", {}).get("3h", 0) for item in items[:8])

    # 7-day daily forecast aggregation
    daily = {}
    for item in items:
        date = item["dt_txt"][:10]
        daily.setdefault(date, {"rain": 0.0, "temp": [], "desc": item["weather"][0]["description"]})
        daily[date]["rain"] += item.get("rain", {}).get("3h", 0)
        daily[date]["temp"].append(item["main"]["temp"])

    daily_forecast = [
        {
            "date":        d,
            "rain_mm":     round(v["rain"], 2),
            "temp_avg":    round(sum(v["temp"]) / len(v["temp"]), 1),
            "description": v["desc"],
        }
        for d, v in sorted(daily.items())
    ][:7]

    # Pad to 7 days if API returned fewer
    if daily_forecast:
        last_date = datetime.strptime(daily_forecast[-1]["date"], "%Y-%m-%d")
    else:
        last_date = datetime.today()
    while len(daily_forecast) < 7:
        last_date += timedelta(days=1)
        daily_forecast.append({
            "date":        last_date.strftime("%Y-%m-%d"),
            "rain_mm":     0.0,
            "temp_avg":    daily_forecast[-1]["temp_avg"] if daily_forecast else 25.0,
            "description": "clear sky",
        })

    # Map OWM description to our weather categories
    desc_lower = res["weather"][0]["description"].lower()
    if "thunderstorm" in desc_lower or "storm" in desc_lower:
        weather_cat = "stormy"
    elif "rain" in desc_lower or "drizzle" in desc_lower:
        weather_cat = "rainy"
    elif "cloud" in desc_lower and "few" not in desc_lower and "scattered" not in desc_lower:
        weather_cat = "cloudy"
    elif "cloud" in desc_lower:
        weather_cat = "partly_cloudy"
    else:
        weather_cat = "clear"

    return {
        "city":             res["name"],
        "temperature":      res["main"]["temp"],
        "humidity":         res["main"]["humidity"],
        "description":      res["weather"][0]["description"],
        "weather_category": weather_cat,
        "forecast_rain_mm": round(rain_24h, 2),
        "daily_forecast":   daily_forecast,
    }

# -------------------------------
# DECISION ENGINE
# Features: crop_type, soil_moisture, humidity, temperature,
#           rainfall_forecast, weather_condition  (as per paper)
# -------------------------------
def make_irrigation_decision(crop, soil_moisture, weather):
    threshold   = CROP_MOISTURE_THRESHOLDS.get(crop, 50)
    temperature = weather["temperature"]
    humidity    = weather["humidity"]
    rain_mm     = weather["forecast_rain_mm"]
    weather_cat = weather["weather_category"]

    # Feature contribution scores (SHAP-style)
    feature_scores = {}

    # 1. Soil moisture
    if soil_moisture < threshold * 0.5:
        feature_scores["soil_moisture"] = +3
    elif soil_moisture < threshold:
        feature_scores["soil_moisture"] = +2
    elif soil_moisture < threshold * 1.2:
        feature_scores["soil_moisture"] = 0
    else:
        feature_scores["soil_moisture"] = -1

    # 2. Rainfall forecast
    if rain_mm >= 20:
        feature_scores["rainfall_forecast"] = -3
    elif rain_mm >= 10:
        feature_scores["rainfall_forecast"] = -2
    elif rain_mm >= 5:
        feature_scores["rainfall_forecast"] = -1
    else:
        feature_scores["rainfall_forecast"] = 0

    # 3. Temperature
    if temperature > 38:
        feature_scores["temperature"] = +2
    elif temperature > 32:
        feature_scores["temperature"] = +1
    elif temperature < 18:
        feature_scores["temperature"] = -1
    else:
        feature_scores["temperature"] = 0

    # 4. Humidity
    if humidity < 25:
        feature_scores["humidity"] = +2
    elif humidity < 40:
        feature_scores["humidity"] = +1
    elif humidity > 80:
        feature_scores["humidity"] = -1
    else:
        feature_scores["humidity"] = 0

    # 5. Weather condition
    weather_adj = {
        "stormy":        -3,
        "rainy":         -2,
        "cloudy":        -1,
        "partly_cloudy":  0,
        "clear":         +1,
    }
    feature_scores["weather_condition"] = weather_adj.get(weather_cat, 0)

    # 6. Crop type water demand
    demand = CROP_WATER_DEMAND.get(crop, "MEDIUM")
    feature_scores["crop_type"] = {"HIGH": +1, "MEDIUM": 0, "LOW": -1}.get(demand, 0)

    base_score = sum(feature_scores.values())

    # ML model prediction (if loaded)
    model_used = False
    if _model and _crop_enc and _weather_enc:
        try:
            crop_enc_val    = _crop_enc.transform([crop])[0] if crop in _crop_enc.classes_ else 0
            weather_enc_val = _weather_enc.transform([weather_cat])[0] if weather_cat in _weather_enc.classes_ else 0
            X          = np.array([[crop_enc_val, soil_moisture, humidity, temperature, rain_mm, weather_enc_val]])
            prediction = int(_model.predict(X)[0])
            proba      = _model.predict_proba(X)[0]
            confidence = round(float(max(proba)) * 100, 1)
            model_used = True
        except Exception:
            prediction = 1 if base_score > 0 else 0
            confidence = min(95, 55 + abs(base_score) * 5)
    else:
        prediction = 1 if base_score > 0 else 0
        confidence = min(95, 55 + abs(base_score) * 5)

    # Human-readable reason
    reasons = []
    if feature_scores["soil_moisture"] > 0:
        reasons.append(f"soil moisture ({soil_moisture:.0f}%) is below {crop} threshold ({threshold}%)")
    elif feature_scores["soil_moisture"] < 0:
        reasons.append(f"soil moisture ({soil_moisture:.0f}%) is adequate for {crop}")
    if feature_scores["rainfall_forecast"] < -1:
        reasons.append(f"heavy rain forecast ({rain_mm:.1f} mm in 24 h)")
    if feature_scores["temperature"] > 1:
        reasons.append(f"high temperature ({temperature:.1f} C) increases evaporation")
    if feature_scores["humidity"] < 0:
        reasons.append(f"high humidity ({humidity}%) reduces water demand")
    elif feature_scores["humidity"] > 0:
        reasons.append(f"low humidity ({humidity}%) increases water demand")
    if not reasons:
        reasons.append("all sensor conditions are within normal range")

    prefix     = "Irrigation recommended: " if prediction else "Irrigation not needed: "
    reason_str = prefix + "; ".join(reasons) + "."

    # Top 3 SHAP-style features
    shap_values = sorted(
        [{"feature": k, "impact": v, "direction": "increase" if v > 0 else "decrease"}
         for k, v in feature_scores.items()],
        key=lambda x: abs(x["impact"]),
        reverse=True,
    )

    return {
        "irrigation_needed": prediction,
        "confidence":        confidence,
        "reason":            reason_str,
        "model_used":        "RandomForest" if model_used else "RuleEngine",
        "feature_scores":    feature_scores,
        "shap_top_features": shap_values[:3],
        "score":             base_score,
    }

# -------------------------------
# ANOMALY DETECTION
# -------------------------------
def detect_anomalies(soil_moisture):
    global _moisture_history
    anomalies = []

    # Rule 1: Out of range
    if soil_moisture < 0 or soil_moisture > 100:
        anomalies.append({
            "type":     "OUT_OF_RANGE",
            "severity": "HIGH",
            "message":  f"Moisture {soil_moisture:.1f}% is outside valid range (0-100%). Hardware fault suspected."
        })
        return anomalies

    # Rule 2: Sudden drop > 40%
    if _moisture_history:
        prev = _moisture_history[-1]
        drop = prev - soil_moisture
        if drop > 40:
            anomalies.append({
                "type":     "SUDDEN_DROP",
                "severity": "HIGH",
                "message":  f"Moisture dropped {drop:.1f}% in one reading ({prev:.1f}% to {soil_moisture:.1f}%). Possible pipe leak or sensor disconnect."
            })

    # Rule 3: Sensor stuck (same value 5+ readings)
    if len(_moisture_history) >= 5:
        last5 = _moisture_history[-5:]
        if (len(set(round(v, 1) for v in last5)) == 1 and
                round(soil_moisture, 1) == round(last5[-1], 1)):
            anomalies.append({
                "type":     "SENSOR_STUCK",
                "severity": "MEDIUM",
                "message":  f"Moisture stuck at {soil_moisture:.1f}% for 6 consecutive readings. Sensor may be disconnected or frozen."
            })

    # Rule 4: Prolonged dry
    critically_dry = sum(1 for v in _moisture_history[-10:] if v < 10)
    if soil_moisture < 10 and critically_dry >= 9:
        anomalies.append({
            "type":     "PROLONGED_DRY",
            "severity": "HIGH",
            "message":  "Moisture critically low (<10%) for 10+ consecutive readings. Immediate irrigation required."
        })

    _moisture_history.append(soil_moisture)
    if len(_moisture_history) > _MAX_HISTORY:
        _moisture_history.pop(0)

    return anomalies

# -------------------------------
# CROP RECOMMENDATION ENGINE
# -------------------------------
def recommend_crop(crop, total_readings, irrigation_count):
    if total_readings <= 0:
        raise ValueError("total_readings must be > 0")

    irrigation_rate = (irrigation_count / total_readings) * 100
    demand          = CROP_WATER_DEMAND.get(crop, "MEDIUM")
    threshold       = CROP_MOISTURE_THRESHOLDS.get(crop, 50)
    alternatives    = CROP_ALTERNATIVES.get(crop, [])

    should_change = (irrigation_rate > 65 and demand == "HIGH") or \
                    (irrigation_rate > 80 and demand == "MEDIUM")

    if should_change:
        reason  = (f"{crop.title()} has {demand} water demand (threshold: {threshold}%) and "
                   f"needed irrigation {irrigation_rate:.0f}% of the time. "
                   f"Switching to a lower-demand crop will significantly reduce water consumption.")
        message = f"Consider switching from {crop.title()} to one of the recommended alternatives."
    else:
        reason  = (f"{crop.title()} needed irrigation {irrigation_rate:.0f}% of the time, "
                   f"which is within acceptable limits for {demand.lower()} water-demand crops.")
        message = f"{crop.title()} water usage is acceptable for current conditions."

    return {
        "should_change":      should_change,
        "current_crop":       crop,
        "water_demand":       demand,
        "irrigation_rate":    round(irrigation_rate, 1),
        "moisture_threshold": threshold,
        "water_usage":        f"{irrigation_rate:.0f}%",
        "reason":             reason,
        "message":            message,
        "alternatives":       alternatives[:3],
        "water_saved_est":    WATER_SAVINGS.get(demand, "Varies"),
    }

# ==============================
# ROUTES
# ==============================

@app.route("/")
def home():
    acc = _metadata.get("accuracy") if _metadata else "N/A"
    return jsonify({"status": "Smart Irrigation API running", "model_accuracy": acc})


@app.route("/weather")
def weather():
    city = request.args.get("city", "Hyderabad")
    try:
        return jsonify(fetch_weather(city))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() 
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    city     = data.get("city", "Hyderabad")
    moisture = data.get("soil_moisture")
    crop     = str(data.get("crop", "rice")).lower().strip()

    if moisture is None:
        return jsonify({"error": "soil_moisture is required"}), 400
    try:
        moisture = float(moisture)
    except (TypeError, ValueError):
        return jsonify({"error": "soil_moisture must be a number"}), 400

    if crop not in CROP_MOISTURE_THRESHOLDS:
        crop = "rice"

    try:
        weather_data = fetch_weather(city)
        decision     = make_irrigation_decision(crop, moisture, weather_data)
        anomalies    = detect_anomalies(moisture)
        crop_rec     = recommend_crop(crop, 1, decision["irrigation_needed"])

        return jsonify({
            "irrigation_needed":   decision["irrigation_needed"],
            "confidence":          decision["confidence"],
            "reason":              decision["reason"],
            "model_used":          decision["model_used"],
            "shap_top_features":   decision["shap_top_features"],
            "feature_scores":      decision["feature_scores"],
            "weather_summary":     weather_data,
            "crop":                crop,
            "soil_moisture":       moisture,
            "anomalies":           anomalies,
            "anomaly_count":       len(anomalies),
            "crop_recommendation": crop_rec,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/anomaly", methods=["POST"])
def anomaly():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400
    try:
        moisture = float(data.get("soil_moisture", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "soil_moisture must be a number"}), 400

    anomalies = detect_anomalies(moisture)
    return jsonify({
        "soil_moisture": moisture,
        "anomalies":     anomalies,
        "anomaly_count": len(anomalies),
        "status":        "ALERT" if anomalies else "NORMAL",
    })


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    crop             = str(data.get("crop", "rice")).lower().strip()
    total_readings   = data.get("total_readings")
    irrigation_count = data.get("irrigation_count")

    if total_readings is None or irrigation_count is None:
        return jsonify({"error": "total_readings and irrigation_count are required"}), 400
    try:
        total_readings   = int(total_readings)
        irrigation_count = int(irrigation_count)
    except (TypeError, ValueError):
        return jsonify({"error": "total_readings and irrigation_count must be integers"}), 400

    if total_readings <= 0:
        return jsonify({"error": "total_readings must be > 0"}), 400
    if irrigation_count > total_readings:
        return jsonify({"error": "irrigation_count cannot exceed total_readings"}), 400

    try:
        return jsonify(recommend_crop(crop, total_readings, irrigation_count))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/info", methods=["GET"])
def model_info():
    if _metadata:
        return jsonify({
            "model":            "RandomForestClassifier",
            "accuracy":         _metadata.get("accuracy"),
            "training_samples": 2160,
            "features":         _metadata.get("features"),
            "crop_types":       _metadata.get("crop_types"),
            "status":           "loaded",
        })
    return jsonify({
        "model":  "RuleEngine (fallback)",
        "status": "run train_model.py to generate artifacts",
    })


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
