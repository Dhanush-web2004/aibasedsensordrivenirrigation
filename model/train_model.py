"""
Smart Irrigation ML Model Training
Generates synthetic 3-month dataset and trains RandomForestClassifier
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import os

# ── Reproducibility ─────────────────────────────────────────────────────────
np.random.seed(42)

# ── Constants ────────────────────────────────────────────────────────────────
CROP_TYPES = ["rice", "wheat", "maize", "sugarcane", "vegetables", "cotton"]
WEATHER_CONDITIONS = ["clear", "cloudy", "rainy", "partly_cloudy", "stormy"]
N_SAMPLES = 2160  # ~24 records/day × 90 days

CROP_MOISTURE_THRESHOLDS = {
    "rice": 70,
    "sugarcane": 65,
    "vegetables": 55,
    "maize": 45,
    "cotton": 40,
    "wheat": 35,
}


def generate_dataset() -> pd.DataFrame:
    crop_types = np.random.choice(CROP_TYPES, N_SAMPLES)
    soil_moisture = np.random.uniform(10, 95, N_SAMPLES)
    humidity = np.random.uniform(20, 95, N_SAMPLES)
    temperature = np.random.uniform(15, 42, N_SAMPLES)
    rainfall_forecast = np.zeros(N_SAMPLES)

    weather_condition_idx = np.random.choice(len(WEATHER_CONDITIONS), N_SAMPLES, p=[0.30, 0.25, 0.20, 0.20, 0.05])
    weather_condition = [WEATHER_CONDITIONS[i] for i in weather_condition_idx]

    # Correlate rainfall with weather condition
    for i, wc in enumerate(weather_condition):
        if wc == "rainy":
            rainfall_forecast[i] = np.random.uniform(5, 40)
        elif wc == "stormy":
            rainfall_forecast[i] = np.random.uniform(20, 80)
        elif wc == "cloudy":
            rainfall_forecast[i] = np.random.uniform(0, 8)
        elif wc == "partly_cloudy":
            rainfall_forecast[i] = np.random.uniform(0, 3)
        else:
            rainfall_forecast[i] = 0.0

    # Determine irrigation_needed label
    irrigation_needed = []
    for i in range(N_SAMPLES):
        crop = crop_types[i]
        threshold = CROP_MOISTURE_THRESHOLDS.get(crop, 50)
        moisture = float(soil_moisture[i])
        rain = float(rainfall_forecast[i])
        temp = float(temperature[i])
        hum = float(humidity[i])

        score = 0
        if moisture < threshold:
            score += 2
        if moisture < threshold * 0.6:
            score += 1  # very dry
        if rain >= 10:
            score -= 2
        elif rain >= 5:
            score -= 1
        if temp > 35:
            score += 1
        if hum < 30:
            score += 1

        # Add slight noise
        score += np.random.uniform(-0.5, 0.5)
        irrigation_needed.append(1 if score > 0 else 0)

    df = pd.DataFrame({
        "crop_type": crop_types,
        "soil_moisture": soil_moisture,
        "humidity": humidity,
        "temperature": temperature,
        "rainfall_forecast": rainfall_forecast,
        "weather_condition": weather_condition,
        "irrigation_needed": irrigation_needed,
    })
    return df


def train_and_save():
    print("📊 Generating synthetic dataset...")
    df = generate_dataset()
    print(f"   Dataset shape: {df.shape}")
    print(f"   Irrigation needed: {df['irrigation_needed'].value_counts().to_dict()}")

    # Encode categoricals
    crop_le = LabelEncoder()
    weather_le = LabelEncoder()
    df["crop_type_enc"] = crop_le.fit_transform(df["crop_type"])
    df["weather_condition_enc"] = weather_le.fit_transform(df["weather_condition"])

    features = ["crop_type_enc", "soil_moisture", "humidity", "temperature",
                "rainfall_forecast", "weather_condition_enc"]
    X = df[features]
    y = df["irrigation_needed"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n🤖 Training RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(clf, "artifacts/model.pkl")
    joblib.dump(crop_le, "artifacts/crop_encoder.pkl")
    joblib.dump(weather_le, "artifacts/weather_encoder.pkl")

    # Save metadata for the backend
    metadata = {
        "crop_types": list(crop_le.classes_),
        "weather_conditions": list(weather_le.classes_),
        "features": features,
        "accuracy": round(acc, 4),
        "crop_moisture_thresholds": CROP_MOISTURE_THRESHOLDS,
    }
    with open("artifacts/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n💾 Saved: artifacts/model.pkl, crop_encoder.pkl, weather_encoder.pkl, metadata.json")
    return acc


if __name__ == "__main__":
    train_and_save()