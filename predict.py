import sys
import json
import pickle
import pandas as pd
import os

base_dir = os.path.dirname(__file__)

stats_path = os.path.join(base_dir, "model_stats.json")
encoder_path = os.path.join(base_dir, "label_encoder_brand.pkl")

try:
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Label encoder not found: {encoder_path}")

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    best_model_name = stats.get("best_model", "Random Forest")
    algorithm_files = stats.get("algorithm_files", {})

    model_file = algorithm_files.get(best_model_name, "model_rf.pkl")
    model_path = os.path.join(base_dir, model_file)

    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "model_rf.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    if len(sys.argv) < 2:
        raise ValueError("No input file provided")

    input_file = sys.argv[1]

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    brand = str(input_data.get("device_brand", "Unknown"))

    if brand in label_encoder.classes_:
        brand_encoded = int(label_encoder.transform([brand])[0])
    else:
        brand_encoded = 0

    row = {
        "device_brand_encoded": brand_encoded,
        "ram": float(input_data.get("ram", 0)),
        "internal_memory": float(input_data.get("internal_memory", 0)),
        "rear_camera_mp": float(input_data.get("rear_camera_mp", 0)),
        "battery": float(input_data.get("battery", 0)),
        "release_year": float(input_data.get("release_year", 0)),
        "battery_health": float(input_data.get("battery_health", 0)),
        "condition_score": float(input_data.get("condition_score", 0)),
        "normalized_used_price": float(input_data.get("normalized_used_price", 0)),
        "device_age": float(input_data.get("device_age", 0))
    }

    df = pd.DataFrame([row])

    prediction = model.predict(df)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)
        score = float(max(proba[0]))
    else:
        score = 0.0

    result = {
        "prediction": int(prediction[0]),
        "score": score,
        "model_used": best_model_name
    }

    print(json.dumps(result))

except Exception as e:
    print(json.dumps({
        "prediction": 0,
        "score": 0.0,
        "model_used": "Unavailable",
        "error": str(e)
    }))