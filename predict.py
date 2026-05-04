import sys
import json
import pickle
import pandas as pd
import os

base_dir = os.path.dirname(__file__)

stats_path   = os.path.join(base_dir, "model_stats.json")
encoder_path = os.path.join(base_dir, "label_encoder_brand.pkl")
scaler_path  = os.path.join(base_dir, "scaler.pkl")

FEATURE_COLS = [
    "device_brand_encoded",
    "ram",
    "internal_memory",
    "rear_camera_mp",
    "battery",
    "release_year",
    "battery_health",
    "condition_score",
    "normalized_used_price",
    "device_age"
]

# Weight each quality class: Poor=0.2, Fair=0.5, Good=0.8, Excellent=1.0
CLASS_WEIGHTS = [0.2, 0.5, 0.8, 1.0]

try:
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Label encoder not found: {encoder_path}")

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    best_model_name = stats.get("best_model", "Random Forest")
    algorithm_files = stats.get("algorithm_files", {})
    model_file      = algorithm_files.get(best_model_name, "model_rf.pkl")
    model_path      = os.path.join(base_dir, model_file)

    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "model_rf.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Load scaler (used by NB, LR, SVM, KNN, Ensemble)
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    needs_scaling = stats.get("algorithms", {}).get(best_model_name, {}).get("scaled", False)

    if len(sys.argv) < 2:
        raise ValueError("No input file provided")
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    brand = str(input_data.get("device_brand", "Unknown"))
    brand_encoded = int(label_encoder.transform([brand])[0]) if brand in label_encoder.classes_ else 0

    row = {
        "device_brand_encoded"  : brand_encoded,
        "ram"                   : float(input_data.get("ram", 0)),
        "internal_memory"       : float(input_data.get("internal_memory", 0)),
        "rear_camera_mp"        : float(input_data.get("rear_camera_mp", 0)),
        "battery"               : float(input_data.get("battery", 0)),
        "release_year"          : float(input_data.get("release_year", 0)),
        "battery_health"        : float(input_data.get("battery_health", 0)),
        "condition_score"       : float(input_data.get("condition_score", 0)),
        "normalized_used_price" : float(input_data.get("normalized_used_price", 0)),
        "device_age"            : float(input_data.get("device_age", 0)),
    }

    df = pd.DataFrame([row])[FEATURE_COLS]

    # Apply scaler if this model needs it
    if needs_scaling and scaler is not None:
        df = pd.DataFrame(scaler.transform(df), columns=FEATURE_COLS)

    prediction = model.predict(df)

    # FIX: use weighted class probabilities, not max(proba).
    # OLD: score = max(proba[0])  → always returns the confidence level (e.g. 0.80),
    #      ignoring WHICH class was predicted. A "Poor" phone with 80% confidence
    #      got the same score as an "Excellent" phone with 80% confidence.
    # NEW: weight each class by quality so score reflects actual phone quality.
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        n     = min(len(proba), len(CLASS_WEIGHTS))
        score = float(sum(proba[i] * CLASS_WEIGHTS[i] for i in range(n)))
        score = round(min(max(score, 0.0), 1.0), 4)
    else:
        pred  = int(prediction[0])
        score = CLASS_WEIGHTS[min(pred, 3)]

    result = {
        "prediction" : int(prediction[0]),
        "score"      : score,
        "model_used" : best_model_name,
    }

    print(json.dumps(result))

except Exception as e:
    print(json.dumps({
        "prediction" : 0,
        "score"      : 0.0,
        "model_used" : "Unavailable",
        "error"      : str(e),
    }))
