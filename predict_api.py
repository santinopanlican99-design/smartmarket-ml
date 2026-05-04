"""
predict_api.py — Smart Market External ML API
Deploy on Render.com so InfinityFree can call
the real Python model via HTTP instead of shell_exec.
"""

import json
import pickle
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))

# ── Load pre-trained artifacts (do NOT retrain on startup) ──────────
# Models are already saved as .pkl files in the repo.
# Re-training on every cold start adds 2-5 minutes of delay,
# causing ml_helper.php (60s timeout) to fall back to PHP scoring.
# ────────────────────────────────────────────────────────────────────

def load_artifacts():
    stats_path   = os.path.join(base_dir, "model_stats.json")
    encoder_path = os.path.join(base_dir, "label_encoder_brand.pkl")
    scaler_path  = os.path.join(base_dir, "scaler.pkl")

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    best_model_name = stats.get("best_model", "Random Forest")
    algorithm_files = stats.get("algorithm_files", {})
    scaled_flag     = stats.get("algorithms", {}).get(best_model_name, {}).get("scaled", False)

    # If best model is Naive Bayes, prefer Random Forest instead.
    # Naive Bayes outputs overconfident probabilities (often 0.0 or 1.0),
    # which made all products score 1.0000.
    if best_model_name == "Naive Bayes":
        rf_path = os.path.join(base_dir, "model_rf.pkl")
        if os.path.exists(rf_path):
            print("Naive Bayes detected — switching to Random Forest for better probability calibration.")
            best_model_name = "Random Forest"
            scaled_flag     = False
            model_path      = rf_path
        else:
            model_file = algorithm_files.get(best_model_name, "model_rf.pkl")
            model_path = os.path.join(base_dir, model_file)
    else:
        model_file = algorithm_files.get(best_model_name, "model_rf.pkl")
        model_path = os.path.join(base_dir, model_file)

    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "model_rf.pkl")

    with open(model_path,   "rb") as f: model         = pickle.load(f)
    with open(encoder_path, "rb") as f: label_encoder = pickle.load(f)
    with open(scaler_path,  "rb") as f: scaler        = pickle.load(f)

    return model, label_encoder, scaler, best_model_name, scaled_flag


model, label_encoder, scaler, best_model_name, use_scaler = load_artifacts()
print(f"Smart Market ML API ready — model: {best_model_name} (scaled={use_scaler})")

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
    "device_age",
]

# Quality weight per class:
# Class 0 = Poor (<65)      → 0.2
# Class 1 = Fair (65-74)    → 0.5
# Class 2 = Good (75-84)    → 0.8
# Class 3 = Excellent (85+) → 1.0
CLASS_WEIGHTS = [0.2, 0.5, 0.8, 1.0]


class PredictHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "ok",
            "model":  best_model_name,
        }).encode())

    def do_POST(self):
        if self.path != "/predict":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)

        try:
            input_data = json.loads(body)
            brand      = str(input_data.get("device_brand", "Unknown"))

            brand_encoded = (
                int(label_encoder.transform([brand])[0])
                if brand in label_encoder.classes_ else 0
            )

            row = {
                "device_brand_encoded" : brand_encoded,
                "ram"                  : float(input_data.get("ram", 0)),
                "internal_memory"      : float(input_data.get("internal_memory", 0)),
                "rear_camera_mp"       : float(input_data.get("rear_camera_mp", 0)),
                "battery"              : float(input_data.get("battery", 0)),
                "release_year"         : float(input_data.get("release_year", 0)),
                "battery_health"       : float(input_data.get("battery_health", 0)),
                "condition_score"      : float(input_data.get("condition_score", 0)),
                "normalized_used_price": float(input_data.get("normalized_used_price", 0)),
                "device_age"           : float(input_data.get("device_age", 0)),
            }

            df = pd.DataFrame([row])[FEATURE_COLS]

            if use_scaler:
                df = pd.DataFrame(scaler.transform(df), columns=FEATURE_COLS)

            prediction = model.predict(df)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(df)[0]
                n     = min(len(proba), len(CLASS_WEIGHTS))
                score = float(sum(proba[i] * CLASS_WEIGHTS[i] for i in range(n)))
                score = round(min(max(score, 0.0), 1.0), 4)
            else:
                pred  = int(prediction[0])
                score = CLASS_WEIGHTS[min(pred, 3)]

            result = {
                "prediction": int(prediction[0]),
                "score":      score,
                "model_used": best_model_name,
            }

        except Exception as e:
            result = {
                "prediction": 0,
                "score":      0.0,
                "model_used": "Error",
                "error":      str(e),
            }

        response = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type",    "application/json")
        self.send_header("Content-Length",  str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Listening on port {port}")
    HTTPServer(("0.0.0.0", port), PredictHandler).serve_forever()
