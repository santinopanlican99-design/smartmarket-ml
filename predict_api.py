"""
predict_api.py — Smart Market External ML API
Deploy this FREE on Render.com so InfinityFree can call
the real Python model via HTTP instead of shell_exec.
"""

import json
import pickle
import os
import subprocess
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))

# ── Auto-retrain if dataset is available ────────────────────────
dataset_path = os.path.join(base_dir, "phones_dataset.csv")
train_script = os.path.join(base_dir, "train_model.py")
if os.path.exists(dataset_path) and os.path.exists(train_script):
    print("Dataset found — retraining models on startup...")
    subprocess.run([sys.executable, train_script], check=False)
    print("Retraining complete.")
else:
    print("No dataset found — using existing model files.")
# ────────────────────────────────────────────────────────────────

def load_artifacts():
    stats_path    = os.path.join(base_dir, "model_stats.json")
    encoder_path  = os.path.join(base_dir, "label_encoder_brand.pkl")
    scaler_path   = os.path.join(base_dir, "scaler.pkl")

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    best_model_name = stats.get("best_model", "Random Forest")
    algorithm_files = stats.get("algorithm_files", {})
    scaled_flag     = stats.get("algorithms", {}).get(best_model_name, {}).get("scaled", False)

    model_file = algorithm_files.get(best_model_name, "model_rf.pkl")
    model_path = os.path.join(base_dir, model_file)
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "model_rf.pkl")

    with open(model_path,   "rb") as f: model         = pickle.load(f)
    with open(encoder_path, "rb") as f: label_encoder = pickle.load(f)
    with open(scaler_path,  "rb") as f: scaler        = pickle.load(f)

    return model, label_encoder, scaler, best_model_name, scaled_flag

model, label_encoder, scaler, best_model_name, use_scaler = load_artifacts()
print(f"Loaded model: {best_model_name} (scaled={use_scaler})")

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

class PredictHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "model": best_model_name}).encode())

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

            brand_encoded = int(label_encoder.transform([brand])[0]) \
                            if brand in label_encoder.classes_ else 0

            row = {col: 0.0 for col in FEATURE_COLS}
            row["device_brand_encoded"]   = brand_encoded
            row["ram"]                    = float(input_data.get("ram", 0))
            row["internal_memory"]        = float(input_data.get("internal_memory", 0))
            row["rear_camera_mp"]         = float(input_data.get("rear_camera_mp", 0))
            row["battery"]                = float(input_data.get("battery", 0))
            row["release_year"]           = float(input_data.get("release_year", 0))
            row["battery_health"]         = float(input_data.get("battery_health", 0))
            row["condition_score"]        = float(input_data.get("condition_score", 0))
            row["normalized_used_price"]  = float(input_data.get("normalized_used_price", 0))
            row["device_age"]             = float(input_data.get("device_age", 0))

            df = pd.DataFrame([row])

            if use_scaler:
                df = pd.DataFrame(scaler.transform(df), columns=FEATURE_COLS)

            prediction = model.predict(df)
            score = float(max(model.predict_proba(df)[0])) \
                    if hasattr(model, "predict_proba") else 0.0

            result = {
                "prediction": int(prediction[0]),
                "score":      score,
                "model_used": best_model_name,
            }
        except Exception as e:
            result = {"prediction": 0, "score": 0.0, "model_used": "Error", "error": str(e)}

        response = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Smart Market ML API running on port {port} — model: {best_model_name}")
    HTTPServer(("0.0.0.0", port), PredictHandler).serve_forever()
