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

# ── Load pre-trained Decision Tree artifact ─────────────────────
def load_artifacts():
    encoder_path = os.path.join(base_dir, "label_encoder_brand.pkl")
    model_path   = os.path.join(base_dir, "model_dt.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Decision Tree model not found: {model_path}")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Label encoder not found: {encoder_path}")

    with open(model_path,   "rb") as f: model         = pickle.load(f)
    with open(encoder_path, "rb") as f: label_encoder = pickle.load(f)

    # Decision Tree does not require scaling
    return model, label_encoder


model, label_encoder = load_artifacts()
print("Smart Market ML API ready — model: Decision Tree (scaled=False)")

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

# Equal-width class weights (evenly spaced, no big jumps):
#   Class 0 = Poor     (score < 40)   -> 0.25
#   Class 1 = Fair     (40 - 60)      -> 0.50
#   Class 2 = Good     (60 - 80)      -> 0.75
#   Class 3 = Excellent(80+)          -> 1.00
CLASS_WEIGHTS = [0.25, 0.50, 0.75, 1.00]


class PredictHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "ok",
            "model":  "Decision Tree",
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

            # No scaling needed for Decision Tree
            prediction = model.predict(df)

            # Weighted score using evenly spaced class weights
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
                "model_used": "Decision Tree",
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
