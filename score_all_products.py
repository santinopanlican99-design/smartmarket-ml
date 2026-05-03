"""
score_all_products.py — Smart Market Scorer
Fetches products from live site, scores them, outputs SQL file.
"""

import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
STATS_PATH  = os.path.join(BASE_DIR, "model_stats.json")
ENC_PATH    = os.path.join(BASE_DIR, "label_encoder_brand.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
SITE_URL    = os.environ.get("SITE_URL", "https://secondhandphones.infinityfreeapp.com")
EXPORT_TOKEN= os.environ.get("EXPORT_TOKEN", "")

FEATURE_COLS = [
    "device_brand_encoded","ram","internal_memory","rear_camera_mp",
    "battery","release_year","battery_health","condition_score",
    "normalized_used_price","device_age"
]

def load_model():
    with open(STATS_PATH) as f:
        stats = json.load(f)
    best_name  = stats.get("best_model", "Random Forest")
    algo_files = stats.get("algorithm_files", {})
    model_file = algo_files.get(best_name, "model_rf.pkl")
    model_path = os.path.join(BASE_DIR, model_file)
    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, "model_rf.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(ENC_PATH, "rb") as f:
        label_enc = pickle.load(f)
    scaler = None
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    needs_scaling = stats.get("algorithms", {}).get(best_name, {}).get("scaled", False)
    print(f"  Model : {best_name} (CV: {stats.get(\"best_accuracy\", \"?\"):.2f}%)")
    return model, label_enc, scaler, needs_scaling

def fetch_products():
    import urllib.request
    # Generate daily token matching PHP: md5(DB_PASS + today)
    import hashlib
    db_pass = os.environ.get("DB_PASS", "SoJ9uOrKQOv9")
    today   = datetime.now().strftime("%Y-%m-%d")
    token   = "sm_export_" + hashlib.md5((db_pass + today).encode()).hexdigest()
    url     = f"{SITE_URL}/api/export_products.php?token={token}"
    print(f"  Fetching: {url}")
    req  = urllib.request.urlopen(url, timeout=30)
    data = json.loads(req.read().decode())
    if "error" in data:
        raise Exception(data["error"])
    print(f"  Got {data[\"total\"]} products")
    return data["products"]

def build_features(product, label_enc, current_year):
    brand = str(product.get("device_brand") or "Unknown")
    try:
        brand_enc = int(label_enc.transform([brand])[0]) if brand in label_enc.classes_ else 0
    except:
        brand_enc = 0
    release_year = float(product.get("release_year") or current_year)
    price = float(product.get("selling_price") or 0)
    return {
        "device_brand_encoded"  : brand_enc,
        "ram"                   : float(product.get("ram") or 0),
        "internal_memory"       : float(product.get("internal_memory") or 0),
        "rear_camera_mp"        : float(product.get("rear_camera_mp") or 0),
        "battery"               : float(product.get("battery") or 0),
        "release_year"          : release_year,
        "battery_health"        : float(product.get("battery_health") or 80),
        "condition_score"       : float(product.get("condition_score") or 50),
        "normalized_used_price" : min(price / 50000.0, 1.0),
        "device_age"            : current_year - release_year
    }

def predict_score(model, scaler, needs_scaling, features):
    df = pd.DataFrame([features])[FEATURE_COLS]
    df_input = pd.DataFrame(scaler.transform(df), columns=FEATURE_COLS) if (needs_scaling and scaler) else df
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df_input)[0]
        weights = [0.2, 0.5, 0.8, 1.0]
        score = sum(proba[i] * weights[i] for i in range(min(len(proba), 4)))
    else:
        pred  = int(model.predict(df_input)[0])
        score = [0.2, 0.5, 0.8, 1.0][min(pred, 3)]
    return round(min(max(float(score), 0.0), 1.0), 4)

def write_sql(updates):
    path = os.path.join(BASE_DIR, "scores_output.sql")
    with open(path, "w") as f:
        f.write("-- Smart Market ML Scores\n")
        f.write(f"-- Generated: {datetime.now()}\n\n")
        for score, pid in updates:
            if score is not None:
                f.write(f"UPDATE products SET ml_score = {score} WHERE id = {pid};\n")
    return path

def main():
    print("\n" + "="*50)
    print("  Smart Market — Product Scorer")
    print("="*50)

    print("\n Loading model...")
    model, label_enc, scaler, needs_scaling = load_model()

    print("\n Fetching products from site...")
    try:
        products = fetch_products()
    except Exception as e:
        print(f"\n ERROR fetching products: {e}")
        sys.exit(1)

    print(f"\n Scoring {len(products)} products...")
    updates = []
    year = datetime.now().year
    errors = 0
    for i, p in enumerate(products, 1):
        try:
            feat  = build_features(p, label_enc, year)
            score = predict_score(model, scaler, needs_scaling, feat)
            updates.append((score, p["id"]))
        except Exception as e:
            errors += 1
            updates.append((None, p["id"]))
        bar = "█" * int(30*i/len(products)) + "░" * (30 - int(30*i/len(products)))
        print(f"\r  [{bar}] {i}/{len(products)}", end="", flush=True)
    print()

    print("\n Writing SQL file...")
    path = write_sql(updates)
    good = [s for s,_ in updates if s is not None]

    print(f"\n" + "="*50)
    print(f"  Done! {len(good)}/{len(products)} products scored")
    if good:
        print(f"  Avg score : {round(sum(good)/len(good), 4)}")
        print(f"  Highest   : {max(good)}")
        print(f"  Lowest    : {min(good)}")
    if errors:
        print(f"  Errors    : {errors}")
    print(f"  SQL file  : {path}")
    print("="*50)
    print("\n Download scores_output.sql from Artifacts and import in phpMyAdmin!\n")

if __name__ == "__main__":
    main()
