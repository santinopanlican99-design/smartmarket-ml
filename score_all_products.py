"""
score_all_products.py — Smart Market Local Scorer
==================================================
Runs locally on your PC, connects to the live InfinityFree DB,
scores ALL products using the best trained ML model, and updates
the database directly.

SETUP (run once in your terminal):
    pip install mysql-connector-python pandas scikit-learn numpy

HOW TO RUN:
    1. Put this file inside your project's  ml/  folder
       (same folder as predict.py, model_stats.json, etc.)
    2. Open terminal / cmd in that folder
    3. Run:  python score_all_products.py

WHAT IT DOES:
    - Loads the best model from model_stats.json
    - Fetches all products from your live DB
    - Predicts an ml_score for each product
    - Updates the ml_score column in the DB
    - Prints a summary when done
"""

import os
import sys
import json
import pickle
import math
import numpy as np
import pandas as pd
import mysql.connector
from datetime import datetime

# ── DB credentials (same as db_connect.php) ──────────────────────────
DB_HOST = "sql301.infinityfree.com"
DB_NAME = "if0_41369385_2ndhandphones"
DB_USER = "if0_41369385"
DB_PASS = "SoJ9uOrKQOv9"
DB_PORT = 3306

# ── Paths (relative to this script's location = ml/ folder) ──────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATS_PATH = os.path.join(BASE_DIR, "model_stats.json")
ENC_PATH   = os.path.join(BASE_DIR, "label_encoder_brand.pkl")
SCALER_PATH= os.path.join(BASE_DIR, "scaler.pkl")

FEATURE_COLS = [
    'device_brand_encoded',
    'ram',
    'internal_memory',
    'rear_camera_mp',
    'battery',
    'release_year',
    'battery_health',
    'condition_score',
    'normalized_used_price',
    'device_age'
]

def load_model():
    """Load best model, label encoder, scaler from ml/ folder."""
    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError(
            f"model_stats.json not found at {STATS_PATH}\n"
            "Run train_model.py first!"
        )

    with open(STATS_PATH, "r") as f:
        stats = json.load(f)

    best_name  = stats.get("best_model", "Random Forest")
    algo_files = stats.get("algorithm_files", {})
    model_file = algo_files.get(best_name, "model_rf.pkl")
    model_path = os.path.join(BASE_DIR, model_file)

    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, "model_rf.pkl")

    print(f"  Model    : {best_name}")
    print(f"  CV Acc   : {stats.get('best_accuracy', '?')}%")
    print(f"  File     : {os.path.basename(model_path)}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(ENC_PATH, "rb") as f:
        label_enc = pickle.load(f)

    scaler = None
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

    needs_scaling = stats.get("algorithms", {}).get(best_name, {}).get("scaled", False)

    return model, label_enc, scaler, needs_scaling, best_name


def connect_db():
    """Connect to the remote InfinityFree MySQL database."""
    print(f"  Host     : {DB_HOST}")
    print(f"  Database : {DB_NAME}")
    conn = mysql.connector.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT,
        connection_timeout=15
    )
    print(f"  Status   : Connected ✅")
    return conn


def fetch_products(conn):
    """Fetch all products needed for scoring."""
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT
            id,
            device_brand,
            ram_gb,
            internal_memory_gb,
            rear_camera_mp,
            battery_mah,
            release_year,
            battery_health,
            condition_score,
            selling_price,
            seller_rating,
            device_age_years,
            ml_score
        FROM products
        ORDER BY id
    """)
    rows = cursor.fetchall()
    cursor.close()
    return rows


def build_features(product, label_enc, current_year=None):
    """Convert a product row into the feature vector for prediction."""
    if current_year is None:
        current_year = datetime.now().year

    brand = str(product.get("device_brand") or "Unknown")
    if brand in label_enc.classes_:
        brand_enc = int(label_enc.transform([brand])[0])
    else:
        brand_enc = 0

    release_year = float(product.get("release_year") or current_year)
    # Prefer stored device_age_years, fall back to computed value
    device_age = float(product.get("device_age_years") or (current_year - release_year))

    # condition_score: DB may store 0-10, model trained on 1-5
    cond_raw = float(product.get("condition_score") or 0)
    if cond_raw > 5:
        cond_score = max(1.0, min(5.0, round(cond_raw / 2, 1)))
    else:
        cond_score = max(1.0, min(5.0, cond_raw if cond_raw > 0 else 3.0))

    # normalized_used_price: model trained on RAW prices (e.g. 8500), NOT divided
    raw_price = float(product.get("selling_price") or 0)

    return {
        "device_brand_encoded"  : brand_enc,
        "ram"                   : float(product.get("ram_gb") or 0),
        "internal_memory"       : float(product.get("internal_memory_gb") or 0),
        "rear_camera_mp"        : float(product.get("rear_camera_mp") or 0),
        "battery"               : float(product.get("battery_mah") or 0),
        "release_year"          : release_year,
        "battery_health"        : float(product.get("battery_health") or 80),
        "condition_score"       : cond_score,
        "normalized_used_price" : raw_price,
        "device_age"            : device_age
    }


def predict_score(model, scaler, needs_scaling, features):
    """Run prediction and return a 0-1 ml_score."""
    df = pd.DataFrame([features])[FEATURE_COLS]

    if needs_scaling and scaler is not None:
        df_input = pd.DataFrame(
            scaler.transform(df), columns=FEATURE_COLS
        )
    else:
        df_input = df

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df_input)[0]
        # Weight classes: Poor=0.2, Fair=0.5, Good=0.8, Excellent=1.0
        class_weights = [0.2, 0.5, 0.8, 1.0]
        n = min(len(proba), len(class_weights))
        score = float(sum(proba[i] * class_weights[i] for i in range(n)))
    else:
        pred = int(model.predict(df_input)[0])
        score = [0.2, 0.5, 0.8, 1.0][min(pred, 3)]

    return round(min(max(score, 0.0), 1.0), 4)


def update_scores(conn, updates):
    """Bulk update ml_score for all products."""
    cursor = conn.cursor()
    cursor.executemany(
        "UPDATE products SET ml_score = %s WHERE id = %s",
        updates
    )
    conn.commit()
    cursor.close()


def main():
    print("\n" + "=" * 55)
    print("  Smart Market — Local Product Scorer")
    print("=" * 55)

    # 1. Load model
    print("\n📦 Loading ML model...")
    model, label_enc, scaler, needs_scaling, best_name = load_model()

    # 2. Connect to DB
    print("\n🔌 Connecting to database...")
    try:
        conn = connect_db()
    except Exception as e:
        print(f"\n❌ DB connection failed: {e}")
        print("\nTip: Make sure you're connected to the internet.")
        sys.exit(1)

    # 3. Fetch products
    print("\n📋 Fetching products...")
    products = fetch_products(conn)
    total = len(products)
    already_scored = sum(1 for p in products if p.get("ml_score") is not None)
    print(f"  Total    : {total} products")
    print(f"  Scored   : {already_scored} already have ml_score")
    print(f"  To score : {total - already_scored} without score")

    # 4. Score all products
    print(f"\n🤖 Scoring all {total} products with {best_name}...")
    updates = []
    current_year = datetime.now().year
    errors = 0

    for i, product in enumerate(products, 1):
        try:
            features = build_features(product, label_enc, current_year)
            score    = predict_score(model, scaler, needs_scaling, features)
            updates.append((score, product["id"]))

            # Progress bar
            bar_len = 30
            filled  = int(bar_len * i / total)
            bar     = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{bar}] {i}/{total}  last: ID#{product['id']} score={score}", end="", flush=True)
        except Exception as e:
            errors += 1
            updates.append((None, product["id"]))

    print()  # newline after progress bar

    # 5. Update DB
    print(f"\n💾 Updating database...")
    update_scores(conn, [(s, pid) for s, pid in updates if s is not None])
    conn.close()

    successful = len([s for s, _ in updates if s is not None])
    scores = [s for s, _ in updates if s is not None]

    print(f"\n{'=' * 55}")
    print(f"  ✅ Done! {successful}/{total} products scored")
    if scores:
        print(f"  📊 Average ML Score : {round(sum(scores)/len(scores), 4)}")
        print(f"  📈 Highest Score    : {max(scores)}")
        print(f"  📉 Lowest Score     : {min(scores)}")
    if errors:
        print(f"  ⚠️  Errors           : {errors} products skipped")
    print(f"{'=' * 55}")
    print("\n🚀 All done! Refresh your Smart Market analytics page.\n")


if __name__ == "__main__":
    main()
