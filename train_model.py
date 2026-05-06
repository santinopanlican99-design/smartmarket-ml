"""
train_model.py — Smart Market
Trains Decision Tree Classifier only.

Run:
    python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings

warnings.filterwarnings('ignore')

base_dir = os.path.dirname(os.path.abspath(__file__))

# ── Support both dataset locations ──────────────────────────────
dataset_candidates = [
    os.path.join(base_dir, 'dataset', 'phones_dataset.csv'),
    os.path.join(base_dir, 'phones_dataset.csv'),
]
dataset_path = next((p for p in dataset_candidates if os.path.exists(p)), None)
if dataset_path is None:
    raise FileNotFoundError("phones_dataset.csv not found. Put it in dataset/ or same folder.")

print("=" * 60)
print("  Smart Market — ML Model Training")
print("  Algorithm : Decision Tree")
print("  Scaler    : MinMaxScaler (0-1)")
print("  Max Depth : 5  (prevents overfitting / overconfident scores)")
print("  Classes   : Poor<40 | Fair 40-60 | Good 60-80 | Excellent 80+")
print("=" * 60)

df = pd.read_csv(dataset_path)
print(f"\n✅ Dataset loaded: {len(df)} phone records")
print(f"   Columns: {list(df.columns)}\n")

# ── Support both column names for the target score ───────
if 'recommendation_score' in df.columns:
    score_col = 'recommendation_score'
elif 'ml_score' in df.columns:
    score_col = 'ml_score'
else:
    raise ValueError("Dataset must have 'recommendation_score' or 'ml_score' column.")

print(f"   Using score column: '{score_col}'")
print(f"   Score range: {df[score_col].min()} - {df[score_col].max()}\n")

# ── Normalize used price to 0-1 ──────────────────────────────────
if df['normalized_used_price'].max() > 10:
    print("   Warning: Detected raw prices — normalizing by /100000")
    df['normalized_used_price'] = df['normalized_used_price'] / 100000

# ── Equal-width class boundaries (20-point bands) ───────────────
#   Poor     : score < 40
#   Fair     : 40 <= score < 60
#   Good     : 60 <= score < 80
#   Excellent: score >= 80
def score_to_class(score):
    score = float(score)
    if score >= 80:   return 3  # Excellent
    elif score >= 60: return 2  # Good
    elif score >= 40: return 1  # Fair
    else:             return 0  # Poor

df['rec_class'] = df[score_col].apply(score_to_class)

le_brand = LabelEncoder()
df['device_brand_encoded'] = le_brand.fit_transform(df['device_brand'].astype(str))

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
    'device_age',
]

FEATURE_LABELS = [
    'Brand', 'RAM', 'Storage', 'Camera (MP)', 'Battery (mAh)',
    'Release Year', 'Battery Health', 'Condition Score', 'Used Price', 'Device Age'
]

CLASS_NAMES = ['Poor (<40)', 'Fair (40-60)', 'Good (60-80)', 'Excellent (80+)']

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")

X = df[FEATURE_COLS].fillna(0)
y = df['rec_class']

print(f"Class distribution:\n{y.value_counts().sort_index().rename({0:'Poor',1:'Fair',2:'Good',3:'Excellent'})}\n")

# ── MinMaxScaler ─────────────────────────────────────────────────
scaler = MinMaxScaler()
scaler.fit(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Train Decision Tree ──────────────────────────────────────────
# max_depth=5  → shallower tree = less overconfident probabilities
# min_samples_split=5 → needs at least 5 samples to split a node
print("Training Decision Tree (max_depth=5, min_samples_split=5)...")
model = DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
cv   = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print(f"\n{'Metric':<20} {'Value':>10}")
print("-" * 32)
print(f"{'Test Accuracy':<20} {acc*100:>9.2f}%")
print(f"{'CV Accuracy':<20} {cv.mean()*100:>9.2f}%")
print(f"{'CV Std Dev':<20} {cv.std()*100:>9.2f}%")
print(f"{'Precision':<20} {prec*100:>9.2f}%")
print(f"{'Recall':<20} {rec*100:>9.2f}%")
print(f"{'F1 Score':<20} {f1*100:>9.2f}%")

# ── Feature Importance ───────────────────────────────────────────
print(f"\nFeature Importance (Decision Tree):")
feat_imp = sorted(zip(FEATURE_LABELS, model.feature_importances_), key=lambda x: -x[1])
for feat, imp in feat_imp:
    print(f"   {feat:<20} {'#'*int(imp*40)} {imp:.3f}")

# ── Classification Report ────────────────────────────────────────
print(f"\nClassification Report (Decision Tree):")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

# ── Save model artifacts ─────────────────────────────────────────
with open(os.path.join(base_dir, 'model_dt.pkl'), 'wb') as f:
    pickle.dump(model, f)

with open(os.path.join(base_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(base_dir, 'label_encoder_brand.pkl'), 'wb') as f:
    pickle.dump(le_brand, f)

results = {
    'Decision Tree': {
        'test_accuracy': round(acc * 100, 2),
        'cv_accuracy':   round(cv.mean() * 100, 2),
        'cv_std':        round(cv.std() * 100, 2),
        'precision':     round(prec * 100, 2),
        'recall':        round(rec * 100, 2),
        'f1_score':      round(f1 * 100, 2),
        'description':   'Single tree (max_depth=5) using Gini impurity. Balanced between accuracy and probability calibration.',
        'scaled':        False,
    }
}

stats = {
    "best_model":     "Decision Tree",
    "best_accuracy":  round(cv.mean() * 100, 2),
    "dataset_size":   len(df),
    "feature_count":  len(FEATURE_COLS),
    "features":       FEATURE_LABELS,
    "classes":        CLASS_NAMES,
    "class_boundaries": {
        "Poor":      "score < 40",
        "Fair":      "40 <= score < 60",
        "Good":      "60 <= score < 80",
        "Excellent": "score >= 80"
    },
    "algorithms":     results,
    "feature_importance": {
        label: round(float(imp), 4)
        for label, imp in zip(FEATURE_LABELS, model.feature_importances_)
    },
    "algorithm_files": {
        "Decision Tree": "model_dt.pkl",
    }
}

with open(os.path.join(base_dir, 'model_stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)

print("\n" + "=" * 60)
print("  Done! Artifacts saved:")
print("    model_dt.pkl            (Decision Tree, max_depth=5)")
print("    scaler.pkl              (MinMaxScaler)")
print("    label_encoder_brand.pkl")
print("    model_stats.json")
print(f"\n  Model: Decision Tree ({cv.mean()*100:.1f}% CV accuracy)")
print(f"  Dataset: {len(df)} phones, {len(FEATURE_COLS)} features")
print("\n  Class bands (equal width, 20 pts each):")
print("    Poor     : score < 40  -> weight 0.25")
print("    Fair     : 40 - 60     -> weight 0.50")
print("    Good     : 60 - 80     -> weight 0.75")
print("    Excellent: 80+         -> weight 1.00")
print("=" * 60)
