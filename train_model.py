"""
train_model.py — Smart Market
Trains 6 Supervised Machine Learning Algorithms + Voting Ensemble

Run:
    python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
print("  Supervised Machine Learning Algorithms")
print("=" * 60)

df = pd.read_csv(dataset_path)
print(f"\n✅ Dataset loaded: {len(df)} phone records")
print(f"   Columns: {list(df.columns)}\n")

# ── FIX 1: Support both column names for the target score ───────
if 'recommendation_score' in df.columns:
    score_col = 'recommendation_score'
elif 'ml_score' in df.columns:
    score_col = 'ml_score'
else:
    raise ValueError("Dataset must have 'recommendation_score' or 'ml_score' column.")

print(f"   Using score column: '{score_col}'")
print(f"   Score range: {df[score_col].min()} – {df[score_col].max()}\n")

# ── FIX 2: Normalize used price to 0-1 (match what ml_helper.php sends) ──
# ml_helper.php sends: selling_price / 100000
# Dataset has raw prices like 7500, 18000 → divide by 100000
if df['normalized_used_price'].max() > 10:
    print("   ⚠️  Detected raw prices in normalized_used_price — normalizing by /100000")
    df['normalized_used_price'] = df['normalized_used_price'] / 100000

def score_to_class(score):
    score = float(score)
    if score >= 85:   return 3  # Excellent
    elif score >= 75: return 2  # Good
    elif score >= 65: return 1  # Fair
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

CLASS_NAMES = ['Poor (<65)', 'Fair (65-74)', 'Good (75-84)', 'Excellent (85+)']

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")

X = df[FEATURE_COLS].fillna(0)
y = df['rec_class']

print(f"Class distribution:\n{y.value_counts().sort_index().rename({0:'Poor',1:'Fair',2:'Good',3:'Excellent'})}\n")

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS)

X_train,  X_test,  y_train, y_test  = train_test_split(X,        y, test_size=0.2, random_state=42, stratify=y)
Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ALGORITHMS = {
    'Random Forest': {
        'model': RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=2, random_state=42),
        'scaled': False,
        'description': 'Ensemble of decision trees using bagging. Reduces overfitting via averaging.'
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(max_depth=10, min_samples_split=3, random_state=42),
        'scaled': False,
        'description': 'Single tree that splits data using Gini impurity. Highly interpretable.'
    },
    'K-Nearest Neighbors': {
        'model': KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='distance'),
        'scaled': True,
        'description': 'Classifies based on the 5 most similar phones in the training set.'
    },
    'Support Vector Machine': {
        'model': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
        'scaled': True,
        'description': 'Finds the optimal hyperplane that separates recommendation classes.'
    },
    'Naive Bayes': {
        'model': GaussianNB(var_smoothing=1e-9),
        'scaled': True,
        'description': 'Probabilistic classifier using Bayes theorem with feature independence assumption.'
    },
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        'scaled': True,
        'description': 'Linear classifier that estimates class probabilities.'
    },
}

print(f"{'Algorithm':<25} {'Test Acc':>9} {'CV Score':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 75)

results       = {}
trained_models = {}

for name, config in ALGORITHMS.items():
    m        = config['model']
    use_X    = X_scaled   if config['scaled'] else X
    use_Xtr  = Xs_train   if config['scaled'] else X_train
    use_Xte  = Xs_test    if config['scaled'] else X_test
    use_ytr  = ys_train   if config['scaled'] else y_train
    use_yte  = ys_test    if config['scaled'] else y_test

    m.fit(use_Xtr, use_ytr)
    y_pred = m.predict(use_Xte)

    acc  = accuracy_score(use_yte, y_pred)
    prec = precision_score(use_yte, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(use_yte, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(use_yte, y_pred, average='weighted', zero_division=0)
    cv   = cross_val_score(m, use_X, y, cv=skf, scoring='accuracy')

    print(f"{name:<25} {acc*100:>8.2f}% {cv.mean()*100:>8.2f}% {prec*100:>9.2f}% {rec*100:>7.2f}% {f1*100:>7.2f}%")

    results[name] = {
        'test_accuracy': round(acc*100, 2),
        'cv_accuracy':   round(cv.mean()*100, 2),
        'cv_std':        round(cv.std()*100, 2),
        'precision':     round(prec*100, 2),
        'recall':        round(rec*100, 2),
        'f1_score':      round(f1*100, 2),
        'description':   config['description'],
        'scaled':        config['scaled'],
    }
    trained_models[name] = m

print("\n🗳️  Training Voting Ensemble (RF + SVM + Logistic Regression)...")
ensemble = VotingClassifier(
    estimators=[
        ('rf',  RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
        ('lr',  LogisticRegression(max_iter=1000, random_state=42)),
    ],
    voting='soft'
)
ensemble.fit(Xs_train, ys_train)
ens_pred = ensemble.predict(Xs_test)
ens_acc  = accuracy_score(ys_test, ens_pred)
ens_cv   = cross_val_score(ensemble, X_scaled, y, cv=skf, scoring='accuracy').mean()

print(f"{'Voting Ensemble':<25} {ens_acc*100:>8.2f}% {ens_cv*100:>8.2f}%")

results['Voting Ensemble'] = {
    'test_accuracy': round(ens_acc*100, 2),
    'cv_accuracy':   round(ens_cv*100, 2),
    'cv_std':        0,
    'precision':     round(precision_score(ys_test, ens_pred, average='weighted', zero_division=0)*100, 2),
    'recall':        round(recall_score(ys_test, ens_pred, average='weighted', zero_division=0)*100, 2),
    'f1_score':      round(f1_score(ys_test, ens_pred, average='weighted', zero_division=0)*100, 2),
    'description':   'Combines RF + SVM + Logistic Regression votes for final prediction.',
    'scaled':        True,
}

best_name = max(results, key=lambda k: results[k]['cv_accuracy'])
best_acc  = results[best_name]['cv_accuracy']
print(f"\n🏆 Best Model: {best_name} (CV Accuracy: {best_acc:.2f}%)")

rf_model = trained_models['Random Forest']
print(f"\n📊 Feature Importance (Random Forest):")
feat_imp = sorted(zip(FEATURE_LABELS, rf_model.feature_importances_), key=lambda x: -x[1])
for feat, imp in feat_imp:
    print(f"   {feat:<20} {'█'*int(imp*40)} {imp:.3f}")

best_model = trained_models.get(best_name, ensemble)
use_Xte    = Xs_test if results[best_name]['scaled'] else X_test
use_yte    = ys_test if results[best_name]['scaled'] else y_test
print(f"\n📋 Classification Report ({best_name}):")
print(classification_report(use_yte, best_model.predict(use_Xte), target_names=CLASS_NAMES, zero_division=0))

# ── Save all models ──────────────────────────────────────────────
model_files = {
    'Random Forest':         'model_rf.pkl',
    'Decision Tree':         'model_dt.pkl',
    'K-Nearest Neighbors':   'model_knn.pkl',
    'Support Vector Machine':'model_svm.pkl',
    'Naive Bayes':           'model_nb.pkl',
    'Logistic Regression':   'model_lr.pkl',
}
for name, fname in model_files.items():
    with open(os.path.join(base_dir, fname), 'wb') as f:
        pickle.dump(trained_models[name], f)

with open(os.path.join(base_dir, 'model_ensemble.pkl'), 'wb') as f:
    pickle.dump(ensemble, f)

with open(os.path.join(base_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(base_dir, 'label_encoder_brand.pkl'), 'wb') as f:
    pickle.dump(le_brand, f)

stats = {
    "best_model":       best_name,
    "best_accuracy":    best_acc,
    "dataset_size":     len(df),
    "feature_count":    len(FEATURE_COLS),
    "features":         FEATURE_LABELS,
    "classes":          CLASS_NAMES,
    "algorithms":       results,
    "feature_importance": {
        label: round(float(imp), 4)
        for label, imp in zip(FEATURE_LABELS, rf_model.feature_importances_)
    },
    "algorithm_files": {
        "Random Forest":         "model_rf.pkl",
        "Decision Tree":         "model_dt.pkl",
        "K-Nearest Neighbors":   "model_knn.pkl",
        "Support Vector Machine":"model_svm.pkl",
        "Naive Bayes":           "model_nb.pkl",
        "Logistic Regression":   "model_lr.pkl",
        "Voting Ensemble":       "model_ensemble.pkl",
    }
}

with open(os.path.join(base_dir, 'model_stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)

print("\n" + "=" * 60)
print("  ✅ All models saved (.pkl)")
print("  ✅ Scaler saved (scaler.pkl)")
print("  ✅ Label encoder saved (label_encoder_brand.pkl)")
print("  ✅ Stats saved (model_stats.json)")
print(f"\n  🏆 Best: {best_name} ({best_acc:.1f}% CV accuracy)")
print(f"  📱 Dataset: {len(df)} phones, {len(FEATURE_COLS)} features")
print("=" * 60)
print("\n🚀 Done!")
