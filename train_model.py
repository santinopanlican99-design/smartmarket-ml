"""
train_model.py — Smart Market
Trains 6 Supervised Machine Learning Algorithms + Voting Ensemble

Run:
    cd ml
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

base_dir = os.path.dirname(__file__)
dataset_path = os.path.join(base_dir, 'dataset', 'phones_dataset.csv')

print("=" * 60)
print("  Smart Market — ML Model Training")
print("  Supervised Machine Learning Algorithms")
print("=" * 60)

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found: {dataset_path}")

df = pd.read_csv(dataset_path)
print(f"\n✅ Dataset loaded: {len(df)} phone records\n")


def score_to_class(score):
    if score < 65:
        return 0
    elif score < 75:
        return 1
    elif score < 85:
        return 2
    else:
        return 3


required_columns = [
    'device_brand',
    'ram',
    'internal_memory',
    'rear_camera_mp',
    'battery',
    'release_year',
    'battery_health',
    'condition_score',
    'normalized_used_price',
    'device_age',
    'ml_score'
]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in dataset: {missing_columns}")

df['rec_class'] = df['ml_score'].apply(score_to_class)

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
    'device_age'
]

FEATURE_LABELS = [
    'Brand',
    'RAM',
    'Storage',
    'Camera (MP)',
    'Battery (mAh)',
    'Release Year',
    'Battery Health',
    'Condition Score',
    'Used Price',
    'Device Age'
]

CLASS_NAMES = ['Poor (<65)', 'Fair (65-74)', 'Good (75-84)', 'Excellent (85+)']

X = df[FEATURE_COLS].fillna(0)
y = df['rec_class']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ALGORITHMS = {
    'Random Forest': {
        'model': RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=2,
            random_state=42
        ),
        'scaled': False,
        'description': 'Ensemble of decision trees using bagging. Reduces overfitting via averaging.'
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=3,
            random_state=42
        ),
        'scaled': False,
        'description': 'Single tree that splits data using Gini impurity. Highly interpretable.'
    },
    'K-Nearest Neighbors': {
        'model': KNeighborsClassifier(
            n_neighbors=5,
            metric='euclidean',
            weights='distance'
        ),
        'scaled': True,
        'description': 'Classifies based on the 5 most similar phones in the training set.'
    },
    'Support Vector Machine': {
        'model': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        ),
        'scaled': True,
        'description': 'Finds the optimal hyperplane that separates recommendation classes.'
    },
    'Naive Bayes': {
        'model': GaussianNB(var_smoothing=1e-9),
        'scaled': True,
        'description': 'Probabilistic classifier using Bayes theorem with feature independence assumption.'
    },
    'Logistic Regression': {
        'model': LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42
        ),
        'scaled': True,
        'description': 'Linear classifier that estimates class probabilities.'
    },
}

print(f"{'Algorithm':<25} {'Test Acc':>9} {'CV Score':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 75)

results = {}
trained_models = {}

for name, config in ALGORITHMS.items():
    model = config['model']
    use_X = X_scaled if config['scaled'] else X
    use_Xtr = Xs_train if config['scaled'] else X_train
    use_Xte = Xs_test if config['scaled'] else X_test
    use_ytr = ys_train if config['scaled'] else y_train
    use_yte = ys_test if config['scaled'] else y_test

    model.fit(use_Xtr, use_ytr)
    y_pred = model.predict(use_Xte)

    acc = accuracy_score(use_yte, y_pred)
    prec = precision_score(use_yte, y_pred, average='weighted', zero_division=0)
    rec = recall_score(use_yte, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(use_yte, y_pred, average='weighted', zero_division=0)

    cv_scores = cross_val_score(model, use_X, y, cv=skf, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    print(f"{name:<25} {acc*100:>8.2f}% {cv_mean*100:>8.2f}% {prec*100:>9.2f}% {rec*100:>7.2f}% {f1*100:>7.2f}%")

    results[name] = {
        'test_accuracy': round(acc * 100, 2),
        'cv_accuracy': round(cv_mean * 100, 2),
        'cv_std': round(cv_std * 100, 2),
        'precision': round(prec * 100, 2),
        'recall': round(rec * 100, 2),
        'f1_score': round(f1 * 100, 2),
        'description': config['description'],
        'scaled': config['scaled'],
    }
    trained_models[name] = model

print("\n🗳️  Training Voting Ensemble (RF + SVM + Logistic Regression)...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ],
    voting='soft'
)
ensemble.fit(Xs_train, ys_train)
ens_pred = ensemble.predict(Xs_test)
ens_acc = accuracy_score(ys_test, ens_pred)
ens_cv = cross_val_score(ensemble, X_scaled, y, cv=skf, scoring='accuracy').mean()

print(f"{'Voting Ensemble':<25} {ens_acc*100:>8.2f}% {ens_cv*100:>8.2f}%")

results['Voting Ensemble'] = {
    'test_accuracy': round(ens_acc * 100, 2),
    'cv_accuracy': round(ens_cv * 100, 2),
    'cv_std': 0,
    'precision': round(precision_score(ys_test, ens_pred, average='weighted', zero_division=0) * 100, 2),
    'recall': round(recall_score(ys_test, ens_pred, average='weighted', zero_division=0) * 100, 2),
    'f1_score': round(f1_score(ys_test, ens_pred, average='weighted', zero_division=0) * 100, 2),
    'description': 'Combines RF + SVM + Logistic Regression votes for final prediction.',
    'scaled': True,
}

best_name = max(results, key=lambda k: results[k]['cv_accuracy'])
best_acc = results[best_name]['cv_accuracy']
print(f"\n🏆 Best Model: {best_name} (CV Accuracy: {best_acc:.2f}%)")

rf_model = trained_models['Random Forest']
print(f"\n📊 Feature Importance (Random Forest):")
feat_imp = sorted(zip(FEATURE_LABELS, rf_model.feature_importances_), key=lambda x: -x[1])
for feat, imp in feat_imp:
    bar = '█' * int(imp * 40)
    print(f"   {feat:<20} {bar} {imp:.3f}")

best_model = trained_models.get(best_name, ensemble)
use_Xte = Xs_test if results[best_name]['scaled'] else X_test
use_yte = ys_test if results[best_name]['scaled'] else y_test

print(f"\n📋 Classification Report ({best_name}):")
print(classification_report(use_yte, best_model.predict(use_Xte), target_names=CLASS_NAMES, zero_division=0))

with open(os.path.join(base_dir, 'model_rf.pkl'), 'wb') as f:
    pickle.dump(trained_models['Random Forest'], f)

with open(os.path.join(base_dir, 'model_dt.pkl'), 'wb') as f:
    pickle.dump(trained_models['Decision Tree'], f)

with open(os.path.join(base_dir, 'model_knn.pkl'), 'wb') as f:
    pickle.dump(trained_models['K-Nearest Neighbors'], f)

with open(os.path.join(base_dir, 'model_svm.pkl'), 'wb') as f:
    pickle.dump(trained_models['Support Vector Machine'], f)

with open(os.path.join(base_dir, 'model_nb.pkl'), 'wb') as f:
    pickle.dump(trained_models['Naive Bayes'], f)

with open(os.path.join(base_dir, 'model_lr.pkl'), 'wb') as f:
    pickle.dump(trained_models['Logistic Regression'], f)

with open(os.path.join(base_dir, 'model_ensemble.pkl'), 'wb') as f:
    pickle.dump(ensemble, f)

with open(os.path.join(base_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(base_dir, 'label_encoder_brand.pkl'), 'wb') as f:
    pickle.dump(le_brand, f)

stats = {
    "best_model": best_name,
    "best_accuracy": best_acc,
    "dataset_size": len(df),
    "feature_count": len(FEATURE_COLS),
    "features": FEATURE_LABELS,
    "classes": CLASS_NAMES,
    "algorithms": results,
    "feature_importance": {
        label: round(float(imp), 4)
        for label, imp in zip(FEATURE_LABELS, rf_model.feature_importances_)
    },
    "algorithm_files": {
        "Random Forest": "model_rf.pkl",
        "Decision Tree": "model_dt.pkl",
        "K-Nearest Neighbors": "model_knn.pkl",
        "Support Vector Machine": "model_svm.pkl",
        "Naive Bayes": "model_nb.pkl",
        "Logistic Regression": "model_lr.pkl",
        "Voting Ensemble": "model_ensemble.pkl",
    }
}

with open(os.path.join(base_dir, 'model_stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)

print("\n" + "=" * 60)
print("  ✅ 6 models + ensemble saved (.pkl)")
print("  ✅ Scaler saved (scaler.pkl)")
print("  ✅ Label encoder saved (label_encoder_brand.pkl)")
print("  ✅ Stats saved (model_stats.json)")
print(f"\n  🏆 Best: {best_name} ({best_acc:.1f}% CV accuracy)")
print(f"  📱 Dataset: {len(df)} phones, {len(FEATURE_COLS)} features")
print("=" * 60)
print("\n🚀 Done!")