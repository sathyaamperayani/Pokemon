"""
Task 3: Classification Model
Loads data/clean_data.csv, trains + compares two classifiers to predict
a Pokemon's primary_type from its base stats, and saves everything the
Streamlit dashboard will need.
"""

import json
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

df = pd.read_csv("data/clean_data.csv")

# ------------------------------------------------------------------
# STEP 1: Features and target
# ------------------------------------------------------------------
# Stats-only features: a user filling in the dashboard would plausibly
# know/guess these for a "new" Pokemon. secondary_type and n_types are
# left out on purpose -- they describe the Pokemon's typing itself,
# which is what we're trying to predict, so including them would leak
# target information rather than genuinely predict from stats.
FEATURES = [
    "height", "weight", "base_experience",
    "hp", "attack", "defense",
    "special_attack", "special_defense", "speed",
]
TARGET = "primary_type"

X = df[FEATURES]
y = df[TARGET]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ------------------------------------------------------------------
# STEP 2: Train/test split (stratified so rare classes appear in both)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ------------------------------------------------------------------
# STEP 3: Scale features
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# STEP 4: Train two models
# ------------------------------------------------------------------
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42
    ),
}

# ------------------------------------------------------------------
# STEP 5: Evaluate and compare
# ------------------------------------------------------------------
results = {}
fitted_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="macro", zero_division=0
    )

    results[name] = {
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
    }
    fitted_models[name] = (model, preds)

    print(f"{name:22s}  acc={acc:.3f}  macro_prec={prec:.3f}  "
          f"macro_rec={rec:.3f}  macro_f1={f1:.3f}")

# Pick the best model by macro F1 (fairer than accuracy given class imbalance)
best_name = max(results, key=lambda n: results[n]["macro_f1"])
best_model, best_preds = fitted_models[best_name]
print(f"\nBest model (by macro F1): {best_name}")

cm = confusion_matrix(y_test, best_preds, labels=range(len(le.classes_)))

# Feature importance: RandomForest gives it natively; for Logistic
# Regression we report the mean absolute coefficient across classes
# as a rough importance proxy.
if best_name == "random_forest":
    importances = dict(zip(FEATURES, best_model.feature_importances_.tolist()))
else:
    mean_abs_coef = np.abs(best_model.coef_).mean(axis=0)
    importances = dict(zip(FEATURES, mean_abs_coef.tolist()))

# ------------------------------------------------------------------
# STEP 6: Save everything the dashboard needs
# ------------------------------------------------------------------
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

with open("model_metrics.json", "w") as f:
    json.dump({
        "all_model_results": results,
        "best_model": best_name,
        "confusion_matrix": cm.tolist(),
        "class_labels": le.classes_.tolist(),
        "feature_importance": importances,
        "features": FEATURES,
    }, f, indent=2)

print("\nSaved model.pkl, scaler.pkl, label_encoder.pkl, model_metrics.json")
