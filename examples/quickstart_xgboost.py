"""Quickstart: Train an XGBoost model and serve it with Timber.

Usage:
    python examples/quickstart_xgboost.py
    timber load xgb_breast_cancer.json --name breast-cancer
    timber serve breast-cancer
"""

import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data.astype(np.float32), data.target,
    test_size=0.2, random_state=42,
)

# 2. Train model
model = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
)
model.fit(X_train, y_train)

# 3. Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Trees: {model.n_estimators}, Features: {X_train.shape[1]}")

# 4. Save for Timber
output_path = "xgb_breast_cancer.json"
model.get_booster().save_model(output_path)
print(f"\nSaved to {output_path}")
print(f"\nNext steps:")
print(f"  timber load {output_path} --name breast-cancer")
print(f"  timber serve breast-cancer")
print(f"  curl http://localhost:11434/api/predict \\")
print(f"    -d '{{\"model\": \"breast-cancer\", \"inputs\": [{X_test[0].tolist()}]}}'")
