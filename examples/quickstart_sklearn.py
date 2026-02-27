"""Quickstart: Train a scikit-learn pipeline and serve it with Timber.

Demonstrates Pipeline(StandardScaler + GradientBoosting) — Timber fuses
the scaler into tree thresholds at compile time, eliminating the
preprocessing step at inference.

Usage:
    python examples/quickstart_sklearn.py
    timber load sklearn_pipeline.pkl --name sklearn-bc
    timber serve sklearn-bc
"""

import pickle
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data.astype(np.float32), data.target,
    test_size=0.2, random_state=42,
)

# 2. Train pipeline (scaler + gradient boosting)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )),
])
pipe.fit(X_train, y_train)

# 3. Evaluate
y_pred = pipe.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Pipeline: StandardScaler → GradientBoosting(50 trees, depth 3)")

# 4. Save for Timber
output_path = "sklearn_pipeline.pkl"
with open(output_path, "wb") as f:
    pickle.dump(pipe, f)
print(f"\nSaved to {output_path}")
print(f"\nNext steps:")
print(f"  timber load {output_path} --name sklearn-bc")
print(f"  timber serve sklearn-bc")
