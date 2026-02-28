---
sidebar_position: 3
title: scikit-learn
---

# scikit-learn Example

Timber supports scikit-learn tree estimators and pipelines, including `StandardScaler` fusion.

## Supported Estimators

- `GradientBoostingClassifier` / `GradientBoostingRegressor`
- `RandomForestClassifier` / `RandomForestRegressor`
- `HistGradientBoostingClassifier` / `HistGradientBoostingRegressor`
- `DecisionTreeClassifier` / `DecisionTreeRegressor`
- `Pipeline` with `StandardScaler` + any of the above

## Pipeline with StandardScaler

This is the most powerful use case — Timber **fuses the scaler into tree thresholds** at compile time, eliminating preprocessing entirely.

```python
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingClassifier(
        n_estimators=50, max_depth=3, random_state=42
    )),
])
pipe.fit(X_train, y_train)

with open("sklearn_pipeline.pkl", "wb") as f:
    pickle.dump(pipe, f)
```

## Load and Serve

```bash
timber load sklearn_pipeline.pkl --name sklearn-bc
timber serve sklearn-bc
```

## How Pipeline Fusion Works

When Timber encounters a `Pipeline(StandardScaler, TreeEstimator)`:

1. The parser emits a `ScalerStage` (with means μᵢ and scales σᵢ) followed by a `TreeEnsembleStage`
2. The **Pipeline Fusion** optimizer pass absorbs the scaler into tree thresholds:
   - New threshold: `θ' = θ × σ + μ`
3. The scaler stage is removed from the IR
4. The compiled code performs inference with **zero preprocessing overhead**

## Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
import pickle

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

with open("rf_model.pkl", "wb") as f:
    pickle.dump(model, f)
```

```bash
timber load rf_model.pkl --name random-forest
```
