---
sidebar_position: 1
title: XGBoost
---

# XGBoost Example

End-to-end: train an XGBoost model, compile it with Timber, and serve it.

## Train and Save

```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = xgb.XGBClassifier(
    n_estimators=50, max_depth=4,
    learning_rate=0.1, random_state=42,
)
model.fit(X_train, y_train)

# Save as JSON (required format for Timber)
model.get_booster().save_model("xgb_model.json")
```

## Load and Serve

```bash
timber load xgb_model.json --name xgb-bc
timber serve xgb-bc
```

## Query

```bash
curl http://localhost:11434/api/predict \
  -d '{"model": "xgb-bc", "inputs": [[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]]}'
```

## Important: base_score Handling

XGBoost stores `base_score` in **probability space** for logistic objectives (e.g., 0.5). Timber automatically converts this to **logit space** via `logit(p) = ln(p/(1-p))` for correct accumulation. This is critical for numerical accuracy — without it, predictions diverge significantly.

## Numerical Accuracy

On 114 test samples (breast cancer dataset):

| Metric | Value |
|--------|-------|
| Max absolute error | < 10⁻⁵ |
| Mean absolute error | < 10⁻⁶ |
| Classification agreement | 100% |
