---
sidebar_position: 2
title: LightGBM
---

# LightGBM Example

## Train and Save

```python
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = lgb.LGBMClassifier(
    n_estimators=50, max_depth=4,
    learning_rate=0.1, random_state=42, verbose=-1,
)
model.fit(X_train, y_train)

# Save as text model
model.booster_.save_model("lgb_model.txt")
```

## Load and Serve

```bash
timber load lgb_model.txt --name lgb-bc
timber serve lgb-bc
```

## Query

```bash
curl http://localhost:11434/api/predict \
  -d '{"model": "lgb-bc", "inputs": [[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]]}'
```

## LightGBM Format Details

Timber parses the LightGBM text model format, handling:

- Negative-indexed leaf references (LightGBM convention)
- Feature names and importance metadata
- Tree structure arrays (split features, thresholds, children, leaf values)

The parser validates the file and rejects empty or malformed models with descriptive error messages.
