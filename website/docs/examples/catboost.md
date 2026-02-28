---
sidebar_position: 4
title: CatBoost
---

# CatBoost Example

## Train and Save

```python
from catboost import CatBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = CatBoostClassifier(iterations=50, depth=4, random_seed=42, verbose=0)
model.fit(X_train, y_train)

# Must export as JSON for Timber
model.save_model("catboost_model.json", format="json")
```

## Load and Serve

CatBoost JSON shares the `.json` extension with XGBoost, so use `--format` to disambiguate:

```bash
timber load catboost_model.json --name catboost-bc --format catboost
```

Timber can also auto-detect CatBoost by looking for the `oblivious_trees` key in the JSON.

```bash
timber serve catboost-bc
```

## CatBoost Oblivious Trees

CatBoost uses **oblivious decision trees** — symmetric trees where every node at the same depth uses the same split feature and threshold. Timber's parser expands these into the general tree representation used by the IR, converting the bottom-up split ordering to the standard top-down format.
