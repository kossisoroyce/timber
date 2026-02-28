# Examples (Runnable End-to-End)

This folder contains runnable examples for XGBoost, LightGBM, and scikit-learn.

## Quickstart Scripts

Run from repo root:

```bash
python examples/quickstart_xgboost.py
python examples/quickstart_lightgbm.py
python examples/quickstart_sklearn.py
```

These scripts train models and write artifacts you can immediately load with Timber.

## Included Model Artifacts

The following generated model files are included for convenience so users can run Timber commands without retraining first:

- `xgb_breast_cancer.json`
- `lgb_breast_cancer.txt`
- `sklearn_pipeline.pkl`

## Run with Timber

```bash
# XGBoost
timber load examples/xgb_breast_cancer.json --name xgb-bc

# LightGBM
timber load examples/lgb_breast_cancer.txt --name lgb-bc

# scikit-learn Pipeline
timber load examples/sklearn_pipeline.pkl --name sklearn-bc

# Serve any loaded model
timber serve xgb-bc
```

## Query

```bash
curl http://localhost:11434/api/predict \
  -d '{"model": "xgb-bc", "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]]}'
```
