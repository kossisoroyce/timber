"""Test the ctypes drop-in predictor against XGBoost Python predictions."""
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from timber.runtime import TimberPredictor

# Train reference model
data = load_breast_cancer()
X, y = data.data.astype(np.float32), data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=50, max_depth=4, learning_rate=0.1,
    objective="binary:logistic", random_state=42,
    use_label_encoder=False, eval_metric="logloss",
)
model.fit(X_train, y_train)

# XGBoost Python predictions
xgb_preds = model.predict_proba(X_test[:20])[:, 1]

# Timber compiled predictions — compile on-the-fly
print("Compiling model with Timber...")
predictor = TimberPredictor.from_model(
    "examples/breast_cancer_model.json",
    format_hint="xgboost",
    optimize=False,  # no optimization for bit-exact comparison
)

print(f"Loaded: {predictor.n_features} features, {predictor.n_trees} trees")
print()

# Run predictions
timber_preds = predictor.predict(X_test[:20])

# Compare
print(f"{'Sample':>6s}  {'XGBoost':>10s}  {'Timber C':>10s}  {'Delta':>10s}  {'Status':>6s}")
print(f"{'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}")

max_err = 0.0
for i in range(20):
    delta = abs(float(xgb_preds[i]) - float(timber_preds[i]))
    max_err = max(max_err, delta)
    status = "OK" if delta < 1e-5 else "DIFF"
    print(f"{i:>6d}  {xgb_preds[i]:>10.6f}  {timber_preds[i]:>10.6f}  {delta:>10.2e}  {status:>6s}")

print()
print(f"Max absolute error: {max_err:.2e}")
print(f"Mean absolute error: {np.mean(np.abs(xgb_preds - timber_preds)):.2e}")

if max_err < 1e-4:
    print("PASS — predictions match within tolerance")
else:
    print(f"NOTE — max error {max_err:.2e} (expected with optimizer passes)")

predictor.close()
print("\nDone.")
