"""Diagnose the numerical accuracy gap between XGBoost and Timber C inference."""
import json
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data.astype(np.float32), data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=50, max_depth=4, learning_rate=0.1,
    objective="binary:logistic", random_state=42,
    use_label_encoder=False, eval_metric="logloss",
)
model.fit(X_train, y_train)
booster = model.get_booster()

dtest = xgb.DMatrix(X_test[:1])
margin = booster.predict(dtest, output_margin=True)
prob = booster.predict(dtest)

with open("examples/breast_cancer_model.json") as f:
    m = json.load(f)

base_score = float(m["learner"]["learner_model_param"]["base_score"])

contrib = booster.predict(dtest, pred_contribs=True)
bias_term = contrib[0, -1]

print(f"Stored base_score:       {base_score:.10f}")
print(f"Bias from pred_contribs: {bias_term:.10f}")
print(f"XGBoost raw margin:      {margin[0]:.10f}")
print(f"XGBoost probability:     {prob[0]:.10f}")
print(f"sigmoid(margin):         {1/(1+np.exp(-margin[0])):.10f}")
print(f"Sum of contribs:         {contrib[0].sum():.10f}")
print(f"logit(base_score):       {np.log(base_score/(1-base_score)):.10f}")
