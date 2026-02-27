"""Pinpoint exact base_score handling for XGBoost binary:logistic."""
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

with open("examples/breast_cancer_model.json") as f:
    m = json.load(f)

stored_bs = float(m["learner"]["learner_model_param"]["base_score"])
trees = m["learner"]["gradient_booster"]["model"]["trees"]

sample = X_test[0]

# Manual tree traversal (matching XGBoost's < for left, >= for right)
def traverse(tree_data, sample):
    si = tree_data["split_indices"]
    sc = tree_data["split_conditions"]
    lc = tree_data["left_children"]
    rc = tree_data["right_children"]
    dl = tree_data.get("default_left", [1]*len(si))
    node = 0
    for _ in range(100):
        if lc[node] == -1:
            return sc[node]
        feat = si[node]
        val = sample[feat]
        if np.isnan(val):
            node = lc[node] if dl[node] else rc[node]
        elif val < sc[node]:
            node = lc[node]
        else:
            node = rc[node]
    return 0.0

tree_sum = sum(traverse(t, sample) for t in trees)

dtest = xgb.DMatrix(sample.reshape(1, -1))
xgb_margin = booster.predict(dtest, output_margin=True)[0]
xgb_prob = booster.predict(dtest)[0]

print(f"Stored base_score:           {stored_bs:.10f}")
print(f"logit(stored_bs):            {np.log(stored_bs/(1-stored_bs)):.10f}")
print(f"Tree outputs sum:            {tree_sum:.10f}")
print(f"XGBoost raw margin:          {xgb_margin:.10f}")
print(f"tree_sum + stored_bs:        {tree_sum + stored_bs:.10f}")
print(f"tree_sum + logit(stored_bs): {tree_sum + np.log(stored_bs/(1-stored_bs)):.10f}")
print()
diff_raw = xgb_margin - tree_sum
print(f"XGBoost margin - tree_sum:   {diff_raw:.10f}")
print(f"=> Implied base for margin:  {diff_raw:.10f}")
print()

# Test: what base_score gives the exact XGBoost result?
implied_base = xgb_margin - tree_sum
timber_prob_with_implied = 1.0 / (1.0 + np.exp(-(tree_sum + implied_base)))
timber_prob_with_stored = 1.0 / (1.0 + np.exp(-(tree_sum + stored_bs)))
timber_prob_with_logit = 1.0 / (1.0 + np.exp(-(tree_sum + np.log(stored_bs/(1-stored_bs)))))

print(f"XGBoost probability:                   {xgb_prob:.10f}")
print(f"Timber prob (implied base {implied_base:.4f}):  {timber_prob_with_implied:.10f}")
print(f"Timber prob (stored base {stored_bs:.4f}):   {timber_prob_with_stored:.10f}")
print(f"Timber prob (logit base {np.log(stored_bs/(1-stored_bs)):.4f}):   {timber_prob_with_logit:.10f}")
