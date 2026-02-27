"""Train a real XGBoost model on the Breast Cancer dataset and save as JSON."""
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
data = load_breast_cancer()
X, y = data.data.astype(np.float32), data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    objective="binary:logistic",
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"Test accuracy: {acc:.4f}")

# Save model as JSON
model.get_booster().save_model("examples/breast_cancer_model.json")
print("Saved examples/breast_cancer_model.json")

# Save a few test samples as CSV for inference validation
header = ",".join(data.feature_names)
np.savetxt("examples/test_samples.csv", X_test[:10], delimiter=",", header=header, comments="")
print(f"Saved examples/test_samples.csv ({X_test[:10].shape[0]} samples)")

# Print reference predictions for comparison
preds = model.predict_proba(X_test[:10])[:, 1]
print("\nReference predictions (probability of class 1):")
for i, p in enumerate(preds):
    print(f"  sample {i}: {p:.6f}")
