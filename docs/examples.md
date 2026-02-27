# Examples

Real-world examples for each supported framework.

## XGBoost

```python
# examples/xgboost_example.py
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Train
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
model = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=42)
model.fit(X_train, y_train)
model.get_booster().save_model("xgb_model.json")
```

```bash
timber load xgb_model.json --name xgb-breast-cancer
timber serve xgb-breast-cancer
```

## LightGBM

```python
# examples/lightgbm_example.py
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
model = lgb.LGBMClassifier(n_estimators=50, max_depth=4, random_state=42)
model.fit(X_train, y_train)
model.booster_.save_model("lgb_model.txt")
```

```bash
timber load lgb_model.txt --name lgb-breast-cancer
timber serve lgb-breast-cancer
```

## scikit-learn

Timber supports `GradientBoosting`, `RandomForest`, `DecisionTree`, and `Pipeline` with `StandardScaler`.

```python
# examples/sklearn_example.py
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

# Pipeline with scaler — Timber fuses the scaler into tree thresholds
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42))
])
pipe.fit(X_train, y_train)

with open("sklearn_pipeline.pkl", "wb") as f:
    pickle.dump(pipe, f)
```

```bash
timber load sklearn_pipeline.pkl --name sklearn-pipeline
timber serve sklearn-pipeline
```

## CatBoost

```python
# examples/catboost_example.py
from catboost import CatBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
model = CatBoostClassifier(iterations=50, depth=4, random_seed=42, verbose=0)
model.fit(X_train, y_train)
model.save_model("catboost_model.json", format="json")
```

```bash
timber load catboost_model.json --name catboost-bc --format catboost
timber serve catboost-bc
```

> **Note:** CatBoost JSON uses the same `.json` extension as XGBoost. Use `--format catboost` to disambiguate, or Timber will auto-detect based on the `oblivious_trees` key.

## ONNX

```python
# examples/onnx_example.py
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

data = load_breast_cancer()
model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
model.fit(data.data, data.target)

initial_type = [("float_input", FloatTensorType([None, 30]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

```bash
timber load model.onnx --name onnx-bc
timber serve onnx-bc
```

## Python Client Example

Once a model is served, you can query it from any language. Here's a Python client:

```python
# examples/client.py
import requests
import numpy as np

# Generate sample input (30 features for breast cancer)
sample = np.random.randn(1, 30).tolist()

# Single prediction
resp = requests.post("http://localhost:11434/api/predict", json={
    "model": "xgb-breast-cancer",
    "inputs": sample
})
print(resp.json())
# {"model": "xgb-breast-cancer", "outputs": [0.97], "n_samples": 1, "latency_us": 91.0, "done": true}

# Batch prediction (10 samples)
batch = np.random.randn(10, 30).tolist()
resp = requests.post("http://localhost:11434/api/predict", json={
    "model": "xgb-breast-cancer",
    "inputs": batch
})
result = resp.json()
print(f"Predicted {result['n_samples']} samples in {result['latency_us']}µs")

# List models
resp = requests.get("http://localhost:11434/api/models")
for m in resp.json()["models"]:
    print(f"  {m['name']}: {m['n_trees']} trees, {m['n_features']} features")

# Health check
resp = requests.get("http://localhost:11434/api/health")
print(resp.json())  # {"status": "ok", "version": "0.1.0"}
```

## Using the Drop-in Predictor (No Server)

If you don't need HTTP, you can use `TimberPredictor` directly:

```python
from timber.runtime.predictor import TimberPredictor
import numpy as np

# Compile and predict in one step
pred = TimberPredictor.from_model("xgb_model.json")

X = np.random.randn(100, 30).astype(np.float32)
outputs = pred.predict(X)
print(f"Predicted {len(outputs)} samples")
print(f"  n_features: {pred.n_features}")
print(f"  n_outputs:  {pred.n_outputs}")
print(f"  n_trees:    {pred.n_trees}")
```

## Embedding in C

After `timber compile`, embed the generated code in your C project:

```c
// main.c
#include "model.h"
#include <stdio.h>

int main() {
    TimberCtx* ctx;
    timber_init(&ctx);

    float inputs[TIMBER_N_FEATURES] = {
        17.99, 10.38, 122.8, 1001.0, /* ... 30 features ... */
    };
    float outputs[TIMBER_N_OUTPUTS];

    int err = timber_infer_single(inputs, outputs, ctx);
    if (err != TIMBER_OK) {
        printf("Error: %s\n", timber_strerror(err));
        return 1;
    }

    printf("Prediction: %f\n", outputs[0]);
    timber_free(ctx);
    return 0;
}
```

```bash
gcc -O2 -o predict main.c model.c model_data.c -lm
./predict
```
