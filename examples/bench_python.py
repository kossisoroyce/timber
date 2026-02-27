"""Benchmark XGBoost Python inference for comparison with Timber C99."""
import time
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

WARMUP = 1000
ITERS = 10000

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

samples = X_test[:10]

print("XGBoost Python Inference Benchmark")
print("=" * 50)
print(f"Trees:    {model.n_estimators}")
print(f"Features: {X_test.shape[1]}")
print(f"Samples:  {len(samples)}")
print(f"Warmup:   {WARMUP} iters")
print(f"Timed:    {ITERS} iters")
print("=" * 50)
print()

# --- Single-sample (batch=1) ---
single = np.ascontiguousarray(samples[:1])
d_single = xgb.DMatrix(single)

for _ in range(WARMUP):
    booster.predict(d_single)

latencies = []
for i in range(ITERS):
    t0 = time.perf_counter_ns()
    booster.predict(d_single)
    t1 = time.perf_counter_ns()
    latencies.append((t1 - t0) / 1000.0)  # to microseconds

latencies.sort()
mean = sum(latencies) / len(latencies)
p50 = latencies[len(latencies) // 2]
p95 = latencies[int(len(latencies) * 0.95)]
p99 = latencies[int(len(latencies) * 0.99)]

print("Single-sample (batch=1):")
print(f"  Mean:  {mean:.2f} us")
print(f"  P50:   {p50:.2f} us")
print(f"  P95:   {p95:.2f} us")
print(f"  P99:   {p99:.2f} us")
print(f"  Throughput: {1e6 / mean:.0f} samples/sec")
print()

# --- Batch inference ---
for bs in [1, 4, 10]:
    batch = np.ascontiguousarray(samples[:bs])
    d_batch = xgb.DMatrix(batch)

    for _ in range(WARMUP):
        booster.predict(d_batch)

    latencies = []
    for i in range(ITERS):
        t0 = time.perf_counter_ns()
        booster.predict(d_batch)
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1000.0)

    latencies.sort()
    mean = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]

    print(f"Batch={bs}:")
    print(f"  Mean:  {mean:.2f} us")
    print(f"  P50:   {p50:.2f} us")
    print(f"  P95:   {p95:.2f} us")
    print(f"  P99:   {p99:.2f} us")
    print(f"  Throughput: {bs * 1e6 / mean:.0f} samples/sec")
    print()
