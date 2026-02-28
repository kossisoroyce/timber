from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from timber.runtime.predictor import TimberPredictor


WARMUP = 1000
ITERS = 10000


@dataclass
class BenchResult:
    name: str
    mean_us: float
    p50_us: float
    p95_us: float
    p99_us: float
    throughput_per_sec: float
    ran: bool
    note: str = ""



def _percentile(sorted_values: list[float], p: float) -> float:
    idx = int(len(sorted_values) * p)
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]



def _measure(fn: Callable[[], None], name: str) -> BenchResult:
    for _ in range(WARMUP):
        fn()

    latencies: list[float] = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1000.0)

    latencies.sort()
    mean_us = statistics.fmean(latencies)
    return BenchResult(
        name=name,
        mean_us=mean_us,
        p50_us=_percentile(latencies, 0.50),
        p95_us=_percentile(latencies, 0.95),
        p99_us=_percentile(latencies, 0.99),
        throughput_per_sec=1_000_000.0 / mean_us,
        ran=True,
    )



def _try_optional(name: str, runner: Callable[[], BenchResult]) -> BenchResult:
    try:
        return runner()
    except Exception as exc:
        return BenchResult(
            name=name,
            mean_us=0.0,
            p50_us=0.0,
            p95_us=0.0,
            p99_us=0.0,
            throughput_per_sec=0.0,
            ran=False,
            note=f"skipped: {type(exc).__name__}: {exc}",
        )



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/results.json")
    args = parser.parse_args()

    data = load_breast_cancer()
    X, y = data.data.astype(np.float32), data.target
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        objective="binary:logistic",
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    booster = model.get_booster()

    sample = np.ascontiguousarray(X_test[:1], dtype=np.float32)
    d_sample = xgb.DMatrix(sample)

    results: list[BenchResult] = []

    results.append(_measure(lambda: booster.predict(d_sample), "python_xgboost"))

    tmp_xgb_json = Path("benchmarks/.tmp_xgb_for_timber.json")
    booster.save_model(tmp_xgb_json)
    pred = TimberPredictor.from_model(tmp_xgb_json)
    results.append(_measure(lambda: pred.predict(sample), "timber_native"))
    pred.close()

    def run_onnx() -> BenchResult:
        import onnxruntime as ort  # type: ignore
        from skl2onnx import convert_sklearn  # type: ignore
        from skl2onnx.common.data_types import FloatTensorType  # type: ignore

        onx = convert_sklearn(model, initial_types=[("input", FloatTensorType([None, sample.shape[1]]))])
        sess = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        return _measure(lambda: sess.run(None, {input_name: sample}), "onnx_runtime")

    results.append(_try_optional("onnx_runtime", run_onnx))

    def run_treelite() -> BenchResult:
        import treelite_runtime  # type: ignore

        tmp_model = Path("benchmarks/.tmp_xgb.json")
        booster.save_model(tmp_model)
        predictor = treelite_runtime.Predictor(str(tmp_model))
        return _measure(lambda: predictor.predict(sample), "treelite_runtime")

    results.append(_try_optional("treelite_runtime", run_treelite))

    def run_lleaves() -> BenchResult:
        import lleaves  # type: ignore
        import lightgbm as lgb

        lgbm = lgb.LGBMClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)
        lgbm.fit(X_train, y_train)
        model_path = Path("benchmarks/.tmp_lgb.txt")
        lgbm.booster_.save_model(model_path)

        ll_model = lleaves.Model(str(model_path))
        ll_model.compile()
        return _measure(lambda: ll_model.predict(sample), "lleaves")

    results.append(_try_optional("lleaves", run_lleaves))

    out = {
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version.split()[0],
        },
        "methodology": {
            "dataset": "breast_cancer",
            "model": {
                "framework": "xgboost",
                "n_trees": 50,
                "max_depth": 4,
                "n_features": int(sample.shape[1]),
            },
            "warmup": WARMUP,
            "iters": ITERS,
            "metric": "single-sample in-process latency (microseconds)",
        },
        "results": [r.__dict__ for r in results],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
