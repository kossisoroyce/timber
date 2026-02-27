"""Example client for a running Timber server.

Prerequisites:
    timber serve <model-name>   # start a server first

Usage:
    python examples/quickstart_client.py
"""

import json
import urllib.request
import numpy as np

BASE_URL = "http://localhost:11434"


def api_get(path: str) -> dict:
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def api_post(path: str, data: dict) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def main():
    # Health check
    health = api_get("/api/health")
    print(f"Health: {health}")

    # List models
    models = api_get("/api/models")
    print(f"\nLoaded models:")
    for m in models["models"]:
        print(f"  {m['name']}: {m['n_trees']} trees, {m['n_features']} features")

    if not models["models"]:
        print("\nNo models loaded. Run: timber load <model> --name <name>")
        return

    model_name = models["models"][0]["name"]
    n_features = models["models"][0]["n_features"]

    # Single prediction
    sample = np.random.randn(1, n_features).tolist()
    result = api_post("/api/predict", {"model": model_name, "inputs": sample})
    print(f"\nSingle prediction:")
    print(f"  Model:   {result['model']}")
    print(f"  Output:  {result['outputs']}")
    print(f"  Latency: {result['latency_us']}µs")

    # Batch prediction (10 samples)
    batch = np.random.randn(10, n_features).tolist()
    result = api_post("/api/predict", {"model": model_name, "inputs": batch})
    print(f"\nBatch prediction (10 samples):")
    print(f"  Outputs: {result['outputs'][:3]}... ({result['n_samples']} total)")
    print(f"  Latency: {result['latency_us']}µs")


if __name__ == "__main__":
    main()
