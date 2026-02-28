---
sidebar_position: 6
title: HTTP Client
---

# HTTP Client Examples

Once a model is served with `timber serve`, you can query it from any language.

## Python

```python
import requests
import numpy as np

# Single prediction
sample = np.random.randn(1, 30).tolist()
resp = requests.post("http://localhost:11434/api/predict", json={
    "model": "my-model",
    "inputs": sample,
})
result = resp.json()
print(f"Output: {result['outputs']}, Latency: {result['latency_us']}µs")

# Batch prediction
batch = np.random.randn(100, 30).tolist()
resp = requests.post("http://localhost:11434/api/predict", json={
    "model": "my-model",
    "inputs": batch,
})
result = resp.json()
print(f"Predicted {result['n_samples']} samples in {result['latency_us']}µs")
```

## cURL

```bash
# Single prediction
curl -s http://localhost:11434/api/predict \
  -H "Content-Type: application/json" \
  -d '{"model": "my-model", "inputs": [[1.0, 2.0, 3.0]]}' | python -m json.tool

# Health check
curl -s http://localhost:11434/api/health

# List models
curl -s http://localhost:11434/api/models | python -m json.tool
```

## JavaScript / Node.js

```javascript
const resp = await fetch("http://localhost:11434/api/predict", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({
    model: "my-model",
    inputs: [[1.0, 2.0, 3.0, /* ... */]],
  }),
});
const result = await resp.json();
console.log(`Prediction: ${result.outputs}, Latency: ${result.latency_us}µs`);
```

## Go

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
)

func main() {
    body, _ := json.Marshal(map[string]interface{}{
        "model":  "my-model",
        "inputs": [][]float64{{1.0, 2.0, 3.0}},
    })

    resp, _ := http.Post(
        "http://localhost:11434/api/predict",
        "application/json",
        bytes.NewBuffer(body),
    )
    defer resp.Body.Close()

    var result map[string]interface{}
    json.NewDecoder(resp.Body).Decode(&result)
    fmt.Printf("Prediction: %v\n", result["outputs"])
}
```

## Rust

```rust
use reqwest;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let resp = client.post("http://localhost:11434/api/predict")
        .json(&json!({
            "model": "my-model",
            "inputs": [[1.0, 2.0, 3.0]]
        }))
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;

    println!("Prediction: {}", resp["outputs"]);
    Ok(())
}
```
