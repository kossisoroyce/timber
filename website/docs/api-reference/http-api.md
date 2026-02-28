---
sidebar_position: 2
title: HTTP API
---

# HTTP API Reference

Default base URL: `http://localhost:11434`

## POST `/api/predict`

Run inference on a loaded model.

### Request

```json
{
  "model": "my-model",
  "inputs": [[1.0, 2.0, 3.0, ...]]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Name of the loaded model |
| `inputs` | float[][] | Yes | 2D array, shape `[n_samples, n_features]` |

### Response (200 OK)

```json
{
  "model": "my-model",
  "outputs": [0.97],
  "n_samples": 1,
  "latency_us": 91.0,
  "done": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Model name |
| `outputs` | float[] | Prediction values, length = n_samples × n_outputs |
| `n_samples` | int | Number of samples processed |
| `latency_us` | float | Inference latency in microseconds (C call only) |
| `done` | bool | Always `true` for non-streaming responses |

### Errors

```json
{"error": "model 'xyz' not loaded"}          // 404
{"error": "expected 30 features, got 10"}     // 400
{"error": "missing 'inputs' field"}           // 400
{"error": "missing 'model' field"}            // 400
```

### Batch Example

Send multiple samples in one request:

```bash
curl http://localhost:11434/api/predict \
  -d '{
    "model": "my-model",
    "inputs": [
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0]
    ]
  }'
```

---

## POST `/api/generate`

Alias for `/api/predict`. Provided for compatibility with Ollama client libraries.

---

## GET `/api/models`

List all loaded models with metadata.

### Response (200 OK)

```json
{
  "models": [
    {
      "name": "fraud-detector",
      "n_features": 30,
      "n_outputs": 1,
      "n_trees": 50,
      "objective": "binary:logistic",
      "framework": "xgboost",
      "format": "xgboost",
      "version": "0.1.0"
    }
  ]
}
```

---

## GET `/api/model/:name`

Get metadata for a specific model.

### Response (200 OK)

```json
{
  "name": "fraud-detector",
  "n_features": 30,
  "n_outputs": 1,
  "n_trees": 50,
  "objective": "binary:logistic",
  "framework": "xgboost"
}
```

### Error (404)

```json
{"error": "model 'xyz' not found"}
```

---

## GET `/api/health`

Health check endpoint.

### Response (200 OK)

```json
{"status": "ok", "version": "0.1.0"}
```
