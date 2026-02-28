---
sidebar_position: 2
title: Serving Models
---

# Serving Models

`timber serve` starts an HTTP server that exposes a compiled model over a REST API.

## Basic Usage

```bash
timber serve my-model
```

This starts the server on `0.0.0.0:11434` — the same default port as Ollama.

## Options

```bash
timber serve my-model --host 127.0.0.1 --port 8080
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `11434` | Bind port |

## Endpoints

### POST `/api/predict`

Run inference. Send a JSON body with `model` and `inputs`:

```bash
curl http://localhost:11434/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0]]
  }'
```

**Response:**

```json
{
  "model": "my-model",
  "outputs": [0.87],
  "n_samples": 1,
  "latency_us": 91.0,
  "done": true
}
```

### Batch Inference

Send multiple samples:

```bash
curl http://localhost:11434/api/predict \
  -d '{
    "model": "my-model",
    "inputs": [
      [1.0, 2.0, 3.0, 4.0, 5.0],
      [5.0, 4.0, 3.0, 2.0, 1.0],
      [2.5, 3.5, 1.5, 4.5, 0.5]
    ]
  }'
```

### GET `/api/models`

List all loaded models:

```bash
curl http://localhost:11434/api/models
```

### GET `/api/health`

Health check:

```bash
curl http://localhost:11434/api/health
# {"status": "ok", "version": "0.1.0"}
```

### POST `/api/generate`

Alias for `/api/predict` — for Ollama client compatibility.

## Architecture

The serving architecture separates concerns:

- **Python** handles: HTTP parsing, JSON serialization, request validation, CORS headers
- **Compiled C** handles: the actual inference computation via `ctypes`

This means Python is **never in the inference hot path**. The C function call takes ~2 µs; the total HTTP round-trip is ~91 µs.

## Error Handling

The server returns structured errors:

```json
{"error": "model 'xyz' not loaded"}
{"error": "expected 30 features, got 10"}
{"error": "missing 'inputs' field"}
```

## Production Deployment

The built-in server uses Python's `http.server`. For production at scale, front it with a reverse proxy:

```nginx
upstream timber {
    server 127.0.0.1:11434;
}

server {
    listen 80;
    location /api/ {
        proxy_pass http://timber;
    }
}
```
