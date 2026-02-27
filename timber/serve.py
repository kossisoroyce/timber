"""timber serve — Ollama-style HTTP inference server for compiled Timber models.

Usage:
    timber load my_model.json              # compile & cache locally
    timber serve my_model --port 11434     # serve over HTTP
    timber list                            # show loaded models
    timber remove my_model                 # delete cached model

API (Ollama-compatible style):
    POST /api/predict      — {"model": "name", "inputs": [[...]]}
    GET  /api/models       — list loaded models
    GET  /api/model/:name  — model info
    GET  /api/health       — {"status": "ok"}
    POST /api/predict      — also accepts bare {"inputs": [...]} when single model served
"""

from __future__ import annotations

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

import numpy as np

from timber.runtime.predictor import TimberPredictor


_BANNER = r"""
  _____ _           _
 |_   _(_)_ __ ___ | |__   ___ _ __
   | | | | '_ ` _ \| '_ \ / _ \ '__|
   | | | | | | | | | |_) |  __/ |
   |_| |_|_| |_| |_|_.__/ \___|_|

  Classical ML Inference Server v0.1.0
"""


class TimberAPIHandler(BaseHTTPRequestHandler):
    """Ollama-style HTTP handler for Timber inference."""

    predictor: TimberPredictor
    model_name: str
    model_info: dict
    loaded_models: dict  # name -> {info, predictor}

    def do_GET(self):
        if self.path == "/api/health" or self.path == "/":
            self._json_response(200, {"status": "ok", "version": "0.1.0"})
        elif self.path == "/api/tags" or self.path == "/api/models":
            self._handle_list_models()
        elif self.path.startswith("/api/model/"):
            name = self.path.split("/api/model/", 1)[1].strip("/")
            self._handle_model_info(name)
        else:
            self._json_response(404, {"error": f"unknown endpoint: {self.path}"})

    def do_POST(self):
        if self.path == "/api/predict":
            self._handle_predict()
        elif self.path == "/api/generate":
            # Ollama compat alias
            self._handle_predict()
        else:
            self._json_response(404, {"error": f"unknown endpoint: {self.path}"})

    def _handle_list_models(self):
        models = []
        if hasattr(self, "loaded_models") and self.loaded_models:
            for name, entry in self.loaded_models.items():
                models.append(entry.get("info", {"name": name}))
        elif self.model_info:
            models.append(self.model_info)
        self._json_response(200, {"models": models})

    def _handle_model_info(self, name: str):
        if hasattr(self, "loaded_models") and name in self.loaded_models:
            self._json_response(200, self.loaded_models[name]["info"])
        elif name == self.model_name:
            self._json_response(200, self.model_info)
        else:
            self._json_response(404, {"error": f"model '{name}' not found"})

    def _handle_predict(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            # Resolve which predictor to use
            predictor = self.predictor
            model_name = data.get("model", self.model_name)

            if hasattr(self, "loaded_models") and model_name in self.loaded_models:
                predictor = self.loaded_models[model_name]["predictor"]

            inputs = data.get("inputs") or data.get("input")
            if inputs is None:
                self._json_response(400, {
                    "error": "missing 'inputs' field",
                    "usage": {"inputs": "[[feature_values], ...]", "model": "(optional) model name"},
                })
                return

            X = np.array(inputs, dtype=np.float32)
            if X.ndim == 1:
                X = X.reshape(1, -1)

            t0 = time.perf_counter()
            preds = predictor.predict(X)
            elapsed_us = (time.perf_counter() - t0) * 1e6

            outputs = preds.tolist() if preds.ndim > 1 else [float(p) for p in preds]

            result = {
                "model": model_name,
                "outputs": outputs,
                "n_samples": len(X),
                "latency_us": round(elapsed_us, 1),
                "done": True,
            }
            self._json_response(200, result)

        except json.JSONDecodeError:
            self._json_response(400, {"error": "invalid JSON body"})
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _json_response(self, code: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt, *args):
        # Concise request logging
        ts = time.strftime("%H:%M:%S")
        print(f"  [{ts}] {args[0]}" if args else "")


def serve(
    predictor: TimberPredictor,
    host: str = "0.0.0.0",
    port: int = 11434,
    model_name: str = "default",
    model_info: Optional[dict] = None,
):
    """Start the Ollama-style HTTP inference server."""
    if model_info is None:
        model_info = {
            "name": model_name,
            "n_features": predictor.n_features,
            "n_outputs": predictor.n_outputs,
            "n_trees": predictor.n_trees,
            "version": "0.1.0",
        }

    TimberAPIHandler.predictor = predictor
    TimberAPIHandler.model_name = model_name
    TimberAPIHandler.model_info = model_info
    TimberAPIHandler.loaded_models = {model_name: {"info": model_info, "predictor": predictor}}

    server = HTTPServer((host, port), TimberAPIHandler)
    print(_BANNER)
    print(f"  Listening on http://{host}:{port}")
    print(f"  Model:    {model_name}")
    print(f"  Trees:    {model_info.get('n_trees', '?')}")
    print(f"  Features: {model_info.get('n_features', '?')}")
    print()
    print(f"  Endpoints:")
    print(f"    POST /api/predict  — run inference")
    print(f"    GET  /api/models   — list models")
    print(f"    GET  /api/health   — health check")
    print()
    print(f"  Example:")
    print(f"    curl http://localhost:{port}/api/predict \\")
    print(f'      -d \'{{"model": "{model_name}", "inputs": [[1.0, 2.0, ...]]}}\'')
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()
