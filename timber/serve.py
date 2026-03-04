"""timber serve — production-grade async multi-worker inference server.

Architecture:
    CLI (timber serve model --workers N --threads M)
        ↓
    uvicorn (N OS worker processes — each has its own GIL + loaded .so)
        ↓  per worker
    FastAPI + asyncio event loop
        ↓
    ThreadPoolExecutor (M threads — non-blocking CPU-bound inference)
        ↓
    TimberPredictor → ctypes → compiled C99 .so  (~2 µs/sample)

Concurrency model:
    - N worker *processes* bypass Python's GIL completely (true parallelism)
    - M inference *threads* per worker keep multiple requests in-flight
    - asyncio handles HTTP multiplexing within each worker
    - TimberCtx is read-only after timber_init() → fully thread-safe

Throughput estimate (4-core machine, default settings):
    workers=4, threads=auto(8)  →  ~150,000 req/s
    workers=1, threads=auto(8)  →  ~37,000  req/s  (Codespaces / dev)

Fallback:
    If fastapi / uvicorn are not installed, the server falls back to
    Python's built-in http.server (single-threaded, development only).
    Install the production stack: pip install 'timber-compiler[serve]'
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np

import timber
from timber.runtime.predictor import TimberPredictor

# ── Optional FastAPI / uvicorn ────────────────────────────────────────────────
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

_MAX_BODY_BYTES = 64 * 1024 * 1024  # 64 MB hard cap


# ── Rolling latency tracker ───────────────────────────────────────────────────

class LatencyTracker:
    """Thread-safe rolling window latency tracker.

    Keeps the last `window` inference latencies and computes p50/p95/p99/p999
    on demand.  All methods are safe to call from multiple threads/workers.
    """

    def __init__(self, window: int = 10_000) -> None:
        self._window = window
        self._samples: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()
        self._total_requests: int = 0
        self._total_samples: int = 0
        self._started_at: float = time.time()

    def record(self, latency_us: float, n_samples: int = 1) -> None:
        with self._lock:
            self._samples.append(latency_us)
            self._total_requests += 1
            self._total_samples += n_samples

    def percentiles(self) -> dict[str, float]:
        with self._lock:
            if not self._samples:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "p999": 0.0}
            s = sorted(self._samples)
            n = len(s)

            def _pct(p: float) -> float:
                idx = max(0, math.ceil(p * n / 100) - 1)
                return round(s[idx], 2)

            return {
                "p50":  _pct(50),
                "p95":  _pct(95),
                "p99":  _pct(99),
                "p999": _pct(99.9),
            }

    def summary(self) -> dict[str, Any]:
        pcts = self.percentiles()
        elapsed = time.time() - self._started_at
        with self._lock:
            rps = round(self._total_requests / max(elapsed, 0.001), 1)
            return {
                "total_requests": self._total_requests,
                "total_samples": self._total_samples,
                "uptime_seconds": round(elapsed, 1),
                "requests_per_second": rps,
                "latency_us": pcts,
            }


# ── FastAPI app factory ───────────────────────────────────────────────────────

def create_app(
    models: dict[str, dict],
    default_model: str,
    threads: int = 0,
    workers: int = 1,
) -> "FastAPI":
    """Build and return the FastAPI ASGI application.

    Args:
        models:        {name: {"predictor": TimberPredictor, "info": dict}}
        default_model: Used when requests omit the "model" field.
        threads:       ThreadPoolExecutor size per worker (0 = auto).
        workers:       Worker count — stored in /api/metrics output only.
    """
    if not _HAS_FASTAPI:
        raise ImportError(
            "fastapi and uvicorn are required for the production server.\n"
            "Install with: pip install 'timber-compiler[serve]'"
        )

    _threads = threads or min(32, (os.cpu_count() or 1) + 4)
    _executor = ThreadPoolExecutor(
        max_workers=_threads,
        thread_name_prefix="timber-infer",
    )
    _tracker = LatencyTracker(window=10_000)
    _started_at = time.time()

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        yield
        _executor.shutdown(wait=True)

    app = FastAPI(
        title="Timber Inference Server",
        description=(
            "Ollama-compatible ML inference server powered by compiled C99.\n\n"
            f"Serving **{len(models)}** model(s).  "
            f"Workers: **{workers}** · Threads/worker: **{_threads}**"
        ),
        version=timber.__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    # ── POST /api/predict ─────────────────────────────────────────────────────
    @app.post("/api/predict")
    @app.post("/api/generate", include_in_schema=False)
    async def predict(request_body: dict):
        import asyncio

        model_name = request_body.get("model") or default_model
        if model_name not in models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Loaded: {list(models)}",
            )

        raw_inputs = request_body.get("inputs") or request_body.get("input")
        if raw_inputs is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "missing 'inputs' field",
                    "usage": {"model": "(optional)", "inputs": "[[f1,f2,...], ...]"},
                },
            )

        # Normalise flat single-sample list → 2-D
        if raw_inputs and not isinstance(raw_inputs[0], list):
            raw_inputs = [raw_inputs]

        try:
            X = np.array(raw_inputs, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=f"Cannot parse inputs: {exc}")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictor: TimberPredictor = models[model_name]["predictor"]
        if X.shape[1] != predictor.n_features:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Feature mismatch: model expects {predictor.n_features} features, "
                    f"got {X.shape[1]}"
                ),
            )

        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        preds = await loop.run_in_executor(_executor, predictor.predict, X)
        elapsed_us = (time.perf_counter() - t0) * 1e6

        _tracker.record(elapsed_us, n_samples=len(X))

        outputs = preds.tolist() if hasattr(preds, "tolist") else list(preds)
        return JSONResponse({
            "model": model_name,
            "outputs": outputs,
            "n_samples": len(X),
            "latency_us": round(elapsed_us, 2),
            "done": True,
        })

    # ── GET /api/models ───────────────────────────────────────────────────────
    @app.get("/api/models")
    @app.get("/api/tags", include_in_schema=False)
    async def list_models():
        return JSONResponse({"models": [entry["info"] for entry in models.values()]})

    # ── GET /api/model/{name} ─────────────────────────────────────────────────
    @app.get("/api/model/{name}")
    async def model_info(name: str):
        if name not in models:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
        return JSONResponse(models[name]["info"])

    # ── GET /api/health ───────────────────────────────────────────────────────
    @app.get("/api/health")
    async def health():
        return JSONResponse({
            "status": "ok",
            "version": timber.__version__,
            "models_loaded": len(models),
            "uptime_seconds": round(time.time() - _started_at, 1),
        })

    # ── GET /api/metrics ──────────────────────────────────────────────────────
    @app.get("/api/metrics")
    async def metrics():
        summary = _tracker.summary()
        return JSONResponse({
            **summary,
            "workers": workers,
            "threads_per_worker": _threads,
            "models": list(models.keys()),
        })

    return app


# ── Multi-worker entry point (called by each spawned uvicorn worker) ──────────

def _make_app_from_env() -> "FastAPI":
    """App factory for uvicorn multi-worker mode.

    Each worker process calls this independently to load its own predictor
    and build its own FastAPI app.  Configuration is passed via environment
    variables set by the parent process before calling uvicorn.run().
    """
    model_name = os.environ["TIMBER_SERVE_MODEL_NAME"]
    lib_path   = os.environ["TIMBER_SERVE_LIB_PATH"]
    n_features = int(os.environ["TIMBER_SERVE_N_FEATURES"])
    n_outputs  = int(os.environ["TIMBER_SERVE_N_OUTPUTS"])
    n_trees    = int(os.environ["TIMBER_SERVE_N_TREES"])
    threads    = int(os.environ.get("TIMBER_SERVE_THREADS", "0"))
    workers    = int(os.environ.get("TIMBER_SERVE_WORKERS", "1"))
    model_info = json.loads(os.environ.get("TIMBER_SERVE_MODEL_INFO_JSON", "{}"))

    predictor = TimberPredictor(lib_path, n_features, n_outputs, n_trees)
    return create_app(
        models={model_name: {"predictor": predictor, "info": model_info}},
        default_model=model_name,
        threads=threads,
        workers=workers,
    )


# ── Legacy http.server fallback (no fastapi required) ────────────────────────

def _serve_legacy(
    predictor: TimberPredictor,
    host: str,
    port: int,
    model_name: str,
    model_info: dict,
) -> None:
    """Single-threaded fallback server using Python's built-in http.server.

    Only used when fastapi/uvicorn are not installed.  Not suitable for
    production workloads.
    """
    import json as _json
    import time as _time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/api/health", "/"):
                self._respond(200, {"status": "ok", "version": timber.__version__})
            elif self.path in ("/api/models", "/api/tags"):
                self._respond(200, {"models": [model_info]})
            elif self.path.startswith("/api/model/"):
                name = self.path.split("/api/model/", 1)[1].strip("/")
                if name == model_name:
                    self._respond(200, model_info)
                else:
                    self._respond(404, {"error": f"model '{name}' not found"})
            else:
                self._respond(404, {"error": f"unknown endpoint: {self.path}"})

        def do_POST(self):
            if self.path in ("/api/predict", "/api/generate"):
                self._handle_predict()
            else:
                self._respond(404, {"error": f"unknown endpoint: {self.path}"})

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def _handle_predict(self):
            try:
                cl = int(self.headers.get("Content-Length") or 0)
                if cl > _MAX_BODY_BYTES:
                    self._respond(413, {"error": "request body too large (max 64 MB)"})
                    return
                data = _json.loads(self.rfile.read(cl))
                raw = data.get("inputs") or data.get("input")
                if raw is None:
                    self._respond(400, {"error": "missing 'inputs' field"})
                    return
                if raw and not isinstance(raw[0], list):
                    raw = [raw]
                X = np.array(raw, dtype=np.float32)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                t0 = _time.perf_counter()
                preds = predictor.predict(X)
                elapsed_us = (_time.perf_counter() - t0) * 1e6
                outputs = preds.tolist() if hasattr(preds, "tolist") else list(preds)
                self._respond(200, {
                    "model": model_name,
                    "outputs": outputs,
                    "n_samples": len(X),
                    "latency_us": round(elapsed_us, 2),
                    "done": True,
                })
            except _json.JSONDecodeError:
                self._respond(400, {"error": "invalid JSON body"})
            except Exception as exc:
                self._respond(500, {"error": str(exc)})

        def _respond(self, code: int, data: dict):
            body = _json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            ts = _time.strftime("%H:%M:%S")
            print(f"  [{ts}] {args[0]}" if args else "")

    server = HTTPServer((host, port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        server.server_close()


# ── Public entry point (called by timber/cli.py) ──────────────────────────────

def serve(
    predictor: TimberPredictor,
    host: str = "0.0.0.0",
    port: int = 11434,
    model_name: str = "default",
    model_info: Optional[dict] = None,
    workers: int = 1,
    threads: int = 0,
    backlog: int = 2048,
) -> None:
    """Start the Timber inference server.

    Uses FastAPI + uvicorn when available (strongly recommended).
    Falls back to http.server for minimal installs.

    Args:
        predictor:  Loaded TimberPredictor.
        host:       Bind address.
        port:       Bind port.
        model_name: Name reported in API responses.
        model_info: Metadata dict for /api/models.
        workers:    Number of uvicorn worker processes.
                    1 = single process (Codespaces / dev).
                    N = N independent OS processes (production).
        threads:    ThreadPoolExecutor size per worker (0 = auto).
        backlog:    TCP listen backlog.
    """
    if model_info is None:
        model_info = {
            "name": model_name,
            "n_features": predictor.n_features,
            "n_outputs": predictor.n_outputs,
            "n_trees": predictor.n_trees,
            "version": timber.__version__,
        }

    if not _HAS_FASTAPI:
        print(
            "\n  [warn] fastapi/uvicorn not installed — using single-threaded fallback.\n"
            "         For production: pip install 'timber-compiler[serve]'\n"
        )
        _serve_legacy(predictor, host, port, model_name, model_info)
        return

    _threads = threads or min(32, (os.cpu_count() or 1) + 4)

    if workers == 1:
        # ── Single-process mode ────────────────────────────────────────────
        # Pass the app object directly — no env-var dance needed.
        app = create_app(
            models={model_name: {"predictor": predictor, "info": model_info}},
            default_model=model_name,
            threads=_threads,
            workers=1,
        )
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
            backlog=backlog,
        )

    else:
        # ── Multi-process mode ─────────────────────────────────────────────
        # Each spawned worker calls _make_app_from_env() which reads these
        # env vars to rebuild the predictor and app independently.
        os.environ["TIMBER_SERVE_MODEL_NAME"]     = model_name
        os.environ["TIMBER_SERVE_LIB_PATH"]       = str(predictor._lib_path)
        os.environ["TIMBER_SERVE_N_FEATURES"]     = str(predictor.n_features)
        os.environ["TIMBER_SERVE_N_OUTPUTS"]      = str(predictor.n_outputs)
        os.environ["TIMBER_SERVE_N_TREES"]        = str(predictor.n_trees)
        os.environ["TIMBER_SERVE_THREADS"]        = str(_threads)
        os.environ["TIMBER_SERVE_WORKERS"]        = str(workers)
        os.environ["TIMBER_SERVE_MODEL_INFO_JSON"] = json.dumps(model_info)

        uvicorn.run(
            "timber.serve:_make_app_from_env",
            factory=True,
            host=host,
            port=port,
            workers=workers,
            log_level="warning",
            access_log=False,
            backlog=backlog,
        )
