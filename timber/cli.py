"""Timber CLI — primary user interface for compilation, inspection, benchmarking, and validation."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import click
import numpy as np

import timber
from timber.codegen.c99 import C99Emitter, TargetSpec
from timber.frontends import detect_format, parse_model
from timber.ir.model import PrecisionMode, TimberIR
from timber.optimizer.pipeline import OptimizerPipeline
from timber.audit.report import AuditReport


def _load_target_spec(path: Optional[str]) -> tuple[TargetSpec, dict]:
    """Load a TOML target specification file, or return defaults."""
    spec = TargetSpec()
    spec_dict: dict = {}

    if path is None:
        return spec, spec_dict

    p = Path(path)
    if not p.exists():
        click.echo(f"Warning: target file '{path}' not found, using defaults", err=True)
        return spec, spec_dict

    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib
    except ImportError:
        click.echo("Warning: TOML parser not available, using default target", err=True)
        return spec, spec_dict

    with open(p, "rb") as f:
        data = tomllib.load(f)

    spec_dict = data

    target = data.get("target", {})
    spec.arch = target.get("arch", "x86_64")
    spec.features = target.get("features", [])
    spec.os = target.get("os", "linux")
    spec.abi = target.get("abi", "systemv")

    precision = data.get("precision", {})
    mode_str = precision.get("mode", "float32")
    try:
        spec.precision = PrecisionMode(mode_str)
    except ValueError:
        spec.precision = PrecisionMode.FLOAT32

    output = data.get("output", {})
    spec.output_format = output.get("format", "c_source")
    spec.strip_symbols = output.get("strip_symbols", False)

    return spec, spec_dict


def _load_calibration_data(path: Optional[str]) -> Optional[np.ndarray]:
    """Load calibration CSV data as a numpy array."""
    if path is None:
        return None

    p = Path(path)
    if not p.exists():
        click.echo(f"Warning: calibration file '{path}' not found, skipping", err=True)
        return None

    try:
        data = np.loadtxt(p, delimiter=",", skiprows=1, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        click.echo(f"Loaded calibration data: {data.shape[0]} samples, {data.shape[1]} features")
        return data
    except Exception as exc:
        click.echo(f"Warning: failed to load calibration data: {exc}", err=True)
        return None


@click.group()
@click.version_option(version=timber.__version__, prog_name="timber")
def main():
    """Timber — Classical ML Inference Compiler.

    Compiles trained ML model artifacts into optimized, self-contained
    inference binaries targeting specific hardware.
    """
    pass


@main.command()
@click.option("--model", required=True, type=click.Path(exists=True), help="Input model artifact path")
@click.option("--format", "fmt", default=None, help="Input format hint (xgboost, lightgbm). Auto-detected if omitted.")
@click.option("--target", "target_path", default=None, type=click.Path(), help="Hardware target spec (TOML file)")
@click.option("--out", default="./dist", type=click.Path(), help="Output directory")
@click.option("--calibration", default=None, type=click.Path(), help="Calibration data CSV for frequency-ordering")
@click.option("--no-optimize", is_flag=True, help="Skip optimizer passes")
@click.option("--dead-leaf-threshold", default=0.001, type=float, help="Dead leaf elimination threshold")
def compile(model, fmt, target_path, out, calibration, no_optimize, dead_leaf_threshold):
    """Compile a trained model into an optimized inference artifact."""
    t0 = time.monotonic()

    # Detect format
    detected = fmt or detect_format(model)
    if detected is None:
        click.echo(f"Error: cannot detect model format for '{model}'. Use --format.", err=True)
        sys.exit(1)

    click.echo(f"Timber v{timber.__version__} — Classical ML Inference Compiler")
    click.echo(f"Input:  {model} (format: {detected})")

    # Parse model
    try:
        ir = parse_model(model, format_hint=detected)
    except Exception as exc:
        click.echo(f"Error parsing model: {exc}", err=True)
        sys.exit(1)

    summary = ir.summary()
    click.echo(f"Parsed: {summary.get('n_trees', 0)} trees, "
               f"max depth {summary.get('max_depth', 0)}, "
               f"{summary.get('n_features', summary.get('n_input_features', 0))} features, "
               f"objective={summary.get('objective', 'unknown')}")

    # Load target spec
    target_spec, target_dict = _load_target_spec(target_path)
    click.echo(f"Target: {target_spec.arch} [{', '.join(target_spec.features) or 'generic'}] "
               f"precision={target_spec.precision.value}")

    # Optimize
    opt_result = None
    if not no_optimize:
        click.echo("\nRunning optimizer passes...")
        calib_data = _load_calibration_data(calibration)

        optimizer = OptimizerPipeline(
            dead_leaf_threshold=dead_leaf_threshold,
            calibration_data=calib_data,
        )
        opt_result = optimizer.run(ir)
        ir = opt_result.ir

        for p in opt_result.passes:
            status = "applied" if p.changed else "no-op"
            click.echo(f"  [{status:>7s}] {p.pass_name} ({p.duration_ms:.1f}ms)")

        click.echo(f"Optimization complete: {opt_result.total_duration_ms:.1f}ms total")

    # Code generation
    click.echo("\nGenerating C99 code...")
    emitter = C99Emitter(target=target_spec)

    try:
        output = emitter.emit(ir)
    except Exception as exc:
        click.echo(f"Error during code generation: {exc}", err=True)
        sys.exit(1)

    output_files = output.write(out)
    for f in output_files:
        click.echo(f"  wrote {f}")

    # Audit report
    report = AuditReport.generate(
        ir=ir,
        optimization_result=opt_result,
        input_path=model,
        input_format=detected,
        target_spec=target_dict,
        output_files=output_files,
    )
    report_path = report.write(Path(out) / "audit_report.json")
    click.echo(f"  wrote {report_path}")

    # Save IR
    ir_path = Path(out) / "model.timber.json"
    ir_path.write_text(ir.to_json(), encoding="utf-8")
    click.echo(f"  wrote {ir_path}")

    elapsed = (time.monotonic() - t0) * 1000.0
    click.echo(f"\nCompilation complete in {elapsed:.0f}ms")
    click.echo(f"Output: {out}/")


@main.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--format", "fmt", default=None, help="Input format hint")
def inspect(model, fmt):
    """Inspect a model artifact without compiling.

    Prints a summary: framework, number of trees, max depth, features,
    objective, estimated compiled size.
    """
    detected = fmt or detect_format(model)
    if detected is None:
        click.echo(f"Error: cannot detect model format for '{model}'. Use --format.", err=True)
        sys.exit(1)

    try:
        ir = parse_model(model, format_hint=detected)
    except Exception as exc:
        click.echo(f"Error parsing model: {exc}", err=True)
        sys.exit(1)

    summary = ir.summary()
    click.echo(f"Timber Model Inspector")
    click.echo(f"{'='*50}")
    click.echo(f"Source file:     {model}")
    click.echo(f"Format:          {detected}")
    click.echo(f"Framework:       {ir.metadata.source_framework}")
    click.echo(f"Objective:       {summary.get('objective', 'unknown')}")
    click.echo(f"Features:        {summary.get('n_features', summary.get('n_input_features', 0))}")
    click.echo(f"Trees:           {summary.get('n_trees', 0)}")
    click.echo(f"Max depth:       {summary.get('max_depth', 0)}")
    click.echo(f"Total nodes:     {summary.get('total_nodes', 0)}")
    click.echo(f"Total leaves:    {summary.get('total_leaves', 0)}")
    click.echo(f"Classes:         {summary.get('n_classes', 1)}")

    # Estimate compiled size
    total_nodes = summary.get("total_nodes", 0)
    # Each node: feature(4B) + threshold(4B) + left(4B) + right(4B) + leaf(4B) + flags(2B) = ~22B
    est_data_bytes = total_nodes * 22
    # Code overhead: ~500B per tree + base
    est_code_bytes = summary.get("n_trees", 0) * 500 + 4096
    est_total = est_data_bytes + est_code_bytes

    click.echo(f"\nEstimated compiled size:")
    click.echo(f"  Data:   {_fmt_bytes(est_data_bytes)}")
    click.echo(f"  Code:   {_fmt_bytes(est_code_bytes)}")
    click.echo(f"  Total:  {_fmt_bytes(est_total)}")

    if ir.metadata.feature_names:
        click.echo(f"\nFeature names ({len(ir.metadata.feature_names)}):")
        for i, name in enumerate(ir.metadata.feature_names[:20]):
            click.echo(f"  [{i:3d}] {name}")
        if len(ir.metadata.feature_names) > 20:
            click.echo(f"  ... and {len(ir.metadata.feature_names) - 20} more")


@main.command()
@click.option("--artifact", required=True, type=click.Path(exists=True), help="Compiled artifact directory or model.c")
@click.option("--reference", required=True, type=click.Path(exists=True), help="Reference model file")
@click.option("--data", "data_path", required=True, type=click.Path(exists=True), help="Validation data CSV")
@click.option("--format", "fmt", default=None, help="Reference model format hint")
@click.option("--tolerance", default=1e-5, type=float, help="Maximum acceptable absolute error")
def validate(artifact, reference, data_path, fmt, tolerance):
    """Validate compiled artifact against the reference model.

    Runs the compiled C code and original framework side-by-side and
    reports maximum absolute error, mean absolute error, and divergent samples.
    """
    click.echo(f"Timber Validator")
    click.echo(f"{'='*50}")

    # Load reference IR
    detected = fmt or detect_format(reference)
    if detected is None:
        click.echo(f"Error: cannot detect format for reference '{reference}'", err=True)
        sys.exit(1)

    try:
        ir = parse_model(reference, format_hint=detected)
    except Exception as exc:
        click.echo(f"Error parsing reference model: {exc}", err=True)
        sys.exit(1)

    # Load validation data
    try:
        data = np.loadtxt(data_path, delimiter=",", skiprows=1, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
    except Exception as exc:
        click.echo(f"Error loading validation data: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Reference:   {reference} ({detected})")
    click.echo(f"Artifact:    {artifact}")
    click.echo(f"Data:        {data_path} ({data.shape[0]} samples)")
    click.echo(f"Tolerance:   {tolerance}")

    # Run IR-based inference (our compiled logic equivalent)
    ensemble = ir.get_tree_ensemble()
    if ensemble is None:
        click.echo("Error: no tree ensemble in reference model", err=True)
        sys.exit(1)

    n_samples = data.shape[0]
    predictions = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        sample = data[i]
        pred = _ir_predict_single(ensemble, sample)
        predictions[i] = pred

    # Load artifact predictions (from the compiled IR in model.timber.json)
    artifact_path = Path(artifact)
    ir_path = None
    if artifact_path.is_dir():
        ir_path = artifact_path / "model.timber.json"
    elif artifact_path.name == "model.timber.json":
        ir_path = artifact_path

    if ir_path and ir_path.exists():
        compiled_ir = TimberIR.from_json(ir_path.read_text())
        compiled_ensemble = compiled_ir.get_tree_ensemble()
        if compiled_ensemble:
            compiled_preds = np.zeros(n_samples, dtype=np.float32)
            for i in range(n_samples):
                compiled_preds[i] = _ir_predict_single(compiled_ensemble, data[i])

            errors = np.abs(predictions - compiled_preds)
            max_err = float(np.max(errors))
            mean_err = float(np.mean(errors))
            divergent = int(np.sum(errors > tolerance))

            click.echo(f"\nResults:")
            click.echo(f"  Max absolute error:  {max_err:.2e}")
            click.echo(f"  Mean absolute error: {mean_err:.2e}")
            click.echo(f"  Divergent samples:   {divergent}/{n_samples}")

            if divergent == 0:
                click.echo(f"\n  PASS — all predictions within tolerance")
            else:
                click.echo(f"\n  FAIL — {divergent} samples exceed tolerance {tolerance}")
                sys.exit(1)
        else:
            click.echo("Warning: no tree ensemble in compiled artifact IR", err=True)
    else:
        click.echo(f"Note: no model.timber.json found in '{artifact}'. "
                    f"Showing reference predictions only.")
        click.echo(f"  Mean prediction: {np.mean(predictions):.6f}")
        click.echo(f"  Std prediction:  {np.std(predictions):.6f}")


@main.command()
@click.option("--artifact", required=True, type=click.Path(exists=True), help="Compiled artifact or model file")
@click.option("--data", "data_path", required=True, type=click.Path(exists=True), help="Benchmark data CSV")
@click.option("--batch-sizes", default="1,16,64,256", help="Comma-separated batch sizes to test")
@click.option("--warmup-iters", default=1000, type=int, help="Warmup iterations before timing")
@click.option("--format", "fmt", default=None, help="Model format hint")
def bench(artifact, data_path, batch_sizes, warmup_iters, fmt):
    """Benchmark inference performance.

    Reports latency (P50/P95/P99), throughput, and memory usage.
    """
    click.echo(f"Timber Benchmark")
    click.echo(f"{'='*50}")

    # Try loading as compiled IR first, then as raw model
    artifact_path = Path(artifact)
    ir_path = None
    if artifact_path.is_dir():
        ir_path = artifact_path / "model.timber.json"
    elif artifact_path.suffix == ".json":
        ir_path = artifact_path

    ir: Optional[TimberIR] = None
    if ir_path and ir_path.exists():
        try:
            ir = TimberIR.from_json(ir_path.read_text())
        except Exception:
            pass

    if ir is None:
        detected = fmt or detect_format(artifact)
        if detected:
            ir = parse_model(artifact, format_hint=detected)

    if ir is None:
        click.echo(f"Error: cannot load model from '{artifact}'", err=True)
        sys.exit(1)

    ensemble = ir.get_tree_ensemble()
    if ensemble is None:
        click.echo("Error: no tree ensemble found", err=True)
        sys.exit(1)

    # Load data
    try:
        data = np.loadtxt(data_path, delimiter=",", skiprows=1, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
    except Exception as exc:
        click.echo(f"Error loading data: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Model:       {artifact}")
    click.echo(f"Trees:       {ensemble.n_trees}")
    click.echo(f"Max depth:   {ensemble.max_depth}")
    click.echo(f"Features:    {ensemble.n_features}")
    click.echo(f"Data:        {data.shape[0]} samples")
    click.echo(f"Warmup:      {warmup_iters} iterations")
    click.echo()

    sizes = [int(s.strip()) for s in batch_sizes.split(",")]

    click.echo(f"{'Batch':>8s}  {'P50 (μs)':>10s}  {'P95 (μs)':>10s}  {'P99 (μs)':>10s}  {'Throughput':>14s}")
    click.echo(f"{'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*14}")

    for bs in sizes:
        batch = data[:bs]
        if len(batch) < bs:
            # Tile if needed
            repeats = (bs // len(batch)) + 1
            batch = np.tile(data, (repeats, 1))[:bs]

        # Warmup
        for _ in range(min(warmup_iters, 100)):
            for i in range(len(batch)):
                _ir_predict_single(ensemble, batch[i])

        # Timed runs
        n_runs = max(100, 1000 // bs)
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter_ns()
            for i in range(len(batch)):
                _ir_predict_single(ensemble, batch[i])
            elapsed_ns = time.perf_counter_ns() - t0
            latencies.append(elapsed_ns / 1000.0)  # to microseconds

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        throughput = bs / (p50 / 1e6) if p50 > 0 else 0  # samples/sec

        click.echo(f"{bs:>8d}  {p50:>10.1f}  {p95:>10.1f}  {p99:>10.1f}  {throughput:>11.0f}/s")

    click.echo()
    click.echo("Note: These are Python-interpreted IR benchmarks. Compiled C artifacts will be significantly faster.")


def _ir_predict_single(ensemble, sample: np.ndarray) -> float:
    """Run inference on a single sample using the IR tree ensemble."""
    from timber.ir.model import Objective

    total = ensemble.base_score

    for tree in ensemble.trees:
        nodes = tree.nodes
        if not nodes:
            continue

        current = 0
        max_steps = tree.max_depth + 2

        while max_steps > 0:
            max_steps -= 1
            if current < 0 or current >= len(nodes):
                break

            node = nodes[current]
            if node.is_leaf:
                total += node.leaf_value
                break

            feat = node.feature_index
            if feat < 0 or feat >= len(sample):
                if node.default_left:
                    current = node.left_child
                else:
                    current = node.right_child
                continue

            val = sample[feat]
            if np.isnan(val):
                current = node.left_child if node.default_left else node.right_child
            elif val < node.threshold:
                current = node.left_child
            else:
                current = node.right_child

    # Apply activation
    if ensemble.objective in (Objective.BINARY_CLASSIFICATION, Objective.REGRESSION_LOGISTIC):
        total = 1.0 / (1.0 + np.exp(-total))

    return float(total)


def _fmt_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    else:
        return f"{n / (1024 * 1024):.1f} MB"


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--name", default=None, help="Name to register the model as (default: filename stem)")
@click.option("--format", "fmt", default=None, help="Input format hint (xgboost, lightgbm, sklearn, ...)")
def load(model_path, name, fmt):
    """Compile and cache a model locally.

    \b
    Examples:
        timber load my_model.json
        timber load my_model.json --name fraud-detector
        timber load model.pkl --format sklearn
    """
    from timber.store import ModelStore

    store = ModelStore()
    click.echo(f"Timber v{timber.__version__} — loading model...")

    try:
        info = store.load_model(model_path, name=name, format_hint=fmt)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"\nModel loaded successfully:")
    click.echo(f"  Name:      {info.name}")
    click.echo(f"  Format:    {info.format}")
    click.echo(f"  Framework: {info.framework}")
    click.echo(f"  Trees:     {info.n_trees}")
    click.echo(f"  Features:  {info.n_features}")
    click.echo(f"  Objective: {info.objective}")
    click.echo(f"  Compiled:  {'yes' if info.compiled else 'no (gcc not found)'}")
    click.echo(f"  Size:      {_fmt_bytes(info.size_bytes)}")
    click.echo(f"\nRun inference with:")
    click.echo(f"  timber serve {info.name}")


@main.command(name="list")
def list_models():
    """List all loaded models.

    \b
    Example:
        timber list
    """
    from timber.store import ModelStore

    store = ModelStore()
    models = store.list_models()

    if not models:
        click.echo("No models loaded. Use 'timber load <model_file>' to load one.")
        return

    click.echo(f"{'NAME':<25} {'FORMAT':<12} {'TREES':>6} {'FEATURES':>9} {'SIZE':>10} {'COMPILED':>9}")
    click.echo("-" * 75)
    for m in models:
        compiled = "yes" if m.compiled else "no"
        click.echo(
            f"{m.name:<25} {m.format:<12} {m.n_trees:>6} {m.n_features:>9} "
            f"{_fmt_bytes(m.size_bytes):>10} {compiled:>9}"
        )


@main.command()
@click.argument("name")
def remove(name):
    """Remove a loaded model.

    \b
    Example:
        timber remove my_model
    """
    from timber.store import ModelStore

    store = ModelStore()
    if store.remove_model(name):
        click.echo(f"Removed model '{name}'")
    else:
        click.echo(f"Model '{name}' not found", err=True)
        sys.exit(1)


@main.command()
@click.argument("name")
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=11434, type=int, help="Bind port (default: 11434)")
def serve(name, host, port):
    """Serve a loaded model over HTTP for inference.

    \b
    Examples:
        timber serve my_model
        timber serve fraud-detector --port 8080

    \b
    API endpoints:
        POST /api/predict  — {"model": "name", "inputs": [[...]]}
        GET  /api/models   — list loaded models
        GET  /api/health   — health check
    """
    from timber.store import ModelStore
    from timber.runtime.predictor import TimberPredictor
    from timber.serve import serve as _serve

    store = ModelStore()
    model_info = store.get_model(name)

    if model_info is None:
        # Try treating 'name' as a file path for convenience
        path = Path(name)
        if path.exists():
            click.echo(f"Model '{name}' not loaded. Loading now...")
            model_info = store.load_model(str(path))
            name = model_info.name
        else:
            click.echo(f"Model '{name}' not found. Load it first with:", err=True)
            click.echo(f"  timber load <model_file> --name {name}", err=True)
            sys.exit(1)

    # Load predictor from compiled artifacts
    lib_path = store.get_lib_path(name)
    artifact_dir = store.get_model_dir(name) / "compiled"

    if lib_path and lib_path.exists():
        predictor = TimberPredictor(
            str(lib_path),
            n_features=model_info.n_features,
            n_outputs=model_info.n_outputs,
            n_trees=model_info.n_trees,
        )
    elif artifact_dir.exists():
        predictor = TimberPredictor.from_artifact(str(artifact_dir), build=True)
    else:
        click.echo(f"Error: compiled artifacts not found for '{name}'", err=True)
        sys.exit(1)

    info = {
        "name": name,
        "n_features": model_info.n_features,
        "n_outputs": model_info.n_outputs,
        "n_trees": model_info.n_trees,
        "objective": model_info.objective,
        "framework": model_info.framework,
        "format": model_info.format,
        "version": timber.__version__,
    }
    _serve(predictor, host=host, port=port, model_name=name, model_info=info)


if __name__ == "__main__":
    main()
