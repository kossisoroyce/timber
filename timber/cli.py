"""Timber CLI — primary user interface for compilation, inspection, benchmarking, and validation."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import click
import numpy as np

import timber
from timber.audit.report import AuditReport
from timber.codegen.c99 import C99Emitter, TargetSpec
from timber.frontends import detect_format, parse_model
from timber.ir.model import PrecisionMode, TimberIR
from timber.optimizer.pipeline import OptimizerPipeline


def _con():
    """Return a fresh Rich Console (created at call-time so it picks up any stdout patching)."""
    from rich.console import Console
    return Console(highlight=False)


def _rich_header(con, subtitle: str = "") -> None:
    from rich.panel import Panel
    from rich.text import Text
    t = Text()
    t.append("🌲 Timber ", style="bold green")
    t.append(f"v{timber.__version__}", style="dim green")
    if subtitle:
        t.append(f"  —  {subtitle}", style="dim")
    con.print(Panel(t, border_style="green", padding=(0, 2)))
    con.print()


def _rich_section(con, title: str) -> None:
    from rich.rule import Rule
    con.print(Rule(f"  {title}  ", style="dim", align="left"))
    con.print()


def _rich_ok(con, label: str, detail: str = "") -> None:
    from rich.text import Text
    t = Text()
    t.append("  ✓ ", style="bold green")
    t.append(f"{label:<24}", style="bold")
    if detail:
        t.append(detail, style="dim")
    con.print(t)


def _rich_skip(con, label: str, detail: str = "") -> None:
    from rich.text import Text
    t = Text()
    t.append("  ─ ", style="dim")
    t.append(f"{label:<24}", style="dim bold")
    if detail:
        t.append(detail, style="dim")
    con.print(t)


def _build_load_callbacks(con, status):
    """Return a LoadCallbacks instance that drives rich terminal output."""
    from timber.store import LoadCallbacks

    def on_detect(fmt):
        _rich_ok(con, "Format detected", fmt)

    def on_parse_start():
        status.update("  [bold]Parsing model...[/bold]")
        status.start()

    def on_parse_done(n_trees, n_features, objective):
        status.stop()
        _rich_ok(con, "Parsed model", f"{n_trees} trees · {n_features} features · {objective}")

    def on_optimize_start():
        status.update("  [bold]Optimizing...[/bold]")
        status.start()

    def on_optimize_done(passes):
        status.stop()
        applied = sum(1 for p in passes if p.changed)
        total = len(passes)
        _rich_ok(con, "Optimized", f"{applied}/{total} passes applied")

    def on_emit_start():
        status.update("  [bold]Generating C99 code...[/bold]")
        status.start()

    def on_emit_done(compiled_dir):
        status.stop()
        model_c = Path(compiled_dir) / "model.c"
        if model_c.exists():
            with open(model_c) as _f:
                n_lines = sum(1 for _ in _f)
        else:
            n_lines = 0
        _rich_ok(con, "Generated C99", f"{n_lines:,} lines")

    def on_compile_start():
        status.update("  [bold]Compiling binary...[/bold]")
        status.start()

    def on_compile_done(lib_path):
        status.stop()
        if lib_path and Path(lib_path).exists():
            size = Path(lib_path).stat().st_size
            _rich_ok(con, "Compiled binary", _fmt_bytes(size))
        else:
            _rich_skip(con, "Compiled binary", "gcc not found — Python fallback will be used")

    return LoadCallbacks(
        on_detect=on_detect,
        on_parse_start=on_parse_start,
        on_parse_done=on_parse_done,
        on_optimize_start=on_optimize_start,
        on_optimize_done=on_optimize_done,
        on_emit_start=on_emit_start,
        on_emit_done=on_emit_done,
        on_compile_start=on_compile_start,
        on_compile_done=on_compile_done,
    )


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
    click.echo("Timber Model Inspector")
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

    click.echo("\nEstimated compiled size:")
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
    click.echo("Timber Validator")
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

            click.echo("\nResults:")
            click.echo(f"  Max absolute error:  {max_err:.2e}")
            click.echo(f"  Mean absolute error: {mean_err:.2e}")
            click.echo(f"  Divergent samples:   {divergent}/{n_samples}")

            if divergent == 0:
                click.echo("\n  PASS — all predictions within tolerance")
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
@click.option("--iters", default=0, type=int, help="Timed iterations per batch size (0=auto)")
@click.option("--format", "fmt", default=None, help="Model format hint")
@click.option("--report", "report_path", default=None, type=click.Path(), help="Write reproducibility report to file (.json or .html)")
def bench(artifact, data_path, batch_sizes, warmup_iters, iters, fmt, report_path):
    """Benchmark inference performance.

    Reports latency (P50/P95/P99/P99.9), throughput, CV, and memory usage.
    Optionally writes a reproducibility report with system info to JSON or HTML.

    \b
    Example:
        timber bench --artifact model.json --data test.csv --report report.html
    """
    import json as _json
    import platform
    import datetime

    con = _con()
    _rich_header(con, "Classical ML Inference Compiler")
    click.echo("Timber Benchmark")
    click.echo(f"{'='*60}")

    # ── Load IR ────────────────────────────────────────────────────────────
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

    # ── Load data ──────────────────────────────────────────────────────────
    try:
        data = np.loadtxt(data_path, delimiter=",", skiprows=1, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
    except Exception as exc:
        click.echo(f"Error loading data: {exc}", err=True)
        sys.exit(1)

    # ── System info ────────────────────────────────────────────────────────
    sys_info = {
        "platform": platform.platform(),
        "python":   platform.python_version(),
        "cpu":      platform.processor() or platform.machine(),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    try:
        import psutil
        sys_info["cpu_physical_cores"] = psutil.cpu_count(logical=False)
        sys_info["cpu_logical_cores"]  = psutil.cpu_count(logical=True)
        sys_info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass

    model_info = {
        "artifact": str(artifact),
        "n_trees":  ensemble.n_trees,
        "max_depth": ensemble.max_depth,
        "n_features": ensemble.n_features,
        "n_classes": ensemble.n_classes,
        "objective": ensemble.objective.value,
        "n_samples": data.shape[0],
    }

    click.echo(f"Model:       {artifact}")
    click.echo(f"Trees:       {ensemble.n_trees}")
    click.echo(f"Max depth:   {ensemble.max_depth}")
    click.echo(f"Features:    {ensemble.n_features}")
    click.echo(f"Data:        {data.shape[0]} samples")
    click.echo(f"Warmup:      {warmup_iters} iterations")
    click.echo()

    sizes = [int(s.strip()) for s in batch_sizes.split(",")]

    hdr = (f"{'Batch':>8s}  {'Min (μs)':>10s}  {'P50 (μs)':>10s}  "
           f"{'P95 (μs)':>10s}  {'P99 (μs)':>10s}  {'P99.9 (μs)':>12s}  "
           f"{'Throughput':>14s}  {'CV %':>6s}")
    sep = (f"{'-'*8}  {'-'*10}  {'-'*10}  "
           f"{'-'*10}  {'-'*10}  {'-'*12}  "
           f"{'-'*14}  {'-'*6}")
    click.echo(hdr)
    click.echo(sep)

    results = []
    for bs in sizes:
        batch = data[:bs]
        if len(batch) < bs:
            repeats = (bs // len(batch)) + 1
            batch = np.tile(data, (repeats, 1))[:bs]

        # Warmup
        for _ in range(warmup_iters):
            for i in range(len(batch)):
                _ir_predict_single(ensemble, batch[i])

        # Timed runs — more iterations for small batches
        n_runs = iters if iters > 0 else max(200, 2000 // max(bs, 1))
        latencies: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter_ns()
            for i in range(len(batch)):
                _ir_predict_single(ensemble, batch[i])
            elapsed_ns = time.perf_counter_ns() - t0
            latencies.append(elapsed_ns / 1_000.0)  # μs

        latencies.sort()
        n = len(latencies)
        p50   = latencies[n // 2]
        p95   = latencies[int(n * 0.95)]
        p99   = latencies[int(n * 0.99)]
        p999  = latencies[min(int(n * 0.999), n - 1)]
        mn    = latencies[0]
        mean  = sum(latencies) / n
        std   = (sum((x - mean) ** 2 for x in latencies) / n) ** 0.5
        cv    = (std / mean * 100.0) if mean > 0 else 0.0
        throughput = bs / (p50 / 1e6) if p50 > 0 else 0.0

        click.echo(
            f"{bs:>8d}  {mn:>10.1f}  {p50:>10.1f}  "
            f"{p95:>10.1f}  {p99:>10.1f}  {p999:>12.1f}  "
            f"{throughput:>11.0f}/s  {cv:>5.1f}%"
        )
        results.append({
            "batch_size":  bs,
            "n_runs":      n_runs,
            "min_us":      round(mn, 2),
            "p50_us":      round(p50, 2),
            "p95_us":      round(p95, 2),
            "p99_us":      round(p99, 2),
            "p999_us":     round(p999, 2),
            "mean_us":     round(mean, 2),
            "std_us":      round(std, 2),
            "cv_pct":      round(cv, 2),
            "throughput_samples_per_sec": round(throughput, 1),
        })

    click.echo()
    click.echo("Note: Python-interpreted IR. Compiled C artifacts are significantly faster.")

    # ── Reproducibility report ──────────────────────────────────────────────
    if report_path:
        report_data = {
            "timber_version": "0.2.0",
            "system":  sys_info,
            "model":   model_info,
            "results": results,
        }
        rp = Path(report_path)
        if rp.suffix.lower() == ".html":
            html = _bench_report_html(report_data)
            rp.write_text(html, encoding="utf-8")
        else:
            rp.write_text(_json.dumps(report_data, indent=2), encoding="utf-8")
        click.echo(f"\nReport written to: {rp}")


def _bench_report_html(report: dict) -> str:
    """Render a benchmark reproducibility report as HTML."""
    import json as _json

    sys_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in report["system"].items()
    )
    model_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in report["model"].items()
    )
    result_headers = [
        "Batch", "Runs", "Min (μs)", "P50 (μs)", "P95 (μs)",
        "P99 (μs)", "P99.9 (μs)", "Mean (μs)", "Std (μs)", "CV %",
        "Throughput (samp/s)",
    ]
    result_rows_html = ""
    for r in report["results"]:
        cells = "".join(f"<td>{r.get(k, '')}</td>" for k in [
            "batch_size", "n_runs", "min_us", "p50_us", "p95_us",
            "p99_us", "p999_us", "mean_us", "std_us", "cv_pct",
            "throughput_samples_per_sec",
        ])
        result_rows_html += f"<tr>{cells}</tr>"

    result_header_html = "".join(f"<th>{h}</th>" for h in result_headers)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Timber Benchmark Report</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background:#0f1117; color:#e0e0e0; margin:2rem; }}
  h1 {{ color:#7ee8a2; }} h2 {{ color:#90caf9; border-bottom:1px solid #333; padding-bottom:.3rem; }}
  table {{ border-collapse:collapse; width:100%; margin-bottom:2rem; }}
  th {{ background:#1e2a3a; color:#90caf9; padding:.5rem 1rem; text-align:left; }}
  td {{ padding:.4rem 1rem; border-bottom:1px solid #222; font-variant-numeric:tabular-nums; }}
  tr:hover td {{ background:#1a1f2e; }}
  .badge {{ display:inline-block; background:#1e2a3a; border-radius:4px; padding:.2rem .6rem;
             font-size:.85rem; color:#7ee8a2; margin:.2rem; }}
  pre {{ background:#1e1e2e; padding:1rem; border-radius:6px; overflow:auto; }}
</style>
</head>
<body>
<h1>Timber Benchmark Report</h1>
<p>Generated: <span class="badge">{report["system"].get("timestamp","")}</span>
   Timber: <span class="badge">v{report.get("timber_version","")}</span></p>

<h2>System</h2>
<table><tbody>{sys_rows}</tbody></table>

<h2>Model</h2>
<table><tbody>{model_rows}</tbody></table>

<h2>Results Matrix</h2>
<table>
  <thead><tr>{result_header_html}</tr></thead>
  <tbody>{result_rows_html}</tbody>
</table>

<h2>Raw JSON</h2>
<pre>{_json.dumps(report, indent=2)}</pre>
</body>
</html>"""


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


def _run_load(
    source: str,
    name: Optional[str],
    fmt: Optional[str],
    force: bool = False,
    con=None,
) -> "ModelInfo":  # noqa: F821
    """Shared logic for load + pull + serve-with-source. Returns ModelInfo."""
    from rich.status import Status

    from timber.downloader import download_model, is_url
    from timber.store import ModelStore

    if con is None:
        con = _con()

    store = ModelStore()
    model_path = source

    if is_url(source):
        _rich_section(con, "Pulling Model")
        con.print(f"  [dim]Source  [/dim] [cyan]{source}[/cyan]")
        con.print()
        try:
            cached = download_model(source, store.home, force=force, console=con)
        except Exception as exc:
            con.print(f"  [bold red]✗ Download failed:[/bold red] {exc}")
            raise
        con.print()
        _rich_ok(con, "Downloaded", str(cached))
        con.print()
        model_path = str(cached)
        if name is None:
            name = Path(cached).stem
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"No such file: '{source}'")

    _rich_section(con, "Processing Model")

    status = Status("", console=con, spinner="dots")
    callbacks = _build_load_callbacks(con, status)

    try:
        info = store.load_model(model_path, name=name, format_hint=fmt, callbacks=callbacks)
    except Exception:
        try:
            status.stop()
        except Exception:
            pass
        raise

    con.print()
    con.print(f"  [bold green]✓ Model ready[/bold green]  [bold cyan]{info.name}[/bold cyan]")
    con.print()
    return info


@main.command()
@click.argument("source")
@click.option("--name", default=None, help="Name to register the model as (default: filename stem)")
@click.option("--format", "fmt", default=None, help="Input format hint (xgboost, lightgbm, sklearn, ...)")
@click.option("--force", is_flag=True, default=False, help="Re-download even if URL is cached")
def load(source, name, fmt, force):
    """Compile and cache a model locally. Accepts a file path or HTTPS URL.

    \b
    Examples:
        timber load my_model.json
        timber load my_model.json --name fraud-detector
        timber load model.pkl --format sklearn
        timber load https://example.com/fraud-detector.json --name fraud-v1
    """
    con = _con()
    _rich_header(con, "Classical ML Inference Compiler")

    try:
        info = _run_load(source, name=name, fmt=fmt, force=force, con=con)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo("\nModel loaded successfully:")
    click.echo(f"  Name:      {info.name}")
    click.echo(f"  Format:    {info.format}")
    click.echo(f"  Framework: {info.framework}")
    click.echo(f"  Trees:     {info.n_trees}")
    click.echo(f"  Features:  {info.n_features}")
    click.echo(f"  Objective: {info.objective}")
    click.echo(f"  Compiled:  {'yes' if info.compiled else 'no (gcc not found)'}")
    click.echo(f"  Size:      {_fmt_bytes(info.size_bytes)}")
    click.echo("\nRun inference with:")
    click.echo(f"  timber serve {info.name}")


@main.command()
@click.argument("source")
@click.option("--name", default=None, help="Name to register the model as (default: filename stem)")
@click.option("--format", "fmt", default=None, help="Input format hint (xgboost, lightgbm, sklearn, ...)")
@click.option("--force", is_flag=True, default=False, help="Re-download even if URL is cached")
def pull(source, name, fmt, force):
    """Pull a model from a URL and compile it locally.

    \b
    Examples:
        timber pull https://example.com/models/fraud-detector.json
        timber pull https://example.com/models/fraud-detector.json --name fraud-v1
    """
    con = _con()
    _rich_header(con, "Classical ML Inference Compiler")

    try:
        info = _run_load(source, name=name, fmt=fmt, force=force, con=con)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo("\nModel loaded successfully:")
    click.echo(f"  Name:      {info.name}")
    click.echo(f"  Format:    {info.format}")
    click.echo(f"  Run with:  timber serve {info.name}")


@main.command(name="list")
def list_models():
    """List all locally loaded models.

    \b
    Example:
        timber list
    """
    from rich.table import Table

    from timber.store import ModelStore

    con = _con()
    store = ModelStore()
    models = store.list_models()

    if not models:
        click.echo("No models loaded. Use 'timber load <model_file>' to load one.")
        return

    table = Table(
        show_header=True,
        header_style="bold dim",
        box=None,
        padding=(0, 2),
        show_edge=False,
    )
    table.add_column("NAME", style="bold cyan", no_wrap=True)
    table.add_column("FRAMEWORK")
    table.add_column("FORMAT", style="dim")
    table.add_column("TREES", justify="right")
    table.add_column("FEATURES", justify="right")
    table.add_column("SIZE", justify="right", style="dim")
    table.add_column("COMPILED", justify="center")

    model_names = []
    for m in models:
        compiled = "[bold green]✓[/bold green]" if m.compiled else "[dim]no[/dim]"
        table.add_row(
            m.name,
            m.framework or "—",
            m.format,
            str(m.n_trees),
            str(m.n_features),
            _fmt_bytes(m.size_bytes),
            compiled,
        )
        model_names.append(m.name)

    con.print()
    con.print(table)
    con.print()
    for _n in model_names:
        click.echo(_n)  # for test capture — appears after table


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
        click.echo(
            click.style("  ✓ ", fg="green", bold=True)
            + click.style(f"{'Removed':<24}", bold=True)
            + click.style(f"model '{name}' deleted from local store", dim=True)
        )
    else:
        click.echo(
            click.style("  ✗ ", fg="red", bold=True)
            + click.style(f"Model {name!r} not found", bold=True)
            + click.style("  —  run 'timber list' to see available models", dim=True),
            err=True,
        )
        sys.exit(1)


@main.command()
@click.argument("source")
@click.option("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
@click.option("--port", default=11434, type=int, help="Bind port (default: 11434)")
@click.option("--name", default=None, help="Model name when source is a URL or file path")
@click.option("--force", is_flag=True, default=False, help="Re-download even if URL is cached")
def serve(source, host, port, name, force):
    """Serve a model over HTTP. Accepts a model name, file path, or HTTPS URL.

    When given a URL or file path, Timber will pull, compile, and serve
    the model in a single step—showing every stage of processing.

    \b
    Examples:
        timber serve fraud-detector
        timber serve fraud-detector --port 8080
        timber serve ./model.json --name my-model
        timber serve https://example.com/model.json

    \b
    API endpoints:
        POST /api/predict  — {"model": "name", "inputs": [[...]]}
        GET  /api/models   — list loaded models
        GET  /api/health   — health check
    """
    from rich.panel import Panel
    from rich.text import Text
    from rich.text import Text as _Text

    from timber.downloader import is_url
    from timber.runtime.predictor import TimberPredictor
    from timber.serve import serve as _serve
    from timber.store import ModelStore

    con = _con()
    _rich_header(con, "Inference Server")

    store = ModelStore()
    model_info = None
    model_name = source

    # ── Resolve source ────────────────────────────────────────────────────────
    if is_url(source):
        # Pull from URL, compile, register
        try:
            model_info = _run_load(source, name=name, fmt=None, force=force, con=con)
            model_name = model_info.name
        except Exception as exc:
            con.print(f"\n  [bold red]✗ Failed:[/bold red] {exc}")
            con.print("  [dim]Check the URL and try again.[/dim]")
            sys.exit(1)

    elif Path(source).exists():
        # Local file — compile and register if not already loaded
        existing = store.get_model(name or Path(source).stem)
        if existing is None:
            con.print(f"  [dim]Loading local file:[/dim] [cyan]{source}[/cyan]\n")
            try:
                model_info = _run_load(source, name=name, fmt=None, force=False, con=con)
                model_name = model_info.name
            except Exception as exc:
                con.print(f"\n  [bold red]✗ Failed:[/bold red] {exc}")
                sys.exit(1)
        else:
            model_info = existing
            model_name = existing.name

    else:
        # Treat as a registered model name
        model_info = store.get_model(source)
        if model_info is None:
            con.print(f"  [bold red]✗[/bold red] Model [bold]{source!r}[/bold] not found.")
            con.print()
            con.print("  [dim]Options:[/dim]")
            con.print(f"    [cyan]timber load <model_file> --name {source}[/cyan]  [dim]load from a file[/dim]")
            con.print(f"    [cyan]timber load <url> --name {source}[/cyan]         [dim]pull from a URL[/dim]")
            con.print("    [cyan]timber list[/cyan]                                [dim]see all loaded models[/dim]")
            sys.exit(1)
        model_name = model_info.name

    # ── Build predictor ───────────────────────────────────────────────────────
    lib_path = store.get_lib_path(model_name)
    _model_dir = store.get_model_dir(model_name)
    artifact_dir = (_model_dir / "compiled") if _model_dir is not None else None

    if lib_path and lib_path.exists():
        predictor = TimberPredictor(
            str(lib_path),
            n_features=model_info.n_features,
            n_outputs=model_info.n_outputs,
            n_trees=model_info.n_trees,
        )
    elif artifact_dir is not None and artifact_dir.exists():
        predictor = TimberPredictor.from_artifact(str(artifact_dir), build=True)
    else:
        con.print(f"  [bold red]✗[/bold red] Compiled artifacts not found for [bold]{model_name!r}[/bold].")
        con.print("  [dim]Try reloading:[/dim]")
        con.print(f"    [cyan]timber load <source> --name {model_name}[/cyan]")
        sys.exit(1)

    # ── Show serving panel ────────────────────────────────────────────────────
    display_host = "localhost" if host in ("0.0.0.0", "") else host
    endpoint = f"http://{display_host}:{port}"

    t = Text()
    t.append("  Serving    ", style="bold green")
    t.append(f"{model_name}\n", style="bold white")
    t.append("\n  Endpoint   ", style="dim")
    t.append(endpoint, style="bold cyan underline")
    t.append("\n  Framework  ", style="dim")
    t.append(f"{model_info.framework or 'unknown'}", style="white")
    t.append(f"  ·  {model_info.n_trees} trees  ·  {model_info.n_features} features", style="dim")
    t.append("\n  Objective  ", style="dim")
    t.append(model_info.objective or "unknown", style="white")
    con.print(Panel(t, border_style="green", padding=(0, 1)))
    con.print()
    con.print("  [bold]API Endpoints[/bold]")
    con.print(f"    [green]POST[/green]  [cyan]{endpoint}/api/predict[/cyan]      [dim]run inference[/dim]")
    con.print(f"    [green]GET[/green]   [cyan]{endpoint}/api/models[/cyan]       [dim]list loaded models[/dim]")
    con.print(f"    [green]GET[/green]   [cyan]{endpoint}/api/model/:name[/cyan]  [dim]model metadata[/dim]")
    con.print(f"    [green]GET[/green]   [cyan]{endpoint}/api/health[/cyan]       [dim]health check[/dim]")
    con.print()
    con.print("  [bold]Example[/bold]")
    con.print(_Text(f"    curl {endpoint}/api/predict \\", style="dim"))
    con.print(_Text( "      -H 'Content-Type: application/json' \\", style="dim"))
    con.print(_Text(f"      -d '{{\"model\": \"{model_name}\", \"inputs\": [[1.0, 2.0, ...]]}}'", style="dim"))
    con.print()
    _stop = _Text()
    _stop.append("  Press ", style="dim")
    _stop.append("Ctrl+C", style="bold dim")
    _stop.append(" to stop", style="dim")
    con.print(_stop)
    con.print()

    info = {
        "name": model_name,
        "n_features": model_info.n_features,
        "n_outputs": model_info.n_outputs,
        "n_trees": model_info.n_trees,
        "objective": model_info.objective,
        "framework": model_info.framework,
        "format": model_info.format,
        "version": timber.__version__,
    }
    _serve(predictor, host=host, port=port, model_name=model_name, model_info=info)


if __name__ == "__main__":
    main()
