"""Timber Accelerate CLI — timber-accel command."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

from timber.accel.version import __version__

console = Console(stderr=True)


@click.group()
@click.version_option(__version__, prog_name="timber-accel")
def main():
    """Timber Accelerate — Hardware acceleration, safety, and deployment for Timber ML."""
    pass


@main.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True), help="Model artifact path.")
@click.option("--target", "-t", required=True, help="Target profile name or TOML path.")
@click.option("--out", "-o", default="./accel_out", help="Output directory.")
@click.option("--deterministic", is_flag=True, help="Apply deterministic transform.")
@click.option("--constant-time", is_flag=True, help="Apply constant-time transform.")
@click.option("--compliance", "-c", default=None, help="Compliance profile (do_178c, iso_26262, iec_62304).")
@click.option("--sign", is_flag=True, help="Sign the output artifact.")
@click.option("--verify", is_flag=True, help="Verify existing signature.")
@click.option("--fmt", default=None, help="Model format hint.")
@click.option("--no-optimize", is_flag=True, help="Skip Timber optimizer passes.")
def compile(model, target, out, deterministic, constant_time, compliance, sign, verify, fmt, no_optimize):
    """Compile model to accelerated C with optional safety transforms."""
    try:
        from timber.accel._util.target_loader import load_target_profile
        from timber.accel.accel.embedded.base import get_embedded_emitter
        from timber.accel.accel.gpu.base import get_gpu_emitter
        from timber.accel.accel.hls.base import get_hls_emitter
        from timber.accel.accel.simd.base import get_simd_emitter
        from timber.frontends import parse_model
        from timber.optimizer.pipeline import OptimizerPipeline

        console.print(f"[bold]timber-accel compile[/bold] v{__version__}")

        # Load target profile
        profile = load_target_profile(target)
        console.print(f"  Target: {profile.name} (arch={profile.target_spec.arch})")

        # Parse model
        ir = parse_model(model, format_hint=fmt)
        console.print(f"  Model: {model} ({ir.schema.n_features} features)")

        # Optimize
        if not no_optimize:
            opt = OptimizerPipeline()
            result = opt.run(ir)
            ir = result.ir
            console.print(f"  Optimizer: {len(result.passes)} passes")

        # Apply safety transforms
        if deterministic:
            from timber.accel.safety.realtime.deterministic import deterministic_pass
            changed, ir, details = deterministic_pass(ir)
            console.print(f"  Deterministic: {details.get('transforms_applied', 0)} transforms")

        if constant_time:
            from timber.accel.safety.realtime.constant_time import constant_time_pass
            changed, ir, details = constant_time_pass(ir)
            console.print(f"  Constant-time: {details.get('branches_removed', 0)} branches removed")

        # Select emitter
        if profile.is_simd:
            emitter = get_simd_emitter(profile)
        elif profile.is_gpu:
            emitter = get_gpu_emitter(profile)
        elif profile.is_hls:
            emitter = get_hls_emitter(profile)
        elif profile.is_embedded:
            emitter = get_embedded_emitter(profile)
        else:
            from timber.codegen.c99 import C99Emitter
            emitter = C99Emitter(target=profile.target_spec)

        # Emit code
        output = emitter.emit(ir)
        out_path = Path(out)
        out_path.mkdir(parents=True, exist_ok=True)
        files = output.write(str(out_path))
        console.print(f"  Output: {len(files)} files → {out_path}")

        # Compliance check
        if compliance:
            from timber.accel.safety.certification.profiles import check_compliance
            report = check_compliance(output.model_c, compliance)
            status = "[green]PASS[/green]" if report["compliant"] else "[red]FAIL[/red]"
            console.print(f"  Compliance ({compliance}): {status}")
            if not report["compliant"]:
                for v in report["violations"][:5]:
                    console.print(f"    - {v}")

        # Sign
        if sign:
            from timber.accel.safety.supply_chain.signing import sign_artifact
            sig_path = sign_artifact(str(out_path))
            console.print(f"  Signed: {sig_path}")

        console.print("[bold green]Done.[/bold green]")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True))
@click.option("--arch", "-a", required=True, help="Target architecture for WCET analysis.")
@click.option("--clock-mhz", default=100.0, help="Clock frequency in MHz.")
@click.option("--safety-margin", default=3.0, type=float, help="Safety margin multiplier for WCET estimate (default 3.0).")
@click.option("--fmt", default=None)
def wcet(model, arch, clock_mhz, safety_margin, fmt):
    """Analyze worst-case execution time."""
    if clock_mhz <= 0:
        click.echo("Error: --clock-mhz must be positive", err=True)
        sys.exit(1)

    from timber.accel.safety.realtime.wcet import DISCLAIMER, analyze_wcet
    from timber.frontends import parse_model

    ir = parse_model(model, format_hint=fmt)
    result = analyze_wcet(ir, arch=arch, clock_mhz=clock_mhz, safety_margin=safety_margin)

    console.print(f"[bold]WCET Analysis[/bold] — {arch} @ {clock_mhz} MHz (safety margin: {safety_margin}x)")
    console.print(f"  Raw cycles (worst): {result['raw_total_cycles_worst']}")
    console.print(f"  Cycles (worst):     {result['total_cycles_worst']}  (with {safety_margin}x margin)")
    console.print(f"  Time (worst):       {result['total_time_us_worst']:.2f} µs")
    console.print(f"  Raw cycles (avg):   {result['raw_total_cycles_avg']}")
    console.print(f"  Cycles (avg):       {result['total_cycles_avg']}  (with {safety_margin}x margin)")
    console.print(f"  Time (avg):         {result['total_time_us_avg']:.2f} µs")
    if result.get("per_stage"):
        for stage in result["per_stage"]:
            console.print(f"    [{stage['stage']}] {stage['cycles_worst']} cycles")
    console.print(f"\n[yellow]{DISCLAIMER}[/yellow]")


@main.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True))
@click.option("--profile", "-p", required=True, help="Compliance profile: do_178c, iso_26262, iec_62304.")
@click.option("--include-wcet", is_flag=True)
@click.option("--output", "-o", default=None)
@click.option("--fmt", default=None)
def certify(model, profile, include_wcet, output, fmt):
    """Generate certification report."""
    from timber.accel.safety.certification.report import generate_certification_report
    from timber.frontends import parse_model

    ir = parse_model(model, format_hint=fmt)
    report = generate_certification_report(ir, profile, include_wcet=include_wcet)

    if output:
        Path(output).write_text(report.to_json())
        console.print(f"Report written to {output}")
    else:
        console.print(report.summary())

    if include_wcet:
        from timber.accel.safety.realtime.wcet import DISCLAIMER as WCET_DISCLAIMER
        console.print(f"\n[yellow]{WCET_DISCLAIMER}[/yellow]")


@main.command("sign")
@click.option("--model", "-m", required=True, type=click.Path(exists=True))
@click.option("--key", "-k", default=None, help="Private key path.")
@click.option("--generate-key", is_flag=True, help="Generate a new key pair.")
def sign_cmd(model, key, generate_key):
    """Sign a model artifact."""
    from timber.accel.safety.supply_chain.signing import generate_keypair, sign_artifact

    if generate_key:
        priv, pub = generate_keypair()
        console.print(f"  Private key: {priv}")
        console.print(f"  Public key:  {pub}")
        key = priv

    if not key:
        console.print("[red]Error: --key or --generate-key required[/red]")
        sys.exit(1)

    sig_path = sign_artifact(model, key_path=key)
    console.print(f"Signature: {sig_path}")


@main.command("verify")
@click.option("--model", "-m", required=True, type=click.Path(exists=True))
@click.option("--sig", "-s", required=True, type=click.Path(exists=True), help="Signature file.")
@click.option("--key", "-k", required=True, type=click.Path(exists=True), help="Public key path.")
def verify_cmd(model, sig, key):
    """Verify a model artifact signature."""
    from timber.accel.safety.supply_chain.verification import verify_artifact

    valid = verify_artifact(model, sig, key)
    if valid:
        console.print("[bold green]Signature valid.[/bold green]")
    else:
        console.print("[bold red]Signature INVALID.[/bold red]")
        sys.exit(1)


@main.command("encrypt")
@click.option("--model", "-m", required=True, type=click.Path(exists=True))
@click.option("--key", "-k", required=True, help="256-bit hex key or path to key file.")
@click.option("--output", "-o", default=None)
def encrypt_cmd(model, key, output):
    """Encrypt a model artifact with AES-256-GCM."""
    from timber.accel.safety.supply_chain.encryption import encrypt_file

    out = encrypt_file(model, key, output_path=output)
    console.print(f"Encrypted: {out}")


@main.command("decrypt")
@click.option("--model", "-m", required=True, type=click.Path(exists=True))
@click.option("--key", "-k", required=True, help="256-bit hex key or path to key file.")
@click.option("--output", "-o", default=None)
def decrypt_cmd(model, key, output):
    """Decrypt an AES-256-GCM encrypted model."""
    from timber.accel.safety.supply_chain.encryption import decrypt_file

    out = decrypt_file(model, key, output_path=output)
    console.print(f"Decrypted: {out}")


@main.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True))
@click.option("--include-source", is_flag=True)
@click.option("--include-cert", is_flag=True)
@click.option("--output", "-o", default=None)
@click.option("--fmt", default=None)
def bundle(model, include_source, include_cert, output, fmt):
    """Create air-gapped deployment bundle."""
    from timber.accel.deploy.bundle.bundler import BundleOptions, DeploymentBundler

    options = BundleOptions(
        model_path=model,
        format_hint=fmt,
        include_source=include_source,
        include_cert_report=include_cert,
        output_dir=str(Path(output).parent) if output else ".",
    )
    bundler = DeploymentBundler(options)
    out = bundler.create()
    console.print(f"Bundle: {out}")


@main.command("serve-native")
@click.option("--model", "-m", required=True, type=click.Path(exists=True))
@click.option("--port", "-p", default=50051, type=int)
@click.option("--grpc", is_flag=True)
def serve_native(model, port, grpc):
    """Generate C++ gRPC inference server."""
    if not (1 <= port <= 65535):
        click.echo("Error: --port must be between 1 and 65535", err=True)
        sys.exit(1)

    from timber.accel.deploy.serve_native.server_gen import NativeServerGenerator, ServerConfig
    from timber.frontends import parse_model

    ir = parse_model(model)
    config = ServerConfig(
        model_name=Path(model).stem,
        ir=ir,
        mode="grpc" if grpc else "http",
        port=port,
    )
    gen = NativeServerGenerator(config)
    out = gen.generate(Path(model).parent / "serve_native")
    console.print(f"Server generated: {out}")
