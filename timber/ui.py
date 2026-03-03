"""Timber terminal UI — rich-based display components for a beautiful CLI experience."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console()


def header(subtitle: str = "") -> None:
    """Print the branded Timber header panel."""
    from timber import __version__

    t = Text()
    t.append("🌲 Timber ", style="bold green")
    t.append(f"v{__version__}", style="dim green")
    if subtitle:
        t.append(f"  —  {subtitle}", style="dim")
    console.print(Panel(t, border_style="green", padding=(0, 2)))
    console.print()


def section(title: str) -> None:
    """Print a labelled section rule."""
    console.print(Rule(f"  {title}  ", style="dim", align="left"))


def ok(label: str, detail: str = "") -> None:
    """Print a ✓ success line."""
    t = Text()
    t.append("  ✓ ", style="bold green")
    t.append(f"{label:<22}", style="bold")
    if detail:
        t.append(detail, style="dim")
    console.print(t)


def skip(label: str, detail: str = "") -> None:
    """Print a ─ skipped/neutral line."""
    t = Text()
    t.append("  ─ ", style="dim")
    t.append(f"{label:<22}", style="dim bold")
    if detail:
        t.append(detail, style="dim")
    console.print(t)


def fail(label: str, detail: str = "") -> None:
    """Print a ✗ failure line."""
    t = Text()
    t.append("  ✗ ", style="bold red")
    t.append(f"{label:<22}", style="bold red")
    if detail:
        t.append(detail, style="red")
    console.print(t)


def hint(message: str) -> None:
    """Print a dim helper hint line."""
    console.print(f"    [dim]{message}[/dim]")


def blank() -> None:
    console.print()


def error_panel(title: str, body: str, hint_text: str = "") -> None:
    """Print a red error panel with optional hint."""
    t = Text()
    t.append(f"{body}", style="red")
    if hint_text:
        t.append(f"\n\n{hint_text}", style="dim")
    console.print(Panel(t, title=f"[bold red]{title}[/bold red]", border_style="red", padding=(0, 2)))


def serving_panel(
    name: str,
    host: str,
    port: int,
    model_info: dict[str, Any],
) -> None:
    """Print the full server startup panel with endpoint list and example."""
    display_host = "localhost" if host in ("0.0.0.0", "") else host
    endpoint = f"http://{display_host}:{port}"
    framework = model_info.get("framework", "unknown")
    n_trees = model_info.get("n_trees", "?")
    n_features = model_info.get("n_features", "?")
    objective = model_info.get("objective", "unknown")

    t = Text()
    t.append("  Serving    ", style="bold green")
    t.append(f"{name}\n", style="bold white")
    t.append("\n  Endpoint   ", style="dim")
    t.append(endpoint, style="bold cyan underline")
    t.append("\n  Framework  ", style="dim")
    t.append(f"{framework}", style="white")
    t.append(f"  ·  {n_trees} trees  ·  {n_features} features", style="dim")
    t.append("\n  Objective  ", style="dim")
    t.append(objective, style="white")

    console.print(Panel(t, border_style="green", padding=(0, 1)))
    console.print()

    console.print("  [bold]API Endpoints[/bold]")
    console.print(f"    [green]POST[/green]  [cyan]{endpoint}/api/predict[/cyan]      [dim]run inference[/dim]")
    console.print(f"    [green]GET[/green]   [cyan]{endpoint}/api/models[/cyan]       [dim]list loaded models[/dim]")
    console.print(f"    [green]GET[/green]   [cyan]{endpoint}/api/model/:name[/cyan]  [dim]model metadata[/dim]")
    console.print(f"    [green]GET[/green]   [cyan]{endpoint}/api/health[/cyan]       [dim]health check[/dim]")
    console.print()
    _ex1 = Text(f"    curl {endpoint}/api/predict \\", style="dim")
    _ex2 = Text( "      -H 'Content-Type: application/json' \\", style="dim")
    _ex3 = Text(f"      -d '{{\"model\": \"{name}\", \"inputs\": [[1.0, 2.0, ...]]}}'" , style="dim")
    _stop = Text()
    _stop.append("  Press ", style="dim")
    _stop.append("Ctrl+C", style="bold dim")
    _stop.append(" to stop the server", style="dim")
    console.print("  [bold]Example[/bold]")
    console.print(_ex1)
    console.print(_ex2)
    console.print(_ex3)
    console.print()
    console.print(_stop)
    console.print()


def models_table(models: list[Any]) -> None:
    """Print a rich table of loaded models."""
    table = Table(
        show_header=True,
        header_style="bold dim",
        border_style="dim",
        box=None,
        padding=(0, 2),
        show_edge=False,
    )
    table.add_column("NAME", style="bold cyan", no_wrap=True)
    table.add_column("FRAMEWORK", style="")
    table.add_column("FORMAT", style="dim")
    table.add_column("TREES", justify="right")
    table.add_column("FEATURES", justify="right")
    table.add_column("SIZE", justify="right", style="dim")
    table.add_column("COMPILED", justify="center")

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

    console.print(table)


def inspect_panel(model_path: str, detected_fmt: str, ir_summary: dict[str, Any]) -> None:
    """Print a detailed model inspection summary."""
    section("Model Summary")
    console.print()
    ok("Source file", model_path)
    ok("Format", detected_fmt)
    ok("Framework", ir_summary.get("source_framework", ir_summary.get("framework", "unknown")))
    ok("Objective", ir_summary.get("objective", "unknown"))
    ok("Features", str(ir_summary.get("n_features", ir_summary.get("n_input_features", 0))))
    ok("Trees", str(ir_summary.get("n_trees", 0)))
    ok("Max depth", str(ir_summary.get("max_depth", 0)))
    ok("Total nodes", str(ir_summary.get("total_nodes", 0)))
    ok("Total leaves", str(ir_summary.get("total_leaves", 0)))
    ok("Classes", str(ir_summary.get("n_classes", 1)))

    total_nodes = ir_summary.get("total_nodes", 0)
    n_trees = ir_summary.get("n_trees", 0)
    est_data = total_nodes * 22
    est_code = n_trees * 500 + 4096
    console.print()
    section("Estimated Compiled Size")
    console.print()
    ok("Data segment", _fmt_bytes(est_data))
    ok("Code segment", _fmt_bytes(est_code))
    ok("Total", _fmt_bytes(est_data + est_code))


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    else:
        return f"{n / (1024 * 1024):.1f} MB"
