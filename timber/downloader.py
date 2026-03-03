"""Timber model downloader — HTTP(S) model download with progress reporting."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

URL_SCHEMES = {"http", "https"}


def is_url(s: str) -> bool:
    """Return True if s looks like an HTTP(S) URL."""
    try:
        r = urlparse(s)
        return r.scheme in URL_SCHEMES and bool(r.netloc)
    except Exception:
        return False


def _url_cache_key(url: str) -> str:
    """Deterministic short cache key from a URL."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _filename_from_url(url: str) -> str:
    """Extract a filename from a URL path, fallback to 'model'."""
    parsed = urlparse(url)
    name = parsed.path.rstrip("/").split("/")[-1]
    # Strip query params if they leaked into name
    name = name.split("?")[0]
    return name if name else "model"


def download_model(
    url: str,
    cache_root: Path,
    force: bool = False,
    console: Optional[object] = None,
) -> Path:
    """Download a model from a URL with a rich progress bar, caching locally.

    Args:
        url:        The HTTP(S) URL to download from.
        cache_root: Root cache directory (e.g. ~/.timber).
        force:      Re-download even if the file is already cached.
        console:    Rich Console instance for output (optional).

    Returns:
        Path to the downloaded (and cached) file.

    Raises:
        ValueError:              If URL scheme is not http/https.
        requests.HTTPError:      On a non-2xx HTTP response.
        requests.ConnectionError: If the host is unreachable.
    """
    import requests
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    con: Console = console or Console()  # type: ignore[assignment]

    parsed = urlparse(url)
    if parsed.scheme not in URL_SCHEMES:
        raise ValueError(
            f"Unsupported URL scheme '{parsed.scheme}'. Only http and https are supported."
        )

    cache_dir = cache_root / "cache" / _url_cache_key(url)
    filename = _filename_from_url(url)
    cached_path = cache_dir / filename

    if cached_path.exists() and not force:
        con.print(f"  [dim]Cached copy found:[/dim] [cyan]{cached_path}[/cyan]")
        return cached_path

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Write to a .part temp file; rename to final path only on success.
    # This ensures a failed/interrupted download never leaves a corrupt cache entry.
    part_path = cached_path.with_suffix(cached_path.suffix + ".part")

    try:
        with Progress(
            TextColumn("  [bold green]Downloading[/bold green]"),
            BarColumn(bar_width=38, complete_style="green", finished_style="bold green"),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            console=con,
            transient=False,
        ) as progress:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                task = progress.add_task("", total=total or None)
                with open(part_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))
    except Exception:
        if part_path.exists():
            part_path.unlink()
        raise

    part_path.rename(cached_path)
    return cached_path
