"""Air-gapped deployment bundle manifest."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from timber.accel._util.crypto import sha256_digest
from timber.accel.version import __version__


@dataclass
class BundleManifest:
    """Manifest for an air-gapped deployment bundle."""
    bundle_id: str
    created_at: str
    timber_accel_version: str = __version__
    files: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    total_size_bytes: int = 0

    def add_file(self, path: str, rel_path: str) -> None:
        data = Path(path).read_bytes()
        self.files.append({
            "path": rel_path,
            "sha256": sha256_digest(data),
            "size_bytes": len(data),
        })
        self.total_size_bytes += len(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "timber_accel_version": self.timber_accel_version,
            "total_size_bytes": self.total_size_bytes,
            "file_count": len(self.files),
            "files": self.files,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def write(self, path: str | Path) -> str:
        p = Path(path)
        p.write_text(self.to_json())
        return str(p)

    @classmethod
    def from_directory(
        cls, directory: str | Path, bundle_id: str, **metadata
    ) -> BundleManifest:
        d = Path(directory)
        manifest = cls(
            bundle_id=bundle_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        )
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.name != "manifest.json":
                manifest.add_file(str(f), str(f.relative_to(d)))
        return manifest
