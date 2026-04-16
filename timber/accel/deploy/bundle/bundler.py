"""Air-gapped deployment bundle creator.

Creates tar.gz archives containing compiled C artifacts, optional source,
certification reports, an integrity manifest, and optional Ed25519 signature.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from timber.accel._util.crypto import KeyPair, sign_data
from timber.accel.deploy.bundle.manifest import BundleManifest
from timber.codegen.c99 import C99Emitter
from timber.frontends import parse_model

logger = logging.getLogger(__name__)


@dataclass
class BundleOptions:
    """Configuration for bundle creation."""

    model_path: str
    target: str = "generic"
    format_hint: str | None = None
    include_source: bool = False
    include_cert_report: bool = False
    cert_report_path: str | None = None
    sign: bool = False
    key_pair: KeyPair | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    output_dir: str = "."


class DeploymentBundler:
    """Creates air-gapped deployment bundles as signed tar.gz archives."""

    def __init__(self, options: BundleOptions) -> None:
        self._opts = options
        self._bundle_id = f"timber-bundle-{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def create(self) -> str:
        """Build the bundle and return the path to the tar.gz archive."""
        with tempfile.TemporaryDirectory(prefix="timber_bundle_") as staging:
            staging_path = Path(staging)

            # 1. Parse & compile the model
            ir = parse_model(self._opts.model_path, self._opts.format_hint)
            emitter = C99Emitter()
            c99_out = emitter.emit(ir)

            artifact_dir = staging_path / "artifacts"
            artifact_dir.mkdir()
            c99_out.write(str(artifact_dir))
            logger.info("Compiled C99 artifacts written to staging area")

            # 2. Optionally include source
            if self._opts.include_source:
                self._copy_source(staging_path)

            # 3. Optionally include certification report
            if self._opts.include_cert_report and self._opts.cert_report_path:
                self._copy_cert_report(staging_path)

            # 4. Write model metadata
            meta_dir = staging_path / "meta"
            meta_dir.mkdir()
            (meta_dir / "model_ir.json").write_text(ir.to_json())

            # 5. Create manifest
            manifest = BundleManifest.from_directory(
                staging_path,
                self._bundle_id,
                target=self._opts.target,
                **self._opts.metadata,
            )
            manifest.write(staging_path / "manifest.json")
            logger.info(
                "Manifest created: %d files, %d bytes",
                len(manifest.files),
                manifest.total_size_bytes,
            )

            # 6. Optional signature
            if self._opts.sign:
                self._sign_manifest(staging_path, manifest)

            # 7. Pack into tar.gz
            archive_path = self._pack(staging_path)

        return archive_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_path(path: str | Path) -> Path:
        """Reject paths that attempt directory traversal."""
        resolved = Path(path).resolve()
        if ".." in resolved.parts:
            raise ValueError(f"Path traversal detected: {path}")
        return resolved

    def _copy_source(self, staging: Path) -> None:
        src_dir = staging / "source"
        src_dir.mkdir()
        src = self._validate_path(self._opts.model_path)
        if src.is_file():
            shutil.copy2(src, src_dir / src.name)
        elif src.is_dir():
            shutil.copytree(src, src_dir / src.name)
        logger.info("Source included in bundle")

    def _copy_cert_report(self, staging: Path) -> None:
        report = self._validate_path(self._opts.cert_report_path)
        if not report.exists():
            logger.warning("Certification report not found: %s", report)
            return
        dest = staging / "cert"
        dest.mkdir()
        shutil.copy2(report, dest / report.name)
        logger.info("Certification report included")

    def _sign_manifest(self, staging: Path, manifest: BundleManifest) -> None:
        if self._opts.key_pair is None:
            logger.warning("Signing requested but no key pair provided; skipping")
            return
        manifest_bytes = manifest.to_json().encode()
        signature = sign_data(manifest_bytes, self._opts.key_pair.private_key)
        sig_path = staging / "manifest.sig"
        sig_path.write_bytes(signature)
        logger.info("Manifest signed (Ed25519)")

    def _pack(self, staging: Path) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_name = f"{self._bundle_id}_{ts}.tar.gz"
        out_dir = Path(self._opts.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        archive_path = out_dir / archive_name

        staging_resolved = staging.resolve()

        def _tar_filter(tarinfo):
            if tarinfo.issym() or tarinfo.islnk():
                return None  # skip symlinks
            return tarinfo

        with tarfile.open(archive_path, "w:gz") as tar:
            for item in sorted(staging.rglob("*")):
                # Ensure the file doesn't escape the staging directory
                item_resolved = item.resolve()
                if not str(item_resolved).startswith(str(staging_resolved)):
                    raise ValueError(f"Path traversal detected: {item}")
                arcname = str(item.relative_to(staging))
                tar.add(str(item), arcname=arcname, filter=_tar_filter)
        logger.info("Bundle archive created: %s", archive_path)
        return str(archive_path)
