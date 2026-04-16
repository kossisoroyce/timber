"""Sign model artifacts (files or directories) using Ed25519.

When signing a directory, all files are hashed recursively and the
combined manifest hash is signed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from timber.accel._util.crypto import KeyPair, sha256_digest, sign_data

logger = logging.getLogger(__name__)

# Default filenames for generated keys
_PRIVATE_KEY_NAME = "timber_signing.key"
_PUBLIC_KEY_NAME = "timber_signing.pub"
_SIGNATURE_SUFFIX = ".sig"


def generate_keypair(output_dir: str | Path | None = None) -> tuple[str, str]:
    """Generate an Ed25519 key pair for artifact signing.

    Args:
        output_dir: Directory to write the key files. Defaults to the
            current working directory.

    Returns:
        Tuple of (private_key_path, public_key_path).
    """
    output_dir = Path(output_dir) if output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    kp = KeyPair.generate()

    private_path = output_dir / _PRIVATE_KEY_NAME
    public_path = output_dir / _PUBLIC_KEY_NAME
    kp.save(private_path, public_path)

    # Restrict permissions on the private key
    os.chmod(private_path, 0o600)

    logger.info("Generated signing key pair in %s", output_dir)
    return str(private_path), str(public_path)


def sign_artifact(
    artifact_path: str | Path,
    key_path: str | Path | None = None,
) -> str:
    """Sign a file or directory artifact.

    If *artifact_path* is a directory, every file is hashed recursively
    (sorted by relative path) and the combined manifest hash is signed.

    Args:
        artifact_path: Path to the file or directory to sign.
        key_path: Path to the Ed25519 private key. When ``None``, looks
            for ``timber_signing.key`` in the current directory.

    Returns:
        Path to the generated ``.sig`` signature file.
    """
    artifact_path = Path(artifact_path)
    if os.path.islink(artifact_path):
        raise ValueError("Refusing to sign symlink target")
    artifact_path = artifact_path.resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    # Resolve private key
    if key_path is None:
        key_path = Path.cwd() / _PRIVATE_KEY_NAME
    key_path = Path(key_path)
    if not key_path.exists():
        raise FileNotFoundError(
            f"Signing key not found: {key_path}. "
            "Generate one with generate_keypair()."
        )

    kp = KeyPair.load(key_path)
    digest = _hash_artifact(artifact_path)
    signature = sign_data(digest, kp.private_key)

    sig_path = artifact_path.parent / (artifact_path.name + _SIGNATURE_SUFFIX)
    sig_path.write_bytes(signature)

    logger.info(
        "Signed %s -> %s (digest=%s)",
        artifact_path.name,
        sig_path.name,
        sha256_digest(digest),
    )
    return str(sig_path)


def _hash_artifact(path: str | Path) -> bytes:
    """Compute a deterministic hash of a file or directory.

    For a single file the SHA-256 digest of its contents is returned as
    raw bytes.  For a directory every file is visited in sorted order
    (by relative POSIX path) and the concatenation of
    ``relative_path + hex_digest`` lines is itself hashed.

    Returns:
        Raw SHA-256 digest bytes (32 bytes).
    """
    path = Path(path)

    if path.is_file():
        data = path.read_bytes()
        hex_digest = sha256_digest(data)
        return bytes.fromhex(hex_digest)

    # Directory: build a sorted manifest
    manifest_lines: list[str] = []
    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file():
            continue
        # Skip signature files themselves
        if file_path.suffix == _SIGNATURE_SUFFIX:
            continue
        rel = file_path.relative_to(path).as_posix()
        hex_digest = sha256_digest(file_path.read_bytes())
        manifest_lines.append(f"{rel}:{hex_digest}")

    if not manifest_lines:
        raise ValueError(f"No files found in directory: {path}")

    manifest = "\n".join(manifest_lines).encode()
    return bytes.fromhex(sha256_digest(manifest))
