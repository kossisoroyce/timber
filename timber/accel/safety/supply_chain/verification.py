"""Verify Ed25519 signatures on model artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

from timber.accel._util.crypto import load_public_key, verify_signature

from .signing import _hash_artifact

logger = logging.getLogger(__name__)


def verify_artifact(
    artifact_path: str | Path,
    signature_path: str | Path,
    public_key_path: str | Path,
) -> bool:
    """Verify the Ed25519 signature of a file or directory artifact.

    Args:
        artifact_path: Path to the signed file or directory.
        signature_path: Path to the ``.sig`` file produced by
            :func:`~timber_accel.safety.supply_chain.signing.sign_artifact`.
        public_key_path: Path to the Ed25519 public key.

    Returns:
        ``True`` if the signature is valid, ``False`` otherwise.
    """
    artifact_path = Path(artifact_path).resolve()
    signature_path = Path(signature_path)
    public_key_path = Path(public_key_path)

    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    if not signature_path.exists():
        raise FileNotFoundError(f"Signature file not found: {signature_path}")
    if not public_key_path.exists():
        raise FileNotFoundError(f"Public key not found: {public_key_path}")

    digest = _hash_artifact(artifact_path)
    signature = signature_path.read_bytes()
    public_key = load_public_key(public_key_path)

    valid = verify_signature(digest, signature, public_key)

    if valid:
        logger.info("Signature verification PASSED for %s", artifact_path.name)
    else:
        logger.warning("Signature verification FAILED for %s", artifact_path.name)

    return valid


def verify_artifact_bytes(
    data: bytes,
    signature: bytes,
    public_key_path: str | Path,
) -> bool:
    """Verify an Ed25519 signature over raw bytes.

    This is a convenience wrapper for verifying in-memory data without
    writing it to disk first.

    Args:
        data: The original bytes that were signed.
        signature: The Ed25519 signature to check.
        public_key_path: Path to the Ed25519 public key.

    Returns:
        ``True`` if the signature is valid, ``False`` otherwise.
    """
    public_key_path = Path(public_key_path)
    if not public_key_path.exists():
        raise FileNotFoundError(f"Public key not found: {public_key_path}")

    public_key = load_public_key(public_key_path)
    return verify_signature(data, signature, public_key)
