"""Encrypt and decrypt model artifacts with AES-256-GCM.

Encrypted file format:
    [12-byte nonce][ciphertext + GCM tag]

The GCM authentication tag is appended to the ciphertext automatically
by the ``cryptography`` library.
"""

from __future__ import annotations

import logging
from pathlib import Path

from timber.accel._util.crypto import decrypt_aes_gcm, encrypt_aes_gcm

logger = logging.getLogger(__name__)

_ENCRYPTED_SUFFIX = ".enc"
_NONCE_SIZE = 12  # AES-GCM standard nonce length


def encrypt_file(
    file_path: str | Path,
    key: str | Path,
    output_path: str | Path | None = None,
    aad: bytes | None = None,
) -> str:
    """Encrypt a file using AES-256-GCM.

    Args:
        file_path: Path to the plaintext file.
        key: Either a 64-character hex string representing a 32-byte key,
            or a path to a file containing the raw 32-byte key.
        output_path: Destination for the encrypted file. Defaults to
            ``<file_path>.enc``.

    Returns:
        Path to the encrypted output file.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    resolved_key = _resolve_key(key)
    plaintext = file_path.read_bytes()

    nonce, ciphertext = encrypt_aes_gcm(plaintext, resolved_key, aad=aad)

    if output_path is None:
        output_path = file_path.parent / (file_path.name + _ENCRYPTED_SUFFIX)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write nonce + ciphertext (tag is appended by the library)
    output_path.write_bytes(nonce + ciphertext)

    logger.info(
        "Encrypted %s -> %s (%d bytes)",
        file_path.name,
        output_path.name,
        output_path.stat().st_size,
    )
    return str(output_path)


def decrypt_file(
    file_path: str | Path,
    key: str | Path,
    output_path: str | Path | None = None,
    aad: bytes | None = None,
) -> str:
    """Decrypt an AES-256-GCM encrypted file.

    Args:
        file_path: Path to the encrypted file (nonce + ciphertext).
        key: Either a 64-character hex string representing a 32-byte key,
            or a path to a file containing the raw 32-byte key.
        output_path: Destination for the decrypted file. Defaults to
            the input path with the ``.enc`` suffix stripped (or
            ``<file_path>.dec`` if no ``.enc`` suffix exists).

    Returns:
        Path to the decrypted output file.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Encrypted file not found: {file_path}")

    resolved_key = _resolve_key(key)
    raw = file_path.read_bytes()

    if len(raw) < _NONCE_SIZE + 16:
        raise ValueError(
            "Encrypted file too short — must contain at least nonce + GCM tag"
        )

    nonce = raw[:_NONCE_SIZE]
    if len(nonce) != _NONCE_SIZE:
        raise ValueError(
            f"Extracted nonce must be exactly {_NONCE_SIZE} bytes, got {len(nonce)}"
        )
    ciphertext = raw[_NONCE_SIZE:]
    plaintext = decrypt_aes_gcm(nonce, ciphertext, resolved_key, aad=aad)

    if output_path is None:
        if file_path.suffix == _ENCRYPTED_SUFFIX:
            output_path = file_path.with_suffix("")
        else:
            output_path = file_path.parent / (file_path.name + ".dec")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(plaintext)

    logger.info(
        "Decrypted %s -> %s (%d bytes)",
        file_path.name,
        output_path.name,
        len(plaintext),
    )
    return str(output_path)


def _resolve_key(key: str | Path) -> bytes:
    """Resolve an AES-256 key from a hex string or file path.

    Args:
        key: A 64-character hex string (32 bytes) or a path to a file
            containing the raw 32-byte key.

    Returns:
        The 32-byte key.
    """
    # Try as hex string first
    if isinstance(key, str) and len(key) == 64:
        try:
            raw = bytes.fromhex(key)
            if len(raw) == 32:
                return raw
        except ValueError:
            raise ValueError(
                "Key looks like hex (64 chars) but contains invalid hex characters"
            )

    # Try as file path
    key_path = Path(key)
    if key_path.is_file():
        raw = key_path.read_bytes()
        if len(raw) != 32:
            raise ValueError(
                f"Key file must contain exactly 32 bytes, got {len(raw)}"
            )
        return raw

    raise ValueError(
        "Key must be a 64-character hex string or a path to a 32-byte key file"
    )
