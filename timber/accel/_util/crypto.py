"""Shared cryptographic utilities for TimberAccelerate."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


@dataclass
class KeyPair:
    """Ed25519 key pair."""
    private_key: Ed25519PrivateKey
    public_key: Ed25519PublicKey

    @classmethod
    def generate(cls) -> KeyPair:
        private = Ed25519PrivateKey.generate()
        return cls(private_key=private, public_key=private.public_key())

    def private_bytes(self) -> bytes:
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def public_bytes(self) -> bytes:
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def save(self, private_path: str | Path, public_path: str | Path) -> None:
        Path(private_path).write_bytes(self.private_bytes())
        Path(public_path).write_bytes(self.public_bytes())

    @classmethod
    def load(cls, private_path: str | Path) -> KeyPair:
        raw = Path(private_path).read_bytes()
        private = Ed25519PrivateKey.from_private_bytes(raw)
        return cls(private_key=private, public_key=private.public_key())


def load_public_key(path: str | Path) -> Ed25519PublicKey:
    raw = Path(path).read_bytes()
    return Ed25519PublicKey.from_public_bytes(raw)


def sign_data(data: bytes, private_key: Ed25519PrivateKey) -> bytes:
    return private_key.sign(data)


def verify_signature(data: bytes, signature: bytes, public_key: Ed25519PublicKey) -> bool:
    if len(signature) != 64:
        raise ValueError(
            f"Ed25519 signature must be exactly 64 bytes, got {len(signature)}"
        )
    try:
        public_key.verify(signature, data)
        return True
    except InvalidSignature:
        return False


def encrypt_aes_gcm(
    plaintext: bytes, key: bytes, aad: bytes | None = None
) -> tuple[bytes, bytes]:
    """Encrypt with AES-256-GCM. Returns (nonce, ciphertext).

    Args:
        plaintext: Data to encrypt.
        key: 32-byte AES-256 key.
        aad: Optional Additional Authenticated Data for domain separation.
    """
    if len(key) != 32:
        raise ValueError("AES-256-GCM requires a 32-byte key")
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, aad)
    return nonce, ciphertext


def decrypt_aes_gcm(
    nonce: bytes, ciphertext: bytes, key: bytes, aad: bytes | None = None
) -> bytes:
    """Decrypt AES-256-GCM ciphertext.

    Args:
        nonce: 12-byte nonce used during encryption.
        ciphertext: The encrypted data (including GCM tag).
        key: 32-byte AES-256 key.
        aad: Optional Additional Authenticated Data (must match encryption).
    """
    if len(key) != 32:
        raise ValueError("AES-256-GCM requires a 32-byte key")
    if len(nonce) != 12:
        raise ValueError(
            f"AES-GCM nonce must be exactly 12 bytes, got {len(nonce)}"
        )
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, aad)


def sha256_digest(data: bytes) -> str:
    """Return hex SHA-256 digest."""
    digest = hashes.Hash(hashes.SHA256())
    digest.update(data)
    return digest.finalize().hex()
