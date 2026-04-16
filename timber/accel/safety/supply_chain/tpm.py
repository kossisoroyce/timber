"""TPM integration hooks for hardware-backed key management."""

from __future__ import annotations

import abc
import os
import re
from dataclasses import dataclass
from typing import Optional

from cryptography.exceptions import InvalidSignature


class TPMInterface(abc.ABC):
    """Abstract TPM interface for key management."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if TPM hardware is available."""
        ...

    @abc.abstractmethod
    def create_key(self, key_id: str) -> bytes:
        """Create a signing key in TPM. Returns public key bytes."""
        ...

    @abc.abstractmethod
    def sign(self, key_id: str, data: bytes) -> bytes:
        """Sign data using TPM-resident key."""
        ...

    @abc.abstractmethod
    def verify(self, key_id: str, data: bytes, signature: bytes) -> bool:
        """Verify signature using TPM-resident key."""
        ...

    @abc.abstractmethod
    def seal(self, data: bytes, pcr_selection: Optional[list[int]] = None) -> bytes:
        """Seal data to current TPM state (PCR values)."""
        ...

    @abc.abstractmethod
    def unseal(self, sealed_data: bytes) -> bytes:
        """Unseal data (only succeeds if PCR values match)."""
        ...

    @abc.abstractmethod
    def get_attestation(self) -> dict:
        """Get TPM attestation report."""
        ...


class LinuxTPM(TPMInterface):
    """Linux TPM 2.0 implementation using /dev/tpmrm0.

    NOTE: This is a stub implementation. Full TPM integration requires
    the tpm2-pytss library and appropriate system configuration.
    """

    def __init__(self, device: str = "/dev/tpmrm0"):
        self.device = device

    def is_available(self) -> bool:
        import os
        return os.path.exists(self.device)

    def create_key(self, key_id: str) -> bytes:
        if not self.is_available():
            raise RuntimeError("TPM device not available")
        # Stub: would use tpm2-pytss to create primary key + signing key
        raise NotImplementedError(
            "TPM key creation requires tpm2-pytss. "
            "Install with: pip install tpm2-pytss"
        )

    def sign(self, key_id: str, data: bytes) -> bytes:
        raise NotImplementedError("TPM signing requires tpm2-pytss")

    def verify(self, key_id: str, data: bytes, signature: bytes) -> bool:
        raise NotImplementedError("TPM verification requires tpm2-pytss")

    def seal(self, data: bytes, pcr_selection: Optional[list[int]] = None) -> bytes:
        raise NotImplementedError("TPM sealing requires tpm2-pytss")

    def unseal(self, sealed_data: bytes) -> bytes:
        raise NotImplementedError("TPM unsealing requires tpm2-pytss")

    def get_attestation(self) -> dict:
        raise NotImplementedError("TPM attestation requires tpm2-pytss")


class SoftwareTPM(TPMInterface):
    """Software-only TPM emulation for testing.

    WARNING: This provides NO actual hardware security guarantees.
    Use only for development and testing.
    """

    def __init__(self):
        self._keys: dict[str, tuple[bytes, bytes]] = {}  # key_id -> (private, public)
        self._seal_key = os.urandom(32)

    def is_available(self) -> bool:
        return True

    @staticmethod
    def _validate_key_id(key_id: str) -> None:
        if not key_id:
            raise ValueError("key_id must be non-empty")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", key_id):
            raise ValueError(
                f"key_id contains unsafe characters: {key_id!r}. "
                "Only alphanumeric, underscore, and hyphen are allowed."
            )

    def create_key(self, key_id: str) -> bytes:
        self._validate_key_id(key_id)
        from timber.accel._util.crypto import KeyPair
        kp = KeyPair.generate()
        self._keys[key_id] = (kp.private_bytes(), kp.public_bytes())
        return kp.public_bytes()

    def sign(self, key_id: str, data: bytes) -> bytes:
        self._validate_key_id(key_id)
        if key_id not in self._keys:
            raise KeyError(f"Key not found: {key_id}")
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        priv = Ed25519PrivateKey.from_private_bytes(self._keys[key_id][0])
        return priv.sign(data)

    def verify(self, key_id: str, data: bytes, signature: bytes) -> bool:
        self._validate_key_id(key_id)
        if key_id not in self._keys:
            raise KeyError(f"Key not found: {key_id}")
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        pub = Ed25519PublicKey.from_public_bytes(self._keys[key_id][1])
        try:
            pub.verify(signature, data)
            return True
        except (ValueError, InvalidSignature):
            return False

    def seal(self, data: bytes, pcr_selection: Optional[list[int]] = None) -> bytes:
        from timber.accel._util.crypto import encrypt_aes_gcm
        nonce, ct = encrypt_aes_gcm(data, self._seal_key)
        return nonce + ct

    def unseal(self, sealed_data: bytes) -> bytes:
        from timber.accel._util.crypto import decrypt_aes_gcm
        nonce = sealed_data[:12]
        ct = sealed_data[12:]
        return decrypt_aes_gcm(nonce, ct, self._seal_key)

    def get_attestation(self) -> dict:
        return {
            "type": "software_emulation",
            "keys": list(self._keys.keys()),
            "warning": "Software TPM — no hardware security guarantees",
        }


def get_tpm(backend: str = "auto") -> TPMInterface:
    """Get TPM implementation.

    Args:
        backend: "linux", "software", or "auto" (tries Linux first, falls back to software)
    """
    if backend == "linux":
        return LinuxTPM()
    elif backend == "software":
        return SoftwareTPM()
    else:  # auto
        tpm = LinuxTPM()
        if tpm.is_available():
            return tpm
        return SoftwareTPM()
