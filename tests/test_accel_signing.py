"""Tests for Ed25519 signing and verification."""

import os
import tempfile

import pytest


class TestKeyGeneration:
    def test_generate_keypair(self):
        from timber.accel._util.crypto import KeyPair

        kp = KeyPair.generate()
        assert len(kp.private_bytes()) == 32
        assert len(kp.public_bytes()) == 32

    def test_save_and_load_keypair(self, tmp_path):
        from timber.accel._util.crypto import KeyPair

        kp = KeyPair.generate()
        priv_path = tmp_path / "test.key"
        pub_path = tmp_path / "test.pub"
        kp.save(priv_path, pub_path)

        loaded = KeyPair.load(priv_path)
        assert loaded.public_bytes() == kp.public_bytes()


class TestSigning:
    def test_sign_and_verify_data(self):
        from timber.accel._util.crypto import KeyPair, sign_data, verify_signature

        kp = KeyPair.generate()
        data = b"test data for signing"
        sig = sign_data(data, kp.private_key)

        assert verify_signature(data, sig, kp.public_key)

    def test_verify_wrong_data_fails(self):
        from timber.accel._util.crypto import KeyPair, sign_data, verify_signature

        kp = KeyPair.generate()
        data = b"correct data"
        sig = sign_data(data, kp.private_key)

        assert not verify_signature(b"wrong data", sig, kp.public_key)

    def test_verify_wrong_key_fails(self):
        from timber.accel._util.crypto import KeyPair, sign_data, verify_signature

        kp1 = KeyPair.generate()
        kp2 = KeyPair.generate()
        data = b"test data"
        sig = sign_data(data, kp1.private_key)

        assert not verify_signature(data, sig, kp2.public_key)


class TestArtifactSigning:
    def test_sign_and_verify_file(self, tmp_path):
        from timber.accel.safety.supply_chain.signing import sign_artifact, generate_keypair
        from timber.accel.safety.supply_chain.verification import verify_artifact

        # Create test artifact
        artifact = tmp_path / "model.c"
        artifact.write_text("int main() { return 0; }")

        # Generate keys
        priv_path, pub_path = generate_keypair(str(tmp_path))

        # Sign
        sig_path = sign_artifact(str(artifact), key_path=priv_path)
        assert os.path.exists(sig_path)

        # Verify
        assert verify_artifact(str(artifact), sig_path, pub_path)

    def test_sign_directory(self, tmp_path):
        from timber.accel.safety.supply_chain.signing import sign_artifact, generate_keypair
        from timber.accel.safety.supply_chain.verification import verify_artifact

        # Create test directory with files
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        (artifact_dir / "model.c").write_text("void infer() {}")
        (artifact_dir / "model.h").write_text("#pragma once")

        priv_path, pub_path = generate_keypair(str(tmp_path))
        sig_path = sign_artifact(str(artifact_dir), key_path=priv_path)

        assert verify_artifact(str(artifact_dir), sig_path, pub_path)
