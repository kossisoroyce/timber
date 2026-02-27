"""Tests for the Timber model store and Ollama-style CLI workflow."""

import json
import numpy as np
import pytest
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from click.testing import CliRunner

from timber.store import ModelStore, ModelInfo
from timber.cli import main


@pytest.fixture
def store(tmp_path):
    """Create a ModelStore backed by a temp directory."""
    return ModelStore(home=tmp_path / ".timber")


@pytest.fixture
def xgb_model_path(tmp_path):
    """Train and save a small XGBoost model, return path."""
    data = load_breast_cancer()
    X, y = data.data.astype(np.float32), data.target
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)
    path = tmp_path / "test_model.json"
    model.get_booster().save_model(str(path))
    return path


class TestModelStore:
    def test_load_model(self, store, xgb_model_path):
        info = store.load_model(xgb_model_path, name="mymodel")
        assert info.name == "mymodel"
        assert info.format == "xgboost"
        assert info.n_trees == 5
        assert info.n_features == 30
        assert info.framework == "xgboost"
        assert info.size_bytes > 0

    def test_load_auto_name(self, store, xgb_model_path):
        info = store.load_model(xgb_model_path)
        assert info.name == "test_model"

    def test_get_model(self, store, xgb_model_path):
        store.load_model(xgb_model_path, name="mymodel")
        info = store.get_model("mymodel")
        assert info is not None
        assert info.name == "mymodel"

    def test_get_model_not_found(self, store):
        assert store.get_model("nonexistent") is None

    def test_list_models(self, store, xgb_model_path):
        store.load_model(xgb_model_path, name="model_a")
        store.load_model(xgb_model_path, name="model_b")
        models = store.list_models()
        names = {m.name for m in models}
        assert names == {"model_a", "model_b"}

    def test_remove_model(self, store, xgb_model_path):
        store.load_model(xgb_model_path, name="to_delete")
        assert store.get_model("to_delete") is not None
        removed = store.remove_model("to_delete")
        assert removed is True
        assert store.get_model("to_delete") is None

    def test_remove_nonexistent(self, store):
        assert store.remove_model("nope") is False

    def test_overwrite_model(self, store, xgb_model_path):
        store.load_model(xgb_model_path, name="same")
        store.load_model(xgb_model_path, name="same")
        models = store.list_models()
        assert len(models) == 1

    def test_compiled_artifacts_exist(self, store, xgb_model_path):
        store.load_model(xgb_model_path, name="compiled_test")
        model_dir = store.get_model_dir("compiled_test")
        assert model_dir is not None
        assert (model_dir / "compiled" / "model.c").exists()
        assert (model_dir / "compiled" / "model.h").exists()
        assert (model_dir / "model_info.json").exists()

    def test_shared_lib_compiled(self, store, xgb_model_path):
        info = store.load_model(xgb_model_path, name="lib_test")
        # gcc may or may not be available, so just check the flag
        if info.compiled:
            lib_path = store.get_lib_path("lib_test")
            assert lib_path is not None
            assert lib_path.exists()

    def test_model_info_json(self, store, xgb_model_path):
        store.load_model(xgb_model_path, name="info_test")
        model_dir = store.get_model_dir("info_test")
        info_path = model_dir / "model_info.json"
        data = json.loads(info_path.read_text())
        assert data["name"] == "info_test"
        assert data["n_trees"] == 5
        assert data["loaded_at"] != ""

    def test_file_not_found(self, store):
        with pytest.raises(FileNotFoundError):
            store.load_model("/nonexistent/model.json")

    def test_name_sanitization(self, store, xgb_model_path):
        info = store.load_model(xgb_model_path, name="My Model/v2")
        assert info.name == "my_model_v2"


class TestCLILoadListRemove:
    """Test the CLI commands using Click's CliRunner."""

    def test_load_command(self, xgb_model_path, tmp_path, monkeypatch):
        monkeypatch.setenv("TIMBER_HOME", str(tmp_path / ".timber"))
        runner = CliRunner()
        result = runner.invoke(main, ["load", str(xgb_model_path), "--name", "cli_test"])
        assert result.exit_code == 0, result.output
        assert "loaded successfully" in result.output
        assert "cli_test" in result.output

    def test_list_command(self, xgb_model_path, tmp_path, monkeypatch):
        monkeypatch.setenv("TIMBER_HOME", str(tmp_path / ".timber"))
        runner = CliRunner()
        runner.invoke(main, ["load", str(xgb_model_path), "--name", "list_test"])
        result = runner.invoke(main, ["list"])
        assert result.exit_code == 0
        assert "list_test" in result.output

    def test_list_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TIMBER_HOME", str(tmp_path / ".timber"))
        runner = CliRunner()
        result = runner.invoke(main, ["list"])
        assert result.exit_code == 0
        assert "No models loaded" in result.output

    def test_remove_command(self, xgb_model_path, tmp_path, monkeypatch):
        monkeypatch.setenv("TIMBER_HOME", str(tmp_path / ".timber"))
        runner = CliRunner()
        runner.invoke(main, ["load", str(xgb_model_path), "--name", "rm_test"])
        result = runner.invoke(main, ["remove", "rm_test"])
        assert result.exit_code == 0
        assert "Removed" in result.output

    def test_remove_nonexistent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TIMBER_HOME", str(tmp_path / ".timber"))
        runner = CliRunner()
        result = runner.invoke(main, ["remove", "nope"])
        assert result.exit_code != 0
