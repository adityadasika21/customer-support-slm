"""Tests for config loading and manifest serialisation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from csbot.config import DataConfig, LoraConfig_, TrainConfig, _from_dict, load_train_config


def test_to_dict_is_json_serialisable():
    """The manifest is written with json.dumps, so nested Paths must be converted.

    Regression test: dataclasses.asdict flattens nested dataclasses into dicts but leaves Path
    values inside them, which crashed manifest writing at the very end of a training run.
    """
    cfg = TrainConfig(model_key="qwen3-1.7b", run_name="t")
    blob = json.dumps(cfg.to_dict())
    assert "artifacts/runs" in blob
    assert "PosixPath" not in blob


def test_to_dict_converts_tuples_to_lists():
    cfg = TrainConfig(model_key="qwen3-1.7b", run_name="t")
    assert isinstance(cfg.to_dict()["lora"]["target_modules"], list)


def test_lora_alpha_defaults_to_twice_rank():
    """alpha = 2r keeps alpha/r constant, so a rank sweep is not secretly a learning-rate sweep."""
    assert LoraConfig_(r=32).alpha == 64
    assert LoraConfig_(r=8, alpha=99).alpha == 99


def test_effective_batch_size():
    cfg = TrainConfig(model_key="m", run_name="r",
                      per_device_batch_size=4, gradient_accumulation_steps=8)
    assert cfg.effective_batch_size == 32


def test_unknown_config_key_rejected():
    """A misspelled key must fail loudly, or a sweep silently runs N identical configs."""
    with pytest.raises(ValueError, match="unknown config keys"):
        _from_dict(DataConfig, {"varient": "raw"})


def test_invalid_variant_rejected():
    with pytest.raises(ValueError, match="variant must be"):
        DataConfig(variant="placeholder")


def test_invalid_scope_rejected():
    with pytest.raises(ValueError, match="scope must be"):
        DataConfig(scope="everything")


def test_load_train_config_applies_overrides(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "model_key: qwen3-1.7b\nrun_name: base\nmax_steps: 100\n"
        "data:\n  variant: raw\n  scope: research\n"
    )
    cfg = load_train_config(path, run_name="overridden", max_steps=250)
    assert cfg.run_name == "overridden"
    assert cfg.max_steps == 250
    assert cfg.data.variant == "raw"
    assert cfg.data.scope == "research"


def test_nested_dicts_become_dataclasses(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text("model_key: m\nrun_name: r\nlora:\n  r: 32\n")
    cfg = load_train_config(path)
    assert isinstance(cfg.lora, LoraConfig_)
    assert cfg.lora.alpha == 64
