"""配置读取、覆盖与校验的单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from itransformer_repro.config import (
    ExperimentConfig,
    apply_overrides,
    coerce_scalar,
    load_config,
    validate_config,
)

MINIMAL = {
    "experiment": "unit_test",
    "seed": 2021,
    "device": "cpu",
    "runs_dir": "runs",
    "data": {"name": "ETTh1", "seq_len": 96, "pred_len": 96},
    "model": {"name": "itransformer", "d_model": 64, "d_ff": 128, "n_heads": 4},
    "train": {"batch_size": 8, "learning_rate": 0.001, "epochs": 1},
}


def write_config(tmp_path: Path, payload: dict) -> Path:
    """把 ``payload`` 写成临时 YAML 并返回路径。"""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


def test_coerce_scalar_recognises_bool_int_float_and_text() -> None:
    assert coerce_scalar("true") is True
    assert coerce_scalar("192") == 192
    assert coerce_scalar("1e-3") == pytest.approx(0.001)
    assert coerce_scalar("ffn") == "ffn"


def test_load_config_applies_overrides(tmp_path: Path) -> None:
    path = write_config(tmp_path, MINIMAL)
    config = load_config(path, {"data.pred_len": 192, "seed": 7})
    assert config.seed == 7
    assert config.data.pred_len == 192
    assert config.data.seq_len == 96


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_apply_overrides_rejects_unknown_section() -> None:
    config = ExperimentConfig(experiment="x")
    with pytest.raises(KeyError):
        apply_overrides(config, {"unknown.field": 1})


def test_apply_overrides_rejects_unknown_field() -> None:
    config = ExperimentConfig(experiment="x")
    with pytest.raises(TypeError):
        apply_overrides(config, {"data.missing": 1})


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"model.name": "mlp"}, "model.name"),
        ({"model.variate_module": "cnn"}, "variate_module"),
        ({"model.temporal_module": "cnn"}, "temporal_module"),
        ({"data.features": "S"}, "features"),
        ({"model.d_model": 10, "model.n_heads": 4}, "整除"),
        ({"train.epochs": 0}, "epochs"),
        ({"train.learning_rate": 0.0}, "learning_rate"),
        ({"seed": -1}, "seed"),
    ],
)
def test_validate_config_rejects_invalid_values(override: dict, message: str) -> None:
    config = apply_overrides(ExperimentConfig(experiment="x"), override)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_none_is_a_valid_ablation_switch() -> None:
    config = apply_overrides(
        ExperimentConfig(experiment="x"),
        {"model.variate_module": "none", "model.temporal_module": "none"},
    )
    validate_config(config)
