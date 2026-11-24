"""端到端冒烟测试：小样本、小模型、1 个 epoch。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from itransformer_repro.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from itransformer_repro.data import DATASET_SPECS, DatasetSpec
from itransformer_repro.evaluate import evaluate_run
from itransformer_repro.train import run_experiment

ROWS = 400
N_VARS = 3
SEQ_LEN = 24
PRED_LEN = 12


@pytest.fixture
def synthetic_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ExperimentConfig:
    """在临时目录里注册一个 3 变量合成数据集并返回对应配置。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rng = np.random.default_rng(7)
    values = rng.normal(size=(ROWS, N_VARS))
    frame = pd.DataFrame(values, columns=[f"v{index}" for index in range(N_VARS)])
    stamps = pd.date_range("2020-01-01", periods=ROWS, freq="h")
    frame.insert(0, "date", stamps.astype(str))
    frame.to_csv(raw_dir / "synthetic.csv", index=False)

    monkeypatch.setitem(
        DATASET_SPECS, "SYN", DatasetSpec("synthetic.csv", N_VARS, "1h", "ratio")
    )
    return ExperimentConfig(
        experiment="smoke_syn",
        seed=7,
        device="cpu",
        runs_dir=str(tmp_path / "runs"),
        data=DataConfig(
            name="SYN",
            root=str(raw_dir),
            file="synthetic.csv",
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
        ),
        model=ModelConfig(
            name="itransformer",
            d_model=16,
            d_ff=32,
            n_heads=2,
            e_layers=1,
            dropout=0.0,
        ),
        train=TrainConfig(batch_size=8, learning_rate=0.001, epochs=1),
    )


def test_run_experiment_writes_complete_evidence(
    synthetic_project: ExperimentConfig,
) -> None:
    result = run_experiment(synthetic_project)
    run_dir = synthetic_project.run_dir()

    for name in ("config.json", "metrics.json", "train.log", "pred.npy", "model.pt"):
        assert (run_dir / name).exists(), name

    assert np.isfinite(result["mse"])
    assert result["mse"] > 0
    assert result["epochs_run"] == 1
    assert result["n_windows"]["train"] > 0
    assert result["device"] == "cpu"

    predictions = np.load(run_dir / "pred.npy")
    assert predictions.shape == (69, PRED_LEN, N_VARS)

    stored = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert stored["seed"] == 7
    assert "environment" in stored


def test_run_experiment_refuses_to_overwrite(
    synthetic_project: ExperimentConfig,
) -> None:
    run_experiment(synthetic_project)
    with pytest.raises(FileExistsError, match="已有结果"):
        run_experiment(synthetic_project)


def test_evaluate_run_recomputes_metrics(
    synthetic_project: ExperimentConfig,
) -> None:
    original = run_experiment(synthetic_project)
    payload = evaluate_run(synthetic_project.run_dir())
    assert payload["recomputed"] is True
    assert payload["mse"] == pytest.approx(original["mse"], rel=1e-4)
    assert payload["mae"] == pytest.approx(original["mae"], rel=1e-4)
