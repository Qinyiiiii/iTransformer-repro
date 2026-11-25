"""命令行入口的参数解析与子命令分发测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import itransformer_repro.cli as cli

CONFIG = {
    "experiment": "cli_test",
    "seed": 2021,
    "device": "cpu",
    "runs_dir": "runs",
    "data": {"name": "ETTh1", "seq_len": 96, "pred_len": 96},
    "model": {"name": "itransformer", "d_model": 64, "d_ff": 128, "n_heads": 4},
    "train": {"batch_size": 8, "learning_rate": 0.001, "epochs": 1},
}


def write_config(tmp_path: Path) -> Path:
    """把测试配置写入临时 YAML。"""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(CONFIG, allow_unicode=True), encoding="utf-8")
    return path


def test_build_parser_requires_subcommand() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_collect_overrides_merges_flags_and_set(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(
        [
            "train",
            "--config",
            str(write_config(tmp_path)),
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--set",
            "model.d_model=32",
            "model.variate_module=ffn",
        ]
    )
    overrides = cli.collect_overrides(args)
    assert overrides == {
        "seed": 7,
        "train.batch_size": 4,
        "model.d_model": 32,
        "model.variate_module": "ffn",
    }


def test_collect_overrides_rejects_bad_set(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(
        ["train", "--config", str(write_config(tmp_path)), "--set", "oops"]
    )
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        cli.collect_overrides(args)


def test_main_dispatches_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    calls: dict[str, object] = {}

    def fake_run(config: object, force: bool = False) -> dict[str, object]:
        calls["config"] = config
        calls["force"] = force
        return {
            "experiment": "cli_test",
            "seed": 2021,
            "mse": 0.5,
            "mae": 0.4,
            "train_time_sec": 1.0,
        }

    monkeypatch.setattr(cli, "run_experiment", fake_run)
    code = cli.main(["train", "--config", str(write_config(tmp_path)), "--force"])
    assert code == 0
    assert calls["force"] is True
    assert "mse=0.500000" in capsys.readouterr().out


def test_main_reports_existing_run_as_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    def fake_run(config: object, force: bool = False) -> dict[str, object]:
        raise FileExistsError("已有结果")

    monkeypatch.setattr(cli, "run_experiment", fake_run)
    code = cli.main(["train", "--config", str(write_config(tmp_path))])
    assert code == 0
    assert "SKIP" in capsys.readouterr().out


def test_main_dispatches_evaluate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    def fake_evaluate(run_dir: Path) -> dict[str, object]:
        assert run_dir == tmp_path / "run"
        return {"mse": 0.25, "mae": 0.5, "device": "cpu", "matches_original": True}

    monkeypatch.setattr(cli, "evaluate_run", fake_evaluate)
    code = cli.main(["evaluate", "--run-dir", str(tmp_path / "run")])
    assert code == 0
    out = capsys.readouterr().out
    assert "recomputed mse=0.250000" in out
    assert "matches_original=True" in out


def test_main_dispatches_check_data_and_writes_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    card = {
        "dataset": "ETTh1",
        "passed": True,
        "rows": 17420,
        "rows_match": False,
        "split_windows": {"train": 1, "val": 1, "test": 1},
    }
    monkeypatch.setattr(cli, "dataset_card", lambda config, strict=False: card)
    out = tmp_path / "card.json"
    code = cli.main(
        ["check-data", "--config", str(write_config(tmp_path)), "--out", str(out)]
    )
    assert code == 0
    assert json.loads(out.read_text(encoding="utf-8"))["rows"] == 17420
    assert "WARN" in capsys.readouterr().out


def test_main_returns_nonzero_when_check_data_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    card = {"dataset": "ETTh1", "passed": False, "rows_match": False}
    monkeypatch.setattr(cli, "dataset_card", lambda config, strict=False: card)
    code = cli.main(["check-data", "--config", str(write_config(tmp_path))])
    assert code == 1
    assert "FAIL" in capsys.readouterr().out
