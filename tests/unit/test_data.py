"""数据切分、标准化与窗口构造的单元测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from itransformer_repro.config import DataConfig
from itransformer_repro.data import (
    DATASET_SPECS,
    DatasetSpec,
    StandardScaler,
    WindowDataset,
    build_datasets,
    dataset_card,
    load_frame,
    split_borders,
)


def make_csv(
    path: Path, rows: int, n_vars: int, with_date: bool = True, seed: int = 0
) -> np.ndarray:
    """生成用于测试的随机 CSV，返回写入的数值矩阵。"""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, n_vars))
    frame = pd.DataFrame(values, columns=[f"v{index}" for index in range(n_vars)])
    if with_date:
        stamps = pd.date_range("2020-01-01", periods=rows, freq="h")
        frame.insert(0, "date", stamps.astype(str))
    frame.to_csv(path, index=False)
    return values


def test_split_borders_follows_official_ett_months() -> None:
    borders = split_borders(17420, DATASET_SPECS["ETTh1"], 96)
    assert borders["train"] == (0, 8640)
    assert borders["val"] == (8544, 11520)
    assert borders["test"] == (11424, 14400)


def test_etth1_window_counts_match_paper_table4_convention() -> None:
    spec = DATASET_SPECS["ETTh1"]
    borders = split_borders(17420, spec, 96)
    strict = {name: end - start - 96 - 96 + 1 for name, (start, end) in borders.items()}
    paper = {name: end - start - 96 + 1 for name, (start, end) in borders.items()}
    assert strict == {"train": 8449, "val": 2785, "test": 2785}
    # 论文表 4 的 8545/2881/2881 = 严格样本数 + pred_len。
    assert paper == {"train": 8545, "val": 2881, "test": 2881}


def test_split_borders_uses_tail_for_ratio_datasets() -> None:
    borders = split_borders(1000, DATASET_SPECS["Weather"], 24)
    assert borders["train"] == (0, 700)
    assert borders["val"] == (676, 800)
    assert borders["test"] == (776, 1000)


def test_split_borders_rejects_short_ett_file() -> None:
    with pytest.raises(ValueError, match="12/4/4"):
        split_borders(100, DATASET_SPECS["ETTh1"], 96)


def test_scaler_roundtrip_restores_original_values() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scaler = StandardScaler().fit(values)
    scaled = scaler.transform(values)
    np.testing.assert_allclose(scaled, [[-1.0, -1.0], [1.0, 1.0]])
    np.testing.assert_allclose(scaler.inverse_transform(scaled), values)


def test_scaler_requires_fit_before_transform() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        StandardScaler().transform(np.zeros((2, 2)))


def test_window_dataset_length_and_shapes() -> None:
    values = np.arange(100 * 3, dtype=np.float32).reshape(100, 3)
    dataset = WindowDataset(values, start=0, end=50, seq_len=24, pred_len=12)
    assert len(dataset) == 15
    inputs, targets = dataset[0]
    assert tuple(inputs.shape) == (24, 3)
    assert tuple(targets.shape) == (12, 3)
    np.testing.assert_allclose(targets[0].numpy(), values[24])


def test_window_dataset_rejects_out_of_range_index() -> None:
    values = np.zeros((100, 3), dtype=np.float32)
    dataset = WindowDataset(values, start=0, end=50, seq_len=24, pred_len=12)
    with pytest.raises(IndexError):
        _ = dataset[15]


def test_load_frame_drops_date_column(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    values = make_csv(path, rows=20, n_vars=4)
    loaded = load_frame(path)
    assert loaded.shape == (20, 4)
    np.testing.assert_allclose(loaded, values.astype(np.float32), rtol=1e-6)


def test_load_frame_rejects_nan(tmp_path: Path) -> None:
    path = tmp_path / "broken.csv"
    frame = pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "v0": [1.0, np.nan]})
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="NaN/inf"):
        load_frame(path)


def test_load_frame_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_frame(tmp_path / "absent.csv")


def test_build_datasets_fits_scaler_on_train_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "synthetic.csv"
    make_csv(path, rows=400, n_vars=3)
    monkeypatch.setitem(
        DATASET_SPECS, "SYN", DatasetSpec("synthetic.csv", 3, "1h", "ratio")
    )
    config = DataConfig(
        name="SYN", root=str(tmp_path), file="synthetic.csv", seq_len=24, pred_len=12
    )
    bundle = build_datasets(config)
    assert bundle.n_rows == 400
    assert len(bundle.train) == 280 - 24 - 12 + 1
    raw = pd.read_csv(path).drop(columns=["date"]).to_numpy()
    np.testing.assert_allclose(bundle.scaler.mean_, raw[:280].mean(axis=0), rtol=1e-5)


def test_build_datasets_rejects_variable_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "synthetic.csv"
    make_csv(path, rows=400, n_vars=3)
    monkeypatch.setitem(
        DATASET_SPECS, "SYN", DatasetSpec("synthetic.csv", 5, "1h", "ratio")
    )
    config = DataConfig(
        name="SYN", root=str(tmp_path), file="synthetic.csv", seq_len=24, pred_len=12
    )
    with pytest.raises(ValueError, match="期望 5 个变量"):
        build_datasets(config)


def test_build_datasets_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="未知数据集"):
        build_datasets(DataConfig(name="NotADataset"))


def test_dataset_card_reports_window_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "synthetic.csv"
    make_csv(path, rows=400, n_vars=3)
    monkeypatch.setitem(
        DATASET_SPECS,
        "SYN",
        DatasetSpec("synthetic.csv", 3, "1h", "ratio", expected_rows=400),
    )
    config = DataConfig(
        name="SYN", root=str(tmp_path), file="synthetic.csv", seq_len=24, pred_len=12
    )
    card = dataset_card(config)
    assert card["passed"] is True
    assert card["rows"] == 400
    assert card["rows_match"] is True
    assert card["size_bytes"] > 0
    assert len(card["sha256"]) == 64
    assert card["split_windows"]["train"] == 245
    assert card["borders"]["test"] == [296, 400]
    # 论文表 4 口径只数回看起点，比严格样本数多 pred_len 个。
    assert card["paper_table4_windows"]["test"] == card["split_windows"]["test"] + 12


def test_dataset_card_warns_on_row_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "synthetic.csv"
    make_csv(path, rows=400, n_vars=3)
    monkeypatch.setitem(
        DATASET_SPECS,
        "SYN",
        DatasetSpec("synthetic.csv", 3, "1h", "ratio", expected_rows=401),
    )
    config = DataConfig(
        name="SYN", root=str(tmp_path), file="synthetic.csv", seq_len=24, pred_len=12
    )
    assert dataset_card(config)["passed"] is True
    assert dataset_card(config, strict=True)["passed"] is False
