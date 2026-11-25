"""数据读取、时间顺序切分与滑窗构造。

切分协议与官方基准（thuml/iTransformer ``data_provider``）保持一致：

* ETT 系列固定使用前 12 / 4 / 4 个月，即 1 小时数据 8640 / 2880 / 2880 行，
  15 分钟数据 34560 / 11520 / 11520 行；文件末尾多出的月份不参与实验。
* 其余基准按 70% / 10% / 20% 时间顺序切分。
* 标准化统计量只在训练段上拟合，验证段与测试段共用同一组统计量，避免信息泄漏。

窗口约定：窗口起点为 ``i`` 时使用 ``data[i : i+T]`` 预测 ``data[i+T : i+T+S]``，
因此某段的窗口数为 ``end - start - T - S + 1``。论文表 4 的 ETT 数字
（如 ETTh1 8545/2881/2881）只数回看起点、不扣 ``pred_len``，比本口径多
``pred_len`` 个；``dataset_card`` 同时给出两种口径以便核对。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from itransformer_repro.config import DataConfig


@dataclass(frozen=True)
class DatasetSpec:
    """公开基准的静态属性。

    Attributes:
        filename: 相对 ``data.root`` 的路径。
        n_vars: 变量（通道）数。
        freq: 采样频率标签，用于 ETT 的月份换算。
        split_rule: ``"ett"`` 表示 12/4/4 个月切分，``"ratio"`` 表示 70/10/20。
        expected_rows: 官方发布文件的数据行数（不含表头）；仅用于完整性告警。
    """

    filename: str
    n_vars: int
    freq: str
    split_rule: str
    expected_rows: int | None = None


DATASET_SPECS: dict[str, DatasetSpec] = {
    "ETTh1": DatasetSpec("ETT-small/ETTh1.csv", 7, "1h", "ett", 17420),
    "ETTh2": DatasetSpec("ETT-small/ETTh2.csv", 7, "1h", "ett", 17420),
    "ETTm1": DatasetSpec("ETT-small/ETTm1.csv", 7, "15min", "ett", 69680),
    "ETTm2": DatasetSpec("ETT-small/ETTm2.csv", 7, "15min", "ett", 69680),
    "Weather": DatasetSpec("weather/weather.csv", 21, "10min", "ratio", 52696),
    "ECL": DatasetSpec("electricity/electricity.csv", 321, "1h", "ratio", 26304),
    "Traffic": DatasetSpec("traffic/traffic.csv", 862, "1h", "ratio", 17544),
    "Exchange": DatasetSpec("exchange_rate/exchange_rate.csv", 8, "1d", "ratio", 7588),
}


class StandardScaler:
    """按列做 z-score 标准化，统计量只由训练段决定。"""

    def __init__(self) -> None:
        self.mean_: NDArray[np.float64] | None = None
        self.std_: NDArray[np.float64] | None = None

    def fit(self, values: NDArray[np.float32]) -> StandardScaler:
        """沿时间轴估计每一列的均值与标准差。

        Args:
            values: 形状 ``(n_rows, n_vars)`` 的训练段数据。

        Returns:
            自身，便于链式调用。

        Raises:
            ValueError: 输入不是二维数组或没有样本。
        """
        if values.ndim != 2:
            raise ValueError(f"期望二维数组，实际形状为 {values.shape}")
        if values.shape[0] == 0:
            raise ValueError("不能用空数组拟合标准化器")
        mean = values.astype(np.float64).mean(axis=0)
        std = values.astype(np.float64).std(axis=0)
        std[std < 1e-8] = 1.0
        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, values: NDArray[np.float32]) -> NDArray[np.float64]:
        """用已拟合的统计量标准化 ``values``。

        Raises:
            RuntimeError: 尚未调用 ``fit``。
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler 必须先 fit 再 transform")
        return (values - self.mean_) / self.std_

    def inverse_transform(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """把标准化数值映射回原始量纲。

        Raises:
            RuntimeError: 尚未调用 ``fit``。
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler 必须先 fit 再 inverse_transform")
        return values * self.std_ + self.mean_


def resolve_path(config: DataConfig) -> Path:
    """返回 ``config`` 描述的 CSV 路径（相对路径以仓库根目录为基准）。"""
    return Path(config.root) / config.file


def load_frame(path: Path) -> NDArray[np.float32]:
    """读取基准文件，去掉时间列后返回 ``float32`` 矩阵。

    Args:
        path: CSV 文件路径。

    Returns:
        形状 ``(n_rows, n_vars)``、无缺失值的数组。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 没有数值列，或解析后存在 NaN/inf。
    """
    if not path.exists():
        raise FileNotFoundError(
            f"数据文件不存在：{path}。请按 data/README.md 的说明下载并放置数据。"
        )
    frame = pd.read_csv(path)
    time_columns = [name for name in frame.columns if str(name).lower() == "date"]
    if time_columns:
        frame = frame.drop(columns=time_columns)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if frame.shape[1] == 0:
        raise ValueError(f"{path} 中没有可用的数值列")

    values = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).sum())
        raise ValueError(f"{path} 解析后存在 {bad} 个 NaN/inf 值")
    return values


def split_borders(
    n_rows: int, spec: DatasetSpec, seq_len: int
) -> dict[str, tuple[int, int]]:
    """计算各段的窗口起止边界。

    Args:
        n_rows: 文件总行数。
        spec: 数据集静态属性。
        seq_len: 回看长度 T（验证/测试段需要向前借 T 行构造输入）。

    Returns:
        映射 ``{"train": (start, end), ...}``，窗口起点 ``i`` 需满足
        ``i + seq_len + pred_len <= end``。

    Raises:
        ValueError: 文件长度不足以支持该协议或该 ``seq_len``。
    """
    if spec.split_rule == "ett":
        month = 30 * 24 * (4 if spec.freq == "15min" else 1)
        num_train, num_val, num_test = 12 * month, 4 * month, 4 * month
        if num_train + num_val + num_test > n_rows:
            raise ValueError(
                f"数据集只有 {n_rows} 行，不足以覆盖 ETT 的 12/4/4 个月协议"
            )
        borders = {
            "train": (0, num_train),
            "val": (num_train - seq_len, num_train + num_val),
            "test": (num_train + num_val - seq_len, num_train + num_val + num_test),
        }
    else:
        num_train = int(n_rows * 0.7)
        num_test = int(n_rows * 0.2)
        num_val = n_rows - num_train - num_test
        borders = {
            "train": (0, num_train),
            "val": (num_train - seq_len, num_train + num_val),
            "test": (n_rows - num_test - seq_len, n_rows),
        }

    for split, (start, end) in borders.items():
        if end - start <= seq_len:
            raise ValueError(
                f"划分 '{split}' 共 {end - start} 行，容不下 seq_len={seq_len}"
            )
    return borders


class WindowDataset(Dataset):
    """滑窗数据集，元素形状为 ``(seq_len, N)`` 与 ``(pred_len, N)``。"""

    def __init__(
        self,
        data: NDArray[np.float32],
        start: int,
        end: int,
        seq_len: int,
        pred_len: int,
    ) -> None:
        """创建某一时间段的窗口视图。

        Args:
            data: 全量标准化数据，形状 ``(n_rows, n_vars)``。
            start: 该段窗口起点的下界（含）。
            end: 该段可用数据的右边界（不含）。
            seq_len: 回看长度 T。
            pred_len: 预测长度 S。

        Raises:
            ValueError: ``data`` 不是二维数组。
        """
        if data.ndim != 2:
            raise ValueError(f"期望二维数组，实际形状为 {data.shape}")
        self._data = torch.from_numpy(np.ascontiguousarray(data))
        self.start = start
        self.end = end
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return max(0, self.end - self.start - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回第 ``index`` 个 ``(输入窗口, 预测目标)``。

        Raises:
            IndexError: 索引越界。
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"索引 {index} 超出范围（共 {len(self)} 个窗口）")
        offset = self.start + index
        window = self._data[offset : offset + self.seq_len]
        horizon = self._data[
            offset + self.seq_len : offset + self.seq_len + self.pred_len
        ]
        return window, horizon


@dataclass(frozen=True)
class DataBundle:
    """训练/验证/测试窗口集合、拟合好的标准化器与维度信息。"""

    train: WindowDataset
    val: WindowDataset
    test: WindowDataset
    n_vars: int
    scaler: StandardScaler
    n_rows: int


def _get_spec(name: str) -> DatasetSpec:
    """按名称取数据集规格。

    Raises:
        ValueError: 名称未注册。
    """
    spec = DATASET_SPECS.get(name)
    if spec is None:
        raise ValueError(f"未知数据集 '{name}'，已注册：{sorted(DATASET_SPECS)}")
    return spec


def _build_bundle(
    values: NDArray[np.float32], spec: DatasetSpec, config: DataConfig
) -> tuple[DataBundle, dict[str, tuple[int, int]]]:
    """在已加载的数值矩阵上完成切分、标准化与窗口构造。"""
    borders = split_borders(len(values), spec, config.seq_len)
    train_start, train_end = borders["train"]
    scaler = StandardScaler().fit(values[train_start:train_end])
    scaled = scaler.transform(values).astype(np.float32)

    bundle = DataBundle(
        train=WindowDataset(scaled, *borders["train"], config.seq_len, config.pred_len),
        val=WindowDataset(scaled, *borders["val"], config.seq_len, config.pred_len),
        test=WindowDataset(scaled, *borders["test"], config.seq_len, config.pred_len),
        n_vars=spec.n_vars,
        scaler=scaler,
        n_rows=int(values.shape[0]),
    )
    return bundle, borders


def build_datasets(config: DataConfig) -> DataBundle:
    """加载基准数据，用训练段拟合标准化器并构造三段窗口集合。

    Args:
        config: 数据配置。

    Returns:
        训练/验证/测试窗口集合。

    Raises:
        ValueError: 数据集未注册，或列数与注册规格不一致。
    """
    spec = _get_spec(config.name)
    path = resolve_path(config)
    values = load_frame(path)
    if values.shape[1] != spec.n_vars:
        raise ValueError(
            f"{config.name} 期望 {spec.n_vars} 个变量，实际在 {path} 中读到 "
            f"{values.shape[1]} 列"
        )
    bundle, _ = _build_bundle(values, spec, config)
    return bundle


def _sha256(path: Path) -> str:
    """以 1 MiB 分块计算文件 SHA256，用于记录数据版本。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_card(config: DataConfig, strict: bool = False) -> dict[str, Any]:
    """生成 ``check-data`` 数据集卡片，用于校验与实验记录。

    Args:
        config: 数据配置。
        strict: 为 ``True`` 时把行数与官方发布版本不一致也视为失败。

    Returns:
        含文件哈希、行数、变量数、切分窗口数、边界以及论文表 4 计数口径的
        字典。``split_windows`` 是严格样本数（扣掉 ``pred_len``），
        ``paper_table4_windows`` 只数回看起点（不扣 ``pred_len``），用于和
        论文表 4 的 8545/2881/2881 对齐。

    Raises:
        FileNotFoundError: 数据文件不存在。
        ValueError: 数据集未注册。
    """
    spec = _get_spec(config.name)
    path = resolve_path(config)
    values = load_frame(path)
    rows, n_vars = values.shape
    vars_ok = int(n_vars) == spec.n_vars
    rows_ok = spec.expected_rows is None or int(rows) == spec.expected_rows
    card: dict[str, Any] = {
        "dataset": config.name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "rows": int(rows),
        "expected_rows": spec.expected_rows,
        "rows_match": rows_ok,
        "n_vars": int(n_vars),
        "expected_n_vars": spec.n_vars,
        "vars_match": vars_ok,
        "freq": spec.freq,
        "split_rule": spec.split_rule,
        "seq_len": config.seq_len,
        "pred_len": config.pred_len,
        "passed": vars_ok and (rows_ok or not strict),
        "split_windows": None,
        "paper_table4_windows": None,
        "borders": None,
    }
    if not vars_ok:
        card["message"] = "变量数与基准规格不一致，请检查数据文件是否完整"
        return card
    if not rows_ok:
        card["message"] = (
            f"行数 {rows} 与官方发布版本 {spec.expected_rows} 不一致；"
            "可能是数据版本不同或文件被截断"
        )
    try:
        bundle, borders = _build_bundle(values, spec, config)
    except ValueError as exc:
        card["passed"] = False
        card["message"] = str(exc)
        return card

    card["split_windows"] = {
        "train": len(bundle.train),
        "val": len(bundle.val),
        "test": len(bundle.test),
    }
    card["paper_table4_windows"] = {
        name: max(0, end - start - config.seq_len + 1)
        for name, (start, end) in borders.items()
    }
    card["borders"] = {key: list(value) for key, value in borders.items()}
    return card
