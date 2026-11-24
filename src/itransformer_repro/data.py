"""数据读取、时间顺序切分与滑窗构造。

切分协议与官方基准（thuml/iTransformer ``data_provider``）保持一致：

* ETT 系列固定使用前 12 / 4 / 4 个月，即 1 小时数据 8640 / 2880 / 2880 行，
  15 分钟数据 34560 / 11520 / 11520 行；文件末尾多出的月份不参与实验。
* 其余基准按 70% / 10% / 20% 时间顺序切分。
* 标准化统计量只在训练段上拟合，验证段与测试段共用同一组统计量，避免信息泄漏。

窗口约定：窗口起点为 ``i`` 时使用 ``data[i : i+T]`` 预测 ``data[i+T : i+T+S]``，
因此某段的窗口数为 ``end - start - T - S + 1``。
"""

from __future__ import annotations

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
    """

    filename: str
    n_vars: int
    freq: str
    split_rule: str


DATASET_SPECS: dict[str, DatasetSpec] = {
    "ETTh1": DatasetSpec("ETT-small/ETTh1.csv", 7, "1h", "ett"),
    "ETTh2": DatasetSpec("ETT-small/ETTh2.csv", 7, "1h", "ett"),
    "ETTm1": DatasetSpec("ETT-small/ETTm1.csv", 7, "15min", "ett"),
    "ETTm2": DatasetSpec("ETT-small/ETTm2.csv", 7, "15min", "ett"),
    "Weather": DatasetSpec("weather/weather.csv", 21, "10min", "ratio"),
    "ECL": DatasetSpec("electricity/electricity.csv", 321, "1h", "ratio"),
    "Traffic": DatasetSpec("traffic/traffic.csv", 862, "1h", "ratio"),
    "Exchange": DatasetSpec("exchange_rate/exchange_rate.csv", 8, "1d", "ratio"),
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


def dataset_card(config: DataConfig) -> dict[str, Any]:
    """生成 ``check-data`` 数据集卡片，用于校验与实验记录。

    Args:
        config: 数据配置。

    Returns:
        含文件路径、行数、变量数、切分窗口数与边界的字典；变量数不符时
        ``passed`` 为 ``False`` 且不给出窗口统计。

    Raises:
        FileNotFoundError: 数据文件不存在。
        ValueError: 数据集未注册。
    """
    spec = _get_spec(config.name)
    path = resolve_path(config)
    values = load_frame(path)
    rows, n_vars = values.shape
    card: dict[str, Any] = {
        "dataset": config.name,
        "path": str(path),
        "rows": int(rows),
        "n_vars": int(n_vars),
        "expected_n_vars": spec.n_vars,
        "freq": spec.freq,
        "split_rule": spec.split_rule,
        "seq_len": config.seq_len,
        "pred_len": config.pred_len,
        "passed": int(n_vars) == spec.n_vars,
        "split_windows": None,
        "borders": None,
    }
    if not card["passed"]:
        card["message"] = "变量数与基准规格不一致，请检查数据文件是否完整"
        return card

    bundle, borders = _build_bundle(values, spec, config)
    card["split_windows"] = {
        "train": len(bundle.train),
        "val": len(bundle.val),
        "test": len(bundle.test),
    }
    card["borders"] = {key: list(value) for key, value in borders.items()}
    return card
