"""预测指标，全部在标准化空间中计算。

与官方基准一致：先在训练段拟合标准化统计量，再在标准化空间上报告 MSE/MAE，
因此不同数据集的数值可以直接横向比较，但也**不能**直接换算回原始单位。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_pair(pred: FloatArray, target: FloatArray) -> tuple[FloatArray, FloatArray]:
    """校验并统一转换为 ``float64`` 数组。

    Args:
        pred: 预测数组，形状与 ``target`` 相同。
        target: 真值数组。

    Returns:
        转换后的 ``(pred, target)``。

    Raises:
        ValueError: 形状不一致或数组为空。
    """
    pred_array = np.asarray(pred, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if pred_array.shape != target_array.shape:
        raise ValueError(
            f"预测与真值形状必须一致，实际为 {pred_array.shape} 与 {target_array.shape}"
        )
    if pred_array.size == 0:
        raise ValueError("预测与真值不能为空数组")
    return pred_array, target_array


def mse(pred: FloatArray, target: FloatArray) -> float:
    """均方误差 MSE。"""
    pred_array, target_array = _as_pair(pred, target)
    return float(np.mean((pred_array - target_array) ** 2))


def mae(pred: FloatArray, target: FloatArray) -> float:
    """平均绝对误差 MAE。"""
    pred_array, target_array = _as_pair(pred, target)
    return float(np.mean(np.abs(pred_array - target_array)))


def rmse(pred: FloatArray, target: FloatArray) -> float:
    """均方根误差 RMSE。"""
    return float(np.sqrt(mse(pred, target)))


def mape(pred: FloatArray, target: FloatArray, eps: float = 1e-8) -> float:
    """平均绝对百分比误差 MAPE（百分数），忽略接近 0 的真值。

    Args:
        pred: 预测数组。
        target: 真值数组。
        eps: 视为 0 的阈值，避免除零。

    Returns:
        MAPE 百分数；当所有真值都接近 0 时返回 ``nan``。
    """
    pred_array, target_array = _as_pair(pred, target)
    mask = np.abs(target_array) > eps
    if not np.any(mask):
        return float("nan")
    ratio = np.abs((pred_array[mask] - target_array[mask]) / target_array[mask])
    return float(np.mean(ratio) * 100.0)


def compute_all(pred: FloatArray, target: FloatArray) -> dict[str, float]:
    """一次返回 MSE、MAE、RMSE、MAPE 四个指标。"""
    return {
        "mse": mse(pred, target),
        "mae": mae(pred, target),
        "rmse": rmse(pred, target),
        "mape": mape(pred, target),
    }
