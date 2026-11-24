"""指标函数的单元测试。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from itransformer_repro.metrics import compute_all, mae, mape, mse, rmse

PRED = np.array([[1.0, 2.0, 3.0]])
TARGET = np.array([[1.0, 2.0, 4.0]])


def test_mse_returns_mean_squared_difference() -> None:
    assert mse(PRED, TARGET) == pytest.approx(1.0 / 3.0)


def test_mae_returns_mean_absolute_difference() -> None:
    assert mae(PRED, TARGET) == pytest.approx(1.0 / 3.0)


def test_rmse_is_square_root_of_mse() -> None:
    assert rmse(PRED, TARGET) == pytest.approx(math.sqrt(1.0 / 3.0))


def test_mape_is_percentage_and_ignores_zero_targets() -> None:
    pred = np.array([[1.0, 2.0, 3.0]])
    target = np.array([[0.0, 2.0, 4.0]])
    assert mape(pred, target) == pytest.approx(25.0 / 2.0)


def test_mape_returns_nan_when_all_targets_are_zero() -> None:
    assert math.isnan(mape(np.zeros((1, 3)), np.zeros((1, 3))))


def test_metrics_reject_different_shapes() -> None:
    with pytest.raises(ValueError, match="形状必须一致"):
        mse(np.zeros((2, 3)), np.zeros((2, 4)))


def test_metrics_reject_empty_arrays() -> None:
    with pytest.raises(ValueError, match="不能为空"):
        mae(np.zeros((0, 3)), np.zeros((0, 3)))


def test_compute_all_returns_four_metrics() -> None:
    metrics = compute_all(PRED, TARGET)
    assert set(metrics) == {"mse", "mae", "rmse", "mape"}
    assert metrics["mse"] == pytest.approx(metrics["rmse"] ** 2)
