"""训练循环、评估流程与单次实验编排。"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from itransformer_repro.config import ExperimentConfig
from itransformer_repro.data import build_datasets
from itransformer_repro.metrics import compute_all
from itransformer_repro.models import build_model
from itransformer_repro.utils import (
    count_parameters,
    environment_info,
    get_logger,
    git_commit,
    resolve_device,
    save_json,
    seed_everything,
)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """把窗口数据集包装为 ``DataLoader``。

    Raises:
        ValueError: 数据集没有任何窗口（通常是切分或 ``seq_len`` 设置错误）。
    """
    if len(dataset) == 0:
        raise ValueError("数据集窗口数为 0，请检查切分边界与 seq_len/pred_len")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """训练一个 epoch，返回按样本数加权的平均损失。"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * inputs.size(0)
        total_samples += inputs.size(0)
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """在 ``loader`` 上推理并计算指标。

    Args:
        model: 待评估模型。
        loader: 数据加载器。
        device: 计算设备。
        criterion: 给定时会额外返回 ``loss`` 指标。

    Returns:
        ``(metrics, predictions, targets)``，两个数组形状均为
        ``(n_windows, pred_len, n_vars)``。
    """
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    prediction_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        predictions = model(inputs)
        if criterion is not None:
            loss_sum += float(criterion(predictions, targets).item()) * inputs.size(0)
        sample_count += inputs.size(0)
        prediction_batches.append(predictions.cpu().numpy())
        target_batches.append(targets.cpu().numpy())

    prediction_array = np.concatenate(prediction_batches, axis=0).astype(np.float64)
    target_array = np.concatenate(target_batches, axis=0).astype(np.float64)
    metrics = compute_all(prediction_array, target_array)
    if criterion is not None:
        metrics["loss"] = loss_sum / max(sample_count, 1)
    return metrics, prediction_array, target_array


def run_experiment(config: ExperimentConfig, force: bool = False) -> dict[str, Any]:
    """训练、验证、测试并落盘一次实验的全部证据。

    产出目录 ``runs/<experiment>/seed<seed>/`` 包含 ``config.json``、
    ``metrics.json``、``train.log``、``pred.npy`` 与 ``model.pt``。

    Args:
        config: 完整实验配置。
        force: 为 ``True`` 时覆盖已存在的运行结果。

    Returns:
        写入 ``metrics.json`` 的指标字典。

    Raises:
        FileExistsError: 目录中已有结果且 ``force`` 为 ``False``。
    """
    run_dir = config.run_dir()
    if (run_dir / "metrics.json").exists() and not force:
        raise FileExistsError(f"{run_dir} 已有结果；如需重跑请加 --force")

    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("itransformer_repro.train", run_dir / "train.log")
    seed_everything(config.seed)
    device = resolve_device(config.device)
    logger.info(
        "experiment=%s seed=%d device=%s",
        config.experiment,
        config.seed,
        device,
    )

    bundle = build_datasets(config.data)
    model = build_model(
        config.model,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        n_vars=bundle.n_vars,
    ).to(device)
    n_params = count_parameters(model)
    logger.info(
        "data=%s n_vars=%d train/val/test=%d/%d/%d params=%d",
        config.data.name,
        bundle.n_vars,
        len(bundle.train),
        len(bundle.val),
        len(bundle.test),
        n_params,
    )

    save_json(
        run_dir / "config.json",
        {
            **config.to_dict(),
            "git_commit": git_commit(),
            "environment": environment_info(),
            "n_params": n_params,
        },
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    train_loader = make_loader(
        bundle.train, config.train.batch_size, True, config.train.num_workers
    )
    val_loader = make_loader(
        bundle.val, config.train.batch_size, False, config.train.num_workers
    )
    test_loader = make_loader(
        bundle.test, config.train.batch_size, False, config.train.num_workers
    )

    best_val_mse = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_times: list[float] = []
    started_at = time.perf_counter()

    for epoch in range(1, config.train.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate_model(model, val_loader, device, criterion)
        duration = time.perf_counter() - epoch_start
        epoch_times.append(round(duration, 4))
        logger.info(
            "epoch %d/%d train_loss=%.6f val_mse=%.6f val_mae=%.6f %.2fs",
            epoch,
            config.train.epochs,
            train_loss,
            val_metrics["mse"],
            val_metrics["mae"],
            duration,
        )
        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                config.train.patience
                and epochs_without_improvement >= config.train.patience
            ):
                logger.info(
                    "验证集连续 %d 轮无提升，提前停止", epochs_without_improvement
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, predictions, _ = evaluate_model(model, test_loader, device, criterion)
    total_time = time.perf_counter() - started_at
    result: dict[str, Any] = {
        "experiment": config.experiment,
        "seed": config.seed,
        "dataset": config.data.name,
        "model": config.model.name,
        "seq_len": config.data.seq_len,
        "pred_len": config.data.pred_len,
        "mse": round(test_metrics["mse"], 6),
        "mae": round(test_metrics["mae"], 6),
        "rmse": round(test_metrics["rmse"], 6),
        "mape": round(test_metrics["mape"], 6),
        "best_val_mse": round(best_val_mse, 6),
        "best_epoch": best_epoch,
        "n_params": n_params,
        "n_windows": {
            "train": len(bundle.train),
            "val": len(bundle.val),
            "test": len(bundle.test),
        },
        "epochs_run": len(epoch_times),
        "epoch_time_sec": epoch_times,
        "train_time_sec": round(total_time, 2),
        "peak_memory_mb": _peak_memory_mb(device),
        "device": str(device),
    }

    save_json(run_dir / "metrics.json", result)
    np.save(run_dir / "pred.npy", predictions)
    torch.save(model.state_dict(), run_dir / "model.pt")
    logger.info(
        "finished test_mse=%.6f test_mae=%.6f time=%.1fs",
        result["mse"],
        result["mae"],
        total_time,
    )
    return result


def _peak_memory_mb(device: torch.device) -> float:
    """返回 GPU 峰值显存（MB）；CPU 上返回 ``0.0``。"""
    if device.type != "cuda":
        return 0.0
    return round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)


def run_exists(run_dir: Path) -> bool:
    """``run_dir`` 中是否已有完成的运行（以 ``metrics.json`` 为准）。"""
    return (run_dir / "metrics.json").exists()
