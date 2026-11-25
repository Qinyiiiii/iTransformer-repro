"""分析单次运行的预测误差，输出逐 horizon、逐变量与最差样本。

用途：
    读取一次训练产出的 ``pred.npy`` 与 ``true.npy``（形状 ``(n_windows, S, N)``，
    标准化空间），计算：

    * 每个预测步的 MSE/MAE，用来看误差是否随 horizon 累积；
    * 每个变量的 MSE/MAE，用来看哪些通道最难预测；
    * 按窗口平均误差排序的最差 ``top-k`` 样本，供失败案例分析。

    结果写入运行目录下的 ``error_analysis.json`` 与 ``worst_cases.csv``。该脚本
    只做分析，不修改 ``metrics.json``。

前置条件：
    该 run 目录中存在 ``pred.npy`` 与 ``true.npy``（``run_experiment`` 会生成）。

示例命令：
    python scripts/analyze_errors.py \
        results/raw/itransformer/ETTh1/h96/seed2021 --top-k 20
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="analyze_errors.py",
        description="分析 pred.npy/true.npy，输出逐 horizon/变量误差与最差样本。",
    )
    parser.add_argument(
        "run_dir", type=Path, help="包含 pred.npy 与 true.npy 的运行目录。"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="输出的最差窗口数，默认 20。",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON 输出路径，默认 <run_dir>/error_analysis.json。",
    )
    return parser


def _load_arrays(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """加载并校验预测/真值数组。"""
    pred_path = run_dir / "pred.npy"
    true_path = run_dir / "true.npy"
    for path in (pred_path, true_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少 {path}；请先完成一次训练")
    pred = np.load(pred_path).astype(np.float64)
    target = np.load(true_path).astype(np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"pred 与 true 形状不一致：{pred.shape} vs {target.shape}")
    if pred.ndim != 3:
        raise ValueError(f"期望 (n_windows, pred_len, n_vars)，实际 {pred.shape}")
    return pred, target


def _summarize_vector(
    pred: np.ndarray, target: np.ndarray, axis: int | tuple[int, ...]
) -> dict[str, list[float]]:
    """沿 ``axis`` 聚合 MSE/MAE，返回 Python 列表。"""
    squared = np.mean((pred - target) ** 2, axis=axis)
    absolute = np.mean(np.abs(pred - target), axis=axis)
    return {
        "mse": [round(float(value), 8) for value in squared],
        "mae": [round(float(value), 8) for value in absolute],
    }


def analyze(run_dir: Path, top_k: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """计算误差摘要与最差样本。

    Args:
        run_dir: 运行目录。
        top_k: 最差样本数量。

    Returns:
        ``(JSON 可序列化的摘要, 最差样本行列表)``。
    """
    pred, target = _load_arrays(run_dir)
    n_windows, pred_len, n_vars = pred.shape
    sample_mse = np.mean((pred - target) ** 2, axis=(1, 2))
    order = np.argsort(sample_mse)[::-1][: max(1, top_k)]
    worst: list[dict[str, Any]] = []
    for rank, index in enumerate(order, start=1):
        worst.append(
            {
                "rank": rank,
                "window_index": int(index),
                "sample_mse": round(float(sample_mse[index]), 8),
                "sample_mae": round(
                    float(np.mean(np.abs(pred[index] - target[index]))), 8
                ),
            }
        )
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "n_windows": int(n_windows),
        "pred_len": int(pred_len),
        "n_vars": int(n_vars),
        "overall": {
            "mse": round(float(np.mean((pred - target) ** 2)), 8),
            "mae": round(float(np.mean(np.abs(pred - target))), 8),
        },
        "by_horizon": _summarize_vector(pred, target, axis=(0, 2)),
        "by_variable": _summarize_vector(pred, target, axis=(0, 1)),
        "worst_windows": worst,
        "note": "误差在标准化空间计算；变量名顺序与数据文件去掉 date 列后的顺序一致。",
    }
    return summary, worst


def write_outputs(
    run_dir: Path,
    out: Path | None,
    summary: dict[str, Any],
    worst: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """写入 JSON 摘要与最差样本 CSV。"""
    out = out or run_dir / "error_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = out.with_name("worst_cases.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["rank", "window_index", "sample_mse", "sample_mae"]
        )
        writer.writeheader()
        writer.writerows(worst)
    return out, csv_path


def main(argv: Sequence[str] | None = None) -> int:
    """执行误差分析并返回退出码。"""
    args = build_parser().parse_args(argv)
    if args.top_k <= 0:
        raise SystemExit("--top-k 必须为正整数")
    try:
        summary, worst = analyze(args.run_dir, args.top_k)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    out, csv_path = write_outputs(args.run_dir, args.out, summary, worst)
    print(f"已写入 {out} 与 {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
