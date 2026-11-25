"""导出变量维注意力权重，用于可解释性分析。

用途：
    读取一次已完成运行目录中的 ``config.json``、``model.pt`` 与数据文件，
    对测试集前若干窗口做前向，调用 :meth:`ITransformer.attention_map` 取出
    指定层的变量间注意力权重，落盘并画热图：

    * ``attention.npy``：形状 ``(窗口数, n_heads, N, N)`` 的原始权重；
    * ``attention_mean.npy``：对窗口与头求平均后的 ``(N, N)``；
    * ``attention_heatmap.png``：``(N, N)`` 热图（变量数 > 32 时省略刻度）；
    * ``attention_topk.json``：注意力最强的变量对，便于写失败案例。

    注意力是"变量 token 之间的相关强度"，不是因果；论文用它说明 iTransformer
    学到的变量间关系与数据集已知结构一致，报告里需要与领域知识对照解读。

前置条件：
    run 目录中已有 ``config.json``、``model.pt``，且数据文件可读；
    模型必须是 iTransformer 且该层 ``variate_module == "attention"``。

示例命令：
    python scripts/attention_maps.py results/raw/itransformer/ETTh1/h96/seed2021
    python scripts/attention_maps.py results/raw/itransformer/ECL/h96/seed2021 \\
        --layer 0 --num-windows 32 --top-k 20
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Windows + Anaconda 下 Pillow/matplotlib 与 torch 各带一份 libiomp5md.dll，
# 画热图时会触发 OMP Error #15。只影响本脚本的绘图进程，不影响 torch 计算结果。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from itransformer_repro.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
    validate_config,
)
from itransformer_repro.data import DataBundle, build_datasets, resolve_path
from itransformer_repro.models import ITransformer, build_model
from itransformer_repro.utils import load_json, save_json

DEFAULT_OUTPUT_NAME = "attention"


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="attention_maps.py",
        description="导出 iTransformer 的变量维注意力权重并画热图。",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="已完成运行的目录，需包含 config.json 与 model.pt。",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="提取哪一层的注意力，默认最后一层（-1）。",
    )
    parser.add_argument(
        "--num-windows",
        type=int,
        default=8,
        help="使用测试集前多少个窗口，默认 8。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="输出注意力最强的变量对数量，默认 15。",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出目录，默认 <run_dir>/attention。",
    )
    return parser


def load_run_config(run_dir: Path) -> ExperimentConfig:
    """从运行目录的 ``config.json`` 重建实验配置。

    Args:
        run_dir: 运行目录。

    Returns:
        校验通过的配置对象。

    Raises:
        FileNotFoundError: 缺少 ``config.json``。
        ValueError: 配置文件字段非法。
    """
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"缺少 {config_path}")
    payload = load_json(config_path)
    config = ExperimentConfig(
        experiment=str(payload["experiment"]),
        seed=int(payload["seed"]),
        device="cpu",
        runs_dir=str(payload.get("runs_dir", "results/raw")),
        data=DataConfig(**payload["data"]),
        model=ModelConfig(**payload["model"]),
        train=TrainConfig(**payload["train"]),
    )
    validate_config(config)
    return config


def read_variable_names(config: DataConfig, n_vars: int) -> list[str]:
    """读取 CSV 表头中的变量名（去掉 ``date`` 列）。

    Args:
        config: 数据配置。
        n_vars: 期望的变量数。

    Returns:
        长度为 ``n_vars`` 的名称列表；表头不可用时退回 ``var0..varN``。
    """
    path = resolve_path(config)
    fallback = [f"var{index}" for index in range(n_vars)]
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
    except (OSError, StopIteration):
        return fallback
    names = [name for name in header if name.strip().lower() != "date"]
    if len(names) != n_vars:
        return fallback
    return [name.strip() or f"var{index}" for index, name in enumerate(names)]


def load_model(run_dir: Path, config: ExperimentConfig, n_vars: int) -> ITransformer:
    """按运行配置重建模型并加载权重。

    Args:
        run_dir: 运行目录。
        config: 实验配置。
        n_vars: 数据集变量数。

    Returns:
        处于 ``eval`` 模式的模型。

    Raises:
        FileNotFoundError: 缺少 ``model.pt``。
        ValueError: 模型不是 iTransformer，或该层没有可提取的注意力。
    """
    if config.model.name != "itransformer":
        raise ValueError(
            f"只有 itransformer 支持变量维注意力导出，当前为 {config.model.name}"
        )
    if config.model.variate_module != "attention":
        raise ValueError(
            "该配置关闭了变量维注意力（variate_module != 'attention'），无法导出"
        )
    weights_path = run_dir / "model.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(f"缺少 {weights_path}")
    model = build_model(
        config.model,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        n_vars=n_vars,
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def collect_inputs(bundle: DataBundle, num_windows: int) -> torch.Tensor:
    """取测试集前 ``num_windows`` 个回看窗口。

    Args:
        bundle: 数据集窗口集合。
        num_windows: 窗口数。

    Returns:
        形状 ``(num_windows, T, N)`` 的输入张量。

    Raises:
        ValueError: 测试集窗口数不足。
    """
    available = len(bundle.test)
    if available < num_windows:
        raise ValueError(
            f"测试集只有 {available} 个窗口，少于 --num-windows={num_windows}"
        )
    return torch.stack([bundle.test[index][0] for index in range(num_windows)])


def summarize_top_pairs(
    mean_map: np.ndarray, names: Sequence[str], top_k: int
) -> list[dict[str, Any]]:
    """取注意力均值矩阵中最强的若干变量对（不含自注意力）。

    Args:
        mean_map: 形状 ``(N, N)`` 的平均注意力。
        names: 变量名。
        top_k: 返回条数。

    Returns:
        按权重降序排列的 ``{"source", "target", "weight"}`` 列表。
    """
    n_vars = mean_map.shape[0]
    pairs: list[tuple[float, int, int]] = []
    for source in range(n_vars):
        for target in range(n_vars):
            if source == target:
                continue
            pairs.append((float(mean_map[source, target]), source, target))
    pairs.sort(reverse=True)
    return [
        {"source": names[source], "target": names[target], "weight": round(weight, 6)}
        for weight, source, target in pairs[: max(0, top_k)]
    ]


def plot_heatmap(
    mean_map: np.ndarray, names: Sequence[str], title: str, path: Path
) -> bool:
    """画出注意力热图。

    Args:
        mean_map: 形状 ``(N, N)`` 的平均注意力。
        names: 变量名。
        title: 图标题。
        path: PNG 输出路径。

    Returns:
        ``True`` 表示成功保存；缺少 matplotlib 时返回 ``False``。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib 在 requirements 中
        return False

    n_vars = mean_map.shape[0]
    side = max(4.0, min(14.0, 0.28 * n_vars))
    figure, axes = plt.subplots(figsize=(side, side))
    image = axes.imshow(mean_map, cmap="viridis")
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    axes.set_title(title, fontsize=9)
    axes.set_xlabel("target variate")
    axes.set_ylabel("source variate")
    if n_vars <= 32:
        positions = np.arange(n_vars)
        axes.set_xticks(positions)
        axes.set_yticks(positions)
        axes.set_xticklabels(names, rotation=90, fontsize=7)
        axes.set_yticklabels(names, fontsize=7)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """执行注意力导出并返回退出码。"""
    args = build_parser().parse_args(argv)
    run_dir: Path = args.run_dir
    if args.num_windows <= 0:
        print("ERROR: --num-windows 必须为正整数")
        return 1

    try:
        config = load_run_config(run_dir)
        bundle = build_datasets(config.data)
        model = load_model(run_dir, config, bundle.n_vars)
        inputs = collect_inputs(bundle, args.num_windows)
        with torch.no_grad():
            attention = model.attention_map(inputs, layer_index=args.layer)
    except (FileNotFoundError, ValueError, RuntimeError, IndexError) as exc:
        print(f"ERROR: {exc}")
        return 1

    weights = attention.detach().cpu().numpy()
    mean_map = weights.mean(axis=(0, 1))
    names = read_variable_names(config.data, mean_map.shape[0])
    out_dir: Path = args.out if args.out is not None else run_dir / DEFAULT_OUTPUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "attention.npy", weights.astype(np.float32))
    np.save(out_dir / "attention_mean.npy", mean_map.astype(np.float32))
    top_pairs = summarize_top_pairs(mean_map, names, args.top_k)
    save_json(
        out_dir / "attention_topk.json",
        {
            "run_dir": str(run_dir),
            "dataset": config.data.name,
            "layer": args.layer,
            "num_windows": args.num_windows,
            "n_vars": int(mean_map.shape[0]),
            "top_pairs": top_pairs,
        },
    )

    title = (
        f"{config.model.name} / {config.data.name} / layer={args.layer} / "
        f"mean over {args.num_windows} windows & heads"
    )
    plotted = plot_heatmap(mean_map, names, title, out_dir / "attention_heatmap.png")

    print(
        f"导出 {config.data.name} 变量注意力：shape={weights.shape} "
        f"layer={args.layer} out={out_dir}"
    )
    if not plotted:
        print("WARN: 未安装 matplotlib，跳过热图")
    print("注意力最强的变量对（不含自注意力）：")
    for pair in top_pairs[: min(5, len(top_pairs))]:
        print(f"  {pair['source']} -> {pair['target']}: {pair['weight']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
