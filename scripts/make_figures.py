"""由汇总 CSV 生成三张结果图（主结果、模型对比、消融）。

用途：
    读取 ``scripts/aggregate.py`` 生成的 ``outputs/summary.csv``，在
    ``outputs/figures/`` 下生成：

    * ``mse_by_horizon.png``：每个数据集一个子图，画出各模型 MSE 随预测长度的
      变化；
    * ``model_comparison.png``：按数据集分组的柱状图，比较各模型跨预测长度的
      平均 MSE（有标准差时附带误差棒）；
    * ``ablation.png``：从 summary 的 model 列中筛出消融配置画柱状图；没有消融
      数据时跳过该图并记录日志，不算失败。

    为避开发行环境的中文字体缺失，图中文字统一使用英文。所有图以 ``Agg`` 后端
    离屏渲染，dpi=150。

前置条件：
    已执行 ``python -m pip install -e ".[dev]"``，且已运行
    ``python scripts/aggregate.py`` 生成汇总 CSV。

示例命令：
    python scripts/make_figures.py
    python scripts/make_figures.py --summary outputs/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("dataset", "model", "pred_len", "mse_mean")

ABLATION_KEYWORDS = (
    "ablation",
    "abl_",
    "no_attn",
    "no_ffn",
    "no_layernorm",
    "pos_embed",
    "wo_attn",
    "wo_ffn",
)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    Returns:
        配置好 ``--summary`` 的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="make_figures.py",
        description="由 outputs/summary.csv 生成 mse_by_horizon、"
        "model_comparison 与 ablation 三张图。",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/summary.csv"),
        help="汇总 CSV 路径，默认 outputs/summary.csv；图片写入其同级 figures/ 目录。",
    )
    return parser


def read_summary(path: Path) -> list[dict[str, str]]:
    """读取汇总 CSV。

    Args:
        path: 汇总 CSV 路径。

    Returns:
        每行一个字符串字典的列表。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 缺少必需列。
    """
    if not path.is_file():
        raise FileNotFoundError(f"汇总文件不存在：{path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(
                f"{path} 缺少列：{', '.join(missing)}；请重新运行 aggregate.py"
            )
        return list(reader)


def mse_value(row: dict[str, str]) -> float | None:
    """把一行的 ``mse_mean`` 转为浮点数。

    Args:
        row: 汇总 CSV 的一行。

    Returns:
        解析成功时返回浮点数，空值或非法值返回 ``None``。
    """
    raw = row.get("mse_mean", "")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def stats_by_model(
    rows: Sequence[dict[str, str]], column: str = "model"
) -> dict[tuple[str, str], tuple[float, float | None]]:
    """按 ``(dataset, column)`` 汇总跨预测长度的 MSE 均值与误差。

    均值是各预测长度 ``mse_mean`` 的算术平均；误差取各预测长度 ``mse_std``
    的平均值（即种子间波动），全部缺失时返回 ``None``。

    Args:
        rows: 汇总 CSV 的行列表。
        column: 分组列，默认 ``model``；画消融图时传 ``experiment``，避免多个
            消融配置被同一个模型名合并。

    Returns:
        映射 ``(dataset, column 取值) -> (mean, error)``。
    """
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    spreads: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = mse_value(row)
        if value is None:
            continue
        key = (row["dataset"], row[column])
        grouped[key].append(value)
        raw_std = row.get("mse_std", "")
        if not raw_std:
            continue
        try:
            spreads[key].append(float(raw_std))
        except ValueError:
            logger.warning("忽略非法标准差 %r", raw_std)

    stats: dict[tuple[str, str], tuple[float, float | None]] = {}
    for key, values in grouped.items():
        spread = spreads.get(key, [])
        error = statistics.fmean(spread) if spread else None
        stats[key] = (statistics.fmean(values), error)
    return stats


def is_ablation_model(name: str) -> bool:
    """按模型名判断是否为消融配置。

    Args:
        name: 汇总 CSV 的 model 列取值。

    Returns:
        名称包含任一消融关键词时为 ``True``。
    """
    lowered = name.lower()
    return any(keyword in lowered for keyword in ABLATION_KEYWORDS)


def plot_mse_by_horizon(rows: Sequence[dict[str, str]], out_path: Path) -> None:
    """每个数据集一个子图，画出各模型 MSE 随预测长度的折线。

    Args:
        rows: 汇总 CSV 的行列表。
        out_path: 输出图片路径。
    """
    datasets = sorted({row["dataset"] for row in rows})
    models = sorted({row["model"] for row in rows})
    n_cols = min(3, len(datasets))
    n_rows = math.ceil(len(datasets) / n_cols)
    figure, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.8 * n_rows), squeeze=False
    )

    for index, dataset in enumerate(datasets):
        axis = axes[index // n_cols][index % n_cols]
        for model in models:
            points: list[tuple[int, float]] = []
            for row in rows:
                if row["dataset"] != dataset or row["model"] != model:
                    continue
                value = mse_value(row)
                if value is None:
                    continue
                points.append((int(row["pred_len"]), value))
            if not points:
                continue
            points.sort()
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            axis.plot(x_values, y_values, marker="o", label=model)
        axis.set_title(dataset)
        axis.set_xlabel("pred_len")
        axis.set_ylabel("MSE (normalized)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    for index in range(len(datasets), n_rows * n_cols):
        figure.delaxes(axes[index // n_cols][index % n_cols])
    figure.suptitle("Test MSE by forecast horizon")
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def plot_model_comparison(rows: Sequence[dict[str, str]], out_path: Path) -> None:
    """按数据集分组的柱状图，比较各模型跨预测长度的平均 MSE。

    Args:
        rows: 汇总 CSV 的行列表。
        out_path: 输出图片路径。
    """
    stats = stats_by_model(rows)
    datasets = sorted({key[0] for key in stats})
    all_models = sorted({key[1] for key in stats})
    models = [
        model
        for model in all_models
        if all((dataset, model) in stats for dataset in datasets)
    ]
    if not models:
        models = all_models

    positions = np.arange(len(datasets), dtype=float)
    width = 0.8 / max(len(models), 1)
    figure, axis = plt.subplots(
        figsize=(max(6.0, 1.6 * len(datasets) * len(models)), 4.5)
    )
    for index, model in enumerate(models):
        heights = [
            stats[(dataset, model)][0] if (dataset, model) in stats else 0.0
            for dataset in datasets
        ]
        stds = [
            stats[(dataset, model)][1] if (dataset, model) in stats else None
            for dataset in datasets
        ]
        error_bars: list[float] | None = None
        if all(value is not None for value in stds):
            error_bars = [0.0 if value is None else value for value in stds]
        axis.bar(
            positions + index * width,
            heights,
            width,
            label=model,
            yerr=error_bars,
            capsize=3,
        )
    axis.set_xticks(positions + width * (len(models) - 1) / 2)
    axis.set_xticklabels(datasets)
    axis.set_ylabel("Mean MSE (normalized)")
    axis.set_title("Model comparison: mean MSE across tested horizons")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def is_ablation_row(row: dict[str, str]) -> bool:
    """按 experiment 名（其次按模型名）判断是否为消融配置。

    Args:
        row: 汇总 CSV 的一行。

    Returns:
        experiment 含 ``ablation`` 或模型名命中消融关键词时为 ``True``。
    """
    experiment = row.get("experiment", "")
    if "ablation" in experiment.lower():
        return True
    return is_ablation_model(row.get("model", ""))


def plot_ablation(rows: Sequence[dict[str, str]], out_path: Path) -> bool:
    """画出消融结果柱状图；没有消融数据时跳过。

    每个消融配置一根柱子，高度为其在各数据集、各预测长度上的 MSE 均值。

    Args:
        rows: 汇总 CSV 的行列表。
        out_path: 输出图片路径。

    Returns:
        ``True`` 表示已生成图片，``False`` 表示 summary 中没有消融行。
    """
    ablation_rows = [row for row in rows if is_ablation_row(row)]
    if not ablation_rows:
        logger.info("summary 中没有消融结果（experiment 名不含 ablation），跳过")
        return False

    stats = stats_by_model(ablation_rows, column="experiment")
    per_experiment: dict[str, list[float]] = defaultdict(list)
    for (_dataset, experiment), (mean, _error) in stats.items():
        per_experiment[experiment].append(mean)
    experiments = sorted(per_experiment)
    heights = [
        statistics.fmean(per_experiment[experiment]) for experiment in experiments
    ]

    positions = np.arange(len(experiments), dtype=float)
    figure, axis = plt.subplots(figsize=(max(6.0, 1.1 * len(experiments)), 4.8))
    axis.bar(positions, heights, 0.6, color="#4c72b0")
    axis.set_xticks(positions)
    axis.set_xticklabels(experiments, rotation=20, ha="right", fontsize=8)
    axis.set_ylabel("Mean MSE (normalized)")
    axis.set_title("Ablation: mean MSE across tested horizons")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """生成图表并返回退出码。

    Args:
        argv: 命令行参数；``None`` 表示使用 ``sys.argv[1:]``。

    Returns:
        生成成功时为 ``0``；缺少汇总文件或没有数据行时为 ``1``。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    try:
        rows = read_summary(args.summary)
    except FileNotFoundError:
        print(f"未找到 {args.summary}，请先运行 python scripts/aggregate.py")
        return 1
    except (OSError, ValueError) as exc:
        logger.error("读取汇总失败：%s", exc)
        return 1

    if not rows:
        print(f"{args.summary} 中没有数据行。")
        return 1

    figures_dir = args.summary.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_mse_by_horizon(rows, figures_dir / "mse_by_horizon.png")
    plot_model_comparison(rows, figures_dir / "model_comparison.png")
    has_ablation = plot_ablation(rows, figures_dir / "ablation.png")

    names = ["mse_by_horizon.png", "model_comparison.png"]
    if has_ablation:
        names.append("ablation.png")
    print(f"已生成 {len(names)} 张图到 {figures_dir}：{', '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
