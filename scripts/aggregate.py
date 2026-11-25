"""汇总 ``results/raw/`` 下的全部运行结果，输出 mean±std 表格与 CSV。

用途：
    递归扫描 ``results/raw/**/seed*/metrics.json``，按
    ``(experiment, dataset, model, seq_len, pred_len)`` 分组，计算 MSE/MAE 的
    均值与样本标准差、种子数、平均参数量、平均训练耗时、推理耗时与峰值显存，
    写入 ``results/summary/summary.csv`` 和 ``results/summary/main_table.md``，
    并在终端打印 Markdown 表格。标准差在种子数小于 2 时留空，避免把单次运行
    误当作稳定结论。

前置条件：
    已执行 ``python -m pip install -r requirements.txt``，且 ``results/raw/`` 中至少有
    一次完成的训练（可用 ``scripts/run_matrix.py`` 生成）。

示例命令：
    python scripts/aggregate.py
    python scripts/aggregate.py --runs-dir results/raw \
        --out results/summary/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from itransformer_repro.utils import load_json

logger = logging.getLogger(__name__)

REQUIRED_METRIC_KEYS = (
    "experiment",
    "dataset",
    "model",
    "seq_len",
    "pred_len",
    "mse",
    "mae",
    "n_params",
    "train_time_sec",
)

SUMMARY_FIELDS = (
    "experiment",
    "dataset",
    "model",
    "seq_len",
    "pred_len",
    "n_seeds",
    "mse_mean",
    "mse_std",
    "mae_mean",
    "mae_std",
    "params_mean",
    "train_time_mean_sec",
    "inference_time_mean_sec",
    "peak_memory_mean_mb",
)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    Returns:
        配置好 ``--runs-dir`` 与 ``--out`` 的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="aggregate.py",
        description="汇总 results/raw 下的 metrics.json，输出 mean±std 表格与 CSV。",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("results/raw"),
        help="运行结果根目录，默认 results/raw。",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/summary/summary.csv"),
        help="汇总 CSV 输出路径，默认 results/summary/summary.csv。",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("results/summary/main_table.md"),
        help="Markdown 主表输出路径，默认 results/summary/main_table.md。",
    )
    return parser


def iter_metrics(runs_dir: Path) -> list[dict[str, Any]]:
    """收集 ``runs_dir`` 下所有 ``seed*/metrics.json``。

    Args:
        runs_dir: 运行结果根目录。

    Returns:
        解析成功的指标字典列表；无法解析或字段不完整的文件会被跳过并记录警告。

    Raises:
        FileNotFoundError: ``runs_dir`` 不存在。
    """
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"结果目录不存在：{runs_dir}")

    records: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_dir.rglob("metrics.json")):
        if not metrics_path.parent.name.startswith("seed"):
            continue
        try:
            payload = load_json(metrics_path)
        except (OSError, ValueError) as exc:
            logger.warning("跳过无法解析的结果 %s：%s", metrics_path, exc)
            continue
        if not set(REQUIRED_METRIC_KEYS).issubset(payload):
            logger.warning("跳过字段不完整的结果 %s", metrics_path)
            continue
        records.append(payload)
    return records


def std_or_none(values: Sequence[float]) -> float | None:
    """返回样本标准差（分母 ``n-1``）。

    Args:
        values: 同一分组的指标值列表。

    Returns:
        样本数不小于 2 时为样本标准差，否则为 ``None``。
    """
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def summarize(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """按数据集、模型与窗口长度聚合运行结果。

    Args:
        records: :func:`iter_metrics` 返回的指标字典列表。

    Returns:
        按 ``(experiment, dataset, model, seq_len, pred_len)`` 排序的汇总行列表；
        每行含均值、样本标准差、种子数、平均参数量与平均训练耗时。
    """
    groups: dict[tuple[str, str, str, int, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for record in records:
        key = (
            str(record["experiment"]),
            str(record["dataset"]),
            str(record["model"]),
            int(record["seq_len"]),
            int(record["pred_len"]),
        )
        groups[key].append(record)

    rows: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        experiment, dataset, model, seq_len, pred_len = key
        mse_values = [float(item["mse"]) for item in items]
        mae_values = [float(item["mae"]) for item in items]
        params_values = [float(item["n_params"]) for item in items]
        time_values = [float(item["train_time_sec"]) for item in items]
        inference_values = [
            float(item["inference_time_sec"])
            for item in items
            if "inference_time_sec" in item
        ]
        memory_values = [
            float(item["peak_memory_mb"]) for item in items if "peak_memory_mb" in item
        ]
        rows.append(
            {
                "experiment": experiment,
                "dataset": dataset,
                "model": model,
                "seq_len": seq_len,
                "pred_len": pred_len,
                "n_seeds": len(items),
                "mse_mean": statistics.fmean(mse_values),
                "mse_std": std_or_none(mse_values),
                "mae_mean": statistics.fmean(mae_values),
                "mae_std": std_or_none(mae_values),
                "params_mean": int(round(statistics.fmean(params_values))),
                "train_time_mean_sec": statistics.fmean(time_values),
                "inference_time_mean_sec": (
                    statistics.fmean(inference_values) if inference_values else None
                ),
                "peak_memory_mean_mb": (
                    statistics.fmean(memory_values) if memory_values else None
                ),
            }
        )
    return rows


def format_float(value: float | None, digits: int = 6) -> str:
    """格式化浮点数。

    Args:
        value: 待格式化的值；``None`` 表示不可计算。
        digits: 小数位数。

    Returns:
        固定小数位字符串；``None`` 返回空字符串。
    """
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def csv_row(row: dict[str, Any]) -> dict[str, str]:
    """把一行汇总结果转换为 CSV 字符串字典。

    Args:
        row: :func:`summarize` 返回的一行。

    Returns:
        各字段均已格式化的字符串字典。
    """
    return {
        "experiment": str(row["experiment"]),
        "dataset": str(row["dataset"]),
        "model": str(row["model"]),
        "seq_len": str(row["seq_len"]),
        "pred_len": str(row["pred_len"]),
        "n_seeds": str(row["n_seeds"]),
        "mse_mean": format_float(row["mse_mean"]),
        "mse_std": format_float(row["mse_std"]),
        "mae_mean": format_float(row["mae_mean"]),
        "mae_std": format_float(row["mae_std"]),
        "params_mean": str(row["params_mean"]),
        "train_time_mean_sec": format_float(row["train_time_mean_sec"], digits=2),
        "inference_time_mean_sec": format_float(
            row["inference_time_mean_sec"], digits=2
        ),
        "peak_memory_mean_mb": format_float(row["peak_memory_mean_mb"], digits=1),
    }


def write_summary(rows: Sequence[dict[str, Any]], out_path: Path) -> None:
    """把汇总行写入 CSV，父目录会自动创建。

    Args:
        rows: :func:`summarize` 的返回值。
        out_path: 目标 CSV 路径。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(csv_row(row))


def markdown_table(rows: Sequence[dict[str, Any]]) -> str:
    """生成 Markdown 汇总表文本。

    Args:
        rows: :func:`summarize` 的返回值。

    Returns:
        可直接写入 ``main_table.md`` 的 Markdown 文本。
    """
    lines = [
        "| experiment | dataset | model | seq_len | pred_len | n_seeds | "
        "mse_mean | mse_std | mae_mean | mae_std | params | train_s | "
        "infer_s | peak_mb |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- | --- |",
    ]
    for row in rows:
        mse_std = format_float(row["mse_std"]) or "-"
        mae_std = format_float(row["mae_std"]) or "-"
        lines.append(
            f"| {row['experiment']} | {row['dataset']} | {row['model']} | "
            f"{row['seq_len']} | "
            f"{row['pred_len']} | {row['n_seeds']} | "
            f"{format_float(row['mse_mean'])} | {mse_std} | "
            f"{format_float(row['mae_mean'])} | {mae_std} | "
            f"{row['params_mean']} | "
            f"{format_float(row['train_time_mean_sec'], digits=2) or '-'} | "
            f"{format_float(row['inference_time_mean_sec'], digits=2) or '-'} | "
            f"{format_float(row['peak_memory_mean_mb'], digits=1) or '-'} |"
        )
    lines.append("")
    lines.append(
        "说明：std 为样本标准差（n-1）；n_seeds=1 时以 '-' 表示不可计算；"
        "peak_mb 在 CPU 上为 0。"
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """汇总运行结果并返回退出码。

    Args:
        argv: 命令行参数；``None`` 表示使用 ``sys.argv[1:]``。

    Returns:
        成功写出汇总时为 ``0``；没有可用结果时为 ``1``。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    try:
        records = iter_metrics(args.runs_dir)
    except FileNotFoundError as exc:
        print(f"错误：{exc}；请先运行 scripts/run_matrix.py 生成结果")
        return 1

    if not records:
        print(f"{args.runs_dir} 下没有任何 metrics.json，请先运行训练。")
        return 1

    rows = summarize(records)
    if not rows:
        print("没有可汇总的完整结果。")
        return 1

    write_summary(rows, args.out)
    table = markdown_table(rows)
    print(table)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(table, encoding="utf-8")
    under_seeded = [row for row in rows if int(row["n_seeds"]) < 3]
    if under_seeded:
        print(
            f"WARN: {len(under_seeded)} 个分组少于 3 个种子，"
            "主结果需补足 3 个种子后再写入报告。"
        )
    print(
        f"已写入 {args.out} 与 {args.markdown_out}"
        f"（{len(rows)} 行，来自 {len(records)} 次运行）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
