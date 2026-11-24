"""把项目结果与论文报告值做量级对照（不判定复现成败）。

用途：
    读取 ``scripts/aggregate.py`` 生成的 ``outputs/summary.csv``，对每个
    ``(dataset, model)`` 先在已跑的预测长度上求算术平均，再与硬编码的论文参考值
    比较，计算相对偏差 ``(ours - ref) / ref``，按 ``--tolerance`` 标注
    ``PASS`` / ``WARN`` 并打印 Markdown 表。

重要说明：
    参考值是论文**全量配置**（完整特征数、更长训练、官方超参）下的结果，本项目
    为课程缩减配置，绝对数值不可直接比较；这里的对比仅作量级参照，**不作为**
    “复现成功/失败”的判据。ETT 系列使用论文中 ETT 平均口径 0.383/0.399，输出
    时会注明“ETT 系列平均”。

前置条件：
    已执行 ``python -m pip install -e ".[dev]"``，且已运行
    ``python scripts/aggregate.py`` 生成汇总 CSV。

示例命令：
    python scripts/compare_paper.py
    python scripts/compare_paper.py --tolerance 0.30
    python scripts/compare_paper.py --fail-on-warn
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

logger = logging.getLogger(__name__)

PAPER_REFERENCE: dict[str, tuple[float, float]] = {
    "ECL": (0.178, 0.270),
    "Traffic": (0.428, 0.282),
    "Weather": (0.258, 0.278),
    "Exchange": (0.360, 0.403),
}

ETT_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
ETT_REFERENCE = (0.383, 0.399)
ETT_SCOPE = "ETT 系列平均"
PAPER_SCOPE = "论文原表"

REQUIRED_COLUMNS = ("dataset", "model", "pred_len", "mse_mean", "mae_mean")


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    Returns:
        配置好 ``--summary``、``--tolerance`` 与 ``--fail-on-warn`` 的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="compare_paper.py",
        description="把项目结果与论文报告值做量级对照（不作为复现成败判据）。",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/summary.csv"),
        help="汇总 CSV 路径，默认 outputs/summary.csv。",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.20,
        help="相对偏差阈值，默认 0.20（即 20%%）。",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="存在 WARN 行时返回非零退出码。",
    )
    return parser


def resolve_reference(dataset: str) -> tuple[float, float, str] | None:
    """返回数据集的论文参考值（4 个预测长度平均，MSE/MAE）。

    Args:
        dataset: 数据集名。

    Returns:
        ``(mse, mae, 口径说明)``；无参考值时返回 ``None``。
    """
    if dataset in PAPER_REFERENCE:
        mse, mae = PAPER_REFERENCE[dataset]
        return mse, mae, PAPER_SCOPE
    if dataset in ETT_DATASETS:
        mse, mae = ETT_REFERENCE
        return mse, mae, ETT_SCOPE
    return None


def read_summary(path: Path) -> list[dict[str, str]]:
    """读取 ``aggregate.py`` 生成的汇总 CSV。

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


def read_values(items: Sequence[dict[str, str]], column: str) -> list[float]:
    """读取某一列并转换为浮点数。

    Args:
        items: 同一分组的汇总行。
        column: 列名。

    Returns:
        有效数值列表；空值与非法数值会被忽略（非法值记录警告）。
    """
    values: list[float] = []
    for item in items:
        raw = item.get(column, "")
        if not raw:
            continue
        try:
            values.append(float(raw))
        except ValueError:
            logger.warning("忽略非法数值 %r（列 %s）", raw, column)
    return values


def compare(rows: Sequence[dict[str, str]], tolerance: float) -> list[dict[str, Any]]:
    """按 ``(dataset, model)`` 聚合已跑预测长度并与论文参考值比较。

    Args:
        rows: :func:`read_summary` 的返回值。
        tolerance: 相对偏差阈值，MSE 与 MAE 均在阈值内才算 ``PASS``。

    Returns:
        对比行列表；每行含预测长度覆盖、项目均值、参考值、相对偏差、参考口径与
        结论。没有参考值的数据集会被跳过。
    """
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        experiment = row.get("experiment", "")
        if "ablation" in experiment.lower():
            continue  # 消融配置与主方法同名，不能混进主结果的平均
        groups[(row["dataset"], row["model"])].append(row)

    comparisons: list[dict[str, Any]] = []
    for (dataset, model), items in sorted(groups.items()):
        reference = resolve_reference(dataset)
        if reference is None:
            logger.warning("数据集 %s 没有论文参考值，跳过", dataset)
            continue
        mse_values = read_values(items, "mse_mean")
        mae_values = read_values(items, "mae_mean")
        if not mse_values or not mae_values:
            logger.warning("%s / %s 的指标为空，跳过", dataset, model)
            continue

        paper_mse, paper_mae, scope = reference
        ours_mse = statistics.fmean(mse_values)
        ours_mae = statistics.fmean(mae_values)
        mse_rel = (ours_mse - paper_mse) / paper_mse
        mae_rel = (ours_mae - paper_mae) / paper_mae
        horizons = sorted({int(item["pred_len"]) for item in items})
        passed = abs(mse_rel) <= tolerance and abs(mae_rel) <= tolerance
        comparisons.append(
            {
                "dataset": dataset,
                "model": model,
                "horizons": ",".join(str(horizon) for horizon in horizons),
                "ours_mse": ours_mse,
                "ours_mae": ours_mae,
                "paper_mse": paper_mse,
                "paper_mae": paper_mae,
                "mse_rel": mse_rel,
                "mae_rel": mae_rel,
                "scope": scope,
                "verdict": "PASS" if passed else "WARN",
            }
        )
    return comparisons


def print_table(comparisons: Sequence[dict[str, Any]]) -> None:
    """打印参考值说明与 Markdown 对比表。

    Args:
        comparisons: :func:`compare` 的返回值。
    """
    print(
        "参考值说明：论文全量配置结果；本项目为课程缩减配置，仅作量级对照，"
        "不作为“复现成功/失败”的判据。"
    )
    print("ETT 系列参考值为四个 ETT 数据集在 4 个预测长度上的平均口径。")
    print()
    print(
        "| dataset | model | horizons | ours MSE | ours MAE | "
        "paper MSE | paper MAE | MSE 偏差 | MAE 偏差 | 参考口径 | 结论 |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in comparisons:
        print(
            f"| {row['dataset']} | {row['model']} | {row['horizons']} | "
            f"{row['ours_mse']:.4f} | {row['ours_mae']:.4f} | "
            f"{row['paper_mse']:.3f} | {row['paper_mae']:.3f} | "
            f"{row['mse_rel']:+.1%} | {row['mae_rel']:+.1%} | "
            f"{row['scope']} | {row['verdict']} |"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """执行论文对照并返回退出码。

    Args:
        argv: 命令行参数；``None`` 表示使用 ``sys.argv[1:]``。

    Returns:
        有对比行时为 ``0``（``--fail-on-warn`` 且有 WARN 时为 ``1``）；没有可对比
        结果或读取失败时为 ``1``。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if args.tolerance < 0:
        parser.error("--tolerance 不能为负数")

    try:
        rows = read_summary(args.summary)
    except FileNotFoundError:
        print(f"未找到 {args.summary}，请先运行 python scripts/aggregate.py")
        return 1
    except (OSError, ValueError) as exc:
        print(f"错误：{exc}")
        return 1

    comparisons = compare(rows, args.tolerance)
    if not comparisons:
        print("没有可对比的结果（汇总为空或数据集缺少论文参考值）。")
        return 1

    print_table(comparisons)
    warn_count = sum(1 for row in comparisons if row["verdict"] == "WARN")
    print(
        f"共 {len(comparisons)} 行对比，PASS "
        f"{len(comparisons) - warn_count}，WARN {warn_count}。"
    )
    if warn_count and args.fail_on_warn:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
