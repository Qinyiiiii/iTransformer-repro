"""把项目结果与论文报告值做量级对照（不判定复现成败）。

用途：
    读取 ``scripts/aggregate.py`` 生成的 ``results/summary/summary.csv``，输出两张表：

    * **表 A（论文对照）**：只比较主方法 iTransformer。对每个数据集先在已跑的
      预测长度上求算术平均，再与硬编码的论文参考值比较，计算相对偏差
      ``(ours - ref) / ref``，按 ``--tolerance``（默认 10%%）标注 ``PASS`` / ``WARN``。
    * **表 B（基线与本项目的表内互比）**：DLinear / PatchTST / Transformer 只和
      本项目同数据集的 iTransformer 结果比较。基线的论文数字来自官方基准仓库
      的**原版实现**，与本项目的简化基线不可直接对照，因此不参与论文对照。

重要说明：
    参考值是论文**全量配置**（完整特征数、更长训练、官方超参）下的结果，本项目
    为课程缩减配置，绝对数值不可直接比较；这里的对比仅作量级参照，**不作为**
    “复现成功/失败”的判据。ETT 参考值是**各数据集自身**在论文表 4 的 4 档预测
    长度上的平均（例如 ETTh1 = 0.454/0.447），不做跨数据集平均。

前置条件：
    已执行 ``python -m pip install -r requirements.txt``，且已运行
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

# 论文表 4 中各 ETT 数据集在 96/192/336/720 四档上的 MSE/MAE 算术平均。
# 数值由原表四档结果直接平均得到，未做任何调参或挑选。
ETT_REFERENCE: dict[str, tuple[float, float]] = {
    "ETTh1": (0.454, 0.447),
    "ETTh2": (0.383, 0.407),
    "ETTm1": (0.407, 0.410),
    "ETTm2": (0.288, 0.332),
}
ETT_SCOPE = "论文表 4（4 档平均）"
PAPER_SCOPE = "论文原表"
MAIN_MODEL = "itransformer"

REQUIRED_COLUMNS = ("dataset", "model", "pred_len", "mse_mean", "mae_mean")
EXPECTED_HORIZONS = {96, 192, 336, 720}


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
        default=Path("results/summary/summary.csv"),
        help="汇总 CSV 路径，默认 results/summary/summary.csv。",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.10,
        help="相对偏差阈值，默认 0.10（即 10%%，与作业要求的量级一致）。",
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
    if dataset in ETT_REFERENCE:
        mse, mae = ETT_REFERENCE[dataset]
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
    # 允许以 ``#`` 开头的说明行（示例 CSV 用它标注“非论文结果”）。
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    reader = csv.DictReader(lines)
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


def _group_rows(
    rows: Sequence[dict[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    """按 ``(dataset, model)`` 分组，并剔除消融行与回看扫描行。

    Args:
        rows: :func:`read_summary` 的返回值。

    Returns:
        分组后的汇总行。
    """
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        experiment = row.get("experiment", "")
        if "ablation" in experiment.lower():
            continue  # 消融配置与主方法同名，不能混进主结果的平均
        if row.get("seq_len", "").isdigit() and int(row["seq_len"]) != 96:
            continue  # 回看扫描行不能混进主结果的平均
        groups[(row["dataset"], row["model"])].append(row)
    return groups


def _mean_pair(items: Sequence[dict[str, str]]) -> tuple[float, float] | None:
    """返回一组汇总行的 ``(MSE 均值, MAE 均值)``；无有效值时返回 ``None``。"""
    mse_values = read_values(items, "mse_mean")
    mae_values = read_values(items, "mae_mean")
    if not mse_values or not mae_values:
        return None
    return statistics.fmean(mse_values), statistics.fmean(mae_values)


def compare(rows: Sequence[dict[str, str]], tolerance: float) -> list[dict[str, Any]]:
    """只对主方法 iTransformer 做论文对照。

    基线（DLinear/PatchTST/Transformer）在本项目里是简化实现，与论文基线的
    官方实现不可直接比较，因此不进入本表；它们由 :func:`baseline_table` 输出。

    Args:
        rows: :func:`read_summary` 的返回值。
        tolerance: 相对偏差阈值，MSE 与 MAE 均在阈值内才算 ``PASS``。

    Returns:
        对比行列表；每行含预测长度覆盖、项目均值、参考值、相对偏差、参考口径与
        结论。没有参考值的数据集或非主方法的行会被跳过。
    """
    comparisons: list[dict[str, Any]] = []
    for (dataset, model), items in sorted(_group_rows(rows).items()):
        if model != MAIN_MODEL:
            continue
        reference = resolve_reference(dataset)
        if reference is None:
            logger.warning("数据集 %s 没有论文参考值，跳过", dataset)
            continue
        means = _mean_pair(items)
        if means is None:
            logger.warning("%s / %s 的指标为空，跳过", dataset, model)
            continue

        paper_mse, paper_mae, scope = reference
        ours_mse, ours_mae = means
        mse_rel = (ours_mse - paper_mse) / paper_mse
        mae_rel = (ours_mae - paper_mae) / paper_mae
        horizons = sorted({int(item["pred_len"]) for item in items})
        complete = set(horizons) == EXPECTED_HORIZONS
        passed = abs(mse_rel) <= tolerance and abs(mae_rel) <= tolerance and complete
        comparisons.append(
            {
                "dataset": dataset,
                "model": model,
                "horizons": ",".join(str(horizon) for horizon in horizons),
                "coverage": f"{len(set(horizons) & EXPECTED_HORIZONS)}/4",
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


def baseline_table(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    """生成基线与主方法的表内互比（同一数据集、同一组预测长度）。

    Args:
        rows: :func:`read_summary` 的返回值。

    Returns:
        每个基线一行，含自身均值以及与同数据集 iTransformer 均值的相对差值；
        若该数据集还没有主方法结果，差值为 ``None``（表格里显示 ``-``）。
    """
    main_means: dict[str, tuple[float, float]] = {}
    for (dataset, model), items in _group_rows(rows).items():
        if model != MAIN_MODEL:
            continue
        means = _mean_pair(items)
        if means is not None:
            main_means[dataset] = means

    table: list[dict[str, Any]] = []
    for (dataset, model), items in sorted(_group_rows(rows).items()):
        if model == MAIN_MODEL:
            continue
        means = _mean_pair(items)
        if means is None:
            continue
        ours_mse, ours_mae = means
        reference = main_means.get(dataset)
        table.append(
            {
                "dataset": dataset,
                "model": model,
                "horizons": ",".join(
                    str(horizon)
                    for horizon in sorted({int(item["pred_len"]) for item in items})
                ),
                "ours_mse": ours_mse,
                "ours_mae": ours_mae,
                "delta_mse": None
                if reference is None
                else (ours_mse - reference[0]) / reference[0],
                "delta_mae": None
                if reference is None
                else (ours_mae - reference[1]) / reference[1],
            }
        )
    return table


def print_table(comparisons: Sequence[dict[str, Any]]) -> None:
    """打印参考值说明与主方法的 Markdown 对比表。

    Args:
        comparisons: :func:`compare` 的返回值。
    """
    print(
        "参考值说明：论文全量配置结果；本项目为课程缩减配置，仅作量级对照，"
        "不作为“复现成功/失败”的判据。"
    )
    print(
        "ETT 参考值为各数据集自身在 4 个预测长度上的平均（论文表 4），不跨数据集平均。"
    )
    print("本表只包含主方法 iTransformer；基线的表内互比见下方表 B。")
    print()
    print(
        "| dataset | model | horizons | coverage | ours MSE | ours MAE | "
        "paper MSE | paper MAE | MSE 偏差 | MAE 偏差 | 参考口径 | 结论 |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in comparisons:
        print(
            f"| {row['dataset']} | {row['model']} | {row['horizons']} | "
            f"{row['coverage']} | "
            f"{row['ours_mse']:.4f} | {row['ours_mae']:.4f} | "
            f"{row['paper_mse']:.3f} | {row['paper_mae']:.3f} | "
            f"{row['mse_rel']:+.1%} | {row['mae_rel']:+.1%} | "
            f"{row['scope']} | {row['verdict']} |"
        )


def print_baseline_table(table: Sequence[dict[str, Any]]) -> None:
    """打印基线与同数据集 iTransformer 的表内互比。

    Args:
        table: :func:`baseline_table` 的返回值。
    """
    print()
    print(
        "表 B：基线与本项目主方法的互比（同一数据集、同一组预测长度）。"
        "基线为简化实现，不能与论文基线数字直接对照。"
    )
    print()
    print(
        "| dataset | model | horizons | MSE | MAE | "
        "相对 iTransformer MSE | 相对 iTransformer MAE |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for row in table:
        delta_mse = "-" if row["delta_mse"] is None else f"{row['delta_mse']:+.1%}"
        delta_mae = "-" if row["delta_mae"] is None else f"{row['delta_mae']:+.1%}"
        print(
            f"| {row['dataset']} | {row['model']} | {row['horizons']} | "
            f"{row['ours_mse']:.4f} | {row['ours_mae']:.4f} | "
            f"{delta_mse} | {delta_mae} |"
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
    baselines = baseline_table(rows)
    if not comparisons and not baselines:
        print("没有可对比的结果（汇总为空或数据集缺少论文参考值）。")
        return 1

    if comparisons:
        print_table(comparisons)
    else:
        print("暂无可与论文对照的 iTransformer 结果；只输出基线互比表。")
    if baselines:
        print_baseline_table(baselines)

    warn_count = sum(1 for row in comparisons if row["verdict"] == "WARN")
    if comparisons:
        print(
            f"共 {len(comparisons)} 行论文对照，PASS "
            f"{len(comparisons) - warn_count}，WARN {warn_count}；"
            f"另有 {len(baselines)} 行基线互比。"
        )
    if warn_count and args.fail_on_warn:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
