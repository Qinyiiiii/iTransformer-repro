"""检查本地基准数据是否就绪，并打印手工下载指引（脚本本身不联网）。

用途：
    遍历 :data:`itransformer_repro.data.DATASET_SPECS`，逐个检查 ``--root`` 下的
    CSV 是否存在、变量数是否与注册规格一致，并在文件缺失时打印数据来源与放置
    位置。脚本不会下载任何数据，也不会创建占位文件。

前置条件：
    已在仓库根目录执行 ``python -m pip install -e ".[dev]"``；数据按 README 的
    说明手工下载并放到 ``data/raw``。

示例命令：
    python scripts/download_data.py
    python scripts/download_data.py --dataset ETTh1 Weather
    python scripts/download_data.py --strict-vars --json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from itransformer_repro.config import DataConfig
from itransformer_repro.data import DATASET_SPECS, resolve_path

logger = logging.getLogger(__name__)

ETT_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
ETT_SOURCE = "https://github.com/zhouhaoyi/ETDataset"
OFFICIAL_SOURCE = "https://github.com/thuml/iTransformer"


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    Returns:
        配置好 ``--root``、``--dataset``、``--json`` 与 ``--strict-vars`` 的
        解析器。
    """
    parser = argparse.ArgumentParser(
        prog="download_data.py",
        description="检查各基准数据文件是否存在并打印下载指引（不联网）。",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw"),
        help="数据根目录，默认 data/raw。",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=sorted(DATASET_SPECS),
        default=None,
        help="只检查指定数据集，可传多个；默认检查全部已注册数据集。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="以 JSON 输出机器可读结果（默认打印人类可读清单）。",
    )
    parser.add_argument(
        "--strict-vars",
        action="store_true",
        help="文件存在但变量数与规格不符时也判定为失败。",
    )
    return parser


def count_variables(path: Path) -> int:
    """读取 CSV 表头，返回去掉时间列后的变量数。

    Args:
        path: CSV 文件路径。

    Returns:
        非 ``date`` 列的数量。

    Raises:
        OSError: 文件无法读取。
        ValueError: CSV 为空或表头无法解析。
    """
    frame = pd.read_csv(path, nrows=0)
    return sum(1 for name in frame.columns if str(name).lower() != "date")


def data_source_hint(dataset: str, path: Path) -> str:
    """返回缺失文件的下载来源与放置位置说明。

    Args:
        dataset: 数据集名。
        path: 期望的文件路径。

    Returns:
        面向终端的一行说明，不含任何密钥。
    """
    if dataset in ETT_DATASETS:
        source = f"{ETT_SOURCE} 中的 {dataset}.csv"
    else:
        source = f"{OFFICIAL_SOURCE} 仓库 README 中的数据链接"
    return f"请从 {source} 下载，并放到 {path.parent}"


def inspect_dataset(root: Path, dataset: str, strict_vars: bool) -> dict[str, Any]:
    """检查单个数据集，返回可直接序列化为 JSON 的记录。

    Args:
        root: 数据根目录，例如 ``data/raw``。
        dataset: 已注册的数据集名。
        strict_vars: 为 ``True`` 时变量数不符也视为失败。

    Returns:
        含 ``dataset``、``path``、``exists``、``expected_n_vars``、``n_vars``、
        ``ok``、``message`` 与 ``hint`` 的字典。
    """
    spec = DATASET_SPECS[dataset]
    config = DataConfig(name=dataset, root=str(root), file=spec.filename)
    path = resolve_path(config)
    record: dict[str, Any] = {
        "dataset": dataset,
        "path": str(path),
        "exists": path.exists(),
        "expected_n_vars": spec.n_vars,
        "n_vars": None,
        "ok": False,
        "message": "",
        "hint": "",
    }
    if not record["exists"]:
        record["message"] = "文件缺失"
        record["hint"] = data_source_hint(dataset, path)
        return record

    try:
        n_vars = count_variables(path)
    except (OSError, ValueError) as exc:
        record["message"] = f"无法解析：{exc}"
        return record

    record["n_vars"] = n_vars
    if n_vars == spec.n_vars:
        record["ok"] = True
        return record

    record["message"] = f"变量数不符：期望 {spec.n_vars}，实际 {n_vars}"
    record["ok"] = not strict_vars
    if not strict_vars:
        logger.warning(
            "%s 变量数不符（期望 %d，实际 %d），未启用 --strict-vars",
            dataset,
            spec.n_vars,
            n_vars,
        )
    return record


def print_record(record: dict[str, Any]) -> None:
    """把单条检查记录打印到终端。

    Args:
        record: :func:`inspect_dataset` 的返回值。
    """
    if record["ok"]:
        status = "[OK]"
    elif not record["exists"]:
        status = "[MISSING]"
    else:
        status = "[MISMATCH]"
    n_vars = record["n_vars"]
    vars_text = "-" if n_vars is None else str(n_vars)
    print(
        f"{status:<10} {record['dataset']:<9} "
        f"vars={vars_text}/{record['expected_n_vars']:<6} {record['path']}"
    )
    if record["message"]:
        print(f"           {record['message']}")
    if record["hint"]:
        print(f"           {record['hint']}")


def main(argv: Sequence[str] | None = None) -> int:
    """检查数据文件并返回退出码。

    Args:
        argv: 命令行参数；``None`` 表示使用 ``sys.argv[1:]``。

    Returns:
        全部就绪（且严格模式下变量数一致）返回 ``0``，否则返回 ``1``。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    datasets = args.dataset or sorted(DATASET_SPECS)
    records = [
        inspect_dataset(args.root, dataset, args.strict_vars) for dataset in datasets
    ]
    all_ok = all(record["ok"] for record in records)

    if args.as_json:
        payload = {
            "root": str(args.root),
            "strict_vars": args.strict_vars,
            "all_ok": all_ok,
            "datasets": records,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for record in records:
            print_record(record)
        missing = sum(1 for record in records if not record["ok"])
        print(f"共检查 {len(records)} 个数据集，{missing} 个未就绪。")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
