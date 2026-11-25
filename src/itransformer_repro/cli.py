"""命令行入口：``train``、``check-data`` 与 ``evaluate``。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from itransformer_repro.config import coerce_scalar, load_config, validate_config
from itransformer_repro.data import dataset_card
from itransformer_repro.evaluate import evaluate_run
from itransformer_repro.train import run_experiment
from itransformer_repro.utils import save_json

_OVERRIDE_FLAGS = {
    "seed": "seed",
    "epochs": "train.epochs",
    "batch_size": "train.batch_size",
    "lr": "train.learning_rate",
    "seq_len": "data.seq_len",
    "pred_len": "data.pred_len",
    "device": "device",
}


def build_parser() -> argparse.ArgumentParser:
    """构造带全部子命令的参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="itransformer-repro",
        description="复现 iTransformer 多变量长期时间序列预测实验。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="按 YAML 配置训练一次实验。")
    train.add_argument("--config", type=Path, required=True, help="YAML 配置文件路径。")
    train.add_argument("--seed", type=int, help="覆盖随机种子。")
    train.add_argument("--epochs", type=int, help="覆盖训练轮数。")
    train.add_argument("--batch-size", type=int, help="覆盖批大小。")
    train.add_argument("--lr", type=float, help="覆盖学习率。")
    train.add_argument("--seq-len", type=int, help="覆盖回看长度 T。")
    train.add_argument("--pred-len", type=int, help="覆盖预测长度 S。")
    train.add_argument("--device", help="cpu、cuda、cuda:1 或 auto。")
    train.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="点号覆盖，例如 model.d_model=256 model.variate_module=ffn。",
    )
    train.add_argument("--force", action="store_true", help="覆盖已存在的运行目录。")

    check = subparsers.add_parser("check-data", help="校验数据集并打印数据集卡片。")
    check.add_argument("--config", type=Path, required=True, help="YAML 配置文件路径。")
    check.add_argument(
        "--strict", action="store_true", help="行数与官方发布版本不符时也返回失败。"
    )
    check.add_argument("--out", type=Path, help="把数据集卡片另存为 JSON。")
    check.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="点号覆盖，例如 data.seq_len=192 data.pred_len=336。",
    )

    evaluate = subparsers.add_parser("evaluate", help="重算一次已完成运行的指标。")
    evaluate.add_argument("--run-dir", type=Path, required=True, help="运行输出目录。")
    return parser


def collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """把专用参数与 ``--set KEY=VALUE`` 合并为一个覆盖字典。

    Raises:
        SystemExit: ``--set`` 的值缺少 ``=``。
    """
    overrides: dict[str, Any] = {}
    for attribute, key in _OVERRIDE_FLAGS.items():
        value = getattr(args, attribute, None)
        if value is not None:
            overrides[key] = value
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set 需要 KEY=VALUE 形式，收到 {item!r}")
        key, raw_value = item.split("=", 1)
        overrides[key.strip()] = coerce_scalar(raw_value)
    return overrides


def command_train(args: argparse.Namespace) -> int:
    """执行 ``train``，打印一行摘要。"""
    config = load_config(args.config, collect_overrides(args))
    validate_config(config)
    try:
        result = run_experiment(config, force=args.force)
    except FileExistsError as exc:
        print(f"SKIP: {exc}")
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(
        f"done experiment={result['experiment']} seed={result['seed']} "
        f"mse={result['mse']:.6f} mae={result['mae']:.6f} "
        f"time={result['train_time_sec']}s"
    )
    return 0


def command_check_data(args: argparse.Namespace) -> int:
    """执行 ``check-data``；变量数（或 strict 下的行数）不符时返回非零。"""
    config = load_config(args.config, collect_overrides(args))
    try:
        card = dataset_card(config.data, strict=args.strict)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    if args.out is not None:
        save_json(args.out, card)
    print(json.dumps(card, indent=2, ensure_ascii=False))
    if not card["passed"]:
        print("FAIL: 数据文件与基准规格不一致")
        return 1
    if not card["rows_match"]:
        print("WARN: 行数与官方发布版本不同；如需严格校验请加 --strict")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    """执行 ``evaluate``，打印重算指标。"""
    try:
        metrics = evaluate_run(args.run_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(
        f"recomputed mse={metrics['mse']:.6f} mae={metrics['mae']:.6f} "
        f"device={metrics['device']}"
    )
    if "matches_original" in metrics:
        print(f"matches_original={metrics['matches_original']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """控制台脚本与 ``python -m itransformer_repro.cli`` 的共同入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        return command_train(args)
    if args.command == "check-data":
        return command_check_data(args)
    if args.command == "evaluate":
        return command_evaluate(args)
    parser.error(f"未知子命令：{args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
