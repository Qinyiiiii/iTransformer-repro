"""按矩阵批量训练：数据集 × 模型 × 预测长度 × 随机种子。

用途：
    以 ``--config`` 指定的 YAML 为模板，用
    :func:`itransformer_repro.config.apply_overrides` 生成每个组合的完整配置，
    再直接调用 :func:`itransformer_repro.train.run_experiment`。已存在
    ``metrics.json`` 的组合会打印 ``SKIP`` 并跳过，不覆盖历史结果；单个任务失败
    会记录后继续，最后汇总成功/跳过/失败计数。

前置条件：
    已执行 ``python -m pip install -r requirements.txt``，且 ``--config`` 模板中的数据集
    文件已经就位（可用 ``scripts/download_data.py`` 检查）。

示例命令：
    python scripts/run_matrix.py --datasets ETTh1 \\
        --models itransformer dlinear --horizons 96 192 --seeds 2021
    python scripts/run_matrix.py --datasets Weather --models patchtst --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from itransformer_repro.config import MODEL_NAMES, ExperimentConfig, load_config
from itransformer_repro.data import DATASET_SPECS
from itransformer_repro.train import run_exists, run_experiment

logger = logging.getLogger(__name__)

# 每个模型单独的超参文件；只取其中的 model 段与学习率/早停设置，避免批量跑
# PatchTST/DLinear 时误用 iTransformer 的 d_model=512 等超参（基线不公平）。
MODEL_PRESET_FILES = {
    "itransformer": "default.yaml",
    "dlinear": "dlinear_etth1_h96.yaml",
    "patchtst": "patchtst_etth1_h96.yaml",
    "transformer": "transformer_etth1_h96.yaml",
}


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    Returns:
        配置好矩阵维度、模板与 ``--dry-run`` 的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="run_matrix.py",
        description="批量训练 iTransformer 复现实验矩阵（直接调用 run_experiment）。",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_SPECS),
        default=["ETTh1"],
        help="数据集列表，默认 ETTh1。",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_NAMES),
        default=["itransformer"],
        help="模型列表，默认 itransformer。",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[96, 192, 336, 720],
        help="预测长度列表，默认 96 192 336 720。",
    )
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=None,
        help="回看长度列表；缺省沿用模板的 96。给多个值即可做回看曲线扫描。",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[2021],
        help="随机种子列表，默认 2021。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="覆盖训练轮数；缺省时沿用模板值。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="覆盖批大小；缺省时沿用模板值。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="作为模板的 YAML 配置路径，默认 configs/default.yaml。",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="模型超参 preset 所在目录，默认 configs。",
    )
    parser.add_argument(
        "--no-model-presets",
        action="store_true",
        help="禁用 per-model preset，所有模型都沿用 --config 模板的超参。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将执行的任务，不进行训练。",
    )
    return parser


def _apply_model_preset(
    config: ExperimentConfig,
    model: str,
    config_dir: Path,
    batch_size_override: int | None = None,
) -> ExperimentConfig:
    """用 per-model preset 替换 model 段，并继承其学习率、批大小与早停设置。"""
    preset_name = MODEL_PRESET_FILES.get(model)
    if preset_name is None:
        return config
    preset_path = config_dir / preset_name
    if not preset_path.is_file():
        logger.warning("找不到 %s 的模型 preset %s，沿用模板超参", model, preset_path)
        return config
    preset = load_config(preset_path)
    model_config = replace(preset.model, name=model)
    train_config = replace(
        config.train,
        learning_rate=preset.train.learning_rate,
        batch_size=(
            preset.train.batch_size
            if batch_size_override is None
            else batch_size_override
        ),
        patience=preset.train.patience,
    )
    return replace(config, model=model_config, train=train_config)


def build_tasks(
    template: Path,
    datasets: Sequence[str],
    models: Sequence[str],
    horizons: Sequence[int],
    seeds: Sequence[int],
    epochs: int | None,
    batch_size: int | None,
    seq_lens: Sequence[int] | None = None,
    config_dir: Path = Path("configs"),
    use_model_presets: bool = True,
) -> list[ExperimentConfig]:
    """按矩阵组合生成经过校验的配置列表。

    Args:
        template: YAML 模板路径。
        datasets: 数据集名列表。
        models: 模型名列表。
        horizons: 预测长度列表。
        seeds: 随机种子列表。
        epochs: 可选，覆盖 ``train.epochs``。
        batch_size: 可选，覆盖 ``train.batch_size``。
        seq_lens: 可选，回看长度列表；缺省沿用模板。
        config_dir: per-model preset 所在目录。
        use_model_presets: 切换到与模板不同的模型时，是否套用该模型的 preset。
            与模板相同的模型保留模板超参，方便 smoke 配置保持小模型。

    Returns:
        与去重后的组合一一对应的配置对象。

    Raises:
        FileNotFoundError: 模板文件不存在。
        ValueError: 模板字段非法，或合并后的配置未通过校验。
        KeyError: 覆盖项包含未知配置段。
        yaml.YAMLError: 模板不是合法 YAML。
    """
    combos = list(
        dict.fromkeys(
            itertools.product(datasets, models, horizons, seq_lens or [None], seeds)
        )
    )
    tasks: list[ExperimentConfig] = []
    for dataset, model, pred_len, seq_len, seed in combos:
        spec = DATASET_SPECS[dataset]
        name = f"{model}_{dataset}_h{pred_len}"
        if seq_len is not None and seq_len != 96:
            name = f"{name}_t{seq_len}"
        overrides: dict[str, Any] = {
            "experiment": name,
            "seed": seed,
            "data.name": dataset,
            "data.file": spec.filename,
            "data.pred_len": pred_len,
        }
        if seq_len is not None:
            overrides["data.seq_len"] = seq_len
        if epochs is not None:
            overrides["train.epochs"] = epochs
        if batch_size is not None:
            overrides["train.batch_size"] = batch_size

        config = load_config(template, overrides)
        if use_model_presets and model != config.model.name:
            config = _apply_model_preset(config, model, config_dir, batch_size)
        config = replace(config, model=replace(config.model, name=model))
        tasks.append(config)
    return tasks


def describe(config: ExperimentConfig) -> str:
    """返回单个任务的单行说明。

    Args:
        config: 已生成的实验配置。

    Returns:
        含实验名、种子、数据文件、预测长度与训练设置的字符串。
    """
    return (
        f"{config.experiment} seed={config.seed} "
        f"dataset={config.data.name} file={config.data.file} "
        f"seq_len={config.data.seq_len} horizon={config.data.pred_len} "
        f"d_model={config.model.d_model} lr={config.train.learning_rate} "
        f"epochs={config.train.epochs} batch_size={config.train.batch_size}"
    )


def run_tasks(
    tasks: Sequence[ExperimentConfig], dry_run: bool
) -> tuple[int, int, int, list[str]]:
    """依次执行或打印任务。

    Args:
        tasks: 待执行的配置列表。
        dry_run: 为 ``True`` 时只打印，不训练。

    Returns:
        ``(成功数, 跳过数, 失败数, 失败标签列表)``。
    """
    succeeded = 0
    skipped = 0
    failed = 0
    failures: list[str] = []
    total = len(tasks)

    for index, config in enumerate(tasks, start=1):
        label = f"{config.experiment} seed{config.seed}"
        prefix = f"[{index}/{total}]"
        if run_exists(config.run_dir()):
            print(f"{prefix} SKIP {label}（已存在 metrics.json）")
            skipped += 1
            continue
        if dry_run:
            print(f"{prefix} DRY-RUN {describe(config)}")
            continue

        print(f"{prefix} RUN {label}")
        try:
            result = run_experiment(config)
        except FileExistsError:
            print(f"{prefix} SKIP {label}（结果已存在）")
            skipped += 1
            continue
        except Exception as exc:
            logger.exception("任务失败：%s", label)
            print(f"{prefix} FAIL {label}: {exc}")
            failures.append(label)
            failed += 1
            continue
        print(
            f"{prefix} OK {label} mse={result['mse']:.6f} "
            f"mae={result['mae']:.6f} time={result['train_time_sec']}s"
        )
        succeeded += 1
    return succeeded, skipped, failed, failures


def main(argv: Sequence[str] | None = None) -> int:
    """执行批量矩阵并返回退出码。

    Args:
        argv: 命令行参数；``None`` 表示使用 ``sys.argv[1:]``。

    Returns:
        无失败任务时为 ``0``，否则为 ``1``。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if any(horizon <= 0 for horizon in args.horizons):
        parser.error("--horizons 必须为正整数")
    if args.seq_lens is not None and any(seq_len <= 0 for seq_len in args.seq_lens):
        parser.error("--seq-lens 必须为正整数")
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds 必须为非负整数")
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs 必须为正整数")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size 必须为正整数")

    try:
        tasks = build_tasks(
            args.config,
            args.datasets,
            args.models,
            args.horizons,
            args.seeds,
            args.epochs,
            args.batch_size,
            args.seq_lens,
            args.config_dir,
            not args.no_model_presets,
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(f"无法根据模板生成配置：{exc}")

    print(
        f"计划执行 {len(tasks)} 个组合，模板：{args.config}，"
        f"runs_dir={tasks[0].runs_dir}，"
        f"model_presets={'off' if args.no_model_presets else 'on'}"
    )
    if args.dry_run:
        print("DRY-RUN 模式：只打印任务，不训练。")

    succeeded, skipped, failed, failures = run_tasks(tasks, args.dry_run)
    print(f"汇总：成功 {succeeded}，跳过 {skipped}，失败 {failed}，计划 {len(tasks)}")
    if failures:
        print("失败任务：" + ", ".join(failures))
    if args.dry_run:
        return 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
