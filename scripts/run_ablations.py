"""批量运行消融设计：每种设计 × 每个随机种子。

用途：
    论文表 3/6 的六种设计对应 ``configs/`` 下的 5 个 ``ablation_*.yaml`` 与主方法
    配置。``run_matrix.py`` 只负责"数据集 × 模型 × 预测长度"，不覆盖消融，因此
    这里提供独立的批处理入口，避免手工循环时漏种子或漏设计。

    每个任务都会写入 ``results/raw/<model>/<dataset>/h<S>/<设计名>/seed<seed>/``，
    与主实验目录互不覆盖；已有 ``metrics.json`` 的任务默认跳过。

前置条件：
    数据已按 ``data/README.md`` 放置；需要先完成一次 ``check-data``。

示例命令：
    python scripts/run_ablations.py --seeds 2021 2022 2023
    python scripts/run_ablations.py --include-main --seeds 2021 2022 2023
    python scripts/run_ablations.py --configs configs/ablation_*.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from itransformer_repro.config import load_config
from itransformer_repro.train import run_exists, run_experiment

logger = logging.getLogger(__name__)

DEFAULT_PATTERN = "configs/ablation_*.yaml"
MAIN_CONFIG = Path("configs/default.yaml")


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="run_ablations.py",
        description="批量运行消融设计（每种设计 × 每个种子）。",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[DEFAULT_PATTERN],
        help=f"配置路径或 glob，默认 {DEFAULT_PATTERN}。",
    )
    parser.add_argument(
        "--include-main",
        action="store_true",
        help="额外运行主方法配置 configs/default.yaml。",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[2021, 2022, 2023],
        help="随机种子列表，默认 2021 2022 2023。",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="覆盖设备（cpu / cuda / auto）；缺省沿用配置文件。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="已有 metrics.json 时也重跑。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将执行的任务，不训练。",
    )
    return parser


def expand_configs(patterns: Sequence[str], include_main: bool) -> list[Path]:
    """展开 glob 并去重，返回待运行的配置文件列表。

    Args:
        patterns: 文件路径或 glob 模式。
        include_main: 是否追加主方法配置。

    Returns:
        排序后的配置文件路径列表。

    Raises:
        FileNotFoundError: 某个模式没有匹配到文件。
    """
    resolved: dict[Path, None] = {}
    for pattern in patterns:
        path = Path(pattern)
        if any(char in pattern for char in "*?["):
            matches = sorted(path.parent.glob(path.name))
            if not matches:
                raise FileNotFoundError(f"模式 {pattern} 没有匹配到配置文件")
            for match in matches:
                resolved[match] = None
        else:
            if not path.is_file():
                raise FileNotFoundError(f"配置文件不存在：{path}")
            resolved[path] = None
    if include_main:
        if not MAIN_CONFIG.is_file():
            raise FileNotFoundError(f"主方法配置不存在：{MAIN_CONFIG}")
        resolved[MAIN_CONFIG] = None
    return sorted(resolved)


def build_jobs(configs: Sequence[Path], seeds: Sequence[int], device: str | None):
    """生成 ``(配置文件, 配置对象)`` 任务列表。

    Args:
        configs: 配置文件列表。
        seeds: 随机种子列表。
        device: 可选的设备覆盖。

    Returns:
        任务列表，顺序为"先设计、后种子"。
    """
    jobs = []
    for path in configs:
        for seed in seeds:
            overrides: dict[str, object] = {"seed": seed}
            if device is not None:
                overrides["device"] = device
            jobs.append((path, load_config(path, overrides)))
    return jobs


def main(argv: Sequence[str] | None = None) -> int:
    """执行批量消融并返回退出码。"""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if not args.seeds:
        print("ERROR: --seeds 不能为空")
        return 1

    try:
        configs = expand_configs(args.configs, args.include_main)
        jobs = build_jobs(configs, args.seeds, args.device)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"共 {len(configs)} 个设计 × {len(args.seeds)} 个种子 = {len(jobs)} 个任务")
    if args.dry_run:
        for path, config in jobs:
            print(f"  {path.name} seed={config.seed} -> {config.run_dir()}")
        return 0

    finished = skipped = failed = 0
    failures: list[str] = []
    for index, (path, config) in enumerate(jobs, start=1):
        run_dir = config.run_dir()
        if run_exists(run_dir) and not args.force:
            print(f"[{index}/{len(jobs)}] SKIP {run_dir}（已有结果）")
            skipped += 1
            continue
        print(
            f"[{index}/{len(jobs)}] RUN {config.experiment} seed={config.seed} "
            f"device={config.device}"
        )
        try:
            result = run_experiment(config, force=args.force)
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
            failed += 1
            failures.append(f"{path.name} seed={config.seed}: {exc}")
            print(f"  ERROR: {exc}")
            continue
        finished += 1
        print(
            f"  done mse={result['mse']:.6f} mae={result['mae']:.6f} "
            f"train={result['train_time_sec']}s"
        )

    print(f"完成 {finished}，跳过 {skipped}，失败 {failed}（共 {len(jobs)} 个任务）")
    for message in failures:
        print(f"  FAILED {message}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
