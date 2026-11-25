"""脚本层单元测试：消融任务展开与注意力 top-k 汇总。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import attention_maps, run_ablations  # noqa: E402


def test_expand_configs_supports_glob_and_dedup(tmp_path: Path) -> None:
    """glob 展开后按路径排序并去重。"""
    first = tmp_path / "ablation_a.yaml"
    second = tmp_path / "ablation_b.yaml"
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")

    patterns = [str(tmp_path / "ablation_*.yaml"), str(first)]
    assert run_ablations.expand_configs(patterns, include_main=False) == [first, second]


def test_expand_configs_raises_for_empty_pattern(tmp_path: Path) -> None:
    """模式无匹配时必须报错，避免静默漏跑。"""
    with pytest.raises(FileNotFoundError):
        run_ablations.expand_configs([str(tmp_path / "nothing_*.yaml")], False)


def test_build_jobs_covers_every_seed() -> None:
    """每个种子都要生成一个任务，且种子写进配置。"""
    config = ROOT / "configs" / "ablation_no_attention.yaml"
    jobs = run_ablations.build_jobs([config], [2021, 2022], device=None)
    assert [job[1].seed for job in jobs] == [2021, 2022]
    assert all(job[0] == config for job in jobs)


def test_summarize_top_pairs_skips_diagonal_and_sorts() -> None:
    """top-k 汇总排除自注意力，并按权重降序返回变量名。"""
    mean_map = np.array(
        [
            [0.9, 0.05, 0.01],
            [0.02, 0.9, 0.30],
            [0.01, 0.10, 0.9],
        ]
    )
    pairs = attention_maps.summarize_top_pairs(mean_map, ["a", "b", "c"], top_k=2)
    assert pairs == [
        {"source": "b", "target": "c", "weight": 0.3},
        {"source": "c", "target": "b", "weight": 0.1},
    ]
