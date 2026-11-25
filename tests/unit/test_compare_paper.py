"""论文对照脚本的单元测试：模型隔离、参考值与判定口径。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import compare_paper  # noqa: E402

HORIZONS = (96, 192, 336, 720)


def make_rows(
    model: str,
    experiment: str,
    mse: float,
    mae: float,
    seq_len: int = 96,
) -> list[dict[str, str]]:
    """构造 aggregate.py 口径的汇总行。"""
    return [
        {
            "dataset": "ETTh1",
            "model": model,
            "experiment": experiment,
            "seq_len": str(seq_len),
            "pred_len": str(horizon),
            "mse_mean": str(mse),
            "mae_mean": str(mae),
        }
        for horizon in HORIZONS
    ]


def test_ett2_reference_matches_paper_table() -> None:
    """ETTh2 的 4 档平均应与论文 F.2 的 0.383/0.407 一致。"""
    assert compare_paper.ETT_REFERENCE["ETTh2"] == (0.383, 0.407)


def test_default_tolerance_is_ten_percent() -> None:
    """命令行默认容差为作业要求的 10%% 量级。"""
    args = compare_paper.build_parser().parse_args([])
    assert args.tolerance == pytest.approx(0.10)


def test_paper_comparison_only_includes_main_model() -> None:
    """基线不能进入论文对照表，只能出现在基线互比表。"""
    rows = make_rows("itransformer", "itransformer_ETTh1_h96", 0.40, 0.41)
    rows += make_rows("dlinear", "dlinear_ETTh1_h96", 0.60, 0.55)

    comparisons = compare_paper.compare(rows, tolerance=0.10)
    assert [row["model"] for row in comparisons] == ["itransformer"]

    baselines = compare_paper.baseline_table(rows)
    assert [row["model"] for row in baselines] == ["dlinear"]
    assert baselines[0]["delta_mse"] == pytest.approx((0.60 - 0.40) / 0.40)
    assert baselines[0]["delta_mae"] == pytest.approx((0.55 - 0.41) / 0.41)


def test_baseline_without_main_result_has_no_delta() -> None:
    """只有基线时，互比差值为 None（表格显示 ``-``）。"""
    rows = make_rows("patchtst", "patchtst_ETTh1_h96", 0.62, 0.58)
    baselines = compare_paper.baseline_table(rows)
    assert baselines[0]["delta_mse"] is None
    assert baselines[0]["delta_mae"] is None


def test_ablation_and_lookback_rows_are_excluded() -> None:
    """消融行与回看扫描行不能混入主结果平均。"""
    rows = make_rows("itransformer", "itransformer_ETTh1_h96", 0.40, 0.41)
    rows += make_rows("itransformer", "ablation_no_attention_ETTh1_h96", 9.0, 9.0)
    rows += make_rows("itransformer", "itransformer_ETTh1_h96", 9.0, 9.0, seq_len=192)

    comparisons = compare_paper.compare(rows, tolerance=0.10)
    assert len(comparisons) == 1
    assert comparisons[0]["ours_mse"] == pytest.approx(0.40)


def test_verdict_requires_full_horizon_coverage() -> None:
    """缺预测长度时即便数值接近也只能是 WARN。"""
    rows = make_rows("itransformer", "itransformer_ETTh1_h96", 0.455, 0.448)
    partial = rows[:2]
    comparisons = compare_paper.compare(partial, tolerance=0.10)
    assert comparisons[0]["coverage"] == "2/4"
    assert comparisons[0]["verdict"] == "WARN"


def test_verdict_pass_when_within_tolerance() -> None:
    """四档齐全且 MSE/MAE 都在阈值内时判 PASS。"""
    rows = make_rows("itransformer", "itransformer_ETTh1_h96", 0.47, 0.46)
    comparisons = compare_paper.compare(rows, tolerance=0.10)
    assert comparisons[0]["coverage"] == "4/4"
    assert comparisons[0]["verdict"] == "PASS"
