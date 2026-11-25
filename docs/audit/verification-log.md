# 审计验证记录（原始输出摘录）

日期：2026-11-24　环境：Windows / Python 3.9.13 / torch 2.8.0+cpu / CPU-only
工作目录：`d:\itransformer-repro`

本文件保存审计期间实际执行的命令与关键输出，供审计报告引用。所有数字均来自真实命令，
未做手工改写。

## 1. 静态检查

```powershell
$env:PYTHONPATH='src'; python -m ruff check .
# All checks passed!

python -m ruff format --check .
# 35 files already formatted
```

## 2. 测试与覆盖率

```powershell
$env:PYTHONPATH='src'; python -m pytest -q --cov=itransformer_repro --cov-report=term
```

结果：`65 passed, 11 warnings in 7.99s`；`Required test coverage of 80.0% reached. Total coverage: 89.08%`。

| 模块 | 覆盖率 |
| --- | --- |
| `__init__.py` | 100% |
| `cli.py` | 87% |
| `config.py` | 90% |
| `data.py` | 89% |
| `evaluate.py` | 85% |
| `metrics.py` | 100% |
| `models.py` | 96% |
| `train.py` | 88% |
| `utils.py` | 77% |
| **TOTAL** | **89.08%** |

## 3. 数据校验（ETTh1，strict）

```powershell
$env:PYTHONPATH='src'; python -m itransformer_repro.cli check-data --config configs/smoke.yaml --strict
```

关键字段：

```json
{
  "dataset": "ETTh1",
  "size_bytes": 2589657,
  "sha256": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
  "rows": 17420,
  "expected_rows": 17420,
  "rows_match": true,
  "n_vars": 7,
  "vars_match": true,
  "split_windows": {"train": 8449, "val": 2785, "test": 2785},
  "paper_table4_windows": {"train": 8545, "val": 2881, "test": 2881},
  "borders": {"train": [0, 8640], "val": [8544, 11520], "test": [11424, 14400]},
  "passed": true
}
```

## 4. 真实 ETTh1 冒烟（1 epoch，非实验证据）

```powershell
$env:PYTHONPATH='src'; python -m itransformer_repro.cli train --config configs/smoke.yaml --device cpu --force
```

```text
data=ETTh1 n_vars=7 train/val/test=8449/2785/2785 params=46048
epoch 1/1 train_loss=0.419134 val_mse=0.698363 val_mae=0.566997 11.31s
finished test_mse=0.428403 test_mae=0.450213 time=17.7s
```

产物目录 `results/raw/itransformer/ETTh1/h96/smoke/seed2021/`：

```text
config.json  metrics.json  metrics_eval.json  train.log
pred.npy     true.npy      model.pt
error_analysis.json  worst_cases.csv
```

## 5. 重评与错误分析

```powershell
python -m itransformer_repro.cli evaluate --run-dir results/raw/itransformer/ETTh1/h96/smoke/seed2021
# recomputed mse=0.428403 mae=0.450213 device=cpu
# matches_original=True

python scripts/analyze_errors.py results/raw/itransformer/ETTh1/h96/smoke/seed2021
# 已写入 error_analysis.json 与 worst_cases.csv
```

## 6. 汇总、对照与绘图

```powershell
python scripts/aggregate.py
# 1 行；WARN: 1 个分组少于 3 个种子
# 已写入 results/summary/summary.csv 与 results/summary/main_table.md

python scripts/compare_paper.py
# ETTh1 h96 coverage=1/4 ours 0.4284/0.4502 paper 0.383/0.399 → WARN

python scripts/make_figures.py
# 已生成 mse_by_horizon.png, model_comparison.png
# 跳过 ablation.png（无消融结果）与 lookback_curve.png（仅一个 seq_len）
```

## 7. 矩阵 dry-run（验证 baseline preset）

```powershell
python scripts/run_matrix.py --models itransformer dlinear patchtst transformer \
  --datasets ETTh1 --horizons 96 --seeds 2021 --dry-run
```

```text
[1/4] itransformer ... d_model=512 lr=0.001 epochs=10 batch_size=16
[2/4] dlinear      ... d_model=512 lr=0.001 epochs=10 batch_size=32
[3/4] patchtst     ... d_model=128 lr=0.0001 epochs=10 batch_size=32
[4/4] transformer  ... d_model=512 lr=0.001 epochs=10 batch_size=32
```

## 8. Git 状态

```text
git log --oneline -8  → 78fe1d1 feat: add itransformer-repro project files
git branch -a         → main（仅本地 main + origin/main）
git tag               → 空
git status            → 大量已修改文件与新增文件尚未提交
```
