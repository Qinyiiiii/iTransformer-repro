# 复现指南

本文给出从零复现本仓库结果的最短路径。命令均在仓库根目录执行，使用
PowerShell 或 bash 均可；Windows 路径用 `\`，Linux/macOS 用 `/`。

## 1. 环境

| 项目 | 课程默认 | 本仓库 |
| --- | --- | --- |
| Python | 3.10 / 3.11 | `>=3.9,<3.13`；本次审计在 3.9.13 验证，推荐 3.10/3.11 |
| PyTorch | 2.x | `>=2.0` |
| CUDA | 11.8 / 12.x | 有 GPU 时自动使用；CPU 也可跑通与冒烟 |
| 依赖 | 根目录 `requirements.txt` | 运行时 numpy/pandas/PyYAML/matplotlib/torch，开发含 pytest/pytest-cov/ruff |

说明：课程推荐 3.10/3.11；本仓库为兼容审计机器保留 3.9 下限，代码不使用 3.10+
专属语法。正式提交实验建议在 3.10/3.11 + CUDA 环境重跑，并重新生成
`environment.txt`。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 记录实际环境（含 pip freeze 与 nvidia-smi）
python scripts/env_report.py --out environment.txt
```

## 2. 数据

原始 CSV 不随仓库分发。按 [data/README.md](data/README.md) 下载并放到
`data/raw/`：

```text
data/raw/ETT-small/ETTh1.csv
data/raw/ETT-small/ETTh2.csv
data/raw/weather/weather.csv
data/raw/electricity/electricity.csv
```

校验文件完整性、变量数、SHA256 与切分窗口：

```bash
itransformer-repro check-data --config configs/default.yaml
itransformer-repro check-data --config configs/default.yaml \
  --out data_card_ETTh1.json
```

ETTh1 在 `T=S=96` 时：严格样本数 `8449/2785/2785`，论文表 4 口径
`8545/2881/2881`（`paper_table4_windows` 字段）。两种口径的差异已写入
[data/README.md](data/README.md) 与 [docs/experiments.md](docs/experiments.md)。

## 3. 冒烟（不进入结果表）

```bash
# 合成数据单元/集成测试：不依赖真实数据集
pytest -q

# 真实 ETTh1、小模型、1 epoch：只验证数据→训练→评估→产物链路
itransformer-repro train --config configs/smoke.yaml --device cpu
```

冒烟结果只用于确认链路，`configs/smoke.yaml` 已在注释中声明不得作为实验证据。

## 4. 主实验

课程主矩阵：ETTh1 / ETTh2 / Weather / ECL × 预测长度 96/192/336/720 ×
iTransformer/DLinear/PatchTST/Vanilla Transformer × 3 个种子。

```bash
python scripts/run_matrix.py \
  --datasets ETTh1 ETTh2 Weather ECL \
  --models itransformer dlinear patchtst transformer \
  --horizons 96 192 336 720 \
  --seeds 2021 2022 2023
```

脚本会为每个模型读取 `configs/` 中对应的 preset（iTransformer 用
`default.yaml`，其余用各自 YAML），保证基线不会误用 `d_model=512` 等主模型超参。
`--no-model-presets` 可关闭该行为。

聚合与出图：

```bash
python scripts/aggregate.py
python scripts/make_figures.py
python scripts/compare_paper.py
```

产物：

- 原始记录：`results/raw/<model>/<dataset>/h<S>/[<variant>/]seed<seed>/`（本地，不入库）
- 汇总：`results/summary/summary.csv`、`results/summary/main_table.md`
- 图：`results/figures/mse_by_horizon.png`、`model_comparison.png`、
  `ablation.png`、`lookback_curve.png`

说明：公开 GitHub 仓库不提交 `results/raw/`、权重、日志或缓存；`results/summary/`
与 `results/figures/` 默认也被忽略，真实结果是否入库需确认后手动放行。
仓库不附带任何示例结果或冒烟产物，只保留三个子目录的 `.gitkeep`；需要核验链路
请按本文第 3 节在本地运行。若课程提交要求原始结果，请把 `results/raw/` 单独打包提交。

## 5. 消融（论文表 3/6 的六种设计）

| 设计 | 命令 |
| --- | --- |
| 完整 iTransformer | `itransformer-repro train --config configs/default.yaml --seed 2021` |
| Replace Attn→Attn | `--config configs/ablation_temporal_attn.yaml` |
| Replace Attn→FFN | `--config configs/ablation_variate_ffn_temporal_attn.yaml` |
| Replace FFN→FFN | `--config configs/ablation_variate_ffn.yaml` |
| w/o Attention | `--config configs/ablation_no_attention.yaml` |
| w/o FFN | `--config configs/ablation_no_ffn.yaml` |

每个配置都需用 3 个种子重复；`aggregate.py` 会按 `experiment` 分组，不会把消融
混进主方法平均值。也可以一次性批量运行（自动跳过已有结果）：

```bash
python scripts/run_ablations.py --seeds 2021 2022 2023
python scripts/run_ablations.py --include-main --seeds 2021 2022 2023
```

## 6. 回看长度扫描

```bash
python scripts/run_matrix.py \
  --datasets ETTh1 \
  --models itransformer transformer \
  --horizons 96 \
  --seq-lens 48 96 192 336 720 \
  --seeds 2021 2022 2023

python scripts/aggregate.py
python scripts/make_figures.py   # 生成 lookback_curve.png
```

`seq_len=96` 复用主实验目录名；其它长度写成 `..._t<S>`，避免覆盖。

## 7. 重评与错误分析

```bash
itransformer-repro evaluate \
  --run-dir results/raw/itransformer/ETTh1/h96/seed2021

python scripts/analyze_errors.py \
  results/raw/itransformer/ETTh1/h96/seed2021 --top-k 20
```

`evaluate` 会重算测试指标并写入 `metrics_eval.json`，其中
`matches_original` 表示与原 `metrics.json` 的 MSE/MAE 是否在 `1e-6` 内一致；
`analyze_errors.py` 输出逐 horizon、逐变量误差与最差窗口 CSV，供失败案例分析。

## 8. 确定性

- `seed_everything` 固定 Python/NumPy/PyTorch 种子，设置
  `torch.use_deterministic_algorithms(True, warn_only=True)`，并关闭 cudnn benchmark；
- CUDA 下设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；
- 每个 run 的最佳权重按验证集 MSE 选择，测试集只评估一次；
- 需要断点状态时在 YAML 设置 `train.save_train_state: true`，会额外保存
  `train_state.pt`（模型、优化器、epoch、三类随机状态）。

## 9. 已知阻塞与如实说明

- 本次审计机器为 CPU-only，只下载并校验了 ETTh1；完整主矩阵未运行，
  README 与 docs 的数值表保持“待运行”，没有填充任何推测数字。
- 论文表 4 的 ETT 窗口数与严格样本数口径不同，已在 `check-data` 中同时给出。
- PatchTST/Vanilla Transformer 为简化基线，`temporal_module=attention` 为分组
  近似实现，详见 [docs/architecture.md](docs/architecture.md) 与
  [docs/experiments.md](docs/experiments.md)。
- 课程要求的团队分工、GitHub 分支/PR/Issue 过程证据需由提交者补齐；
  见 [DECLARATION.md](DECLARATION.md) 与 [CONTRIBUTING.md](CONTRIBUTING.md)。
