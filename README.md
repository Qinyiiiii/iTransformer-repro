# itransformer-repro

面向课程期末项目的 iTransformer（ICLR 2024）可复现实现：在统一的 PyTorch
数据/训练/评测管道上，实现倒置 Transformer 主模型、三个对照模型、论文表 3/6 的
六种消融设计，以及从原始运行到汇总表/图的完整实验流水线。

## 1. 问题与目标

多变量长期时间序列预测：给定回看窗口 `X ∈ R^{T×N}`，预测未来 `Y ∈ R^{S×N}`。
本项目要回答三个问题：

1. 在**变量维**做注意力、在**时间维**做 FFN 的倒置结构，是否优于在时间维做注意
   力的原始 Transformer；
2. 变量维注意力与时间维 FFN 各自贡献多少（六种消融）；
3. 在相同数据切分、标准化、指标与训练协议下，iTransformer 与 DLinear、PatchTST、
   Vanilla Transformer 的差距如何。

范围：ETTh1、ETTh2、Weather、ECL 四个数据集，预测长度 96/192/336/720，至少 3 个
随机种子。论文其余数据集（Traffic、Exchange、Solar、PEMS）不在主矩阵内。

## 2. 方法概览

iTransformer 把每个变量的整段回看序列嵌入成一个 token，在 **N 个变量 token**
之间做自注意力，再对每个 token 做时间维 FFN，最后把每个 token 投影到预测长度：

```text
(B,T,N) → (B,N,T) → Linear(T,D) → L × [变量维 Attention + 时间维 FFN]
        → LayerNorm → Linear(D,S) → (B,S,N)
```

- 位置编码默认关闭（倒置后变量 token 没有天然时间顺序）；
- 消融开关：`variate_module ∈ {attention, ffn, none}`、
  `temporal_module ∈ {ffn, attention, none}`、`use_layernorm`、`use_pos_embed`；
- 对照模型：DLinear（趋势/季节分解 + 线性）、PatchTST（实例标准化 + patch +
  Transformer）、Vanilla Transformer（时间步 token + 位置编码）。

与论文/官方实现的差异（训练预算、简化基线、时间维注意力近似、ETT 窗口计数口径）
集中记录在 [docs/experiments.md](docs/experiments.md) 第 5 节和
[docs/architecture.md](docs/architecture.md) 第 6 节，报告中必须如实转述。

## 3. 环境与安装

| 项目 | 要求 |
| --- | --- |
| Python | 课程默认 3.10/3.11；本仓库 `>=3.9,<3.13`，本次审计在 3.9.13 验证 |
| PyTorch | `>=2.0`，CPU 可运行；推荐 CUDA 11.8/12.x |
| 依赖 | 根目录 `requirements.txt`：numpy、pandas、PyYAML、matplotlib、torch；开发另含 pytest、pytest-cov、ruff |

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 记录实际环境（Python/PyTorch/CUDA/GPU/pip freeze/nvidia-smi）
python scripts/env_report.py --out environment.txt
```

`requirements.txt` 声明运行时与开发依赖，并包含 `-e .`，因此上面一条命令即可
装齐依赖并以可编辑模式安装本包（提供 `itransformer-repro` 命令）。仓库根目录的
`environment.txt` 是审计时的实际环境记录；换机器或升级依赖后必须重新生成，精确
版本以它的 `pip_freeze` 为准。

## 4. 数据准备

原始 CSV 不随仓库分发。按 [data/README.md](data/README.md) 下载到 `data/raw/`：

```text
data/raw/ETT-small/ETTh1.csv
data/raw/ETT-small/ETTh2.csv
data/raw/weather/weather.csv
data/raw/electricity/electricity.csv
```

校验文件、变量数、SHA256 与切分窗口：

```bash
itransformer-repro check-data --config configs/default.yaml
itransformer-repro check-data --config configs/default.yaml \
  --out data_card_ETTh1.json
```

`check-data` 输出 `rows`、`sha256`、`split_windows` 与
`paper_table4_windows`。ETTh1 在 `T=S=96` 时严格样本数为
`8449/2785/2785`；论文表 4 的 `8545/2881/2881` 是“只数回看起点、不扣
`pred_len`”的口径，等于前者加 `pred_len`。两种口径都记录，不用论文数字覆盖实际
切分。

## 5. 最小示例

不依赖真实数据的完整链路测试（合成数据，秒级）：

```bash
pytest -q
```

真实 ETTh1 的单 epoch 小模型冒烟（只验证数据→训练→评估→落盘，不进入结果表）：

```bash
itransformer-repro train --config configs/smoke.yaml --device cpu
```

预期产物（路径随配置）：

```text
results/raw/itransformer/ETTh1/h96/smoke/seed2021/
├── config.json
├── metrics.json
├── train.log
├── pred.npy
├── true.npy
└── model.pt
```

`config.json` 含 `git_commit`、`git_dirty`、`environment`、`n_params`；
`metrics.json` 含 `mse/mae/rmse/mape`（标准化空间）、`best_epoch`、
`train_time_sec`（纯训练，不含验证）、`val_time_sec`（逐轮验证）、
`inference_time_sec`（测试集推理）、`total_time_sec`、`peak_memory_mb`。
冒烟结果不得写入主表。这些文件属于本地运行产物，`results/raw/` 整体被
`.gitignore` 排除，不会进入公开仓库；仓库只保留
[results/README.md](results/README.md) 与三个子目录的 `.gitkeep`。

## 6. 训练与评估

### 6.1 单次运行

```bash
# 主方法
itransformer-repro train --config configs/default.yaml --seed 2021

# 换预测长度 / 设备
itransformer-repro train --config configs/default.yaml \
  --pred-len 336 --device cuda

# 任意点号覆盖
itransformer-repro train --config configs/default.yaml \
  --set model.d_model=256 train.epochs=3
```

运行目录规范：`results/raw/<model>/<dataset>/h<S>/[<variant>/]seed<seed>/`。

### 6.2 主实验矩阵

```bash
python scripts/run_matrix.py \
  --datasets ETTh1 ETTh2 Weather ECL \
  --models itransformer dlinear patchtst transformer \
  --horizons 96 192 336 720 \
  --seeds 2021 2022 2023
```

脚本会为每个模型套用 `configs/` 中对应的 preset（PatchTST 的 `d_model=128`、
`lr=1e-4`、`batch_size=32` 等），避免基线误用 iTransformer 的 `d_model=512`；
`--no-model-presets` 可关闭。已存在 `metrics.json` 的组合会打印 `SKIP`。

### 6.3 聚合、对照与出图

```bash
python scripts/aggregate.py       # results/summary/summary.csv + main_table.md
python scripts/make_figures.py    # results/figures/*.png
python scripts/compare_paper.py   # 表 A 论文对照 + 表 B 基线互比
```

`aggregate.py` 输出每个（实验 × 数据集 × 模型 × seq_len × pred_len）的
`mean ± std`、种子数、参数量、训练/推理耗时与峰值显存；少于 3 个种子会打印 WARN。
`make_figures.py` 生成 `mse_by_horizon.png`、`model_comparison.png`、
`ablation.png`，存在多个 `seq_len` 时额外生成 `lookback_curve.png`。
`compare_paper.py` 只对主方法 iTransformer 做论文对照（默认容差 10%，同时要求
4 档预测长度齐全才判 PASS）；基线是简化实现，单独输出"与同数据集 iTransformer
互比"的表 B，不与论文基线数字直接对照。

### 6.4 复评与错误分析

```bash
itransformer-repro evaluate \
  --run-dir results/raw/itransformer/ETTh1/h96/seed2021

python scripts/analyze_errors.py \
  results/raw/itransformer/ETTh1/h96/seed2021 --top-k 20
```

复评写入 `metrics_eval.json`，含与原 `metrics.json` 的
`mse_delta/mae_delta/matches_original`；错误分析输出逐 horizon、逐变量误差与
`worst_cases.csv`，用于失败案例分析。

### 6.5 消融、回看扫描与注意力导出

六种消融设计的配置与命令见 [REPRODUCTION.md](REPRODUCTION.md) 第 5 节。批量
运行全部消融设计（每种设计 × 每个种子，自动跳过已有结果）：

```bash
python scripts/run_ablations.py --seeds 2021 2022 2023            # 5 个消融设计
python scripts/run_ablations.py --include-main --seeds 2021 2022 2023
python scripts/run_ablations.py --dry-run                          # 只打印任务
```

回看扫描：

```bash
python scripts/run_matrix.py \
  --datasets ETTh1 --models itransformer transformer \
  --horizons 96 --seq-lens 48 96 192 336 720 --seeds 2021 2022 2023
```

变量注意力导出（可解释性分析）：

```bash
python scripts/attention_maps.py results/raw/itransformer/ETTh1/h96/seed2021
```

输出 `attention/attention.npy`（`(窗口, 头, N, N)`）、`attention_mean.npy`、
`attention_heatmap.png` 与 `attention_topk.json`；`--layer` 选层，`--num-windows`
控制窗口数。

## 7. 主要结果

> 状态：**待运行**。本次审计环境为 CPU-only；ETTh1、ETTh2、Weather、ECL 四个
> 主数据集均已下载并通过 `check-data` 校验（Traffic/Exchange/Solar/PEMS 的原始
> 文件也在 `data/raw`，不在本次范围内），但完整
> `4 数据集 × 4 预测长度 × 4 模型 × 3 种子` 矩阵未运行。下表不使用论文值
> 冒充实测值，也不填任何推测数字。

| 数据集 | 模型 | 96 | 192 | 336 | 720 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | iTransformer | - | - | - | - | 待运行 |
| ETTh1 | DLinear | - | - | - | - | 待运行 |
| ETTh1 | PatchTST | - | - | - | - | 待运行 |
| ETTh1 | Transformer | - | - | - | - | 待运行 |
| ETTh2 | iTransformer | - | - | - | - | 待运行 |
| Weather | iTransformer | - | - | - | - | 待运行 |
| ECL | iTransformer | - | - | - | - | 待运行 |

运行完成后，用 `scripts/aggregate.py` 的 `results/summary/main_table.md` 替换本节；
每个数字必须能追溯到 `results/raw/**/seed*/metrics.json` 与 `config.json` 的
`git_commit`。论文参考值见 [docs/experiments.md](docs/experiments.md) 第 7 节，
只作量级参照。

## 8. 实验设计与产物规范

完整矩阵、消融、回看扫描、失败案例分析、图表清单与“不要做的事”见
[docs/experiments.md](docs/experiments.md)。结果目录规范见
[results/README.md](results/README.md)：

```text
results/
├── raw/<model>/<dataset>/h<S>/[<variant>/]seed<seed>/   # 本地运行产物（gitignore）
├── summary/summary.csv + main_table.md                  # 由 aggregate.py 生成
└── figures/*.png                                        # 由 make_figures.py 生成
```

`raw/` 是唯一事实来源；`summary/` 与 `figures/` 只能由脚本生成。公开仓库不提交
`raw/`、`.npy`/`.pt`、日志或缓存，只保留 `README.md` 与三个子目录的 `.gitkeep`。
真实汇总是否入库需人工确认后再放行。

## 9. 项目结构

```text
.
├── README.md
├── REPRODUCTION.md                   # 从零复现清单
├── DECLARATION.md                    # 代码/数据来源、AI 使用声明、团队分工
├── CONTRIBUTING.md                   # 分支、提交、Issue、PR 规范
├── CITATION.cff
├── LICENSE                           # MIT
├── environment.txt                   # 审计时实际环境（换机器后重新生成）
├── requirements.txt                  # 运行时 + 开发依赖（含 -e .）
├── pyproject.toml                    # 包元信息 + ruff/pytest 默认配置
├── .gitignore                        # 排除 results/raw/**、__pycache__ 等
├── .github/
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── ISSUE_TEMPLATE/{bug_report,experiment_task}.md
│   └── workflows/ci.yml              # ruff format → ruff check → pytest
│
├── configs/                           # YAML 配置（超参、消融开关）
│   ├── default.yaml                  # iTransformer 主配置
│   ├── smoke.yaml                    # 冒烟测试（小模型 + 1 epoch）
│   ├── dlinear_etth1_h96.yaml        # DLinear 基线 preset
│   ├── patchtst_etth1_h96.yaml       # PatchTST 基线 preset
│   ├── transformer_etth1_h96.yaml    # Vanilla Transformer 基线 preset
│   ├── ablation_no_attention.yaml    # 消融：w/o Attention
│   ├── ablation_no_ffn.yaml          # 消融：w/o FFN
│   ├── ablation_temporal_attn.yaml   # 消融：Replace Attn→Attn
│   ├── ablation_variate_ffn.yaml     # 消融：Replace FFN→FFN
│   └── ablation_variate_ffn_temporal_attn.yaml  # 消融：Replace FFN→Attn
│
├── data/
│   ├── README.md
│   └── raw/                          # 原始数据（gitignore 排除）
│
├── docs/
│   ├── architecture.md               # 倒置结构详解 + 与官方实现对照
│   ├── experiments.md                # 实验设计、已知偏差、图表清单
│   ├── remote-training-prompt.md     # 远程 GPU AI 助手训练指令
│   └── audit/
│       ├── audit-report.md
│       └── verification-log.md
│
├── results/                          # 运行产物（gitignore 排除 raw/**）
│   ├── README.md                     # 产物目录规范
│   ├── raw/.gitkeep                  # 单次 run 的原始证据（config/metrics/log/npy/pt）
│   ├── summary/.gitkeep              # aggregate.py 生成 summary.csv + main_table.md
│   └── figures/.gitkeep              # make_figures.py 生成 *.png
│
├── scripts/
│   ├── run_matrix.py                 # 主实验矩阵批量训练（dry-run + SKIP 续跑）
│   ├── run_ablations.py              # 消融设计批量运行
│   ├── aggregate.py                  # 扫描 raw → mean±std 汇总表
│   ├── compare_paper.py              # 论文量级对照 + 基线互比
│   ├── make_figures.py               # 生成折线图/柱状图
│   ├── attention_maps.py             # 导出变量注意力权重（npy + heatmap PNG）
│   ├── analyze_errors.py             # 错误案例分析
│   ├── env_report.py                 # 生成 environment.txt
│   └── download_data.py              # 数据下载
│
├── src/itransformer_repro/
│   ├── config.py                     # dataclass 配置 + 加载 + 覆盖 + 校验
│   ├── data.py                       # 数据集规格 + 切分 + 标准化 + 滑窗
│   ├── models.py                     # iTransformer + 消融变体 + 3 基线
│   ├── train.py                      # 训练循环 + run_experiment（证据落盘）
│   ├── evaluate.py                   # 独立评估入口
│   ├── metrics.py                    # MSE/MAE/RMSE/MAPE
│   ├── utils.py                      # 确定性 + 设备 + git + 环境信息
│   └── cli.py                        # itransformer-repro train/evaluate/check-data
│
└── tests/
    ├── conftest.py
    ├── unit/                         # 8 份：config, data, models, fidelity, metrics, cli, compare_paper, scripts
    └── integration/                  # 1 份：完整冒烟训练链路
```

## 10. 测试与质量

```bash
ruff format --check .
ruff check .
pytest -q
```

CI（`.github/workflows/ci.yml`）在 Python 3.11 上执行同样的三步，权限仅
`contents: read`。测试覆盖模型形状、六种消融、注意力权重形状、数据切分与标准化、
指标手算、配置校验、完整训练/复评链路。

## 11. 可复现性与确定性

- 固定 Python/NumPy/PyTorch 种子；`torch.use_deterministic_algorithms(True,
  warn_only=True)`；关闭 cudnn benchmark；CUDA 下设置
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`；
- 每个 run 在本地保存 `config.json`（含 `git_commit/git_dirty/environment`）、
  `metrics.json`、`train.log`、`pred.npy`、`true.npy`；这些属于本地证据，
  `results/raw/` 整体不入库；
- 最佳权重按**验证集** MSE 选择，测试集只评估一次；测试集不参与调参；
- 需要断点状态时设置 `train.save_train_state: true`，额外保存模型、优化器、
  epoch 与三类随机状态；
- 数据版本由 `check-data` 的 SHA256 与行数记录；未下载的数据不会用占位文件冒充。

## 12. 团队分工、限制与许可

团队分工与 AI 使用声明见 [DECLARATION.md](DECLARATION.md)；提交前必须填写真实
姓名与学号。协作流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。

已知限制：

- 本次审计未运行主实验，结果表待补；完整矩阵建议在 GPU 上执行
  [REPRODUCTION.md](REPRODUCTION.md) 第 4 节；
- PatchTST 与 Vanilla Transformer 是简化基线，`temporal_module=attention` 是
  分组近似实现，不能声称逐行复刻官方/论文；
- 训练预算为 10 epoch、无学习率调度、无超参网格搜索，绝对数值预计低于论文；
- ETT 窗口计数与论文表 4 的口径差异见 `data/README.md` 第 5 节；
- Python 兼容下限为 3.9，但课程默认环境为 3.10/3.11，正式提交建议在 3.10/3.11
  重跑并更新 `environment.txt`。

引用：iTransformer（arXiv:2310.06625, ICLR 2024）与
[thuml/iTransformer](https://github.com/thuml/iTransformer)；数据引用见
[data/README.md](data/README.md)。代码许可见 [LICENSE](LICENSE)。
