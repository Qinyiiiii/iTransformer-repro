# iTransformer 复现（多变量长期时间序列预测）

本仓库复现 ICLR 2024 论文 **iTransformer: Inverted Transformers Are Effective for
Time Series Forecasting** 的核心结论：把自注意力放在变量维、FFN 放在时间维的倒置
结构，在相同数据划分与评价口径下优于线性模型与通道独立 Transformer。

## 功能与方法

- 主方法：iTransformer（变量维注意力 + 时间维 FFN，无位置编码）
- 基线：DLinear、PatchTST，以及未倒置的 Vanilla Transformer 对照
- 消融：`variate_module ∈ {attention, ffn, none}` × `temporal_module ∈ {ffn, attention, none}`，
  覆盖论文表 3 的 FFN-FNN、Attn-Attn、w/o Attn、w/o FFN，另有去 LayerNorm、加位置编码开关
- 统一协议：时间顺序切分、只用训练段拟合标准化、MSE/MAE 在标准化空间计算、
  固定随机种子、每个配置落盘完整证据
- 任务口径：多变量输入 → 多变量输出（`features="M"`）；单变量与 MS 模式未实现，
  配置校验会直接报错而不是静默降级

## 项目结构

```text
itransformer-repro/
├── .github/workflows/ci.yml        # 【自写】Ruff + pytest 质量门禁
├── configs/                        # 【自写】9 个版本化 YAML 实验配置
├── data/README.md                  # 【自写】数据来源、放置方式、划分规则
├── docs/architecture.md            # 【自写】结构、公式与消融语义
├── docs/experiments.md             # 【自写】实验矩阵与论文差异表
├── scripts/                        # 【自写】数据检查、批量训练、汇总、对照、绘图
├── src/itransformer_repro/
│   ├── models.py                   # 【参考官方重写】iTransformer/消融/三个基线
│   ├── data.py                     # 【自写】切分、标准化、滑窗（协议依据官方 data_provider）
│   ├── train.py  evaluate.py       # 【自写】训练、选权、证据落盘、复评
│   ├── config.py  cli.py  utils.py  metrics.py   # 【自写】
│   └── __init__.py                 # 【自写】
├── tests/                          # 【自写】单元测试 + 端到端冒烟测试
└── README.md  pyproject.toml  LICENSE  .gitignore  # 【自写】
```

## 代码来源与原创性

本仓库**没有逐行复制**任何外部仓库文件。标注含义如下：

| 标注 | 含义 | 涉及文件 |
| --- | --- | --- |
| 【参考官方重写】 | 模块划分、公式与超参依据 [thuml/iTransformer](https://github.com/thuml/iTransformer)（MIT）与论文重写，代码为本项目自行实现 | `src/itransformer_repro/models.py` |
| 【自写】 | 完全自主实现，仅借鉴公开基准的数据切分协议 | 其余全部文件 |

- `models.py` 对应官方 `DataEmbedding_inverted`、`EncoderLayer`、`DLinear`、`PatchTST` 的语义：
  逐变量嵌入、变量维注意力、post-norm 残差、投影回预测长度；官方原代码以 Conv1d(kernel=1)
  实现 FFN，本项目用等价的 `nn.Linear`。
- 数据切分遵循官方 `data_provider` 的 ETT 12/4/4 个月与其余 70/10/20 协议，但读取、标准化、
  滑窗、数据集卡片均为自写代码。
- 官方仓库的 MIT 许可证与出处已记录在 `LICENSE` 与本文"数据与第三方代码来源"一节；
  如需在报告中说明改动，可直接引用本节。

## 环境要求

- Python 3.9–3.12（课程环境为 3.9；CI 使用 3.11）
- PyTorch ≥ 2.0（CPU 可跑通冒烟测试；正式实验建议 CUDA GPU）
- 其他依赖见 `pyproject.toml`

## 安装

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

CUDA 用户如默认源装的是 CPU 版，请先按官方索引安装对应版本，例如：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 数据准备

数据不随仓库分发，请从官方链接下载后放到 `data/raw/`（详见 `data/README.md`）：

```text
data/raw/
├── ETT-small/ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv
├── weather/weather.csv
├── electricity/electricity.csv
├── traffic/traffic.csv
└── exchange_rate/exchange_rate.csv
```

ETT 采用官方协议的"前 12/4/4 个月"，文件末尾多余月份不参与；其余数据集按
70/10/20 时间顺序切分。细节与许可证见 [data/README.md](data/README.md)。

下载后先校验文件、变量数与切分点数：

```bash
itransformer-repro check-data --config configs/default.yaml
```

## 快速开始（最小可运行示例）

```bash
# 1) 1 个 epoch 的小配置，验证环境与链路
itransformer-repro train --config configs/smoke.yaml --device cpu

# 2) 正式配置，ETTh1 + 预测 96 步 + 种子 2021
itransformer-repro train --config configs/default.yaml --seed 2021
```

运行产物写入 `runs/<experiment>/seed<seed>/`：

```text
config.json    # 全部超参 + 环境 + 代码版本
metrics.json   # test MSE / MAE / RMSE + 参数量 + 耗时
train.log      # 每个 epoch 的 loss 与耗时
pred.npy       # 测试集预测，形状 (n_windows, pred_len, n_vars)
model.pt       # 权重（仅本地，已被 .gitignore 排除）
```

## 训练

单个配置（可用命令行覆盖少量参数）：

```bash
itransformer-repro train --config configs/default.yaml \
  --seed 2022 --epochs 10 --batch-size 16 --pred-len 192
```

批量矩阵（3 模型 × 多个数据集 × 4 个预测长度 × 3 个种子）：

```bash
python scripts/run_matrix.py \
  --datasets ETTh1 Weather \
  --models itransformer dlinear patchtst \
  --horizons 96 192 336 720 \
  --seeds 2021 2022 2023 \
  --epochs 10
```

已存在 `metrics.json` 的运行会自动跳过并打印 `SKIP`，不会覆盖历史结果。

## 评估与测试

```bash
# 复评某个已完成运行
itransformer-repro evaluate --run-dir runs/itransformer_ETTh1_h96/seed2021

# 汇总 runs/ 下全部结果，输出 mean±std
python scripts/aggregate.py

# 与论文报告值逐档对照
python scripts/compare_paper.py

# 生成图表（按预测长度的 MSE、模型对比、消融）
python scripts/make_figures.py

# 代码质量与测试
ruff format --check .
ruff check .
pytest -q
```

## 主要结果

下表由 `scripts/aggregate.py` 从原始运行结果生成，禁止手工填写；未运行的行保持空白。

| 数据集 | 预测长度 | iTransformer | DLinear | PatchTST | 种子数 |
| --- | --- | --- | --- | --- | --- |
| ETTh1 | 96 | 待运行 | 待运行 | 待运行 | 0 |
| ETTh1 | 192 | 待运行 | 待运行 | 待运行 | 0 |
| Weather | 96 | 待运行 | 待运行 | 待运行 | 0 |

论文报告值（4 个预测长度平均，MSE/MAE）仅作量级参照：ETT 系列平均 0.383/0.399、
Weather 0.258/0.278、ECL 0.178/0.270。课程为缩减配置，绝对数值不可直接比较。

## 团队分工

| 成员 | 负责模块 | 主要提交 |
| --- | --- | --- |
| （待填） | 环境与数据管道 | — |
| （待填） | 模型与消融 | — |
| （待填） | 训练/评估与图表 | — |

## 已知限制

- 未包含 PEMS、Market、Solar 等数据集（Solar 原始文件无表头，需要单独的解析分支）。
  Traffic(N=862) 与 ECL(N=321) 已注册，但显存不足时应降低 batch，并在差异表中记录。
- `temporal_module="attention"` 为消融近似实现：把 D 维表征按 8 组切分作为时间表征
  token 做注意力，语义与论文文字一致，分组方式是本项目的实现细节（见
  [docs/architecture.md](docs/architecture.md)）。
- PatchTST 与 Vanilla Transformer 为简化基线（未做超参搜索），与官方实现存在实现差异。
- 论文正文与代码存在若干不一致（PEMS 步长、batch、学习率、指标口径），本项目
  的处理方式记录在 [docs/experiments.md](docs/experiments.md) 的差异表中。
- 指标在标准化空间计算，不与原始量纲换算；跨数据集比较只看相对量级。

## 数据与第三方代码来源

- 论文：Liu et al., *iTransformer: Inverted Transformers Are Effective for Time
  Series Forecasting*, ICLR 2024, arXiv:2310.06625
- 代码：模型结构参照 https://github.com/thuml/iTransformer （MIT License），
  本项目重写训练/评估框架并新增消融开关与实验管理
- 数据：ETT / Weather / ECL 等公开基准，来源与许可证见 `data/README.md`

## 许可证

MIT，见 `LICENSE`。第三方代码与数据的许可说明附在同一文件中。
