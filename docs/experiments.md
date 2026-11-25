# 实验手册

本文说明实验矩阵、常用命令、运行产物与字段含义、复现纪律，并集中记录论文（与
官方代码）中与本项目实现不一致之处。本文不报告任何实验数值：所有结果均为
"待运行"，由 `scripts/aggregate.py` 从 `results/raw/` 下的原始 `metrics.json`
汇总生成。

## 1. 实验矩阵

主实验按下面的笛卡尔积执行：

| 维度 | 取值 | 数量 |
| --- | --- | --- |
| 数据集 | ETTh1、ETTh2、Weather、ECL | 4 |
| 预测长度 `pred_len` | 96、192、336、720 | 4 |
| 模型 | `itransformer`、`dlinear`、`patchtst`、`transformer` | 4 |
| 随机种子 `seed` | 2021、2022、2023 | 3 |

总计 `4 × 4 × 4 × 3 = 192` 次独立运行。固定 `seq_len=96`（`configs/default.yaml`），
每个配置训练 `epochs=10`，不启用早停（`patience=0`）。

三个种子的作用是估计随机性带来的方差：同一配置的三个种子结果先各自落盘为独立的
`results/raw/<model>/<dataset>/h<S>/[<variant>/]seed<seed>/metrics.json`，再由
`scripts/aggregate.py` 汇总为 **mean ± std**。`seed_everything` 同时固定 Python `random`、NumPy、PyTorch 与
`PYTHONHASHSEED`，并请求确定性算子；跨硬件无法保证逐位一致，因此只承诺统计结论
一致（见 `utils.seed_everything` 的说明）。

命名约定：`experiment` 形如 `itransformer_ETTh1_h96`，运行目录由
`ExperimentConfig.run_dir()` 生成为
`results/raw/itransformer/ETTh1/h96/seed2021`；消融变体在 `h96/` 下多一层
`ablation_*` 目录。同一配置的三个种子分别落在 `seed2021`、`seed2022`、
`seed2023`。批量脚本需要保证 `experiment` 唯一，否则会写到同一目录而触发
`FileExistsError`（或打印 `SKIP`）。

## 2. 常用命令

以下命令都在仓库根目录执行，且需先 `python -m pip install -r requirements.txt`（安装控制台
脚本 `itransformer-repro`）；未安装时可用 `python -m itransformer_repro.cli` 替代
命令前缀。

### 2.1 数据校验

```bash
itransformer-repro check-data --config configs/default.yaml
```

输出数据集卡片 JSON（`rows`、`expected_rows`、`sha256`、`n_vars`、
`split_windows`、`paper_table4_windows`、`borders` 等），变量数不符时返回非零
退出码；`--strict` 时行数与官方发布版本不符也失败。数据放置方式见
`data/README.md`。

### 2.2 单次训练

```bash
# 正式配置：ETTh1、h96、种子 2021
itransformer-repro train --config configs/default.yaml

# 冒烟链路（1 epoch、小模型，结果不得作为实验证据）
itransformer-repro train --config configs/smoke.yaml --device cpu
```

### 2.3 修改预测长度 / 数据集

```bash
# 方式一：专用参数
itransformer-repro train --config configs/default.yaml --pred-len 192

# 方式二：点号覆盖，同时切换数据集（路径相对 data/raw）
itransformer-repro train --config configs/default.yaml \
  --set data.pred_len=336 data.name=Weather data.file=weather/weather.csv
```

`--seed`、`--epochs`、`--batch-size`、`--lr`、`--seq-len`、`--pred-len`、`--device`
是 `cli.py` 中的专用覆盖参数，其余字段用 `--set KEY=VALUE`（如
`model.d_model=256`、`model.variate_module=ffn`、`model.use_pos_embed=true`）。
`--force` 覆盖已存在的运行目录，否则已有 `metrics.json` 的运行会被跳过。

### 2.4 跑完整矩阵

```bash
python scripts/run_matrix.py \
  --datasets ETTh1 ETTh2 Weather ECL \
  --models itransformer dlinear patchtst transformer \
  --horizons 96 192 336 720 \
  --seeds 2021 2022 2023 \
  --epochs 10
```

`scripts/` 是实验编排层：它不随本包一起打包（`pyproject.toml` 只打包
`src/itransformer_repro`），直接调用 `train.run_experiment`，已存在
`metrics.json` 的运行跳过并打印 `SKIP`。脚本会为每个模型读取 `configs/` 中对应的
preset（如 `patchtst_etth1_h96.yaml`），只替换 `model` 段与学习率/批大小/早停，
避免批量跑基线时误用 iTransformer 的 `d_model=512`；`--no-model-presets` 可关闭。
`--seq-lens 48 96 192 336 720` 可在同一矩阵上增加回看长度扫描。

### 2.5 汇总、对照与绘图

```bash
# 汇总 results/raw/ 下全部结果，输出 mean±std CSV 与 Markdown 主表
python scripts/aggregate.py

# 与论文报告值逐档对照
python scripts/compare_paper.py

# 生成主结果、消融与回看曲线图
python scripts/make_figures.py

# 单次运行的逐 horizon / 逐变量误差与最差样本
python scripts/analyze_errors.py \
  results/raw/itransformer/ETTh1/h96/seed2021 --top-k 20
```

### 2.6 重评已完成的运行

```bash
itransformer-repro evaluate \
  --run-dir results/raw/itransformer/ETTh1/h96/seed2021
```

`evaluate_run` 读取该目录的 `config.json` 与 `model.pt`，重建数据与模型后重算测试集
指标，并写入 `metrics_eval.json`（含 `mse/mae/rmse/mape`、`recomputed: true`、
`device`，以及与原 `metrics.json` 的 `mse_delta/mae_delta/matches_original`）。
该文件与 `metrics.json` 并存，不会覆盖原始记录。

### 2.7 代码质量与测试

```bash
ruff format --check .
ruff check .
pytest -q
```

## 3. 运行产物与字段

每次运行写入 `config.run_dir()`，即：

```text
results/raw/<model>/<dataset>/h<S>/[<variant>/]seed<seed>/
├── config.json     # 全部超参 + git_commit/git_dirty + environment + n_params
├── metrics.json    # 测试指标、最佳 epoch、参数量、窗口数、耗时、推理耗时、显存
├── train.log       # 每个 epoch 的 train_loss/val_mse/val_mae 与耗时
├── pred.npy        # 测试集预测，形状 (n_windows, pred_len, n_vars)，标准化空间
├── true.npy        # 对应真值，用于 analyze_errors.py
├── model.pt        # 最佳验证 MSE 对应的权重（被 .gitignore 排除）
└── train_state.pt  # 可选：模型 + 优化器 + epoch + 随机状态（被 .gitignore 排除）
```

`config.json` 由 `run_experiment` 写出，除 `ExperimentConfig.to_dict()` 的全部字段
（`experiment`、`seed`、`device`、`runs_dir`、`data`、`model`、`train`）外，还包含：

- `git_commit`：`utils.git_commit()` 返回的短提交哈希，非 git 环境为 `"unknown"`；
- `git_dirty`：运行时代码是否有未提交改动（`true` 时结果不可严格复现）；
- `environment`：`python`、`platform`、`torch`、`numpy`、`cuda_available`、
  `cuda_version`、`cudnn_version`、`gpu`（CPU 上为 `"cpu"`），CUDA 可用时另有
  `gpu_memory_gb`；
- `n_params`：可训练参数总数。

`metrics.json` 字段：

| 字段 | 含义 |
| --- | --- |
| `mse` / `mae` / `rmse` / `mape` | 测试集指标，在标准化空间计算（`mape` 为百分数，忽略接近 0 的真值） |
| `best_val_mse` | 训练过程中验证集的最低 MSE |
| `best_epoch` | 取得最低验证 MSE 的 epoch（从 1 开始；若全程无提升则为 0） |
| `n_params` | 可训练参数总数（`count_parameters`） |
| `n_windows` | 对象，含 `train`、`val`、`test` 三段窗口数 |
| `epochs_run` | 实际执行的 epoch 数（早停时会小于 `train.epochs`） |
| `epoch_time_sec` | 列表，每个 epoch 的耗时（训练 + 验证），保留 4 位小数 |
| `train_time_sec` | 纯训练耗时（含反向与参数更新，不含验证） |
| `val_time_sec` | 每轮验证前向与指标计算的累计耗时 |
| `inference_time_sec` | 测试集前向与指标计算耗时 |
| `total_time_sec` | 整次运行端到端耗时（数据加载 + 训练 + 测试） |
| `peak_memory_mb` | CUDA 峰值显存（MB，按 run 重置统计）；CPU 上固定为 `0.0` |
| `device` | 实际使用的设备字符串，如 `"cpu"`、`"cuda"` |

另含用于索引的 `experiment`、`seed`、`dataset`、`model`、`seq_len`、`pred_len`。
表格汇总时 `epoch_time_sec` 是逐 epoch 列表，`n_windows` 是嵌套对象，不要当作标量。

## 4. 复现纪律

1. **验证集只用于选模**：`val` 只用于挑选最佳 epoch 与触发早停
   （`train.patience > 0` 时），不得参与超参搜索之外的任何调参决策。
2. **测试集只评估一次**：方案冻结后（配置、种子、epoch 数确定）才在 `test` 上运行，
   且同一配置只报告一次测试结果。
3. **不因测试结果回头调参**：看到测试指标后再改配置属于信息泄漏；如需改动，必须
   作为新的预注册配置重新运行，并在报告中说明。
4. **每次报告附三样证据**：`git_commit`、`environment`（来自 `config.json`）、
   `seed`（来自运行目录名与 `metrics.json`）。
5. **冒烟结果不上报告**：`configs/smoke.yaml` 只验证链路，文件内已注明其结果不得
   作为实验证据。
6. **不手工填表**：结果表与图只能由 `scripts/aggregate.py`、`scripts/make_figures.py`
   从 `results/raw/` 生成。
7. **公开基准的标签可见**：ETT/Weather/ECL 等公开数据集测试标签公开，存在被间接
   利用的风险；本项目测试集只评估一次、不参与调参与早停，报告中仍应声明这一潜在偏差。

## 5. 论文与本项目的差异表

下表的"论文说法"仅针对 iTransformer 原文（arXiv:2310.06625 / ICLR 2024）与官方
仓库 <https://github.com/thuml/iTransformer>；本项目一律以 `src/`、`configs/` 中的
实际实现为准。

| 差异点 | 论文说法 | 官方代码或本仓库做法 | 本项目处理方式 |
| --- | --- | --- | --- |
| PEMS 预测步长 | 正文写 PEMS 的步长为 `{12, 24, 36, 48}`，表 4 写 `{12, 24, 48, 96}` | 官方脚本按具体数据集分别配置 | 未实现 PEMS 数据集，不参与实验；此处只记录原文自相矛盾 |
| 批大小 | 论文配方写 32 | 官方脚本在小数据集上常用 8/16 | `configs/default.yaml` 用 16；`dlinear`/`patchtst`/`transformer` 用 32；`smoke.yaml` 用 8，`TrainConfig` 的 dataclass 默认值为 32。实际值均写入 `config.json` |
| 学习率 | 论文只给网格 `{1e-3, 5e-4, 1e-4}`，未说明逐数据集取值 | 官方代码按数据集调整 | 不做网格搜索：iTransformer 固定 `1e-3`，PatchTST 固定 `1e-4`（见对应 YAML），差异在报告中说明 |
| 学习率调度与早停 | 论文未交代调度策略与早停条件 | 官方训练脚本使用固定学习率与早停回调 | 本项目实现 `train.patience` 早停（默认 0 即关闭）；**未实现**学习率调度器，全程固定 lr |
| 指标口径 | 在标准化空间报告 MSE/MAE，未反归一化 | 官方同样在标准化空间评估 | `metrics.compute_all` 在标准化空间计算，`pred.npy` 亦为标准化值；不同数据集的绝对数值不可直接换算回原始单位 |
| Weather 表 3 与表 6 | 表 3 的 MAE 为 0.278，表 6 为 0.279；表 6 的 `Attn-Attn` 变体 MSE 0.255，略优于完整模型 0.258 | 无论文附加说明 | 本项目如实报告：消融不保证完整模型最优；两个表号与数值原样记录，不取平均、不四舍五入成一致 |
| ETT 窗口计数 | 表 4 报告 ETTh1 的 train/val/test 为 `8545/2881/2881` | 官方 `Dataset.__len__` 为"段长 − T − S + 1"，且只取前 20 个月 | 本项目沿用官方公式与 12/4/4 月份协议，ETTh1 在 `T=S=96` 时为 `8449/2785/2785`；论文数字=本口径+`pred_len`（只数回看起点），`check-data` 的 `paper_table4_windows` 同步给出 `8545/2881/2881`，两种口径都记录，不修改协议去对齐论文 |
| ETT 数据范围 | 只说明 12/4/4 个月 | 官方只用文件前 20 个月，末尾月份丢弃 | `split_borders` 只使用前 `num_train+num_val+num_test` 行，文件多余部分不参与 |
| 特征模式 | 含 `S`、`MS`、`M` 三种 | 官方 `--features` 可选 | 仅支持 `M`（`FEATURE_MODES = ("M",)`），其余取值被 `validate_config` 拒绝 |
| 数据集范围 | 8 个基准含 PEMS、Solar | 官方提供完整数据链接 | 注册表含 8 个 CSV 基准，但主实验矩阵只跑 ETTh1/ETTh2/Weather/ECL；PEMS、Solar 未实现，Traffic/Exchange 已注册但不在矩阵内 |
| 基线与消融实现 | PatchTST 为完整方法；消融按"变量维/时间维"替换 | 官方不提供本项目所用的时间维注意力消融参考实现 | PatchTST 与 Vanilla Transformer 为简化基线；`temporal_module=attention` 为分组近似实现（见 `docs/architecture.md`），与原文文字对应但不保证逐行一致 |

## 6. 报告写作要点

- **均值 ± 标准差**：每个（数据集 × 预测长度 × 模型）单元报告三个种子的
  `mean ± std`，并注明种子数；只有单个种子时不得写成均值。
- **硬件与耗时口径**：报告 GPU 型号与显存（`config.json` 的 `environment.gpu`、
  `gpu_memory_gb`）、CUDA/torch 版本；耗时说明口径——`train_time_sec` 为纯训练、
  `val_time_sec` 为逐轮验证、`inference_time_sec` 为测试推理、`total_time_sec`
  为端到端（含数据加载与测试），`epoch_time_sec` 是单 epoch 的训练 + 验证耗时，
  `peak_memory_mb` 仅 CUDA 有值，CPU 上为 0。
- **图表必须由脚本生成**：主结果表、预测曲线、消融柱状图一律来自
  `scripts/aggregate.py` 与 `scripts/make_figures.py`，不手工绘制、不手抄数值；
  变量注意力热图与 top-k 变量对由 `scripts/attention_maps.py` 生成。
- **如实报告差距**：本课程配置为 10 epoch、无网格搜索、简化基线，绝对数值预计低于
  论文；应写明差距及可能原因（训练预算、超参、实现简化），不得只挑好看的单元格。
- **可追溯**：正文每个结果标注
  `results/raw/<model>/<dataset>/h<S>/[<variant>/]seed<seed>` 与对应 `git_commit`，
  附录列出全部配置文件的哈希或内容。
- **区分证据等级**：冒烟运行（`smoke.yaml`）只证明代码可运行，不进入结果表。
- **当前状态**：审计环境为 CPU-only，仅 ETTh1 已下载并完成数据校验；完整
  `4 数据集 × 4 预测长度 × 4 模型 × 3 种子` 矩阵尚未运行，本文与 README 的数值表
  保持"待运行"，不以任何方式推测或填充。

## 7. 论文参考值（仅作量级参照）

下表为论文报告的 4 个预测长度（96/192/336/720）平均 MSE/MAE，仅用于判断复现结果
是否在同一量级，不可与本项目的 10-epoch 配置直接比较。

| 数据集 | MSE | MAE |
| --- | --- | --- |
| ECL | 0.178 | 0.270 |
| Traffic | 0.428 | 0.282 |
| Weather | 0.258 | 0.278 |
| Exchange | 0.360 | 0.403 |
| ETT 系列（ETTh1/ETTh2/ETTm1/ETTm2 平均） | 0.383 | 0.399 |

说明：

- Traffic（N=862）与 Exchange 已注册但不在主实验矩阵内，参考值仅供扩展实验使用。
- 论文表格内部存在不一致（见第 5 节 Weather 行与 ETT 窗口计数行），引用时以表格
  编号注明出处。
