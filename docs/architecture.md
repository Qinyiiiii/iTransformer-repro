# 架构说明

本文说明 `src/itransformer_repro/` 的模块边界、iTransformer 的数据流与公式、
四个模型的结构差异、消融开关的实现语义，以及本项目与官方实现的差异。文中出现的
类名、函数名、字段名均可在源码中直接检索到。

## 1. 模块职责

| 模块 | 职责 | 关键函数 / 类 |
| --- | --- | --- |
| `config.py` | 读取 YAML、点号覆盖、取值校验；配置是参数唯一来源 | `load_config`、`apply_overrides`、`validate_config`、`coerce_scalar`、`ExperimentConfig`、`DataConfig`、`ModelConfig`、`TrainConfig` |
| `data.py` | CSV 读取、时间顺序切分、训练段标准化、滑窗构造、数据集卡片 | `DATASET_SPECS`、`DatasetSpec`、`load_frame`、`split_borders`、`StandardScaler`、`WindowDataset`、`DataBundle`、`build_datasets`、`dataset_card` |
| `models.py` | iTransformer 主模型、消融组件、三个对照模型与工厂函数 | `ITransformer`、`InvertedBlock`、`VariateAttention`、`VariateMlp`、`PositionwiseFFN`、`TemporalAttention`、`DLinear`、`PatchTST`、`VanillaTransformer`、`build_model` |
| `train.py` | 训练循环、验证集选模、测试评估、运行产物落盘 | `make_loader`、`train_one_epoch`、`evaluate_model`、`run_experiment`、`run_exists` |
| `evaluate.py` | 加载已保存的 `config.json` 与 `model.pt` 复评测试集 | `config_from_dict`、`evaluate_run` |
| `metrics.py` | 标准化空间内的指标计算 | `mse`、`mae`、`rmse`、`mape`、`compute_all` |
| `utils.py` | 随机性控制、日志、JSON 读写、设备解析、git 与环境记录 | `seed_everything`、`get_logger`、`save_json`、`load_json`、`count_parameters`、`resolve_device`、`git_commit`、`environment_info` |
| `cli.py` | 命令行入口与子命令分发 | `main`、`build_parser`、`collect_overrides`、`command_train`、`command_check_data`、`command_evaluate` |

依赖方向是单向的：`cli → train/evaluate → data/models/metrics/utils → config`，
`config` 不依赖其它业务模块。

## 2. iTransformer 数据流

```
输入 (B, T, N)
  └─ transpose(1, 2)              → (B, N, T)        每个变量一条完整回看序列
  └─ Linear(T, D)                 → (B, N, D)        每个变量嵌入成一个 token
  └─ L × InvertedBlock            → (B, N, D)        变量维注意力 + 时间维 FFN
  └─ LayerNorm                    → (B, N, D)
  └─ Linear(D, S)                 → (B, N, S)        每个 token 投影到预测长度
  └─ transpose(1, 2)              → (B, S, N)        输出与输入同布局
```

关键点：

- **无位置编码**：`use_pos_embed` 默认 `False`，`position_embedding` 为 `None`。
  若显式打开，位置编码形状是 `(1, N, D)`，加在**变量 token**上，而不是时间步上。
- **变量维注意力**：`VariateAttention` 用 `nn.MultiheadAttention(batch_first=True)`
  在 `N` 个维度为 `D` 的 token 之间做自注意力，注意力权重
  $A_{i,j} \propto \exp\left(q_i^{\top} k_j / \sqrt{d_k}\right)$，即变量 $i$ 与变量
  $j$ 的依赖强度。
- **残差 + 后归一化（post-norm）**：`InvertedBlock.forward` 先计算
  `norm1(x + Variate(x))`，再计算 `norm2(x + Temporal(x))`，与官方
  `EncoderLayer` 的顺序一致；两个子层都可被开关置空（置空时连同残差一起跳过）。
- **嵌入后 dropout**：`ITransformer.forward` 在 `Linear(T, D)` 与可选位置编码之后、
  进入编码块之前有一层 `nn.Dropout`；全部编码块之后再对 token 做一次最终
  LayerNorm，然后才进入投影层。
- **变量维 LayerNorm**：`nn.LayerNorm(d_model)` 作用在 `(B, N, D)` 的最后一维，即
  对**每个变量 token 的 D 维表征**做归一化（论文所称"在变量维做 LayerNorm"），
  不是跨 `N` 求统计量。

对应论文的三个公式（`X_{:,n} ∈ R^T` 表示第 $n$ 个变量的整段回看序列）：

$$
h_n^{0} = \mathrm{Embedding}(X_{:,n}) = W_e X_{:,n} + b_e,\qquad h_n^{0}\in\mathbb{R}^{D}
$$

$$
H^{l+1} = \mathrm{TrmBlock}\left(H^{l}\right),\qquad
H^{l}=\left[h_1^{l},\dots,h_N^{l}\right]\in\mathbb{R}^{N\times D}
$$

$$
Y_{:,n} = \mathrm{Projection}\left(h_n^{L}\right) = W_p h_n^{L} + b_p,\qquad
Y_{:,n}\in\mathbb{R}^{S}
$$

在代码中的落点：

| 公式 | 代码 |
| --- | --- |
| $h_n^{0}=\mathrm{Embedding}(X_{:,n})$ | `ITransformer.embedding = nn.Linear(seq_len, d_model)`，输入已转置为 `(B, N, T)` |
| $H^{l+1}=\mathrm{TrmBlock}(H^{l})$ | `ITransformer.blocks` 中的 `L` 个 `InvertedBlock`（`e_layers`） |
| $Y_{:,n}=\mathrm{Projection}(h_n^{L})$ | `ITransformer.projector = nn.Linear(d_model, pred_len)`，最后转置回 `(B, S, N)` |

## 3. 四个模型的结构差异

| 模型（`model.name`） | 核心结构 | 通道处理 | 位置编码 | 备注 |
| --- | --- | --- | --- | --- |
| `itransformer` | 变量 token 化 + 变量维自注意力 + 时间维 FFN | 跨通道联合建模（token 之间互相关注） | 无（默认） | 本项目的目标方法；消融开关均在此模型上 |
| `dlinear` | `SeriesDecomposition(kernel_size=25)` 分解趋势/季节项，各接一个 `Linear(T, S)` | 逐通道独立，季节/趋势头权重**跨通道共享** | 无 | 构造时 `n_vars` 被丢弃，参数量与通道数无关 |
| `patchtst` | 实例标准化 → `unfold` 切 patch（16/8）→ `Linear(patch_len, D)` → `TransformerEncoder` | 通道独立，batch 与通道合并为 `B*N` | patch 级可学习位置编码 `(1, num_patches, D)` | 简化基线，不含官方 RevIN 的全部选项 |
| `transformer` | 每个时间步 `Linear(1, D)` 作为 token → `TransformerEncoder` | 通道独立，`B*N` 条单变量序列 | 时间步位置编码 `(1, T, D)` | 未倒置的"原结构"对照，正是论文批评的设计 |

### 3.1 为什么三个基线是公平对照

- **同一数据管道**：都走 `build_datasets`，切分边界、窗口公式、训练段标准化完全一致。
- **同一训练协议**：`run_experiment` 中统一使用 `nn.MSELoss`、`torch.optim.Adam`、
  同一套早停逻辑与 `evaluate_model`，测试集只评估一次。
- **同一评估口径**：所有模型都通过 `metrics.compute_all` 在标准化空间报告
  MSE/MAE/RMSE/MAPE，并使用相同的 `pred_len`、`seq_len`、种子集合。
- **唯一变量是结构**：因此差异可归因于"变量维注意力 vs 线性映射 / patch 化 /
  时间步注意力"，这正是论文的对照逻辑。

需要注意的公平性约束（应在报告中注明）：

- 基线超参沿用各自常用配方而非与主模型完全一致：`configs/patchtst_etth1_h96.yaml`
  使用 `lr=1e-4`、`batch_size=32`、`e_layers=3`，`configs/default.yaml` 的
  iTransformer 使用 `lr=1e-3`、`batch_size=16`。这是"按各自官方配方调优"与"完全
  同参"之间的折中，不能声称是受控的同参对比。
- `dlinear`、`patchtst`、`transformer` 的参数量通常远小于 iTransformer（见
  `metrics.json` 的 `n_params`），报告中应同时给出参数量。

## 4. 消融开关的实现语义

所有消融都在 `itransformer` 上通过 `ModelConfig` 字段控制，由 `InvertedBlock`
组装。`variate_module`、`temporal_module` 的合法取值分别在 `config.py` 的
`VARIATE_MODULES = ("attention", "ffn", "none")` 与
`TEMPORAL_MODULES = ("ffn", "attention", "none")` 中定义，非法值在
`validate_config` 阶段报错。

### 4.1 `variate_module`（变量维混合）

| 取值 | 实际模块 | 语义 |
| --- | --- | --- |
| `attention` | `VariateAttention` | 在 `N` 个变量 token 之间做多头自注意力，token 可互相读取信息 |
| `ffn` | `VariateMlp` | 在 **N 维**做 token mixing：`(B,N,D)` 转置为 `(B,D,N)` → `Linear(N, 2N)` → `GELU` → `Linear(2N, N)` → 转置回 `(B,N,D)` |
| `none` | `None` | 整个变量子层**及其残差**被移除，`InvertedBlock` 只做时间维混合 |

### 4.2 `temporal_module`（时间维混合）

| 取值 | 实际模块 | 语义 |
| --- | --- | --- |
| `ffn` | `PositionwiseFFN` | 对每个变量 token 独立施加 `Linear(D, d_ff) → GELU → Dropout → Linear(d_ff, D)`，即论文的时间维 FFN |
| `attention` | `TemporalAttention` | **近似实现**：把每个 token 的 `D` 维表征切成 `n_groups=8` 组，把每组当作一个"时间 token"，在组间做多头注意力，再拼回 `D` 维 |
| `none` | `None` | 整个时间子层**及其残差**被移除 |

`TemporalAttention` 的实现细节（与本项目实现绑定，原文只给定性描述）：

- `n_groups` 在 `_build_temporal` 中**固定为 8**（不是配置项）；要求
  `d_model % 8 == 0`，否则构造时抛出 `ValueError`。
- 组维 `group_dim = d_model / n_groups`；头数取 `n_heads`，当
  `group_dim % n_heads != 0` 时**退化为单头**（`heads = 1`）。例如默认
  `d_model=512` 时 `group_dim=64`，可被 8 头整除，实际为 8 头。
- 该模块接收 `(B, N, D)`，内部 reshape 为 `(B*N, n_groups, group_dim)` 做注意力，
  形状不变，因此可直接替换 FFN 而不改动其余结构。
- **不保证与官方消融代码逐行一致**：官方 iTransformer 代码并未提供"时间维注意力"
  的参考实现，本项目只保证语义与论文文字（"在时间维用注意力"）对应，且形状与主
  模型兼容。

### 4.3 另外两个开关

| 字段 | 取值与含义 |
| --- | --- |
| `use_layernorm` | `True` 时 `norm1`、`norm2` 与最终 `norm` 为 `nn.LayerNorm(d_model)`；`False` 时替换为 `nn.Identity()`，用于"去 LayerNorm"消融 |
| `use_pos_embed` | `True` 时创建形状 `(1, N, D)` 的可学习参数并加到变量 token 上；主方法默认 `False` |

`configs/` 已附四份消融配置：

| 配置文件 | 开关 |
| --- | --- |
| `configs/ablation_no_attention.yaml` | `variate_module: none` |
| `configs/ablation_no_ffn.yaml` | `temporal_module: none` |
| `configs/ablation_variate_ffn.yaml` | `variate_module: ffn` |
| `configs/ablation_temporal_attn.yaml` | `temporal_module: attention` |

"去 LayerNorm"与"加位置编码"没有单独的 YAML，可用
`--set model.use_layernorm=false` 或 `--set model.use_pos_embed=true` 覆盖后运行。

## 5. 切分与窗口（与 `data.py` 一致）

记总行数 `n`、回看长度 `T`、预测长度 `S`。

ETT 系列（`split_rule="ett"`，`scale=4` 表示 15 分钟、否则为 1）：

- `num_train = 12 * 30 * 24 * scale`，`num_val = num_test = 4 * 30 * 24 * scale`；
- 边界：
  - `train = (0, num_train)`
  - `val = (num_train - T, num_train + num_val)`
  - `test = (num_train + num_val - T, num_train + num_val + num_test)`

其余数据集（`split_rule="ratio"`）：

- `num_train = int(0.7n)`，`num_test = int(0.2n)`，`num_val = n - num_train - num_test`；
- 边界：
  - `train = (0, num_train)`
  - `val = (num_train - T, num_train + num_val)`
  - `test = (n - num_test - T, n)`

窗口数为 `end - start - T - S + 1`，窗口起点 `i` 使用 `data[i : i+T]` 预测
`data[i+T : i+T+S]`。标准化器只在训练段拟合，再 `transform` 全量数据。

## 6. 与官方实现的差异

| 方面 | 官方 `thuml/iTransformer` | 本项目 |
| --- | --- | --- |
| 包结构 | 训练脚本与 `model/`、`data_provider/` 平铺在仓库根 | 改为 `src/itransformer_repro/` 的 src 布局，`pyproject.toml` 的 `packages = ["src/itransformer_repro"]` |
| 参数入口 | `run.py` 的 argparse 参数（`--model_id`、`--seq_len` 等） | YAML 配置 + `--set KEY=VALUE` 点号覆盖 + `validate_config` 校验 |
| 命令行 | `python run.py ...` | 控制台脚本 `itransformer-repro`，子命令 `train` / `check-data` / `evaluate` |
| 指标体系 | 各脚本内分别计算 MSE/MAE | 统一到 `metrics.compute_all`，四个模型共享 |
| PatchTST | 完整的官方实现（RevIN、channel-independence 等） | 简化基线：实例标准化 + patch + `TransformerEncoder`，只保留核心思想 |
| Vanilla Transformer | 官方仓库不提供同口径的未倒置基线 | 自实现 `VanillaTransformer`（时间步 token + 位置编码），作为"原结构"对照 |
| 特征模式 | 支持 `features ∈ {S, MS, M}` | 仅 `M`（`FEATURE_MODES = ("M",)`），`validate_config` 拒绝其它取值 |
| 数据集 | 含 PEMS、Solar、Market 等 | 仅注册 8 个基准，无 PEMS/Solar，也未实现 `.txt` 读取 |
| 标准化与指标 | 训练段标准化，指标在标准化空间 | 相同；`pred.npy` 与指标均为标准化空间，未反归一化 |
| 训练设施 | 早停、学习率策略、可选多 GPU | 早停仅由 `train.patience` 控制（默认 0 关闭），无 scheduler、无分布式 |
| 超参搜索 | 每个数据集搜索 batch/lr | 固定超参写在 `configs/*.yaml`，不做网格搜索 |

## 7. 代码可维护性：新增一个模型

保持现有接口（输入 `(B, T, N)`，输出 `(B, S, N)`）即可接入，共三步：

1. **写模型类**：在 `models.py` 增加一个 `nn.Module` 子类，`forward` 接收
   `(B, T, N)`、返回 `(B, S, N)`；如需新超参，在 `ModelConfig` 中加带默认值的字段
   （`load_config` 会用 dataclass 字段名校验 YAML，未知字段会报错）。
2. **接入工厂与注册表**：在 `build_model` 中增加 `if config.name == "<name>"` 分支，
   并把名称加入 `config.py` 的 `MODEL_NAMES`（否则 `validate_config` 拒绝该配置）。
3. **加配置并冒烟**：在 `configs/` 新增一份 YAML，先以 1 个 epoch 的小模型跑通
   `itransformer-repro train --config configs/<new>.yaml --device cpu`，再提交。

若新模型需要新的数据字段（例如协变量），还需同步修改 `DataConfig` 与
`DataBundle`；仅新增结构时无需改动数据层。
