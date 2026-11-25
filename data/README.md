# 数据准备与放置说明

本目录只保存数据说明文档。原始 CSV **不随仓库分发**，需要按本文说明自行下载到
`data/raw/`（该目录已被 `.gitignore` 排除）。数据集名称、变量数、路径与切分规则
均以 `src/itransformer_repro/data.py` 中的 `DATASET_SPECS` 为唯一事实来源，本文与
代码保持一致。

## 1. 已注册数据集

`data.name` 区分大小写，必须与下表第一列完全一致。

| 数据集 | 变量数 N | 官方行数 | 采样频率 | CSV 相对路径（以 `data/raw` 为根） | 切分规则 |
| --- | --- | --- | --- | --- | --- |
| ETTh1 | 7 | 17420 | 1h | `ETT-small/ETTh1.csv` | ETT 固定前 12/4/4 个月 |
| ETTh2 | 7 | 17420 | 1h | `ETT-small/ETTh2.csv` | ETT 固定前 12/4/4 个月 |
| ETTm1 | 7 | 69680 | 15min | `ETT-small/ETTm1.csv` | ETT 固定前 12/4/4 个月 |
| ETTm2 | 7 | 69680 | 15min | `ETT-small/ETTm2.csv` | ETT 固定前 12/4/4 个月 |
| Weather | 21 | 52696 | 10min | `weather/weather.csv` | 70/10/20 时间顺序 |
| ECL | 321 | 26304 | 1h | `electricity/electricity.csv` | 70/10/20 时间顺序 |
| Traffic | 862 | 17544 | 1h | `traffic/traffic.csv` | 70/10/20 时间顺序 |
| Exchange | 8 | 7588 | 1d | `exchange_rate/exchange_rate.csv` | 70/10/20 时间顺序 |

说明：

- `ECL` 即 Electricity，变量数 321；`Exchange` 为日频汇率数据，变量数 8。
- 变量数由代码注册表固定，读取到的列数与之不符时 `build_datasets` 会直接报错。
- Solar（`solar/solar_AL.txt`）与 PEMS 系列**未注册**：Solar 原始文件无表头、需要
  单独的解析分支，PEMS 的预测步长在论文中也存在前后不一致；两者都不在本项目范围
  内，无需下载。

## 2. 目录结构

下载完成后，`data/raw/` 应呈现如下结构（文件名大小写敏感）：

```text
data/raw/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── weather/
│   └── weather.csv
├── electricity/
│   └── electricity.csv
├── traffic/
│   └── traffic.csv
└── exchange_rate/
    └── exchange_rate.csv
```

配置中的 `data.root` 默认为 `data/raw`，`data.file` 为相对该根的路径，最终路径由
`data.py` 的 `resolve_path` 拼接得到（相对路径以仓库根目录为工作目录）。

## 3. CSV 格式要求

1. 第一列是 `date` 时间列，其余列均为数值列（与官方基准文件一致）。
2. `load_frame` 读取流程：`pd.read_csv` → 删除列名（不区分大小写）等于 `"date"`
   的时间列 → `pd.to_numeric(errors="coerce")` → 转为 `float32` 矩阵。
3. 校验规则：
   - 文件不存在：抛出 `FileNotFoundError`，提示按本文下载数据；
   - 删除时间列后没有数值列：抛出 `ValueError`；
   - 解析后存在任何 `NaN` 或 `inf`：抛出 `ValueError`，并报告异常值个数。
     本项目不会做插值或填充，缺失值必须在数据准备阶段解决。

## 4. 切分规则与窗口数

切分在 `data.py` 的 `split_borders` 中实现，记总行数为 `n`、回看长度为 `T`、
预测长度为 `S`。

### 4.1 ETT 系列（`split_rule="ett"`）

固定使用文件**前 12 / 4 / 4 个月**，其余月份不参与任何实验：

- `month = 30 * 24 * scale`，其中 15 分钟数据的 `scale = 4`，小时数据 `scale = 1`；
- 1 小时数据：`num_train/num_val/num_test = 8640/2880/2880`；
- 15 分钟数据：`num_train/num_val/num_test = 34560/11520/11520`。

### 4.2 其余数据集（`split_rule="ratio"`）

按时间顺序 70/10/20 切分，不打乱：

- `num_train = int(0.7n)`，`num_test = int(0.2n)`，`num_val = n - num_train - num_test`。

### 4.3 边界与窗口数

验证段与测试段的窗口起点需要向前借用 `T` 行构造输入，因此边界如下：

| 段 | ETT | 70/10/20 |
| --- | --- | --- |
| train | `(0, num_train)` | `(0, num_train)` |
| val | `(num_train - T, num_train + num_val)` | `(num_train - T, num_train + num_val)` |
| test | `(num_train + num_val - T, num_train + num_val + num_test)` | `(n - num_test - T, n)` |

任意一段的窗口数为 `end - start - T - S + 1`（见 `WindowDataset.__len__`）。
标准化统计量只在训练段 `(0, num_train)` 上拟合，随后应用到全量数据，验证/测试段
共用同一组均值与标准差。

## 5. ETT 只取前 20 个月：窗口计数口径

官方协议只使用 ETT 文件的**前 20 个月**（12+4+4），文件末尾多出的月份被丢弃：
ETTh1 官方文件共 17420 行，但只有前 14400 行进入切分；文件不足 14400 行时
`split_borders` 会报错"不足以覆盖 12/4/4 个月协议"。

因此本仓库的窗口数与论文表 4 中 ETTh1 的 `8545/2881/2881` **不是同一计数口径**：

- 按本仓库公式（`T = 96`、`S = 96`），ETTh1 三段窗口数为 `8449/2785/2785`；
- 论文表 4 的 `8545/2881/2881` 等于"只数回看起点、不扣 `pred_len`"的计数，即
  本仓库口径 + `S`；`check-data` 的 `paper_table4_windows` 字段会同时给出这组数；
- 两种口径在 `data.py` 模块注释、本文与 `docs/experiments.md` 差异表中均有说明；
  本项目**不修改切分协议去凑论文数字**，实际参与训练的窗口数以 `split_windows`
  为准。

数值请以 `check-data` 打印的 `split_windows` 与 `paper_table4_windows` 为准，
不要手工抄写上面手算的示例值。

## 6. 下载来源

数据不随仓库分发，请通过下列渠道获取：

- **ETT 数据集**（ETTh1/ETTh2/ETTm1/ETTm2）：<https://github.com/zhouhaoyi/ETDataset>，
  使用其中的 `ETT-small` 目录。
- **Weather / ECL (Electricity) / Traffic / Exchange**：按官方仓库
  <https://github.com/thuml/iTransformer> 的 README 提供的数据链接下载。

本文不给出任何其他下载地址；若官方仓库链接发生变化，以官方仓库 README 为准。
下载后请按第 1 节表格中的相对路径命名并放置（大小写敏感）。

本次审计的实际下载记录（可作为版本参照）：

| 数据集 | 下载日期 | 文件大小（bytes） | SHA256 |
| --- | --- | --- | --- |
| ETTh1 | 2026-11-24 | 2589657 | `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066` |

ETT 官方仓库无版本标签，上表以固定文件哈希作为数据版本；重新下载后请用
`check-data` 复核 `size_bytes`、`sha256` 与 `rows`，不一致时先查明来源再训练。

## 7. 校验命令

放置完成后，先校验文件、变量数与切分后各段窗口数：

```bash
itransformer-repro check-data --config configs/default.yaml
```

该命令输出一段 JSON 数据集卡片（由 `dataset_card` 生成），字段包括：

| 字段 | 含义 |
| --- | --- |
| `dataset` | 配置中的 `data.name` |
| `path` | 解析后的 CSV 绝对/相对路径 |
| `size_bytes` / `sha256` | 文件大小与 SHA256，用于记录数据版本 |
| `rows` / `expected_rows` / `rows_match` | 实际数据行数（不含表头与时间列）/ 官方行数 / 是否一致 |
| `n_vars` / `expected_n_vars` | 实际变量数 / 注册表期望变量数 |
| `vars_match` / `passed` | 变量数是否一致 / 总体是否通过 |
| `freq` / `split_rule` | 采样频率标签 / 切分规则 |
| `seq_len` / `pred_len` | 本次配置的 `T` 与 `S` |
| `split_windows` | 严格样本数口径的 `train/val/test` 窗口数 |
| `paper_table4_windows` | 只数回看起点（不扣 `S`）的窗口数，用于对齐论文表 4 |
| `borders` | 三段边界 `[start, end]`（变量数不符时为 `null`） |

变量数不符时命令会打印 `FAIL` 并返回非零退出码；行数与官方版本不同时默认只打印
`WARN`，加 `--strict` 可让其返回失败。`check-data` 支持 `--set` 覆盖（例如
`--set data.seq_len=192 data.pred_len=336 --out data_card.json`）；`configs/` 现有
配置默认指向 ETTh1，校验其它数据集可用 `--set data.name=Weather` 等覆盖，或复制
YAML 修改 `data.name`、`data.file`。

## 8. 版本控制

`data/raw/`、`data/processed/`、`*.npy`、`*.pt` 均已在 `.gitignore` 中排除。
公开 GitHub 仓库**不提交任何原始数据**，只保留本说明文档与校验信息；
`results/raw/` 下的 `config.json`、`metrics.json`、`train.log` 等属于本地运行产物，
默认也不入库（见 `results/README.md`）。如需向课程提交原始结果，请单独打包，
**不要提交原始数据或模型权重**。

## 9. 许可证与引用

- ETT 数据集（ETDataset）由 Informer 作者发布，引用格式请遵循其仓库要求：
  Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series
  Forecasting*, AAAI 2021。
- Weather、ECL、Traffic、Exchange 等基准遵循各自原始发布方的许可；本仓库不重新
  分发这些数据，仅说明获取方式。
- 参考代码 <https://github.com/thuml/iTransformer> 采用 MIT License，本项目的数据
  管道与切分协议参照其 `data_provider` 实现。
