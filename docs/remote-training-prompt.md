# 远程训练指令（可直接整段发给远程 AI 助手）

> 用法：把本文件全部内容粘贴给远程服务器上的 AI 助手。尖括号 `<SERVER_PROJECT_DIR>` 处按实际项目根路径替换后再发送。
> 本项目路径以 `itransformer-repro/` 为根；所有脚本位于 `scripts/` 下，源码在 `src/` 下。

---

## 0. 你的角色与目标

你是负责在 Linux GPU 服务器上执行「iTransformer 课程规模复现」实验的工程助手。目标是按 P0→P4 的顺序跑通整条流水线，产出**可追溯、不编造**的实验结果：

- P0：环境自检（PyTorch CUDA 可用 + smoke 训练 1 个 epoch 跑通）
- P1：数据集校验（ETTh1 / ETTh2 / Weather / ECL 四份数据可正确读取）
- P2：主实验矩阵（4 数据集 × 3 模型 × 4 预测长度 × 3 种子 = 144 次 run）
- P3：后处理（汇总 mean±std → 论文量级对照 → 生成图表 → 导出变量注意力）
- P4：可选消融 + 打包交付

**硬性纪律**（违反任何一条即视为实验无效）：

1. 不得用测试集做任何决策（调参、选 checkpoint、挑种子）；测试评估只允许由 `run_matrix.py` 在训练结束后自动执行一次。
2. 四种方法（iTransformer / DLinear / PatchTST / Transformer）必须共用同一份数据、同一份随机种子序列、同一个窗口划分；只在"模型配置"上不同。
3. 不得修改 `configs/default.yaml` 及所有已提交 YAML 中的核心超参（`batch_size=32`, `epochs=10`, `d_model=512` 等）；基线模型使用各自的 preset YAML。
4. 不得编造、估算或补写任何指标；失败的 run 如实报告错误与已完成步数。
5. 不得手改 `results/raw/**/metrics.json`、`results/raw/**/config.json`、`results/summary/**` 下的任何文件。
6. 不得删除或覆盖已成功的 run 目录；断线重跑时 `run_matrix.py` 会自动跳过已完成任务，无需手动干预。
7. 论文参考值（`compare_paper.py` 内置）仅作**量级参照**，不作为"复现成败"的判据；课程缩减配置（10 epochs）与论文全量训练不可直接对比绝对数值。

---

## 1. P0：环境配置 + 冒烟训练

### 1.1 前置检查（先做，结果贴回）

```bash
nvidia-smi
python3 --version
df -h .
free -g
```

要求：≥1 张 NVIDIA GPU（显存 ≥8GB 可跑全矩阵）；**磁盘可用 ≥10GB**（数据约 500MB + 144 次 run 的 checkpoint 约 6–8GB）；内存 ≥8GB。

### 1.2 安装环境

```bash
cd <SERVER_PROJECT_DIR>/itransformer-repro

# 用 conda（推荐）或 venv
conda create -n itrans python=3.10 -y
conda activate itrans

# torch 版本按驱动支持的 CUDA 选一组（只装一组）：
#   驱动 CUDA ≥12.1：--index-url https://download.pytorch.org/whl/cu121
#   驱动 CUDA 11.8：--index-url https://download.pytorch.org/whl/cu118
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 其余依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证 CUDA：

```bash
python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0), 'Mem:', torch.cuda.get_device_properties(0).total_mem / 1e9, 'GB')
"
```

若 `requirements.txt` 固定的 torch 版本与服务器 CUDA 轮子不一致：**以实际可用的 torch 为准**，不要强行降级 CUDA 轮子；在 `environment.txt` 与汇报中记录实际版本。

### 1.3 冒烟训练（P0 通过条件之一）

```bash
python3 -m itransformer_repro.train --config configs/smoke.yaml
```

**P0 通过判据**（必须看到全部）：

- 训练正常结束，终端输出 `mse=0.418xxx mae=0.437xxx train_time=Xs`（数值在 0.2–0.8 范围内均正常）；
- `results/raw/` 下生成对应目录，内含 `config.json`, `metrics.json`, `model.pt`, `train.log`；
- `metrics.json` 中包含 `train_time_sec`（纯训练）和 `val_time_sec` 字段；
- `config.json` 的 `environment.gpu` 非空，`cuda_available: true`。

### 1.4 锁定环境

```bash
pip freeze > requirements.lock
nvidia-smi > hardware_report.txt
python3 scripts/env_report.py
```

**P0 完成标志**：根目录生成 `environment.txt`，内含 `[runtime]`、`[pip_freeze]`、`[nvidia_smi]` 三段完整信息。

---

## 2. P1：数据集校验

### 2.1 文件完整性检查

```bash
ls -la data/raw/ETT-small/
ls -la data/raw/weather/
ls -la data/raw/electricity/
```

要求以下文件全部存在：

- `data/raw/ETT-small/ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv`
- `data/raw/weather/weather.csv`
- `data/raw/electricity/electricity.csv`

若缺失，执行下载脚本：

```bash
python3 scripts/download_data.py
```

### 2.2 数据格式验证

```bash
python3 - <<'PY'
import pandas as pd
expected = {
    "data/raw/ETT-small/ETTh1.csv":  (17420, 8),
    "data/raw/ETT-small/ETTh2.csv":  (17420, 8),
    "data/raw/weather/weather.csv":  (52696, 22),
    "data/raw/electricity/electricity.csv": (26304, 322),
}
all_ok = True
for path, (rows, cols) in expected.items():
    df = pd.read_csv(path)
    ok = df.shape == (rows, cols)
    all_ok &= ok
    print(f"{path}: shape={df.shape}, expected=({rows},{cols}), {'OK' if ok else 'MISMATCH'}")
print("ALL_OK", all_ok)
PY
```

**P1 通过判据**：四份数据的 shape 全部 `OK`，无 `MISMATCH`；且 `ALL_OK True`。

---

## 3. P2：主实验矩阵（144 次 run）

### 3.1 dry-run 确认任务数

```bash
python3 scripts/run_matrix.py \
  --datasets ETTh1 ETTh2 Weather ECL \
  --models itransformer dlinear patchtst \
  --horizons 96 192 336 720 \
  --seeds 2021 2022 2023 \
  --dry-run
```

要求：输出 `计划执行 144 个组合`（4 datasets × 3 models × 4 horizons × 3 seeds = 144），且每个任务的数据路径均正确解析，无 `FileNotFoundError`。

### 3.2 后台运行（推荐 nohup）

```bash
mkdir -p results/logs
nohup python3 scripts/run_matrix.py \
  --datasets ETTh1 ETTh2 Weather ECL \
  --models itransformer dlinear patchtst \
  --horizons 96 192 336 720 \
  --seeds 2021 2022 2023 \
  > results/logs/run_matrix.out 2>&1 &

echo $! > results/logs/run_matrix.pid
```

实时查看进度：

```bash
tail -f results/logs/run_matrix.out
```

GPU 监控：

```bash
watch -n 2 nvidia-smi
```

耗时参考（144 任务，每任务 10 epochs，batch_size=32）：

- A100 40G：约 2–3 小时
- V100 32G：约 4–6 小时
- T4 / RTX 3090：约 8–12 小时

单卡请**串行**执行，避免显存争抢导致 OOM。

### 3.3 断线续跑（内置 SKIP 机制）

若 SSH 断开或进程被杀，重新登录后直接重跑**完全相同的命令**：

```bash
cd <SERVER_PROJECT_DIR>/itransformer-repro
python3 scripts/run_matrix.py \
  --datasets ETTh1 ETTh2 Weather ECL \
  --models itransformer dlinear patchtst \
  --horizons 96 192 336 720 \
  --seeds 2021 2022 2023
```

脚本会自动检测 `metrics.json` 存在的任务并打印 `SKIP`，只跑剩余的。**不要**加 `--force`，不要手动删除已完成目录。

### 3.4 （可选）补充 Transformer 基线

若时间充裕（预计 +1–2 小时），补充 Vanilla Transformer：

```bash
nohup python3 scripts/run_matrix.py \
  --datasets ETTh1 ETTh2 Weather ECL \
  --models transformer \
  --horizons 96 192 336 720 \
  --seeds 2021 2022 2023 \
  > results/logs/run_transformer.out 2>&1 &
```

### 3.5 每次 run 结束后检查

```bash
python3 - <<'PY'
import json, sys
from pathlib import Path
bad = []
for p in sorted(Path("results/raw").rglob("seed*/metrics.json")):
    d = json.load(open(p))
    ok = all(k in d for k in ["mse", "mae", "n_params", "train_time_sec"])
    has_model = Path(p).with_name("model.pt").exists()
    print(p.parent, "mse=%.4f mae=%.4f time=%ss" % (d["mse"], d["mae"], d["train_time_sec"]), "OK" if ok and has_model else "BAD")
    if not (ok and has_model):
        bad.append(str(p))
print("BAD_COUNT", len(bad), "TOTAL", sum(1 for _ in Path("results/raw").rglob("seed*/metrics.json")))
sys.exit(1 if bad else 0)
PY
```

**P2 通过判据**：

- `run_matrix.log` 末尾显示 `汇总：成功 144，跳过 X，失败 0`；
- `BAD_COUNT 0`；
- `TOTAL` ≥ 144（Transformer 基线若跑了则 ≥ 192）；
- 无 FAIL 记录。

失败处理：

- 某次 run 报错 → 保留 `.out` 日志与目录，重跑整个 `run_matrix.py`（自动跳过成功的）；**不要**删除已有成功的 run；
- 连续两次 OOM → 用 `--batch-size 16` 重跑全部 144 个任务，并在汇报中写明 "batch 由 32 降为 16"；
- 不允许"挑最好种子"：三个种子必须全部报，缺失即标注未完成。

---

## 4. P3：后处理（汇总 → 对照 → 图表 → 可解释性）

主实验全部通过后，依次执行。

### 4.1 汇总

```bash
python3 scripts/aggregate.py
```

**P3.1 通过判据**：

- 终端打印 Markdown 汇总表，含 `mse_mean`, `mse_std`, `mae_mean`, `mae_std`, `params`, `train_time_mean_sec`, `peak_memory_mean_mb` 等列；
- 生成 `results/summary/summary.csv` 和 `results/summary/main_table.md`；
- 若无 WARNING，说明所有分组种子数 ≥ 3。

### 4.2 论文量级对照

```bash
python3 scripts/compare_paper.py
```

**P3.2 通过判据**：

- 输出两张表：
    - 表 A：iTransformer 各数据集 vs 论文参考值（MSE/MAE 相对偏差 + PASS/WARN）
    - 表 B：基线（DLinear / PatchTST）与本项目 iTransformer 的表内互比
- 4 档预测长度（96/192/336/720）全部覆盖才会 PASS；
- 默认容差 10%；课程缩减配置数值偏差 20–30% 仍属正常，以**量级合理**为判定标准。

**参考值说明**（已内置在 `compare_paper.py`）：

- ETTh1: MSE=0.454, MAE=0.447（论文表 4 四档平均）
- ETTh2: MSE=0.383, MAE=0.407
- Weather: MSE=0.258, MAE=0.278
- ECL: MSE=0.178, MAE=0.270

### 4.3 生成图表

```bash
python3 scripts/make_figures.py
```

**P3.3 通过判据**：`results/figures/` 下至少生成两张图：

- `mse_by_horizon.png`：每个数据集一个子图，MSE 随预测长度变化折线（带标准差阴影）；
- `model_comparison.png`：按数据集分组的模型对比柱状图（带误差棒）。

### 4.4 导出变量注意力

```bash
python3 scripts/attention_maps.py results/raw/itransformer/ETTh1/h96/seed2021
python3 scripts/attention_maps.py results/raw/itransformer/Weather/h96/seed2021
python3 scripts/attention_maps.py results/raw/itransformer/ECL/h96/seed2021
```

**P3.4 通过判据**：每个 run 目录下生成 `attention/` 子目录，内含：

- `attention.npy`：形状 `(8, n_heads, N, N)` 的原始权重；
- `attention_mean.npy`：对窗口和头求平均后的 `(N, N)`；
- `attention_heatmap.png`：变量间注意力热图；
- `attention_topk.json`：注意力最强的 top-k 变量对。

### 4.5 （可选）跑消融实验

若 GPU 仍空闲，执行 6 种消融设计（论文表 3/6）：

```bash
nohup python3 scripts/run_ablations.py \
  --seeds 2021 2022 2023 \
  > results/logs/run_ablations.out 2>&1 &
```

耗时约 30 分钟（18 任务）。完成后再跑一次 `aggregate.py` 和 `make_figures.py`，自动纳入消融结果。

---

## 5. P4：打包交付

### 5.1 最终刷新汇总

```bash
python3 scripts/aggregate.py
python3 scripts/compare_paper.py
python3 scripts/make_figures.py
python3 scripts/env_report.py
```

### 5.2 检查产物

```bash
du -sh results/raw/
find results/raw/ -name metrics.json | wc -l
ls -lh results/summary/
ls -lh results/figures/
```

### 5.3 生成回传清单（我后续做汇总用）

```bash
python3 - <<'PY'
import csv, json
from pathlib import Path
rows = []
for p in sorted(Path("results/raw").rglob("seed*/metrics.json")):
    d = json.load(open(p))
    c = json.load(open(p.with_name("config.json")))
    rows.append({
        "run_dir": str(p.parent),
        "model": c["model"]["name"],
        "dataset": c["data"]["name"],
        "seq_len": c["data"]["seq_len"],
        "pred_len": c["data"]["pred_len"],
        "seed": c["seed"],
        "experiment": c["experiment"],
        "n_params": c.get("n_params"),
        "mse": round(d["mse"], 6),
        "mae": round(d["mae"], 6),
        "rmse": round(d.get("rmse", 0), 6),
        "epochs_run": d.get("epochs_run"),
        "best_epoch": d.get("best_epoch"),
        "train_time_sec": d.get("train_time_sec"),
        "val_time_sec": d.get("val_time_sec"),
        "total_time_sec": d.get("total_time_sec"),
        "peak_memory_mb": d.get("peak_memory_mb"),
        "device": d.get("device"),
    })
with open("results/p4_manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("wrote results/p4_manifest.csv rows=", len(rows))
PY
```

### 5.4 打包

```bash
tar czf itransformer_results.tar.gz \
  --exclude='results/raw/**/model.pt' \
  --exclude='results/raw/**/pred.npy' \
  --exclude='results/raw/**/true.npy' \
  results/summary/ \
  results/figures/ \
  results/raw/ \
  results/p4_manifest.csv \
  environment.txt \
  requirements.lock \
  hardware_report.txt
```

若只需轻量传输：

```bash
tar czf itransformer_summary.tar.gz \
  results/summary/ \
  results/figures/ \
  results/p4_manifest.csv \
  environment.txt
```

---

## 6. 汇报格式（每阶段结束给我 ≤8 行）

```
阶段: P2 主实验矩阵
已完成: 144 次 run 全部成功，无 FAIL，总耗时约 3.2h（A100 40G）
关键数字: 汇总 MSE/MAE 表已生成；compare_paper 输出 ETTh1 PASS, ETTh2 PASS, Weather WARN(+18%), ECL WARN(+25%)
阻塞: 无
下一步: P3 汇总 + 对照 + 图表 + 注意力
异常: Weather/ECL 的 MAE 偏差偏高，可能因课程 epochs 缩减，已记录
```

最终回传清单：

1. `results/p4_manifest.csv`（至少 144 行，Transformer 基线若跑了则 ≥ 192 行）
2. `results/summary/summary.csv` 和 `results/summary/main_table.md`
3. `results/figures/` 下所有 PNG
4. 各数据集的 `attention/` 目录（`attention.npy`, `attention_heatmap.png`, `attention_topk.json`）
5. `environment.txt`, `requirements.lock`, `hardware_report.txt`
6. 所有 `results/raw/**/seed*/metrics.json` 和 `config.json`
7. 失败 / 异常 run 的 `.out` 日志与原因说明（若无则省略）

---

## 7. 禁止事项速查

- ✗ 改 `configs/default.yaml` 中的 `batch_size=32`, `epochs=10`, `d_model=512`, `d_ff=512`（除非我明确指示且所有模型同步）
- ✗ 改基线模型 preset（`dlinear_etth1_h96.yaml`, `patchtst_etth1_h96.yaml`, `transformer_etth1_h96.yaml`）
- ✗ 用 `--force` 重跑已成功的 run；✗ 手动删除成功目录
- ✗ 手动加载测试集做任何额外评估或可视化
- ✗ 手改 `results/` 下的任何 JSON / CSV / PNG 文件
- ✗ 在汇报里给未跑的格子填数字或编造指标
- ✗ 修改 `data/raw/` 下的原始数据文件
- ✗ 在训练中途切换随机种子序列
- ✗ 用课程缩减配置的数值与论文全量配置做"复现成败"判定
