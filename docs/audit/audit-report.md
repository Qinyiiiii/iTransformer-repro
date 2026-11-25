# iTransformer 复现仓库审计报告

- 审计对象：`d:\itransformer-repro`（分支 `main`，当前工作区有未提交修改）
- 审计日期：2026-11-24
- 审计依据：
  1. 《人工智能》课程期末项目《Python 与 GitHub 工程代码规范》（下称“课程规范”）
  2. 角色与任务提示词（下称“角色提示词”，含 §A 论文规格、§B 课程交付规格）
  3. iTransformer 论文（ICLR 2024，arXiv:2310.06625v4）
- 审计环境：Windows / Python 3.9.13 / torch 2.8.0+cpu / **CPU-only**，详见仓库根目录的
  [environment.txt](../../environment.txt)
- 验证记录：[verification-log.md](verification-log.md)
- 说明：课程规范与论文原文为外部输入，未随仓库分发；本报告只引用其条款与结论。

> 链接约定：本文件中的相对链接均相对 `docs/audit/` 目录；仓库根目录是 `../../`。

---

## 0. 结论摘要

| 维度 | 结论 |
| --- | --- |
| 仓库结构与文档 | **已补齐**：src 布局、README 12 节、REPRODUCTION、DECLARATION、CONTRIBUTING、CITATION、PR/Issue 模板齐全 |
| 环境与依赖 | **可运行**：`pip install -r requirements.txt` 后按 README 可完成 ETTh1 冒烟；`environment.txt` 已记录 Python/torch/CPU/pip freeze |
| 代码质量 | **通过**：`ruff check .`、`ruff format --check .` 全绿；`pytest` 65 项全过；总覆盖率 **89.08%**（CI 阈值 80%） |
| 可运行性 | **已验证**：真实 ETTh1 单 epoch 冒烟完成 train → evaluate → analyze → aggregate → figures → compare 全链路 |
| 论文保真 | **核心结构一致**：变量维注意力 + 时间维 FFN、post-norm、ReLU、无位置编码、`Linear(T,D)`/`Linear(D,S)`；六种消融可前向。差异见 §3.3 |
| 主实验 | **缺失（如实标注）**：CPU-only 未运行 `4 数据集 × 4 步长 × 4 模型 × 3 种子`；README/docs 数值表保持“待运行”，无任何编造数字 |
| Git/GitHub 协作 | **不合规**：仅 1 个提交、无分支/PR/Issue/标签；DECLARATION 团队姓名为“待填” |
| 需用户决策 | ① 团队姓名/学号；② 是否允许在 CPU 上跑缩减版真实实验；③ `data/raw` 与角色提示词 `dataset/` 命名冲突；④ 是否由用户完成提交/PR/打标签 |

三类清单汇总见 §2，论文核对见 §3，本次补齐内容见 §4，验证证据见 §5，剩余阻塞见 §6。

---

## 1. 审计方法

1. 从课程规范与论文原文逐条摘出 MUST / SHOULD / MAY；提取用的临时文件未随仓库保留。
2. 对仓库执行文件级核对：根目录、`src/`、`tests/`、`configs/`、`scripts/`、`docs/`、
   `.github/`、`results/`、`data/`；读取全部核心模块与配置。
3. 执行静态检查、测试、覆盖率、数据校验、真实冒烟、矩阵 dry-run、汇总/绘图脚本。
4. 以论文 §3、式 (1)–(3)、表 3/4/6、附录 A.2 为基准核对模型、数据协议、训练配方与消融。
5. 将缺口直接补齐；无法用现有算力完成的实验，保留“待运行”并写明阻塞原因，不填推测值。
6. 审计清单形成后已先期向会话汇报一次；本报告为完整交付。

---

## 2. 课程规范逐条对照

状态说明：**已满足**＝有实现且有证据；**缺失**＝无产物或无证据；**不合规**＝有产物但违反
MUST；**不适用**＝项目类型不涉及。

### 2.1 仓库结构与必要文件（课程规范 §2、§8；角色提示词 B3/B8）

| # | 要求（等级） | 仓库现状与证据 | 判定 |
| --- | --- | --- | --- |
| S1 | 根目录 README/pyproject/.gitignore/LICENSE（MUST） | 四者均存在 | 已满足 |
| S2 | 业务代码在 `src/`，测试在 `tests/`（MUST） | `src/itransformer_repro/*`、`tests/unit`、`tests/integration` | 已满足 |
| S3 | 不得把代码堆在根目录（MUST） | 根目录仅文档/配置/入口文件，无 `final*.py` | 已满足 |
| S4 | `configs/` 可版本化、无密钥/绝对路径（MUST） | 10 份 YAML，路径均为相对路径 | 已满足 |
| S5 | `data/README.md` 说明数据（MUST） | [data/README.md](../../data/README.md) 含来源、行数、切分、SHA、许可证、下载日期 | 已满足 |
| S6 | `docs/` 放架构与实验长文（SHOULD） | `docs/architecture.md`、`docs/experiments.md` | 已满足 |
| S7 | `scripts/` 有帮助信息与错误提示（MUST） | 7 个脚本均用 argparse 且带 `--help` | 已满足 |
| S8 | `results/{raw,summary,figures}` 分离，汇总/图由脚本生成（MUST） | [results/README.md](../../results/README.md)；`aggregate.py`/`make_figures.py` 生成 | 已满足 |
| S9 | Notebook 不得承载核心实现（MUST） | 仓库无 Notebook，核心全在 `src/` | 不适用 |
| S10 | `notebooks/`、`.env.example`、SECURITY.md（MAY/条件） | 无 Notebook；无外部服务故无 `.env.example`/SECURITY.md | 不适用 |
| S11 | CONTRIBUTING.md（多人项目 SHOULD） | [CONTRIBUTING.md](../../CONTRIBUTING.md) | 已满足 |
| S12 | CITATION.cff（复现项目 MAY） | [CITATION.cff](../../CITATION.cff) | 已满足 |
| S13 | REPRODUCTION.md（角色提示词 B8 MUST） | [REPRODUCTION.md](../../REPRODUCTION.md) | 已满足 |
| S14 | DECLARATION.md（角色提示词 B8 MUST） | [DECLARATION.md](../../DECLARATION.md) 存在，但**团队姓名/学号为“待填”** | **不合规** |
| S15 | CHANGELOG.md（MAY） | 未提供 | 缺失（可选） |
| S16 | 数据目录命名 `dataset/`（角色提示词 B3） | 实际为 `data/raw/`；与课程规范 §2.1 的 `data/` 一致，与角色提示词 B3 不一致 | **偏离（需决策）** |

### 2.2 Python 编码规范（课程规范 §3）

| # | 要求（等级） | 仓库现状与证据 | 判定 |
| --- | --- | --- | --- |
| C1 | 命名 snake_case/PascalCase/UPPER（MUST） | 模块/函数 snake_case，类 PascalCase，常量 UPPER | 已满足 |
| C2 | Ruff 88 列、4 空格、UTF-8（MUST） | `pyproject.toml` 设 `line-length=88`；`ruff format --check` 通过 | 已满足 |
| C3 | 导入顺序、禁止星号导入（MUST） | ruff `I` 规则通过 | 已满足 |
| C4 | 公共函数类型标注 + Google 文档字符串（MUST） | 核心模块均有 `from __future__ import annotations`、Args/Returns/Raises | 已满足 |
| C5 | 训练/数据/指标分离（MUST） | `train.py`/`data.py`/`metrics.py`/`evaluate.py` 独立 | 已满足 |
| C6 | 禁止空 `except`/静默忽略（MUST） | 仅捕获 `FileNotFoundError`/`ValueError`/`RuntimeError` 等具体异常 | 已满足 |
| C7 | 正式代码用 logging，print 仅 CLI（MUST） | `train.py` 用 logger；`cli.py` 用 print 面向用户 | 已满足 |
| C8 | 不得用 assert 校验用户输入（MUST） | `src/` 中无 `assert`（grep 为空） | 已满足 |
| C9 | 模块导入不得自动训练/下载/改文件（MUST） | 所有执行逻辑在 `main()`/函数内，`if __name__ == "__main__"` 保护 | 已满足 |
| C10 | 路径优先 `pathlib.Path`（SHOULD） | 配置、数据、脚本均使用 `Path` | 已满足 |

### 2.3 配置、依赖与环境（课程规范 §4）

| # | 要求（等级） | 仓库现状与证据 | 判定 |
| --- | --- | --- | --- |
| E1 | 声明 Python 版本与直接依赖（MUST） | `pyproject.toml` 声明 `requires-python >=3.9,<3.13`；直接依赖见 `requirements.txt`（numpy/pandas/PyYAML/matplotlib/torch） | 已满足 |
| E2 | README 写明课程默认 Python 及偏差（MUST） | README §3、REPRODUCTION §1 均说明课程 3.10/3.11 与本地 3.9.13 | 已满足 |
| E3 | 运行/开发依赖分组（MUST） | `requirements.txt` 分“运行时依赖”与“开发与质量门禁”两段（pytest/pytest-cov/ruff） | 已满足 |
| E4 | 依赖快照 + CUDA/驱动/硬件（MUST） | [environment.txt](../../environment.txt) 含 Python、torch、CUDA 可用性、CPU、pip freeze；CPU 无 `nvidia-smi` 已注明 | 已满足 |
| E5 | editable 安装、不改 `sys.path`（MUST） | README 用 `pip install -r requirements.txt`（含 `-e .`）；无 `sys.path` 修改 | 已满足 |
| E6 | 可变参数进配置/命令行、相对路径、`--help`（MUST） | 10 份 YAML + `--set/--seed/...`；全部子命令支持 `--help` | 已满足 |
| E7 | 禁止密钥/Token/隐私（MUST） | `.gitignore` 含 `.env*`；仓库无密钥；`grep` 未发现敏感值 | 已满足 |

### 2.4 AI 项目专项（课程规范 §5）

| # | 要求（等级） | 仓库现状与证据 | 判定 |
| --- | --- | --- | --- |
| A1 | 记录数据来源/链接/版本/许可证/下载日期（MUST） | data/README §6 + §9，新增 ETTh1 下载日期与 SHA256 | 已满足 |
| A2 | 说明划分规则/比例/种子/防泄漏（MUST） | data/README §4；`split_borders`；scaler 仅训练段拟合 | 已满足 |
| A3 | 预处理为可重复代码（MUST） | `data.py` 读取/校验/标准化/窗口化 | 已满足 |
| A4 | 记录文件大小/校验值/样本数/字段（MUST） | `check-data` 输出 `size_bytes/sha256/rows/n_vars` | 已满足 |
| A5 | 原始大数据不进 Git（MUST） | `.gitignore`：`data/raw/`、`*.npy`、`*.pt` | 已满足 |
| A6 | 训练/验证/测试职责分离，测试集不调参（MUST） | 最佳模型按 val MSE 选择；test 只评估一次；`train.py` 无 test 反传 | 已满足 |
| A7 | 记录 commit/配置/种子/数据版本/环境/指标（MUST） | `config.json` 含 `git_commit/git_dirty/environment/n_params`；`metrics.json` 含指标与耗时 | 已满足 |
| A8 | 随机实验多次重复并报均值±波动（MUST） | 代码支持 `--seeds`，`aggregate.py` 算 mean/std；**实际只跑了 1 个种子的冒烟** | **缺失**（主实验） |
| A9 | 区分训练/推理/端到端时间与硬件（MUST） | `metrics.json` 有 `train_time_sec/inference_time_sec/total_time_sec/peak_memory_mb` | 已满足 |
| A10 | 固定所有随机源、确定性算法、cudnn 设置（MUST） | `utils.seed_everything`；`CUBLAS_WORKSPACE_CONFIG`；`use_deterministic_algorithms(True, warn_only=True)` | 已满足 |
| A11 | 只保存必要权重、附代码/配置/输入规范/指标（MUST） | `model.pt` 旁有 `config.json`/`metrics.json`；权重不进 Git | 已满足（sidecar 方式） |
| A12 | 图表由脚本生成、保留脚本与原始指标（MUST） | `make_figures.py` 从 `results/summary/summary.csv` 生成图；原始指标保存在 `results/raw/**/metrics.json` | 已满足 |
| A13 | 结果命名含实验名/关键变量，禁止 result1/best_final2（MUST） | `results/raw/<model>/<dataset>/h<S>/[variant]/seed<S>/` | 已满足 |
| A14 | 断点保存模型/优化器/随机状态（SHOULD） | `train.save_train_state=true` 时保存 `train_state.pt` | 已满足 |
| A15 | 主表 3 个种子、均值±std（角色提示词 B1/B5 MUST） | 脚本就绪；**实际未运行 3 种子** | **缺失** |
| A16 | 至少 2 基线 + 主方法 + 1 项深入分析（B1 MUST） | DLinear/PatchTST/Transformer + iTransformer + 6 消融配置 + 回看扫描脚本；**未实际跑** | **缺失**（实验未跑） |

### 2.5 测试与质量（课程规范 §6）

| # | 要求（等级） | 仓库现状与证据 | 判定 |
| --- | --- | --- | --- |
| T1 | 关键数据/指标/规则单元测试（MUST） | `tests/unit/test_data.py`、`test_metrics.py`、`test_models.py`、`test_config.py`、`test_cli.py` | 已满足 |
| T2 | 至少一个不依赖大数据的端到端冒烟（MUST） | `tests/integration/test_smoke.py` 用合成数据跑完整 train/evaluate | 已满足 |
| T3 | 测试快速、独立、可重复（MUST） | 65 项约 8 秒；tmp_path/monkeypatch；无网络 | 已满足 |
| T4 | 覆盖率建议 ≥70%（SHOULD） | 实测 **89.08%**；CI `fail_under=80` | 已满足 |
| T5 | 测试命名描述行为（MUST） | 如 `test_etth1_window_counts_match_paper_table4_convention` | 已满足 |
| T6 | 提交前检查命令（MUST） | README §10、CI 均给出 ruff/pytest | 已满足 |
| T7 | 四项必测：输出形状、切分、指标口径、六种消融（角色提示词 B3 MUST） | `test_models.py` 覆盖形状/注意力/六消融；`test_data.py` 覆盖切分；`test_metrics.py` 手算 | 已满足 |

### 2.6 Git 与 GitHub 协作（课程规范 §7、§8.3）

| # | 要求（等级） | 仓库现状与证据 | 判定 |
| --- | --- | --- | --- |
| G1 | main 稳定，新工作走短分支 + PR（MUST） | 仅 `main` 一个分支；无 feature/fix 分支 | **不合规** |
| G2 | 提交小而清晰，Conventional Commits（MUST） | 仅 1 个提交 `feat: add itransformer-repro project files`；本次修改未提交 | **不合规** |
| G3 | PR 说明变更/验证/结果影响（MUST） | 仅有模板 [.github/PULL_REQUEST_TEMPLATE.md](../../.github/PULL_REQUEST_TEMPLATE.md)，无实际 PR | **缺失** |
| G4 | Issue 记录任务/验收条件/负责人（MUST） | 仅有模板 `.github/ISSUE_TEMPLATE/`，无实际 Issue | **缺失** |
| G5 | 代码审查记录（MUST） | 无 | **缺失** |
| G6 | 版本冻结标签 v1.0.0（MUST） | `git tag` 为空 | **缺失** |
| G7 | 团队贡献可追踪（MUST） | DECLARATION 姓名为“待填”；提交历史无法区分成员 | **不合规** |

### 2.7 CI（课程规范 §9）

| # | 要求（等级） | 仓库现状与证据 | 判定 |
| --- | --- | --- | --- |
| I1 | push main + PR 触发（MUST） | [ci.yml](../../.github/workflows/ci.yml) | 已满足 |
| I2 | 格式 + 静态检查 + 测试（MUST） | `ruff format --check`、`ruff check`、`pytest --cov` | 已满足 |
| I3 | 不在 CI 跑长训练，用小样本冒烟（MUST） | CI 只跑测试（合成数据冒烟在 pytest 内） | 已满足 |
| I4 | 最小权限（MUST） | `permissions: contents: read` | 已满足 |
| I5 | main 最新提交 CI 绿色（MUST） | **本地无法验证 GitHub Actions 实际运行**；仓库尚无本次提交 | **缺失**（证据） |

### 2.8 最终验收清单（课程规范 §11）

| 检查项 | 现状 | 判定 |
| --- | --- | --- |
| 仓库入口：README 完整、5 分钟内找到安装与最小运行 | README §3–§5 | 已满足 |
| 项目结构：src/tests/configs/docs 职责清楚 | 是 | 已满足 |
| 环境：干净环境可按文档安装 | 依赖声明 + editable 安装 | 已满足（未在全新机器实测） |
| 运行：最小示例/训练/评估可执行 | 真实 ETTh1 冒烟通过 | 已满足 |
| 质量：Ruff 通过、无调试残留 | 全绿 | 已满足 |
| 测试：pytest 通过、关键逻辑有测试 | 65 passed，89.08% | 已满足 |
| 复现：数据/种子/配置/环境/指标/结果可追溯 | 冒烟级别可追溯；主实验缺失 | 部分满足（主实验**缺失**） |
| 数据合规：来源/许可/划分/隐私 | data/README | 已满足 |
| 安全：无密钥/隐私/绝对路径/大文件 | 代码与配置干净；`environment.txt` 的 pip freeze 含 conda 构建路径（非个人路径） | 已满足 |
| Git 历史：提交清楚、分支/PR、贡献可追踪 | 单提交、无 PR、姓名待填 | **不合规** |
| CI：main 最新提交全绿 | 配置就绪，未实际运行 | **缺失**（证据） |
| 版本冻结：标签对应代码/文档/配置/结果一致 | 无标签；主结果未跑 | **缺失** |

### 2.9 三类清单汇总

**已满足（52 条 + §11 的 8 项）**：S1–S8、S11–S13、C1–C10、E1–E7、A1–A7、A9–A14、
T1–T7、I1–I4；§11 的入口/结构/环境/运行/质量/测试/数据/安全 8 项。

**缺失（9 项）**：S15（CHANGELOG，可选）、A8（主实验多种子）、A15（3 种子均值±std）、
A16（基线/消融实际运行）、G3（PR）、G4（Issue）、G5（审查）、G6（版本标签）、
I5（CI 运行证据）；另主结果表与图表未生成。

**不合规（4 项）**：S14（DECLARATION 姓名待填）、G1（无分支/PR）、G2（单提交且本次修改未提交）、
G7（团队贡献不可追踪）；S16 为与角色提示词 `dataset/` 的命名偏离（课程规范本身允许 `data/`）。

---

## 3. 论文保真核对（iTransformer, arXiv:2310.06625v4）

### 3.1 模型结构

| 论文要求（§3、式 1–3） | 仓库实现 | 判定 |
| --- | --- | --- |
| 逐变量嵌入 `Linear(T, D)` | `ITransformer.embedding = nn.Linear(seq_len, d_model)`，输入先转置 `(B,T,N)→(B,N,T)` | 一致 |
| 在 N 个变量 token 上做多头自注意力 | `VariateAttention`（`nn.MultiheadAttention`，token 数 = N） | 一致 |
| 每个 token 独立做时间维 FFN | `PositionwiseFFN`（Linear→ReLU→Dropout→Linear），块内再 dropout | 一致（官方用 Conv1d 实现，逐 token 等价） |
| post-norm：`norm(x + sublayer(x))` | `InvertedBlock.forward`：`norm1(x + dropout1(variate(x)))`，再 `norm2(...)` | 一致 |
| 激活函数 ReLU | `nn.ReLU()`（曾为 GELU，已修正） | 一致 |
| 不使用位置编码 | `use_pos_embed=False`；为消融保留可选参数 | 一致 |
| 输出 `Linear(D, S)` 后转置回 `(B,S,N)` | `projector` + `transpose(1,2)` | 一致 |
| 注意力权重可解释性 | `ITransformer.attention_map(inputs, layer_index)` 返回 `(B, heads, N, N)` | 已提供 |
| 每层参数是否共享 | 角色提示词 A2 写“每层共享同一 FFN 与注意力权重”，官方代码为**每层独立参数** | 按官方实现（`ModuleList` 独立层），冲突已记录 |

### 3.2 六种消融（论文表 3/6）

| 设计 | 配置 | `variate_module` | `temporal_module` | 状态 |
| --- | --- | --- | --- | --- |
| 完整 iTransformer | `configs/default.yaml` | attention | ffn | 可运行 |
| Replace Attention→Attention | `ablation_temporal_attn.yaml` | attention | attention | 可运行 |
| Replace FFN→Attention | `ablation_variate_ffn_temporal_attn.yaml` | ffn | attention | 可运行 |
| Replace FFN→FFN | `ablation_variate_ffn.yaml` | ffn | ffn | 可运行 |
| w/o Attention | `ablation_no_attention.yaml` | none | ffn | 可运行 |
| w/o FFN | `ablation_no_ffn.yaml` | attention | none | 可运行 |

`tests/unit/test_models.py` 对六种组合逐一断言前向形状 `(B,S,N)`；此前测试缺少
`(ffn, attention)` 且多出 `(none, none)`，已修正为与论文完全一致的六种。

### 3.3 数据与实验设置

| 项目 | 论文 | 仓库 | 判定 |
| --- | --- | --- | --- |
| 回看/预测 | T=96；S∈{96,192,336,720} | 配置默认 96/96；`run_matrix --horizons` 支持 4 档；`--seq-lens` 支持扫描 | 一致 |
| ETT 切分 | 12/4/4 个月（8640/2880/2880 行），只用前 20 个月 | `split_borders` 按 12/4/4 且只用前 14400 行 | 一致 |
| 其余数据集 | 70/10/20 时间顺序 | `split_rule="ratio"`，val/test 向前借 T 行 | 一致（官方同款边界） |
| 标准化 | 仅训练段拟合 | `build_datasets` 只在 train 段 `fit`，全量 `transform` | 一致 |
| 窗口数 | 表 4 ETTh1 `8545/2881/2881` | 严格样本数 `8449/2785/2785`；`paper_table4_windows` 同时给出 `8545/2881/2881` | 已解释：论文只数回看起点，差 `S` |
| 指标 | MSE/MAE，标准化空间 | `metrics.compute_all` 在标准化空间；额外给 RMSE/MAPE | 一致 |
| 优化器/损失 | Adam + MSE | `Adam` + `nn.MSELoss` | 一致 |
| 学习率 | 网格 {1e-3,5e-4,1e-4} | iTransformer 1e-3；PatchTST 1e-4（记录于 YAML） | 简化，已记录 |
| batch | 32 | 默认 16（显存折中）；DLinear/PatchTST/Transformer 32；smoke 8 | 偏离，已记录 |
| epoch/早停 | 固定 10 epoch，无早停 | 默认 10；`patience=0` 即关闭早停；无 lr 调度 | 一致 |
| 层数/维度/头数 | L∈{2,3,4}，D∈{256,512} | L=2、D=512、heads=8、d_ff=512、dropout=0.1 | 取论文默认档 |
| 种子 | 论文 5 个；课程最低 3 个 | 支持 2021/2022/2023；**未实际跑** | 缺失（实验） |
| 基线 | DLinear、PatchTST、vanilla Transformer | 均有实现，但 PatchTST/Transformer 为**简化基线** | 已声明，不声称逐行复刻 |
| 结果表 | 表 1/4/5/6 | **未运行**，README 标“待运行” | 缺失（如实标注） |
| 测试标签可见性 | 公开基准标签可见 | 测试集只评估一次、不参与调参/早停；已在 `docs/experiments.md` §4 声明潜在偏差 | 一致 |

### 3.4 论文 ↔ 代码/提示词差异表（覆盖角色提示词 A7 全部条目）

| # | 差异点 | 处理 |
| --- | --- | --- |
| 1 | PEMS 步长表 1 与表 4 自相矛盾 | 不做 PEMS；记录于 `docs/experiments.md` §5 |
| 2 | 论文 batch 32，官方脚本常用 8/16 | 默认 16，基线 32，smoke 8；写入 `config.json` |
| 3 | 学习率网格 vs 脚本逐数据集固定 | 不做网格；固定值写入 YAML |
| 4 | 固定 10 epoch vs 官方 `--patience/--lradj` | `patience=0` 关闭早停；不实现 lr 调度 |
| 5 | 指标是否反归一化 | 统一标准化空间，`pred.npy` 亦为标准化值 |
| 6 | 表 3/6 的 Weather MAE 0.278 vs 0.279 | 原样记录，不取平均；`docs/experiments.md` §5 |
| 7 | ETT 窗口计数 8545/2881/2881 vs 严格样本数 | 两种口径同时输出，不改协议 |
| 8 | 代码公开时点/commit | `config.json` 记录 `git_commit` 与 `git_dirty`；本次审计代码尚未提交（见 §6） |
| 9 | 角色提示词 A2 “每层共享权重” vs 官方独立层 | 按官方实现，记录冲突 |

### 3.5 发现的实现错误（已修复）

1. **激活函数错误**：`VariateMlp`/`PositionwiseFFN` 原用 GELU，官方为 ReLU → 已改。
2. **FFN dropout 重复**：原 FFN 内部末尾 dropout 与块内 `dropout2` 叠加，官方只有
   激活后与子层输出两处 → 已移除 FFN 内部末尾 dropout，并更新架构文档。
3. **训练 DataLoader 未 drop_last**：官方训练集 `drop_last=True` → 已改为训练集
   `drop_last=True`（小数据集自动保护）。
4. **消融测试错误**：原测试含 `(none,none)`、缺 `(ffn,attention)` → 已改为论文六种。
5. **注意力权重不可导出**：原 `forward` 丢弃权重 → 新增 `attention_map()` 并测试形状与
   `softmax` 行和。
6. **日志器复用旧文件**：同进程多次运行时 `get_logger` 复用首个 FileHandler → 已修复。
7. **CUDA 峰值显存未重置**：多 run 同进程统计错误 → 每 run 重置。
8. **`run_matrix` 给基线套用主模型超参**：已加 per-model preset（PatchTST d_model=128/lr=1e-4，
   基线 batch=32）。
9. **`evaluate` 不验证重算一致性**：已加 `mse_delta/mae_delta/matches_original`。
10. **缺少 `true.npy`**：预测数组无法复核 → 已保存并与 `pred.npy` 同形状。
11. **`git_commit` 依赖当前工作目录** → 已改为从项目根目录解析，并记录 `git_dirty`。
12. **确定性不足**：未设 `CUBLAS_WORKSPACE_CONFIG` → 已补。

---

## 4. 本次补齐内容（文件级）

### 4.1 环境与依赖

- `pyproject.toml`：保留 `>=3.9,<3.13` 与 ruff/pytest/coverage 配置；新增
  `[tool.coverage.report] fail_under=80`。
- `requirements.txt`：运行时与开发依赖统一声明（版本下限与审计环境一致），末尾
  `-e .` 以可编辑模式安装本包。
- `environment.txt`（仓库根目录）：真实环境记录（含 pip freeze、CPU、CUDA 可用性）。
- `README.md` §3、`REPRODUCTION.md` §1：安装、版本偏差、环境记录命令。

### 4.2 代码与配置

- `src/itransformer_repro/models.py`：ReLU、post-norm/dropout 对齐官方、`attention_map`、
  六消融开关。
- `src/itransformer_repro/config.py`：`runs_dir=results/raw`；`save_train_state`；层级
  run 目录 `<model>/<dataset>/h<S>/[variant]/seed<S>`。
- `src/itransformer_repro/train.py`：`drop_last`、峰值显存重置、`git_dirty`、CPU 权重快照、
  `true.npy`、时间口径拆分、可选 `train_state.pt`。
- `src/itransformer_repro/data.py`：官方行数注册、SHA256、`expected_rows`、
  `paper_table4_windows`、strict 校验。
- `src/itransformer_repro/evaluate.py`：重算并比对原指标。
- `src/itransformer_repro/utils.py`：CUBLAS 设置、日志器修复、`project_root/git_dirty`、
  `platform`/`cudnn` 环境字段。
- `src/itransformer_repro/cli.py`：`check-data --strict/--out/--set`；`evaluate` 错误处理。
- `configs/`：新增 `ablation_variate_ffn_temporal_attn.yaml`；全部 `runs_dir` 改为
  `results/raw`；修正消融注释。

### 4.3 实验脚本与结果规范

- `scripts/aggregate.py`：递归发现、`results/summary/{summary.csv,main_table.md}`、均值/std、
  耗时/显存、种子数 WARN。
- `scripts/run_matrix.py`：per-model preset、`--seq-lens`、`--config-dir`、`--no-model-presets`。
- `scripts/make_figures.py`：误差条、主协议过滤、`lookback_curve.png`。
- `scripts/compare_paper.py`：覆盖度列、4 档齐全才 PASS。
- 新增 `scripts/analyze_errors.py`（逐 horizon/变量误差 + `worst_cases.csv`）。
- 新增 `scripts/env_report.py`（环境快照）。
- `results/README.md`、`results/summary/.gitkeep`、`results/figures/.gitkeep`。

### 4.4 文档与仓库模板

- 重写 `README.md`（问题、方法、环境、数据、最小示例、命令、待运行结果、结构、测试、
  确定性、团队/限制/许可）。
- 新增 `REPRODUCTION.md`、`DECLARATION.md`、`CONTRIBUTING.md`、`CITATION.cff`。
- 新增 `.github/PULL_REQUEST_TEMPLATE.md`、`.github/ISSUE_TEMPLATE/{bug_report,experiment_task}.md`。
- 更新 `docs/architecture.md`、`docs/experiments.md`、`data/README.md`。
- CI：最小权限、并发取消、20 分钟超时、覆盖率阈值。

### 4.5 测试

- 修正 `tests/unit/test_models.py` 六消融与 `attention_map` 测试。
- 扩充 `tests/unit/test_data.py`：论文表 4 口径、SHA/行数、strict。
- 扩充 `tests/integration/test_smoke.py`：`true.npy`、新指标字段、`git_dirty`、
  `matches_original`。
- 新增 `tests/unit/test_cli.py`：子命令解析、覆盖合并、分发与错误码。

---

## 5. 验证证据

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| 静态检查 | `ruff check .` | All checks passed |
| 格式 | `ruff format --check .` | 35 files already formatted |
| 测试 | `pytest -q` | **65 passed**（约 8 s） |
| 覆盖率 | `pytest --cov=itransformer_repro` | **89.08%**，`fail_under=80` 达标 |
| 数据 | `check-data --strict`（ETTh1） | rows=17420、7 变量、SHA 匹配、passed=true |
| 冒烟 | `train --config configs/smoke.yaml --device cpu` | test MSE 0.428403 / MAE 0.450213，1 epoch，11.31 s |
| 复评 | `evaluate --run-dir .../smoke/seed2021` | `matches_original=True` |
| 错误分析 | `analyze_errors.py ...` | 生成 `error_analysis.json`、`worst_cases.csv` |
| 汇总/图 | `aggregate.py`、`make_figures.py` | `summary.csv`、`main_table.md`、2 张 PNG（审计结束后已删除这些本地产物） |
| 论文对照 | `compare_paper.py` | ETTh1 h96：0.4284/0.4502 vs 论文 ETT 平均 0.383/0.399，`WARN`（1/4 档，仅量级参照） |
| 矩阵 dry-run | `run_matrix.py ... --dry-run` | 基线 preset 生效（PatchTST d_model=128/lr=1e-4/batch=32） |

冒烟产物当时位于 `results/raw/itransformer/ETTh1/h96/smoke/seed2021/`（本地目录，公开
仓库不入库），**只证明链路可运行，不进入主结果表**。审计结束后这些本地冒烟产物
（含临时示例目录）已全部删除，`results/` 只保留目录约定与三个子目录的 `.gitkeep`；
正式提交前必须用主实验重新生成 `results/summary/` 与 `results/figures/`。

原始输出摘录见 [verification-log.md](verification-log.md)。

---

## 6. 剩余阻塞与需用户决策

1. **算力阻塞（无法由代码补齐）**：当前机器 CPU-only，完整
   `4 数据集 × 4 步长 × 4 模型 × 3 种子 = 192+ 次训练` 不可行。因此主结果表、
   3 种子均值±std、消融图、回看曲线、与论文的系统对照均保持“待运行”。
   需要 GPU（建议 ≥16 GB）或用户批准“缩减版真实实验”（如 ETTh1 × {96,720} × 2 模型 × 3 种子）。
2. **数据阻塞**：仅 ETTh1 已下载并通过校验；ETTh2/Weather/ECL 需按 data/README 下载。
3. **团队信息（需用户填写）**：`DECLARATION.md` 与 README 的姓名/学号为“待填”，
   否则“团队分工与贡献”不合规。
4. **Git/GitHub 过程证据（需用户执行）**：仓库只有 1 个提交、无分支/PR/Issue/tag。
   本次修改未提交（遵循“不主动 commit”）。建议按 `CONTRIBUTING.md` 拆分为
   `feat(model): ...`、`fix(train): ...`、`docs: ...`、`test: ...` 等提交，经 PR 合并，
   最后打 `v1.0.0`。
5. **目录命名冲突**：角色提示词 B3 要求 `dataset/`，课程规范 §2.1 用 `data/`；本仓库采用
   `data/raw/`（与课程规范一致）。若助教按角色提示词检查，需要改名或加说明。
6. **Python 版本**：课程默认 3.10/3.11，本地审计在 3.9.13 完成；已在 README/REPRODUCTION
   说明，正式提交建议在 3.10/3.11 重跑并更新 `environment.txt`。
7. **CI 实际状态未知**：工作流配置正确，但未在 GitHub 上运行；需要推送后确认绿灯。

---

## 7. 复现命令（PowerShell）

```powershell
cd d:\itransformer-repro

# 1) 安装
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 2) 环境记录
python scripts/env_report.py --out environment.txt

# 3) 数据校验（先按 data/README 放置 ETTh1.csv）
$env:PYTHONPATH='src'
python -m itransformer_repro.cli check-data --config configs/smoke.yaml --strict

# 4) 冒烟（非实验证据）
python -m itransformer_repro.cli train --config configs/smoke.yaml --device cpu --force
python -m itransformer_repro.cli evaluate --run-dir results/raw/itransformer/ETTh1/h96/smoke/seed2021

# 5) 质量门禁
python -m ruff format --check .
python -m ruff check .
python -m pytest -q --cov=itransformer_repro --cov-report=term-missing

# 6) 主实验（需 GPU 与全部数据）
python scripts/run_matrix.py `
  --datasets ETTh1 ETTh2 Weather ECL `
  --models itransformer dlinear patchtst transformer `
  --horizons 96 192 336 720 `
  --seeds 2021 2022 2023
python scripts/aggregate.py
python scripts/make_figures.py
python scripts/compare_paper.py
```

---

## 8. 审计材料索引

| 文件 | 内容 |
| --- | --- |
| [environment.txt](../../environment.txt)（仓库根目录） | 审计环境快照（pip freeze、CPU/CUDA、平台） |
| [audit-report.md](audit-report.md) | 本报告 |
| [verification-log.md](verification-log.md) | 本次全部命令与原始输出 |

课程规范与论文原文属于外部输入，未随仓库分发；如需复核条款，请使用课程下发的
PDF 与 arXiv:2310.06625v4。

---

## 9. 代码修订记录（2026-11-25）

本轮只修代码与文档，未运行的实验仍保持"待运行"状态。修订项：

| 项 | 内容 | 影响文件 |
| --- | --- | --- |
| 对照隔离 | 论文对照只保留主方法 iTransformer；基线改为"与同数据集 iTransformer 互比"的表 B，避免基线被误比论文值 | `scripts/compare_paper.py` |
| 参考值修正 | ETTh2 的 4 档平均 MAE 由 0.399 改为论文值 0.407（MSE 0.383 不变） | `scripts/compare_paper.py` |
| 判定口径 | 默认容差由 20% 改为作业要求的 10%，并要求 4 档预测长度齐全才判 PASS | `scripts/compare_paper.py` |
| 可解释性 | 新增变量注意力导出脚本：`attention.npy`、`attention_mean.npy`、热图与 top-k 变量对 JSON | `scripts/attention_maps.py` |
| 批处理 | 新增消融批量运行入口（6 个设计 × 多种子，自动跳过已完成运行） | `scripts/run_ablations.py` |
| 计时口径 | `train_time_sec` 只统计纯训练，新增 `val_time_sec`；`epoch_time_sec` 仍为训练 + 验证 | `src/itransformer_repro/train.py` |
| 测试 | 新增默认语义等价性测试（与 post-norm 参考实现逐权重对照）、对照脚本测试、脚本层测试；测试数 65 → 79 | `tests/unit/test_fidelity.py`、`test_compare_paper.py`、`test_scripts.py` |
| 文档 | README / REPRODUCTION / experiments 同步更新数据状态、脚本用法与耗时口径 | `README.md`、`REPRODUCTION.md`、`docs/experiments.md` |
