# 贡献指南

本仓库是课程复现项目，仍按真实开源仓库的流程协作：小步提交、分支开发、PR 评审、
CI 通过后合并。

## 1. 开发环境

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
python -m pip install -U pip
python -m pip install -r requirements.txt
ruff format .
ruff check .
pytest -q
```

提交前必须四条全部通过。CI 使用 Python 3.11，在 PR 上运行
`ruff format --check`、`ruff check` 与 `pytest --cov`。

## 2. 分支与提交

- 分支命名：`feat/<topic>`、`fix/<topic>`、`docs/<topic>`、`exp/<topic>`。
- 一个 PR 只做一件事；不要把数据文件、权重、格式化全仓库混进功能 PR。
- 提交信息使用 Conventional Commits：
  - `feat: add lookback sweep`
  - `fix: isolate per-run log file`
  - `docs: record ETT window-count convention`
  - `exp: add ETTh1 h96 seed 2021 metrics`
- 每个 PR 至少一位其他成员评审；评审意见解决后再合并，禁止直接 push `main`。

## 3. Issue

- 功能/实验任务先用 Issue 描述目标、验收标准与负责人。
- Bug Issue 附复现命令、完整错误、环境（`environment.txt` 摘要）与期望行为。
- 实验结果 Issue 附 run 目录路径与 `config.json` 的 `git_commit`。

## 4. 代码约定

- 公共函数与类必须有类型注解和简洁 docstring；仅在逻辑不直观处写注释。
- 配置是参数唯一来源：新增超参先加到 `config.py` 的 dataclass，再在 YAML 中给
  默认值；不要在训练函数里硬编码。
- 指标改动必须同步更新 `tests/unit/test_metrics.py` 的手算用例。
- 数据切分改动必须同步更新 `tests/unit/test_data.py` 与
  `docs/experiments.md` 的差异表。
- 新模型必须实现 `(B, T, N) -> (B, S, N)`，在 `MODEL_NAMES` 注册，并附一份
  可冒烟的 YAML。
- 不提交 `data/raw/`、`*.npy`、`*.pt`、虚拟环境、缓存与密钥。

## 5. 实验与结果

- 主结果至少 3 个种子；单种子只能作为冒烟或调试，不得写入主表。
- 原始结果写入本地 `results/raw/`（整目录被 `.gitignore` 排除，不进入公开仓库）；
  汇总与图只能由脚本生成：`aggregate.py` → `summary.csv/main_table.md`，
  `make_figures.py` → `figures/`。公开仓库只保留目录约定与三个子目录的
  `.gitkeep`，不附带任何示例或冒烟产物。
- 每个结果必须能指向唯一 `config.json`；配置变更后旧结果不覆盖，改
  `experiment` 或删除后重跑。
- 未跑完的实验在 README/报告中标注“待运行”并写清数据或算力阻塞，不填占位数字。

## 6. PR 检查清单

- [ ] `ruff format --check .` 通过
- [ ] `ruff check .` 通过
- [ ] `pytest -q` 通过
- [ ] 新配置已用 `--dry-run` 或 1 epoch 冒烟
- [ ] 文档/差异表已同步（如涉及数据、指标、模型结构）
- [ ] 未提交数据、权重、密钥与大体积数组
- [ ] PR 描述包含改动动机、验证命令与结果路径
