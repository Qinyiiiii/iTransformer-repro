# 结果目录规范

`results/` 只保留课程约定的三层结构（`raw / summary / figures`）与目录说明；
训练、汇总、绘图产生的文件都是本地生成物，默认不进入公开仓库。

## 公开仓库提交策略

| 路径 | 是否入库 | 说明 |
| --- | --- | --- |
| `results/README.md`、`results/*/.gitkeep` | 是 | 目录约定 |
| `results/raw/**` | 否 | 原始运行目录：配置、指标、日志、预测、权重 |
| `results/summary/*` | 否（默认） | 由 `scripts/aggregate.py` 生成；入库前需人工确认 |
| `results/figures/*` | 否（默认） | 由 `scripts/make_figures.py` 生成；入库前需人工确认 |
| `data/raw/**` | 否 | 原始数据集，仅保留 `data/README.md` 的下载与校验说明 |

`.gitignore` 已覆盖上述路径，三个子目录各保留一个 `.gitkeep` 维持结构。若课程
提交确实要求附带结果，请把 `results/raw/` 单独打包提交，或在确认内容后显式
`git add -f`，不要修改默认忽略规则。

## 目录

```text
results/
├── README.md
├── raw/                         # 训练产物（仅 .gitkeep 入库）
│   └── <model>/<dataset>/h<S>/[<variant>/]seed<seed>/
│       ├── config.json          # 完整配置 + git commit + 环境
│       ├── metrics.json         # 测试指标、参数量、耗时、峰值显存
│       ├── train.log            # 训练日志
│       ├── pred.npy             # (n_windows, S, N) 标准化空间预测
│       ├── true.npy             # (n_windows, S, N) 标准化空间真值
│       ├── model.pt             # 最佳验证 MSE 权重
│       ├── metrics_eval.json    # 可选：evaluate 重评结果
│       ├── error_analysis.json  # 可选：analyze_errors.py 输出
│       ├── worst_cases.csv      # 可选：analyze_errors.py 输出
│       └── train_state.pt       # 可选：模型+优化器+epoch+随机状态
├── summary/                     # 由 scripts/aggregate.py 生成
│   ├── .gitkeep
│   ├── summary.csv
│   └── main_table.md
└── figures/                     # 由 scripts/make_figures.py 生成
    ├── .gitkeep
    ├── mse_by_horizon.png
    ├── model_comparison.png
    ├── ablation.png             # 有消融结果时生成
    └── lookback_curve.png       # 有多个 seq_len 时生成
```

## 约定

- `raw/` 是本地唯一事实来源；`summary/` 与 `figures/` 只能由
  `scripts/aggregate.py` 和 `scripts/make_figures.py` 从 `raw/` 生成，不手工填写。
- `itransformer-repro train` 写入
  `raw/<model>/<dataset>/h<S>/[<variant>/]seed<seed>/`，默认根目录为
  `results/raw`（由配置的 `runs_dir` 决定）；`evaluate` 与 `analyze_errors.py`
  的补充产物写回同一个 run 目录。
- 主结果至少 3 个种子；少于 3 个种子时 `aggregate.py` 会打印 WARN，报告里不得
  把单种子结果写成稳定结论。
- 未运行的实验保持缺失状态，不创建占位数字；`README.md` 的“主结果”表必须标注
  “待运行”并说明数据/算力阻塞。
- 仓库不附带任何示例结果或冒烟产物；需要核验链路时按 `REPRODUCTION.md` 在本地
  运行，产物只落在上述三个目录，不会被提交。
