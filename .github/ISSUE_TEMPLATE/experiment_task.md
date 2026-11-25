---
name: Experiment task
about: 登记一次主结果、消融或回看扫描任务
title: "[Exp] "
labels: experiment
assignees: ""
---

## 目标

<!-- 例如：ETTh1 h96 iTransformer 3 seeds 主结果 -->

## 配置与命令

```bash

```

## 验收标准

- [ ] 3 个种子全部完成
- [ ] `results/raw/**/metrics.json`、`train.log`、`config.json` 齐全
- [ ] `aggregate.py` 生成对应 summary 行
- [ ] 如进入报告，已附 `git_commit` 与 `environment` 摘要

## 阻塞 / 备注

<!-- 数据、GPU、依赖等问题 -->
