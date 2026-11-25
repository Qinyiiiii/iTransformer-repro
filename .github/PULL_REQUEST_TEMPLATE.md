## 改动概述

<!-- 这个 PR 做了什么，为什么需要它。关联 Issue：#<id> -->

## 改动类型

- [ ] 新功能 / 模型
- [ ] Bug 修复
- [ ] 数据 / 指标 / 切分
- [ ] 实验脚本 / 结果
- [ ] 文档
- [ ] 重构 / 测试

## 验证

```bash
ruff format --check .
ruff check .
pytest -q
# 如涉及训练：
# itransformer-repro train --config configs/smoke.yaml --device cpu
```

- 验证结果（粘贴关键输出或截图）：
- 新增/修改的结果路径（如 `results/raw/...`）：

## 规范检查

- [ ] 未提交 `data/raw/`、`*.npy`、`*.pt`、密钥或虚拟环境
- [ ] 新超参已写入 `config.py` 与 YAML，未硬编码
- [ ] 数据/指标/模型结构变更已同步测试与 `docs/experiments.md`
- [ ] 主结果未使用单种子冒充稳定结论
- [ ] 未运行完的实验已标注“待运行”与阻塞原因
- [ ] 至少一位成员已评审

## 截图 / 备注

<!-- 可选 -->
