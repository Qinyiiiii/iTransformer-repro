"""在不重新训练的前提下复评一次已保存的运行。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from itransformer_repro.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from itransformer_repro.data import build_datasets
from itransformer_repro.models import build_model
from itransformer_repro.train import evaluate_model, make_loader
from itransformer_repro.utils import (
    load_json,
    resolve_device,
    save_json,
    seed_everything,
)


def config_from_dict(payload: dict[str, Any]) -> ExperimentConfig:
    """从落盘的 ``config.json`` 还原配置对象。

    Raises:
        ValueError: 缺少必需字段。
    """
    try:
        return ExperimentConfig(
            experiment=str(payload["experiment"]),
            seed=int(payload["seed"]),
            device=str(payload.get("device", "auto")),
            runs_dir=str(payload.get("runs_dir", "runs")),
            data=DataConfig(**payload["data"]),
            model=ModelConfig(**payload["model"]),
            train=TrainConfig(**payload["train"]),
        )
    except KeyError as exc:
        raise ValueError(f"保存的配置缺少字段：{exc}") from exc


def evaluate_run(run_dir: Path) -> dict[str, Any]:
    """加载 ``run_dir`` 中的权重并重算测试集指标。

    Args:
        run_dir: ``itransformer-repro train`` 产生的运行目录。

    Returns:
        重算后的指标，同时写入 ``metrics_eval.json``。

    Raises:
        FileNotFoundError: 配置或权重文件缺失。
    """
    config_path = run_dir / "config.json"
    weights_path = run_dir / "model.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"缺少配置：{config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"缺少权重：{weights_path}")

    config = config_from_dict(load_json(config_path))
    seed_everything(config.seed)
    device = resolve_device(config.device)
    bundle = build_datasets(config.data)
    model = build_model(
        config.model,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        n_vars=bundle.n_vars,
    ).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    loader = make_loader(bundle.test, config.train.batch_size, shuffle=False)
    metrics, _, _ = evaluate_model(model, loader, device)
    payload = {
        **{key: round(value, 6) for key, value in metrics.items()},
        "recomputed": True,
        "device": str(device),
    }
    save_json(run_dir / "metrics_eval.json", payload)
    return payload
