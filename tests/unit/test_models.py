"""模型结构与输出形状的单元测试。"""

from __future__ import annotations

import pytest
import torch

from itransformer_repro.config import ModelConfig
from itransformer_repro.models import build_model

SEQ_LEN = 24
PRED_LEN = 12
N_VARS = 5
BATCH = 4


def make_config(**overrides: object) -> ModelConfig:
    """构造小尺寸模型配置，保证测试快速运行。"""
    base: dict[str, object] = {
        "d_model": 16,
        "d_ff": 32,
        "n_heads": 2,
        "e_layers": 1,
        "dropout": 0.0,
    }
    base.update(overrides)
    return ModelConfig(**base)


@pytest.mark.parametrize(
    ("variate_module", "temporal_module"),
    [
        ("attention", "ffn"),
        ("ffn", "ffn"),
        ("attention", "attention"),
        ("none", "ffn"),
        ("attention", "none"),
        ("none", "none"),
    ],
)
def test_itransformer_variants_return_expected_shape(
    variate_module: str, temporal_module: str
) -> None:
    config = make_config(variate_module=variate_module, temporal_module=temporal_module)
    model = build_model(config, SEQ_LEN, PRED_LEN, N_VARS)
    output = model(torch.randn(BATCH, SEQ_LEN, N_VARS))
    assert tuple(output.shape) == (BATCH, PRED_LEN, N_VARS)
    assert torch.isfinite(output).all()


def test_itransformer_accepts_optional_position_embedding() -> None:
    config = make_config(use_pos_embed=True)
    model = build_model(config, SEQ_LEN, PRED_LEN, N_VARS)
    output = model(torch.randn(BATCH, SEQ_LEN, N_VARS))
    assert tuple(output.shape) == (BATCH, PRED_LEN, N_VARS)


def test_itransformer_without_layernorm_still_runs() -> None:
    config = make_config(use_layernorm=False)
    model = build_model(config, SEQ_LEN, PRED_LEN, N_VARS)
    output = model(torch.randn(BATCH, SEQ_LEN, N_VARS))
    assert tuple(output.shape) == (BATCH, PRED_LEN, N_VARS)


@pytest.mark.parametrize("name", ["dlinear", "patchtst", "transformer"])
def test_baselines_return_expected_shape(name: str) -> None:
    config = make_config(name=name, kernel_size=3, patch_len=8, patch_stride=4)
    model = build_model(config, SEQ_LEN, PRED_LEN, N_VARS)
    output = model(torch.randn(BATCH, SEQ_LEN, N_VARS))
    assert tuple(output.shape) == (BATCH, PRED_LEN, N_VARS)
    assert torch.isfinite(output).all()


def test_build_model_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="未知模型"):
        build_model(make_config(name="mlp"), SEQ_LEN, PRED_LEN, N_VARS)


def test_patchtst_rejects_patch_longer_than_input() -> None:
    with pytest.raises(ValueError, match="patch_len"):
        build_model(
            make_config(name="patchtst", patch_len=64), SEQ_LEN, PRED_LEN, N_VARS
        )


def test_dlinear_requires_odd_kernel_size() -> None:
    with pytest.raises(ValueError, match="奇数"):
        build_model(
            make_config(name="dlinear", kernel_size=4), SEQ_LEN, PRED_LEN, N_VARS
        )
