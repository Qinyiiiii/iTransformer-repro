"""iTransformer 与"倒置 + post-norm 编码器"参考实现的一致性测试。

作业要求（提示词 A4）允许在不包装官方入口时提供替代证据：逐层权重对照 +
同一输入下输出一致。本测试把项目权重逐层拷入 ``nn.TransformerEncoderLayer``
（post-norm、ReLU、batch_first）搭成的参考实现，断言两者输出差异在数值噪声
范围内。这样任何改动破坏默认语义（嵌入、注意力、FFN、归一化顺序、投影）都会
立刻失败。

参考实现关闭 dropout；项目在 ``dropout=0`` 时也退化为同一计算图。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from itransformer_repro.config import ModelConfig, load_config
from itransformer_repro.models import ITransformer, build_model

SEQ_LEN = 24
PRED_LEN = 12
N_VARS = 6
D_MODEL = 16
D_FF = 32
N_HEADS = 2
E_LAYERS = 2
BATCH = 3
ATOL = 1e-5


class ReferenceInvertedEncoder(nn.Module):
    """直接使用 ``nn.TransformerEncoderLayer``（post-norm）的参考实现。"""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Linear(SEQ_LEN, D_MODEL)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    D_MODEL,
                    N_HEADS,
                    D_FF,
                    dropout=0.0,
                    activation="relu",
                    batch_first=True,
                    norm_first=False,
                )
                for _ in range(E_LAYERS)
            ]
        )
        self.norm = nn.LayerNorm(D_MODEL)
        self.projector = nn.Linear(D_MODEL, PRED_LEN)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.embedding(inputs.transpose(1, 2))
        for layer in self.layers:
            tokens = layer(tokens)
        return self.projector(self.norm(tokens)).transpose(1, 2)


def build_project_model() -> ITransformer:
    """构造与参考实现同尺寸、dropout 为 0 的项目模型。"""
    config = ModelConfig(
        d_model=D_MODEL,
        d_ff=D_FF,
        n_heads=N_HEADS,
        e_layers=E_LAYERS,
        dropout=0.0,
    )
    model = build_model(config, seq_len=SEQ_LEN, pred_len=PRED_LEN, n_vars=N_VARS)
    assert isinstance(model, ITransformer)
    return model.eval()


def copy_weights(model: ITransformer, reference: ReferenceInvertedEncoder) -> None:
    """把项目模型权重逐层拷入参考实现（结构一一对应）。"""
    with torch.no_grad():
        reference.embedding.weight.copy_(model.embedding.weight)
        reference.embedding.bias.copy_(model.embedding.bias)
        for index, block in enumerate(model.blocks):
            layer = reference.layers[index]
            attention = block.variate.attention
            layer.self_attn.in_proj_weight.copy_(attention.in_proj_weight)
            layer.self_attn.in_proj_bias.copy_(attention.in_proj_bias)
            layer.self_attn.out_proj.weight.copy_(attention.out_proj.weight)
            layer.self_attn.out_proj.bias.copy_(attention.out_proj.bias)
            layer.linear1.weight.copy_(block.temporal.net[0].weight)
            layer.linear1.bias.copy_(block.temporal.net[0].bias)
            layer.linear2.weight.copy_(block.temporal.net[3].weight)
            layer.linear2.bias.copy_(block.temporal.net[3].bias)
            layer.norm1.weight.copy_(block.norm1.weight)
            layer.norm1.bias.copy_(block.norm1.bias)
            layer.norm2.weight.copy_(block.norm2.weight)
            layer.norm2.bias.copy_(block.norm2.bias)
        reference.norm.weight.copy_(model.norm.weight)
        reference.norm.bias.copy_(model.norm.bias)
        reference.projector.weight.copy_(model.projector.weight)
        reference.projector.bias.copy_(model.projector.bias)


def test_default_block_matches_reference_postnorm_encoder() -> None:
    """默认配置下项目模型与参考实现输出一致（差异 < 1e-5）。"""
    torch.manual_seed(0)
    model = build_project_model()
    reference = ReferenceInvertedEncoder().eval()
    copy_weights(model, reference)

    inputs = torch.randn(BATCH, SEQ_LEN, N_VARS)
    with torch.no_grad():
        ours = model(inputs)
        expected = reference(inputs)

    assert tuple(ours.shape) == (BATCH, PRED_LEN, N_VARS)
    assert torch.allclose(ours, expected, atol=ATOL, rtol=0.0)


def test_attention_map_equals_reference_attention_weights() -> None:
    """导出的变量注意力权重与参考实现同层权重一致。"""
    torch.manual_seed(0)
    model = build_project_model()
    reference = ReferenceInvertedEncoder().eval()
    copy_weights(model, reference)

    inputs = torch.randn(BATCH, SEQ_LEN, N_VARS)
    with torch.no_grad():
        tokens = reference.embedding(inputs.transpose(1, 2))
        for layer in reference.layers[:-1]:
            tokens = layer(tokens)
        _, expected = reference.layers[-1].self_attn(
            tokens, tokens, tokens, need_weights=True, average_attn_weights=False
        )
        ours = model.attention_map(inputs)

    assert tuple(ours.shape) == (BATCH, N_HEADS, N_VARS, N_VARS)
    assert torch.allclose(ours, expected, atol=ATOL, rtol=0.0)


def test_default_config_keeps_paper_defaults() -> None:
    """仓库默认配置必须保持论文主方法语义（A2/A10 的默认要求）。"""
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "default.yaml").model
    assert config.name == "itransformer"
    assert config.variate_module == "attention"
    assert config.temporal_module == "ffn"
    assert config.use_layernorm is True
    assert config.use_pos_embed is False
    assert config.dropout == pytest.approx(0.1)
