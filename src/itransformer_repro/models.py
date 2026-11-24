"""模型：iTransformer、其消融变体以及三个对照模型。

所有预测模块都把 ``(batch, seq_len, n_vars)`` 的输入映射为
``(batch, pred_len, n_vars)`` 的输出。iTransformer 遵循官方实现：先转置为
``(batch, n_vars, seq_len)``，把每个变量的整段回看序列嵌入成一个 token，
再在变量维度上做自注意力，最后把每个 token 投影到预测长度；不使用位置编码。
"""

from __future__ import annotations

import torch
from torch import nn

from itransformer_repro.config import ModelConfig


class VariateAttention(nn.Module):
    """变量维多头自注意力：在 N 个维度为 D 的 token 之间做注意力。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Args:
            tokens: 形状 ``(B, N, D)`` 的变量 token。

        Returns:
            注意力输出，形状不变。
        """
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        return attended


class VariateMlp(nn.Module):
    """变量维的 token-mixing MLP，用作"去掉注意力"的替代混合模块。"""

    def __init__(self, n_vars: int, expansion: int = 2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_vars, n_vars * expansion)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(n_vars * expansion, n_vars)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Args:
            tokens: 形状 ``(B, N, D)`` 的变量 token。

        Returns:
            在 N 维上混合后的 token，形状不变。
        """
        hidden = tokens.transpose(1, 2)
        hidden = self.fc2(self.activation(self.fc1(hidden)))
        return hidden.transpose(1, 2)


class PositionwiseFFN(nn.Module):
    """标准的逐 token 前馈网络，作用在 D（时间表征）维上。"""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """对每个 token 独立地施加同一个 MLP。"""
        return self.net(tokens)


class TemporalAttention(nn.Module):
    """消融用：把逐 token 的 FFN 换成时间维注意力。

    倒置结构里每个 token 已经是"一个变量的完整时间表征"，不存在时间步 token。
    这里把 D 维表征切成 ``n_groups`` 组当作时间 token 做注意力，是"时间维用
    注意力替代 FFN"最接近原文、且保持形状不变的实现方式；分组数是本项目的
    实现细节（见 docs/architecture.md）。

    Raises:
        ValueError: ``d_model`` 不能被 ``n_groups`` 整除。
    """

    def __init__(
        self,
        d_model: int,
        n_groups: int = 8,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_groups != 0:
            raise ValueError(f"d_model={d_model} 必须能被 n_groups={n_groups} 整除")
        self.n_groups = n_groups
        group_dim = d_model // n_groups
        heads = n_heads if group_dim % n_heads == 0 else 1
        self.attention = nn.MultiheadAttention(
            group_dim, heads, dropout=dropout, batch_first=True
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Args:
            tokens: 形状 ``(B, N, D)`` 的变量 token。

        Returns:
            时间分组被注意力混合后的 token，形状不变。
        """
        batch, n_vars, d_model = tokens.shape
        grouped = tokens.reshape(
            batch * n_vars, self.n_groups, d_model // self.n_groups
        )
        attended, _ = self.attention(grouped, grouped, grouped, need_weights=False)
        return attended.reshape(batch, n_vars, d_model)


class InvertedBlock(nn.Module):
    """倒置编码块：先变量维混合、再时间维混合，各自残差后归一化。

    归一化顺序与官方 ``EncoderLayer`` 一致（post-norm）：
    ``x = LayerNorm(x + Variate(x))``，随后 ``x = LayerNorm(x + Temporal(x))``。
    """

    def __init__(
        self,
        n_vars: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        variate_module: str,
        temporal_module: str,
        use_layernorm: bool,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.variate = _build_variate(n_vars, d_model, n_heads, dropout, variate_module)
        self.temporal = _build_temporal(
            d_model, d_ff, n_heads, dropout, temporal_module
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """先变量混合后时间混合，每个子层为"残差 + 归一化"。"""
        if self.variate is not None:
            tokens = self.norm1(tokens + self.variate(tokens))
        if self.temporal is not None:
            tokens = self.norm2(tokens + self.temporal(tokens))
        return tokens


def _build_variate(
    n_vars: int, d_model: int, n_heads: int, dropout: float, name: str
) -> nn.Module | None:
    """按名称构造变量维混合模块；``"none"`` 返回 ``None``。

    Raises:
        ValueError: 名称未知。
    """
    if name == "attention":
        return VariateAttention(d_model, n_heads, dropout)
    if name == "ffn":
        return VariateMlp(n_vars)
    if name == "none":
        return None
    raise ValueError(f"未知 variate_module：{name}")


def _build_temporal(
    d_model: int, d_ff: int, n_heads: int, dropout: float, name: str
) -> nn.Module | None:
    """按名称构造时间维混合模块；``"none"`` 返回 ``None``。

    Raises:
        ValueError: 名称未知。
    """
    if name == "ffn":
        return PositionwiseFFN(d_model, d_ff, dropout)
    if name == "attention":
        return TemporalAttention(d_model, n_groups=8, n_heads=n_heads, dropout=dropout)
    if name == "none":
        return None
    raise ValueError(f"未知 temporal_module：{name}")


class ITransformer(nn.Module):
    """倒置 Transformer 主模型。

    前向过程对应论文的三个公式：``h_n = Embedding(X[:, n])``、
    ``H^{l+1} = TrmBlock(H^l)``、``Y[:, n] = Projection(h_n^L)``。
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int,
        d_model: int = 512,
        d_ff: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        dropout: float = 0.1,
        variate_module: str = "attention",
        temporal_module: str = "ffn",
        use_layernorm: bool = True,
        use_pos_embed: bool = False,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.embedding = nn.Linear(seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.position_embedding = (
            nn.Parameter(torch.zeros(1, n_vars, d_model)) if use_pos_embed else None
        )
        self.blocks = nn.ModuleList(
            [
                InvertedBlock(
                    n_vars=n_vars,
                    d_model=d_model,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    dropout=dropout,
                    variate_module=variate_module,
                    temporal_module=temporal_module,
                    use_layernorm=use_layernorm,
                )
                for _ in range(e_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.projector = nn.Linear(d_model, pred_len)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Args:
            inputs: 形状 ``(B, T, N)`` 的回看窗口。

        Returns:
            形状 ``(B, S, N)`` 的预测。
        """
        tokens = self.embedding(inputs.transpose(1, 2))
        if self.position_embedding is not None:
            tokens = tokens + self.position_embedding
        tokens = self.embed_dropout(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return self.projector(tokens).transpose(1, 2)


class SeriesDecomposition(nn.Module):
    """DLinear 使用的移动平均趋势/季节项分解。

    Raises:
        ValueError: ``kernel_size`` 为偶数（会改变序列长度）。
    """

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须为奇数，以保持序列长度不变")
        self.kernel_size = kernel_size
        self.average = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, series: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            series: 形状 ``(B, N, T)`` 的序列。

        Returns:
            ``(seasonal, trend)``，与输入同形状。
        """
        trend = self.average(series)
        return series - trend, trend


class DLinear(nn.Module):
    """线性基线：序列分解 + 共享权重线性映射（各通道共用同一组权重）。"""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int,
        kernel_size: int = 25,
    ) -> None:
        super().__init__()
        del n_vars  # 共享权重设计下与通道数无关，仅为统一构造接口而保留参数
        self.decomposition = SeriesDecomposition(kernel_size)
        self.seasonal_head = nn.Linear(seq_len, pred_len)
        self.trend_head = nn.Linear(seq_len, pred_len)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Args:
            inputs: 形状 ``(B, T, N)`` 的回看窗口。

        Returns:
            形状 ``(B, S, N)`` 的预测。
        """
        series = inputs.transpose(1, 2)
        seasonal, trend = self.decomposition(series)
        forecast = self.seasonal_head(seasonal) + self.trend_head(trend)
        return forecast.transpose(1, 2)


class PatchTST(nn.Module):
    """PatchTST 基线：通道独立、patch 化 + 实例标准化。"""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int,
        patch_len: int = 16,
        patch_stride: int = 8,
        d_model: int = 128,
        d_ff: int = 256,
        n_heads: int = 16,
        e_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Args:
            seq_len: 回看长度 T。
            pred_len: 预测长度 S。
            n_vars: 变量数（通道独立，不参与计算）。
            patch_len: 每个 patch 覆盖的时间步数。
            patch_stride: patch 之间的步长。
            d_model: patch 嵌入维度。
            d_ff: 前馈隐层宽度。
            n_heads: 注意力头数。
            e_layers: 编码层数。
            dropout: Dropout 比例。

        Raises:
            ValueError: ``patch_len`` 大于 ``seq_len``。
        """
        super().__init__()
        del n_vars
        if patch_len > seq_len:
            raise ValueError("patch_len 不能大于 seq_len")
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.num_patches = (seq_len - patch_len) // patch_stride + 1
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            d_ff,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, e_layers)
        self.head = nn.Linear(self.num_patches * d_model, pred_len)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Args:
            inputs: 形状 ``(B, T, N)`` 的回看窗口。

        Returns:
            形状 ``(B, S, N)`` 的预测。
        """
        batch, _, n_vars = inputs.shape
        mean = inputs.mean(dim=1, keepdim=True)
        std = inputs.std(dim=1, keepdim=True).clamp_min(1e-5)
        normalised = (inputs - mean) / std

        series = normalised.transpose(1, 2).reshape(batch * n_vars, -1)
        patches = series.unfold(
            dimension=1, size=self.patch_len, step=self.patch_stride
        )
        tokens = self.patch_embedding(patches) + self.position_embedding
        encoded = self.encoder(tokens).reshape(batch * n_vars, -1)
        forecast = self.head(encoded).reshape(batch, n_vars, -1)
        return (forecast.transpose(1, 2) * std) + mean


class VanillaTransformer(nn.Module):
    """未倒置的通道独立 Transformer，作为"原结构"对照组。

    时间步是注意力 token 且保留位置编码，正是 iTransformer 论文所批评的设计。
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int,
        d_model: int = 64,
        d_ff: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        del n_vars
        self.seq_len = seq_len
        self.embedding = nn.Linear(1, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, e_layers)
        self.head = nn.Linear(seq_len * d_model, pred_len)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Args:
            inputs: 形状 ``(B, T, N)`` 的回看窗口。

        Returns:
            形状 ``(B, S, N)`` 的预测。
        """
        batch, _, n_vars = inputs.shape
        channels = inputs.transpose(1, 2).reshape(batch * n_vars, self.seq_len, 1)
        tokens = self.embedding(channels) + self.position_embedding
        encoded = self.encoder(tokens).reshape(batch * n_vars, -1)
        forecast = self.head(encoded).reshape(batch, n_vars, -1)
        return forecast.transpose(1, 2)


def build_model(
    config: ModelConfig, seq_len: int, pred_len: int, n_vars: int
) -> nn.Module:
    """按 ``config.name`` 实例化对应模型。

    Args:
        config: 模型配置。
        seq_len: 回看长度 T。
        pred_len: 预测长度 S。
        n_vars: 变量数 N。

    Returns:
        未搬到设备上的模型实例。

    Raises:
        ValueError: 模型名称未知。
    """
    if config.name == "itransformer":
        return ITransformer(
            seq_len=seq_len,
            pred_len=pred_len,
            n_vars=n_vars,
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            e_layers=config.e_layers,
            dropout=config.dropout,
            variate_module=config.variate_module,
            temporal_module=config.temporal_module,
            use_layernorm=config.use_layernorm,
            use_pos_embed=config.use_pos_embed,
        )
    if config.name == "dlinear":
        return DLinear(
            seq_len=seq_len,
            pred_len=pred_len,
            n_vars=n_vars,
            kernel_size=config.kernel_size,
        )
    if config.name == "patchtst":
        return PatchTST(
            seq_len=seq_len,
            pred_len=pred_len,
            n_vars=n_vars,
            patch_len=config.patch_len,
            patch_stride=config.patch_stride,
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            e_layers=config.e_layers,
            dropout=config.dropout,
        )
    if config.name == "transformer":
        return VanillaTransformer(
            seq_len=seq_len,
            pred_len=pred_len,
            n_vars=n_vars,
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            e_layers=config.e_layers,
            dropout=config.dropout,
        )
    raise ValueError(f"未知模型：{config.name}")
