"""实验配置：YAML 读取、点号覆盖与合法性校验。

配置是参数的唯一事实来源：批大小、学习率、数据目录、随机种子等可变参数只写在
YAML 或命令行里，不允许散落为训练脚本中的魔法数字。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

MODEL_NAMES = ("itransformer", "dlinear", "patchtst", "transformer")
VARIATE_MODULES = ("attention", "ffn", "none")
TEMPORAL_MODULES = ("ffn", "attention", "none")
FEATURE_MODES = ("M",)


@dataclass(frozen=True)
class DataConfig:
    """数据集与滑窗设置。

    Attributes:
        name: 基准数据集名，例如 ``"ETTh1"``，需在
            :data:`itransformer_repro.data.DATASET_SPECS` 中注册。
        root: 存放各数据集的目录。
        file: 相对 ``root`` 的 CSV 路径。
        target: 目标列名（仅当 ``features="S"`` 时使用，本项目固定为 ``"M"``）。
        features: ``"M"`` 表示多变量输入、多变量输出。
        seq_len: 回看窗口长度 T。
        pred_len: 预测步长 S。
    """

    name: str = "ETTh1"
    root: str = "data/raw"
    file: str = "ETT-small/ETTh1.csv"
    target: str = "OT"
    features: str = "M"
    seq_len: int = 96
    pred_len: int = 96


@dataclass(frozen=True)
class ModelConfig:
    """模型超参数与消融开关。

    Attributes:
        name: ``itransformer`` / ``dlinear`` / ``patchtst`` / ``transformer``。
        d_model: 每个变量 token 的嵌入维度 D。
        d_ff: 前馈网络隐层宽度。
        n_heads: 注意力头数，必须整除 ``d_model``。
        e_layers: 编码层数 L。
        dropout: Dropout 比例。
        variate_module: 变量维混合方式，``attention`` / ``ffn`` / ``none``。
        temporal_module: 时间维混合方式，``ffn`` / ``attention`` / ``none``。
        use_layernorm: 是否使用倒置的 LayerNorm（对每个变量 token 归一化）。
        use_pos_embed: 是否在变量 token 上加可学习位置编码（主方法默认关闭）。
        patch_len: PatchTST 的 patch 长度。
        patch_stride: PatchTST 的 patch 步长。
        kernel_size: DLinear 移动平均分解核大小，必须为奇数。
    """

    name: str = "itransformer"
    d_model: int = 512
    d_ff: int = 512
    n_heads: int = 8
    e_layers: int = 2
    dropout: float = 0.1
    variate_module: str = "attention"
    temporal_module: str = "ffn"
    use_layernorm: bool = True
    use_pos_embed: bool = False
    patch_len: int = 16
    patch_stride: int = 8
    kernel_size: int = 25


@dataclass(frozen=True)
class TrainConfig:
    """优化器设置。论文配方：Adam、MSE 损失、学习率 1e-3、10 个 epoch、无早停。"""

    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 10
    patience: int = 0
    num_workers: int = 0
    save_train_state: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    """一次完整训练记录所需的全部信息。"""

    experiment: str
    seed: int = 2021
    device: str = "auto"
    runs_dir: str = "results/raw"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的字典，用于写入 ``config.json``。"""
        return asdict(self)

    def run_dir(self) -> Path:
        """返回该次运行的输出目录。

        目录遵循 ``results/raw/<model>/<dataset>/h<S>/<variant>/seed<seed>``：
        主实验省略 ``variant``，消融或其它变体使用去掉 ``_<dataset>_h<S>`` 后缀
        的实验名，保证每个结果都有唯一且可读的路径。
        """
        base = (
            Path(self.runs_dir)
            / self.model.name
            / self.data.name
            / f"h{self.data.pred_len}"
        )
        default_name = f"{self.model.name}_{self.data.name}_h{self.data.pred_len}"
        variant = self.experiment
        if variant == default_name:
            variant = ""
        else:
            suffix = f"_{self.data.name}_h{self.data.pred_len}"
            if variant.endswith(suffix):
                variant = variant[: -len(suffix)]
        if variant:
            base = base / variant
        return base / f"seed{self.seed}"


def coerce_scalar(text: str) -> bool | int | float | str:
    """把命令行字符串转换为 ``bool`` / ``int`` / ``float`` / ``str``。

    Args:
        text: 形如 ``"true"``、``"192"``、``"1e-3"``、``"ffn"`` 的原始文本。

    Returns:
        类型推断后的标量；无法解析时原样返回字符串。
    """
    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def apply_overrides(
    config: ExperimentConfig, overrides: dict[str, Any]
) -> ExperimentConfig:
    """在 ``config`` 的副本上应用点号键覆盖。

    Args:
        config: 基础配置。
        overrides: 形如 ``{"data.pred_len": 192, "seed": 2022}`` 的映射；
            不含点号的键视为顶层字段。

    Returns:
        应用覆盖后的新配置对象（原对象不变）。

    Raises:
        KeyError: 段名不是 ``data`` / ``model`` / ``train`` 之一。
        TypeError: 字段名在目标 dataclass 中不存在。
    """
    section_values: dict[str, dict[str, Any]] = {"data": {}, "model": {}, "train": {}}
    top_values: dict[str, Any] = {}
    for key, value in overrides.items():
        if "." not in key:
            top_values[key] = value
            continue
        section, field_name = key.split(".", 1)
        if section not in section_values:
            raise KeyError(f"未知配置段 '{section}'（覆盖项 '{key}'）")
        section_values[section][field_name] = value

    updated = replace(config, **top_values) if top_values else config
    for section, values in section_values.items():
        if values:
            nested = replace(getattr(updated, section), **values)
            updated = replace(updated, **{section: nested})
    return updated


def validate_config(config: ExperimentConfig) -> None:
    """检查配置取值，避免非法参数在训练深处才报错。

    Args:
        config: 待校验配置。

    Raises:
        ValueError: 任一取值超出允许范围。
    """
    model = config.model
    data = config.data
    if model.name not in MODEL_NAMES:
        raise ValueError(f"model.name 必须是 {MODEL_NAMES} 之一，当前为 {model.name!r}")
    if model.variate_module not in VARIATE_MODULES:
        raise ValueError(f"variate_module 必须是 {VARIATE_MODULES} 之一")
    if model.temporal_module not in TEMPORAL_MODULES:
        raise ValueError(f"temporal_module 必须是 {TEMPORAL_MODULES} 之一")
    if data.features not in FEATURE_MODES:
        raise ValueError(
            f"features 仅支持 {FEATURE_MODES}：本项目复现多变量输入/多变量输出，"
            "单变量与 MS 模式未实现"
        )
    if data.seq_len <= 1 or data.pred_len <= 0:
        raise ValueError("seq_len 必须大于 1，pred_len 必须大于 0")
    if model.d_model <= 0 or model.n_heads <= 0 or model.d_ff <= 0:
        raise ValueError("d_model、n_heads、d_ff 必须为正整数")
    if model.d_model % model.n_heads != 0:
        raise ValueError(
            f"d_model({model.d_model}) 必须能被 n_heads({model.n_heads}) 整除"
        )
    if model.e_layers <= 0:
        raise ValueError("e_layers 必须为正整数")
    if not 0.0 <= model.dropout < 1.0:
        raise ValueError("dropout 必须落在 [0, 1) 区间")
    if config.train.epochs <= 0 or config.train.batch_size <= 0:
        raise ValueError("epochs 与 batch_size 必须为正整数")
    if config.train.learning_rate <= 0:
        raise ValueError("learning_rate 必须为正数")
    if config.train.patience < 0:
        raise ValueError("patience 不能为负数（0 表示关闭早停）")
    if config.seed < 0:
        raise ValueError("seed 必须为非负整数")


def load_config(
    path: Path, overrides: dict[str, Any] | None = None
) -> ExperimentConfig:
    """读取 YAML 配置，应用覆盖并完成校验。

    Args:
        path: 含 ``experiment``、``seed``、``data``、``model``、``train``
            以及可选 ``device`` / ``runs_dir`` 的 YAML 文件。
        overrides: 可选的覆盖项，键使用点号路径。

    Returns:
        校验通过的配置对象。

    Raises:
        FileNotFoundError: 配置文件不存在。
        ValueError: YAML 内容不是映射，或合并后的字段非法。
    """
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"配置文件顶层必须是映射：{path}")

    try:
        config = ExperimentConfig(
            experiment=str(raw.get("experiment", path.stem)),
            seed=int(raw.get("seed", 2021)),
            device=str(raw.get("device", "auto")),
            runs_dir=str(raw.get("runs_dir", "results/raw")),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
        )
    except TypeError as exc:
        raise ValueError(f"配置文件中存在未知字段：{exc}") from exc
    if overrides:
        config = apply_overrides(config, overrides)
    validate_config(config)
    return config
