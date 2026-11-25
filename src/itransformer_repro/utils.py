"""通用工具：随机性控制、日志、JSON 读写与环境记录。"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def seed_everything(seed: int) -> None:
    """固定流程中所有随机源。

    对确定性算子只做"尽力而为"的请求：部分 CUDA 算子即便开启相关开关仍可能
    非确定。因此本项目的复现目标是"流程与统计结论一致"，而非跨硬件逐位相同。

    Args:
        seed: 非负整数种子。

    Raises:
        ValueError: 种子为负数。
    """
    if seed < 0:
        raise ValueError(f"seed 必须为非负整数，当前为 {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        # CUDA >= 10.2 要求固定 workspace 配置，确定性 cuBLAS 才可复现。
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """创建（或复用）同时输出到终端与文件的日志器。

    正式代码统一用 logging 记录阶段、参数与指标；``print`` 只留给 CLI 的简短
    用户提示。

    Args:
        name: 日志器名称，通常传 ``__name__``。
        log_file: 给定时会新建文件处理器并覆盖写入。

    Returns:
        已配置的日志器；重复调用不会重复添加处理器。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)
    has_stream = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )
    if not has_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        # 同一进程连续跑多次实验时，必须把文件处理器切换到本次 run 的日志，
        # 否则后续实验的日志会全部写进第一次实验的 train.log。
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """把 ``payload`` 写为 UTF-8 JSON，必要时创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 对象。

    Args:
        path: JSON 文件路径。

    Returns:
        解析后的字典。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 文件内容不是 JSON 对象。
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"期望 JSON 对象，实际为其他类型：{path}")
    return payload


def count_parameters(model: torch.nn.Module) -> int:
    """返回可训练标量参数总数。"""
    total = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            total += parameter.numel()
    return total


def resolve_device(requested: str = "auto") -> torch.device:
    """把设备字符串解析为 ``torch.device``。

    Args:
        requested: ``"auto"``、``"cpu"``、``"cuda"`` 或 ``"cuda:1"``。

    Returns:
        解析后的设备。

    Raises:
        RuntimeError: 请求 CUDA 但当前环境不可用。
        ValueError: 设备字符串无法识别。
    """
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "请求了 CUDA，但 torch.cuda.is_available() 为 False；"
            "请安装 CUDA 版 PyTorch 或改用 --device cpu"
        )
    if requested != "cpu" and not requested.startswith("cuda"):
        raise ValueError(f"不支持的设备：{requested}")
    return torch.device(requested)


def project_root() -> Path:
    """返回仓库根目录；安装到 site-packages 时退回当前工作目录。"""
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / ".git").exists() else Path.cwd()


def git_commit(cwd: Path | None = None) -> str:
    """返回工作区当前短提交哈希；不在 git 仓库时返回 ``"unknown"``。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd or project_root(),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def git_dirty(cwd: Path | None = None) -> bool | None:
    """工作区是否有未提交改动；不在 git 仓库时返回 ``None``。"""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd or project_root(),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def environment_info() -> dict[str, Any]:
    """收集运行时版本与硬件信息，写入实验记录以支持复现。"""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": "cpu",
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        info["gpu_memory_gb"] = round(total_bytes / 1024**3, 2)
    return info
