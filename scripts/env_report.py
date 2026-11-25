"""生成可复现实验环境记录 ``environment.txt``。

用途：
    收集 Python、PyTorch、CUDA/cuDNN、GPU 型号与显存、``pip freeze`` 以及
    ``nvidia-smi`` 输出，写入仓库根目录的 ``environment.txt``。该文件是课程
    规范要求的复现证据；更换机器或升级依赖后应重新生成并提交。

示例命令：
    python scripts/env_report.py
    python scripts/env_report.py --out environment.txt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from itransformer_repro.utils import environment_info, project_root


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="env_report.py",
        description="收集环境信息、pip freeze 与 GPU 状态，输出 environment.txt。",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("environment.txt"),
        help="输出路径，默认 environment.txt。",
    )
    return parser


def _run(command: list[str]) -> str:
    """执行命令并返回输出；失败时返回可读的错误说明。"""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return f"<无法执行 {' '.join(command)}: {exc}>"
    output = (result.stdout or "").strip()
    error = (result.stderr or "").strip()
    if result.returncode != 0:
        return f"<命令返回 {result.returncode}: {error or output}>"
    return output or "<无输出>"


def collect() -> str:
    """收集环境信息并返回文本。"""
    info: dict[str, Any] = environment_info()
    lines = [
        "# 由 scripts/env_report.py 生成；升级依赖或更换机器后请重新生成。",
        f"generated_by: Python {sys.version.split()[0]}",
        "",
        "[runtime]",
    ]
    lines.extend(f"{key}: {value}" for key, value in info.items())
    lines.extend(
        [
            "",
            "[pip_freeze]",
            _run([sys.executable, "-m", "pip", "freeze"]),
            "",
            "[nvidia_smi]",
            _run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv",
                ]
            ),
            "",
            "[note]",
            "CUDA/GPU 字段来自本机实际环境；CPU-only 机器上 cuda_available=False。",
            "完整训练矩阵建议使用 CUDA 11.8/12.x + PyTorch 2.x。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """写入环境记录并返回退出码。"""
    args = build_parser().parse_args(argv)
    out = args.out if args.out.is_absolute() else project_root() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(collect(), encoding="utf-8")
    print(f"已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
