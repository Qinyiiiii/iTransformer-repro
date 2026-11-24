"""pytest 共享配置。"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True)
def clean_train_logger() -> None:
    """每个测试结束后关闭训练日志器的文件句柄。

    训练日志器按名称缓存处理器；在 Windows 上未关闭的文件句柄会阻止临时目录
    被清理，因此在测试间显式释放。
    """
    yield
    logger = logging.getLogger("itransformer_repro.train")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
