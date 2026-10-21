"""日志配置：统一格式、级别与第三方库降噪。

设计要点：
    - 提供 setup_logging() 供应用启动时调用一次。
    - 第三方库（pymilvus / transformers / httpx）默认噪音大，需单独降级。
    - 日志级别来自 Settings.log_level。

中文 Windows 的编码陷阱：
    控制台默认是 GBK(cp936)。业务日志里有中文（错误消息尤其多），
    一旦 cp936 编不出某个字符，logging 会抛 UnicodeEncodeError —— 而它发生在
    异常处理路径上，会把真正的错误盖掉，非常难查。
    所以启动时把 stdout/stderr 显式切到 UTF-8 并 errors="replace"。
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

# 需要降噪的第三方库
NOISY_LOGGERS: tuple[str, ...] = (
    "pymilvus",
    "transformers",
    "FlagEmbedding",
    "datasets",
    "urllib3",
    "httpx",
    "httpcore",
    "watchfiles",
)

_LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def _ensure_utf8_console() -> None:
    """把标准输出/错误流切到 UTF-8，避免中文日志把进程打挂。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - 环境不支持时静默跳过，不影响启动
            pass


def setup_logging(level: str = "INFO") -> None:
    """初始化全局日志配置。

    Args:
        level: 根日志级别，如 "INFO" / "DEBUG"。
    """
    _ensure_utf8_console()

    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        force=True,  # 重复调用（如 reload）时覆盖旧配置，避免日志重复输出
    )

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger(__name__).debug("日志已初始化，级别=%s", logging.getLevelName(numeric_level))
