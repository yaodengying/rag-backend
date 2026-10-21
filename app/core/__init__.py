"""横切关注点：配置、日志、异常、依赖容器。

本层可被任意上层导入；除 core 自身的模块外，core 不反向依赖任何业务层。
"""

from .config import Settings, get_settings
from .exceptions import (
    AppError,
    ConfigurationError,
    NotFoundError,
    RetrievalError,
    UpstreamError,
    ValidationError,
)

__all__ = [
    "Settings",
    "get_settings",
    "AppError",
    "ConfigurationError",
    "NotFoundError",
    "RetrievalError",
    "UpstreamError",
    "ValidationError",
]
