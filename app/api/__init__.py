"""接口层：HTTP 路由与依赖注入。

本层只做三件事：
    1. 解析与校验请求（交给 Pydantic 模型）；
    2. 调用应用服务；
    3. 把结果序列化为响应 / 把异常交给统一处理器。

禁止在本层写业务逻辑（如相似度过滤、提示词拼接）。
"""

from .deps import (
    get_chat_service,
    get_collection_service,
    get_ingest_service,
    get_retrieval_service,
)

__all__ = [
    "get_chat_service",
    "get_collection_service",
    "get_ingest_service",
    "get_retrieval_service",
]
