"""对外 DTO（Pydantic 模型）。

与 domain.models 的分工：
    schemas 是**接口契约**，随 API 版本演进，字段可增可减；
    domain 模型是**业务语义**，保持稳定。
    两者由 services 层显式转换，不要让 ORM/领域对象直接序列化返回——
    否则数据库字段一改就会破坏对外契约。

命名约定：请求以 ...Request 结尾，响应以 ...Response 结尾。
"""

from .auth import LoginRequest, RegisterRequest, TokenOut, UserOut
from .chat import ChatRequest, ChatResponse, CitationOut, SourceOut
from .collection import (
    KnowledgeBaseCreate,
    KnowledgeBaseOut,
    KnowledgeBaseUpdate,
)
from .common import ErrorResponse, PageMeta
from .document import DocumentOut, UploadResponse

__all__ = [
    "LoginRequest",
    "RegisterRequest",
    "TokenOut",
    "UserOut",
    "ChatRequest",
    "ChatResponse",
    "CitationOut",
    "SourceOut",
    "KnowledgeBaseCreate",
    "KnowledgeBaseOut",
    "KnowledgeBaseUpdate",
    "ErrorResponse",
    "PageMeta",
    "DocumentOut",
    "UploadResponse",
]
