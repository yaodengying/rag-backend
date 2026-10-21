"""通用 DTO：错误响应与分页元信息。

统一响应结构的意义：前端只需写一次错误处理逻辑。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """统一错误响应体。"""

    code: str = Field(..., description="机器可读的错误码，如 not_found")
    message: str = Field(..., description="面向用户的错误描述")
    detail: Any | None = Field(None, description="附加信息，生产环境可省略")


class PageMeta(BaseModel):
    """分页元信息。"""

    total: int = Field(..., description="总条数")
    page: int = Field(1, description="当前页码，1-based")
    page_size: int = Field(20, description="每页条数")
