"""知识库相关 DTO。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class KnowledgeBaseCreate(BaseModel):
    """新建知识库请求。"""

    name: str = Field(..., min_length=1, max_length=100, description="知识库名称")
    description: str = Field("", max_length=500, description="描述")


class KnowledgeBaseUpdate(BaseModel):
    """更新知识库请求。"""

    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field("", max_length=500)


class KnowledgeBaseOut(BaseModel):
    """知识库响应。"""

    id: str
    name: str
    description: str = ""
    doc_count: int = 0
    created_at: datetime | None = None
