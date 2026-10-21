"""文档相关 DTO。

status 用 Literal 收窄取值范围：上游写错状态时在序列化处就暴露，
而不是让前端拿到一个它不认识的值。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    """文档元数据响应。

    status 语义：
        processing 解析中（前端应轮询或提示等待）
        ready      可检索
        failed     失败，error 字段给出原因
    """

    id: str
    kb_id: str
    filename: str
    status: Literal["processing", "ready", "failed"] = "processing"
    chunk_count: int = 0
    error: str = ""
    created_at: datetime | None = None


class UploadResponse(BaseModel):
    """单文件上传结果。"""

    document: DocumentOut
    message: str = Field("", description="附加提示，如失败原因")
