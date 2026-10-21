"""会话历史相关 DTO。

命名对齐约定：
    `stored_message.citations` 的元素就是问答接口的 `CitationOut`——
    同一条引用在「刚回答完」与「三天后翻历史」两个场景里必须是同一套
    字段（filename / page / snippet），否则前端要写两套渲染。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.chat import CitationOut


class StoredMessageOut(BaseModel):
    """会话中的一条消息。"""

    role: str = Field(..., description="user / assistant")
    content: str
    created_at: str = Field(..., description='形如 "2026-10-17 12:00:00" 的本地时间')
    citations: list[CitationOut] = Field(
        default_factory=list,
        description="assistant 消息的溯源来源（文档名 + 页码）；user 消息恒为空",
    )


class ConversationSummaryOut(BaseModel):
    """会话列表项。"""

    id: str
    title: str = Field(..., description="取首条用户提问（截断），会话中途不改")
    kb_id: str = Field("", description="最近一次提问针对的知识库")
    created_at: str = ""
    updated_at: str = ""
    message_count: int = Field(..., description="消息条数（user+assistant 合计）")
    citation_count: int = Field(
        ..., description="本会话全部回答携带的引用总条数（知识库页统计的数据源）"
    )


class ConversationDetailOut(ConversationSummaryOut):
    """会话全量（含消息序列，按时间正序）。"""

    messages: list[StoredMessageOut] = Field(default_factory=list)
