# -*- coding: utf-8 -*-
"""会话历史服务：/conversations 一族接口的编排在这一层。

为什么不让路由直接摸 store：
    - 「REDIS_URL 未配置」的降级语义只该写一次：所有会话接口从这里统一
      拿到同一个明确报错（503），而不是各自碰 None 再各自编话术；
    - 领域对象 → DTO 的映射集中在 mappers，接口层不认识 dataclass。

与 ChatService 的分工：
    ChatService 只在问答链路里调 store（读历史、写一轮），不暴露 CRUD；
    本服务面向「翻历史」的读删场景。两边各取所需，谁也不依赖谁。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.core.exceptions import ServiceUnavailableError

if TYPE_CHECKING:
    from app.domain.models import Conversation, ConversationSummary
    from app.domain.ports import ConversationStorePort

logger = logging.getLogger(__name__)

# 未配置 REDIS_URL 时，会话接口统一报这一句——失败面收敛，运维一眼知道缺什么
_NOT_CONFIGURED = (
    "会话存储未配置（环境变量 REDIS_URL 为空）——"
    "请在 .env 中配置（如 redis://127.0.0.1:6379/0）后重启服务"
)


class ConversationService:
    """会话历史的读删编排。store 为 None 表示本进程未启用会话能力。"""

    def __init__(self, store: "ConversationStorePort | None") -> None:
        self._store = store

    @property
    def enabled(self) -> bool:
        """会话能力是否已配置（供能力发现接口 / 前端判断要不要隐藏入口）。"""
        return self._store is not None

    def _require_store(self) -> "ConversationStorePort":
        if self._store is None:
            raise ServiceUnavailableError(_NOT_CONFIGURED, detail={"reason": "redis_url_empty"})
        return self._store

    def list_conversations(
        self,
        kb_id: str | None = None,
        keyword: str | None = None,
        user_id: str | None = None,
    ) -> list["ConversationSummary"]:
        """列出会话；user_id 非 None 时只返回该账号名下的。"""
        return self._require_store().list_conversations(
            kb_id=kb_id, keyword=keyword, user_id=user_id
        )

    def get_conversation(
        self, conversation_id: str, user_id: str | None = None
    ) -> "Conversation":
        """读取会话详情；user_id 非 None 时归属不符抛 NotFoundError。"""
        return self._require_store().get_conversation(conversation_id, user_id)

    def delete_conversation(self, conversation_id: str, user_id: str | None = None) -> None:
        """删除会话；user_id 非 None 时归属不符抛 NotFoundError。"""
        self._require_store().delete_conversation(conversation_id, user_id)
