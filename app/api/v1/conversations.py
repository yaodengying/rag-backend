# -*- coding: utf-8 -*-
"""会话历史接口（多轮对话历史存 Redis，本路由只负责「翻历史」）。

路由处理函数写同步 `def`：Redis 客户端是同步阻塞 IO（本机 unix socket/TCP，
毫秒级），挂 async 反而要另起 to_thread，收益为零。

会话的「写」不在这里：问答链路（/chat、/chat/stream）在回答收尾时把
user+assistant 两轮连同溯源（文档名 + 页码）写进 Redis——历史是问答的
副产品，不需要前端再发一次保存请求（少一个可漏的步骤）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Query, status

from app.api.deps import get_conversation_service
from app.api.v1.auth import require_current_user, router_dependencies
from app.schemas.auth import UserOut
from app.schemas.chat import CitationOut
from app.schemas.conversation import ConversationDetailOut, ConversationSummaryOut, StoredMessageOut

if TYPE_CHECKING:
    from app.domain.models import ConversationSummary, StoredMessage
    from app.services.conversation_service import ConversationService

router = APIRouter(dependencies=router_dependencies)


def _summary_out(item: "ConversationSummary") -> ConversationSummaryOut:
    return ConversationSummaryOut(
        id=item.id,
        title=item.title,
        kb_id=item.kb_id,
        created_at=item.created_at,
        updated_at=item.updated_at,
        message_count=item.message_count,
        citation_count=item.citation_count,
    )


def _message_out(message: "StoredMessage") -> StoredMessageOut:
    return StoredMessageOut(
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        citations=[
            CitationOut(
                doc_id=c.doc_id,
                filename=c.filename,
                page=c.page,
                snippet=c.snippet,
            )
            for c in message.citations
        ],
    )


@router.get(
    "/conversations",
    response_model=list[ConversationSummaryOut],
    summary="会话列表（按最近更新倒序）",
)
def list_conversations(
    kb_id: str | None = Query(None, description="只列该知识库的会话"),
    keyword: str | None = Query(None, description="标题包含的关键词（不区分大小写）"),
    current_user: UserOut = Depends(require_current_user),
    service: "ConversationService" = Depends(get_conversation_service),
) -> list[ConversationSummaryOut]:
    """列出**当前账号**名下的会话。

    **不能只按 kb_id 过滤**：会话正文就是用户问过的内容，而 kb_id 是
    可猜的（它出现在 URL 里）。只按 kb 过滤等于"知道库 ID 就能读全部对话"。
    """
    return [
        _summary_out(item)
        for item in service.list_conversations(
            kb_id=kb_id, keyword=keyword, user_id=current_user.id
        )
    ]


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetailOut,
    summary="会话详情（消息按时间正序，assistant 消息带溯源）",
)
def get_conversation(
    conversation_id: str,
    current_user: UserOut = Depends(require_current_user),
    service: "ConversationService" = Depends(get_conversation_service),
) -> ConversationDetailOut:
    """会话详情。别人（或不存在）的会话一律 404，同一句话。"""
    conversation = service.get_conversation(conversation_id, current_user.id)
    return ConversationDetailOut(
        **_summary_out(conversation).model_dump(),
        messages=[_message_out(m) for m in conversation.messages],
    )


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除会话（幂等）",
)
def delete_conversation(
    conversation_id: str,
    current_user: UserOut = Depends(require_current_user),
    service: "ConversationService" = Depends(get_conversation_service),
) -> None:
    """删除会话。删除自己的会话是幂等的；删别人的返回 404。"""
    service.delete_conversation(conversation_id, current_user.id)
