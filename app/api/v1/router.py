"""v1 路由汇总。

在此聚合各子路由，main.py 只挂载一次，避免到处 add_api_route。

注意 path 前缀的分工：
    - /knowledge-bases 由 collections 路由自带（它的路径就是 ""）；
    - documents 路由的路径里已含 /knowledge-bases/{kb_id}，
      所以**不能**再给它加 prefix，否则会变成
      /knowledge-bases/knowledge-bases/{kb_id}/documents。
       retrieval 路由同理（它的路径是 /knowledge-bases/{kb_id}/search）。
    - chat 路由自带 /chat 与 /chat/stream；它**不**按知识库分路径，
      因为 kb_id 在请求体里（见 schemas/chat.py 的 ChatRequest）——
      将来要支持多库问答时，改的是请求体字段，不必动 URL 结构。
    - auth 路由自带 /auth/login 与 /auth/register；它只**发**令牌，
      不给任何路由加鉴权（见 api/v1/auth.py 的模块文档）。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import auth, chat, collections, conversations, documents, health, retrieval

api_router = APIRouter()

api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(
    collections.router, prefix="/knowledge-bases", tags=["knowledge-bases"]
)
api_router.include_router(documents.router, tags=["documents"])
api_router.include_router(retrieval.router, tags=["retrieval"])
api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(conversations.router, tags=["conversations"])
