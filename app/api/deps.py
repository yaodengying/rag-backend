"""FastAPI 依赖注入工厂。

作用：把 core/container.py 里的单例包装成 FastAPI 的依赖项，
接口层用 `service: ChatService = Depends(get_chat_service)` 取用，
从而不在路由函数里 import 容器（便于测试时 override）。

FastAPI 测试中覆盖依赖的方式：
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.auth_service import AuthService
    from app.services.chat_service import ChatService
    from app.services.collection_service import CollectionService
    from app.services.conversation_service import ConversationService
    from app.services.ingest_service import IngestService
    from app.services.retrieval_service import RetrievalService


def get_auth_service() -> "AuthService":
    """提供认证服务（注册 / 登录 / 令牌校验）。"""
    from app.core import container

    return container.get_auth_service()


def get_collection_service() -> "CollectionService":
    """提供知识库管理服务。"""
    from app.core import container

    return container.get_collection_service()


def get_ingest_service() -> "IngestService":
    """提供入库服务。"""
    from app.core import container

    return container.get_ingest_service()


def get_retrieval_service() -> "RetrievalService":
    """提供检索服务。"""
    from app.core import container

    return container.get_retrieval_service()


def get_chat_service() -> "ChatService":
    """提供问答服务。"""
    from app.core import container

    return container.get_chat_service()


def get_conversation_service() -> "ConversationService":
    """提供会话历史服务（REDIS_URL 未配置时服务内部降级为 503）。"""
    from app.core import container

    return container.get_conversation_service()
