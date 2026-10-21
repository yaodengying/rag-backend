"""知识库管理路由（**全部需要登录**）。

REST 设计：
    GET    /knowledge-bases              列表（仅本人名下）
    POST   /knowledge-bases              新建（自动归属当前账号）
    PUT    /knowledge-bases/{kb_id}      更新
    DELETE /knowledge-bases/{kb_id}      删除（级联清理文档与向量）

注意 DELETE 语义：会连带删除该库下全部文档、Milvus 向量与上传文件，
属于不可逆操作，前端应二次确认。

鉴权的两层含义（缺一不可）：
    1. **登录**：每个处理函数都依赖 `current_user`，无令牌直接 401。
       这是由 `router = APIRouter(dependencies=[Depends(require_current_user)])`
       统一挂的——写在路由级而不是逐个函数，是因为"漏挂一个"就是一次
       数据泄漏，而这种事不会有任何报错。
    2. **归属**：`current_user.id` 一路传进服务层作为 owner_id。
       只做第 1 层的话，任何登录用户都能读写别人的知识库——
       门锁在楼门口而不是每户门口。

为什么处理函数要单独声明 `current_user` 参数：
    路由级 dependencies 只负责**拦截**，拿不到用户对象。要把 owner_id
    传下去，就必须在处理函数签名里再取一次。FastAPI 对同一个依赖在同一
    请求内做缓存，所以这里不会重复解令牌、重复查库。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, status

from app.api.deps import get_collection_service
from app.api.v1.auth import require_current_user, router_dependencies
from app.schemas.auth import UserOut
from app.schemas.collection import (
    KnowledgeBaseCreate,
    KnowledgeBaseOut,
    KnowledgeBaseUpdate,
)
from app.services.collection_service import CollectionService

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=router_dependencies)


@router.get("", response_model=list[KnowledgeBaseOut], summary="知识库列表")
async def list_knowledge_bases(
    current_user: UserOut = Depends(require_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> list[KnowledgeBaseOut]:
    """列出**当前账号**名下的知识库。"""
    return [
        KnowledgeBaseOut.model_validate(item)
        for item in service.list_knowledge_bases(current_user.id)
    ]


@router.post(
    "",
    response_model=KnowledgeBaseOut,
    status_code=status.HTTP_201_CREATED,
    summary="新建知识库",
)
async def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    current_user: UserOut = Depends(require_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> KnowledgeBaseOut:
    """新建知识库，归属强制为当前账号。

    归属**不由请求体决定**：schema 里根本没有 owner_id 字段。
    让客户端能指定归属，等于留了一个"把库建到别人名下"的接口。
    """
    created = service.create_knowledge_base(
        payload.name, payload.description, current_user.id
    )
    return KnowledgeBaseOut.model_validate(created)


@router.put("/{kb_id}", response_model=KnowledgeBaseOut, summary="更新知识库")
async def update_knowledge_base(
    kb_id: str,
    payload: KnowledgeBaseUpdate,
    current_user: UserOut = Depends(require_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> KnowledgeBaseOut:
    """更新知识库名称与描述。

    Raises:
        NotFoundError: kb_id 不存在**或不属于当前账号**（同一句话）
    """
    updated = service.update_knowledge_base(
        kb_id, payload.name, payload.description, current_user.id
    )
    return KnowledgeBaseOut.model_validate(updated)


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT, summary="删除知识库")
async def delete_knowledge_base(
    kb_id: str,
    current_user: UserOut = Depends(require_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> None:
    """删除知识库及其全部文档、向量与文件（不可逆）。

    归属校验在服务层**清理动作之前**：否则一次越权请求就会连带删掉
    别人的 Milvus 向量与上传文件——那不只是读越权，是破坏。
    """
    service.delete_knowledge_base(kb_id, current_user.id)
