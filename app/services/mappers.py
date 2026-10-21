"""领域模型 → 对外 DTO 的转换。

为什么单独放一个模块：
    架构约定「domain 模型与 HTTP 契约由 services 层做显式转换」。
    ingest 与 collection 两个服务都要把 Document 转成响应结构，
    与其在两处各写一遍，不如集中在这里——将来 DocumentOut 加字段时
    只用改一个地方。

注意：这里返回的是**普通 dict** 而不是 Pydantic 模型，
      接口层的 response_model 会在返回时做校验与序列化，
      这样 services 层不必依赖 schemas 层（保持依赖单向）。
"""

from __future__ import annotations

from app.domain.models import Chunk, Document, RetrievedChunk, User


def user_to_dict(user: User) -> dict:
    """User → 与 schemas.UserOut 对齐的字典。

    **刻意不含 password_hash**：转换函数是"领域模型 → 对外视图"的唯一出口，
    哈希从这里漏出去就再没有第二道闸门了。要哈希的调用方（登录校验）
    直接读领域对象，不要改这个函数。
    """
    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name or user.username,
        "is_active": user.is_active,
        "created_at": user.created_at,
    }


def document_to_dict(document: Document) -> dict:
    """Document → 与 schemas.DocumentOut 对齐的字典。"""
    return {
        "id": document.id,
        "kb_id": document.kb_id,
        "filename": document.filename,
        "status": document.status,
        "chunk_count": document.chunk_count,
        "error": document.error,
        "created_at": document.created_at,
    }


def rejected_document_dict(kb_id: str, filename: str, error: str) -> dict:
    """构造一个「未入库就被拒绝」的文档结构。

    用于批量上传：某个文件因扩展名 / 体积被拒时，仍要给出这一条结果，
    否则前端无法告诉用户"是这个文件没传上去、为什么"。
    此结构**没有**对应的数据库行，故 id 为空串。
    """
    return {
        "id": "",
        "kb_id": kb_id,
        "filename": filename,
        "status": "failed",
        "chunk_count": 0,
        "error": error,
        "created_at": None,
    }


def chunk_preview(chunk: Chunk, limit: int = 200) -> str:
    """片段正文的截断预览，用于引用展示。"""
    text = (chunk.text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def retrieved_to_dict(item: RetrievedChunk, preview_limit: int = 400) -> dict:
    """RetrievedChunk → 与 schemas.RetrievedChunkOut 对齐的字典。

    四个分数全部保留、且**允许为 None**，这是有意的：
    调参时最需要的信息恰恰是"这一路到底有没有召回它"。
    把缺失的分数补成 0 会让"没召回"与"召回了但得 0 分"看起来一样，
    而这两者的修法完全不同（前者改召回策略，后者改模型/语料）。
    """
    return {
        "chunk_id": item.chunk.id,
        "doc_id": item.chunk.doc_id,
        "filename": item.chunk.filename,
        "page": item.chunk.page_label,
        "chunk_index": item.chunk.chunk_index,
        "text": chunk_preview(item.chunk, preview_limit),
        "vector_score": item.vector_score,
        "bm25_score": item.bm25_score,
        "fusion_score": item.fusion_score,
        "rerank_score": item.rerank_score,
    }
