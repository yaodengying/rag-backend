"""持久化层：SQLite 元数据存储（SQLAlchemy 2.x）。

存什么、不存什么：
    ✅ 存：知识库、文档的元数据（名称、状态、片段数、时间戳）
    ❌ 不存：片段正文与向量 —— 正文在 uploads + Milvus，向量在 Milvus

为什么用 SQLite 而不是 Milvus 承担元数据：
    Milvus 擅长向量相似度检索，但不适合做事务性的 CRUD 与关系查询。
    元数据量小、需要强一致与简单查询，SQLite 足够且零运维。
"""

from .database import Base, Database, get_session
from .models import DocumentRow, KnowledgeBaseRow
from .repositories import SqlAlchemyChunkRepository

__all__ = [
    "Base",
    "Database",
    "get_session",
    "DocumentRow",
    "KnowledgeBaseRow",
    "SqlAlchemyChunkRepository",
]
