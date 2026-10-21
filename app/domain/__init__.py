"""领域层：领域模型与端口（抽象接口）。

本层是整个项目的最内层，**不得导入**任何框架、驱动或上层模块：

    ❌ 不导入 fastapi / langchain / pymilvus / sqlalchemy
    ✅ 只依赖标准库与 typing

上层的 infrastructure 通过实现 ports.py 中的 Protocol 接入具体能力。
"""

from .models import Answer, Chunk, Document, RetrievedChunk
from .ports import (
    ChunkRepositoryPort,
    DocumentParserPort,
    EmbedderPort,
    LLMPort,
    RerankerPort,
    SplitterPort,
    VectorStorePort,
)

__all__ = [
    "Answer",
    "Chunk",
    "Document",
    "RetrievedChunk",
    "ChunkRepositoryPort",
    "DocumentParserPort",
    "EmbedderPort",
    "LLMPort",
    "RerankerPort",
    "SplitterPort",
    "VectorStorePort",
]
