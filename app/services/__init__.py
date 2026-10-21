"""应用服务层：用例编排。

职责边界：
    - 只做「用例层面的流程组织」与事务边界，不写算法细节、不碰 HTTP。
    - 依赖 domain/ports.py 的协议，具体实现由 core/container.py 注入。
    - 把领域模型转换成对外的 Pydantic DTO（schemas）返回给接口层。

与 rag 层的分工：
    rag 关注「单次问答/单次入库怎么走」；
    services 关注「什么条件下走、走之前之后还要做什么」——
    例如删文档时既要清元数据也要清 Milvus 向量，这种跨存储的一致性由服务层负责。
"""

from .chat_service import ChatService
from .collection_service import CollectionService
from .ingest_service import IngestService
from .retrieval_service import RetrievalService

__all__ = ["ChatService", "CollectionService", "IngestService", "RetrievalService"]
