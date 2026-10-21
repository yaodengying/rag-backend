"""依赖容器：集中装配单例，供接口层通过 Depends 取用。

为什么要单独一层：
    - 解析器、向量模型、Milvus 客户端、LLM 客户端都是重量级对象，
      必须进程内复用，不能每个请求重新构造。
    - 把「构造什么实现」与「业务怎么用」解耦：
      services 只依赖 domain/ports.py 中的协议，具体实现由本容器注入。
    - 测试时可替换为 Fake 实现，无需改动业务代码。

对象生命周期：
    - 单例（lru_cache）：Embedder、MilvusStore、HybridRetriever、DB、两个仓储、
      Pipeline、各 Service、LLM 客户端、两张编译后的图
    - 每请求：DB Session（见 app/db/database.get_session）

装配顺序（自底向上，逆过来会拿到半成品对象）：
    settings → database → repository / user_repository
             → parser / splitter / embedder / vector_store → pipeline → services
             → hybrid_retriever → retrieval_service
             → llm → (rag_graph | retrieval_graph) → chat_service
             → auth_service
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from app.db.database import Database
    from app.domain.ports import (
        ConversationStorePort,
        EmbedderPort,
        LLMPort,
        RerankerPort,
        VectorStorePort,
    )
    from app.infrastructure.parsing import CompositeDocumentParser
    from app.infrastructure.splitting import LangChainSplitter
    from app.rag.ingest import IngestionPipeline
    from app.rag.retrieval import HybridRetriever
    from app.services.auth_service import AuthService
    from app.services.chat_service import ChatService
    from app.services.collection_service import CollectionService
    from app.services.conversation_service import ConversationService
    from app.services.ingest_service import IngestService
    from app.services.retrieval_service import RetrievalService


# ---------- 基础设施单例 ----------


@lru_cache
def get_database() -> "Database":
    """SQLite 元数据库（引擎 + Session 工厂）单例。"""
    from app.core.config import get_settings
    from app.db.database import Database

    return Database(get_settings().sqlite_path)


@lru_cache
def get_repository():
    """元数据仓储单例（实现 ChunkRepositoryPort）。"""
    from app.db.repositories import SqlAlchemyChunkRepository

    return SqlAlchemyChunkRepository(get_database())


@lru_cache
def get_user_repository():
    """账号仓储单例（实现 UserRepositoryPort）。

    与 get_repository() 分开：账号是身份数据，与知识库/文档的生命周期不同，
    见 domain/ports.py 的 UserRepositoryPort 文档。
    """
    from app.db.repositories import SqlAlchemyUserRepository

    return SqlAlchemyUserRepository(get_database())


@lru_cache
def get_parser() -> "CompositeDocumentParser":
    """文档解析器单例（无状态，但复用可省去重复建索引）。"""
    from app.infrastructure.parsing import CompositeDocumentParser

    return CompositeDocumentParser()


@lru_cache
def get_splitter() -> "LangChainSplitter":
    """文本切分器单例。"""
    from app.core.config import get_settings
    from app.infrastructure.splitting import LangChainSplitter

    settings = get_settings()
    return LangChainSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


@lru_cache
def get_vector_store() -> "VectorStorePort":
    """向量库适配器单例（Milvus 实现）。"""
    from app.core.config import get_settings
    from app.infrastructure.vectorstore import MilvusVectorStore

    settings = get_settings()
    return MilvusVectorStore(
        uri=settings.milvus_uri,
        collection=settings.milvus_collection,
        user=settings.milvus_user or None,
        password=settings.milvus_password or None,
    )


@lru_cache
def get_embedder() -> "EmbedderPort":
    """向量化适配器单例（本地 BGE 权重）。重量级，务必复用。"""
    from app.infrastructure.embedding import BgeEmbedder

    return BgeEmbedder()


@lru_cache
def get_reranker() -> "RerankerPort | None":
    """重排适配器单例；settings.rerank_enabled 为 False 时返回 None。

    注意返回类型里的 `| None` 是**语义的一部分**：None 表示"明确关闭了重排"，
    而不是"没实现/装配失败"。上层据此把链路降级为「按融合分截断」，
    而不是报错——这与 embedding / vector_store 缺了就不可用不同。
    重排是精度增强，不是链路必需件。
    """
    from app.core.config import get_settings

    if not get_settings().rerank_enabled:
        return None

    from app.infrastructure.rerank import BgeReranker

    return BgeReranker()


@lru_cache
def get_llm() -> "LLMPort":
    """LLM 适配器单例（OpenAI 兼容协议，默认本机 Ollama）。

    构造期**不建立连接**：这样"服务能不能起来"不取决于 Ollama 是否在跑。
    真实连接推迟到第一次生成时才做（见 infrastructure/llm.py）。
    """
    from app.infrastructure.llm import OpenAICompatibleLLM

    return OpenAICompatibleLLM()


@lru_cache
def get_rag_graph() -> "CompiledStateGraph":
    """编译后的完整问答图单例（含改写重试与短路拒答）。

    图是编译产物、可并发复用，不要每请求重新 build——compile() 有实打实的
    开销，而且每次 build 都会重新注册节点闭包。
    """
    from app.rag.graph import build_rag_graph

    return build_rag_graph(
        retriever=get_hybrid_retriever(),
        reranker=get_reranker(),
        llm=get_llm(),
    )


@lru_cache
def get_retrieval_graph() -> "CompiledStateGraph":
    """检索子图单例（START → retrieve → rerank → END），供 SSE 流式路径使用。

    与 `get_rag_graph()` 共用同一个 `get_hybrid_retriever()` 与 `get_reranker()`，
    所以"流式看到的来源"与"非流式看到的来源"必然是同一批。
    """
    from app.rag.graph import build_retrieval_graph

    return build_retrieval_graph(retriever=get_hybrid_retriever(), reranker=get_reranker())


# ---------- 入库链路 ----------


@lru_cache
def get_ingestion_pipeline() -> "IngestionPipeline":
    """入库流水线单例。"""
    from app.core.config import get_settings
    from app.rag.ingest import IngestionPipeline

    return IngestionPipeline(
        parser=get_parser(),
        splitter=get_splitter(),
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        repository=get_repository(),
        upload_dir=get_settings().upload_dir,
    )


# ---------- 应用服务单例 ----------


@lru_cache
def get_collection_service() -> "CollectionService":
    """知识库管理服务单例。"""
    from app.core.config import get_settings
    from app.services.collection_service import CollectionService

    return CollectionService(
        repository=get_repository(),
        vector_store=get_vector_store(),
        upload_dir=get_settings().upload_dir,
        # 注入会话存储：删库时要连带清理该 kb 的会话，否则留下孤儿记录。
        # get_conversation_store() 在未配置 REDIS_URL 时返回 None，
        # CollectionService 内部据此跳过——删库不该因为没开会话能力而失败。
        conversation_store=get_conversation_store(),
    )


@lru_cache
def get_ingest_service() -> "IngestService":
    """文档入库服务单例。"""
    from app.services.ingest_service import IngestService

    return IngestService(
        pipeline=get_ingestion_pipeline(),
        repository=get_repository(),
    )


@lru_cache
def get_hybrid_retriever() -> "HybridRetriever":
    """混合检索器单例（两路召回 + RRF 融合）。

    单独暴露出来，是为了让 `get_retrieval_service()` 与将来的
    `get_rag_graph()` 拿到**同一个实例**：融合参数是配置的一部分，
    两个消费者各造一个的话，"接口查到的"与"问答用的"就可能不是同一套检索。
    检索器本身无状态，共用一个实例也是安全的。
    """
    from app.core.config import get_settings
    from app.rag.retrieval import HybridRetriever

    settings = get_settings()
    return HybridRetriever(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        top_k=settings.retrieve_top_k,
        hybrid_enabled=settings.hybrid_enabled,
        rrf_k=settings.hybrid_rrf_k,
        vector_weight=settings.hybrid_vector_weight,
        bm25_weight=settings.hybrid_bm25_weight,
        # 刻意不传 min_similarity：它按模式取不同的默认值
        # （混合模式用 HYBRID_DENSE_MIN_SIMILARITY，纯向量模式用 MIN_SIMILARITY）。
        # 在这里硬传 settings.min_similarity 会让混合模式也套上 0.35，
        # 而那正是会把稠密路静默清空的取值——见 HybridRetriever 的模块文档。
    )


@lru_cache
def get_retrieval_service() -> "RetrievalService":
    """检索服务单例（混合召回 + 可选重排）。"""
    from app.services.retrieval_service import RetrievalService

    return RetrievalService(
        retriever=get_hybrid_retriever(),
        reranker=get_reranker(),
        # 仓储只用于归属校验：检索本身不落数据，但"能读到谁的片段"必须受控
        repository=get_repository(),
    )


@lru_cache
def get_auth_service() -> "AuthService":
    """认证服务单例（注册 / 登录 / 令牌校验）。

    只注入仓储：密钥、有效期、迭代次数、是否开放注册都从 Settings 读，
    不在这里展开成参数——容器越"聪明"，配置漂移越难追。
    """
    from app.services.auth_service import AuthService

    return AuthService(repository=get_user_repository())


@lru_cache
def get_conversation_store() -> "ConversationStorePort | None":
    """会话存储单例；REDIS_URL 未配置时返回 None（能力整体关闭）。

    Redis 客户端**惰性建连**（redis-py 默认行为）：构造时连不上不炸，
    首次命令才失败——所以这里不做连通性预检，失败语义统一在
    RedisConversationStore._run 里包成 UpstreamError。
    """
    from app.core.config import get_settings
    from app.infrastructure.redis_store import RedisConversationStore

    url = get_settings().redis_url.strip()
    if not url:
        return None
    import redis

    client = redis.Redis.from_url(url, decode_responses=False)
    return RedisConversationStore(
        client=client,
        ttl_seconds=get_settings().conversation_ttl_hours * 3600,
    )


@lru_cache
def get_conversation_service() -> "ConversationService":
    """会话历史服务单例。"""
    from app.services.conversation_service import ConversationService

    return ConversationService(store=get_conversation_store())


@lru_cache
def get_chat_service() -> "ChatService":
    """问答服务单例（配置了会话存储时自动具备多轮历史与落库能力）。"""
    from app.services.chat_service import ChatService

    return ChatService(
        graph=get_rag_graph(),
        llm=get_llm(),
        retrieval_graph=get_retrieval_graph(),
        conversation_store=get_conversation_store(),
        # 归属校验用（知识库是否属于当前账号）
        repository=get_repository(),
    )


# ---------- 生命周期 ----------


def _is_built(cached) -> bool:
    """判断 lru_cache 包装的函数是否已经构造过对象。"""
    try:
        return cached.cache_info().currsize > 0
    except AttributeError:  # pragma: no cover - 非 lru_cache 包装
        return False


def shutdown() -> None:
    """释放全局资源（Milvus 连接、线程池等），由 lifespan 在关闭时调用。

    只清理**已经构造过**的单例：惰性装配意味着某些对象可能从未被创建，
    直接调用它们的构造函数会平白多连一次 Milvus。
    """
    import logging

    logger = logging.getLogger(__name__)

    if _is_built(get_vector_store):
        try:
            get_vector_store().close()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            logger.exception("关闭 Milvus 连接失败")
        get_vector_store.cache_clear()

    if _is_built(get_database):
        try:
            get_database().dispose()
        except Exception:  # noqa: BLE001
            logger.exception("释放数据库连接池失败")
        get_database.cache_clear()

    for clearable in (
        get_chat_service,
        get_conversation_service,
        get_conversation_store,
        get_auth_service,
        get_rag_graph,
        get_retrieval_graph,
        get_ingestion_pipeline,
        get_ingest_service,
        get_collection_service,
        get_retrieval_service,
        get_hybrid_retriever,
        get_llm,
        get_reranker,
        get_embedder,
        get_splitter,
        get_parser,
        get_user_repository,
        get_repository,
    ):
        if _is_built(clearable):
            clearable.cache_clear()
