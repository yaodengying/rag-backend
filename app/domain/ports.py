"""端口：外部能力的抽象接口（Protocol）。

这一层是「架构的铰链」：
    - 上层（rag / services）只依赖本文件的协议，不知道背后是 Milvus 还是别的库；
    - 下层（infrastructure）实现这些协议，替换实现不影响业务代码；
    - 测试时用 Fake 实现即可脱离真实模型与数据库跑通链路。

命名约定：接口以 Port 结尾，实现类不加该后缀（如 BgeEmbedder 实现 EmbedderPort）。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    Chunk,
    Conversation,
    ConversationSummary,
    Document,
    Page,
    RetrievedChunk,
    StoredMessage,
    User,
)


# =============================================================================
# 入库链路
# =============================================================================


@runtime_checkable
class DocumentParserPort(Protocol):
    """文档解析：文件 → 带页码的文本页列表。"""

    supported_extensions: tuple[str, ...]

    def parse(self, file_path: str) -> list[Page]:
        """把磁盘文件解析为页列表。

        Args:
            file_path: 本地文件绝对路径

        Returns:
            Page 列表；无页码格式统一记为第 1 页。

        Raises:
            ValidationError: 扩展名不受支持
            AppError: 解析失败（文件损坏、加密等）
        """
        ...


@runtime_checkable
class SplitterPort(Protocol):
    """文本切分：长文本 → 带页码溯源的片段列表。"""

    def split(self, pages: list[Page]) -> list[Chunk]:
        """按配置的块大小与重叠切分，并回填每块覆盖的页码范围。

        Args:
            pages: 解析得到的页列表

        Returns:
            片段列表（chunk_index 已按顺序填好）。

        注意：
            中文场景块过大会显著稀释相似度与重排得分，
            默认 CHUNK_SIZE 建议 150~300 字符。
        """
        ...


@runtime_checkable
class EmbedderPort(Protocol):
    """向量化：文本 → 稠密向量。"""

    dimension: int
    """向量维度。必须是**模型实际输出**的维度（如 BGE 权重 config.json 的
    hidden_size），而不是配置里写的期望值——建 collection 时以它为准。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """编码文档块，**不加**检索指令前缀。

        批大小属于实现细节：调用方直接传整份列表即可，
        分批与限流由实现自己决定。

        Returns:
            与输入等长的向量列表，已做 L2 归一化（余弦检索必需）；
            输入为空列表时返回空列表。

        Raises:
            ValidationError: 存在空/纯空白文本——BGE 对空串不会报错，
                而是返回一条无语义的 padding 向量，静默污染索引。
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """编码用户查询，**加**检索指令前缀（BGE 中文指令）。

        Returns:
            单条向量，已做 L2 归一化。

        注意：
            query 与 passage 若走了不同的编码路径，会落到不同语义空间，
            相似度整体偏移。务必分开调用本方法与 embed_documents()。
        """
        ...


@runtime_checkable
class VectorStorePort(Protocol):
    """向量库：片段的写入、检索与删除。"""

    def ensure_collection(self, dimension: int) -> None:
        """确保 collection 与索引存在；维度不符时应显式报错而非静默重建。"""
        ...

    def upsert(self, chunks: list[Chunk]) -> int:
        """写入或覆盖一批片段（要求 embedding 非空）。

        Returns:
            实际写入条数。
        """
        ...

    def search(
        self,
        kb_id: str,
        query_embedding: list[float],
        top_k: int,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """在指定知识库内做**稠密向量**检索。

        Args:
            kb_id: 知识库 ID，作为过滤条件
            query_embedding: 查询向量
            top_k: 返回条数上限
            min_score: 余弦相似度下限，None 时用 settings.min_similarity

        Returns:
            按相似度降序的结果列表，vector_score 已填。
        """
        ...

    def search_bm25(
        self,
        kb_id: str,
        query: str,
        top_k: int,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """在指定知识库内做**BM25 关键词**检索（混合检索的第二路召回）。

        与 `search()` 是并列的两路召回，结果各自排序后由
        `app.rag.fusion.fuse` 用 RRF 融合——本方法不做融合。

        Args:
            kb_id: 知识库 ID，作为过滤条件
            query: 查询原文（非向量）；分词由存储层在服务端完成
            top_k: 返回条数上限
            min_score: BM25 分下限。None 表示不过滤——BM25 分无界且
                随语料漂移，默认不该设阈值，精度交给重排阶段。

        Returns:
            按 BM25 分降序的结果列表，bm25_score 已填；无命中返回空列表。

        约定：
            底层不支持全文检索的实现应返回空列表（并自行记日志），
            让上层退化成纯向量检索，而不是把整个问答链路打断。
        """
        ...

    def delete_by_doc(self, doc_id: str) -> int:
        """删除某文档的全部片段。"""
        ...

    def delete_by_kb(self, kb_id: str) -> int:
        """删除某知识库的全部片段。"""
        ...


# =============================================================================
# 问答链路
# =============================================================================


@runtime_checkable
class RerankerPort(Protocol):
    """重排：对初检结果做交叉编码器精排。"""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """按 (query, passage) 相关性重排并截断。

        Args:
            query: 用户问题原文
            candidates: 向量/混合检索召回结果
            top_n: 返回条数上限
            min_score: 原始 logits 下限（交叉编码器输出可为负）

        Returns:
            重排后的结果，rerank_score 已回填。
        """
        ...


@runtime_checkable
class LLMPort(Protocol):
    """大模型：一次性生成与流式生成。

    通过 OpenAI 兼容协议接入，默认指向本机 Ollama /v1，
    不绑定任何云端厂商 SDK。
    """

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """一次性生成完整回答。"""
        ...

    def stream(self, system_prompt: str, user_prompt: str):
        """流式生成，逐段 yield 文本增量（async generator）。"""
        ...


# =============================================================================
# 持久化
# =============================================================================


@runtime_checkable
class ChunkRepositoryPort(Protocol):
    """知识库与文档元数据的仓储接口（SQLite 实现）。

    只负责元数据；片段正文与向量不经过本接口。
    """

    def list_knowledge_bases(self, owner_id: str) -> list[dict]:
        """列出该账号名下的知识库（别人的一条都不会出现）。"""
        ...

    def get_knowledge_base(self, kb_id: str, owner_id: str) -> dict | None:
        """按 ID 取单个知识库；不存在**或不属于该账号**返回 None。

        `owner_id` 是**必传**参数而非可选：给默认值意味着某天漏传时
        会静默退化成"谁都能读"，而这种退化成不了任何错误日志。
        """
        ...

    def create_knowledge_base(self, name: str, description: str, owner_id: str) -> dict:
        """创建知识库并归属于 owner_id。"""
        ...

    def update_knowledge_base(
        self, kb_id: str, name: str, description: str, owner_id: str
    ) -> dict:
        """更新知识库名称与描述（仅限本人名下）。

        Raises:
            NotFoundError: kb_id 不存在**或不属于该账号**（同一句话，防枚举）
        """
        ...

    def delete_knowledge_base(self, kb_id: str, owner_id: str) -> bool:
        """删除知识库（仅限本人名下）；不存在或非本人返回 False。"""
        ...

    def list_documents(self, kb_id: str) -> list[Document]:
        """列出某知识库下的全部文档（含各状态）。

        **不做归属校验**：传入的 kb_id 必须已经由服务层
        （`ensure_kb_owned` / `CollectionService.ensure_exists`）验过。
        本方法拿到的 kb_id 若来自未校验的输入，越权方就能列出别人的文档。

        为什么不用 `owner_id` 参数：文档的归属完全由它所在的库决定，
        在仓储层再查一次库表属于重复校验，且会让"库属于谁"这件事
        出现两个判断点——两处不一致时以哪处为准没有答案。
        """
        ...

    def get_document(self, doc_id: str) -> Document | None:
        """按 ID 取单个文档；不存在返回 None。

        重试链路（`IngestionPipeline.reprocess`）靠它拿回 `kb_id` 与
        `filename`：原始文件的落盘路径是
        `uploads/{kb_id}/{doc_id}_{filename}`，只凭 doc_id 猜不出后两段。
        """
        ...

    def create_document(self, kb_id: str, filename: str, content_hash: str = "") -> Document:
        """登记文档（status=processing）。

        `content_hash` 是正文 SHA-256，用于阻止同一份内容重复入库；
        空串表示调用方不参与去重（历史数据/脚本）。
        """
        ...

    def find_document_by_hash(self, kb_id: str, content_hash: str) -> Document | None:
        """按正文 hash 查一个已成功入库的文档；没有则 None。"""
        ...

    def mark_document_ready(self, doc_id: str, chunk_count: int) -> None:
        """入库成功后落终态：status=ready，记录切片数。

        `chunk_count` 是**实际写入的片段数**，不是估算值——
        前端拿它展示"这篇文档被切成几段"，检索出问题时这是第一个要看的数。
        片段为 0 不该记成 ready（那多半是解析出空文本），由调用方判断。
        """
        ...

    def mark_document_failed(self, doc_id: str, error: str) -> None:
        """入库失败后落终态：status=failed，并把失败原因原样存下来。

        `error` 会直接展示给用户，因此要写成**人能看懂的一句话**
        （"PDF 已加密，无法读取"而不是 `PDFEncryptionError: 0x4`）。
        原始异常类型该进日志，不该进这个字段。

        记录必须保留而不是删掉：用户要能看到"我传过这份文件、它挂了、
        为什么挂"，这是"原地重试"入口存在的前提。
        """
        ...

    def mark_document_processing(self, doc_id: str) -> None:
        """把文档重新置为处理中（重试入口用）。

        实现要**同时清空 `error` 与 `chunk_count`**，否则会留下
        「processing 但带着上次失败原因」的自相矛盾行。
        """
        ...

    def delete_document(self, doc_id: str) -> str | None:
        """删除文档元数据。

        Returns:
            被删除文档所属的 kb_id；不存在则 None。
        """
        ...


@runtime_checkable
class UserRepositoryPort(Protocol):
    """账号仓储接口（SQLite 实现）。

    与 ChunkRepositoryPort 分开而不是合二为一：两者的生命周期与权限边界不同
    （知识库是业务数据，账号是身份数据），将来账号可能落到另一张库甚至外部
    IdP，合在一个仓储里会变成"改账号要动检索的依赖"。

    关于 password_hash：仓储**只负责存取**，不做哈希也不做校验——
    哈希算法属于安全实现细节（infrastructure/security.py），
    混进仓储会让"换算法"变成"改数据访问层"。
    """

    def create_user(
        self, username: str, password_hash: str, display_name: str = ""
    ) -> User:
        """新建账号并返回。

        Args:
            username: 已规范化（小写、去空白）的登录名
            password_hash: 已算好的口令哈希，仓储不解释其内容

        Raises:
            ConflictError: 登录名已存在（唯一约束，必须由存储层保证——
                靠"先查再写"挡不住并发注册）
        """
        ...

    def get_user_by_username(self, username: str) -> User | None:
        """按登录名取账号；不存在返回 None。"""
        ...

    def get_user_by_id(self, user_id: str) -> User | None:
        """按 ID 取账号；不存在返回 None。

        令牌里存的是 id 而不是 username：登录名可以被改，
        而令牌一旦签发就无法回收，用可变标识做主体会留下旧令牌指向新用户的缝。
        """
        ...

    def list_users(self) -> list[User]:
        """列出全部账号（供运维脚本与自检使用，按创建时间排序）。"""
        ...

    def update_password_hash(self, user_id: str, password_hash: str) -> bool:
        """覆盖某账号的口令哈希。

        Returns:
            是否命中（False 表示账号不存在）。

        只提供"覆盖"而不提供"读取"：哈希一旦写进去就没有任何业务需要再读出来，
        少一个出口就少一条泄露路径。
        """
        ...


# =============================================================================
# 多轮对话存储（Redis）
# =============================================================================


@runtime_checkable
class ConversationStorePort(Protocol):
    """多轮对话历史的持久化接口。

    与 ChunkRepositoryPort 的分工：
        元数据库（SQLite）只存「库 / 文档 / 片段」的目录事实，不存对话；
        会话是另一类数据——按会话 ID 聚合、整轮追加、带 TTL 过期，
        用 Redis 的 list + hash + zset 表达比关系表省事得多。

    约定：
        - 会话 ID 由调用方（问答接口）生成或透传，本接口不生成；
        - 「会话不存在」统一抛 NotFoundError，「存储不可用」统一抛
          UpstreamError——错误语义只在实现层出现一次；
        - 时间字符串格式固定 "YYYY-MM-DD HH:mm:ss"（前端历史页按前 10 位
          分组，格式属于契约的一部分）。
    """

    def append_turn(
        self,
        conversation_id: str,
        kb_id: str,
        user_message: "StoredMessage",
        assistant_message: "StoredMessage",
        user_id: str = "",
    ) -> "ConversationSummary":
        """把一轮问答（用户消息 + 助手消息）追加进会话；会话不存在则创建。

        Args:
            conversation_id: 会话 ID
            kb_id: 本轮问答针对的知识库（同一会话换库继续问时以最后一次为准）
            user_message: 用户消息
            assistant_message: 助手消息（citations 带文档名 + 页码）
            user_id: 会话归属账号。**只在首轮生效**，续聊不会覆盖已有归属——
                否则一次越权请求就能把别人的会话过继给自己。

        Returns:
            更新后的会话列表项（标题、计数、时间戳都已落库）。

        Raises:
            ValidationError: 会话 ID 或消息为空
            UpstreamError: Redis 不可用
        """
        ...

    def list_conversations(
        self,
        kb_id: str | None = None,
        keyword: str | None = None,
        user_id: str | None = None,
    ) -> list["ConversationSummary"]:
        """列出会话（按最近更新倒序）。

        Args:
            kb_id: 只列该知识库的会话；None 列全部
            keyword: 标题包含的关键词（不区分大小写）；None 不过滤
            user_id: 只列该账号的会话；None **不按归属过滤**（仅运维脚本）。
                HTTP 层必须传，因为会话正文就是用户问过的内容。

        Raises:
            UpstreamError: Redis 不可用
        """
        ...

    def get_conversation(
        self, conversation_id: str, user_id: str | None = None
    ) -> "Conversation":
        """取会话全量（含消息序列，按时间正序）。

        Raises:
            NotFoundError: 会话不存在、已过期，**或不属于 user_id**
                （两种情况同一句话，避免用 ID 枚举别人的会话）
            UpstreamError: Redis 不可用
        """
        ...

    def conversation_exists(self, conversation_id: str) -> bool:
        """会话是否仍存在（过期即不存在；供问答链路判断要不要按新会话处理）。"""
        ...

    def delete_conversation(self, conversation_id: str, user_id: str | None = None) -> None:
        """删除会话。

        Args:
            user_id: 归属过滤。None 表示不校验（级联清理与运维脚本用）；
                HTTP 层传当前账号，不符抛 NotFoundError。

        会话不存在时**静默成功**（删除是幂等的）——但"不属于你"必须报错，
        否则删别人的会话会返回 204，等于确认了它存在。

        Raises:
            NotFoundError: user_id 非 None 且会话不属于该账号
            UpstreamError: Redis 不可用
        """
        ...

    def delete_by_kb(self, kb_id: str) -> int:
        """删除某知识库下的全部会话（含消息与索引项）。

        存在的理由：知识库是会话的**拥有者**。删库时若不连带删会话，
        这些会话会变成「列表里看得见、点进去 kb_id 已不存在」的孤儿——
        用户看到的是报错或空白页，且无法自行清理（删会话的入口在会话页，
        而会话页本身可能因知识库缺失而打不开）。

        与 `delete_conversation` 一样是幂等的：该知识库没有会话时返回 0。

        Returns:
            实际删除的会话数。

        Raises:
            UpstreamError: Redis 不可用
        """
        ...
