"""领域模型：项目内部流转的核心数据结构。

这些是「内部模型」，与对外的 Pydantic DTO（app/schemas）刻意分开：
    - schemas 负责 HTTP 契约与校验，可随接口演进；
    - domain 模型负责业务语义，保持稳定，不被传输格式绑架。
两者之间由 services 层做显式转换（to_dto / from_dto）。

本模块只允许标准库依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# =============================================================================
# 文档与切块
# =============================================================================


@dataclass(slots=True)
class Page:
    """文档解析出的一页（或一个逻辑段落块）。

    Attributes:
        page_no: 页码，1-based；对无页码格式（txt/md/docx）恒为 1；
            **None 表示「该页存在页码元数据但无法解析」**——此时不猜一个
            数字，而是如实标记未知，引用溯源里不显示页码（见 parsing.py）。
        text: 该页纯文本
    """

    page_no: int | None
    text: str


@dataclass(slots=True)
class Chunk:
    """切分后的知识片段，是入库与检索的最小单位。

    Attributes:
        id: 片段主键（UUID）
        kb_id: 所属知识库
        doc_id: 所属文档
        text: 片段正文
        chunk_index: 在文档内的序号，0-based，用于展示顺序
        page_start: 起始页码，未知为 None
        page_end: 结束页码（含），未知为 None
        filename: 来源文件名，便于回答中引用
        embedding: 向量；入库前为 None，检索命中后必有值
        metadata: 其他附加信息，透传给上层
    """

    id: str
    kb_id: str
    doc_id: str
    text: str
    chunk_index: int = 0
    page_start: int | None = None
    page_end: int | None = None
    filename: str = ""
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_label(self) -> str | None:
        """把页码区间渲染成可读标签（如「第3页」「第3-5页」）。

        页码的三种状态各有对应的呈现方式，不能混为一谈：
            page_start 为 None        → 返回 None（未知，前端不显示页码）
            单页                       → "第3页"
            多页（按页切分时不会出现，但格式上仍要支持） → "第3-5页"

        放在领域模型上而不是各处的工具函数里，是因为它被三个层次同时需要：
            - `rag/prompts.py` 要在参考资料里标注来源（在 services 之下，
              不能反向 import services）；
            - `services/mappers.py` 要在接口 DTO 里给出 page；
            - 引用溯源展示。
        分层依赖是单向的（services → rag → domain），所以实现只能落在 domain。
        """
        if self.page_start is None:
            return None
        if self.page_end is None or self.page_end == self.page_start:
            return f"第{self.page_start}页"
        return f"第{self.page_start}-{self.page_end}页"


@dataclass(slots=True)
class RetrievedChunk:
    """检索结果：片段 + 各阶段得分。

    Attributes:
        chunk: 命中的片段
        vector_score: 向量余弦相似度
        bm25_score: BM25 分（未启用混合检索时为 None）
        fusion_score: RRF 融合分
        rerank_score: 交叉编码器重排分（未启用重排时为 None）
    """

    chunk: Chunk
    vector_score: float | None = None
    bm25_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None


@dataclass(slots=True)
class Citation:
    """回答中引用的来源，用于前端展示溯源信息。"""

    doc_id: str
    filename: str
    page: str | None
    snippet: str
    rerank_score: float | None = None


@dataclass(slots=True)
class Answer:
    """问答结果。

    Attributes:
        text: 模型生成的回答；拒答时为空串
        citations: 引用来源列表
        refused: 是否因资料不足而拒答
    """

    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False


# =============================================================================
# 元数据实体（对应 SQLite 表）
# =============================================================================


@dataclass(slots=True)
class KnowledgeBase:
    """知识库。"""

    id: str
    name: str
    description: str = ""
    doc_count: int = 0
    created_at: datetime | None = None


@dataclass(slots=True)
class Document:
    """文档元数据。

    注意：SQLite 只存元数据，**不存**片段正文与向量；
    正文切片与向量分别落在 uploads 目录与 Milvus。

    Attributes:
        id: 文档主键（UUID）
        kb_id: 所属知识库
        filename: 原始文件名
        status: 处理状态
        chunk_count: 成功入库的片段数
        error: 失败原因（status=failed 时有值）
    """

    id: str
    kb_id: str
    filename: str
    status: str = "processing"
    chunk_count: int = 0
    error: str = ""
    created_at: datetime | None = None


# 文档处理状态取值
DOC_STATUS_PROCESSING = "processing"
DOC_STATUS_READY = "ready"
DOC_STATUS_FAILED = "failed"


# =============================================================================
# 身份
# =============================================================================


@dataclass(slots=True)
class User:
    """账号。

    Attributes:
        id: 账号主键（UUID）
        username: 登录名。**统一按小写规范化后存储**（见 auth_service）——
            否则 SQLite 里 `Admin` 与 `admin` 会是两个账号，而用户认为它们是同一个。
        password_hash: 口令哈希（自描述格式，含算法/迭代次数/随机盐）。
            `repr=False` 是硬要求：数据类默认的 repr 会把字段打进异常栈与日志，
            一次 `logger.info("%s", user)` 就把全库口令哈希泄出去了。
        display_name: 展示名，可为空（前端回落显示 username）
        is_active: 是否启用。停用不改密码，只挡登录
        created_at: 创建时间（UTC）
    """

    id: str
    username: str
    password_hash: str = field(default="", repr=False)
    display_name: str = ""
    is_active: bool = True
    created_at: datetime | None = None


# =============================================================================
# 多轮对话（Redis 会话存储）
# =============================================================================


@dataclass(slots=True)
class StoredMessage:
    """会话里的一条消息（Redis 持久化的最小单元）。

    Attributes:
        role: "user" 或 "assistant"
        content: 消息正文
        created_at: 本地时间字符串 "YYYY-MM-DD HH:mm:ss"——与前端历史页
            「按前 10 位分组」的口径一致，格式不能变。
        citations: assistant 消息的溯源来源（文档名 + 页码）；user 消息恒为空。
            存进会话而不只是即时返回，历史消息才能在换设备后仍可核对页码。
    """

    role: str
    content: str
    created_at: str = ""
    citations: list[Citation] = field(default_factory=list)


@dataclass(slots=True)
class ConversationSummary:
    """会话列表项。"""

    id: str
    title: str
    kb_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    message_count: int = 0
    citation_count: int = 0


@dataclass(slots=True)
class Conversation(ConversationSummary):
    """会话全量：列表项 + 消息序列。"""

    messages: list[StoredMessage] = field(default_factory=list)
