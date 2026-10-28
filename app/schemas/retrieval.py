"""检索相关 DTO。

这些结构的存在意义是**让检索可观测**：
    问答接口只回一个答案，看不出"是没召回到还是模型没答好"。
    把检索单独暴露成接口，并且把四个阶段的分数原样带出来，
    调参时才有据可依，而不是凭感觉改 CHUNK_SIZE。

分数语义（来自 `app.domain.models.RetrievedChunk`）：
    vector_score  余弦相似度，落在 [-1,1]，落在 [0,1] 是常态
    bm25_score    BM25 分，**无界**，只可用于同一查询内的相对比较
    fusion_score  加权 RRF 分，只在同一查询内可比；权重整体放大不影响排序
    rerank_score  交叉编码器原始 logits，可为负（实测正负相差约 15~18）
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.config import RERANK_K_HARD_LIMIT, TOP_K_HARD_LIMIT


class RetrievalRequest(BaseModel):
    """检索请求。"""

    query: str = Field(..., min_length=1, max_length=2000, description="查询文本")
    top_k: int | None = Field(
        None,
        ge=1,
        le=TOP_K_HARD_LIMIT,
        description="每路召回条数；不传则用服务端配置（RETRIEVE_TOP_K）。"
        "adaptive 模式下这是上限，不保证一定返回这么多条",
    )
    rerank_k: int | None = Field(
        None,
        ge=1,
        le=RERANK_K_HARD_LIMIT,
        description="重排后保留条数；不传则用服务端配置（RERANK_TOP_N）",
    )
    debug: bool = Field(
        False,
        description="是否附带两路召回的逐条明细（用于调参；数据量较大）",
    )


class RetrievedChunkOut(BaseModel):
    """单条检索结果。

    四个分数**允许缺省**：缺省表示"这一路没有召回它"，
    与"召回了但得 0 分"是两件事，不能混为一谈（见 `mappers.retrieved_to_dict`）。
    """

    chunk_id: str
    doc_id: str = ""
    filename: str = ""
    page: str | None = Field(None, description="形如「第3页」或「第3-5页」；未知为 null")
    chunk_index: int = 0
    text: str = Field("", description="片段正文预览（默认截断到 400 字）")
    vector_score: float | None = Field(None, description="余弦相似度")
    bm25_score: float | None = Field(None, description="BM25 分（无界，仅同查询内可比）")
    fusion_score: float | None = Field(None, description="加权 RRF 融合分（仅同查询内可比）")
    rerank_score: float | None = Field(None, description="交叉编码器原始 logits，可为负")


class RecallStageOut(BaseModel):
    """单路召回的阶段信息。"""

    channel: str = Field(..., description="dense（向量）或 bm25（关键词）")
    count: int = 0
    results: list[RetrievedChunkOut] = Field(default_factory=list)


class RetrievalDebugOut(BaseModel):
    """调参用的中间态快照。"""

    hybrid_enabled: bool = Field(..., description="本次是否跑了 BM25 路")
    dense: RecallStageOut
    bm25: RecallStageOut
    fused: list[RetrievedChunkOut] = Field(default_factory=list)


class RetrievalResponse(BaseModel):
    """检索响应。

    results 是**最终结果**（已重排并按相关性降序）；
    debug 为空时表示未请求明细。
    """

    query: str
    kb_id: str
    count: int = 0
    reranked: bool = Field(False, description="是否经过交叉编码器重排")
    results: list[RetrievedChunkOut] = Field(default_factory=list)
    debug: RetrievalDebugOut | None = None
