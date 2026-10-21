"""检索服务：对外暴露「只检索不生成」的能力。

为什么单独拆一个服务：
    调试期的第一诉求是「看看召回了什么」。把检索独立出来，
    可以在不调用大模型的情况下验证切分、向量化、索引、阈值是否合理，
    排查问题时能快速定位是「检索没召回」还是「模型没答好」。

检索链路：
    向量召回 + BM25 召回 → RRF 融合 → 交叉编码器重排 → 截断 rerank_k

    第 1~2 步在 `app.rag.retrieval.HybridRetriever` 里（与 LangGraph 的
    retrieve 节点**共用同一份实现**），第 3~4 步在本服务。
    这样"接口查到的"与"问答用的"必然是同一套检索。

重要经验：
    检索不出结果，绝大多数时候不是模型问题，而是三个可量化的原因：
      1. 分块过大 —— 长块稀释相似度与重排得分（中文建议 150~300 字符）；
      2. 相似度阈值过高 —— 中文短查询对长块，余弦相似度本就偏低；
      3. 重排阈值过高 —— 交叉编码器输出是 logits，尺度与 [0,1] 完全不同。
    调参前先用 `recall()` 打印各阶段真实分数，不要凭感觉改。

    还有一个混合检索特有的坑：**BM25 分与余弦相似度不可比**，
    所以别去比较 "vector_score=0.62 与 bm25_score=2.5 谁更相关"——
    融合只看排名（见 `app/rag/fusion.py`）。
"""

from __future__ import annotations

import logging

from app.core.config import get_settings
from app.core.exceptions import NotFoundError
from app.domain.models import RetrievedChunk
from app.domain.ports import ChunkRepositoryPort, RerankerPort
from app.rag.retrieval import HybridRetriever, RecallTrace
from app.services.ownership import ensure_kb_owned

logger = logging.getLogger(__name__)


class RetrievalService:
    """检索服务：混合召回 + 可选重排。"""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: RerankerPort | None = None,
        repository: "ChunkRepositoryPort | None" = None,
    ) -> None:
        """注入检索器与重排器。

        Args:
            retriever: 混合检索器（两路召回 + RRF 融合）
            reranker: 重排器；None 表示未启用，此时退化为"按融合分截断"
            repository: 元数据仓储，**仅用于归属校验**（判断 kb_id 是否属于
                当前账号）。为 None 时 `owner_id` 必须也为 None（内部脚本路径）；
                容器装配时一定会传，HTTP 层因此总能拿到归属校验。
                这里收一个仓储而不是"让路由先查一次再传 kb 进来"，是为了
                把校验与使用放在同一个对象的同一次调用里——分成两步时，
                新增的调用方很容易只做第二步。

        说明：这里收的是已经装配好的 `HybridRetriever`，而不是
        embedder + vector_store 两个端口——因为 retrieve 节点也要用同一份
        检索逻辑，让两个消费者各自构造一个 HybridRetriever 会留下
        "两边配置不一致"的隐患（融合是一个可配置的复合对象，不是无状态工具）。
        """
        self._retriever = retriever
        self._reranker = reranker
        self._repository = repository
        self._rerank_top_n = int(get_settings().rerank_top_n)

    # ---------- 只读属性（供自检与日志读取）----------

    @property
    def reranker_enabled(self) -> bool:
        """是否启用了重排。"""
        return self._reranker is not None

    @property
    def retriever(self) -> HybridRetriever:
        """底层混合检索器（自检脚本要用它的 recall 看逐路分数）。"""
        return self._retriever

    # ---------- 主入口 ----------

    def retrieve(
        self,
        kb_id: str,
        query: str,
        top_k: int | None = None,
        rerank_k: int | None = None,
        owner_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """执行完整检索链路。

        Args:
            kb_id: 知识库 ID
            query: 查询文本
            top_k: 每路召回条数；None 时取 settings.retrieve_top_k
            rerank_k: 重排后保留条数；None 时取 settings.rerank_top_n
            owner_id: 当前登录账号 ID。**None 表示不做归属校验**，
                只用于内部脚本与自检（HTTP 层永远显式传入）。
                注意：检索本身不落任何数据，'不校验'的危害是读到别人的文档内容，
                所以生产路径必须传。

        Returns:
            按相关性降序的片段列表；无命中返回空列表。
            未启用重排时按 fusion_score 降序截断，此时 rerank_score 为 None。
        """
        return self.retrieve_with_trace(
            kb_id, query, top_k=top_k, rerank_k=rerank_k, owner_id=owner_id
        )[1]

    def retrieve_with_trace(
        self,
        kb_id: str,
        query: str,
        top_k: int | None = None,
        rerank_k: int | None = None,
        owner_id: str | None = None,
    ) -> tuple[RecallTrace, list[RetrievedChunk]]:
        """同 `retrieve()`，但同时把两路召回的原始明细一并返回。

        为什么不让调用方自己先调 `recall()` 再调 `retrieve()`：
            那会把查询向量化与两路检索**各做两遍**（一次 Embedding 前向
            + 两次 Milvus 往返），检索接口因此平白慢一倍。
            中间态就在这里一次性产出，谁需要谁拿走。

        Args:
            owner_id: 见 `retrieve()`。校验发生在**检索之前**——
                否则越权请求照样会去 Milvus 里查一遍，等于把别人的片段
                读进了进程内存，只是最后没返回而已。

        Returns:
            (召回过程快照, 最终结果列表)

        Raises:
            NotFoundError: owner_id 非 None 且知识库不属于该账号
        """
        self._ensure_owned(kb_id, owner_id)
        keep = self._rerank_top_n if rerank_k is None else int(rerank_k)
        trace = self._retriever.recall(kb_id, query, top_k=top_k)

        if keep <= 0 or not trace.fused:
            if not trace.fused:
                logger.info("检索无候选（kb_id=%s），跳过重排", kb_id)
            return trace, []

        if self._reranker is None:
            # 未启用重排：直接按融合分截断。fusion_score 由融合阶段统一给出，
            # 即使只跑了一路也有值，所以这里不必区分"纯向量"与"混合"。
            logger.info("未启用重排，按融合分截断取前 %s 条", keep)
            return trace, trace.fused[:keep]

        ranked = self._reranker.rerank(query, trace.fused, top_n=keep)
        logger.info(
            "重排完成：%s 条候选 → %s 条可用（top_n=%s）",
            len(trace.fused),
            len(ranked),
            keep,
        )
        return trace, ranked

    def recall(
        self,
        kb_id: str,
        query: str,
        top_k: int | None = None,
        owner_id: str | None = None,
    ) -> RecallTrace:
        """只跑到融合为止，返回两路原始结果与融合结果。

        用于调参：一眼看清是"向量路没召回"还是"BM25 路没召回"，
        以及融合把谁排到了前面。不涉及重排，因此不需要加载交叉编码器。

        Raises:
            NotFoundError: owner_id 非 None 且知识库不属于该账号
        """
        self._ensure_owned(kb_id, owner_id)
        return self._retriever.recall(kb_id, query, top_k=top_k)

    def _ensure_owned(self, kb_id: str, owner_id: str | None) -> None:
        """归属校验的转发口（实现见 `services/ownership.py`）。

        `owner_id` 为 None 时跳过校验（内部脚本与自检的逃生舱）；
        本服务未装配仓储却收到 owner_id 时按"查不到"拒绝。

        Raises:
            NotFoundError: 知识库不存在、不属于该账号，或本服务未装配仓储
        """
        ensure_kb_owned(self._repository, kb_id, owner_id)
