"""检索路由：只检索不生成，用于验证召回质量与调参。

为什么值得单独开一个接口：
    问答接口（/chat）把检索、重排、生成串成一条链，只回一个答案。
    当答案不对时，它是黑盒——无法区分下面三种情况：
        ① 没召回到（分块 / 阈值 / 模型问题）
        ② 召回到了但被重排丢了（rerank_min_score 过高）
        ③ 都正常，是 LLM 没答好（提示词问题）
    本接口把①和②的结果连同四个阶段的分数原样暴露，
    一次请求就能把问题定位到具体环节。

与 /chat 的关系：
    两者共用同一个 `HybridRetriever`（见 container.get_hybrid_retriever），
    所以这里看到的结果与问答实际用的是同一套检索，不存在"调试口径不一致"。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.api.deps import get_collection_service, get_retrieval_service
from app.api.v1.auth import require_current_user, router_dependencies
from app.schemas.auth import UserOut
from app.schemas.retrieval import (
    RecallStageOut,
    RetrievalDebugOut,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunkOut,
)
from app.services.collection_service import CollectionService
from app.services.mappers import retrieved_to_dict
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=router_dependencies)


@router.post(
    "/knowledge-bases/{kb_id}/search",
    response_model=RetrievalResponse,
    summary="检索（向量 + BM25 混合，只检索不生成）",
)
async def search(
    kb_id: str,
    payload: RetrievalRequest,
    current_user: UserOut = Depends(require_current_user),
    retrieval: RetrievalService = Depends(get_retrieval_service),
    collections: CollectionService = Depends(get_collection_service),
) -> RetrievalResponse:
    """在指定知识库内检索，返回片段与各阶段得分。

    先校验知识库存在，再检索：知识库 ID 打错时给 404，
    比返回一个空的 results 更容易发现（"没有相关内容"与"库不存在"
    对使用者是两条完全不同的排查路径）。

    Raises:
        NotFoundError: 知识库不存在或不属于当前账号（404，两者同一句话）
        ValidationError: kb_id 或查询非法（400）
        RetrievalError: 向量库侧失败（502）
    """
    collections.ensure_exists(kb_id, current_user.id)

    # 一次跑完，同时拿到中间态：分两次调用会让 Embedding 前向与两路检索各做两遍
    trace, results = retrieval.retrieve_with_trace(
        kb_id,
        payload.query,
        top_k=payload.top_k,
        rerank_k=payload.rerank_k,
        owner_id=current_user.id,
    )

    debug = None
    if payload.debug:
        debug = RetrievalDebugOut(
            hybrid_enabled=trace.hybrid_enabled,
            dense=RecallStageOut(
                channel="dense",
                count=len(trace.vector_hits),
                results=[
                    RetrievedChunkOut.model_validate(retrieved_to_dict(item))
                    for item in trace.vector_hits
                ],
            ),
            bm25=RecallStageOut(
                channel="bm25",
                count=len(trace.bm25_hits),
                results=[
                    RetrievedChunkOut.model_validate(retrieved_to_dict(item))
                    for item in trace.bm25_hits
                ],
            ),
            fused=[
                RetrievedChunkOut.model_validate(retrieved_to_dict(item))
                for item in trace.fused
            ],
        )

    logger.info(
        "检索接口：kb=%s query=%r 向量=%s BM25=%s 融合=%s 最终=%s 重排=%s",
        kb_id,
        payload.query,
        len(trace.vector_hits),
        len(trace.bm25_hits),
        len(trace.fused),
        len(results),
        retrieval.reranker_enabled,
    )

    return RetrievalResponse(
        query=payload.query,
        kb_id=kb_id,
        count=len(results),
        reranked=retrieval.reranker_enabled,
        results=[
            RetrievedChunkOut.model_validate(retrieved_to_dict(item)) for item in results
        ],
        debug=debug,
    )
