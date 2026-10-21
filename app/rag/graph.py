# -*- coding: utf-8 -*-
"""LangGraph 状态图装配。

图的形状：

    START
      │
      ▼
    retrieve ──────────┐
      │                │
      ▼                │
    rerank             │
      │                │
      ├─ 有可用片段 ──► generate ──► END
      │
      └─ 无可用片段 ─┬─ 未超重试上限 ──► rewrite_query ──┐
                     │                                  │
                     └─ 已超重试上限 ──► refuse ──► END  │
                                                        │
                     ◄──────────────────────────────────┘
                              （回到 retrieve）

条件边（conditional edge）是这张图存在的意义：
    「检索不足 → 改写重试」这个回路用直线链表达不了。

为什么要两张图（`build_rag_graph` 与 `build_retrieval_graph`）：
    `/chat/stream` 要真的逐 token 推给前端，而 token 只能来自
    `llm.stream()`——图节点返回的是**最终字段值**，不是增量。
    所以流式走「检索子图 + 服务层直接流式生成」：
    `build_retrieval_graph` 复用同一批节点工厂（同一个 HybridRetriever、
    同一个 reranker 实例），保证"流式与非流式检索到的东西完全一致"。

    流式路径的改写重试**不在图里**，而在 `ChatService.ask_stream`
    （按请求用 enable_retry 开启）。原因不只是"生成要逐 token"：重试时
    还得**在改写发生的当下**往前端吐一个 retry 事件，告诉用户"在等什么"，
    而图是跑完才返回最终状态的，插不进这条时间线。
    但**次数与判据仍然是共用的**——服务层调的就是本模块的
    `route_after_rerank` 与 `state.MAX_RETRY`，不是另写一套。

为什么 `build_rag_graph` 的参数是 `retriever` 而不是 `embedder + vector_store`：
    骨架期写的是后者，实现时改了。`make_retrieve_node` 需要一个**已经装配好的**
    HybridRetriever，因为融合权重/阈值是配置的一部分；
    让图与接口各自 new 一个，就会出现"接口查到的和问答用的不是同一套检索"
    ——这类偏差只在生产环境、只在大库上才会暴露，是最难查的一种。

依赖通过参数注入而不是在节点内直接 import container：
    这样测试时可以传入 Fake 实现，无需真实模型与数据库
    （见 scripts/check_chat.py，它在不启动 Ollama 与 Milvus 的情况下
    覆盖了正常、拒答、改写重试三条路径）。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from app.domain.ports import LLMPort, RerankerPort
from app.rag.nodes import (
    make_generate_node,
    make_rerank_node,
    make_retrieve_node,
    make_rewrite_node,
    refuse_node,
)
from app.rag.retrieval import HybridRetriever
from app.rag.state import MAX_RETRY, RAGState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

# 节点名常量。用常量而不是字面量，是因为它们要在三处保持一致：
# add_node 的注册名、add_edge 的两端、以及 route 函数返回的字符串。
# 任何一处拼错，LangGraph 只会在编译期抛一个不指向"拼错了"的错误。
NODE_RETRIEVE = "retrieve"
NODE_RERANK = "rerank"
NODE_REWRITE = "rewrite_query"
NODE_GENERATE = "generate"
NODE_REFUSE = "refuse"


def build_rag_graph(
    retriever: HybridRetriever,
    reranker: RerankerPort | None,
    llm: LLMPort,
) -> "CompiledStateGraph":
    """构建并编译完整的问答状态图。

    Args:
        retriever: 已装配的混合检索器（与检索接口共用同一实例）
        reranker: 重排器；None 表示未启用重排
        llm: 大模型客户端

    Returns:
        编译后的图。可并发复用，应在进程内缓存而非每请求重建
        （见 container.get_rag_graph 的 lru_cache）。
    """
    graph = StateGraph(RAGState)

    graph.add_node(NODE_RETRIEVE, make_retrieve_node(retriever))
    graph.add_node(NODE_RERANK, make_rerank_node(reranker))
    graph.add_node(NODE_REWRITE, make_rewrite_node(llm))
    graph.add_node(NODE_GENERATE, make_generate_node(llm))
    graph.add_node(NODE_REFUSE, refuse_node)

    graph.add_edge(START, NODE_RETRIEVE)
    graph.add_edge(NODE_RETRIEVE, NODE_RERANK)
    graph.add_conditional_edges(
        NODE_RERANK,
        route_after_rerank,
        {
            NODE_GENERATE: NODE_GENERATE,
            NODE_REWRITE: NODE_REWRITE,
            NODE_REFUSE: NODE_REFUSE,
        },
    )
    # 重试回路：改写完回到检索。由 route_after_rerank 里的 retry_count
    # 判定终止，因此这个环不会无限转。
    graph.add_edge(NODE_REWRITE, NODE_RETRIEVE)
    graph.add_edge(NODE_GENERATE, END)
    graph.add_edge(NODE_REFUSE, END)

    compiled = graph.compile()
    logger.info(
        "问答图已编译：%s → %s → [条件边] generate | rewrite(≤%s 次) | refuse",
        NODE_RETRIEVE,
        NODE_RERANK,
        MAX_RETRY,
    )
    return compiled


def build_retrieval_graph(
    retriever: HybridRetriever,
    reranker: RerankerPort | None,
) -> "CompiledStateGraph":
    """构建并编译「只检索不生成」的子图：START → retrieve → rerank → END。

    供 `/chat/stream` 使用：先跑完检索拿到 ranked（推 sources 事件），
    再由服务层直接 `llm.stream()` 逐 token 生成。

    与完整图共用同一批节点工厂，所以两边的检索、重排行为一致；
    差别只在**没有生成节点**——因而也没有短路拒答，这一层由服务层
    就近处理（ranked 为空 → 按拒答话术推一条 delta）。
    改写重试同样在服务层，但它复用本模块的 `route_after_rerank`
    做判据与限次（见 ChatService.ask_stream 的 enable_retry）。
    """
    graph = StateGraph(RAGState)
    graph.add_node(NODE_RETRIEVE, make_retrieve_node(retriever))
    graph.add_node(NODE_RERANK, make_rerank_node(reranker))
    graph.add_edge(START, NODE_RETRIEVE)
    graph.add_edge(NODE_RETRIEVE, NODE_RERANK)
    graph.add_edge(NODE_RERANK, END)
    return graph.compile()


def route_after_rerank(state: dict) -> str:
    """重排后的路由决策。

    Returns:
        "generate"  —— 有可用片段
        "rewrite"   —— 无可用片段且 retry_count < MAX_RETRY
        "refuse"    —— 无可用片段且已达重试上限

    注意判据是 `ranked` 而**不是** `candidates`：
        「召回到了」与「召回的能用」是两件事。重排可能把全部候选都判为
        不相关（实测过：问题问的是巡检周期，召回的长块里混着报销与固件内容，
        logits 直接是负的）。此时 candidates 非空、ranked 为空，
        盲目拿 candidates 去生成，模型就会在一堆无关文字里"找"答案——
        这正是幻觉的高发场景。
    """
    if state.get("ranked"):
        return NODE_GENERATE
    if int(state.get("retry_count") or 0) < MAX_RETRY:
        return NODE_REWRITE
    return NODE_REFUSE


__all__ = [
    "NODE_GENERATE",
    "NODE_RERANK",
    "NODE_RETRIEVE",
    "NODE_REWRITE",
    "NODE_REFUSE",
    "build_rag_graph",
    "build_retrieval_graph",
    "route_after_rerank",
]
