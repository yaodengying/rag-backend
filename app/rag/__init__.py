"""编排层：LangGraph 状态图与入库链。

本层回答「一次问答/一次入库，按什么顺序经过哪些步骤」，
不关心任何具体实现（那是 infrastructure 的事）。

    state.py     RAGState：图的状态定义，节点间的唯一数据契约
    nodes.py     各节点实现（检索/重排/生成/自检/拒答）
    graph.py     StateGraph 装配与条件边（含「检索不足 → 改写重试」回路）
    prompts.py   提示词集中管理（防幻觉约束写在这里）
    ingest.py    入库链：解析 → 切分 → 向量化 → 落库

为什么用图而不是一条直线链：
    真实 RAG 需要条件分支与回路——检索结果太差要改写查询重试、
    资料不足要拒答而不是硬编。LangGraph 把这些控制流显式建模成节点与边，
    可调试、可观测、可暂停（未来可加人工审核节点）。
"""

from .graph import build_rag_graph
from .ingest import IngestionPipeline
from .state import RAGState

__all__ = ["build_rag_graph", "IngestionPipeline", "RAGState"]
