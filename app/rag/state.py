"""LangGraph 图状态定义。

状态是节点之间**唯一**的数据契约：
    每个节点接收完整 state，返回需要更新的字段（增量合并语义）。
    新增字段务必给默认值，否则图的首次执行会 KeyError。

用 TypedDict 而不是 dataclass 的原因：LangGraph 原生按 TypedDict 做
字段级合并，且 Annotated + reducer 可以自定义合并策略（如列表累加）。
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict


def _keep_last(_old: Any, new: Any) -> Any:
    """默认 reducer：后写覆盖先写。"""
    return new


class RAGState(TypedDict, total=False):
    """一次问答在图内流转的完整状态。

    Attributes:
        kb_id: 目标知识库 ID（多库检索时改为 kb_ids: list[str]）
        question: 用户原始问题
        history: 多轮对话历史，元素形如 {"role": ..., "content": ...}
        top_k: 向量召回条数
        rerank_k: 重排后保留条数
        rewritten_question: 查询改写后的文本；未改写时等于 question
        candidates: 混合检索召回的候选片段
        ranked: 重排后的片段（送入 LLM 的最终上下文）
        usage: 本次上下文装箱的用量（token 数、丢了几条、利用率）；
            由 generate 节点写入，供服务层回传与日志观测
        answer: 模型生成的回答文本
        refused: 是否因资料不足而拒答
        retry_count: 改写重试次数，用于限制回路深度
        error: 链路中捕获的错误信息
    """

    # ---- 输入 ----
    kb_id: str
    question: str
    history: Annotated[list[dict[str, str]], _keep_last]
    top_k: int
    rerank_k: int

    # ---- 中间态 ----
    rewritten_question: Annotated[str, _keep_last]
    candidates: Annotated[list, _keep_last]
    ranked: Annotated[list, _keep_last]
    # 上下文装箱的用量快照（见 rag/budget.py 的 ContextPlan.as_usage）。
    # 用 dict 而不是 dataclass：状态要能被日志/trace 直接序列化，
    # 也不该把 "预算" 这个纯计算模块的对象带进图的生命周期里。
    usage: Annotated[dict, _keep_last]

    # ---- 输出 ----
    answer: Annotated[str, _keep_last]
    refused: Annotated[bool, _keep_last]

    # ---- 控制 ----
    retry_count: Annotated[int, _keep_last]
    error: Annotated[str | None, _keep_last]


# 最大改写重试次数：防止「检索不足 → 改写 → 仍不足」无限循环
MAX_RETRY = 1

# 资料不足时的固定拒答话术。
# 必须与 prompts.py 中系统提示里的话术保持一致，便于上层做字符串判定。
NO_ANSWER_TEXT = "无法从知识库中找到与该问题相关的答案。"
