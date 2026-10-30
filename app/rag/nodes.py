# -*- coding: utf-8 -*-
"""LangGraph 节点实现。

每个节点是一个纯函数：接收 RAGState，返回需要更新的字段子集。
节点内**不做** I/O 装配（依赖由 build_rag_graph 通过闭包注入），
这样才能单测——节点逻辑与具体实现解耦（见 scripts/check_chat.py）。

节点清单：

    rewrite_query   查询改写（仅在首轮检索不足时触发）
    retrieve        混合检索：向量召回 + BM25 → RRF 融合
    rerank          交叉编码器精排，截断 top_n
    generate        组装提示词 → 调用 LLM → 生成回答
    refuse          资料不足时的短路拒答（不调用 LLM，省一次推理）

两条贯穿所有节点的约定：

  1. **查询口径统一**：读 `rewritten_question or question`。
     只有 rewrite 节点写 `rewritten_question`，所以"检索用的查询"
     与"重排/生成用的查询"必然是同一个值。分开传会是两类经典 bug：
     重排用了原问题、生成用了改写问题，或者三轮检索用了三个不同查询。

  2. **异常不吞**：节点只捕获它**明确知道怎么降级**的异常
     （改写失败 → 用原问题重试，降级实现见 `rag/rewrite.py`；检索无候选 → 空列表）。
     其余异常直接向上抛，由接口层统一转成 5xx——
     会把"没有找到答案"的一场错误让用户以为知识库里没这条，
     而实际上是系统坏了，这是最误导人的一种降级。

"""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import get_settings
from app.domain.ports import LLMPort, RerankerPort
from app.rag.prompts import build_answer_prompts_planned
from app.rag.retrieval import HybridRetriever
from app.rag.rewrite import rewrite_query
from app.rag.state import MAX_RETRY, NO_ANSWER_TEXT, RAGState
from app.rag.topk import RERANK_K_HARD_LIMIT, resolve_k

logger = logging.getLogger(__name__)


def _query_of(state: RAGState) -> str:
    """取本次链路统一的检索/生成查询（见模块文档第 1 条）。"""
    return str(state.get("rewritten_question") or state.get("question") or "").strip()


def make_retrieve_node(retriever: HybridRetriever):
    """构造检索节点（注入混合检索器）。

    Args:
        retriever: 与 RetrievalService 共用的同一个 `HybridRetriever`。
            注意不是 embedder + vector_store——那样两个消费者会各自
            构造一份，融合参数就可能悄悄不一致。

    Returns:
        符合 LangGraph 节点签名的可调用对象：
        (state: RAGState) -> dict[str, Any]

    逻辑要点：
        - 用 rewritten_question（若有）否则 question 作为检索查询；
        - 混合检索：向量召回 + BM25 召回，用 RRF 融合两路排名；
        - 结果写入 candidates；无命中时写空列表并记日志，
          由条件边决定是走改写重试还是直接拒答。
    """

    def retrieve(state: RAGState) -> dict[str, Any]:
        """混合检索节点：向量 + BM25 双路召回，RRF 融合后写入 `candidates`。

        返回的字段集合是 LangGraph 的**部分更新**语义——只写 `candidates`
        （以及入参不全时顺带把 `rewritten_question` 落成空查询），
        其余状态字段原样保留。节点千万不要整份 state 返回：
        那会把并发分支各自写的字段互相覆盖掉。

        入参不全（缺 kb_id 或查询为空）时**不抛异常**而是返回空候选：
        这类问题多半来自上游拼装错误，让条件边走"拒答"路径比 500 更合适——
        用户看到的是"资料不足"，日志里有 warning 可查。
        """
        kb_id = str(state.get("kb_id") or "")
        query = _query_of(state)
        top_k = state.get("top_k")

        if not kb_id or not query:
            logger.warning("检索节点入参不全：kb_id=%r query=%r，直接返回空候选", kb_id, query)
            return {"candidates": [], "rewritten_question": query}

        # top_k 由 state 覆盖（接口层可传），None 时用检索器自身的配置
        trace = retriever.recall(kb_id, query, top_k=top_k)
        if not trace.fused:
            logger.info("检索无候选：%s", trace.describe())

        return {"candidates": trace.fused, "rewritten_question": query}

    return retrieve


def make_rerank_node(reranker: RerankerPort | None):
    """构造重排节点。

    Args:
        reranker: 重排器；为 None 时退化为「按融合分截断 top_n」。

    Returns:
        节点可调用对象。

    为什么 reranker 为 None 时是"截断"而不是"报错"：
        `rerank_enabled=false` 是一个合法的部署选择（省掉交叉编码器的
        内存与延迟）。此时链路依然要能回答问题，只是排序精度降级为
        融合分。这与 embedding / vector_store 缺失不同——那两个是必需件。
    """

    def rerank(state: RAGState) -> dict[str, Any]:
        """精排节点：交叉编码器打分后截断，写入 `ranked`。

        三条降级路径，都不抛异常：
          1. 无候选 / `rerank_k <= 0` → `ranked` 为空（下游走拒答）；
          2. 未装配 reranker → 按 RRF 融合分直接截断（可用性优先于精度）；
          3. 整批最高分低于闸门 → 当作"一条都对不上"，`ranked` 清空。

        第 3 条与逐条阈值不是一回事：逐条只看分数正负，挡不住"字面高度
        重叠、语义不对"的片段（详见下方行内注释）。清空后仍把候选留在
        `candidates` 里不动——排查时要能看到"到底召回了什么、为什么被丢"。
        """
        query = _query_of(state)
        candidates = list(state.get("candidates") or [])
        # rerank_k 走统一解析口子：None/缺字段才用配置默认值，0 与负数原样
        # 保留（下游按 "keep <= 0" 返回空 → 拒答）。改造前这里是
        # `int(state.get("rerank_k") or 默认值)`，0 会被 `or` 悄悄换成 4，
        # 与 RetrievalService 里 `is None` 的判断给出相反结果——同一份配置、
        # 两条路径、两种行为，而且都没有日志。
        keep = resolve_k(
            state.get("rerank_k"),
            default=get_settings().rerank_top_n,
            hard_limit=RERANK_K_HARD_LIMIT,
        )

        if not candidates or keep <= 0:
            logger.info("重排跳过：候选 %s 条，rerank_k=%s", len(candidates), keep)
            return {"ranked": []}

        if reranker is None:
            logger.info("未启用重排，按融合分取前 %s 条", keep)
            return {"ranked": candidates[:keep]}

        ranked = reranker.rerank(query, candidates, top_n=keep)

        # 「整批最高分」闸门（未配置 RERANK_MIN_TOP_SCORE 时关闭）。
        #
        # 逐条闸门（RERANK_MIN_SCORE）只挡负分，挡不住这一类：
        # 问题与某个片段**字面高度重叠但语义不匹配**（问"备件能加急吗"，
        # 库里只有"备件 5 个工作日内发出"），交叉编码器会给正分，
        # 于是模型拿着一段周边规则硬答——这正是"检索幻觉"最常见的来源。
        # 整批最高分都很低，说明这一轮**整批都不对题**，宁可当作没找到。
        #
        # 注意只在 reranker 真的给了分数时才判：reranker=None 时分数全 None，
        # 这里若照判会把所有请求都拒掉。
        floor = get_settings().rerank_min_top_score
        if floor is not None and ranked:
            top = max(
                (item.rerank_score for item in ranked if item.rerank_score is not None),
                default=None,
            )
            if top is not None and top < floor:
                logger.info(
                    "重排最高分 %.3f 低于整批闸门 %.3f（%s 条候选），视为无可用片段",
                    top,
                    floor,
                    len(ranked),
                )
                return {"ranked": []}

        logger.info(
            "重排完成：%s 条候选 → %s 条可用（top_n=%s）；全部低于阈值时 ranked 为空，"
            "由条件边决定改写重试还是拒答",
            len(candidates),
            len(ranked),
            keep,
        )
        return {"ranked": ranked}

    return rerank


def make_rewrite_node(llm: LLMPort):
    """构造查询改写节点。

    仅在「检索/重排后无可用片段」且未超出 MAX_RETRY 时被条件边触发。

    改写本身（提示词 + 失败降级 + 无效判定）在 `rag/rewrite.py` 里实现，
    本节点只负责**图语义**那一层：读 `retry_count`、拦住超限、
    把结果写回 state。流式路径的可选重试调的是同一个 `rewrite_query`，
    所以两条路的改写行为不会各走各的。
    """

    def rewrite_query_node(state: RAGState) -> dict[str, Any]:
        """改写节点：把问题换一种说法，供下一轮检索使用。

        写 `rewritten_question` 与 `retry_count` 两个字段。
        `retry_count` 是**额度**而不是次数统计：条件边用它判断还能不能重试，
        所以每进一次就必须 +1，否则会原地打转。

        改写失败（LLM 超时、返回空、格式不对）时 `rewrite_query` 内部已兜底
        返回原问题，这里不做二次判断——第二轮检索结果不变，路由自然走向拒答，
        这比抛错让整轮问答失败对用户友好得多。
        """
        question = str(state.get("question") or "").strip()
        count = int(state.get("retry_count") or 0)

        if count >= MAX_RETRY:
            # 正常不会走到这里（条件边已判过），兜一层防止图被改坏后无限循环
            logger.warning("改写节点被调用但已达重试上限（%s），保持原查询", count)
            return {"retry_count": count}

        # 改写无效时拿到的是原问题：第二轮检索结果不变，路由随即走向拒答。
        rewritten = rewrite_query(llm, question, attempt=count + 1)
        return {"rewritten_question": rewritten, "retry_count": count + 1}

    return rewrite_query_node


def make_generate_node(llm: LLMPort):
    """构造生成节点。

    逻辑要点：
        - 用 prompts.build_answer_prompts 把 ranked 拼成参考资料；
        - 调用 llm.generate(system, user)；
        - 若模型回复里出现 NO_ANSWER_TEXT，置 refused=True。
    """

    def generate(state: RAGState) -> dict[str, Any]:
        """生成节点：拼提示词 → 调 LLM → 写 `answer` 与 `refused`。

        `refused` 不由"回答长短"或关键词猜测，而是看模型是否原样吐出了
        `NO_ANSWER_TEXT`——拒答话术由提示词约定，是**唯一**的拒答信号。

        收到空 `ranked` 时转短路拒答：条件边正常不会这么走，但图被改动后
        可能发生；让模型在没有资料的情况下自由发挥，是本项目最不能接受的
        失败模式（用户无法分辨那是模型编的还是文档里的）。
        """
        ranked = list(state.get("ranked") or [])
        question = str(state.get("question") or "")

        if not ranked:
            # 双保险：条件边正常情况下不会把空 ranked 送到这里。
            # 真送到这里说明图被改坏了，短路拒答好过让模型凭空作答。
            logger.warning("生成节点收到空上下文，转为短路拒答")
            return refuse_node(state)

        system, user, plan = build_answer_prompts_planned(
            ranked, question, history=state.get("history")
        )
        answer = llm.generate(system, user).strip()
        refused = _is_refusal(answer)
        logger.info(
            "生成完成：候选 %s 条 → 实际送入 %s 条（约 %s token，窗口利用率 %.1f%%），"
            "回答 %s 字，refused=%s",
            len(ranked),
            len(plan.evidence),
            plan.prompt_tokens,
            plan.utilization * 100,
            len(answer),
            refused,
        )
        # 回写 `ranked` 为**实际送入模型**的片段：预算丢掉的片段不应再出现在
        # citations / sources 里（否则用户会看到"引用了但模型根本没读过"）。
        # 拒答路径由条件边决定，不经过这里，所以不影响路由判据。
        return {
            "answer": answer,
            "refused": refused,
            "usage": plan.as_usage(),
            "ranked": list(plan.evidence),
        }

    return generate


def refuse_node(state: RAGState) -> dict[str, Any]:
    """短路拒答节点：不调用 LLM，直接返回固定话术。

    这样做的价值：无检索结果时调模型不但浪费推理，
    还给了模型「自由发挥」的机会，是幻觉的主要来源之一。

    注意 `retry_count` 一并回填：state 是增量合并的，
    拒答路径如果漏了它，图最终状态里就少了这个字段，
    排查"为什么没重试"时看不到关键证据。
    """
    logger.info(
        "短路拒答：无可用片段（kb=%s，已重试 %s 次）",
        state.get("kb_id"),
        state.get("retry_count") or 0,
    )
    return {
        "answer": NO_ANSWER_TEXT,
        "refused": True,
        "retry_count": int(state.get("retry_count") or 0),
    }


def _is_refusal(answer: str) -> bool:
    """判断模型是否拒答。

    判据是**包含**而不是**相等**：模型常给固定话术加上前后缀
    （"抱歉，无法从知识库中找到与该问题相关的答案。"）。
    要求逐字相等会让这些情况被判成"已作答"，用户看到一条
    明说找不到答案、却被系统标成正常回答的结果。

    反向的风险（回答里恰好包含这句话但实为作答）在实践中不存在——
    提示词明确要求"资料不足时原样回复该句"，模型不会在正文里引用它。
    """
    return NO_ANSWER_TEXT in (answer or "")
