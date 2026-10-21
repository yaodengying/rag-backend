# -*- coding: utf-8 -*-
"""查询改写：把「检索不到时的二次尝试」收敛成**唯一一份**实现。

为什么要单独一个模块（而不是留在节点里）：
    改写有**两个消费者**——图里的 `rewrite_query` 节点（非流式路径）
    与 `ChatService.ask_stream` 的可选重试（流式路径，按请求开启）。
    提示词、失败降级、"改写是否无效"的判定各写一遍，两边必然漂移，
    典型症状是「非流式改写失败会沿用原问题、流式却直接报错」——
    而这种偏差只在 LLM 抽风时暴露，是最难查的一类。
    所以本模块是改写语义的唯一实现，节点与服务都调它。

  与 `rag/prompts.py` 的分工：
    prompts 只负责「文案怎么写」（纯字符串模板）；
    本模块负责「怎么用、失败了怎么办」（调用 + 降级 + 判定）。

为什么降级是「原样返回 question」而不是抛错：
    改写用的也是 LLM，而"检索不到"的常见原因之一恰恰是上游模型不稳。
    若改写失败就抛错，一次检索失败会被升级成整个请求 500，
    而用户的真实处境只是"没找到"。返回原问题让调用方的重试回路
    自然终止（第二轮结果与第一轮一致 → 走拒答），
    语义与"确实没找到"完全一致。
"""

from __future__ import annotations

import logging

from app.core.exceptions import AppError
from app.domain.ports import LLMPort
from app.rag.prompts import build_rewrite_prompts

logger = logging.getLogger(__name__)


def rewrite_query(llm: LLMPort, question: str, *, attempt: int | None = None) -> str:
    """用 LLM 把问题改写成更适合检索的查询。

    Args:
        llm: 大模型客户端
        question: 用户原始问题
        attempt: 第几次改写，仅用于日志（让"改写了几轮"在日志里可数）

    Returns:
        改写后的查询；**改写失败或产出无效时原样返回 question**。
        调用方只需比较返回值与入参是否相同，就能判断本次改写是否有产出——
        不必再去解析异常或比对日志。

    "产出无效"的两种情形（都返回原问题）：
        - 模型返回空串：上游超时或被截断时的常见表现；
        - 模型原样抄回问题：对检索毫无改变，再跑一遍检索纯属浪费。

    只捕获 `AppError`（LLM 客户端自己抛的领域异常）。其余异常放行——
    那属于客户端适配器的 bug，不该在这里被伪装成"没找到"。
    """
    try:
        system, user = build_rewrite_prompts(question)
        rewritten = llm.generate(system, user).strip()
    except AppError as exc:
        logger.warning("查询改写失败，沿用原问题：%s", exc)
        return question

    if not rewritten or rewritten == question:
        logger.info("改写无有效产出（空串或与原问题相同），沿用原问题")
        return question

    label = f"（第 {attempt} 次）" if attempt else ""
    logger.info("查询改写%s：%r → %r", label, question, rewritten)
    return rewritten


__all__ = ["rewrite_query"]
