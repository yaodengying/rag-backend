# -*- coding: utf-8 -*-
"""在进程内跑真实问答链路，收集每一题的预测。

为什么是"进程内"而不是调 HTTP：
    评估要拿到**送给模型的上下文原文**（`ChatResult.sources[*].chunk.text`）。
    走 `/chat` 只有 `snippet`（截断到 400 字）与文件名/页码，够溯源不够判分：
    faithfulness 要判断回答里的每句话能否被上下文支持，截断的片段会导致误判。
    进程内直调还省掉 SSE 编码与网络往返，18 题的批量评估更稳。

为什么**不用** `get_chat_service()`：
    容器里那一份挂载了 Redis 会话存储。评估是只读地"考"系统，
    不该在用户的会话列表里留下 18 条测试记录。这里显式传
    `conversation_store=None`，既不改动生产行为，也不污染数据。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.core.exceptions import AppError
from app.domain.models import Citation
from app.services.chat_service import ChatResult, ChatService
from scripts.rag_eval.dataset import EvalSample

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Prediction:
    """一条样本的预测结果。

    Attributes:
        sample: 原始样本（含标准答案与期望来源）
        answer: 模型回答原文；失败时为空
        contexts: 送进模型的上下文原文，按名次排列
        citations: 对外返回的引用（含文件名 + 页码）
        refused: 是否拒答
        latency_ms: 本题端到端耗时（检索 + 重排 + 生成）
        error: 失败原因；非空表示这题没跑出结果
        usage: 上下文用量（输入 token / 丢弃条数 / 利用率）。
            来自 ChatResult.usage，用于 skill（rag-context-eval）的系统层指标：
            只看质量不看 token，上线后成本会失控
        truncated: 回答是否被输出上限截断（true 时该题的忠实度/完整度不可信）
    """

    sample: EvalSample
    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    latency_ms: int = 0
    error: str = ""
    usage: dict | None = None
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return not self.error


class RagPipeline:
    """问答链路的评估夹具（延迟构建，只建一次）。"""

    def __init__(self) -> None:
        self._service: ChatService | None = None

    @property
    def service(self) -> ChatService:
        if self._service is None:
            # 延迟到第一次真正提问再装配：构建会加载 BGE 与 reranker 权重，
            # 而 `--help` / 数据集校验这些路径不该等模型加载。
            from app.core.container import get_llm, get_rag_graph, get_retrieval_graph

            self._service = ChatService(
                graph=get_rag_graph(),
                llm=get_llm(),
                retrieval_graph=get_retrieval_graph(),
                conversation_store=None,
            )
            logger.info("问答链路已装配（无会话存储，评估不落库）")
        return self._service

    def run(self, sample: EvalSample) -> Prediction:
        """对一条样本跑一次完整问答。

        失败不抛出：把原因记在 `Prediction.error` 上继续跑下一题。
        评估是批量任务，第 7 题炸掉不该让前 6 题的结果一起丢。
        """
        started = time.perf_counter()
        try:
            result: ChatResult = self.service.ask(
                kb_id=sample.kb_id,
                question=sample.question,
            )
        except AppError as exc:
            # 已知业务错误（检索故障 / 上游不可用 / 参数非法）
            latency = int((time.perf_counter() - started) * 1000)
            logger.warning("[%s] 失败：%s", sample.id, exc)
            return Prediction(
                sample=sample, latency_ms=latency, error=f"{type(exc).__name__}: {exc}"
            )
        except Exception as exc:  # noqa: BLE001 - 未知异常也要落到样本上
            latency = int((time.perf_counter() - started) * 1000)
            logger.exception("[%s] 未预期异常", sample.id)
            return Prediction(
                sample=sample,
                latency_ms=latency,
                error=f"{type(exc).__name__}: {exc}",
            )

        latency = int((time.perf_counter() - started) * 1000)
        return Prediction(
            sample=sample,
            answer=result.answer.text,
            contexts=[item.chunk.text for item in result.sources],
            citations=list(result.answer.citations),
            refused=result.answer.refused,
            latency_ms=latency,
            usage=result.usage,
            truncated=result.truncated,
        )
