# -*- coding: utf-8 -*-
"""把项目自己的能力接到 RAGAS 需要的 langchain 接口上。

RAGAS 的指标分两类需求：
    - 需要裁判模型（faithfulness / context_recall / context_precision / 问题反推）
    - 需要向量化（answer_relevancy 用"反推问题与原问题的余弦相似度"）

本模块只做适配，不做业务判断：
    - 用**同一份** `.env` 里的 LLM 端点当裁判，保证"评估的模型就是回答的模型"；
    - 用项目自己的 BGE（EmbedderPort）算相似度，而不是另下一套 embedding，
      否则 answer_relevancy 衡量的语义空间与检索用的不是同一个。
"""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings

from app.core.config import get_settings
from app.domain.ports import EmbedderPort
from scripts.rag_eval.compat import ensure_ragas_importable

logger = logging.getLogger(__name__)


class SentenceSimilarityEmbeddings(Embeddings):
    """用于「问句 ↔ 问句」相似度的向量化（RAGAS 的 answer_relevancy 用）。

    为什么两侧都走文档编码（**对称**编码）：
        BGE 的 `embed_query` 会加检索指令前缀（"为这个句子生成表示以用于检索
        相关文章："），那是为「短查询 vs 长段落」的非对称检索设计的。
        而 answer_relevancy 比较的是「用户原问题」与「模型反推出的问题」——
        两个都是问句，属于对称语义相似度任务，两侧应当同样编码。

        实测（4 题对照，同一批回答、同一裁判，只换编码方式）：
            非对称：q03=0.317  q08=0.777  q05=0.000  q14=0.925
            对称：  q03=  —    q08=0.797  q05=0.000  q14=0.946
            （q03 那次对照因事件循环复用失败，无值）

        结论要诚实：**影响是温和的（+0.02 量级），不是数量级差别**。
        保留对称编码的理由是任务本身对称（问句 ↔ 问句），
        而不是因为它能让分数变好看；非对称编码在原理上是给
        「短查询 vs 长段落」用的，用在这里属于借错工具。

        另外：这个指标单题波动很大（同一题两次跑可能 0.37 / 0.80），
        因为反推问题只有一两个。**不要用单题分数下结论**，看均值，
        且 0 分要回看原文——裁判可能把切题回答判成"含糊其辞"。

    继承 langchain 的 `Embeddings` 而不是鸭子类型：RAGAS 的
    `LangchainEmbeddingsWrapper` 依赖 `embed_documents` / `embed_query`
    这两个方法名与返回结构。
    """

    def __init__(self, embedder: EmbedderPort) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        # 刻意**不**调 embed_query：见类文档的对照数据。
        return self._embedder.embed_documents([text])[0]


def build_judge_llm():
    """构造 RAGAS 裁判模型（`LangchainLLMWrapper`）。

    Returns:
        ragas 的 `BaseRagasLLM` 实例。

    取舍：温度强制 0。生产问答用 `LLM_TEMPERATURE`（默认 0.2）让回答更自然，
    但裁判要的是**可复现的判定**——同一份数据跑两次分数不该漂。
    """
    ensure_ragas_importable()

    from langchain_openai import ChatOpenAI
    from ragas.llms.base import LangchainLLMWrapper

    settings = get_settings()
    if not settings.llm_base_url or not settings.llm_model:
        raise RuntimeError(
            "裁判模型未配置：请先在 .env 里设置 LLM_BASE_URL 与 LLM_MODEL"
        )

    chat = ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key_or_placeholder,
        model=settings.llm_model,
        temperature=0.0,
        timeout=settings.llm_timeout,
        # 重试交给 RAGAS 的 RunConfig：SDK 内部重试会放大单题耗时，
        # 而评估是批量任务，宁可让 RAGAS 计数并记录失败的样本。
        max_retries=0,
    )
    logger.info("裁判模型：%s @ %s", settings.llm_model, settings.llm_base_url)
    # bypass_n：Ollama 的 OpenAI 兼容端点不支持 n>1。answer_relevancy 默认
    # strictness=3（生成 3 个反推问题），不声明这一项时每次都会打印
    # "LLM returned 1 generations instead of requested 2" 并退化——显式关掉更诚实。
    return LangchainLLMWrapper(chat, bypass_n=True)


def build_judge_embeddings():
    """构造 RAGAS 用的向量化（项目自带的 BGE，对称编码）。

    维度与检索一致，因此不需要再校验 1024 之类的常量：
    维度的唯一真相是模型权重，`get_embedder()` 已经保证过。
    """
    ensure_ragas_importable()

    from ragas.embeddings.base import LangchainEmbeddingsWrapper

    from app.core.container import get_embedder

    return LangchainEmbeddingsWrapper(SentenceSimilarityEmbeddings(get_embedder()))
