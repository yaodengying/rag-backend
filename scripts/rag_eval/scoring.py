# -*- coding: utf-8 -*-
"""评分：RAGAS 指标（模型判定）+ 确定性检索指标（无需模型）。

两套指标并存是刻意的：

1. **RAGAS 指标**衡量语义层面——忠实度、答案相关性、上下文召回/精确率。
   它们依赖裁判模型，7B 本地模型判得动，但分数带噪声，且不可复现到小数点。

2. **确定性指标**衡量"检索有没有真的命中该命中的那一段"。
   本项目的卖点是**页码级溯源**，而 RAGAS 没有页码概念。用
   `expected_sources`（文件名 + 页码）与返回的 `citations` 直接比对，
   得到可复现的 命中率 / 页码召回率 / MRR——同一份数据跑一百次都是同一个数。
   评审时要"指标不能讲故事"时，看这一组。

`answerable=false` 的样本只进拒答统计：RAGAS 没有"应当拒答"的度量，
把它塞进 faithfulness / context_recall 的均值里会让指标虚高（拒答话术
天然与上下文无关，反而容易拿满分）。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable

from app.domain.models import Citation
from scripts.rag_eval.dataset import ExpectedSource
from scripts.rag_eval.pipeline import Prediction


# -----------------------------------------------------------------------------
# 一、RAGAS 指标装配
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BuiltMetric:
    """一个已装配好的 RAGAS 指标。

    Attributes:
        key: 命令行/报告里用的短名
        label: 报告里的中文名
        column: RAGAS 结果里的列名（= metric.name，不是 key，两者不一定相同）
        metric: 传给 `evaluate()` 的指标对象
    """

    key: str
    label: str
    column: str
    metric: Any


# key -> (中文名, 构造函数)。构造函数签名为 (llm, embeddings) -> Metric。
_METRIC_CATALOG: dict[str, tuple[str, Callable[[Any, Any], Any]]] = {}


def _register(key: str, label: str, factory: Callable[[Any, Any], Any]) -> None:
    _METRIC_CATALOG[key] = (label, factory)


def _build_catalog() -> None:
    """延迟注册：没装 ragas 时，导入本模块不该失败。"""
    if _METRIC_CATALOG:
        return

    from scripts.rag_eval.compat import ensure_ragas_importable

    ensure_ragas_importable()

    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    _register("faithfulness", "忠实度", lambda llm, emb: Faithfulness(llm=llm))
    _register(
        "answer_relevancy",
        "答案相关性",
        # strictness 是"反推几个问题"的上限；本机 Ollama 不支持 n>1，
        # 适配层已声明 bypass_n，实际只反推 1 个，此处保留 3 只作为意图表达。
        lambda llm, emb: ResponseRelevancy(llm=llm, embeddings=emb, strictness=3),
    )
    _register(
        "context_recall",
        "上下文召回率",
        lambda llm, emb: LLMContextRecall(llm=llm),
    )
    _register(
        "context_precision",
        "上下文精确率",
        lambda llm, emb: LLMContextPrecisionWithReference(llm=llm),
    )


DEFAULT_METRIC_KEYS: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
)


def available_metric_keys() -> list[str]:
    """列出可用的 RAGAS 指标短名。"""
    _build_catalog()
    return list(_METRIC_CATALOG)


def build_metrics(
    keys: list[str] | tuple[str, ...],
    llm: Any,
    embeddings: Any,
) -> list[BuiltMetric]:
    """按短名装配指标。

    Raises:
        ValueError: 出现未知短名（列出来，而不是静默忽略）
    """
    _build_catalog()
    unknown = [key for key in keys if key not in _METRIC_CATALOG]
    if unknown:
        raise ValueError(
            f"未知指标：{', '.join(unknown)}；可选：{', '.join(_METRIC_CATALOG)}"
        )

    built: list[BuiltMetric] = []
    for key in keys:
        label, factory = _METRIC_CATALOG[key]
        metric = factory(llm, embeddings)
        built.append(BuiltMetric(key=key, label=label, column=metric.name, metric=metric))
    return built


def ragas_rows(predictions: list[Prediction]) -> tuple[list[str], list[dict]]:
    """把预测转成 RAGAS 的样本行。

    Returns:
        (对应样本 id 列表, 行列表)。只包含「可答且跑成功」的样本——
        失败样本没有回答，算忠实度没有意义。
    """
    ids: list[str] = []
    rows: list[dict] = []
    for pred in predictions:
        if not pred.ok or pred.sample.is_refusal_case:
            continue
        ids.append(pred.sample.id)
        rows.append(
            {
                "user_input": pred.sample.question,
                "retrieved_contexts": pred.contexts,
                "response": pred.answer,
                "reference": pred.sample.reference,
                "reference_contexts": list(pred.sample.reference_contexts),
            }
        )
    return ids, rows


def score_with_ragas(
    predictions: list[Prediction],
    metrics: list[BuiltMetric],
    llm: Any,
    embeddings: Any,
    run_config: Any = None,
    show_progress: bool = True,
) -> dict[str, dict[str, float | None]]:
    """跑 RAGAS 评分。

    Returns:
        {样本 id: {指标短名: 分数或 None}}。None 表示该题该项算不出
        （模型输出无法解析等），与 0 分严格区分。
    """
    from scripts.rag_eval.compat import ensure_ragas_importable

    ensure_ragas_importable()

    from ragas import EvaluationDataset, RunConfig, evaluate

    ids, rows = ragas_rows(predictions)
    if not rows:
        return {}

    result = evaluate(
        dataset=EvaluationDataset.from_list(rows),
        metrics=[item.metric for item in metrics],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config or RunConfig(),
        show_progress=show_progress,
        # 单题失败不该让整批评估前功尽弃；NaN 会在下面被统计出来并显式提示。
        raise_exceptions=False,
    )

    scores = result.scores or []
    if len(scores) != len(ids):
        # 宁可显式报错也不要错位映射：分数对错样本比没有分数更糟
        raise RuntimeError(
            f"RAGAS 返回 {len(scores)} 条分数，但送了 {len(ids)} 条样本；无法对齐"
        )

    table: dict[str, dict[str, float | None]] = {}
    for sample_id, row in zip(ids, scores):
        table[sample_id] = {
            item.key: _clean(row.get(item.column)) for item in metrics
        }
    return table


def _clean(value: Any) -> float | None:
    """把 numpy / NaN 收敛成 float 或 None。"""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, 6)


# -----------------------------------------------------------------------------
# 二、确定性检索指标（页码级）
# -----------------------------------------------------------------------------

# 页码标签形如「第3页」「第2-4页」（跨页块）。只认阿拉伯数字：
# `Chunk.page_label` 就是这么产出的，罗马数字在页码规则里不参与。
_PAGE_LABEL = re.compile(r"第\s*(\d+)(?:\s*[-–—~]\s*(\d+))?\s*页")


def parse_pages(label: str | None) -> frozenset[int]:
    """从页码标签解析出覆盖的页码集合。"""
    if not label:
        return frozenset()
    pages: set[int] = set()
    for match in _PAGE_LABEL.finditer(label):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if end >= start:
            pages.update(range(start, end + 1))
    return frozenset(pages)


def _citation_matches(citation: Citation, expected: ExpectedSource) -> bool:
    if citation.filename.strip() != expected.filename.strip():
        return False
    if expected.page is None:
        return True
    return expected.page in parse_pages(citation.page)


@dataclass(slots=True)
class RetrievalScore:
    """一道题的检索质量。

    Attributes:
        hit: 是否命中任一期望来源（0/1）
        page_recall: 命中的期望来源数 / 期望来源总数
        mrr: 首个命中来源的名次倒数（第 1 名命中为 1.0）
    """

    hit: float | None = None
    page_recall: float | None = None
    mrr: float | None = None


def retrieval_score(pred: Prediction) -> RetrievalScore:
    """按「文件名 + 页码」比对引用与期望来源。"""
    expected = pred.sample.expected_sources
    if not expected or not pred.ok:
        return RetrievalScore()

    matched = 0
    seen: set[int] = set()
    first_rank: int | None = None
    for index, citation in enumerate(pred.citations):
        for position, item in enumerate(expected):
            if position in seen:
                continue
            if _citation_matches(citation, item):
                seen.add(position)
                matched += 1
                if first_rank is None:
                    first_rank = index + 1

    return RetrievalScore(
        hit=1.0 if matched else 0.0,
        page_recall=matched / len(expected),
        mrr=1.0 / first_rank if first_rank else 0.0,
    )


# -----------------------------------------------------------------------------
# 三、聚合
# -----------------------------------------------------------------------------

def _mean(values: list[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return round(sum(kept) / len(kept), 4) if kept else None


def aggregate(
    predictions: list[Prediction],
    ragas_scores: dict[str, dict[str, float | None]],
    metrics: list[BuiltMetric],
) -> dict:
    """汇总成报告需要的结构（纯数据，格式化交给 report.py）。"""
    answerable = [p for p in predictions if p.sample.answerable]
    refusal_cases = [p for p in predictions if p.sample.is_refusal_case]

    retr = {p.sample.id: retrieval_score(p) for p in predictions}

    ragas_block = []
    for item in metrics:
        values = [ragas_scores.get(p.sample.id, {}).get(item.key) for p in answerable]
        ragas_block.append(
            {
                "key": item.key,
                "label": item.label,
                "mean": _mean(values),
                "valid": sum(1 for v in values if v is not None),
                "total": len(answerable),
            }
        )

    hit_values = [retr[p.sample.id].hit for p in answerable]
    recall_values = [retr[p.sample.id].page_recall for p in answerable]
    mrr_values = [retr[p.sample.id].mrr for p in answerable]

    # 分母只用「跑通了」的样本：调用失败是环境故障，不是模型拒答。
    # 把两者混在一个比率里，会让"Ollama 挂了"看起来像"模型不肯回答"。
    answerable_ok = [p for p in answerable if p.ok]
    refusal_ok = [p for p in refusal_cases if p.ok]
    answered = sum(1 for p in answerable_ok if not p.refused)
    refused_right = sum(1 for p in refusal_ok if p.refused)
    latencies = [p.latency_ms for p in predictions if p.ok]

    return {
        "counts": {
            "total": len(predictions),
            "answerable": len(answerable),
            "refusal_cases": len(refusal_cases),
            "failed": sum(1 for p in predictions if not p.ok),
        },
        "ragas": ragas_block,
        "retrieval": {
            "hit_rate": _mean(hit_values),
            "page_recall": _mean(recall_values),
            "mrr": _mean(mrr_values),
            "avg_citations": _mean([float(len(p.citations)) for p in answerable if p.ok]),
        },
        "generation": {
            "answer_rate": round(answered / len(answerable_ok), 4) if answerable_ok else None,
            "refusal_accuracy": (
                round(refused_right / len(refusal_ok), 4) if refusal_ok else None
            ),
            "answerable_ok": len(answerable_ok),
            "refusal_cases_ok": len(refusal_ok),
        },
        "latency_ms": {
            "mean": int(sum(latencies) / len(latencies)) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "errors": [
            {"id": p.sample.id, "error": p.error} for p in predictions if not p.ok
        ],
    }
