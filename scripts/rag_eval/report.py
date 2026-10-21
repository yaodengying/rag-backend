# -*- coding: utf-8 -*-
"""结果落盘：summary.json / samples.jsonl / scores.csv / report.md。

四份文件各有用途，不是重复：
    summary.json  机器读（CI 阈值、版本间对比）
    samples.jsonl 逐题全量（含上下文原文，回答"为什么这题分低"）
    scores.csv    人读 + Excel（评审/汇报时贴表）
    report.md     结论 + 口径 + 局限（发给不跑代码的人）
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from scripts.rag_eval.pipeline import Prediction
from scripts.rag_eval.scoring import BuiltMetric, retrieval_score

_EMPTY = "—"


def _fmt(value: float | int | None, digits: int = 4) -> str:
    """浮点固定小数位（1.0 显示为 1.0000），整数原样（毫秒数等）。"""
    if value is None:
        return _EMPTY
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _cell(text: str, limit: int = 40) -> str:
    """表格单元格：去掉换行与竖线，避免把 markdown 表格撑坏。"""
    flat = " ".join((text or "").split())
    flat = flat.replace("|", "\\|")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def build_payload(
    *,
    dataset_path: Path,
    predictions: list[Prediction],
    ragas_scores: dict[str, dict[str, float | None]],
    metrics: list[BuiltMetric],
    summary: dict,
    config: dict,
    duration_s: float,
) -> dict:
    """把所有结果拼成一个可序列化的结构。"""
    now = datetime.now()
    per_sample = []
    for pred in predictions:
        retr = retrieval_score(pred)
        per_sample.append(
            {
                "id": pred.sample.id,
                "kb_id": pred.sample.kb_id,
                "answerable": pred.sample.answerable,
                "question": pred.sample.question,
                "reference": pred.sample.reference,
                "answer": pred.answer,
                "refused": pred.refused,
                "expected_sources": [s.describe() for s in pred.sample.expected_sources],
                "citations": [
                    {"filename": c.filename, "page": c.page, "snippet": c.snippet}
                    for c in pred.citations
                ],
                "contexts": pred.contexts,
                "retrieval": {
                    "hit": retr.hit,
                    "page_recall": retr.page_recall,
                    "mrr": retr.mrr,
                },
                "scores": ragas_scores.get(pred.sample.id, {}),
                "latency_ms": pred.latency_ms,
                "error": pred.error,
            }
        )

    return {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "path": str(dataset_path),
            "samples": summary["counts"]["total"],
            "answerable": summary["counts"]["answerable"],
            "refusal_cases": summary["counts"]["refusal_cases"],
        },
        "config": config,
        "metrics": [m.key for m in metrics],
        "summary": summary,
        "samples": per_sample,
        "duration_s": round(duration_s, 1),
    }


def write_reports(out_dir: Path, payload: dict) -> dict[str, Path]:
    """写出四份结果文件，返回 {用途: 路径}。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written["summary"] = summary_path

    samples_path = out_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for sample in payload["samples"]:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    written["samples"] = samples_path

    scores_path = out_dir / "scores.csv"
    metric_keys = payload["metrics"]
    with scores_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["id", "kb_id", "answerable", "question", "refused",
             "retrieval_hit", "page_recall", "mrr", "n_citations",
             *metric_keys, "latency_ms", "error"]
        )
        for sample in payload["samples"]:
            retr = sample["retrieval"]
            writer.writerow(
                [
                    sample["id"],
                    sample["kb_id"],
                    sample["answerable"],
                    sample["question"],
                    sample["refused"],
                    _csv_num(retr["hit"]),
                    _csv_num(retr["page_recall"]),
                    _csv_num(retr["mrr"]),
                    len(sample["citations"]),
                    *[_csv_num(sample["scores"].get(key)) for key in metric_keys],
                    sample["latency_ms"],
                    sample["error"],
                ]
            )
    written["scores"] = scores_path

    report_path = out_dir / "report.md"
    report_path.write_text(render_markdown(payload), encoding="utf-8")
    written["report"] = report_path
    return written


def _csv_num(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    counts = summary["counts"]
    cfg = payload["config"]
    lines: list[str] = []

    lines.append("# RAG 评估报告（RAGAS）")
    lines.append("")
    lines.append(f"- 生成时间：{payload['generated_at']}")
    lines.append(
        f"- 数据集：`{payload['dataset']['path']}`"
        f"（{counts['total']} 题 = {counts['answerable']} 可答 + "
        f"{counts['refusal_cases']} 应拒答）"
    )
    lines.append(f"- 裁判模型：`{cfg.get('llm_model')}` @ `{cfg.get('llm_base_url')}`")
    lines.append(f"- 向量模型：`{cfg.get('embedding_model')}`（{cfg.get('embedding_device')}）")
    lines.append(
        f"- 检索配置：top_k={cfg.get('retrieve_top_k')}，"
        f"rerank_k={cfg.get('rerank_top_n')}，"
        f"重排={'开' if cfg.get('rerank_enabled') else '关'}"
    )
    lines.append(f"- 总耗时：{payload['duration_s']} 秒")
    lines.append("")

    # ---- 一、RAGAS ----
    lines.append("## 一、语义指标（RAGAS，模型判定）")
    lines.append("")
    if payload["metrics"]:
        lines.append("| 指标 | 均值 | 有效样本 |")
        lines.append("| --- | ---: | ---: |")
        for item in summary["ragas"]:
            lines.append(
                f"| {item['label']}（{item['key']}） | {_fmt(item['mean'])} | "
                f"{item['valid']}/{item['total']} |"
            )
    else:
        lines.append("_本次未启用 RAGAS 指标。_")
    lines.append("")
    lines.append(
        "> 有效样本 < 总数表示有样本模型输出无法解析（NaN），已排除在均值外，"
        "不与 0 分混为一谈。"
    )
    lines.append("")

    # ---- 二、检索 ----
    retr = summary["retrieval"]
    lines.append("## 二、检索质量（确定性指标，可复现）")
    lines.append("")
    lines.append("| 指标 | 值 | 口径 |")
    lines.append("| --- | ---: | --- |")
    lines.append(
        f"| 命中率 Hit@k | {_fmt(retr['hit_rate'])} | 期望来源（文件名+页码）"
        "是否出现在引用里 |"
    )
    lines.append(
        f"| 页码级召回率 | {_fmt(retr['page_recall'])} | 命中的期望来源数 / 期望来源总数 |"
    )
    lines.append(f"| MRR | {_fmt(retr['mrr'])} | 首个命中来源名次的倒数 |")
    lines.append(f"| 平均引用数 | {_fmt(retr['avg_citations'])} | 每题返回几条引用 |")
    lines.append("")

    # ---- 三、生成与拒答 ----
    gen = summary["generation"]
    lines.append("## 三、生成与拒答")
    lines.append("")
    lines.append("| 指标 | 值 | 说明 |")
    lines.append("| --- | ---: | --- |")
    lines.append(
        f"| 答出率 | {_fmt(gen['answer_rate'])} | 可答题中未拒答的比例"
        f"（分母 {gen.get('answerable_ok')} 题，不含运行失败） |"
    )
    lines.append(
        f"| 拒答正确率 | {_fmt(gen['refusal_accuracy'])} | 应拒答题中确实拒答的比例"
        f"（分母 {gen.get('refusal_cases_ok')} 题） |"
    )
    latency = summary["latency_ms"]
    lines.append(
        f"| 平均耗时 | {_fmt(latency['mean'])} ms | 检索+重排+生成；最慢 "
        f"{_fmt(latency['max'])} ms |"
    )
    lines.append("")

    # ---- 四、逐题 ----
    lines.append("## 四、逐题明细")
    lines.append("")
    header = "| # | 问题 | 命中 | 忠实度 | 相关性 | 上下文召回 | 上下文精确 | 耗时(s) |"
    lines.append(header)
    lines.append("| --- | --- | :---: | ---: | ---: | ---: | ---: | ---: |")
    for sample in payload["samples"]:
        scores = sample["scores"]
        hit = sample["retrieval"]["hit"]
        hit_cell = "✅" if hit == 1.0 else ("❌" if hit == 0.0 else _EMPTY)
        if not sample["answerable"]:
            hit_cell = "拒答" if sample["refused"] else "应拒答"
        lines.append(
            "| {id} | {q} | {hit} | {f} | {ar} | {cr} | {cp} | {lat} |".format(
                id=sample["id"],
                q=_cell(sample["question"], 30),
                hit=hit_cell,
                f=_fmt(scores.get("faithfulness")),
                ar=_fmt(scores.get("answer_relevancy")),
                cr=_fmt(scores.get("context_recall")),
                cp=_fmt(scores.get("context_precision")),
                lat=f"{sample['latency_ms'] / 1000:.1f}",
            )
        )
    lines.append("")

    # ---- 五、失败 ----
    lines.append("## 五、异常样本")
    lines.append("")
    if summary["errors"]:
        for item in summary["errors"]:
            lines.append(f"- `{item['id']}`：{item['error']}")
    else:
        lines.append("_无（全部样本跑通）_")
    lines.append("")

    # ---- 六、口径 ----
    lines.append("## 六、指标口径与局限")
    lines.append("")
    lines.append(
        "1. **裁判就是被测模型**：RAGAS 用 `.env` 里的同一个 LLM 判分。"
        "小模型自评会偏宽松，绝对分数只适合横向比较（换配置前/后），不宜对外宣称。"
    )
    lines.append(
        "2. **上下文召回率 ≠ 检索召回率**：RAGAS 的 context_recall 由 LLM 判断"
        "「标准答案里的信息点有多少被上下文覆盖」，看的是语义；"
        "第二节的页码级召回率比对的是**文件名+页码**，看的是溯源是否点对地方。"
    )
    lines.append(
        "3. **语料规模决定指标上限**：当前知识库只有个位数文档/片段，"
        "检索几乎不会跑偏，指标偏乐观；扩库后应重跑并与本次对比。"
    )
    lines.append(
        "4. **应拒答题不进 RAGAS 均值**：RAGAS 没有「应当拒答」的度量，"
        "把拒答话术塞进忠实度只会虚高，故单列拒答正确率。"
    )
    lines.append(
        "5. **answer_relevancy 的 0 分要回看原文**：该指标先由裁判判断回答是否"
        "「含糊其辞」，一旦全判为是就直接记 0。本地 7B 偶有误判（切题的回答被判 0），"
        "因此这一项适合看整体趋势，单题结论以 `samples.jsonl` 里的原文为准。"
    )
    lines.append("")
    return "\n".join(lines)


def render_console(payload: dict) -> str:
    """控制台摘要（不追求对齐，追求能一眼看出好坏）。"""
    summary = payload["summary"]
    counts = summary["counts"]
    lines = ["", "=" * 66, "评估完成", "=" * 66]
    lines.append(
        f"样本 {counts['total']} 题（可答 {counts['answerable']} / 应拒答 "
        f"{counts['refusal_cases']}），失败 {counts['failed']}，"
        f"耗时 {payload['duration_s']}s"
    )
    if summary["ragas"]:
        lines.append("-" * 66)
        lines.append("RAGAS（模型判定）")
        for item in summary["ragas"]:
            lines.append(
                f"  {item['label']:<12}{_fmt(item['mean']):>8}"
                f"   （有效 {item['valid']}/{item['total']}）"
            )
    retr = summary["retrieval"]
    lines.append("-" * 66)
    lines.append("检索（确定性，页码级）")
    lines.append(f"  {'命中率':<12}{_fmt(retr['hit_rate']):>8}")
    lines.append(f"  {'页码召回率':<10}{_fmt(retr['page_recall']):>8}")
    lines.append(f"  {'MRR':<13}{_fmt(retr['mrr']):>8}")
    gen = summary["generation"]
    lines.append("-" * 66)
    lines.append(f"  答出率 {_fmt(gen['answer_rate'])}   拒答正确率 {_fmt(gen['refusal_accuracy'])}")
    lines.append("=" * 66)
    return "\n".join(lines)
