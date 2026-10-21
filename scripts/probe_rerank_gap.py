# -*- coding: utf-8 -*-
"""量一下「整批最高分闸门」（RERANK_MIN_TOP_SCORE）能不能用——用你自己的语料。

为什么阈值不能拍脑袋：
    交叉编码器的分数是**原始 logits**，尺度随模型与语料一起漂。
    本机 bge-reranker-base 的地板约 -10.2、相关片段轻松到 +5，
    换一个模型或换一份语料，同一个数字的含义立刻不同。
    所以这个脚本只做一件事：把「有答案的问题」和「库里没有答案但主题
    相邻的问题」各跑一遍召回，看两组的最高分有没有**间隔**。
    有间隔 → 阈值得以标定；没有间隔 → 这个闸门在你的语料上就不该开。

判据：
    正样本 = 数据集里 answerable=true 的题（应当被召回并回答）
    负样本 = 你提供的「主题相邻但库里没有答案」的问题（应当被拒答）
    只要 max(负样本) < min(正样本)，中点就是一个可用的阈值；
    区间重叠就说明当前重排分数不足以区分这两类，别硬设。

用法：
    # 用 eval/dataset/qa_dataset.jsonl 的正样本 + 内置的负样本示例
    .venv\\Scripts\\python.exe scripts\\probe_rerank_gap.py

    # 换成自己的负样本（每行一个问题，可选「kb_id<TAB>问题」）
    .venv\\Scripts\\python.exe scripts\\probe_rerank_gap.py --negative neg.txt

    # 指定数据集与检索深度
    .venv\\Scripts\\python.exe scripts\\probe_rerank_gap.py \\
        --dataset eval/dataset/qa_dataset.jsonl --top-k 8

需要 Milvus 在跑、BGE 权重在场（与线上同一条检索链路），**不需要 LLM**。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import os  # noqa: E402

os.chdir(PROJECT_ROOT)
# Milvus 走 gRPC，全局代理会让它 502
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from app.core.container import get_retrieval_graph  # noqa: E402

# 内置示例：主题相邻但库里没有答案的问题（对本项目 3 篇语料）。
# 它们比"完全无关"更有价值——完全无关的问题连召回都没有，测不出区分度。
DEFAULT_NEGATIVES = [
    "保修期可以付费延长吗？",
    "退货的运费由谁承担？",
    "备件发货可以加急吗？",
    "MACD 指标的默认参数是多少？",
    "切片数量上限是多少？",
    "设备的电池续航多久？",
]


def _top_score(graph, kb_id: str, question: str, top_k: int, rerank_k: int):
    """跑一次检索子图，返回 (最高重排分, 可用片段数, 最高向量分)。"""
    state = {
        "kb_id": kb_id,
        "question": question,
        "history": [],
        "top_k": top_k,
        "rerank_k": rerank_k,
        "retry_count": 0,
    }
    ranked = graph.invoke(state).get("ranked") or []
    rerank = [item.rerank_score for item in ranked if item.rerank_score is not None]
    vector = [item.vector_score for item in ranked if item.vector_score is not None]
    return (max(rerank) if rerank else None), len(ranked), (max(vector) if vector else None)


def _load_dataset(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("answerable") and row.get("kb_id"):
            rows.append(row)
    return rows


def _load_negatives(path: Path | None, fallback_kb: str) -> list[tuple[str, str]]:
    if path is None:
        return [(fallback_kb, question) for question in DEFAULT_NEGATIVES]
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            kb_id, question = line.split("\t", 1)
        else:
            kb_id, question = fallback_kb, line
        rows.append((kb_id.strip(), question.strip()))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="量整批最高分闸门的分离度")
    parser.add_argument("--dataset", default="eval/dataset/qa_dataset.jsonl")
    parser.add_argument("--negative", default="", help="负样本文件：每行一个问题，或 kb_id<TAB>问题")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rerank-k", type=int, default=4)
    args = parser.parse_args()

    dataset = _load_dataset(Path(args.dataset))
    if not dataset:
        print(f"[!!] {args.dataset} 里没有 answerable=true 的样本")
        return 1
    kb_default = dataset[0]["kb_id"]
    negatives = _load_negatives(Path(args.negative) if args.negative else None, kb_default)

    graph = get_retrieval_graph()

    print("== 正样本（有答案，应当召回）==")
    print(f"{'id':6s} {'top1':>8s} {'n':>3s} {'vec':>7s}  问题")
    positives: list[float] = []
    for row in dataset:
        score, count, vec = _top_score(graph, row["kb_id"], row["question"], args.top_k, args.rerank_k)
        if score is not None:
            positives.append(score)
        shown = f"{score:8.3f}" if score is not None else "     nil"
        print(f"{row['id']:6s} {shown} {count:3d} {vec if vec is not None else -1:7.3f}  {row['question'][:34]}")

    print()
    print("== 负样本（主题相邻，库里没有答案，应当拒答）==")
    print(f"{'top1':>8s} {'n':>3s} {'vec':>7s}  问题")
    negatives_scores: list[float] = []
    for kb_id, question in negatives:
        score, count, vec = _top_score(graph, kb_id, question, args.top_k, args.rerank_k)
        if score is not None:
            negatives_scores.append(score)
        shown = f"{score:8.3f}" if score is not None else "     nil"
        print(f"{shown} {count:3d} {vec if vec is not None else -1:7.3f}  {question[:34]}")

    print()
    print("== 结论 ==")
    if not positives:
        print("正样本一条都没召回，先修检索再谈阈值。")
        return 1
    print(
        f"正样本 top1：min={min(positives):.3f} 中位={statistics.median(positives):.3f} "
        f"max={max(positives):.3f}（{len(positives)} 条）"
    )
    if negatives_scores:
        print(
            f"负样本 top1：min={min(negatives_scores):.3f} 中位={statistics.median(negatives_scores):.3f} "
            f"max={max(negatives_scores):.3f}（{len(negatives_scores)} 条）"
        )
    zero_recall = len(negatives) - len(negatives_scores)
    if zero_recall:
        print(f"另有 {zero_recall} 条负样本**零召回**（被逐条闸门或召回深度挡在门外，这是最好的结果）")

    if not negatives_scores:
        print(
            "所有负样本都零召回 → 现有闸门已经够用，RERANK_MIN_TOP_SCORE 保持 0.0，"
            "别为了让配置看起来「更安全」而加一个没有数据支撑的阈值。"
        )
        return 0

    low_positive = min(positives)
    high_negative = max(negatives_scores)
    if high_negative < low_positive:
        gap = low_positive - high_negative
        suggest = round((low_positive + high_negative) / 2, 2)
        print(
            f"两组可分：负样本最高 {high_negative:.3f} < 正样本最低 {low_positive:.3f}，"
            f"间隔 {gap:.3f} → 建议 RERANK_MIN_TOP_SCORE={suggest}"
        )
        if gap < 0.5:
            print(
                "⚠️ 间隔小于 0.5，说明这两类在重排分上挨得很近，阈值会随语料漂："
                "上线前请补更多负样本，并盯住评估集的拒答率与 answer_relevancy。"
            )
        return 0

    print(
        f"⚠️ 两组重叠：负样本最高 {high_negative:.3f} ≥ 正样本最低 {low_positive:.3f}。"
        "此时**任何**单一阈值都会同时误杀正样本或放过负样本 —— 不要设这个闸门，"
        "改用「加负样本进评估集 + 提升切分/元数据质量」或语义级核验（成本更高）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
