# -*- coding: utf-8 -*-
"""检索参数调优：扫 top_k / 两路权重 / rerank_k / 重排阈值 / chunk_size。

为什么需要这个脚本，而不是直接改 .env 试：
    参数是耦合的（top_k 放大后重排才有得挑；重排阈值提高又需要更大的 top_k），
    一次改一个值，改动之间互相掩盖，永远收敛不了。
    这里把每个维度分开扫，用**同一份确定性指标**打分，
    才能说"这个值比那个值好"而不是"感觉好点了"。

为什么还要一份**合成基准**（本脚本的重点）：
    项目现有语料只有 3 篇、共 15 个片段、4 条事实，且每个事实独占一页。
    在这种语料上，`top_k=2` 与 `top_k=16`、纯向量与混合、块大小 150 与 500
    **全部得到满分**（实测 20 组参数并列 1.0）——它测不出任何差异，
    用它调参只是在给"当前配置"盖章。
    所以另建一份可区分的基准：24 条结构相似、仅参数不同的设备要求段落，
    检索必须在**近邻干扰项**之间挑出正确的那一段。这才是参数真正起作用的地方。

三个阶段（默认 all）：

    recall  top_k × 两路 RRF 权重            —— 答案片段是否进 top-k、排第几
    rerank  rerank_k × 重排 logits 下限       —— 同上，另附"无重排"对照
    chunk   chunk_size（150 / 300 / 500）     —— 命中率与 top1 余弦（稀释效应）

指标（全部确定性，不需要 LLM，因此可反复重跑）：
    hit@k    正确片段进入前 k 的比例
    mrr      正确片段名次倒数（第 1 名 = 1.0）
    top1_cos 命中的第 1 条片段的余弦相似度均值（衡量"块越大越稀释"）

实现取舍：
    1. **top_k 不需要重新检索**：两路结果按分数降序，切片后融合等价于用更小的
       top_k 检索（RRF 只吃排名）。所以一次 max-depth 召回 + 内存切片即可扫完。
    2. **chunk 阶段必须换 collection**：切分参数变了，向量本身就得重算。
       临时集合名 `<collection>__tune_<size>_<overlap>`，默认跑完删除。
    3. 合成基准的"命中"按**正文包含**判定，不看页码：合成语料是单页 txt，
       同一文件的任意块页码都一样，用页码判等于没判。

用法：
    .venv\\Scripts\\python.exe scripts/tune_retrieval.py
    .venv\\Scripts\\python.exe scripts/tune_retrieval.py --stage chunk --keep-temp
    .venv\\Scripts\\python.exe scripts/tune_retrieval.py --out eval/tuning.json
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.core.container import (  # noqa: E402
    get_embedder,
    get_reranker,
    get_vector_store,
)
from app.domain.models import Page, RetrievedChunk  # noqa: E402
from app.infrastructure.splitting import LangChainSplitter  # noqa: E402
from app.infrastructure.vectorstore import MilvusVectorStore  # noqa: E402
from app.rag.fusion import DEFAULT_RRF_K, fuse_detailed  # noqa: E402
from app.rag.retrieval import HybridRetriever  # noqa: E402
from scripts.rag_eval.dataset import EvalSample, ExpectedSource, load_dataset  # noqa: E402
from scripts.rag_eval.scoring import parse_pages  # noqa: E402

DATASET = Path(__file__).resolve().parent.parent / "eval" / "dataset" / "qa_dataset.jsonl"

TOP_K_CHOICES = (2, 4, 6, 8, 12)
# 两路权重候选。**只有比值有意义**：0.5/0.5 与 1/1 排序相同。
WEIGHT_CHOICES = ((0.5, 0.5), (0.7, 0.3), (0.3, 0.7), (1.0, 0.0))
RERANK_K_CHOICES = (2, 3, 4, 6)
# bge-reranker-base 实测相关 +5.16 / 无关 -10.19，所以 0.0 = 丢掉负分，
# 2.0 = 只留明确相关；-10 等价于不设阈值（用来观察阈值本身的作用）。
RERANK_SCORE_CHOICES = (-10.0, 0.0, 2.0)
CHUNK_SIZE_CHOICES = (150, 300, 500)


# -----------------------------------------------------------------------------
# 指标
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class Metrics:
    """一组参数下的检索质量。"""

    hit: float = 0.0
    mrr: float = 0.0
    top1_cosine: float | None = None
    count: int = 0

    def as_row(self) -> dict:
        return {
            "hit": round(self.hit, 4),
            "mrr": round(self.mrr, 4),
            "top1_cos": None if self.top1_cosine is None else round(self.top1_cosine, 4),
            "n": self.count,
        }


def _mean_metrics(rows: list[Metrics]) -> Metrics:
    rows = [row for row in rows if row.count]
    if not rows:
        return Metrics()
    n = len(rows)
    cosines = [row.top1_cosine for row in rows if row.top1_cosine is not None]
    return Metrics(
        hit=sum(row.hit for row in rows) / n,
        mrr=sum(row.mrr for row in rows) / n,
        top1_cosine=(sum(cosines) / len(cosines)) if cosines else None,
        count=n,
    )


def _rank_of_first(hits: list[RetrievedChunk], contains: str) -> int | None:
    """正确片段（正文包含该事实）的 1 起名次；没进榜返回 None。"""
    for rank, item in enumerate(hits, start=1):
        if contains in item.chunk.text:
            return rank
    return None


def _match_pages(item: RetrievedChunk, expected: ExpectedSource) -> bool:
    """真实语料的命中判定：文件名相同 + 页码落在该块区间内。

    只比文件名不够——同一份 PDF 的 4 页都会算命中，
    那样"页码级溯源"这个能力就完全测不出来。
    """
    if item.chunk.filename.strip() != expected.filename.strip():
        return False
    if expected.page is None:
        return True
    return expected.page in parse_pages(item.chunk.page_label)


def evaluate_real(hits: list[RetrievedChunk], sample: EvalSample) -> Metrics:
    """真实语料：按页码判定。"""
    expected = list(sample.expected_sources)
    if not expected:
        return Metrics()
    first: int | None = None
    for rank, item in enumerate(hits, start=1):
        if any(_match_pages(item, target) for target in expected):
            first = rank
            break
    cosine = hits[0].vector_score if hits else None
    return Metrics(
        hit=1.0 if first else 0.0,
        mrr=1.0 / first if first else 0.0,
        top1_cosine=cosine,
        count=1,
    )


# -----------------------------------------------------------------------------
# 合成基准：24 条结构相似的设备要求
# -----------------------------------------------------------------------------

_PARAM_POOL: tuple[str, ...] = ("工作电压", "防护等级", "保修期限", "校准间隔")

# 同一个参数名在 24 台设备上各有一条段落、措辞几乎逐字相同，
# **只有设备型号与数值不同**——这是检索真正会翻车的形态：
# 向量相似度全都高，BM25 也同时命中「防护等级」这个共同词，
# 必须靠「型号」这个稀疏 token 才能把正确那条挑出来。
# 第一版合成语料给每台设备配了**不同的**参数名（工作电压/待机功耗/…），
# 结果正确段落是唯一同时命中「型号+参数名」的那条，20 组参数全部满分，
# 与真实语料一样饱和——所以改成现在这种「同参数名、异型号」的硬负样本。
_VALUE_POOL: dict[str, tuple[str, ...]] = {
    "工作电压": ("220 伏", "230 伏", "240 伏", "110 伏"),
    "防护等级": ("IP54", "IP65", "IP66", "IP67"),
    "保修期限": ("12 个月", "18 个月", "24 个月", "36 个月"),
    "校准间隔": ("6 个月", "12 个月", "18 个月", "24 个月"),
}

DEVICE_COUNT = 24

SYNTHETIC_KB = "kb-tune"
SYNTHETIC_FILE = "synthetic-params.txt"


@dataclass(slots=True)
class Fact:
    """合成语料里的一条事实：问题 + 用来判定命中的原句。"""

    question: str
    sentence: str


def _device(index: int) -> str:
    return f"XG-{8800 + index}"


def _value(param: str, device_index: int) -> str:
    """同一参数在不同设备上取不同的值（同值会让"值"本身成为命中线索）。"""
    pool = _VALUE_POOL[param]
    return pool[(device_index + _PARAM_POOL.index(param)) % len(pool)]


def synthetic_corpus() -> tuple[str, list[Fact]]:
    """生成合成语料正文与对应问题。

    24 台设备 × 4 个参数 = 96 条互相干扰的段落；问句随机挑「每台设备一个参数」，
    共 24 条。同一参数的 24 条段落彼此是**近邻干扰项**——
    top_k 太小会把正确段落挤出去，这正是参数调优要量化的东西。
    """
    blocks: list[str] = []
    facts: list[Fact] = []
    for index in range(DEVICE_COUNT):
        device = _device(index)
        for param in _PARAM_POOL:
            value = _value(param, index)
            sentence = f"{device} 的{param}为 {value}"
            blocks.append(
                f"{device} 设备参数表\n"
                f"{sentence}。现场核对时应先确认铭牌与运行工况一致，"
                f"读数记入巡检表；若实测值与本条不符，立即停机并联系售后。"
            )
        # 每台设备只问一个参数，避免 96 道题把运行时间拖成 4 倍
        param = _PARAM_POOL[index % len(_PARAM_POOL)]
        facts.append(
            Fact(
                question=f"{device} 的{param}是多少？",
                sentence=f"{device} 的{param}为 {_value(param, index)}",
            )
        )
    return "\n\n".join(blocks), facts


@dataclass(slots=True)
class SyntheticIndex:
    """一份已建好的临时索引（切分参数固定）。"""

    store: MilvusVectorStore
    retriever: HybridRetriever
    facts: list[Fact] = field(default_factory=list)
    chunk_count: int = 0
    avg_chars: float = 0.0


def build_synthetic_index(size: int, overlap: int, *, base_name: str) -> SyntheticIndex:
    """按指定切分参数把合成语料入库到临时 collection。"""
    embedder = get_embedder()
    text, facts = synthetic_corpus()
    name = f"{base_name}__tune_{size}_{overlap}"

    store = MilvusVectorStore(collection=name)
    if store.client.has_collection(name):
        # 上次中断的残留：尺寸不一定一致，直接重建
        store.client.drop_collection(name)
    store.ensure_collection(embedder.dimension)

    splitter = LangChainSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = splitter.split([Page(page_no=1, text=text)])
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk.id = str(uuid.uuid4())
        chunk.kb_id = SYNTHETIC_KB
        chunk.doc_id = f"doc-tune-{size}"
        chunk.filename = SYNTHETIC_FILE
        chunk.embedding = vector
    store.upsert(chunks)

    return SyntheticIndex(
        store=store,
        retriever=HybridRetriever(embedder, store, top_k=max(TOP_K_CHOICES)),
        facts=facts,
        chunk_count=len(chunks),
        avg_chars=round(len(text) / len(chunks), 1),
    )


def evaluate_synthetic(
    index: SyntheticIndex,
    *,
    top_k: int,
    vector_weight: float,
    bm25_weight: float,
    keep: int,
    floor: float | None = None,
    reranker=None,
) -> Metrics:
    """在合成基准上评估一组参数。

    Args:
        top_k: 每路召回深度
        keep: 最终保留条数（送进上下文的数量）
        floor: 重排 logits 下限；reranker 为 None 时忽略
        reranker: 重排器；None 表示按融合分截断（无重排基线）
    """
    rows: list[Metrics] = []
    for fact in index.facts:
        trace = index.retriever.recall(SYNTHETIC_KB, fact.question, top_k=top_k)
        fused = fuse_detailed(
            trace.vector_hits,
            trace.bm25_hits,
            k=DEFAULT_RRF_K,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            top_k=top_k,
        ).items
        final: list[RetrievedChunk]
        if reranker is not None and fused:
            scored = reranker.rerank(
                fact.question, fused, top_n=len(fused), min_score=-1e9
            )
            final = [
                item for item in scored if (item.rerank_score or 0.0) >= (floor or 0.0)
            ][:keep]
        else:
            final = fused[:keep]

        rank = _rank_of_first(final, fact.sentence)
        rows.append(
            Metrics(
                hit=1.0 if rank else 0.0,
                mrr=1.0 / rank if rank else 0.0,
                top1_cosine=final[0].vector_score if final else None,
                count=1,
            )
        )
    return _mean_metrics(rows)


# -----------------------------------------------------------------------------
# 阶段
# -----------------------------------------------------------------------------


def stage_real_baseline(samples: list[EvalSample], retriever: HybridRetriever) -> dict:
    """真实语料：只作回归对照，并明确报告是否饱和。"""
    results: list[dict] = []
    for depth in (4, 8):
        for vector_weight, bm25_weight in WEIGHT_CHOICES:
            rows: list[Metrics] = []
            for sample in samples:
                trace = retriever.recall(sample.kb_id, sample.question, top_k=depth)
                fused = fuse_detailed(
                    trace.vector_hits,
                    trace.bm25_hits,
                    k=DEFAULT_RRF_K,
                    vector_weight=vector_weight,
                    bm25_weight=bm25_weight,
                    top_k=depth,
                ).items
                rows.append(evaluate_real(fused, sample))
            results.append(
                {
                    "top_k": depth,
                    "vector_weight": vector_weight,
                    "bm25_weight": bm25_weight,
                    **_mean_metrics(rows).as_row(),
                }
            )

    saturated = all(row["hit"] == 1.0 and row["mrr"] == 1.0 for row in results)
    _print_table(
        "真实语料：召回深度 × 权重（满分并列说明语料已饱和，不能用于选参）",
        ["top_k", "vector_weight", "bm25_weight", "hit", "mrr", "top1_cos"],
        results,
    )
    if saturated:
        print("  ⚠ 全部组合满分 → 现有语料无法区分参数；选参请看合成基准\n")
    return {"saturated": saturated, "grid": results}


def stage_recall(index: SyntheticIndex) -> dict:
    """合成基准：top_k × 两路权重。"""
    print(f"  合成语料：{index.chunk_count} 块，{len(index.facts)} 条事实")
    results: list[dict] = []
    for depth in TOP_K_CHOICES:
        for vector_weight, bm25_weight in WEIGHT_CHOICES:
            metrics = evaluate_synthetic(
                index,
                top_k=depth,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                keep=4,
            )
            results.append(
                {
                    "top_k": depth,
                    "vector_weight": vector_weight,
                    "bm25_weight": bm25_weight,
                    **metrics.as_row(),
                }
            )
    results.sort(key=lambda row: (-row["mrr"], -row["hit"]))
    _print_table(
        "合成基准：召回深度 × 权重（keep=4，先按 MRR 排序）",
        ["top_k", "vector_weight", "bm25_weight", "hit", "mrr", "top1_cos"],
        results,
    )
    return {"grid": results, "best": results[0] if results else None}


def stage_rerank(
    index: SyntheticIndex, top_k: int, vector_weight: float, bm25_weight: float
) -> dict:
    """合成基准：rerank_k × 重排阈值，附无重排对照。"""
    reranker = get_reranker()
    if reranker is None:
        print("  RERANK_ENABLED=false，跳过（无重排时按融合分截断即可）")
        return {"skipped": "reranker disabled"}

    baseline = evaluate_synthetic(
        index, top_k=top_k, vector_weight=vector_weight, bm25_weight=bm25_weight, keep=4
    )
    results: list[dict] = []
    for keep in RERANK_K_CHOICES:
        for floor in RERANK_SCORE_CHOICES:
            metrics = evaluate_synthetic(
                index,
                top_k=top_k,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                keep=keep,
                floor=floor,
                reranker=reranker,
            )
            results.append({"rerank_k": keep, "min_score": floor, **metrics.as_row()})
    results.sort(key=lambda row: (-row["mrr"], -row["hit"]))
    _print_table(
        "合成基准：重排保留条数 × logits 下限（先按 MRR 排序）",
        ["rerank_k", "min_score", "hit", "mrr", "top1_cos"],
        results,
    )
    print(f"  对照：不重排、按融合分截断 top4 → {baseline.as_row()}\n")
    return {"grid": results, "no_rerank": baseline.as_row()}


def stage_chunk(base_name: str, *, top_k: int) -> dict:
    """合成基准：chunk_size 对"答案是否进上下文"和"相似度稀释"的影响。"""
    results: list[dict] = []
    created: list[str] = []
    try:
        for size in CHUNK_SIZE_CHOICES:
            overlap = max(1, size // 6)
            index = build_synthetic_index(size, overlap, base_name=base_name)
            created.append(index.store.collection_name)
            metrics = evaluate_synthetic(
                index, top_k=top_k, vector_weight=0.5, bm25_weight=0.5, keep=4
            )
            results.append(
                {
                    "chunk_size": size,
                    "chunk_overlap": overlap,
                    "chunks": index.chunk_count,
                    "avg_chars": index.avg_chars,
                    **metrics.as_row(),
                }
            )
            print(
                f"  chunk_size={size}: {index.chunk_count} 块，"
                f"平均 {index.avg_chars} 字/块，hit@4={metrics.hit:.2f}"
            )
    finally:
        for name in created:
            try:
                get_vector_store().client.drop_collection(name)
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠ 删除临时集合 {name} 失败：{exc}")

    best = max(results, key=lambda row: (row["mrr"], row["hit"])) if results else None
    _print_table(
        "合成基准：切分块大小（块越大，单块越多干扰事实，相似度被稀释）",
        ["chunk_size", "chunk_overlap", "chunks", "avg_chars", "hit", "mrr", "top1_cos"],
        results,
    )
    return {"grid": results, "best": best}


# -----------------------------------------------------------------------------
# 输出
# -----------------------------------------------------------------------------


def _print_table(title: str, columns: list[str], rows: list[dict]) -> None:
    """打印等宽表格（输出要能直接贴进 PR / 文档）。"""
    print(f"\n{title}")
    widths = {column: len(column) for column in columns}
    for row in rows:
        for column in columns:
            widths[column] = max(widths[column], len(str(row.get(column, ""))))
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            "  ".join(
                str(row.get(column, "")).ljust(widths[column]) for column in columns
            )
        )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检索参数调优（确定性指标，不需要 LLM）")
    parser.add_argument(
        "--stage", choices=("all", "real", "recall", "rerank", "chunk"), default="all"
    )
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--keep-temp", action="store_true", help="保留临时 collection 便于手动检查")
    parser.add_argument("--out", default="", help="把结果写成 JSON（便于对比两次调参）")
    args = parser.parse_args(argv)

    settings = get_settings()
    base_name = get_vector_store().collection_name
    payload: dict = {
        "config": {
            "retrieve_top_k": settings.retrieve_top_k,
            "rerank_top_n": settings.rerank_top_n,
            "rerank_min_score": settings.rerank_min_score,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "hybrid_weights": [settings.hybrid_vector_weight, settings.hybrid_bm25_weight],
        }
    }

    if args.stage in ("all", "real"):
        samples = [
            sample
            for sample in load_dataset(args.dataset)
            if sample.answerable and sample.expected_sources
        ]
        print(f"\n[真实语料] 可答题 {len(samples)} 道")
        retriever = HybridRetriever(get_embedder(), get_vector_store())
        payload["real"] = stage_real_baseline(samples, retriever)

    index: SyntheticIndex | None = None
    if args.stage in ("all", "recall", "rerank"):
        print("[合成基准] 建临时索引（chunk_size=300 / overlap=50）")
        index = build_synthetic_index(300, 50, base_name=base_name)

    kept = False
    try:
        if args.stage in ("all", "recall") and index is not None:
            print("\n[阶段一] 召回深度 × 权重")
            payload["recall"] = stage_recall(index)
        best_top_k = settings.retrieve_top_k
        if payload.get("recall", {}).get("best"):
            best_top_k = int(payload["recall"]["best"]["top_k"])
        if args.stage in ("all", "rerank") and index is not None:
            print("\n[阶段二] 重排条数 × 阈值")
            payload["rerank"] = stage_rerank(index, best_top_k, 0.5, 0.5)
        if args.stage in ("all", "chunk"):
            print("\n[阶段三] 切分块大小")
            payload["chunk"] = stage_chunk(base_name, top_k=best_top_k)
    finally:
        if index is not None:
            name = index.store.collection_name
            if args.keep_temp:
                kept = True
                print(f"  保留临时集合：{name}")
            else:
                try:
                    get_vector_store().client.drop_collection(name)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ⚠ 删除临时集合 {name} 失败：{exc}")
    if kept:
        print("  （--keep-temp 生效，临时集合未删除）")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
