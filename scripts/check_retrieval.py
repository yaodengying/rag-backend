# -*- coding: utf-8 -*-
"""混合检索自检：验证「向量 + BM25 双路召回 → RRF 融合 → 重排」这一段是对的。

为什么需要这个脚本：
    前四步（加载 / 切分 / 向量化 / 入库）都有各自的自检，
    但**检索**才是 RAG 里最容易「看起来在工作、实际在乱排序」的一段：
    融合写错时结果依然非空、依然像那么回事，只是排错了——
    这类错误只有把每一路的原始输出拆开看才能发现。

判据（不只断言「不报错」）：

    | 检查项            | 期望                                                            |
    |------------------|-----------------------------------------------------------------|
    | 端口契约          | MilvusVectorStore 满足 VectorStorePort（含 search_bm25）          |
    | schema            | 稀疏字段由函数输出、text 开了 analyzer、存在 BM25 索引              |
    | schema 漂移       | 旧 schema（无 sparse）→ ConfigurationError 且提示重建，且**不重建** |
    | RRF 纯函数        | 公式、去重、权重、k、top_k、一路为空、权重为 0 跳过                 |
    | RRF 参数护栏      | k<1 / 负权重 / 未排序输入 / 分数缺失 / id 为空 → ValueError        |
    | 跨副本去重        | 同名文件+同正文合并、**先于 top_k 截断**、不误伤跨文件同文        |
    | 错误语义          | 上游返回未排序结果 → RetrievalError(502) 而非裸 ValueError(500)   |
    | 重排失败语义      | 前向异常 / 条数不符 / NaN → RerankError(502)，配置错误原样透传     |
    | 阈值下发          | BM25 阈值 0 → None（不设限）、正值原样下发                        |
    | BM25 检索         | 中文分词命中、kb 隔离、空查询短路、脏 kb_id → ValidationError       |
    | BM25 噪声         | 共享停用词也会命中但分数显著低于真命中（记录真分数，供调参）          |
    | 混合 > 纯向量     | **关键词文档在纯向量下召不回，混合检索下召得回**（本脚本的核心断言）  |
    | 语义 > 关键词     | 用词不同但语义相近的文档由向量路召回，BM25 路召不到                  |
    | 权重生效          | 向量权重压倒时向量路第一名胜出；反之 BM25 路第一名胜出               |
    | 单路跳过          | 权重为 0 的那一路**不发起检索**（用 spy 记录调用次数）              |
    | 降级              | BM25 路抛错时退化为纯向量检索，不打断整条链路                       |
    | 结果顺序          | 融合结果按 fusion_score 降序；同分时按 best_rank、再按 id 定序       |
    | 重排              | 假重排器：分数回填、按重排分排序、min_score 过滤、top_n 截断         |
    | 无重排            | reranker=None → 按融合分截断，rerank_score 全为 None               |
    | 只跑一次召回      | retrieve_with_trace 不得让 Embedding/两路检索各做两遍               |
    | 图节点            | make_retrieve_node：用 question、优先 rewritten_question、缺参短路  |
    | 稠密路阈值        | 纯向量模式用 MIN_SIMILARITY；混合模式默认不设限（否则向量路被静默清空）|
    | 动态 k            | 解析：None→配置、显式→硬上限、0 原样；adaptive 按本路分数断层收缩  |

    最后一行是修掉一个真实故障后补的回归断言：短查询对长块的余弦相似度只有
    0.28~0.35，配上 MIN_SIMILARITY=0.35 会让整条稠密路静默失活，
    而混合检索会**掩盖**它（BM25 仍在出结果）。详见 retrieval.py 的模块文档。

安全约定（与 check_vectorstore.py 一致）：
    全程使用**独立的临时 collection**（`rag_backend_check_retrieval_<pid>`），
    结束后删除。**绝不触碰** `MILVUS_COLLECTION` 指向的真实集合；
    脚本结尾复核真实集合行数前后一致，不一致直接判失败。

运行（必须用项目自己的 .venv，且 Milvus 容器已启动）：
    .venv\\Scripts\\python.exe scripts\\check_retrieval.py

    需要本机 Milvus（默认 http://127.0.0.1:19530），**不需要 GPU 与 BGE 权重**
    ——用确定性的假 embedder 与假重排器，让检查聚焦在检索编排逻辑本身。
    真实模型的端到端验证走 HTTP（见 README 的验证清单）。

退出码：0 全部通过；1 存在失败项。
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# settings 的 env_file 是相对路径，必须先切到项目根
os.chdir(PROJECT_ROOT)
# Milvus 走 gRPC，全局代理会把它拦成 502
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",localhost,127.0.0.1"
os.environ["no_proxy"] = os.environ["NO_PROXY"]

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from app.core.config import get_settings  # noqa: E402
from app.core.exceptions import (  # noqa: E402
    ConfigurationError,
    RetrievalError,
    ValidationError,
)
from app.domain.models import Chunk, RetrievedChunk  # noqa: E402
from app.domain.ports import VectorStorePort  # noqa: E402
from app.infrastructure.vectorstore import MilvusVectorStore  # noqa: E402
from app.rag.fusion import DEFAULT_RRF_K, fuse, fuse_detailed, rrf_score  # noqa: E402
from app.rag.nodes import make_retrieve_node  # noqa: E402
from app.rag.retrieval import HybridRetriever  # noqa: E402
from app.rag.topk import TOP_K_HARD_LIMIT, adaptive_k, resolve_k  # noqa: E402
from app.services.retrieval_service import RetrievalService  # noqa: E402

DIM = 8
COLLECTION = f"rag_backend_check_retrieval_{os.getpid()}"
LEGACY_COLLECTION = f"rag_backend_check_legacy_{os.getpid()}"

# ---------------------------------------------------------------------------
# 极简断言器：收集全部失败项，而不是第一个就退出
# ---------------------------------------------------------------------------

_failures: list[str] = []
_checks = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    """记录一项断言结果。"""
    global _checks
    _checks += 1
    if condition:
        print(f"  [ok]   {label}")
    else:
        print(f"  [FAIL] {label}" + (f"  -> {detail}" if detail else ""))
        _failures.append(label)


def expect_error(label: str, exc_type: type[Exception], keyword: str, action) -> None:
    """断言 action 抛出指定类型的异常，且消息里含关键词。"""
    global _checks
    _checks += 1
    try:
        action()
    except exc_type as exc:
        if keyword in str(exc):
            print(f"  [ok]   {label}（{type(exc).__name__}: {str(exc)[:70]}…）")
        else:
            print(f"  [FAIL] {label}  -> 消息里没有 {keyword!r}：{exc}")
            _failures.append(label)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {label}  -> 期望 {exc_type.__name__}，实际 {type(exc).__name__}: {exc}")
        _failures.append(label)
    else:
        print(f"  [FAIL] {label}  -> 没有抛异常")
        _failures.append(label)


def section(title: str) -> None:
    """打印分组标题。"""
    print()
    print(f"=== {title} ===")


# ---------------------------------------------------------------------------
# 造数据的小工具
# ---------------------------------------------------------------------------


def _onehot(axis: int) -> list[float]:
    """单位基向量：不同轴之间余弦相似度恰好为 0，可精确构造"召不回"。"""
    return [1.0 if i == axis else 0.0 for i in range(DIM)]


def _chunk(
    text: str,
    index: int,
    *,
    kb_id: str,
    doc_id: str,
    vector: list[float],
    filename: str = "sample.txt",
    page: int = 1,
    chunk_id: str | None = None,
) -> Chunk:
    """构造一个可入库的片段。"""
    return Chunk(
        id=chunk_id or str(uuid.uuid4()),
        kb_id=kb_id,
        doc_id=doc_id,
        text=text,
        chunk_index=index,
        page_start=page,
        page_end=page,
        filename=filename,
        embedding=list(vector),
    )


def _hit(chunk_id: str, text: str, *, vector: float | None = None, bm25: float | None = None):
    """构造一条只带指定分值的检索结果（用于纯函数测试）。"""
    chunk = Chunk(id=chunk_id, kb_id="kb-t", doc_id="doc-t", text=text)
    return RetrievedChunk(chunk=chunk, vector_score=vector, bm25_score=bm25)


# ---------------------------------------------------------------------------
# 假的上下游
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """按文本查表返回向量的假向量化器。

    未登记的文本返回 `onehot(DIM-1)`——与查询向量（`onehot(0)`）正交，
    COSINE 相似度为 0，必然低于 MIN_SIMILARITY，于是"未登记"就等于
    "稠密路一定召不回"。这个可控性正是构造
    「BM25 召得回、向量召不回」场景所必需的。
    """

    dimension = DIM

    def __init__(self, table: dict[str, list[float]] | None = None) -> None:
        self._table = dict(table or {})
        self.query_calls: list[str] = []
        self.doc_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.doc_calls.extend(texts)
        return [self._vec(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._vec(text)

    def _vec(self, text: str) -> list[float]:
        return list(self._table.get(text, _onehot(DIM - 1)))


class FakeReranker:
    """假重排器：按登记表给分，行为与 BgeReranker 的契约一致。

    故意复用真实现的处理顺序（先阈值过滤、再降序、再截断），
    否则重排阶段的"低分项占掉 top_n 名额"这个坑在自检里会漏掉。
    """

    def __init__(self, table: dict[str, float] | None = None, min_score: float = 0.0) -> None:
        self._table = dict(table or {})
        self._min_score = min_score
        self.calls: list[tuple[str, int, int]] = []  # (query, 候选数, top_n)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, len(candidates), top_n))
        if not candidates or top_n <= 0:
            return []
        threshold = self._min_score if min_score is None else float(min_score)
        scored = [(item, float(self._table.get(item.chunk.id, self._min_score - 1.0)))
                  for item in candidates]
        kept = [pair for pair in scored if pair[1] >= threshold]
        kept.sort(key=lambda pair: pair[1], reverse=True)
        out: list[RetrievedChunk] = []
        for item, score in kept[:top_n]:
            item.rerank_score = score
            out.append(item)
        return out


class RecordingStore:
    """包装真 Milvus 适配器，记录两路各被调用了几次（并可选让 BM25 路失败）。

    用包装而不是继承：目的是观察**调用行为**（谁被调了几次），
    而不是替换实现——两路都必须真的打到 Milvus，检查才有意义。
    """

    def __init__(self, inner: MilvusVectorStore, *, fail_bm25: bool = False) -> None:
        self._inner = inner
        self.fail_bm25 = fail_bm25
        self.dense_calls = 0
        self.bm25_calls = 0

    # ---- 转发（保持 VectorStorePort 的表面）----
    def ensure_collection(self, dimension: int) -> None:
        self._inner.ensure_collection(dimension)

    def upsert(self, chunks: list[Chunk]) -> int:
        return self._inner.upsert(chunks)

    def search(self, kb_id, query_embedding, top_k, min_score=None):
        self.dense_calls += 1
        return self._inner.search(kb_id, query_embedding, top_k, min_score)

    def search_bm25(self, kb_id, query, top_k, min_score=None):
        self.bm25_calls += 1
        if self.fail_bm25:
            from app.core.exceptions import RetrievalError

            raise RetrievalError("（自检构造的失败）BM25 索引不可用")
        return self._inner.search_bm25(kb_id, query, top_k, min_score)

    def delete_by_doc(self, doc_id: str) -> int:
        return self._inner.delete_by_doc(doc_id)

    def delete_by_kb(self, kb_id: str) -> int:
        return self._inner.delete_by_kb(kb_id)


# ---------------------------------------------------------------------------
# 1. 端口契约与 schema
# ---------------------------------------------------------------------------


def check_contract_and_schema(store: MilvusVectorStore) -> None:
    section("1. 端口契约与混合检索 schema")
    check("MilvusVectorStore 满足 VectorStorePort", isinstance(store, VectorStorePort))
    check("search_bm25 已进入端口契约", hasattr(store, "search_bm25"))

    store.ensure_collection(DIM)
    store.ensure_collection(DIM)  # 幂等

    description = store.client.describe_collection(COLLECTION)
    fields = {field["name"]: field for field in description.get("fields", [])}
    check("schema 含稀疏字段 sparse", "sparse" in fields)
    check(
        "sparse 是 BM25 函数的输出（不是被写入的普通字段）",
        bool(fields.get("sparse", {}).get("is_function_output")),
        f"is_function_output={fields.get('sparse', {}).get('is_function_output')!r}",
    )
    check(
        "text 开启了 analyzer（analyzer 配置压在 params 下且是字符串）",
        str((fields.get("text", {}).get("params") or {}).get("enable_analyzer", "")).lower()
        == "true",
        f"params={fields.get('text', {}).get('params')}",
    )
    check(
        "analyzer 是中文分析器",
        "chinese" in str((fields.get("text", {}).get("params") or {}).get("analyzer_params", "")),
        f"params={fields.get('text', {}).get('params')}",
    )

    functions = description.get("functions") or []
    check(
        "schema 里登记了 BM25 函数",
        any("bm25" in str(fn.get("name", "")).lower() for fn in functions),
        f"functions={functions}",
    )

    index_infos = [
        store.client.describe_index(COLLECTION, name)
        for name in store.client.list_indexes(COLLECTION)
    ]
    metrics = {str(info.get("metric_type", "")).upper() for info in index_infos}
    check("存在 BM25 索引", "BM25" in metrics, f"实际 {metrics}")
    check(
        "稠密索引仍是配置的度量类型",
        get_settings().milvus_metric_type.upper() in metrics,
        f"实际 {metrics}",
    )
    check("两路索引字段互不重叠", len({info.get("field_name") for info in index_infos}) == 2)


def check_schema_drift(store: MilvusVectorStore) -> None:
    section("2. schema 漂移检测：旧 collection 必须显式报错而不是静默重建")

    # 手工建一个"上一版"的 collection：没有 sparse 字段、text 没开 analyzer
    from pymilvus import DataType

    legacy = MilvusVectorStore(collection=LEGACY_COLLECTION)
    try:
        schema = legacy.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("kb_id", DataType.VARCHAR, max_length=64)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("filename", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("page_start", DataType.INT64)
        schema.add_field("page_end", DataType.INT64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIM)
        index_params = legacy.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        legacy.client.create_collection(
            collection_name=LEGACY_COLLECTION, schema=schema, index_params=index_params
        )

        expect_error(
            "旧 schema 上 ensure_collection 抛 ConfigurationError",
            ConfigurationError,
            "init_milvus.py --recreate",
            lambda: legacy.ensure_collection(DIM),
        )

        # 关键：必须"只报错不重建"——重建会丢光该库的向量
        after = legacy.client.describe_collection(LEGACY_COLLECTION)
        names = {f["name"] for f in after.get("fields", [])}
        check(
            "报错后旧 collection 未被重建（仍是旧字段集）",
            "sparse" not in names,
            f"实际字段 {sorted(names)}",
        )
        check(
            "旧 collection 里仍能查到它自己的字段（没被删表）",
            "text" in names and "embedding" in names,
        )

        # 提示里应逐条列出缺什么，便于一次改完
        try:
            legacy.ensure_collection(DIM)
        except ConfigurationError as exc:
            message = str(exc)
            check("报错把缺失项逐条列出（sparse / analyzer / 索引）",
                  "sparse" in message and "enable_analyzer" in message and "BM25" in message,
                  message[:200])
    finally:
        try:
            if legacy.client.has_collection(LEGACY_COLLECTION):
                legacy.client.drop_collection(LEGACY_COLLECTION)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# 3. RRF 融合纯函数
# ---------------------------------------------------------------------------


def check_fusion_pure() -> None:
    section("3. RRF 融合（纯函数，不碰 Milvus）")

    check("rrf_score 公式正确", abs(rrf_score(1, 60, 0.5) - 0.5 / 61) < 1e-12)
    check("rrf_score 排名越靠前分越高", rrf_score(1, 60) > rrf_score(2, 60) > rrf_score(10, 60))

    # 两路都命中的片段应当排最前（RRF 的核心性质）
    dense = [_hit("a", "甲", vector=0.9), _hit("b", "乙", vector=0.8)]
    sparse = [_hit("b", "乙", bm25=9.0), _hit("c", "丙", bm25=3.0)]
    fused = fuse(dense, sparse)
    check(
        "两路都命中的片段排第一（b 在向量第2、BM25 第1）",
        fused[0].chunk.id == "b",
        f"实际顺序 {[item.chunk.id for item in fused]}",
    )
    check(
        "融合结果里同一 id 只出现一次（已去重）",
        len({item.chunk.id for item in fused}) == len(fused) == 3,
        f"{[item.chunk.id for item in fused]}",
    )
    b_item = next(item for item in fused if item.chunk.id == "b")
    check(
        "去重时两路分数都回填（vector_score 与 bm25_score 同时在）",
        b_item.vector_score == 0.8 and b_item.bm25_score == 9.0,
        f"vector={b_item.vector_score} bm25={b_item.bm25_score}",
    )
    check(
        "只在一路出现的片段，另一路分数为 None（不是 0）",
        next(i for i in fused if i.chunk.id == "a").bm25_score is None,
    )
    check("融合分 = 两路贡献之和",
          abs(b_item.fusion_score - (0.5 / (DEFAULT_RRF_K + 2) + 0.5 / (DEFAULT_RRF_K + 1))) < 1e-12,
          f"实际 {b_item.fusion_score}")

    # 权重：只有比值有意义
    only_dense = fuse(dense, sparse, vector_weight=1.0, bm25_weight=0.0)
    only_sparse = fuse(dense, sparse, vector_weight=0.0, bm25_weight=1.0)
    check("向量权重单边时，向量路第一名排最前", only_dense[0].chunk.id == "a")
    check("BM25 权重单边时，BM25 路第一名排最前", only_sparse[0].chunk.id == "b")

    scaled = fuse(dense, sparse, vector_weight=10.0, bm25_weight=10.0)
    check(
        "权重整体放大 20 倍不改变排序（只有比值有意义）",
        [i.chunk.id for i in scaled] == [i.chunk.id for i in fused],
        f"{[i.chunk.id for i in scaled]} vs {[i.chunk.id for i in fused]}",
    )
    check(
        "权重整体放大后 fusion_score 也放大（绝对值不可当阈值用）",
        scaled[0].fusion_score > fused[0].fusion_score * 19,
    )

    # k 的影响：k 越小，单路靠前的优势越大
    k_small = fuse([_hit("a", "甲", vector=1.0)], [_hit("z", "癸", bm25=1.0)], k=1)
    check("k=1 时两路各自的第 1 名并列（同分）", abs(k_small[0].fusion_score - k_small[1].fusion_score) < 1e-12)

    # top_k 与空输入
    check("top_k 截断生效", len(fuse(dense, sparse, top_k=2)) == 2)
    check("一路为空仍可用（退化为单路排序）",
          [i.chunk.id for i in fuse(dense, [])] == ["a", "b"])
    check("两路都空返回空列表", fuse([], []) == [])
    check("top_k=0 返回空列表", fuse(dense, sparse, top_k=0) == [])

    # 权重为 0 的路：即使传入的结果没排序也不该校验（因为整条跳过）
    unsorted = [_hit("x", "某", bm25=1.0), _hit("y", "某", bm25=5.0)]
    check("权重为 0 的路被整条跳过（不做排序校验）",
          len(fuse(dense, unsorted, bm25_weight=0.0)) == 2)

    # 参数与输入护栏
    expect_value_error("k<1", lambda: fuse(dense, sparse, k=0))
    expect_value_error("负权重", lambda: fuse(dense, sparse, vector_weight=-0.1))
    expect_value_error(
        "输入未按分数降序（排名取自列表位置，顺序错了融合分就没意义）",
        lambda: fuse([_hit("a", "甲", vector=0.1), _hit("b", "乙", vector=0.9)], []),
    )
    expect_value_error(
        "输入缺少自身那一路的分数",
        lambda: fuse([RetrievedChunk(chunk=Chunk(id="a", kb_id="k", doc_id="d", text="甲"))], []),
    )
    expect_value_error(
        "chunk.id 为空无法去重",
        lambda: fuse([_hit("", "甲", vector=0.9)], []),
    )


def expect_value_error(label: str, action) -> None:
    """断言 action 抛 ValueError。"""
    global _checks
    _checks += 1
    try:
        action()
    except ValueError as exc:
        print(f"  [ok]   {label} → ValueError（{str(exc)[:60]}…）")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {label} → 期望 ValueError，实际 {type(exc).__name__}: {exc}")
        _failures.append(label)
    else:
        print(f"  [FAIL] {label} → 没有抛异常")
        _failures.append(label)


# ---------------------------------------------------------------------------
# 3b. 跨副本去重 + 「上游违约」的错误语义
# ---------------------------------------------------------------------------


def _dup(
    chunk_id: str,
    text: str,
    *,
    filename: str = "手册.txt",
    doc_id: str = "doc-a",
    vector: float,
) -> RetrievedChunk:
    """构造一条带正文与文件名的检索结果（去重键就是这两个字段）。"""
    chunk = Chunk(
        id=chunk_id, kb_id="kb-t", doc_id=doc_id, text=text, filename=filename
    )
    return RetrievedChunk(chunk=chunk, vector_score=vector)


def check_fusion_dedupe() -> None:
    """重复上传的副本会挤占召回名额——去重必须发生在 top_k 截断之前。

    这是修掉的一个真实故障：同一个 txt 被传 4 次后，向量路对 4 条副本给出
    **逐位相同**的相似度，融合后 4 条并列占掉 top_k=8 的一半名额，
    真正不同的片段被挤到截断线外。检索"看起来正常工作"，只是召回变差了。
    """
    section("3b. 跨副本去重（重复上传挤占召回名额）")

    copies = [
        _dup(f"copy-{i}", "XG-8800 的工作电压为 220 伏", doc_id=f"doc-{i}", vector=0.40)
        for i in range(4)
    ]
    others = [
        _dup("other-1", "XG-8801 的待机功耗为 3.5 瓦", doc_id="doc-9", vector=0.30),
        _dup("other-2", "XG-8802 的防护等级为 IP65", doc_id="doc-9", vector=0.20),
    ]
    dense = copies + others

    detailed = fuse_detailed(dense, [], top_k=3)
    check(
        "4 份重复副本合并为 1 条（另 3 条计入 duplicates_removed）",
        detailed.duplicates_removed == 3,
        f"duplicates_removed={detailed.duplicates_removed}",
    )
    check(
        "top_k=3 拿到的 3 条正文互不相同",
        len({item.chunk.text for item in detailed.items}) == 3,
        f"{[item.chunk.id for item in detailed.items]}",
    )
    check(
        "被重复副本挤掉的片段因此回到结果里（名额真的释放了）",
        {item.chunk.id for item in detailed.items} == {"copy-0", "other-1", "other-2"},
        f"{[item.chunk.id for item in detailed.items]}",
    )

    naive = fuse_detailed(dense, [], top_k=3, dedupe=False)
    check(
        "对照：不去重时 top_k=3 全是同一份文件的副本（故障原貌）",
        all(item.chunk.id.startswith("copy-") for item in naive.items),
        f"{[item.chunk.id for item in naive.items]}",
    )
    check("dedupe=False 时 duplicates_removed 恒为 0", naive.duplicates_removed == 0)
    check(
        "fuse() 是薄包装：结果与 fuse_detailed().items 一致",
        [i.chunk.id for i in fuse(dense, [], top_k=3)]
        == [i.chunk.id for i in detailed.items],
    )

    high_low = [
        _dup("high", "同一段正文", doc_id="doc-h", vector=0.90),
        _dup("low", "同一段正文", doc_id="doc-l", vector=0.50),
    ]
    check(
        "同内容只保留融合分最高的那条",
        [i.chunk.id for i in fuse_detailed(high_low, []).items] == ["high"],
        f"{[i.chunk.id for i in fuse_detailed(high_low, []).items]}",
    )

    cross_file = [
        _dup("a", "同一段公告", filename="甲.txt", vector=0.9),
        _dup("b", "同一段公告", filename="乙.txt", vector=0.8),
    ]
    cross_result = fuse_detailed(cross_file, [])
    check(
        "不同文件名的相同正文不合并（两篇文档引用同一段公告是合法的）",
        len(cross_result.items) == 2 and cross_result.duplicates_removed == 0,
        f"条数={len(cross_result.items)} 去重={cross_result.duplicates_removed}",
    )

    spaced = [
        _dup("s1", "甲  乙\n丙", vector=0.9),
        _dup("s2", "甲 乙 丙", vector=0.8),
    ]
    check(
        "正文的空白差异视为同一副本（切分会改写块内空白）",
        fuse_detailed(spaced, []).duplicates_removed == 1,
    )

    blanks = [_dup("e1", "", vector=0.9), _dup("e2", "", vector=0.8)]
    check(
        "空正文不参与去重（无法判断是否同内容，各自保留）",
        len(fuse_detailed(blanks, []).items) == 2,
    )

    scores = [item.fusion_score for item in detailed.items]
    check(
        "去重后融合分仍单调不增（没有二次排序需求）",
        all(a >= b - 1e-12 for a, b in zip(scores, scores[1:])),
        f"{scores}",
    )


class _RecordingStore:
    """假向量库：记录收到的阈值，原样返回预置结果。

    用于验证两件事：BM25 阈值的 0→None 转换、以及"上游返回未排序结果"
    是否被翻译成 502 而不是 500。
    """

    def __init__(
        self,
        dense: list[RetrievedChunk] | None = None,
        sparse: list[RetrievedChunk] | None = None,
    ) -> None:
        self._dense = list(dense or [])
        self._sparse = list(sparse or [])
        self.seen: dict[str, float | None] = {}

    def search(self, kb_id, query_embedding, top_k, min_score=None):
        self.seen["dense_min_score"] = min_score
        return self._dense

    def search_bm25(self, kb_id, query, top_k, min_score=None):
        self.seen["bm25_min_score"] = min_score
        return self._sparse


def check_error_semantics() -> None:
    section("3c. 错误语义（上游违约 → 502，阈值 0 → 不设限）")

    unsorted_store = _RecordingStore(
        dense=[
            _dup("a", "甲", vector=0.10),
            _dup("b", "乙", vector=0.90),
        ]
    )
    retriever = HybridRetriever(
        FakeEmbedder(), unsorted_store, top_k=4, hybrid_enabled=True
    )
    expect_error(
        "上游返回未排序结果 → RetrievalError（502），不是裸 ValueError(500)",
        RetrievalError,
        "融合失败",
        lambda: retriever.retrieve("kb-t", "查询"),
    )

    zero_store = _RecordingStore()
    HybridRetriever(
        FakeEmbedder(), zero_store, top_k=4, hybrid_enabled=True, bm25_min_score=0.0
    ).retrieve("kb-t", "查询")
    check(
        "BM25 阈值 0 显式转成 None（传 0 等于什么都没做，会让人以为设了阈值）",
        zero_store.seen.get("bm25_min_score", "missing") is None,
        f"实际 {zero_store.seen}",
    )

    positive_store = _RecordingStore()
    HybridRetriever(
        FakeEmbedder(), positive_store, top_k=4, hybrid_enabled=True, bm25_min_score=2.5
    ).retrieve("kb-t", "查询")
    check(
        "BM25 阈值为正时原样下发",
        positive_store.seen.get("bm25_min_score") == 2.5,
        f"实际 {positive_store.seen}",
    )


# ---------------------------------------------------------------------------
# 3d. 重排的失败语义
# ---------------------------------------------------------------------------


class _BrokenRerankModel:
    """按需抛错 / 返回非法分数的假交叉编码器（注入 BgeReranker._model）。"""

    def __init__(self, *, exc: Exception | None = None, scores=None) -> None:
        self._exc = exc
        self._scores = scores

    def compute_score(self, pairs, **_kwargs):
        if self._exc is not None:
            raise self._exc
        return self._scores


def check_rerank_failures() -> None:
    """重排失败必须收敛成 RerankError(502)，与 EmbeddingError 对称。

    没有包装时 `compute_score` 的异常会冒到兜底处理器变成 500，
    用户看到「服务内部错误」，而真正该做的是查重排模型或调小 RERANK_TOP_N。
    """
    section("3d. 重排失败语义（RerankError 502）")

    from app.core.exceptions import ConfigurationError, RerankError
    from app.infrastructure.rerank import BgeReranker

    reranker = BgeReranker()
    candidates = [
        _dup("a", "正文甲", vector=0.9),
        _dup("b", "正文乙", vector=0.8),
    ]

    def run(model, top_n: int = 4):
        reranker._model = model  # noqa: SLF001 - 自检专用注入点
        try:
            return reranker.rerank("查询", list(candidates), top_n=top_n, min_score=-1e9)
        finally:
            reranker._model = None  # noqa: SLF001

    oom = None
    try:
        run(_BrokenRerankModel(exc=RuntimeError("CUDA out of memory")))
    except RerankError as exc:
        oom = exc
    check("前向 OOM → RerankError（不是裸 RuntimeError）", oom is not None)
    check(
        "OOM 提示里给出可执行动作（RERANK_TOP_N / 换 CPU）",
        oom is not None and "RERANK_TOP_N" in oom.message,
        getattr(oom, "message", ""),
    )
    check(
        "错误里带候选对数与 max_length（否则无从判断该改哪个配置）",
        oom is not None
        and oom.detail.get("pairs") == len(candidates)
        and "max_length" in (oom.detail or {}),
        str(getattr(oom, "detail", {})),
    )

    expect_error(
        "分数条数少于候选数（会让候选与分数张冠李戴）",
        RerankError,
        "条数",
        lambda: run(_BrokenRerankModel(scores=[1.0])),
    )
    expect_error(
        "分数含 NaN（NaN 与阈值比较恒为 False → 全部候选被静默丢弃）",
        RerankError,
        "NaN/Inf",
        lambda: run(_BrokenRerankModel(scores=[1.0, float("nan")])),
    )
    expect_error(
        "ConfigurationError 不被包成 RerankError（该改 .env）",
        ConfigurationError,
        "重排",
        lambda: run(_BrokenRerankModel(exc=ConfigurationError("重排权重目录不存在"))),
    )
    check(
        "RerankError 是 502 且 code=rerank_error",
        RerankError.http_status == 502 and RerankError.code == "rerank_error",
        f"{RerankError.http_status} / {RerankError.code}",
    )
    check(
        "RerankError 继承 RetrievalError（与 Milvus / Embedding 故障同类）",
        issubclass(RerankError, RetrievalError),
    )
    check(
        "一次失败后重排器仍可用（没留下半加载状态）",
        [item.chunk.id for item in run(_BrokenRerankModel(scores=[0.5, 2.0]))] == ["b", "a"],
        "分数 0.5/2.0 应排成 b、a",
    )
    check(
        "min_score 过滤仍然生效（合法分数不受影响）",
        [item.chunk.id for item in ruff_filter(reranker, candidates)] == ["b"],
        "阈值 1.0 只留 2.0 那条",
    )


def ruff_filter(reranker, candidates) -> list:
    """辅助：注入「一好一差」的假模型，验证 min_score 过滤。"""
    model = _BrokenRerankModel(scores=[0.5, 2.0])
    reranker._model = model  # noqa: SLF001
    try:
        return reranker.rerank("查询", list(candidates), top_n=4, min_score=1.0)
    finally:
        reranker._model = None  # noqa: SLF001


# ---------------------------------------------------------------------------
# 4. BM25 检索语义（真 Milvus）
# ---------------------------------------------------------------------------


def check_bm25_search(store: MilvusVectorStore) -> None:
    section("4. BM25 全文检索语义（真 Milvus）")
    kb_a = f"kb-{uuid.uuid4().hex[:8]}"
    kb_b = f"kb-{uuid.uuid4().hex[:8]}"
    doc = str(uuid.uuid4())

    store.upsert(
        [
            _chunk("BM25 是一种稀疏检索算法，擅长关键词匹配。", 0, kb_id=kb_a, doc_id=doc,
                   vector=_onehot(3)),
            _chunk("红烧肉的做法需要冰糖和老抽。", 1, kb_id=kb_a, doc_id=doc, vector=_onehot(4)),
            _chunk("BM25 也会出现在另一个知识库里。", 0, kb_id=kb_b, doc_id=str(uuid.uuid4()),
                   vector=_onehot(5)),
        ]
    )

    hits = store.search_bm25(kb_a, "BM25 稀疏检索", top_k=10)
    texts = [hit.chunk.text for hit in hits]
    check("BM25 命中含关键词的片段", any("稀疏检索算法" in t for t in texts), f"命中 {texts}")
    check("BM25 分数已回填到 bm25_score", all(hit.bm25_score is not None for hit in hits))
    check("BM25 结果里 vector_score 为 None（没跑向量路）",
          all(hit.vector_score is None for hit in hits))
    check("BM25 结果按分数降序",
          [h.bm25_score for h in hits] == sorted([h.bm25_score for h in hits], reverse=True))
    check("BM25 结果不串库",
          all(hit.chunk.kb_id == kb_a for hit in hits), f"{[h.chunk.kb_id for h in hits]}")
    print(f"         [实测分数] {[round(h.bm25_score, 4) for h in hits]}")

    # 停用词噪声：中文分析器不过滤"的"，共享一个助词就能命中
    noise = store.search_bm25(kb_a, "完全无关的东西", top_k=10)
    good = max((h.bm25_score or 0.0) for h in hits)
    noise_max = max((h.bm25_score or 0.0) for h in noise) if noise else 0.0
    print(f"         [停用词噪声] 真命中最高 {good:.4f} / 噪声最高 {noise_max:.4f}")
    check(
        "共享停用词的噪声命中分数显著低于真命中（BM25 噪声可被重排兜住）",
        noise_max < good,
        f"噪声 {noise_max:.4f} ≥ 真命中 {good:.4f}",
    )

    # 阈值与边界
    filtered = store.search_bm25(kb_a, "BM25 稀疏检索", top_k=10, min_score=good + 1.0)
    check("min_score 生效（阈值高于最高分 → 空）", filtered == [])
    check("top_k 截断生效", len(store.search_bm25(kb_a, "BM25", top_k=1)) <= 1)
    check("空查询短路返回空列表", store.search_bm25(kb_a, "", top_k=10) == [])
    check("纯空白查询短路返回空列表", store.search_bm25(kb_a, "   ", top_k=10) == [])
    check("top_k=0 短路返回空列表", store.search_bm25(kb_a, "BM25", top_k=0) == [])
    check("无交集查询返回空列表", store.search_bm25(kb_a, "量子纠缠", top_k=10) == [])
    expect_error(
        "脏 kb_id → ValidationError（不是被包成 502）",
        ValidationError,
        "kb_id",
        lambda: store.search_bm25('kb" or 1 == 1', "BM25", top_k=5),
    )


# ---------------------------------------------------------------------------
# 5. HybridRetriever：混合检索的真实价值
# ---------------------------------------------------------------------------

KEYWORD_DOC = "BM25 是一种稀疏检索算法，擅长关键词匹配。"
SEMANTIC_DOC = "向量检索使用双塔模型计算余弦相似度。"
QUERY = "BM25"


def check_hybrid_value(store: MilvusVectorStore) -> None:
    section("5. HybridRetriever：混合检索是否真的强于纯向量")
    kb = f"kb-{uuid.uuid4().hex[:8]}"
    doc = str(uuid.uuid4())

    # 关键词文档的向量与查询正交（模拟"训练语料里没见过这个词"），
    # 语义文档的向量与查询同向（模拟"模型认为它最相关"）
    embedder = FakeEmbedder({QUERY: _onehot(0), KEYWORD_DOC: _onehot(1), SEMANTIC_DOC: _onehot(0)})
    kw_chunk_id = str(uuid.uuid4())
    store.upsert(
        [
            _chunk(KEYWORD_DOC, 0, kb_id=kb, doc_id=doc, vector=_onehot(1), chunk_id=kw_chunk_id),
            _chunk(SEMANTIC_DOC, 1, kb_id=kb, doc_id=doc, vector=_onehot(0)),
        ]
    )

    dense_only = HybridRetriever(
        embedder=embedder, vector_store=store, hybrid_enabled=False,
        top_k=10, min_similarity=0.35, vector_weight=1.0, bm25_weight=0.0,
    )
    dense_ids = [item.chunk.id for item in dense_only.retrieve(kb, QUERY)]
    check(
        "纯向量检索：关键词文档召回不到（构造前提成立）",
        kw_chunk_id not in dense_ids,
        f"densely 召回了 {dense_ids}",
    )
    check("纯向量检索：语义相近的文档召回到了", len(dense_ids) == 1)

    hybrid = HybridRetriever(
        embedder=embedder, vector_store=store, hybrid_enabled=True,
        top_k=10, min_similarity=0.35, vector_weight=0.5, bm25_weight=0.5,
    )
    trace = hybrid.recall(kb, QUERY)
    fused_ids = [item.chunk.id for item in trace.fused]
    check(
        "混合检索：关键词文档被 BM25 路召回并进入融合结果 ★ 核心价值",
        kw_chunk_id in fused_ids,
        f"融合结果 {fused_ids}",
    )
    check(
        "混合检索：语义文档也还在（两路互补而非替代）",
        any(item.chunk.id != kw_chunk_id for item in trace.fused),
    )
    check(
        "语义文档没被 BM25 路召回（关键词匹配对语义改写无能为力）",
        all(item.chunk.id != next(
            (i.chunk.id for i in trace.vector_hits if i.chunk.text == SEMANTIC_DOC), None
        ) for item in trace.bm25_hits),
        f"BM25 命中 {[i.chunk.text for i in trace.bm25_hits]}",
    )
    print(f"         [召回明细] {trace.describe()}")

    # 权重方向性
    vector_heavy = HybridRetriever(
        embedder=embedder, vector_store=store, hybrid_enabled=True,
        top_k=10, min_similarity=0.35, vector_weight=1.0, bm25_weight=0.0001,
    )
    bm25_heavy = HybridRetriever(
        embedder=embedder, vector_store=store, hybrid_enabled=True,
        top_k=10, min_similarity=0.35, vector_weight=0.0001, bm25_weight=1.0,
    )
    v_first = vector_heavy.retrieve(kb, QUERY)[0].chunk.text
    b_first = bm25_heavy.retrieve(kb, QUERY)[0].chunk.text
    check("权重压倒向量时，语义文档排第一", v_first == SEMANTIC_DOC, f"实际 {v_first!r}")
    check("权重压倒 BM25 时，关键词文档排第一", b_first == KEYWORD_DOC, f"实际 {b_first!r}")

    # top_k 覆盖
    check("top_k 覆写生效", len(hybrid.retrieve(kb, QUERY, top_k=1)) == 1)

    # 空查询
    check("空查询返回空列表", hybrid.retrieve(kb, "   ") == [])
    check("空查询不产生任何召回调用", hybrid.recall(kb, "").fused == [])


def check_single_path_skipped(store: MilvusVectorStore) -> None:
    section("6. 单路跳过与失败降级（用 spy 观察调用行为）")
    kb = f"kb-{uuid.uuid4().hex[:8]}"
    doc = str(uuid.uuid4())
    dense_doc = "向量检索使用双塔模型计算余弦相似度。"
    embedder = FakeEmbedder({QUERY: _onehot(0), KEYWORD_DOC: _onehot(1)})
    # 两篇：一篇只有 BM25 召得回（向量正交），一篇只有向量召得回（不含关键词）。
    # 这样"降级为纯向量"才可能还有结果——否则两路同时失效，测不出降级是否真的生效。
    store.upsert(
        [
            _chunk(KEYWORD_DOC, 0, kb_id=kb, doc_id=doc, vector=_onehot(1)),
            _chunk(dense_doc, 1, kb_id=kb, doc_id=doc, vector=_onehot(0)),
        ]
    )

    spy = RecordingStore(store)
    # 显式给阈值保持"向量路召不回关键词文档"这个前提（小节的场景依赖它）
    both = HybridRetriever(embedder=embedder, vector_store=spy, hybrid_enabled=True, top_k=5,
                           min_similarity=0.35)
    both.retrieve(kb, QUERY)
    check("两路权重都非 0 → 两路各跑一次", spy.dense_calls == 1 and spy.bm25_calls == 1,
          f"dense={spy.dense_calls} bm25={spy.bm25_calls}")

    spy.dense_calls = spy.bm25_calls = 0
    bm25_only = HybridRetriever(
        embedder=embedder, vector_store=spy, hybrid_enabled=True, top_k=5,
        vector_weight=0.0, bm25_weight=1.0,
    )
    bm25_only.retrieve(kb, QUERY)
    check("vector_weight=0 → **不发起**向量检索（省一次往返）",
          spy.dense_calls == 0, f"dense={spy.dense_calls}")
    check("vector_weight=0 → BM25 路照跑", spy.bm25_calls == 1)

    spy.dense_calls = spy.bm25_calls = 0
    dense_only = HybridRetriever(
        embedder=embedder, vector_store=spy, hybrid_enabled=True, top_k=5,
        vector_weight=1.0, bm25_weight=0.0,
    )
    dense_only.retrieve(kb, QUERY)
    check("bm25_weight=0 → **不发起** BM25 检索", spy.bm25_calls == 0, f"bm25={spy.bm25_calls}")
    check("bm25_weight=0 → 向量路照跑", spy.dense_calls == 1)

    spy.dense_calls = spy.bm25_calls = 0
    disabled = HybridRetriever(embedder=embedder, vector_store=spy, hybrid_enabled=False, top_k=5)
    disabled.retrieve(kb, QUERY)
    check("hybrid_enabled=False → 只跑向量路", spy.dense_calls == 1 and spy.bm25_calls == 0)

    # 降级：BM25 路报错不应打断整条链路
    broken = RecordingStore(store, fail_bm25=True)
    degraded = HybridRetriever(embedder=embedder, vector_store=broken, hybrid_enabled=True,
                               top_k=5, min_similarity=0.35)
    results = degraded.retrieve(kb, QUERY)
    check("BM25 路抛错时降级为纯向量检索（向量路的结果仍返回）", len(results) == 1,
          f"实际 {len(results)} 条：{[item.chunk.text for item in results]}")
    check("降级后的结果正是向量路能召回的那篇",
          bool(results) and results[0].chunk.text == dense_doc,
          f"实际 {[item.chunk.text for item in results]}")
    check("降级后结果不带 bm25_score", all(item.bm25_score is None for item in results))
    check("降级只影响 BM25 路（向量路仍被调用）", broken.dense_calls == 1 and broken.bm25_calls == 1)

    # 两路权重同时为 0 属于配置错误，构造期就该拒绝
    try:
        HybridRetriever(embedder=embedder, vector_store=store, hybrid_enabled=True,
                        vector_weight=0.0, bm25_weight=0.0)
    except ValueError as exc:
        check("两路权重同时为 0 → 构造期 ValueError", "0" in str(exc))
    except Exception as exc:  # noqa: BLE001
        check("两路权重同时为 0 → 构造期 ValueError", False, f"实际 {type(exc).__name__}")
    else:
        check("两路权重同时为 0 → 构造期 ValueError", False, "没有抛异常")


# ---------------------------------------------------------------------------
# 7. RetrievalService：重排与截断
# ---------------------------------------------------------------------------


def _build_service(store, *, reranker=None, embedder=None) -> RetrievalService:
    """装配一个用假模型驱动、但检索真的打到 Milvus 的检索服务。"""
    retriever = HybridRetriever(
        embedder=embedder or FakeEmbedder({QUERY: _onehot(0), KEYWORD_DOC: _onehot(1)}),
        vector_store=store,
        hybrid_enabled=True,
        top_k=10,
        min_similarity=0.35,
        vector_weight=0.5,
        bm25_weight=0.5,
    )
    return RetrievalService(retriever=retriever, reranker=reranker)


def check_retrieval_service(store: MilvusVectorStore) -> None:
    section("7. RetrievalService：重排、截断与只跑一次召回")
    kb = f"kb-{uuid.uuid4().hex[:8]}"
    doc = str(uuid.uuid4())
    limit_doc = "预算不得超过十万元，超出部分需重新审批。"
    other_doc = "报销需要提交发票原件。"
    # other_doc 的向量与查询成 45°：余弦 ≈ 0.707，高于 MIN_SIMILARITY 会被召回，
    # 但排在 limit_doc（余弦 1.0）之后——这样"两条候选、谁先谁后"都是确定的
    tilted = [1.0, 1.0] + [0.0] * (DIM - 2)
    embedder = FakeEmbedder({QUERY: _onehot(0)})
    store.upsert(
        [
            _chunk(limit_doc, 0, kb_id=kb, doc_id=doc, vector=_onehot(0)),
            _chunk(other_doc, 1, kb_id=kb, doc_id=doc, vector=tilted),
        ]
    )

    # --- 无重排：按融合分截断 ---
    plain = _build_service(store, embedder=embedder)
    check("未启用重排时 reranker_enabled 为 False", plain.reranker_enabled is False)
    plain_results = plain.retrieve(kb, QUERY, top_k=10, rerank_k=2)
    check("无重排时两条候选都在", len(plain_results) == 2,
          f"实际 {[i.chunk.text for i in plain_results]}")
    check("无重排时 rerank_score 全为 None",
          all(item.rerank_score is None for item in plain_results))
    check("无重排时按 fusion_score 降序",
          [i.fusion_score for i in plain_results] == sorted(
              [i.fusion_score for i in plain_results], reverse=True))
    check("rerank_k 截断生效", len(plain.retrieve(kb, QUERY, top_k=10, rerank_k=1)) == 1)

    # --- 有重排 ---
    limit_id = next(i.chunk.id for i in plain_results if i.chunk.text == limit_doc)
    other_id = next(i.chunk.id for i in plain_results if i.chunk.text == other_doc)
    reranker = FakeReranker({limit_id: 7.5, other_id: -9.0}, min_score=0.0)
    service = _build_service(store, reranker=reranker, embedder=embedder)
    check("启用重排时 reranker_enabled 为 True", service.reranker_enabled is True)

    ranked = service.retrieve(kb, QUERY, top_k=10, rerank_k=5)
    check("重排分数已回填", all(item.rerank_score is not None for item in ranked))
    check("重排按 rerank_score 降序",
          [i.rerank_score for i in ranked] == sorted([i.rerank_score for i in ranked], reverse=True))
    check("低于重排阈值的候选被丢弃（-9.0 < 0.0）",
          all(item.chunk.text != other_doc for item in ranked),
          f"实际 {[i.chunk.text for i in ranked]}")
    check("重排收到了全部 2 条候选（先过滤再截断的前提）",
          reranker.calls and reranker.calls[-1][1] == 2, f"实际 {reranker.calls}")
    check("重排 top_n 由 rerank_k 传入", reranker.calls[-1][2] == 5, f"实际 {reranker.calls}")

    reranker.calls.clear()
    check("rerank_k 在重排阶段同样截断",
          len(service.retrieve(kb, QUERY, top_k=10, rerank_k=1)) == 1)

    # --- 空查询与无候选 ---
    check("空查询返回空列表", service.retrieve(kb, "   ") == [])
    reranker.calls.clear()
    check("无候选时不调用重排器",
          service.retrieve(f"kb-{uuid.uuid4().hex[:8]}", QUERY) == []
          and not reranker.calls,
          f"实际 {reranker.calls}")

    # --- 回归：不能把一次检索跑成两次 ---
    spy = RecordingStore(store)
    service_spy = _build_service(
        spy, reranker=FakeReranker({limit_id: 1.0, other_id: 1.0}), embedder=embedder
    )

    # 清空此前所有调用记录，确保计数只反映这一次 retrieve_with_trace
    embedder.query_calls.clear()
    spy.dense_calls = spy.bm25_calls = 0
    service_spy.retrieve_with_trace(kb, QUERY, top_k=10, rerank_k=2)
    check("retrieve_with_trace 只编码查询一次", len(embedder.query_calls) == 1,
          f"实际 {len(embedder.query_calls)} 次")
    check("retrieve_with_trace 两路各只打一次 Milvus",
          spy.dense_calls == 1 and spy.bm25_calls == 1,
          f"dense={spy.dense_calls} bm25={spy.bm25_calls}")


# ---------------------------------------------------------------------------
# 8. LangGraph retrieve 节点
# ---------------------------------------------------------------------------


def check_graph_node(store: MilvusVectorStore) -> None:
    section("8. LangGraph retrieve 节点复用同一套检索")
    kb = f"kb-{uuid.uuid4().hex[:8]}"
    doc = str(uuid.uuid4())
    store.upsert([_chunk(KEYWORD_DOC, 0, kb_id=kb, doc_id=doc, vector=_onehot(1))])

    retriever = HybridRetriever(
        embedder=FakeEmbedder({QUERY: _onehot(0), KEYWORD_DOC: _onehot(1)}),
        vector_store=store, hybrid_enabled=True, top_k=5,
        vector_weight=0.0, bm25_weight=1.0,
    )
    node = make_retrieve_node(retriever)

    out = node({"kb_id": kb, "question": QUERY, "top_k": 5})
    check("节点用 question 检索并写入 candidates", len(out["candidates"]) == 1, f"{out}")
    check("节点把 rewritten_question 初始化为本次查询",
          out["rewritten_question"] == QUERY)
    check("节点返回的候选带融合分",
          out["candidates"][0].fusion_score is not None)

    out2 = node({"kb_id": kb, "question": "完全没有的东西", "rewritten_question": QUERY, "top_k": 5})
    check("rewritten_question 优先于 question", len(out2["candidates"]) == 1, f"{out2}")

    out3 = node({"kb_id": "", "question": QUERY})
    check("缺 kb_id 时短路返回空候选（不抛异常）", out3["candidates"] == [])

    out4 = node({"kb_id": kb, "question": "   "})
    check("空问题时短路返回空候选", out4["candidates"] == [])


def check_dense_threshold_mode(store: MilvusVectorStore) -> None:
    """稠密路的相似度下限按模式取不同默认值（实测出来的一个隐蔽故障）。"""
    section("9. 稠密路阈值：混合模式必须不设限（否则向量路会被静默清空）")
    kb = f"kb-{uuid.uuid4().hex[:8]}"
    doc = str(uuid.uuid4())
    # 构造"相似度落在 0.28~0.35"的真实处境：
    # 查询与文档成 72° → 余弦 = cos(72°) ≈ 0.309，低于 MIN_SIMILARITY(0.35)
    weak_doc = "巡检周期为每季度一次，发现异常需在四小时内上报运维中心。"
    weak = [1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    store.upsert([_chunk(weak_doc, 0, kb_id=kb, doc_id=doc, vector=weak)])
    embedder = FakeEmbedder({QUERY: _onehot(0)})

    norm = sum(v * v for v in weak) ** 0.5
    cosine = 1.0 / norm
    print(f"         [构造] 该片段与查询的余弦≈{cosine:.4f}（MIN_SIMILARITY={get_settings().min_similarity}）")
    check("构造前提：该片段的相似度低于 MIN_SIMILARITY",
          cosine < get_settings().min_similarity, f"{cosine:.4f}")

    # ① 纯向量模式：阈值是唯一闸门，必须设 → 召不回
    pure = HybridRetriever(embedder=embedder, vector_store=store, hybrid_enabled=False, top_k=5)
    check("纯向量模式：默认用 MIN_SIMILARITY",
          pure.dense_min_similarity == get_settings().min_similarity,
          f"实际 {pure.dense_min_similarity}")
    check("纯向量模式：低于阈值的片段被滤掉（阈值仍是必需闸门）",
          pure.retrieve(kb, QUERY) == [],
          f"实际 {[i.chunk.text for i in pure.retrieve(kb, QUERY)]}")

    # ② 混合模式：默认不设限 → 召得回
    hybrid = HybridRetriever(embedder=embedder, vector_store=store, hybrid_enabled=True, top_k=5)
    check("混合模式：默认用 HYBRID_DENSE_MIN_SIMILARITY（0 = 不设限）",
          hybrid.dense_min_similarity == get_settings().hybrid_dense_min_similarity,
          f"实际 {hybrid.dense_min_similarity}")
    check("混合模式：同样这条片段不再被滤掉 ★ 本轮修掉的故障",
          len(hybrid.retrieve(kb, QUERY)) == 1,
          f"实际 {[i.chunk.text for i in hybrid.retrieve(kb, QUERY)]}")
    check("召回结果里带 vector_score（证明是稠密路召回的）",
          hybrid.retrieve(kb, QUERY)[0].vector_score is not None)

    # ③ 显式传参能覆盖默认值（留出的退路）
    forced = HybridRetriever(
        embedder=embedder, vector_store=store, hybrid_enabled=True, top_k=5, min_similarity=0.35
    )
    check("显式传 min_similarity 可覆盖模式默认值（留出的退路）",
          forced.dense_min_similarity == 0.35 and forced.retrieve(kb, QUERY) == [])

    # ④ trace 必须带上这个值：它设错时不会报错、只会变差
    trace = hybrid.recall(kb, QUERY)
    check("RecallTrace 带上了生效的稠密下限（可在一行日志里看到）",
          trace.dense_min_similarity == hybrid.dense_min_similarity)
    check("日志摘要里含该值", "稠密下限" in trace.describe(), trace.describe())


# ---------------------------------------------------------------------------
# 10. 动态 k
# ---------------------------------------------------------------------------


class _ScoringStore:
    """按预设分数返回结果的假向量库，并记录每次 limit。

    为什么要记录 limit：adaptive 的第一步是"按上限过采样、再截断"，
    若实现成"先按 K_MIN 取"，索引里排第 5 的正确片段根本不会出现在结果里，
    断崖也就永远看不到——这是个不会报错、只会变差的实现错误。
    """

    def __init__(self, dense: list[float], sparse: list[float]) -> None:
        self._dense = dense
        self._sparse = sparse
        self.dense_limits: list[int] = []
        self.bm25_limits: list[int] = []

    def search(self, kb_id, query_embedding, top_k, min_score=None):  # noqa: ANN001
        self.dense_limits.append(int(top_k))
        return [
            _hit(f"d{i}", f"稠密片段{i}", vector=score)
            for i, score in enumerate(self._dense[: int(top_k)], start=1)
        ]

    def search_bm25(self, kb_id, query, top_k, min_score=None):  # noqa: ANN001
        self.bm25_limits.append(int(top_k))
        return [
            _hit(f"b{i}", f"稀疏片段{i}", bm25=score)
            for i, score in enumerate(self._sparse[: int(top_k)], start=1)
        ]


def check_dynamic_k() -> None:
    """动态 k：解析只留一处 + adaptive 只收缩、不放大。"""
    section("10. 动态 k：统一解析 + 按本路分数的断层截断")

    # --- 10.1 resolve_k：请求值 → 配置 → 硬上限 ---
    check("resolve_k：None 用配置默认值", resolve_k(None, default=8) == 8)
    check("resolve_k：显式值覆盖默认值", resolve_k(3, default=8) == 3)
    check(
        "resolve_k：超过硬上限被夹住（HTTP 的 le 不是唯一防线）",
        resolve_k(10**6, default=8) == TOP_K_HARD_LIMIT,
        f"实际 {resolve_k(10**6, default=8)}",
    )
    check(
        "resolve_k：0 原样返回，不偷偷改成 1（下游按空结果处理）",
        resolve_k(0, default=8) == 0,
    )

    # --- 10.2 adaptive_k：纯函数，先把算法钉住 ---
    check(
        "adaptive_k：明显断层 → 断在 2 条",
        adaptive_k([0.90, 0.88, 0.30, 0.28], k_min=2, k_max=8, drop_ratio=0.25) == 2,
    )
    check(
        "adaptive_k：平滑衰减不构成断层 → 取满上限",
        adaptive_k([0.30, 0.29, 0.28, 0.27], k_min=2, k_max=8, drop_ratio=0.25) == 4,
    )
    check(
        "adaptive_k：结果不足下限时不凭空造结果",
        adaptive_k([0.31], k_min=4, k_max=8, drop_ratio=0.25) == 1,
    )
    check(
        "adaptive_k：空输入 → 0",
        adaptive_k([], k_min=4, k_max=16, drop_ratio=0.25) == 0,
    )
    check(
        "adaptive_k：全 0 分没有可比信息 → 取满上限",
        adaptive_k([0.0, 0.0, 0.0, 0.0], k_min=2, k_max=8, drop_ratio=0.25) == 4,
    )

    # --- 10.3 检索器：adaptive 的过采样与截断 ---
    store = _ScoringStore(
        dense=[0.90, 0.88, 0.30, 0.28],
        sparse=[9.0, 8.9, 8.8, 8.7],
    )
    adaptive = HybridRetriever(
        embedder=FakeEmbedder(),
        vector_store=store,
        hybrid_enabled=True,
        top_k=8,
        k_mode="adaptive",
        k_min=2,
        k_max=8,
        k_drop_ratio=0.25,
    )
    trace = adaptive.recall("kb-dyn", "查询")
    check(
        "adaptive：两路各按上限取（8 条），没有提前截断",
        store.dense_limits == [8] and store.bm25_limits == [8],
        f"dense={store.dense_limits} bm25={store.bm25_limits}",
    )
    check(
        "adaptive：稠密路断在 2 条、BM25 路取满 4 条（各看各的分数）",
        trace.k_dense == 2 and trace.k_bm25 == 4,
        f"dense={trace.k_dense} bm25={trace.k_bm25}",
    )
    check(
        "adaptive：融合后保留 max(2,4)=4 条",
        trace.k_selected == 4 and len(trace.fused) == 4,
        f"k_selected={trace.k_selected} fused={len(trace.fused)}",
    )
    check(
        "adaptive：日志摘要能直接看出本次用的 k",
        "k=adaptive:4" in trace.describe(),
        trace.describe(),
    )

    # --- 10.4 fixed 模式必须与改造前完全一致 ---
    fixed_store = _ScoringStore(dense=[0.9, 0.8, 0.7, 0.6], sparse=[9, 8, 7, 6])
    fixed = HybridRetriever(
        embedder=FakeEmbedder(),
        vector_store=fixed_store,
        hybrid_enabled=True,
        top_k=3,
        k_mode="fixed",
    )
    fixed_trace = fixed.recall("kb-dyn", "查询")
    check(
        "fixed：两路各取 3 条（与改造前一致）",
        fixed_store.dense_limits == [3] and fixed_store.bm25_limits == [3],
        f"dense={fixed_store.dense_limits} bm25={fixed_store.bm25_limits}",
    )
    check(
        "fixed：融合后截 3 条、日志标注 fixed",
        fixed_trace.k_selected == 3 and "k=fixed:3" in fixed_trace.describe(),
        fixed_trace.describe(),
    )

    # --- 10.5 adaptive 不放大调用方要求的深度 ---
    capped_store = _ScoringStore(dense=[0.9, 0.8, 0.7, 0.6, 0.5], sparse=[])
    capped = HybridRetriever(
        embedder=FakeEmbedder(),
        vector_store=capped_store,
        hybrid_enabled=False,
        top_k=8,
        k_mode="adaptive",
        k_min=2,
        k_max=4,
    )
    capped.recall("kb-dyn", "查询", top_k=50)
    check(
        "adaptive：K_MAX 拦住显式传大的 top_k（只取 4 条）",
        capped_store.dense_limits == [4],
        f"dense={capped_store.dense_limits}",
    )

    small_store = _ScoringStore(dense=[0.9, 0.8], sparse=[])
    small = HybridRetriever(
        embedder=FakeEmbedder(),
        vector_store=small_store,
        hybrid_enabled=False,
        top_k=8,
        k_mode="adaptive",
        k_min=4,
        k_max=8,
    )
    small_trace = small.recall("kb-dyn", "查询", top_k=1)
    check(
        "adaptive：请求深度小于 K_MIN 时不放大（K_MIN 不是强制下限）",
        small_trace.k_selected == 1 and len(small_trace.fused) == 1,
        f"k_selected={small_trace.k_selected} fused={len(small_trace.fused)}",
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _count_real(store: MilvusVectorStore, collection: str) -> int:
    """统计真实 collection 的行数（只读，用于证明本次自检没碰它）。"""
    store.client.load_collection(collection)
    rows = store.client.query(
        collection_name=collection, filter='id != ""', output_fields=["count(*)"]
    )
    if not rows:
        return 0
    return int(next(iter(rows[0].values())))


def main() -> int:
    settings = get_settings()
    print("混合检索（向量 + BM25 → RRF）自检")
    print(f"  Milvus URI       : {settings.milvus_uri}")
    print(f"  临时 collection  : {COLLECTION}（维度 {DIM}）")
    print(f"  真实 collection  : {settings.milvus_collection}（全程不触碰）")
    print(f"  混合检索配置      : enabled={settings.hybrid_enabled} "
          f"k={settings.hybrid_rrf_k} 权重={settings.hybrid_vector_weight}/"
          f"{settings.hybrid_bm25_weight} 召回深度={settings.retrieve_top_k}"
          f" 模式={settings.retrieve_k_mode}")

    real_rows_before: int | None = None
    store = MilvusVectorStore(collection=COLLECTION)
    try:
        if store.client.has_collection(settings.milvus_collection):
            real_rows_before = _count_real(store, settings.milvus_collection)
            print(f"  真实集合现有行数   : {real_rows_before}（结束时复核）")

        check_contract_and_schema(store)
        check_schema_drift(store)
        check_fusion_pure()
        check_fusion_dedupe()
        check_error_semantics()
        check_rerank_failures()
        check_bm25_search(store)
        check_hybrid_value(store)
        check_single_path_skipped(store)
        check_retrieval_service(store)
        check_graph_node(store)
        check_dense_threshold_mode(store)
        check_dynamic_k()

        section("11. 收尾与残留检查")
        dropped = store.drop_collection()
        check("临时 collection 已删除", dropped and not store.client.has_collection(COLLECTION))
        check("旧 schema 的临时 collection 也未残留",
              not store.client.has_collection(LEGACY_COLLECTION))

        if real_rows_before is not None:
            after = _count_real(store, settings.milvus_collection)
            check(
                f"真实集合 {settings.milvus_collection} 未被触碰（行数仍为 {real_rows_before}）",
                after == real_rows_before,
                f"实际 {after}",
            )
    finally:
        try:
            for name in (COLLECTION, LEGACY_COLLECTION):
                if store.client.has_collection(name):
                    store.client.drop_collection(name)
                    print(f"\n[清理] 已删除临时 collection {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[清理失败] {exc}")
        store.close()

    print()
    if _failures:
        print(f"结果：{_checks} 项断言，{len(_failures)} 项失败")
        for label in _failures:
            print(f"  - {label}")
        return 1
    print(f"结果：{_checks} 项断言全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
