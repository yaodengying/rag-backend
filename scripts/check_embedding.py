# -*- coding: utf-8 -*-
"""向量化自检：验证「文本 → 稠密向量」这一步本身是对的，以及它在流水线第 5 步里的行为。

为什么需要这个脚本：
    `check_models.py` 把关的是**模型**（维度、区分度、重排尺度），
    `check_loading.py` / `check_splitting.py` 把关的是第 3、4 步。
    但"向量化作为入库流水线的第 5 步"此前没有独立检查，而它的失效
    **多数是静默的**，不会抛异常，只会让检索质量悄悄变差：
      - 向量少一条 → 只有数量不匹配才报错，恰好配对时就悄悄丢块；
      - L2 没归一化 → 内积不再等于余弦，Milvus 的 COSINE 度量整体失真；
      - 分批处错位 → 块与向量对不上，检索命中了却引用到别的原文；
      - 空文本被编码 → BGE 不报错，返回一条无语义的 padding 向量污染索引。

判据（不只断言"不报错"）：

    | 检查项        | 期望                                                          |
    |--------------|---------------------------------------------------------------|
    | 维度来源      | embedder.dimension == 权重 config.json 的 hidden_size          |
    | 配置交叉校验   | 权重维度 != MILVUS_DIMENSION 时构造即抛 ConfigurationError      |
    | 配置缺失降级   | 读不到 config.json 时退回 MILVUS_DIMENSION（不崩，告警）        |
    | 形状          | 返回条数与输入等长；每条 == dimension；元素为 float             |
    | L2 归一化     | 每条向量 L2 范数 = 1.0（余弦检索的前提）                        |
    | 确定性        | 同一批文本两次编码逐位一致                                      |
    | 指令隔离      | embed_query(q) != embed_documents([q])[0]（指令只加 query 侧）  |
    | 空输入        | embed_documents([]) == []（FlagEmbedding 对空列表会抛异常）      |
    | 空白拒绝      | 空串 / 纯空白 → ValidationError（不写入无意义向量）             |
    | 跨批一致性    | 条数超过 batch_size 时不错位、不丢条；与逐条编码的差异仅为浮点末位（<1e-5）|
    | 步骤层        | 真 embedder + 假上下游跑 IngestionPipeline：块全带向量、一一对应 |
    | 失败语义      | 写入失败 → failed + 清理残留；切分为空 → failed；空白块 → failed |
    | 向量化异常    | 前向异常 / 条数不符 / 维度不符 / NaN → EmbeddingError(502)       |
    | 配置异常透传  | ConfigurationError 不被包成 EmbeddingError（该改 .env）          |

运行（必须用项目自己的 .venv）：
    .venv\\Scripts\\python.exe scripts\\check_embedding.py

    需要本地 BGE 权重（EMBEDDING_MODEL_PATH），走 CPU，不需要 GPU。
    步骤层用例使用**假的** parser / splitter / 仓储 / 向量库：
    不写 SQLite、不连 Milvus、不碰 data/uploads。

退出码：0 全部通过；1 存在失败项。
"""

from __future__ import annotations

import asyncio
import math
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# settings 的 env_file 是相对路径，必须先切到项目根
os.chdir(PROJECT_ROOT)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from app.core.config import get_settings  # noqa: E402
from app.core.exceptions import (  # noqa: E402
    ConfigurationError,
    EmbeddingError,
    RetrievalError,
    ValidationError,
)
from app.domain.models import (  # noqa: E402
    DOC_STATUS_FAILED,
    DOC_STATUS_READY,
    Chunk,
    Document,
    Page,
)
from app.domain.ports import (  # noqa: E402
    ChunkRepositoryPort,
    DocumentParserPort,
    EmbedderPort,
    SplitterPort,
    VectorStorePort,
)
from app.infrastructure.embedding import BgeEmbedder, read_declared_dimension  # noqa: E402
from app.rag.ingest import IngestionPipeline  # noqa: E402

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
            print(f"  [ok]   {label}（{type(exc).__name__}: {str(exc)[:60]}…）")
        else:
            print(f"  [FAIL] {label}  -> 消息里没有 {keyword!r}：{exc}")
            _failures.append(label)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {label}  -> 期望 {exc_type.__name__}，实际 {type(exc).__name__}: {exc}")
        _failures.append(label)
    else:
        print(f"  [FAIL] {label}  -> 没有抛异常")
        _failures.append(label)


def l2(vector: list[float]) -> float:
    """向量的 L2 范数。"""
    return math.sqrt(sum(x * x for x in vector))


# ---------------------------------------------------------------------------
# 假的上下游：只为了脱离 Milvus / SQLite 跑流水线第 5 步
# 刻意不继承端口（Protocol 的意义就是结构化鸭子类型），
# 但下面会显式断言它们确实满足各自的端口契约。
# ---------------------------------------------------------------------------


class FakeParser:
    """假解析器：不读磁盘，直接返回预置页。"""

    supported_extensions = (".txt",)

    def __init__(self, pages: list[Page]) -> None:
        self._pages = pages

    def parse(self, file_path: str) -> list[Page]:
        return list(self._pages)


class FakeSplitter:
    """假切分器：直接返回预置块（复制一份，避免被流水线就地改写污染用例）。"""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def split(self, pages: list[Page]) -> list[Chunk]:
        return [
            Chunk(
                id=c.id,
                kb_id=c.kb_id,
                doc_id=c.doc_id,
                text=c.text,
                chunk_index=c.chunk_index,
                page_start=c.page_start,
                page_end=c.page_end,
                filename=c.filename,
            )
            for c in self._chunks
        ]


class FakeRepository:
    """假仓储：只记录调用，不落库。"""

    def __init__(self) -> None:
        self.created: list[tuple[str, str]] = []
        self.ready: tuple[str, int] | None = None
        self.failed: tuple[str, str] | None = None
        self.processing: list[str] = []

    def list_knowledge_bases(self) -> list[dict]:
        return []

    def get_knowledge_base(self, kb_id: str) -> dict | None:
        return {"id": kb_id, "name": "fake"}

    def create_knowledge_base(self, name: str, description: str) -> dict:
        return {"id": str(uuid.uuid4()), "name": name, "description": description}

    def update_knowledge_base(self, kb_id: str, name: str, description: str) -> dict:
        return {"id": kb_id, "name": name, "description": description}

    def delete_knowledge_base(self, kb_id: str) -> bool:
        return True

    def list_documents(self, kb_id: str) -> list[Document]:
        return []

    def get_document(self, doc_id: str) -> Document | None:
        return None

    def create_document(self, kb_id: str, filename: str, content_hash: str = "") -> Document:
        document = Document(id=str(uuid.uuid4()), kb_id=kb_id, filename=filename)
        self.created.append((kb_id, filename))
        return document

    def find_document_by_hash(self, kb_id: str, content_hash: str) -> Document | None:
        # 本脚本不测去重；端口要求实现，返回 None 表示"没查到重复"
        return None

    def mark_document_ready(self, doc_id: str, chunk_count: int) -> None:
        self.ready = (doc_id, chunk_count)

    def mark_document_failed(self, doc_id: str, error: str) -> None:
        self.failed = (doc_id, error)

    def mark_document_processing(self, doc_id: str) -> None:
        self.processing.append(doc_id)

    def delete_document(self, doc_id: str) -> str | None:
        return None


class FakeVectorStore:
    """假向量库：记录写入内容；可被要求在某一步抛错，用于验证失败语义。"""

    def __init__(self, fail_on_upsert: bool = False) -> None:
        self.written: list[Chunk] = []
        self.deleted: list[str] = []
        self.ensured: list[int] = []
        self._fail_on_upsert = fail_on_upsert

    def ensure_collection(self, dimension: int) -> None:
        self.ensured.append(dimension)

    def upsert(self, chunks: list[Chunk]) -> int:
        if self._fail_on_upsert:
            raise RuntimeError("模拟向量库写入失败")
        self.written.extend(chunks)
        return len(chunks)

    def search(self, kb_id, query_embedding, top_k, min_score=None):  # noqa: ANN001, ANN201
        return []

    def search_bm25(self, kb_id, query, top_k, min_score=None):  # noqa: ANN001, ANN201
        """本步骤（向量化）用不到全文检索，返回空即符合端口约定。

        按 `VectorStorePort.search_bm25` 的约定：不支持全文检索的实现
        返回空列表而不是抛错，让上层退化成纯向量检索。
        """
        return []

    def delete_by_doc(self, doc_id: str) -> int:
        self.deleted.append(doc_id)
        return 0

    def delete_by_kb(self, kb_id: str) -> int:
        return 0


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------

# 内容互不相同，避免"跨批结果无重复"这类断言因文本雷同而失真
_SENTENCES = (
    "Milvus 是向量数据库，支持余弦相似度检索。",
    "交叉编码器把查询与文档拼在一起过模型，精度高但慢。",
    "食堂今天供应红烧肉与清炒时蔬。",
    "文心一格可以按文本描述生成插图。",
    "FastAPI 用依赖注入组织请求级资源。",
    "递归字符切分先按段落断，再退到句子。",
)


def make_chunks(total: int, page: int = 1) -> list[Chunk]:
    """造 total 个内容互不相同的块。"""
    return [
        Chunk(
            id="",
            kb_id="",
            doc_id="",
            text=f"第 {i} 段：{_SENTENCES[i % len(_SENTENCES)]}",
            chunk_index=i,
            page_start=page,
            page_end=page,
        )
        for i in range(total)
    ]


def case_capability(embedder: BgeEmbedder) -> None:
    """0. 端口契约：实现类真的满足 EmbedderPort。"""
    print("\n[0] 端口契约")
    check("BgeEmbedder 满足 EmbedderPort（runtime_checkable）", isinstance(embedder, EmbedderPort))
    check("FakeParser 满足 DocumentParserPort", isinstance(FakeParser([]), DocumentParserPort))
    check("FakeSplitter 满足 SplitterPort", isinstance(FakeSplitter([]), SplitterPort))
    check("FakeRepository 满足 ChunkRepositoryPort", isinstance(FakeRepository(), ChunkRepositoryPort))
    check("FakeVectorStore 满足 VectorStorePort", isinstance(FakeVectorStore(), VectorStorePort))


def case_dimension(embedder: BgeEmbedder, workdir: Path) -> None:
    """1. 维度来源：权重为权威，配置做交叉校验。"""
    print("\n[1] 维度来源与交叉校验")
    settings = get_settings()

    declared = read_declared_dimension(settings.embedding_model_path)
    check(
        "能从权重 config.json 读出 hidden_size",
        declared is not None,
        f"读不到：{settings.embedding_model_path}\\config.json 缺失或字段名不符",
    )
    check(
        "权重声明维度 == MILVUS_DIMENSION",
        declared == settings.milvus_dimension,
        f"权重 {declared} vs 配置 {settings.milvus_dimension}",
    )
    check(
        "embedder.dimension 取自权重（而不是配置值）",
        embedder.dimension == declared,
        f"{embedder.dimension} vs {declared}",
    )
    check("dimension 为正整数", isinstance(embedder.dimension, int) and embedder.dimension > 0)

    # 维度不符：必须构造即失败，而不是先把 collection 按错的维度建出来
    mismatch_dir = workdir / "mismatch-model"
    mismatch_dir.mkdir(parents=True, exist_ok=True)
    wrong = 768 if embedder.dimension != 768 else 512
    (mismatch_dir / "config.json").write_text(
        f'{{"model_type": "bert", "hidden_size": {wrong}}}', encoding="utf-8"
    )
    expect_error(
        f"权重声明 {wrong} 维 != MILVUS_DIMENSION 时抛 ConfigurationError",
        ConfigurationError,
        "向量维度不一致",
        lambda: BgeEmbedder(model_path=str(mismatch_dir)),
    )

    # 读不到 config.json：降级为声明值，不崩（最终由加载时实测兜底）
    blank_dir = workdir / "no-config-model"
    blank_dir.mkdir(parents=True, exist_ok=True)
    check(
        "读不到 config.json 时 read_declared_dimension 返回 None",
        read_declared_dimension(str(blank_dir)) is None,
    )
    check(
        "读不到 config.json 时降级为 MILVUS_DIMENSION",
        BgeEmbedder(model_path=str(blank_dir)).dimension == settings.milvus_dimension,
    )


def case_contract(embedder: BgeEmbedder) -> None:
    """2. 向量契约：形状、L2、类型、确定性、指令隔离。"""
    print("\n[2] 向量契约（形状 / L2 / 类型 / 确定性）")
    texts = [_SENTENCES[0], _SENTENCES[2]]

    vectors = embedder.embed_documents(texts)
    check("返回条数与输入等长", len(vectors) == len(texts), f"{len(vectors)} vs {len(texts)}")
    check(
        "每条维度 == embedder.dimension",
        all(len(v) == embedder.dimension for v in vectors),
        f"{[len(v) for v in vectors]}",
    )
    check("元素都是 float（Milvus 与 JSON 可直接消费）", all(isinstance(x, float) for v in vectors for x in v))
    norms = [l2(v) for v in vectors]
    check(
        "L2 范数 == 1.0（余弦检索的前提）",
        all(abs(n - 1.0) < 1e-3 for n in norms),
        f"{[round(n, 6) for n in norms]}",
    )
    check(
        "分量落在 [-1, 1]",
        all(-1.0001 <= x <= 1.0001 for v in vectors for x in v),
    )
    check("每条向量是独立对象（无共享引用）", vectors[0] is not vectors[1])
    check("两次编码逐位一致（CPU 上可复现）", embedder.embed_documents(texts) == vectors)

    query = "哪个是向量数据库？"
    query_vector = embedder.embed_query(query)
    check("查询向量维度正确", len(query_vector) == embedder.dimension)
    check("查询向量 L2 == 1.0", abs(l2(query_vector) - 1.0) < 1e-3)
    check(
        "embed_query(q) != embed_documents([q])（检索指令只加在 query 侧）",
        query_vector != embedder.embed_documents([query])[0],
        "两者相等，说明指令没生效、或两侧走了同一条编码路径",
    )


def case_edges(embedder: BgeEmbedder) -> None:
    """3. 边界：空输入短路、空白文本拒绝。"""
    print("\n[3] 边界：空输入与空白文本")
    check(
        "空列表返回空列表（FlagEmbedding 对 encode_corpus([]) 会抛 AttributeError）",
        embedder.embed_documents([]) == [],
    )
    expect_error(
        "空串 → ValidationError",
        ValidationError,
        "待编码文本为空",
        lambda: embedder.embed_documents([""]),
    )
    expect_error(
        "纯空白 → ValidationError",
        ValidationError,
        "待编码文本为空",
        lambda: embedder.embed_documents(["  \n\t "]),
    )
    expect_error(
        "混在正常文本里也要报错，且指出是第几条",
        ValidationError,
        "第 1 条",
        lambda: embedder.embed_documents(["正常文本", ""]),
    )
    expect_error(
        "空查询 → ValidationError",
        ValidationError,
        "查询文本为空",
        lambda: embedder.embed_query("   "),
    )


def case_batching(embedder: BgeEmbedder) -> None:
    """4. 跨批一致性：条数超过 batch_size 时不许错位或丢条。"""
    batch = embedder.batch_size
    total = batch * 2 + 3
    print(f"\n[4] 跨批一致性（{total} 条 > batch_size {batch}）")

    texts = [f"第 {i} 段：{_SENTENCES[i % len(_SENTENCES)]}" for i in range(total)]
    batched = embedder.embed_documents(texts)
    check("跨批返回条数正确", len(batched) == total, f"{len(batched)} vs {total}")
    check("每条维度正确", all(len(v) == embedder.dimension for v in batched))
    check("所有向量互不相同（未跨批复制同一向量）", len({tuple(v) for v in batched}) == total)
    check("同批形状重跑逐位一致（可复现）", embedder.embed_documents(texts) == batched)

    # 分批 vs 逐条：**不能断言逐位相等**。批形状不同会让矩阵乘的归约顺序
    # 不同，float32 末位必然有差异（本机实测最大 1.19e-7 = 2^-23，正好是
    # float32 在 1.0 处的 eps；余弦相似度 0.99999985）。这是数值性质而非 bug，
    # 断言要按容差比，否则会得到一个永远修不好的假失败。
    singles = [embedder.embed_documents([text])[0] for text in texts]
    max_diff = max(
        max(abs(a - b) for a, b in zip(batched_vec, single_vec))
        for batched_vec, single_vec in zip(batched, singles)
    )
    check(
        f"跨批结果与逐条编码一致（最大绝对差 {max_diff:.2e} < 1e-5）",
        max_diff < 1e-5,
        "差异过大，说明不是浮点末位问题，而是分批逻辑丢条或错位",
    )
    lowest_cosine = min(
        sum(a * b for a, b in zip(batched_vec, single_vec))
        for batched_vec, single_vec in zip(batched, singles)
    )
    check(
        f"跨批与逐条的余弦相似度 ≈ 1（最低 {lowest_cosine:.8f}）",
        lowest_cosine > 1 - 1e-6,
        "相似度偏低：同一文本在不同批次下编码结果对不上",
    )


def case_pipeline_success(embedder: BgeEmbedder, workdir: Path) -> None:
    """5. 步骤层：入库流水线第 5 步（真 embedder + 假上下游）。"""
    total = embedder.batch_size * 2 + 6
    print(f"\n[5] 步骤层：IngestionPipeline 第 5 步（{total} 块，跨 {total // embedder.batch_size + 1} 个批次）")

    chunks = make_chunks(total)
    repository = FakeRepository()
    store = FakeVectorStore()
    pipeline = IngestionPipeline(
        parser=FakeParser([Page(page_no=1, text="假文档正文")]),
        splitter=FakeSplitter(chunks),
        embedder=embedder,
        vector_store=store,
        repository=repository,
        upload_dir=str(workdir / "uploads"),
    )

    document = asyncio.run(pipeline.run("kb-fake", "sample.txt", b"fake bytes"))
    check("文档状态为 ready", document.status == DOC_STATUS_READY, document.error)
    check("chunk_count == 切分块数", document.chunk_count == total, f"{document.chunk_count} vs {total}")
    check("向量库收到全部块", len(store.written) == total, f"{len(store.written)} vs {total}")
    check(
        "每个块都回填了向量，且维度正确",
        all(c.embedding and len(c.embedding) == embedder.dimension for c in store.written),
        "存在为空或维度不符的向量",
    )
    check("每个块都被分配了 id（切分器不填，由流水线补）", all(c.id for c in store.written))
    check(
        "块的 kb_id / doc_id / filename 已补齐",
        all(
            c.kb_id == "kb-fake" and c.doc_id == document.id and c.filename == "sample.txt"
            for c in store.written
        ),
    )
    check("chunk_index 保持顺序未被打乱", [c.chunk_index for c in store.written] == list(range(total)))
    check("仓储收到 ready 标记且条数一致", repository.ready == (document.id, total))
    check("成功路径不触发残留清理", store.deleted == [])

    # 向量与文本的对应关系：按写入顺序重编码一次，逐条比对
    reencoded = embedder.embed_documents([c.text for c in store.written])
    check(
        "向量与块文本一一对应（未错位）",
        [c.embedding for c in store.written] == reencoded,
        "第 i 条向量与第 i 条文本对不上：回填时可能错位",
    )


def case_pipeline_failure(embedder: BgeEmbedder, workdir: Path) -> None:
    """6. 步骤层失败语义：写入失败 / 切分为空 / 上游漏出空白块。"""
    print("\n[6] 步骤层失败语义")

    def build(store: FakeVectorStore, repository: FakeRepository, chunks: list[Chunk], tag: str):
        pipeline = IngestionPipeline(
            parser=FakeParser([Page(page_no=1, text="假文档正文")]),
            splitter=FakeSplitter(chunks),
            embedder=embedder,
            vector_store=store,
            repository=repository,
            upload_dir=str(workdir / f"uploads-{tag}"),
        )
        return asyncio.run(pipeline.run("kb-fake", "sample.txt", b"fake bytes"))

    # 6a 向量库写入失败 → failed，且尝试清理已写入的残留
    store = FakeVectorStore(fail_on_upsert=True)
    repository = FakeRepository()
    document = build(store, repository, make_chunks(4), "upsert-fail")
    check("向量库写入失败时文档标记 failed", document.status == DOC_STATUS_FAILED, document.status)
    check("error 里带上原始原因", "模拟向量库写入失败" in document.error, document.error)
    check("仓储也记录了 failed（不留下 processing 僵尸记录）", repository.failed is not None and repository.failed[0] == document.id)
    check("失败时按 doc_id 清理残留向量", store.deleted == [document.id], f"{store.deleted}")

    # 6b 切分为空 → failed，提示可读（而不是抛到接口层变成 500）
    repository = FakeRepository()
    document = build(FakeVectorStore(), repository, [], "empty-split")
    check("切分结果为空时文档标记 failed", document.status == DOC_STATUS_FAILED, document.status)
    check("提示指向「切分结果为空」", "切分结果为空" in document.error, document.error)

    # 6c 上游漏出空白块 → 向量化拒绝，整篇 failed（而不是写入无意义向量）
    blank_chunks = make_chunks(3)
    blank_chunks[1].text = "   "
    repository = FakeRepository()
    document = build(FakeVectorStore(), repository, blank_chunks, "blank-chunk")
    check("上游漏出空白块时文档标记 failed", document.status == DOC_STATUS_FAILED, document.status)
    check("错误指向具体是第几条为空", "第 1 条待编码文本为空" in document.error, document.error)

    # 6d 向量化失败不该留下"已写向量但状态 ready"的错配
    check("上述失败都没有走到 ready 标记", repository.ready is None)


class _BrokenModel:
    """按需抛异常 / 返回非法输出的假模型（替换 embedder._model 用）。

    只实现 encode_corpus / encode_queries 两个入口——适配器对模型的所有
    调用都必须经过这两者，所以用它就能覆盖全部失败面而不加载真权重。
    """

    def __init__(self, *, exc: Exception | None = None, output=None) -> None:
        self._exc = exc
        self._output = output
        self.calls = 0

    def encode_corpus(self, texts, **_kwargs):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return self._output

    def encode_queries(self, texts, **_kwargs):
        return self.encode_corpus(texts)


def case_failure_semantics(embedder: BgeEmbedder) -> None:
    """向量化失败必须收敛成 EmbeddingError(502)，而不是 500 或裸 torch 异常。

    为什么值得单测：这类故障原本会一路冒到兜底处理器变成
    「服务内部错误」+ 一段没人读的 torch 栈，而处置动作明明很具体
    （调小批大小 / 换 CPU / 换权重）。
    """
    print("\n[8] 失败语义：EmbeddingError(502)")

    dim = embedder.dimension

    def run_docs(model, texts: list[str]):
        embedder._model = model  # noqa: SLF001 - 自检专用注入点
        try:
            return embedder.embed_documents(texts)
        finally:
            embedder._model = None  # noqa: SLF001

    def run_query(model, text: str):
        embedder._model = model  # noqa: SLF001
        try:
            return embedder.embed_query(text)
        finally:
            embedder._model = None  # noqa: SLF001

    expect_error(
        "显存不足（前向抛 CUDA OOM）",
        EmbeddingError,
        "向量化失败",
        lambda: run_docs(_BrokenModel(exc=RuntimeError("CUDA out of memory")), ["甲"]),
    )
    oom = None
    try:
        run_docs(_BrokenModel(exc=RuntimeError("CUDA out of memory")), ["甲"])
    except EmbeddingError as exc:
        oom = exc
    check(
        "OOM 的提示里给出可执行的处置动作（调小批大小 / 换 CPU）",
        oom is not None and "EMBEDDING_BATCH_SIZE" in oom.message,
        getattr(oom, "message", ""),
    )
    check(
        "错误里带上 device 与 batch（否则无从判断该改哪个配置）",
        oom is not None
        and oom.detail.get("device") == embedder.device
        and oom.detail.get("batch_size") == embedder.batch_size,
        str(getattr(oom, "detail", {})),
    )

    expect_error(
        "输出条数少于输入（会让块与向量静默错位）",
        EmbeddingError,
        "条数与输入不一致",
        lambda: run_docs(
            _BrokenModel(output=[[0.0] * dim]), ["甲", "乙"]
        ),
    )
    expect_error(
        "输出维度与权重不符",
        EmbeddingError,
        "维度",
        lambda: run_docs(_BrokenModel(output=[[0.0] * (dim + 1)]), ["甲"]),
    )
    nan_vector = [0.0] * dim
    nan_vector[0] = float("nan")
    expect_error(
        "输出含 NaN（会污染索引，对任意查询都给可疑相似度）",
        EmbeddingError,
        "NaN/Inf",
        lambda: run_docs(_BrokenModel(output=[nan_vector]), ["甲"]),
    )
    expect_error(
        "查询编码失败同样被包成 EmbeddingError",
        EmbeddingError,
        "查询编码",
        lambda: run_query(_BrokenModel(exc=RuntimeError("shape mismatch")), "问题"),
    )
    expect_error(
        "ConfigurationError 不被包成 EmbeddingError（该改 .env 而不是查显存）",
        ConfigurationError,
        "权重",
        lambda: run_docs(_BrokenModel(exc=ConfigurationError("权重目录不存在")), ["甲"]),
    )

    check(
        "EmbeddingError 是 502 且 code=embedding_error（与 Milvus 故障可区分）",
        EmbeddingError.http_status == 502 and EmbeddingError.code == "embedding_error",
        f"{EmbeddingError.http_status} / {EmbeddingError.code}",
    )
    check(
        "EmbeddingError 继承 RetrievalError（上层按检索链路失败统一处理）",
        issubclass(EmbeddingError, RetrievalError),
    )
    check(
        "编码失败后 embedder 仍可继续使用（没有留下半加载状态）",
        run_docs(_BrokenModel(output=[[0.0] * dim]), ["甲"]) == [[0.0] * dim],
    )


def main() -> int:
    settings = get_settings()
    print("=" * 64)
    print("向量化自检（FlagEmbedding）")
    print(f"  权重      : {settings.embedding_model_path}")
    print(f"  声明维度  : {settings.milvus_dimension}")
    print(f"  设备/批   : {settings.embedding_device} / {settings.embedding_batch_size}")
    print("=" * 64)

    workdir = Path(tempfile.mkdtemp(prefix="check-embedding-"))
    # 用真实 embedder：本脚本要验的就是它
    embedder = BgeEmbedder()
    print(f"  embedder  : dimension={embedder.dimension} batch={embedder.batch_size}")

    try:
        case_capability(embedder)
        case_dimension(embedder, workdir)
        case_contract(embedder)
        case_edges(embedder)
        case_batching(embedder)
        case_pipeline_success(embedder, workdir)
        case_pipeline_failure(embedder, workdir)
        case_failure_semantics(embedder)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print("\n" + "=" * 64)
    if _failures:
        print(f"未通过 {len(_failures)}/{_checks} 项：")
        for item in _failures:
            print(f"  - {item}")
        return 1
    print(f"全部通过（{_checks} 项）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
