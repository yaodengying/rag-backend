# -*- coding: utf-8 -*-
"""向量库（入库）自检：验证「片段 + 向量 → Milvus」这一步本身是对的。

为什么需要这个脚本：
    `check_embedding.py` 把关的是第 5 步「文本 → 向量」，
    但**向量写进 Milvus 这一步此前没有任何独立检查**，而它的失败模式
    两头都很难受：一头是「静默写歪」——写进去了但字段被截断/页码丢失/
    向量与正文错位；另一头是「整篇崩掉」——某一行触发 Milvus 的参数校验，
    报错里只有一个批内行号，定位不到是哪一块。

判据（不只断言「不报错」）：

    | 检查项        | 期望                                                            |
    |--------------|------------------------------------------------------------------|
    | 端口契约      | MilvusVectorStore 满足 VectorStorePort（runtime_checkable）       |
    | 建表          | schema 九个字段类型正确；索引 metric/index_type 与配置一致        |
    | 维度不符      | ensure_collection(不同维度) → ConfigurationError，且**不重建**    |
    | 空输入        | upsert([]) == 0（不发起请求）                                     |
    | 缺向量        | 片段 embedding 为空 → ValidationError                            |
    | 脏 ID         | kb_id/doc_id/id 含引号或空格 → ValidationError（否则那批向量删不掉）|
    | 写入条数      | 返回值 == 实际落库行数；query count(*) 对得上                     |
    | 可见性        | 写入并 flush 后**立刻**可检索命中                                 |
    | 幂等          | 同一 id 二次 upsert → 行数不增（覆盖而非追加）                    |
    | 跨批          | 条数跨过批边界时一条不丢、不错位                                  |
    | 字节截断      | 超长 text/filename 按 UTF-8 **字节**截断且不切出半个字符          |
    | 字段往返      | 中英文/emoji 正文、filename、页码哨兵 None→-1→None 均无损          |
    | 检索排序      | COSINE 下更相似的排在前（不是按 id 或写入顺序）                     |
    | 库隔离        | 只返回目标 kb_id 的片段，别的库不串味                              |
    | 阈值/条数     | min_score 生效；top_k 截断生效                                     |
    | 查询维度      | 查询向量维度与 collection 不符 → ValidationError（不是 Milvus 生硬报错）|
    | 删除          | delete_by_doc / delete_by_kb 返回条数，且删完**立刻**检索不到      |
    | 探活          | ping() == True                                                    |
    | 步骤层        | 真 Milvus + 假上下游跑 IngestionPipeline 第 6 步：向量真落库、      |
    |              | 字段正确；写入失败（坏维度）→ failed + **无残留行**（含部分成功时） |

安全约定（重要）：
    全程使用**独立的临时 collection**（`rag_backend_check_<pid>`），
    结束后删除。**绝不触碰** `MILVUS_COLLECTION` 指向的真实集合；
    脚本结尾会核对真实集合的行数前后一致，不一致直接判失败。

运行（必须用项目自己的 .venv，且 Milvus 容器已启动）：
    .venv\\Scripts\\python.exe scripts\\check_vectorstore.py

    需要本机 Milvus（默认 http://127.0.0.1:19530），不需要 GPU 与 BGE 权重
    ——步骤层用的是**假** embedder，返回固定维度的确定性向量。

退出码：0 全部通过；1 存在失败项。
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
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
from app.core.exceptions import ConfigurationError, ValidationError  # noqa: E402
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
from app.infrastructure.vectorstore import (  # noqa: E402
    _MAX_FILENAME,
    _MAX_VARCHAR,
    _PAGE_UNKNOWN,
    MilvusVectorStore,
    _clip_utf8,
)
from app.rag.ingest import IngestionPipeline  # noqa: E402

# 独立临时 collection，避免碰到真实数据
DIM = 8
COLLECTION = f"rag_backend_check_{os.getpid()}"

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


def section(title: str) -> None:
    """打印分组标题。"""
    print()
    print(f"=== {title} ===")


# ---------------------------------------------------------------------------
# 造数据与读数据的小工具
# ---------------------------------------------------------------------------


def _vec(seed: int = 0) -> list[float]:
    """确定性 8 维向量（未归一化，COSINE 度量下 Milvus 自行处理）。"""
    return [float((seed + i) % 5 + 1) for i in range(DIM)]


def _onehot(axis: int) -> list[float]:
    """单位基向量，用于构造「谁更相似」这种可断言的场景。"""
    return [1.0 if i == axis else 0.0 for i in range(DIM)]


def _chunk(
    text: str,
    index: int,
    *,
    kb_id: str,
    doc_id: str,
    filename: str = "sample.txt",
    page: int | None = 1,
    vector: list[float] | None = None,
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
        embedding=list(vector) if vector is not None else _vec(index),
    )


def _count(store: MilvusVectorStore, expression: str) -> int:
    """统计满足条件的行数。"""
    rows = store.client.query(
        collection_name=store.collection_name,
        filter=expression,
        output_fields=["count(*)"],
    )
    if not rows:
        return 0
    return int(next(iter(rows[0].values())))


def _fetch(
    store: MilvusVectorStore, expression: str, fields: list[str], limit: int = 100
) -> list[dict]:
    """按表达式取回若干行的指定字段（按 chunk_index 排序，便于比对）。"""
    rows = store.client.query(
        collection_name=store.collection_name,
        filter=expression,
        output_fields=fields,
        limit=limit,
    )
    if "chunk_index" in fields:
        rows.sort(key=lambda row: row.get("chunk_index", 0))
    return rows


# ---------------------------------------------------------------------------
# 假的上下游：只为了脱离磁盘 / SQLite / 模型跑流水线第 6 步
# 刻意不继承端口（Protocol 的用法就是结构化鸭子类型），
# 但下面会显式断言它们确实满足各自的端口契约。
# ---------------------------------------------------------------------------


class FakeParser:
    """假解析器：不读磁盘，按文件名返回预置页。"""

    supported_extensions = (".txt",)

    def __init__(self, pages: list[Page]) -> None:
        self._pages = pages

    def parse(self, file_path: str) -> list[Page]:
        return list(self._pages)


class FakeSplitter:
    """假切分器：一页一块，页码溯源照真实约定填。"""

    def split(self, pages: list[Page]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for index, page in enumerate(pages):
            chunks.append(
                Chunk(
                    id="",  # 归属由流水线补齐，这里的空值正是"切分器不越权"的体现
                    kb_id="",
                    doc_id="",
                    text=page.text,
                    chunk_index=index,
                    page_start=page.page_no,
                    page_end=page.page_no,
                )
            )
        return chunks


class FakeEmbedder:
    """假向量化器：返回确定性向量；带标记的文本返回**错误维度**以触发写入失败。"""

    dimension = DIM

    def __init__(self, poison_marker: str = "") -> None:
        self._poison = poison_marker

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            if self._poison and self._poison in text:
                out.append([0.1] * (DIM - 1))  # 维度不符 → Milvus 侧拒绝
            else:
                out.append(_vec(len(text)))
        return out

    def embed_query(self, text: str) -> list[float]:
        return _vec(len(text))


class FakeRepository:
    """假仓储：只记状态，不写 SQLite。

    刻意把 ChunkRepositoryPort 的**全部**方法都实现掉（哪怕流水线的第 6 步
    只用到 3 个），这样下面的 isinstance 断言才有意义：端口增删方法时
    它会立刻变红，提示"假实现已与端口脱节"。
    """

    def __init__(self) -> None:
        self.ready: list[tuple[str, int]] = []
        self.failed: list[tuple[str, str]] = []
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
        return Document(id=str(uuid.uuid4()), kb_id=kb_id, filename=filename)

    def find_document_by_hash(self, kb_id: str, content_hash: str) -> Document | None:
        # 本脚本不测去重；端口要求实现，返回 None 表示"没查到重复"
        return None

    def mark_document_ready(self, doc_id: str, chunk_count: int) -> None:
        self.ready.append((doc_id, chunk_count))

    def mark_document_failed(self, doc_id: str, error: str) -> None:
        self.failed.append((doc_id, error))

    def mark_document_processing(self, doc_id: str) -> None:
        self.processing.append(doc_id)

    def delete_document(self, doc_id: str) -> str | None:
        return None


# ---------------------------------------------------------------------------
# 各组检查
# ---------------------------------------------------------------------------


def check_contract(store: MilvusVectorStore, settings) -> None:
    section("1. 端口契约与建表")
    check("MilvusVectorStore 满足 VectorStorePort", isinstance(store, VectorStorePort))
    check(
        f"批大小取自配置（当前 {store.upsert_batch_size}）",
        store.upsert_batch_size == min(int(settings.milvus_upsert_batch_size), 2000),
        f"{store.upsert_batch_size} vs {settings.milvus_upsert_batch_size}",
    )

    store.ensure_collection(DIM)
    store.ensure_collection(DIM)  # 幂等：重复调用不应报错，也不应重建
    check("重复 ensure_collection 幂等（不报错、不重建）", store.client.has_collection(COLLECTION))

    description = store.client.describe_collection(COLLECTION)
    fields = {field["name"]: field for field in description.get("fields", [])}
    expected_fields = {
        "id",
        "kb_id",
        "doc_id",
        "text",
        "filename",
        "chunk_index",
        "page_start",
        "page_end",
        "embedding",
        # 稀疏字段：混合检索的 BM25 路用它，由 schema 里的 BM25 函数自动生成
        # （见 scripts/check_retrieval.py 对全文检索能力的专门检查）
        "sparse",
    }
    check("schema 字段齐全", set(fields) == expected_fields, f"实际 {sorted(fields)}")

    from pymilvus import DataType

    type_expectations = {
        "id": DataType.VARCHAR,
        "kb_id": DataType.VARCHAR,
        "doc_id": DataType.VARCHAR,
        "text": DataType.VARCHAR,
        "filename": DataType.VARCHAR,
        "chunk_index": DataType.INT64,
        "page_start": DataType.INT64,
        "page_end": DataType.INT64,
        "embedding": DataType.FLOAT_VECTOR,
        "sparse": DataType.SPARSE_FLOAT_VECTOR,
    }
    wrong_types = [
        name
        for name, wanted in type_expectations.items()
        if name in fields and fields[name].get("type") != wanted
    ]
    check("schema 字段类型正确", not wrong_types, f"类型不符：{wrong_types}")

    check(
        f"id 是主键且 max_length=64",
        bool(fields.get("id", {}).get("is_primary"))
        and int(fields.get("id", {}).get("params", {}).get("max_length", 0)) == 64,
    )
    check(
        f"text 的 max_length={_MAX_VARCHAR}",
        int(fields.get("text", {}).get("params", {}).get("max_length", 0)) == _MAX_VARCHAR,
    )
    check(
        f"filename 的 max_length={_MAX_FILENAME}",
        int(fields.get("filename", {}).get("params", {}).get("max_length", 0)) == _MAX_FILENAME,
    )
    check(
        f"embedding 维度 = {DIM}",
        int(fields.get("embedding", {}).get("params", {}).get("dim", 0)) == DIM,
    )

    indexes = store.client.list_indexes(COLLECTION)
    index_info = [store.client.describe_index(COLLECTION, name) for name in indexes]
    # 现在有两个索引：embedding（稠密，COSINE）+ sparse（全文，BM25）。
    # 不能再断言"所有索引的 metric 相同"，要按字段分别核对。
    by_field = {info.get("field_name"): info for info in index_info}
    check(
        f"embedding 索引 metric_type = {settings.milvus_metric_type}",
        (by_field.get("embedding") or {}).get("metric_type") == settings.milvus_metric_type,
        f"实际 {by_field.get('embedding')}",
    )
    check(
        "sparse 索引 metric_type = BM25（全文检索能力）",
        str((by_field.get("sparse") or {}).get("metric_type", "")).upper() == "BM25",
        f"实际 {by_field.get('sparse')}",
    )

    # 维度不符必须报错且**不重建**
    expect_error(
        "ensure_collection(维度不符) 抛 ConfigurationError",
        ConfigurationError,
        "维度",
        lambda: store.ensure_collection(DIM + 1),
    )
    dim_after = store.client.describe_collection(COLLECTION)
    kept = [
        int(f.get("params", {}).get("dim", 0))
        for f in dim_after.get("fields", [])
        if f["name"] == "embedding"
    ]
    check("维度不符时原 collection 未被重建", kept == [DIM], f"实际 dim={kept}")


def check_write_and_visibility(store: MilvusVectorStore) -> None:
    section("2. 写入、可见性与幂等")
    kb = f"kb-{uuid.uuid4().hex[:8]}"
    doc = str(uuid.uuid4())

    check("upsert([]) == 0（空输入不发起请求）", store.upsert([]) == 0)

    no_vector = _chunk("缺向量的片段", 0, kb_id=kb, doc_id=doc)
    no_vector.embedding = None
    expect_error(
        "缺少向量 → ValidationError",
        ValidationError,
        "缺少向量",
        lambda: store.upsert([no_vector]),
    )

    bad_id = _chunk("脏 kb_id", 0, kb_id='kb" or kb_id != "', doc_id=doc)
    expect_error(
        "kb_id 含引号 → ValidationError（否则该批向量将无法删除）",
        ValidationError,
        "kb_id",
        lambda: store.upsert([bad_id]),
    )

    bad_doc = _chunk("脏 doc_id", 0, kb_id=kb, doc_id="doc id with space")
    expect_error(
        "doc_id 含空格 → ValidationError",
        ValidationError,
        "doc_id",
        lambda: store.upsert([bad_doc]),
    )

    written = store.upsert(
        [_chunk(f"第 {i} 段正文", i, kb_id=kb, doc_id=doc) for i in range(3)]
    )
    check("upsert 返回写入条数 = 3", written == 3, f"实际 {written}")
    stored_rows = _count(store, f'doc_id == "{doc}"')
    check("落库行数与写入条数一致", stored_rows == 3, f"实际 {stored_rows}")

    # 幂等：同一个 id 再写一次，行数不应增加
    same_id = str(uuid.uuid4())
    store.upsert([_chunk("旧正文", 0, kb_id=kb, doc_id=doc, chunk_id=same_id)])
    store.upsert([_chunk("新正文", 0, kb_id=kb, doc_id=doc, chunk_id=same_id)])
    rows = _fetch(store, f'id == "{same_id}"', ["id", "text"])
    check(
        "同一 id 二次 upsert 是覆盖而非追加",
        len(rows) == 1 and rows[0]["text"] == "新正文",
        f"行数 {len(rows)}，正文 {rows[0]['text'] if rows else None!r}",
    )

    # 可见性：写完立刻可检索（依赖 upsert 末尾的 flush）
    hits = store.search(kb, _vec(0), top_k=10, min_score=-2.0)
    check(
        "写入后立刻可检索到（flush 生效）",
        len(hits) >= 3,
        f"命中 {len(hits)} 条",
    )


def check_cross_batch(store: MilvusVectorStore) -> None:
    section("3. 跨批写入（批大小临时调小以快速覆盖多批）")
    kb = f"kb-{uuid.uuid4().hex[:8]}"
    doc = str(uuid.uuid4())
    original = store._upsert_batch_size  # noqa: SLF001 - 自检脚本刻意改动内部批大小
    store._upsert_batch_size = 7  # noqa: SLF001
    try:
        total = 23  # 23 / 7 → 4 批（7,7,7,2），必然覆盖"最后一批不满"
        written = store.upsert(
            [_chunk(f"跨批正文 {i}", i, kb_id=kb, doc_id=doc) for i in range(total)]
        )
        check(f"跨批写入返回 {total} 条", written == total, f"实际 {written}")
        rows = _fetch(store, f'doc_id == "{doc}"', ["id", "chunk_index", "text"], limit=100)
        check("跨批后一条不丢", len(rows) == total, f"实际 {len(rows)}")
        check(
            "跨批后 chunk_index 连续且无空洞",
            [row["chunk_index"] for row in rows] == list(range(total)),
            f"实际 {[row['chunk_index'] for row in rows]}",
        )
        check(
            "跨批后正文与序号一一对应（不错位）",
            all(row["text"] == f"跨批正文 {row['chunk_index']}" for row in rows),
        )
    finally:
        store._upsert_batch_size = original  # noqa: SLF001
    check("批大小已恢复", store.upsert_batch_size == original)


def check_field_guards(store: MilvusVectorStore) -> None:
    section("4. 字段护栏：按 UTF-8 字节截断")
    kb = f"kb-{uuid.uuid4().hex[:8]}"

    # 纯函数层：不切出半个字符
    long_cn = "汉" * 30000
    clipped = _clip_utf8(long_cn, _MAX_VARCHAR)
    check(
        f"_clip_utf8 截断到 {_MAX_VARCHAR} 字节内",
        len(clipped.encode("utf-8")) <= _MAX_VARCHAR,
        f"实际 {len(clipped.encode('utf-8'))} 字节",
    )
    check("_clip_utf8 不产生替换字符（没切断多字节序列）", "\ufffd" not in clipped)
    check("_clip_utf8 结果是原文前缀", long_cn.startswith(clipped) and clipped.strip("汉") == "")
    check(
        "_clip_utf8 对未超限的文本原样返回",
        _clip_utf8("短文本", _MAX_VARCHAR) == "短文本",
    )

    # 端到端层：超长字段真的能写进去，而不是让整篇失败
    doc = str(uuid.uuid4())
    long_text = "测" * 30000  # 90000 字节 > 65535
    long_name = "文" * 600  # 1800 字节 > 512
    try:
        store.upsert(
            [
                _chunk(long_text, 0, kb_id=kb, doc_id=doc, filename=long_name),
                _chunk("正常片段", 1, kb_id=kb, doc_id=doc),
            ]
        )
        ok = True
        error = ""
    except Exception as exc:  # noqa: BLE001
        ok = False
        error = f"{type(exc).__name__}: {exc}"
    check("超长 text/filename 不再让整篇入库失败", ok, error)

    if ok:
        rows = _fetch(store, f'doc_id == "{doc}"', ["chunk_index", "text", "filename"])
        first = next((row for row in rows if row["chunk_index"] == 0), None)
        check("超长 text 已落库", first is not None)
        if first:
            stored_bytes = len(first["text"].encode("utf-8"))
            check(
                f"落库 text 字节数 ≤ {_MAX_VARCHAR}",
                stored_bytes <= _MAX_VARCHAR,
                f"实际 {stored_bytes}",
            )
            check("落库 text 是原文前缀", long_text.startswith(first["text"]))
            name_bytes = len(first["filename"].encode("utf-8"))
            check(
                f"落库 filename 字节数 ≤ {_MAX_FILENAME}",
                name_bytes <= _MAX_FILENAME,
                f"实际 {name_bytes}",
            )
            check("落库 filename 是原名前缀", long_name.startswith(first["filename"]))

    # 边界：正好等于上限时不应被改动
    doc2 = str(uuid.uuid4())
    exact = "边" * (65535 // 3)  # 每字 3 字节 → 正好 65535 字节
    store.upsert([_chunk(exact, 0, kb_id=kb, doc_id=doc2)])
    rows2 = _fetch(store, f'doc_id == "{doc2}"', ["text"])
    check(
        "正好等于字节上限的正文未被截断",
        rows2 and rows2[0]["text"] == exact,
        f"落库 {len(rows2[0]['text'].encode('utf-8')) if rows2 else 0} 字节",
    )


def check_roundtrip_and_search(store: MilvusVectorStore) -> None:
    section("5. 字段往返与检索语义")
    kb_a = f"kb-{uuid.uuid4().hex[:8]}"
    kb_b = f"kb-{uuid.uuid4().hex[:8]}"
    doc_a = str(uuid.uuid4())
    doc_b = str(uuid.uuid4())

    unicode_text = "向量检索：Milvus 支持 COSINE。繁体「檢索」与 emoji 🚀 也要能往返。"
    chunk_id = str(uuid.uuid4())
    store.upsert(
        [
            _chunk(unicode_text, 0, kb_id=kb_a, doc_id=doc_a, filename="中文 文件名.txt",
                   page=3, vector=_onehot(0), chunk_id=chunk_id),
            _chunk("另一个方向的片段", 1, kb_id=kb_a, doc_id=doc_a, vector=_onehot(1)),
            _chunk("页码未知的片段", 2, kb_id=kb_a, doc_id=doc_a, vector=_onehot(0), page=None),
            _chunk("别的知识库的片段", 0, kb_id=kb_b, doc_id=doc_b, vector=_onehot(0)),
        ]
    )

    hits = store.search(kb_a, _onehot(0), top_k=10, min_score=-2.0)
    ids = [hit.chunk.id for hit in hits]
    check("检索结果只含目标知识库", all(hit.chunk.kb_id == kb_a for hit in hits), f"{[h.chunk.kb_id for h in hits]}")
    check("别的知识库的同向片段没有串味", chunk_id in ids and len(hits) == 3, f"命中 {len(hits)} 条")
    check(
        "COSINE：更相似的排在前",
        len(hits) >= 2 and hits[0].vector_score >= hits[1].vector_score,
        f"分数 {[round(h.vector_score, 4) for h in hits]}",
    )
    check(
        "同向片段得分高于正交片段",
        hits[0].vector_score > 0.99 and min(h.vector_score for h in hits[1:]) < 0.5,
        f"分数 {[round(h.vector_score, 4) for h in hits]}",
    )

    target = next((hit for hit in hits if hit.chunk.id == chunk_id), None)
    check("unicode 正文往返一致", target is not None and target.chunk.text == unicode_text)
    check("中文 filename 往返一致", target is not None and target.chunk.filename == "中文 文件名.txt")
    check("页码往返一致（第 3 页）", target is not None and target.chunk.page_start == 3)

    unknown = next((hit for hit in hits if hit.chunk.text == "页码未知的片段"), None)
    check(
        f"页码哨兵往返：None → {_PAGE_UNKNOWN} → None",
        unknown is not None and unknown.chunk.page_start is None and unknown.chunk.page_end is None,
        f"page_start={unknown.chunk.page_start if unknown else 'N/A'}",
    )
    raw = _fetch(store, f'id == "{unknown.chunk.id}"', ["page_start", "page_end"]) if unknown else []
    check(
        f"落库的是 {_PAGE_UNKNOWN} 而非 NULL",
        raw and raw[0]["page_start"] == _PAGE_UNKNOWN,
        f"实际 {raw}",
    )

    # top_k 与 min_score
    limited = store.search(kb_a, _onehot(0), top_k=2, min_score=-2.0)
    check("top_k 截断生效", len(limited) == 2, f"命中 {len(limited)} 条")
    filtered = store.search(kb_a, _onehot(0), top_k=10, min_score=0.5)
    check(
        "min_score 阈值生效（正交片段被滤掉）",
        all(hit.vector_score >= 0.5 for hit in filtered) and 0 < len(filtered) < len(hits),
        f"命中 {len(filtered)} 条，分数 {[round(h.vector_score, 4) for h in filtered]}",
    )
    expect_error(
        "查询向量维度不符 → ValidationError",
        ValidationError,
        "维度",
        lambda: store.search(kb_a, [0.1] * (DIM - 1), top_k=3),
    )
    expect_error(
        "检索时 kb_id 非法 → ValidationError",
        ValidationError,
        "kb_id",
        lambda: store.search('kb" or 1 == 1', _onehot(0), top_k=3),
    )


def check_delete(store: MilvusVectorStore) -> None:
    section("6. 删除与立即失效")
    kb = f"kb-{uuid.uuid4().hex[:8]}"
    doc1 = str(uuid.uuid4())
    doc2 = str(uuid.uuid4())
    store.upsert([_chunk(f"文档一的第 {i} 段", i, kb_id=kb, doc_id=doc1) for i in range(3)])
    store.upsert([_chunk(f"文档二的第 {i} 段", i, kb_id=kb, doc_id=doc2) for i in range(2)])
    check("删除前库内共 5 条", _count(store, f'kb_id == "{kb}"') == 5)

    removed = store.delete_by_doc(doc1)
    check("delete_by_doc 返回删除条数 3", removed == 3, f"实际 {removed}")
    check("被删文档的行数归零", _count(store, f'doc_id == "{doc1}"') == 0)
    hits = store.search(kb, _vec(0), top_k=10, min_score=-2.0)
    check(
        "删除后立刻检索不到（不需要重启或等待 compaction）",
        all(hit.chunk.doc_id != doc1 for hit in hits) and len(hits) == 2,
        f"命中 {len(hits)} 条，doc 集合 {sorted({h.chunk.doc_id for h in hits})}",
    )

    removed_kb = store.delete_by_kb(kb)
    check("delete_by_kb 返回剩余条数 2", removed_kb == 2, f"实际 {removed_kb}")
    check("整库删除后计数归零", _count(store, f'kb_id == "{kb}"') == 0)
    check("整库删除后检索为空", store.search(kb, _vec(0), top_k=10, min_score=-2.0) == [])
    check("删除不存在的 doc_id 返回 0", store.delete_by_doc(str(uuid.uuid4())) == 0)


def check_ping(store: MilvusVectorStore) -> None:
    section("7. 探活")
    check("ping() == True", store.ping() is True)


def check_pipeline_step(store: MilvusVectorStore) -> None:
    section("8. 步骤层：真 Milvus + 假上下游跑 IngestionPipeline 第 6 步")
    check("假 parser 满足 DocumentParserPort", isinstance(FakeParser([]), DocumentParserPort))
    check("假 splitter 满足 SplitterPort", isinstance(FakeSplitter(), SplitterPort))
    check("假 embedder 满足 EmbedderPort", isinstance(FakeEmbedder(), EmbedderPort))
    check("假仓储满足 ChunkRepositoryPort", isinstance(FakeRepository(), ChunkRepositoryPort))

    pages = [Page(page_no=1, text="第一页的正文内容。"), Page(page_no=2, text="第二页的正文内容。")]
    kb = f"kb-{uuid.uuid4().hex[:8]}"
    upload_dir = tempfile.mkdtemp(prefix="rag_check_upload_")
    original = store._upsert_batch_size  # noqa: SLF001

    # --- 成功路径 ---
    repo = FakeRepository()
    pipeline = IngestionPipeline(
        parser=FakeParser(pages),
        splitter=FakeSplitter(),
        embedder=FakeEmbedder(),
        vector_store=store,
        repository=repo,
        upload_dir=upload_dir,
    )
    doc = asyncio.run(pipeline.run(kb, "fake.txt", b"content-not-used"))
    check("成功路径 status=ready", doc.status == DOC_STATUS_READY, f"实际 {doc.status} / {doc.error}")
    check("chunk_count = 2", doc.chunk_count == 2, f"实际 {doc.chunk_count}")
    check("仓储收到了 ready 标记", repo.ready == [(doc.id, 2)])
    check("失败标记未被触发", repo.failed == [])

    rows = _fetch(store, f'doc_id == "{doc.id}"', ["chunk_index", "text", "page_start", "page_end", "filename", "kb_id"], limit=50)
    check("向量真的落进了 Milvus", len(rows) == 2, f"实际 {len(rows)} 行")
    check("页码随片段落库（第 1、2 页）", [row["page_start"] for row in rows] == [1, 2])
    check("page_start == page_end（逐页独立切分）", all(row["page_start"] == row["page_end"] for row in rows))
    check("正文与页码对应（不错位）", [row["text"] for row in rows] == [page.text for page in pages])
    check("filename / kb_id 落库正确", all(row["filename"] == "fake.txt" and row["kb_id"] == kb for row in rows))
    check("检索能命中刚入库的文档", len(store.search(kb, _vec(100), top_k=10, min_score=-2.0)) == 2)

    # --- 失败路径 A：第一批就失败 → 无残留 ---
    store._upsert_batch_size = 1  # noqa: SLF001
    repo2 = FakeRepository()
    pipeline2 = IngestionPipeline(
        parser=FakeParser(pages),
        splitter=FakeSplitter(),
        embedder=FakeEmbedder(poison_marker="第二页"),  # 第二块维度错，但批大小为 1 → 第一批成功
        vector_store=store,
        repository=repo2,
        upload_dir=upload_dir,
    )
    doc2 = asyncio.run(pipeline2.run(kb, "poison.txt", b"x"))
    check("写入失败 → status=failed", doc2.status == DOC_STATUS_FAILED, f"实际 {doc2.status}")
    # 钉的是 `[code]` 前缀而不是异常类名：code 是给前端/CLI 分支用的稳定标识，
    # 类名会随重构改（这条断言就是被 ingest.py 的格式变更打红后改过来的）。
    check(
        "失败原因带可分支的 [retrieval_error] 前缀",
        "[retrieval_error]" in doc2.error,
        f"实际 {doc2.error[:120]}",
    )
    check(
        "失败原因里仍保留可读描述（不是只剩一个 code）",
        "Milvus" in doc2.error,
        f"实际 {doc2.error[:120]}",
    )
    check("失败时有 failed 标记", len(repo2.failed) == 1)
    check(
        "部分成功后失败：已写入的残留行被清理干净",
        _count(store, f'doc_id == "{doc2.id}"') == 0,
        f"残留 {_count(store, f'doc_id == \"{doc2.id}\"')} 行",
    )
    check("清理没有误删别的文档", _count(store, f'doc_id == "{doc.id}"') == 2)

    # --- 失败路径 B：坏维度导致整批失败 ---
    repo3 = FakeRepository()
    pipeline3 = IngestionPipeline(
        parser=FakeParser(pages),
        splitter=FakeSplitter(),
        embedder=FakeEmbedder(poison_marker="第一页"),
        vector_store=store,
        repository=repo3,
        upload_dir=upload_dir,
    )
    doc3 = asyncio.run(pipeline3.run(kb, "poison2.txt", b"x"))
    check("整批失败 → status=failed", doc3.status == DOC_STATUS_FAILED)
    check("整批失败后无残留", _count(store, f'doc_id == "{doc3.id}"') == 0)

    store._upsert_batch_size = original  # noqa: SLF001


# ---------------------------------------------------------------------------


def main() -> int:
    settings = get_settings()
    print("向量库（Milvus 入库）自检")
    print(f"  Milvus URI      : {settings.milvus_uri}")
    print(f"  临时 collection : {COLLECTION}（维度 {DIM}）")
    print(f"  真实 collection : {settings.milvus_collection}（全程不触碰）")

    real_rows_before: int | None = None
    store = MilvusVectorStore(collection=COLLECTION)
    try:
        if store.client.has_collection(settings.milvus_collection):
            real_rows_before = _count_real(store, settings.milvus_collection)
            print(f"  真实集合现有行数 : {real_rows_before}（结束时复核）")

        check_contract(store, settings)
        check_write_and_visibility(store)
        check_cross_batch(store)
        check_field_guards(store)
        check_roundtrip_and_search(store)
        check_delete(store)
        check_ping(store)
        check_pipeline_step(store)

        section("9. 收尾与残留检查")
        dropped = store.drop_collection()
        check("临时 collection 已删除", dropped and not store.client.has_collection(COLLECTION))
        check("重复 drop 返回 False", store.drop_collection() is False)

        if real_rows_before is not None:
            after = _count_real(store, settings.milvus_collection)
            check(
                f"真实集合 {settings.milvus_collection} 未被触碰（行数仍为 {real_rows_before}）",
                after == real_rows_before,
                f"实际 {after}",
            )
    finally:
        try:
            if store.client.has_collection(COLLECTION):
                store.client.drop_collection(COLLECTION)
                print(f"\n[清理] 已删除临时 collection {COLLECTION}")
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


def _count_real(store: MilvusVectorStore, collection: str) -> int:
    """统计真实 collection 的行数（只读，用于证明本次自检没碰它）。"""
    store.client.load_collection(collection)
    rows = store.client.query(
        collection_name=collection, filter='id != ""', output_fields=["count(*)"]
    )
    if not rows:
        return 0
    return int(next(iter(rows[0].values())))


if __name__ == "__main__":
    sys.exit(main())
