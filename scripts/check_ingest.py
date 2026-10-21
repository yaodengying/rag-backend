# -*- coding: utf-8 -*-
"""入库链路自检：首次入库与「失败后原地重试」的状态机。

为什么需要这个脚本：
    `check_embedding.py` 测的是第 5 步本身，`check_vectorstore.py` 测第 6 步，
    但**「失败之后能不能救回来」**这条路径一直没人测——它此前根本不存在：
    失败时原始文件是保留的（`ingest.py` 的注释也这么写、"便于重试"），
    可是没有任何入口去用它。用户只能重传，而重传会生成新的 doc_id，
    留下「一条 failed + 一条 ready」，并把同一份正文的向量写两遍
    （实测库里同一份 txt 曾攒到 4 条 ready，检索时互相挤占名额）。

判据：

    | 检查项            | 期望                                                        |
    |------------------|-------------------------------------------------------------|
    | 首次入库         | status=ready、chunk_count=写入数、error 清空                 |
    | 原始文件          | 落在 uploads/{kb_id}/{doc_id}_{filename}，内容与入参逐字节一致 |
    | 片段归属          | 每块的 kb_id / doc_id / filename 由流水线回填，id 互不相同     |
    | 首次失败          | status=failed、error 带 `[corrupted_document]`、残留向量被清理 |
    | 重试成功          | 同一个 doc_id 从 failed → ready，**不新建元数据行**            |
    | 重试复用文件      | parser 收到的路径就是上次落盘的那一份，内容一致                |
    | 重试前清理        | 再调一次 delete_by_doc（上次清理自己失败时的兜底，幂等）      |
    | 原始文件丢失      | 抛 NotFoundError(404)，消息提示「重新上传」                    |
    | 状态前置          | processing / ready → ConflictError(409)，且不触发解析          |
    | 文档不存在        | NotFoundError(404)                                          |
    | 重试再失败        | 仍是 failed，error 换成新原因，doc_id 不变                    |
    | 异常语义          | AppError 原样带 code；非 AppError 保留类名                    |
    | 路径安全          | 文件名只取最后一段；kb_id 带 .. 不能逃出 upload_dir            |

运行（必须用项目自己的 .venv）：
    .venv\\Scripts\\python.exe scripts\\check_ingest.py

    全程使用**假的** parser / splitter / embedder / 向量库 / 仓储，
    不连 Milvus、不读 BGE 权重、不碰真实的 data/uploads，秒级完成。

退出码：0 全部通过；1 存在失败项。
"""

from __future__ import annotations

import asyncio
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

from app.core.exceptions import (  # noqa: E402
    ConflictError,
    CorruptedDocumentError,
    NotFoundError,
)
from app.domain.models import (  # noqa: E402
    DOC_STATUS_FAILED,
    DOC_STATUS_PROCESSING,
    DOC_STATUS_READY,
    Chunk,
    Document,
    Page,
)
from app.domain.ports import ChunkRepositoryPort  # noqa: E402
from app.rag.concurrency import ConcurrencyGate  # noqa: E402
from app.rag.ingest import IngestionPipeline  # noqa: E402
from app.services.ingest_service import IngestService  # noqa: E402

# ---------------------------------------------------------------------------
# 极简断言器
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


async def expect_error_async(
    label: str, exc_type: type[Exception], keyword: str, awaitable
) -> None:
    """断言 awaitable 抛出指定类型异常，且消息里含关键词。

    必须用 async 版本：`asyncio.run()` 不能在已运行的事件循环里调用，
    否则捕获到的是 RuntimeError 而不是被测的异常，断言会「假通过/假失败」。
    """
    global _checks
    _checks += 1
    try:
        await awaitable
    except exc_type as exc:
        if keyword in str(exc):
            print(f"  [ok]   {label}")
        else:
            print(f"  [FAIL] {label}  -> 消息里没有「{keyword}」：{exc}")
            _failures.append(label)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {label}  -> 抛的是 {type(exc).__name__}：{exc}")
        _failures.append(label)
    else:
        print(f"  [FAIL] {label}  -> 没有抛异常")
        _failures.append(label)


# ---------------------------------------------------------------------------
# 假实现
# ---------------------------------------------------------------------------


class FakeParser:
    """假解析器：把文件内容按换行切成若干页。可配置成抛错。"""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.raise_with: Exception | None = None

    def parse(self, path: str) -> list[Page]:
        self.calls.append(path)
        if self.raise_with is not None:
            raise self.raise_with
        text = Path(path).read_text(encoding="utf-8")
        return [Page(page_no=i + 1, text=line) for i, line in enumerate(text.splitlines())]


class FakeSplitter:
    """假切分器：一页一块。可配置成返回空列表（触发「切分结果为空」）。"""

    def __init__(self) -> None:
        self.empty = False

    def split(self, pages: list[Page]) -> list[Chunk]:
        if self.empty:
            return []
        return [
            Chunk(
                id="",
                kb_id="",
                doc_id="",
                text=page.text,
                chunk_index=index,
                page_start=page.page_no,
                page_end=page.page_no,
            )
            for index, page in enumerate(pages)
        ]


class FakeEmbedder:
    """假向量化器：按条数返回定长向量，不做任何校验。"""

    dimension = 4

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class FakeVectorStore:
    """假向量库：记录 upsert 与 delete_by_doc 的调用。"""

    def __init__(self) -> None:
        self.upserts: list[list[Chunk]] = []
        self.deleted: list[str] = []

    def upsert(self, chunks: list[Chunk]) -> int:
        self.upserts.append(list(chunks))
        return len(chunks)

    def delete_by_doc(self, doc_id: str) -> int:
        self.deleted.append(doc_id)
        return 0


class MemoryRepository:
    """内存仓储：实现 ChunkRepositoryPort 的**全部**方法。

    全实现（而非只实现用到的那几个）是有意的：端口增删方法时
    `isinstance` 断言会立刻变红，提示「假实现已与端口脱节」。
    """

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.created: list[str] = []
        self.processing: list[str] = []
        self.ready: list[tuple[str, int]] = []
        self.failed: list[tuple[str, str]] = []
        # (kb_id, content_hash) -> doc_id：去重链路要能按正文 hash 回查
        self.hashes: dict[tuple[str, str], str] = {}

    # ----- 知识库（本脚本不关心归属，只按端口签名「有」这几个参数） -----
    def list_knowledge_bases(self, owner_id: str) -> list[dict]:
        return []

    def get_knowledge_base(self, kb_id: str, owner_id: str) -> dict | None:
        return {"id": kb_id, "name": "fake"}

    def create_knowledge_base(self, name: str, description: str, owner_id: str) -> dict:
        return {"id": str(uuid.uuid4()), "name": name, "description": description}

    def update_knowledge_base(
        self, kb_id: str, name: str, description: str, owner_id: str
    ) -> dict:
        return {"id": kb_id, "name": name, "description": description}

    def delete_knowledge_base(self, kb_id: str, owner_id: str) -> bool:
        return True

    # ----- 文档 -----
    def list_documents(self, kb_id: str) -> list[Document]:
        return [doc for doc in self.documents.values() if doc.kb_id == kb_id]

    def get_document(self, doc_id: str) -> Document | None:
        return self.documents.get(doc_id)

    def create_document(self, kb_id: str, filename: str, content_hash: str = "") -> Document:
        document = Document(id=str(uuid.uuid4()), kb_id=kb_id, filename=filename)
        self.documents[document.id] = document
        self.hashes[(kb_id, content_hash)] = document.id
        self.created.append(document.id)
        return document

    def find_document_by_hash(self, kb_id: str, content_hash: str) -> Document | None:
        if not content_hash:
            return None
        doc_id = self.hashes.get((kb_id, content_hash))
        document = self.documents.get(doc_id) if doc_id else None
        if document is not None and document.status == DOC_STATUS_READY:
            return document
        return None

    def mark_document_ready(self, doc_id: str, chunk_count: int) -> None:
        self.ready.append((doc_id, chunk_count))
        document = self.documents.get(doc_id)
        if document is not None:
            document.status = DOC_STATUS_READY
            document.chunk_count = chunk_count
            document.error = ""

    def mark_document_failed(self, doc_id: str, error: str) -> None:
        self.failed.append((doc_id, error))
        document = self.documents.get(doc_id)
        if document is not None:
            document.status = DOC_STATUS_FAILED
            document.chunk_count = 0
            document.error = error

    def mark_document_processing(self, doc_id: str) -> None:
        self.processing.append(doc_id)
        document = self.documents.get(doc_id)
        if document is not None:
            document.status = DOC_STATUS_PROCESSING
            document.chunk_count = 0
            document.error = ""

    def delete_document(self, doc_id: str) -> str | None:
        document = self.documents.pop(doc_id, None)
        return document.kb_id if document else None


class Fixture:
    """一套装配好的流水线 + 服务 + 假件，供每个用例重置。"""

    def __init__(self, upload_dir: Path) -> None:
        self.parser = FakeParser()
        self.splitter = FakeSplitter()
        self.embedder = FakeEmbedder()
        self.vector_store = FakeVectorStore()
        self.repository = MemoryRepository()
        self.pipeline = IngestionPipeline(
            parser=self.parser,
            splitter=self.splitter,
            embedder=self.embedder,
            vector_store=self.vector_store,
            repository=self.repository,
            upload_dir=str(upload_dir),
        )
        self.service = IngestService(
            pipeline=self.pipeline,
            repository=self.repository,
            allowed_extensions=(".txt",),
            max_upload_bytes=10 * 1024 * 1024,
            # 静态安全检查由 check_filesafety.py 单独覆盖；
            # 这里放行，避免"测试内容恰好被判成二进制"污染入库链路的断言。
            inspector=lambda _name, _data: None,
            # 并发闸门设 1：下面的用例要断言"同时提交两份，峰值并发确实是 1"
            gate=ConcurrencyGate(1),
        )
        self.upload_dir = Path(upload_dir)

    def original_path(self, document: Document) -> Path:
        """按流水线的命名规则推出原始文件路径。

        必须 `.resolve()`：Windows 的 `tempfile.mkdtemp()` 返回 8.3 短名
        （`ADMINI~1`），而流水线内部的 `_upload_dir` 已 resolve 成长名，
        不归一化就会拿短名去比长名，断言无缘无故失败。
        """
        return (self.upload_dir / document.kb_id / f"{document.id}_{document.filename}").resolve()


# ---------------------------------------------------------------------------
# 第 1 组：首次入库
# ---------------------------------------------------------------------------


async def case_first_ingest(fixture: Fixture) -> Document:
    print("\n[1] 首次入库：解析 → 切分 → 向量化 → 落库")
    content = "first line\nsecond line\n".encode()
    document = await fixture.pipeline.run("kb-1", "note.txt", content)

    check("status=ready", document.status == DOC_STATUS_READY, document.status)
    check("chunk_count=写入条数（2 行 → 2 块）", document.chunk_count == 2, str(document.chunk_count))
    check("error 为空", document.error == "", repr(document.error))
    check("仓储里有一条元数据", len(fixture.repository.documents) == 1)
    check("create_document 只调了一次", len(fixture.repository.created) == 1)

    path = fixture.original_path(document)
    check("原始文件按 uploads/{kb}/{doc}_{name} 落盘", path.is_file(), str(path))
    check("落盘内容与入参逐字节一致", path.read_bytes() == content)

    chunks = fixture.vector_store.upserts[-1]
    check("每块的 kb_id 被回填", all(chunk.kb_id == "kb-1" for chunk in chunks))
    check("每块的 doc_id 被回填", all(chunk.doc_id == document.id for chunk in chunks))
    check("每块的 filename 被回填", all(chunk.filename == "note.txt" for chunk in chunks))
    check("块 id 互不相同且非空", len({chunk.id for chunk in chunks}) == 2 and all(chunk.id for chunk in chunks))
    check("块 id 不是空串（切分器留空、流水线补 UUID）", all(len(chunk.id) == 36 for chunk in chunks))
    return document


# ---------------------------------------------------------------------------
# 第 2 组：首次失败
# ---------------------------------------------------------------------------


async def case_first_failure(fixture: Fixture) -> Document:
    print("\n[2] 首次失败：损坏文件 → failed + 清理残留 + 元数据行保留")
    fixture.parser.raise_with = CorruptedDocumentError("PDF 结构损坏，无法解析")
    document = await fixture.pipeline.run("kb-1", "broken.txt", b"whatever\n")

    check("status=failed", document.status == DOC_STATUS_FAILED, document.status)
    check(
        "error 带稳定的 code 前缀 [corrupted_document]",
        document.error.startswith("[corrupted_document]"),
        document.error,
    )
    check("chunk_count 归零", document.chunk_count == 0)
    check("元数据行仍在（用户要看得到失败原因）", document.id in fixture.repository.documents)
    check("失败状态落到了仓储", fixture.repository.failed[-1][0] == document.id)
    check("清理过残留向量", document.id in fixture.vector_store.deleted)
    check("原始文件仍保留（重试的前提）", fixture.original_path(document).is_file())
    return document


# ---------------------------------------------------------------------------
# 第 3 组：原地重试成功
# ---------------------------------------------------------------------------


async def case_reprocess_success(fixture: Fixture, failed: Document) -> None:
    print("\n[3] 原地重试：同一 doc_id 从 failed → ready，不产生新记录")
    created_before = len(fixture.repository.created)
    parse_calls_before = len(fixture.parser.calls)
    deleted_before = len(fixture.vector_store.deleted)

    # 修好"文件"（假 parser 只是不再抛错），模拟用户在服务端修好依赖后重试
    fixture.parser.raise_with = None
    result = await fixture.service.reprocess(failed.id, owner_id=None)

    check("返回 status=ready", result["status"] == DOC_STATUS_READY, str(result["status"]))
    check("doc_id 不变", result["id"] == failed.id, result["id"])
    check("没有新建元数据行", len(fixture.repository.created) == created_before)
    check("仓储里仍只有原来的记录数", len(fixture.repository.documents) == created_before)
    check("mark_document_processing 被调用", failed.id in fixture.repository.processing)
    check("解析被重新触发", len(fixture.parser.calls) == parse_calls_before + 1)
    check(
        "解析用的是上次落盘的那一份文件",
        fixture.parser.calls[-1] == str(fixture.original_path(failed)),
        fixture.parser.calls[-1],
    )
    check(
        "解析读到的内容就是原始字节",
        Path(fixture.parser.calls[-1]).read_bytes() == b"whatever\n",
    )
    check("error 被清空", result["error"] == "", repr(result["error"]))
    check("chunk_count 是新写入的条数", result["chunk_count"] == 1, str(result["chunk_count"]))
    check("重试前又清了一次残留向量（幂等兜底）", len(fixture.vector_store.deleted) > deleted_before)


# ---------------------------------------------------------------------------
# 第 4 组：重试的前置条件
# ---------------------------------------------------------------------------


async def case_reprocess_preconditions(fixture: Fixture) -> None:
    print("\n[4] 前置条件：文件丢失 → 404；状态不对 → 409；文档不存在 → 404")

    # 4.1 原始文件丢失
    lost = await fixture.pipeline.run("kb-1", "gone.txt", b"bye\n")
    fixture.original_path(lost).unlink()
    fixture.repository.mark_document_failed(lost.id, "[upstream_timeout] 解析超时")
    lost.status = DOC_STATUS_FAILED

    parse_calls_before = len(fixture.parser.calls)
    await expect_error_async(
        "原始文件丢失 → NotFoundError(404)",
        NotFoundError,
        "重新上传",
        fixture.pipeline.reprocess(lost),
    )
    check("文件丢失时没有触发解析", len(fixture.parser.calls) == parse_calls_before)

    # 4.2 处理中
    busy = Document(id=str(uuid.uuid4()), kb_id="kb-1", filename="busy.txt", status=DOC_STATUS_PROCESSING)
    fixture.repository.documents[busy.id] = busy
    fixture.original_path(busy).parent.mkdir(parents=True, exist_ok=True)
    fixture.original_path(busy).write_bytes(b"busy\n")
    await expect_error_async(
        "processing → ConflictError(409)",
        ConflictError,
        "正在处理中",
        fixture.service.reprocess(busy.id, owner_id=None),
    )

    # 4.3 已成功
    done = Document(id=str(uuid.uuid4()), kb_id="kb-1", filename="done.txt", status=DOC_STATUS_READY)
    fixture.repository.documents[done.id] = done
    await expect_error_async(
        "ready → ConflictError(409)，且提示先删除再重建",
        ConflictError,
        "先删除",
        fixture.service.reprocess(done.id, owner_id=None),
    )

    # 4.4 不存在
    await expect_error_async(
        "文档不存在 → NotFoundError(404)",
        NotFoundError,
        "文档不存在",
        fixture.service.reprocess("no-such-doc", owner_id=None),
    )

    check("ConflictError 的 HTTP 语义是 409", ConflictError.http_status == 409)
    check("NotFoundError 的 HTTP 语义是 404", NotFoundError.http_status == 404)
    check("两个状态都拒了、且只拒了这两次（没有额外解析）", len(fixture.parser.calls) == parse_calls_before)


# ---------------------------------------------------------------------------
# 第 5 组：重试再次失败
# ---------------------------------------------------------------------------


async def case_reprocess_failure_again(fixture: Fixture) -> None:
    print("\n[5] 重试再失败：仍是 failed，error 换成新原因，doc_id 不变")
    document = await fixture.pipeline.run("kb-1", "retry-me.txt", b"content\n")
    # 先人为把它打成 failed，模拟"上次失败"
    fixture.repository.mark_document_failed(document.id, "[embedding_error] 旧原因")
    document.status = DOC_STATUS_FAILED

    # 这次切分为空 → 流水线内抛 AppError
    fixture.splitter.empty = True
    result = await fixture.service.reprocess(document.id, owner_id=None)
    fixture.splitter.empty = False

    check("仍是 failed", result["status"] == DOC_STATUS_FAILED, str(result["status"]))
    check("doc_id 不变", result["id"] == document.id)
    check("error 换成了新原因", "切分结果为空" in result["error"], result["error"])
    check("新原因覆盖了旧原因", "旧原因" not in result["error"], result["error"])
    check("文档仍在列表里（没有被删掉）", document.id in fixture.repository.documents)


# ---------------------------------------------------------------------------
# 第 6 组：路径与异常语义
# ---------------------------------------------------------------------------


async def case_paths_and_semantics(fixture: Fixture, upload_dir: Path) -> None:
    print("\n[6] 路径安全与非业务异常语义")

    # 6.1 同名文件两次入库不互相覆盖（doc_id 前缀的作用）
    first = await fixture.pipeline.run("kb-1", "same.txt", b"one\n")
    second = await fixture.pipeline.run("kb-1", "same.txt", b"two\n")
    check(
        "同名文件按 doc_id 区分，不互相覆盖",
        fixture.original_path(first) != fixture.original_path(second)
        and fixture.original_path(first).read_bytes() == b"one\n"
        and fixture.original_path(second).read_bytes() == b"two\n",
    )

    # 6.2 文件名带路径 → 只取最后一段
    #     走服务层入口（`ingest_file`），因为规范化发生在那里；
    #     直接调 pipeline.run 会原样落 `../`，那不是生产路径。
    result = await fixture.service.ingest_file("kb-1", "../../etc/passwd.txt", b"x\n", owner_id=None)
    document = fixture.repository.get_document(result["id"])
    check(
        "文件名里的 ../ 被剥离，落盘仍在 uploads/kb-1 下",
        fixture.original_path(document).parent == (upload_dir / "kb-1").resolve(),
        str(fixture.original_path(document)),
    )
    check("落盘文件名就是 passwd.txt", fixture.original_path(document).name.endswith("_passwd.txt"),
          fixture.original_path(document).name)

    # 6.3 目录穿越的 kb_id 被清洗（不是抛错，而是收敛成单层目录名）
    escaped = await fixture.pipeline.run("../evil", "a.txt", b"x\n")
    written = Path(fixture.parser.calls[-1]).resolve()
    check(
        "kb_id 带 .. 被清洗，落盘仍在 upload_dir 之内（没逃出去）",
        upload_dir.resolve() in written.parents,
        str(written),
    )
    check("清洗后仍能正常入库，不是靠抛错兜底", escaped.status == DOC_STATUS_READY, escaped.status)

    # 6.4 非业务异常保留类名
    fixture.parser.raise_with = RuntimeError("模拟未预期的解析器崩溃")
    broken = await fixture.pipeline.run("kb-1", "crash.txt", b"y\n")
    fixture.parser.raise_with = None
    check(
        "非 AppError 保留类名（便于搜 issue）",
        broken.error.startswith("RuntimeError: "),
        broken.error,
    )
    check("非 AppError 也让 status=failed", broken.status == DOC_STATUS_FAILED)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


async def case_concurrency(fixture: Fixture) -> None:
    """并发与去重：同一份正文只该入库一次，且入库任务不并行抢占 CPU。

    这两条都不是"最佳实践"，是修过的真实故障：
      - 库里同一份 txt 攒到过 4 条 ready，向量分逐位相同、在 top_k 里互相挤占；
      - 多份大文件同时向量化时，每份都要几百 MB 内存与前向计算，一起提交
        只会让所有请求的 P99 一起变差。
    """
    print("\n[6] 并发与去重")

    before = len(fixture.repository.created)
    first = await fixture.service.ingest_file("kb-1", "same.txt", b"same body\n", owner_id=None)
    check("首次上传 → ready", first["status"] == DOC_STATUS_READY, str(first))
    check("首次上传产生 1 条元数据行", len(fixture.repository.created) == before + 1)

    await expect_error_async(
        "同内容换文件名再传 → 409（按正文哈希，不看文件名）",
        ConflictError,
        "已在此知识库",
        fixture.service.ingest_file("kb-1", "renamed.txt", b"same body\n", owner_id=None),
    )
    check(
        "重复上传没有新建元数据行",
        len(fixture.repository.created) == before + 1,
        f"created={len(fixture.repository.created) - before}",
    )

    # 同名不同内容必须放行：用户改了文档再传是正常操作，不能按名字判重
    changed = await fixture.service.ingest_file("kb-1", "same.txt", b"edited body\n", owner_id=None)
    check("同名不同内容 → 允许（改名不改内容才是重复）", changed["status"] == DOC_STATUS_READY)

    # 两个请求同时提交同一份内容：锁内查重 + 唯一索引双保险，只能成功一个
    results = await asyncio.gather(
        fixture.service.ingest_file("kb-1", "race-a.txt", b"race body\n", owner_id=None),
        fixture.service.ingest_file("kb-1", "race-b.txt", b"race body\n", owner_id=None),
        return_exceptions=True,
    )
    conflicts = [r for r in results if isinstance(r, ConflictError)]
    successes = [r for r in results if isinstance(r, dict)]
    check(
        "并发提交同一份内容 → 只成功一个，另一个 409",
        len(successes) == 1 and len(conflicts) == 1,
        f"success={len(successes)} conflict={len(conflicts)}",
    )

    # 并发闸门：两份**不同**内容同时提交时，峰值并发必须是 1（排队而非抢占）
    await asyncio.gather(
        fixture.service.ingest_file("kb-1", "p1.txt", b"parallel one\n", owner_id=None),
        fixture.service.ingest_file("kb-1", "p2.txt", b"parallel two\n", owner_id=None),
    )
    check(
        "入库并发受闸门限制（峰值 = 1）",
        fixture.service.gate.peak == 1,
        f"peak={fixture.service.gate.peak}",
    )
    check("锁池用完即回收（不随文档数增长）", fixture.service.locks.size == 0, f"size={fixture.service.locks.size}")


async def main() -> int:
    print("=" * 72)
    print("入库链路自检：首次入库 / 失败 / 原地重试")
    print("=" * 72)

    temp_dir = Path(tempfile.mkdtemp(prefix="rag-ingest-"))
    try:
        fixture = Fixture(temp_dir / "uploads")
        fixture.upload_dir.mkdir(parents=True, exist_ok=True)

        document = await case_first_ingest(fixture)
        failed = await case_first_failure(fixture)
        await case_reprocess_success(fixture, failed)
        await case_reprocess_preconditions(fixture)
        await case_reprocess_failure_again(fixture)
        await case_paths_and_semantics(fixture, temp_dir / "uploads")
        await case_concurrency(fixture)

        print("\n[7] 端口一致性")
        # 只断言仓储：本次端口新增了 get_document / mark_document_processing，
        # 假实现最容易在这里脱节。解析器 / 向量库 / 向量化器的完整假件
        # 分别在 check_loading / check_vectorstore / check_embedding 里，
        # 本脚本的假件是"够用就好"的最小实现，不强求满足全部端口方法。
        check(
            "IngestionPipeline 的仓储满足 ChunkRepositoryPort",
            isinstance(fixture.repository, ChunkRepositoryPort),
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 72)
    if _failures:
        print(f"失败 {len(_failures)} / {_checks} 项：")
        for label in _failures:
            print(f"  - {label}")
        return 1
    print(f"全部通过：{_checks} 项断言")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
