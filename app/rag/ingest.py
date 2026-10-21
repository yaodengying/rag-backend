"""入库链：文件 → 解析 → 切分 → 向量化 → 落库。

与问答图的区别：入库是**线性流程**，没有条件分支，因此不用 LangGraph，
用一个显式的 Pipeline 类即可，可读性更好。

流水线各步与其副作用：

    1. 登记元数据      写 SQLite，status=processing
    2. 原始文件落盘    data/uploads/{kb_id}/{doc_id}_{filename}
    3. 加载 + 解析     LangChain Loader → Document → list[Page]（带页码）
    4. 切分            list[Page] → list[Chunk]（带页码溯源）
    5. 向量化          list[Chunk] → 回填 embedding（分批与进度由向量化器自理）
    6. 写入向量库      Milvus upsert
    7. 更新元数据      status=ready 且写入 chunk_count

失败处理约定：
    任何一步抛错都要把文档标记为 status=failed 并记录 error，
    **不要**留下永远 processing 的僵尸记录——前端会一直转圈。
    元数据行保留（便于用户看到失败原因），原始文件也保留（便于重试）。

失败后的重试：
    `reprocess()` 复用**已落盘的原始文件**原地重跑，doc_id 不变、不产生新记录。
    这条路径存在的理由就是上面那句"原始文件也保留"——否则用户重试只能重传，
    而重传会生成第二个 doc_id，留下「一个 failed + 一个 ready」两条记录，
    并且把同一份正文的向量写两遍（实测库里同一份 txt 曾攒到 4 条 ready）。
    因此 `reprocess()` 只在**原始文件还在**时可用；文件丢了只能重传。

重要：SQLite 只存元数据，不存片段正文与向量。
      正文的权威副本在 uploads 原始文件 + Milvus 的 text 字段。

实现说明：
    - 解析 / 切分 / 向量化 / 写入都是**阻塞的 CPU 或网络调用**，
      统一用 asyncio.to_thread 挪到线程池，避免卡死事件循环
      （向量化一份 50 MB 文档可能要几十秒，同步执行会让整个服务失去响应）。
    - 编号顺序有意保持：先登记元数据再落盘，这样"文件已落盘但没元数据"
      这种孤儿状态不会出现；反过来"元数据 processing 但文件没落盘"
      在失败时会被标记为 failed，是可解释的。
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path

from app.core.exceptions import AppError, NotFoundError
from app.domain.models import (
    DOC_STATUS_FAILED,
    DOC_STATUS_PROCESSING,
    DOC_STATUS_READY,
    Chunk,
    Document,
)
from app.domain.ports import (
    ChunkRepositoryPort,
    DocumentParserPort,
    EmbedderPort,
    SplitterPort,
    VectorStorePort,
)

logger = logging.getLogger(__name__)

# Windows / POSIX 双平台的文件名非法字符
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class IngestionPipeline:
    """文档入库流水线。"""

    def __init__(
        self,
        parser: DocumentParserPort,
        splitter: SplitterPort,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        repository: ChunkRepositoryPort,
        upload_dir: str,
    ) -> None:
        """注入全部依赖（便于测试替换）。

        Args:
            parser: 文档解析器
            splitter: 文本切分器
            embedder: 向量化器
            vector_store: 向量库
            repository: 元数据仓储
            upload_dir: 原始文件落盘根目录
        """
        self._parser = parser
        self._splitter = splitter
        self._embedder = embedder
        self._vector_store = vector_store
        self._repository = repository
        self._upload_dir = Path(upload_dir).expanduser().resolve()
        self._upload_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        kb_id: str,
        filename: str,
        content: bytes,
        content_hash: str = "",
    ) -> Document:
        """执行完整入库流程。

        Args:
            kb_id: 目标知识库 ID
            filename: 原始文件名（用于展示与落盘命名）
            content: 文件二进制内容
            content_hash: 正文 SHA-256；由服务层算好传入，用于阻止重复入库。
                空串表示不参与去重（脚本、历史调用方）。

        Returns:
            处理后的文档元数据；失败时 status=failed 且 error 非空。

        Raises:
            NotFoundError: 知识库不存在
            ConflictError: 同库已有相同 hash 的文档（并发兜底，见仓储实现）
        """
        # 步骤 1：登记元数据。kb 不存在时在这里就抛 NotFoundError，
        # 此时尚未落盘、未写向量，无需清理。
        document = self._repository.create_document(
            kb_id, filename, content_hash=content_hash
        )

        try:
            # 步骤 2
            file_path = self._save_original(kb_id, document.id, filename, content)
        except Exception as exc:  # noqa: BLE001 - 落盘失败同样要标 failed
            # 落盘失败时**还没有**元数据之外的副作用，清理向量是空操作（幂等）
            return self._mark_failed(document, exc, filename)

        return await self._process(document, file_path)

    async def reprocess(self, document: Document) -> Document:
        """原地重跑一份已存在的文档（失败重试用）。

        与 `run()` 的差别只有三处：
            1. **不新建元数据行**，doc_id 保持不变（否则就是"重传"而不是"重试"）；
            2. **不接收文件内容**，直接读 uploads 里上次落盘的原始文件；
            3. 先把状态翻回 processing 并清空 error，再走同一套处理步骤。

        前置条件由调用方（`IngestService.reprocess`）校验：只有失败的文档
        才允许重试，processing / ready 分别会抛 409。本方法只处理
        "原始文件还在不在"这一个物理前提。

        Args:
            document: 待重试的文档元数据（必须已存在于仓储中）

        Returns:
            处理后的文档元数据；失败时 status=failed 且 error 非空。

        Raises:
            NotFoundError: 原始文件已不在 uploads 目录（被手工清理或换了机器）
        """
        file_path = self._original_path(document.kb_id, document.id, document.filename)
        if not file_path.is_file():
            raise NotFoundError(
                f"原始文件已丢失，无法原地重试：{document.filename}。请重新上传该文件"
            )

        # 上一次失败可能残留了半截向量（首次失败时 `_cleanup_partial` 自己也
        # 可能失败，它的异常是被吞掉的）。重试前再清一次是幂等的：
        # 没写进去时 delete 返回 0，不会报错。
        self._cleanup_partial(document.id)

        self._repository.mark_document_processing(document.id)
        document.status = DOC_STATUS_PROCESSING
        document.chunk_count = 0
        document.error = ""

        return await self._process(document, str(file_path))

    async def _process(self, document: Document, file_path: str) -> Document:
        """执行步骤 3~7（解析 → 切分 → 向量化 → 落库 → 更新状态）。

        `run()` 与 `reprocess()` 共用这一段：两者的差别只在"文件从哪来"
        和"元数据行是否新建"，处理逻辑必须只有一份，否则重试路径会慢慢
        与首次入库漂移（例如忘了清理残留向量、忘了带上 chunk 的归属字段）。
        """
        kb_id = document.kb_id
        filename = document.filename
        try:
            # 步骤 3
            pages = await asyncio.to_thread(self._parser.parse, file_path)

            # 步骤 4
            chunks = await asyncio.to_thread(self._splitter.split, pages)
            if not chunks:
                raise AppError("切分结果为空：文档没有可用文本")

            # 补齐片段归属（切分器只负责文本层面）
            for chunk in chunks:
                chunk.id = str(uuid.uuid4())
                chunk.kb_id = kb_id
                chunk.doc_id = document.id
                chunk.filename = filename

            # 步骤 5
            await asyncio.to_thread(self._attach_embeddings, chunks)

            # 步骤 6
            written = await asyncio.to_thread(self._vector_store.upsert, chunks)

            # 步骤 7
            self._repository.mark_document_ready(document.id, written)
            document.status = DOC_STATUS_READY
            document.chunk_count = written
            document.error = ""
            logger.info(
                "入库完成：kb=%s doc=%s file=%s chunks=%s",
                kb_id,
                document.id,
                filename,
                written,
            )
            return document

        except Exception as exc:  # noqa: BLE001 - 兜住一切，转为 status=failed
            return self._mark_failed(document, exc, filename)

    # ---------- 内部 ----------

    def _mark_failed(self, document: Document, exc: Exception, filename: str) -> Document:
        """清理残留向量、写失败状态，并回填到领域对象上。

        统一入口：`run()`（落盘失败）、`_process()`（处理失败）都走这里，
        避免两处各写一遍"清理 + 拼错误消息 + 兜住标记失败时的二次异常"。
        """
        self._cleanup_partial(document.id)
        # 业务异常带上稳定的 code（`[corrupted_document] ...`）：前端要能
        # 把「文件坏了，重新导出」与「服务坏了，等运维」分开提示，
        # 而 error 字段是它们唯一能看到的线索。非业务异常保留类名，
        # 便于直接搜 issue（那种情况人不该从消息里读结论）。
        if isinstance(exc, AppError):
            message = f"[{exc.code}] {exc.message}"
        else:
            message = f"{type(exc).__name__}: {exc}"
        try:
            self._repository.mark_document_failed(document.id, message)
        except Exception:  # noqa: BLE001
            logger.exception("标记文档失败状态时又出错：doc=%s", document.id)
        document.status = DOC_STATUS_FAILED
        document.chunk_count = 0
        document.error = message
        logger.warning("入库失败：doc=%s file=%s err=%s", document.id, filename, message)
        return document

    def _original_path(self, kb_id: str, doc_id: str, filename: str) -> Path:
        """原始文件的落盘路径（不写盘，只拼路径）。

        与 `_save_original` 共用同一套命名规则，重试时才能找到上次那份文件。
        拆成独立方法是为了让"路径怎么来"只有一处定义——两处各写一遍
        `f"{doc_id}_{filename}"`，改一处忘一处就会变成"重试永远报文件丢失"。
        """
        target_dir = (self._upload_dir / self._safe_component(kb_id)).resolve()
        # 防目录穿越：kb_id 若被构造成 ../../.. 也不能逃出 upload_dir
        if target_dir != self._upload_dir and self._upload_dir not in target_dir.parents:
            raise AppError(f"非法的知识库目录：{kb_id}")
        return target_dir / f"{doc_id}_{self._safe_filename(filename)}"

    def _save_original(self, kb_id: str, doc_id: str, filename: str, content: bytes) -> str:
        """把原始文件写入 uploads 目录并返回绝对路径。

        注意：
            文件名要带 doc_id 前缀，避免同名文件互相覆盖。
            同时需防目录穿越（filename 可能含 ../）。
        """
        path = self._original_path(kb_id, doc_id, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return str(path)

    def _attach_embeddings(self, chunks: list[Chunk]) -> list[Chunk]:
        """为片段批量回填 embedding 字段。

        整份列表直接交给向量化器，**本层不再自己分组**：
        批大小是模型自己的事（``settings.embedding_batch_size``，由适配器
        分批并报进度）。两处各写一个 64 只会随时间互相漂移，
        而且会让人误以为改这里能影响编码批大小。
        """
        vectors = self._embedder.embed_documents([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise AppError(
                f"向量化结果数量不匹配：期望 {len(chunks)}，实际 {len(vectors)}"
            )
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.embedding = vector
        return chunks

    def _cleanup_partial(self, doc_id: str) -> None:
        """失败时清理已写入的向量，避免半成品残留。

        upsert 是单次调用，但 Milvus 内部可能部分成功；
        按 doc_id 删除是幂等的，没有写入时 delete_count 为 0。
        清理本身失败只记日志——不能让它盖掉真正的失败原因。
        """
        try:
            removed = self._vector_store.delete_by_doc(doc_id)
            if removed:
                logger.info("已清理失败文档的残留向量：doc=%s 条数=%s", doc_id, removed)
        except Exception:  # noqa: BLE001
            logger.exception("清理残留向量失败：doc=%s", doc_id)

    @staticmethod
    def _safe_component(value: str) -> str:
        """把任意字符串收敛成单一安全的路径片段。"""
        cleaned = _INVALID_FILENAME_CHARS.sub("_", value or "").strip(" .")
        if not cleaned:
            raise AppError("路径片段非法")
        return cleaned

    @classmethod
    def _safe_filename(cls, filename: str) -> str:
        """取文件名的最后一段并清理非法字符。"""
        # Path(...).name 在 Windows 上同时识别 \\ 与 /，
        # 因此 "..\\..\\etc\\passwd" 会被收敛成 "passwd"
        name = Path(filename or "").name
        cleaned = _INVALID_FILENAME_CHARS.sub("_", name).strip(" .")
        return cleaned or "unnamed"
