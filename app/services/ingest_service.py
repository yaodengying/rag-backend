"""入库服务：把上传的文件交给入库流水线，并回传文档状态。

与 IngestionPipeline 的分工：
    IngestionPipeline 是纯流程（解析→切分→向量化→落库），不关心 HTTP；
    IngestService 负责入口校验（知识库是否存在、扩展名是否允许、
    体积是否超限、内容是否安全）与结果转换（domain.Document → schema.DocumentOut）。

校验为什么放在这一层（而不是只放接口层）：
    接口层校验可以被绕过——脚本、定时任务、别的服务都可能直接调本层。
    服务层才是可信边界，所以白名单、体积上限、静态安全检查在这里**再校验一次**。

多人并发在这里解决三件事（细节见 `app/rag/concurrency.py`）：
    ① **同内容去重**：按正文 SHA-256 查重，命中直接 409（带已有 doc_id），
       不再靠"同一份文件传 4 遍、向量挤占 top_k"来暴露问题；
    ② **同 key 串行**：上传与重试都在 `(kb_id, hash)` / `doc_id` 的锁内，
       避免"检查通过 → 另一请求也通过 → 两份向量"的竞态；
    ③ **全局排队**：入库并发受 `INGEST_MAX_CONCURRENCY` 限制，超出的请求等待。

失败语义：
    - 入口校验失败（扩展名/体积/不安全内容/重复）→ 抛异常，**不产生元数据行**；
    - 流水线内部失败（解析/向量化/写入）→ 返回 status=failed 的文档，
      HTTP 仍是 201，由 status 与 error 字段告诉前端具体原因。
      这样用户能在文档列表里看到失败记录和原因，而不是只看到一个报错弹窗。
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable

from app.core.config import get_settings
from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.domain.models import DOC_STATUS_PROCESSING, DOC_STATUS_READY
from app.domain.ports import ChunkRepositoryPort
from app.infrastructure.filesafety import SafetyLimits, inspect_upload
from app.rag.concurrency import ConcurrencyGate, DocumentLockPool
from app.rag.ingest import IngestionPipeline
from app.services.mappers import document_to_dict, rejected_document_dict
from app.services.ownership import ensure_kb_owned

logger = logging.getLogger(__name__)

# 静态检查的签名：只为自检能替换成"放行"或"专抛某错"的假实现
Inspector = Callable[[str, bytes], None]


class IngestService:
    """文档入库服务。"""

    def __init__(
        self,
        pipeline: IngestionPipeline,
        repository: ChunkRepositoryPort,
        allowed_extensions: tuple[str, ...] | None = None,
        max_upload_bytes: int | None = None,
        inspector: Inspector | None = None,
        locks: DocumentLockPool | None = None,
        gate: ConcurrencyGate | None = None,
    ) -> None:
        """注入流水线与仓储。

        Args:
            pipeline: 入库流水线
            repository: 元数据仓储
            allowed_extensions: 扩展名白名单；None 时取 settings
            max_upload_bytes: 单文件体积上限；None 时取 settings
            inspector: 内容静态检查；None 时用 filesafety（取当前配置阈值）
            locks: 按 key 的互斥锁池；None 时新建（单例由容器注入）
            gate: 全局并发闸门；None 时按 settings 新建
        """
        settings = get_settings()
        self._pipeline = pipeline
        self._repository = repository
        self._allowed = tuple(
            ext.lower() for ext in (allowed_extensions or settings.upload_extension_list)
        )
        self._max_bytes = int(
            max_upload_bytes if max_upload_bytes is not None else settings.max_upload_bytes
        )
        self._inspect: Inspector = inspector or self._default_inspector
        self._locks = locks or DocumentLockPool()
        self._gate = gate or ConcurrencyGate(settings.ingest_max_concurrency)

    @staticmethod
    def _default_inspector(filename: str, content: bytes) -> None:
        inspect_upload(filename, content, limits=SafetyLimits.from_settings())

    @property
    def allowed_extensions(self) -> tuple[str, ...]:
        """当前生效的扩展名白名单。"""
        return self._allowed

    @property
    def max_upload_bytes(self) -> int:
        """当前生效的单文件体积上限（字节）。"""
        return self._max_bytes

    @property
    def gate(self) -> ConcurrencyGate:
        """并发闸门（自检要断言它真的生效）。"""
        return self._gate

    @property
    def locks(self) -> DocumentLockPool:
        """锁池（自检要断言用完会回收）。"""
        return self._locks

    async def ingest_file(
        self, kb_id: str, filename: str, content: bytes, owner_id: str
    ) -> dict:
        """处理单个上传文件。

        Args:
            kb_id: 目标知识库
            filename: 原始文件名
            content: 文件内容
            owner_id: 当前登录账号 ID（用于归属校验，**不传就无法上传到别人的库**）

        Returns:
            文档 DTO（含 status 与 chunk_count）。

        Raises:
            NotFoundError: 知识库不存在或不属于该账号
            ValidationError: 扩展名不支持或超出体积上限
            UnsafeFileError: 内容未通过静态检查
            ConflictError: 同一知识库已有相同正文的文档
        """
        self._ensure_owned(kb_id, owner_id)
        return await self._ingest_one(kb_id, filename, content)

    async def ingest_batch(
        self, kb_id: str, files: list[tuple[str, bytes]], owner_id: str
    ) -> list[dict]:
        """批量入库，返回每个文件的处理结果。

        单个文件失败不中断整批：入口校验失败的记一条 failed 结果，
        流水线失败的由流水线自己返回 status=failed。

        Raises:
            NotFoundError: 知识库不存在或不属于该账号
        """
        # 知识库是否存在是**请求级**判断，不是文件级问题：
        # 在这里先拦一次，避免"库不存在"被降级成 N 条逐文件失败。
        # 归属校验同属请求级：越权不该表现为"每个文件都失败"，
        # 那会让攻击者以为"库存在但格式不对"。
        self._ensure_owned(kb_id, owner_id)

        results: list[dict] = []
        for filename, content in files:
            display_name = self._normalize_filename(filename)
            try:
                results.append(await self._ingest_one(kb_id, display_name, content))
            except (ValidationError, ConflictError) as exc:
                # UnsafeFileError / 重复上传都不该拖垮整批：
                # 逐条记原因（前端要能指出是哪几个文件、为什么没进）。
                logger.warning("批量上传中跳过 %s：%s", display_name, exc)
                results.append(rejected_document_dict(kb_id, display_name, str(exc)))
        return results

    async def reprocess(self, doc_id: str, owner_id: str) -> dict:
        """原地重试一份失败的文档（复用已落盘的原始文件）。

        为什么需要这个方法：
            失败时原始文件是保留的，但此前没有任何入口去用它，
            用户只能重新上传 —— 而重传会生成**新的 doc_id**：库里多一条
            ready 记录、失败记录还挂着，同一份正文的向量也写了两份
            （实测同一份 txt 攒到过 4 条 ready，检索时互相挤占名额）。

        只允许重试 `failed`：
            - `processing`：正在跑，再来一次就是同一份文档两个线程写向量；
            - `ready`：已经成功，再跑一遍是"重建索引"而非"重试"，
              语义不同（需要先清空再写，重建窗口内检索不到它），
              不在本方法职责内 —— 要重建请先删除再上传。

        状态检查放在**锁内重读**：调用方拿到的 document 可能已经过期
        （几十毫秒前另一个请求把它置成了 processing）。用旧快照判断
        "可以重试"是这条链路唯一的竞态点。

        Args:
            doc_id: 文档 ID
            owner_id: 当前登录账号 ID

        Returns:
            文档 DTO；重试仍失败时 status=failed、error 为新原因。

        Raises:
            NotFoundError: 文档不存在、不属于该账号，或原始文件已丢失
            ConflictError: 文档正在处理中，或已成功入库
        """
        async with self._locks.acquire(f"doc:{doc_id}"):
            document = self._repository.get_document(doc_id)
            if document is None:
                raise NotFoundError(f"文档不存在：{doc_id}")
            # 归属校验放在锁内、状态判断之前：越权者不该从"正在处理中/已成功"
            # 这类提示里推断出别人的文档处于什么状态。
            self._ensure_owned(document.kb_id, owner_id)

            if document.status == DOC_STATUS_PROCESSING:
                raise ConflictError(
                    "该文档正在处理中，请等待完成后再试；重复触发会让同一份文档被写两遍向量"
                )
            if document.status == DOC_STATUS_READY:
                raise ConflictError(
                    "该文档已成功入库，无需重试；若要重建索引，请先删除再重新上传"
                )

            async with self._gate.slot():
                retried = await self._pipeline.reprocess(document)
            return document_to_dict(retried)

    # ---------- 内部 ----------

    async def _ingest_one(self, kb_id: str, filename: str, content: bytes) -> dict:
        """校验 → 去重 → 排队入库。上传与批量共用这一条路径。

        锁的 key 用 `(kb_id, 正文 hash)` 而不是 `(kb_id, 文件名)`：
        不同名字、同样内容的文件才是真正会重复写入的情形；
        同名不同内容（用户改了文档再传）应当允许。
        """
        display_name = self._normalize_filename(filename)
        self._validate(display_name, content)

        digest = hashlib.sha256(content).hexdigest()
        async with self._locks.acquire(f"hash:{kb_id}:{digest}"):
            existing = self._repository.find_document_by_hash(kb_id, digest)
            if existing is not None:
                raise ConflictError(
                    f"该文件内容已在此知识库中（文档 {existing.filename} / "
                    f"{existing.id}），无需重复上传",
                    detail={"doc_id": existing.id, "filename": existing.filename},
                )
            async with self._gate.slot():
                document = await self._pipeline.run(
                    kb_id, display_name, content, content_hash=digest
                )
        return document_to_dict(document)

    @staticmethod
    def _normalize_filename(filename: str | None) -> str:
        """去掉路径成分，只保留文件名。"""
        name = Path(filename or "").name.strip()
        return name or "unnamed"

    def _ensure_owned(self, kb_id: str, owner_id: str) -> None:
        """归属校验的转发口（实现见 `services/ownership.py`）。

        本服务的 repository 是**必填**构造参数，所以不会出现
        "传了 owner_id 却没有仓储"的装配错误；统一的实现里仍保留那条
        防御分支，给 repository 可选的服务共用。

        Raises:
            NotFoundError: 知识库不存在或不属于该账号
        """
        ensure_kb_owned(self._repository, kb_id, owner_id)

    def _validate(self, filename: str, content: bytes) -> None:
        """入口校验：扩展名白名单、体积上限、内容静态安全。

        **顺序有意**：先做最便宜的判断（空、后缀、体积），再看内容。
        一个 500MB 的假 PDF 应当在读魔数之前就被体积挡住。
        """
        if not content:
            raise ValidationError(f"文件内容为空：{filename}")

        suffix = Path(filename).suffix.lower()
        if suffix not in self._allowed:
            supported = "、".join(self._allowed)
            raise ValidationError(
                f"不支持的扩展名：{suffix or '(无扩展名)'}；当前支持 {supported}"
            )

        size = len(content)
        if size > self._max_bytes:
            raise ValidationError(
                f"文件超出体积上限：{_human_size(size)} > {_human_size(self._max_bytes)}"
            )

        self._inspect(filename, content)


def _human_size(num_bytes: int) -> str:
    """把字节数渲染成带单位的大小，避免出现「0.0 MB > 0 MB」这种废话。"""
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / 1024 / 1024:.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"
