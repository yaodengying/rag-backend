"""文档管理路由：上传、列表、删除。

上传注意：
    - 需校验扩展名白名单与体积上限，两者都要在服务层再校验一次
      （接口层校验可被绕过，服务层才是可信边界）。
    - 内容类型不能只看 Content-Type，要按扩展名分发解析器。

关于"流式写盘"与"读进内存"的取舍：
    骨架注释原本建议用 UploadFile 流式写盘。但入库流水线的签名是
    `run(kb_id, filename, content: bytes)`——切分与向量化都需要完整字节串，
    落盘只是"顺带留档"，所以这里按 1 MB 分片累加读取。

    关键点是**读取时就封顶**：最多读 `max_upload_bytes + 1` 字节。
    多读的那 1 字节是有意为之——它让服务层能判定"超限"并给出统一错误
    消息，同时保证 1 GB 的文件不会先被读进内存再报错。
    体积校验因此只有一处实现（服务层），接口层不重复判断。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.api.deps import get_collection_service, get_ingest_service
from app.api.v1.auth import require_current_user, router_dependencies
from app.schemas.auth import UserOut
from app.core.config import get_settings
from app.core.exceptions import ValidationError
from app.schemas.document import DocumentOut
from app.services.collection_service import CollectionService
from app.services.ingest_service import IngestService

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=router_dependencies)

_READ_CHUNK = 1024 * 1024  # 1 MB


async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    """分片读取上传内容，最多读 max_bytes + 1 字节。

    多读的那 1 字节用于让服务层判定超限（见模块 docstring）。
    """
    hard_limit = max_bytes + 1
    pieces: list[bytes] = []
    total = 0
    while total < hard_limit:
        piece = await file.read(min(_READ_CHUNK, hard_limit - total))
        if not piece:
            break
        pieces.append(piece)
        total += len(piece)
    return b"".join(pieces)


@router.get(
    "/knowledge-bases/{kb_id}/documents",
    response_model=list[DocumentOut],
    summary="文档列表",
)
async def list_documents(
    kb_id: str,
    current_user: UserOut = Depends(require_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> list[DocumentOut]:
    """列出某知识库下的文档（仅限本人名下）。

    Raises:
        NotFoundError: 知识库不存在或不属于当前账号
    """
    return [
        DocumentOut.model_validate(item)
        for item in service.list_documents(kb_id, current_user.id)
    ]


@router.post(
    "/knowledge-bases/{kb_id}/documents",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
    summary="上传单个文档",
)
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    current_user: UserOut = Depends(require_current_user),
    service: IngestService = Depends(get_ingest_service),
) -> DocumentOut:
    """上传并入库单个文档。

    支持格式：.pdf / .docx / .txt / .md / .pptx

    Raises:
        ValidationError: 扩展名不支持或超出体积上限
        NotFoundError: 知识库不存在或不属于当前账号
    """
    content = await _read_upload(file, get_settings().max_upload_bytes)
    result = await service.ingest_file(
        kb_id, file.filename or "unnamed", content, current_user.id
    )
    return DocumentOut.model_validate(result)


@router.post(
    "/knowledge-bases/{kb_id}/documents/batch",
    response_model=list[DocumentOut],
    summary="批量上传文档",
)
async def upload_documents_batch(
    kb_id: str,
    files: list[UploadFile] = File(...),
    current_user: UserOut = Depends(require_current_user),
    service: IngestService = Depends(get_ingest_service),
) -> list[DocumentOut]:
    """批量上传；单个文件失败不影响其余文件。

    返回结果与入参**顺序一一对应**：被拒绝的文件也会出现在结果里，
    status=failed 且 error 说明原因（如扩展名不支持、超出体积上限），
    否则前端无法知道"是哪几个文件没成功"。

    两个**整批**闸门（单文件上限管不住它们）：
      - 文件数 > `MAX_BATCH_FILES` → 400，一个都不处理；
      - 累计体积 > `MAX_BATCH_TOTAL_MB` → 400，同样整批拒绝。
        累计检查在**读的过程中**做：超限后立刻停读，剩下的文件
        既不入内存也不落盘——否则"拒绝"只是嘴上说说，字节已经进来了。
    """
    settings = get_settings()
    if len(files) > settings.max_batch_files:
        raise ValidationError(
            f"一次最多上传 {settings.max_batch_files} 个文件，当前 {len(files)} 个；"
            "请分批上传"
        )

    max_bytes = settings.max_upload_bytes
    total_limit = settings.max_batch_total_bytes
    payload: list[tuple[str, bytes]] = []
    total = 0
    for file in files:
        # 单个文件的读取仍然封顶在 max_upload_bytes + 1：
        # 超限的那份要能被服务层识别成"超限"并逐条记 failed，
        # 而不是整批 400（批量上传的语义是"尽力而为"）。
        content = await _read_upload(file, max_bytes)
        total += len(content)
        if total > total_limit:
            raise ValidationError(
                f"本批累计 {total / 1024 / 1024:.1f} MB，超过上限 "
                f"{settings.max_batch_total_mb} MB；请分批上传"
            )
        payload.append((file.filename or "unnamed", content))

    results = await service.ingest_batch(kb_id, payload, current_user.id)
    return [DocumentOut.model_validate(item) for item in results]


@router.post(
    "/documents/{doc_id}/reprocess",
    response_model=DocumentOut,
    summary="重试失败的文档",
)
async def reprocess_document(
    doc_id: str,
    current_user: UserOut = Depends(require_current_user),
    service: IngestService = Depends(get_ingest_service),
) -> DocumentOut:
    """原地重跑一份失败的文档，复用上次落盘的原始文件。

    不需要重新上传文件——失败时原始文件是保留的，本接口直接读它重跑
    「解析 → 切分 → 向量化 → 落库」，**doc_id 不变**（重传则会产生新记录）。

    只对 `status=failed` 有效：
      - `processing` → 409（正在跑，重复触发会写两遍向量）；
      - `ready` → 409（已成功，要重建索引请先删除再上传）。

    注意返回语义与上传一致：**处理阶段再次失败仍是 200**，由响应体的
    `status=failed` 与 `error` 告知原因；只有"文档不存在 / 原始文件丢失"
    （404）、"状态不允许"（409）才是 HTTP 错误。
    """
    result = await service.reprocess(doc_id, current_user.id)
    return DocumentOut.model_validate(result)


@router.delete(
    "/documents/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除文档",
)
async def delete_document(
    doc_id: str,
    current_user: UserOut = Depends(require_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> None:
    """删除文档及其向量与原始文件（仅限本人名下知识库中的文档）。"""
    service.delete_document(doc_id, current_user.id)
