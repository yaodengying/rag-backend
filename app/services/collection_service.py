"""知识库管理服务：增删查与文档增删。

跨存储一致性（本层最重要的职责）：
    删除知识库 / 文档时，必须同时清理四处：
        1. SQLite 元数据
        2. Milvus 中的向量
        3. uploads 中的原始文件
        4. Redis 中该知识库的会话历史
    任何一处漏掉都会留下孤儿数据——向量还在但不能被检索、
    文件还在但用户看不到、会话还在但点进去知识库已不存在。
    实现策略是：先删元数据（用户视角立即生效），
    再尽力清理向量、文件与会话，清理失败只记日志不回滚。

为什么第 4 项（会话）必须在这里清：
    会话的 kb_id 指向知识库，知识库是这个关系的"拥有者"。删库不删会话，
    用户会在会话列表里看到一批点不开的记录，而清理入口（会话页）本身
    可能因知识库缺失而无法渲染——等于把脏数据的清理责任推给了用户。

为什么先删元数据而不是最后删：
    用户的意图是"删掉它"。若先删向量再删元数据，中间失败会留下
    「列表里还看得见、但已经搜不出内容」的僵尸条目，更难解释。
    反过来先删元数据，最坏情况是磁盘上多几个没人引用的文件/向量，
    属于可后台清理的"脏数据"，而不是"错数据"。
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.exceptions import KB_NOT_FOUND_MESSAGE, NotFoundError
from app.domain.ports import ChunkRepositoryPort, VectorStorePort
from app.services.mappers import document_to_dict

if TYPE_CHECKING:
    from app.domain.ports import ConversationStorePort

logger = logging.getLogger(__name__)


class CollectionService:
    """知识库与文档的管理服务。"""

    def __init__(
        self,
        repository: ChunkRepositoryPort,
        vector_store: VectorStorePort,
        upload_dir: str | None = None,
        conversation_store: "ConversationStorePort | None" = None,
    ) -> None:
        """注入仓储、向量库与会话存储。

        Args:
            repository: 元数据仓储
            vector_store: 向量库（删除时需同步清理）
            upload_dir: 原始文件根目录；None 时取 settings.upload_dir
            conversation_store: 会话存储；None 表示未启用会话能力
                （REDIS_URL 为空），此时跳过会话清理而不是报错——
                没会话能力就没有会话可清，不该让删库因此失败。
        """
        from app.core.config import get_settings

        self._repository = repository
        self._vector_store = vector_store
        self._upload_dir = Path(upload_dir or get_settings().upload_dir).expanduser().resolve()
        self._conversation_store = conversation_store

    # ---------- 知识库 ----------

    def list_knowledge_bases(self, owner_id: str) -> list[dict]:
        """列出当前账号名下的知识库。

        Args:
            owner_id: 当前登录账号 ID（多租户隔离的唯一依据）
        """
        return self._repository.list_knowledge_bases(owner_id)

    def ensure_exists(self, kb_id: str, owner_id: str) -> dict:
        """确认知识库存在**且属于该账号**，否则抛 NotFoundError。

        独立成一个方法而不是各处内联判断：检索路由也需要"知识库不存在
        就 404"，而它并不想顺带把文档列表查出来——只为校验而拉全量数据
        是明显的浪费。

        **越权一律按"不存在"处理**（404 而不是 403）。理由：
        403 等于告诉对方"这个 ID 确实存在，只是不给你"，
        配合遍历就能枚举出别人有哪些知识库、进而猜出业务规模与命名习惯。
        404 让"不存在"与"不是你的"在响应上逐字相同。

        Args:
            kb_id: 知识库 ID
            owner_id: 当前登录账号 ID

        Returns:
            该知识库的字典表示（避免调用方再查一次）。

        Raises:
            NotFoundError: 知识库不存在，或不属于该账号
        """
        kb = self._repository.get_knowledge_base(kb_id, owner_id)
        if kb is None:
            raise NotFoundError(KB_NOT_FOUND_MESSAGE)
        return kb

    def create_knowledge_base(self, name: str, description: str, owner_id: str) -> dict:
        """新建知识库（归属为当前账号）。"""
        return self._repository.create_knowledge_base(name, description, owner_id)

    def update_knowledge_base(
        self, kb_id: str, name: str, description: str, owner_id: str
    ) -> dict:
        """更新知识库名称与描述（仅限本人名下）。

        Raises:
            NotFoundError: 知识库不存在或不属于该账号
        """
        return self._repository.update_knowledge_base(kb_id, name, description, owner_id)

    def delete_knowledge_base(self, kb_id: str, owner_id: str) -> bool:
        """删除知识库及其全部文档、向量、文件与会话。

        四步清理，元数据先删（用户视角立即生效），其余尽力而为：
            元数据 → Milvus 向量 → 上传文件 → Redis 会话

        Raises:
            NotFoundError: 知识库不存在或不属于该账号（**必须在清理之前**判定，
                否则会去删别人的向量与文件）
        """
        # 归属校验先做：这一步同时挡掉了"用别人的 kb_id 触发级联清理"——
        # 那不只是读越权，而是**删除**别人的全部数据。
        self.ensure_exists(kb_id, owner_id)

        deleted = self._repository.delete_knowledge_base(kb_id, owner_id)

        # 元数据已删，下面三步是"尽力而为"：失败只记日志
        self._best_effort(
            lambda: self._vector_store.delete_by_kb(kb_id),
            f"清理知识库向量（kb={kb_id}）",
        )
        self._best_effort(
            lambda: self._remove_tree(self._upload_dir / kb_id),
            f"清理知识库上传文件（kb={kb_id}）",
        )
        if self._conversation_store is not None:
            self._best_effort(
                lambda: self._conversation_store.delete_by_kb(kb_id),
                f"清理知识库会话历史（kb={kb_id}）",
            )
        return deleted

    # ---------- 文档 ----------

    def list_documents(self, kb_id: str, owner_id: str) -> list[dict]:
        """列出知识库下的文档（仅限本人名下）。

        Raises:
            NotFoundError: 知识库不存在或不属于该账号
        """
        self.ensure_exists(kb_id, owner_id)
        return [document_to_dict(doc) for doc in self._repository.list_documents(kb_id)]

    def delete_document(self, doc_id: str, owner_id: str) -> bool:
        """删除文档及其向量与原始文件（仅限本人名下知识库中的文档）。

        Raises:
            NotFoundError: 文档不存在，或其所属知识库不属于该账号
        """
        # 先按 doc_id 取回它所属的 kb，再做归属校验。
        # **不能**跳过这一步直接删：文档的归属是经由 kb 间接表达的，
        # 而 delete_document 会同时清掉 Milvus 向量与磁盘文件——
        # 那是不可逆操作，绝不能靠"猜这个 doc 大概是我的"。
        doc = self._repository.get_document(doc_id)
        if doc is None:
            raise NotFoundError(f"文档不存在：{doc_id}")
        self.ensure_exists(doc.kb_id, owner_id)

        kb_id = self._repository.delete_document(doc_id)
        if kb_id is None:
            raise NotFoundError(f"文档不存在：{doc_id}")

        self._best_effort(
            lambda: self._vector_store.delete_by_doc(doc_id),
            f"清理文档向量（doc={doc_id}）",
        )
        self._best_effort(
            lambda: self._remove_document_files(kb_id, doc_id),
            f"清理文档上传文件（doc={doc_id}）",
        )
        return True

    # ---------- 内部工具 ----------

    @staticmethod
    def _best_effort(action: Callable[[], Any], label: str) -> None:
        """执行清理动作，失败只记日志（不回滚已完成的删除）。"""
        try:
            action()
        except Exception:  # noqa: BLE001
            logger.exception("%s 失败；已删除的元数据不回滚", label)

    def _remove_document_files(self, kb_id: str, doc_id: str) -> int:
        """删除某文档的原始文件（命名规则 {doc_id}_{filename}）。"""
        target_dir = self._upload_dir / kb_id
        if not target_dir.is_dir():
            return 0
        removed = 0
        for path in target_dir.glob(f"{doc_id}_*"):
            if path.is_file():
                path.unlink()
                removed += 1
        return removed

    def _remove_tree(self, target: Path) -> None:
        """删除知识库的上传目录。

        安全护栏：只允许删除 upload_dir 之下的目录，避免 kb_id 被构造成
        `../../` 之类时误删项目其他内容。
        """
        resolved = target.resolve()
        if resolved == self._upload_dir or self._upload_dir not in resolved.parents:
            raise ValueError(f"拒绝删除 upload_dir 之外的路径：{resolved}")
        if resolved.is_dir():
            shutil.rmtree(resolved)
