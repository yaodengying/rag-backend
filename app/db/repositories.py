"""仓储实现：元数据的读写。

两个仓储：
    SqlAlchemyChunkRepository —— 知识库 / 文档元数据，实现 ChunkRepositoryPort
    SqlAlchemyUserRepository  —— 账号，实现 UserRepositoryPort

仓储层的价值：把「怎么存」与「业务怎么用」隔开。
上层 service 只调 list_documents / get_user_by_username 这类语义化方法，
不写 SQL，也不碰 Session。

实现说明：
    - 每个方法用 Database.session() 上下文，事务边界即方法边界。
    - SQLite 只存元数据；片段正文与向量不经过本模块。
    - `doc_count` 用实时 COUNT 计算，不落冗余列，避免与 documents 表不一致。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError

from app.core.exceptions import KB_NOT_FOUND_MESSAGE, ConflictError, NotFoundError
from app.db.database import Database
from app.db.models import DocumentRow, KnowledgeBaseRow, UserRow
from app.domain.models import (
    DOC_STATUS_FAILED,
    DOC_STATUS_PROCESSING,
    DOC_STATUS_READY,
    Document,
    User,
)


def _to_dt(value: str | None) -> datetime | None:
    """ISO8601 字符串 → datetime；空值或格式非法时返回 None。"""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _kb_to_dict(row: KnowledgeBaseRow, doc_count: int) -> dict:
    """ORM 行 → 对外字典（键与 schemas.KnowledgeBaseOut 对齐）。"""
    return {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "doc_count": int(doc_count or 0),
        "created_at": row.created_at,
    }


def _doc_to_domain(row: DocumentRow) -> Document:
    """ORM 行 → 领域模型。"""
    return Document(
        id=row.id,
        kb_id=row.kb_id,
        filename=row.filename,
        status=row.status,
        chunk_count=row.chunk_count,
        error=row.error,
        created_at=_to_dt(row.created_at),
    )


def _user_to_domain(row: UserRow) -> User:
    """ORM 行 → 领域模型。"""
    return User(
        id=row.id,
        username=row.username,
        password_hash=row.password_hash,
        display_name=row.display_name or "",
        is_active=bool(row.is_active),
        created_at=_to_dt(row.created_at),
    )


class SqlAlchemyChunkRepository:
    """基于 SQLAlchemy 的元数据仓储。

    实现 ChunkRepositoryPort 协议。
    """

    def __init__(self, database: Database) -> None:
        """注入 Database（提供 session 上下文）。

        Args:
            database: app.db.database.Database 实例
        """
        self._db = database

    # ---------- 知识库 ----------

    def list_knowledge_bases(self, owner_id: str) -> list[dict]:
        """列出某账号名下的知识库（含文档数）。

        `owner_id` **没有默认值**，是刻意的：给它一个 None 表示"全部"的
        重载，就等于留了一个"忘了传就全库返回"的洞。多租户隔离里最难查的
        漏洞从来不是"判断写错"，而是"某个新加的方法忘了加过滤条件"——
        强制参数让漏传在**导入/调用时**就报错，而不是在上线后静默泄漏。

        Args:
            owner_id: 当前登录账号 ID。空串表示只匹配「无主」历史数据，
                普通用户传空串会得到空列表（不是全量）——这一点由
                `owner_id == ""` 的比较语义保证，见 _owner_filter。
        """
        counts = (
            select(DocumentRow.kb_id, func.count(DocumentRow.id).label("cnt"))
            .group_by(DocumentRow.kb_id)
            .subquery()
        )
        stmt = (
            select(KnowledgeBaseRow, func.coalesce(counts.c.cnt, 0))
            .outerjoin(counts, counts.c.kb_id == KnowledgeBaseRow.id)
            .where(KnowledgeBaseRow.owner_id == owner_id)
            .order_by(KnowledgeBaseRow.created_at)
        )
        with self._db.session() as session:
            rows = session.execute(stmt).all()
            return [_kb_to_dict(row, cnt) for row, cnt in rows]

    def get_knowledge_base(self, kb_id: str, owner_id: str) -> dict | None:
        """按 ID 取单个知识库；不存在**或不属于该账号**时返回 None。

        归属不符返回 None 而不是抛异常，是为了让上层的
        `get_knowledge_base_or_raise` 能把"不存在"与"不是你的"收敛成
        同一个 404——两者的响应必须逐字相同，否则就成了一个
        「拿随机 ID 试，看返回 404 还是 403」的知识库枚举接口。
        """
        counts = (
            select(DocumentRow.kb_id, func.count(DocumentRow.id).label("cnt"))
            .group_by(DocumentRow.kb_id)
            .subquery()
        )
        stmt = (
            select(KnowledgeBaseRow, func.coalesce(counts.c.cnt, 0))
            .outerjoin(counts, counts.c.kb_id == KnowledgeBaseRow.id)
            .where(
                KnowledgeBaseRow.id == kb_id,
                KnowledgeBaseRow.owner_id == owner_id,
            )
        )
        with self._db.session() as session:
            row = session.execute(stmt).first()
            if row is None:
                return None
            return _kb_to_dict(row[0], row[1])

    def create_knowledge_base(self, name: str, description: str, owner_id: str) -> dict:
        """新建知识库并返回其字典表示（归属为 owner_id）。"""
        row = KnowledgeBaseRow(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            owner_id=owner_id,
        )
        with self._db.session() as session:
            session.add(row)
        return _kb_to_dict(row, 0)

    def update_knowledge_base(
        self, kb_id: str, name: str, description: str, owner_id: str
    ) -> dict:
        """更新知识库名称与描述（仅限本人名下）。

        Raises:
            NotFoundError: kb_id 不存在**或不属于该账号**（同一句话，防枚举）
        """
        with self._db.session() as session:
            row = session.scalar(
                select(KnowledgeBaseRow).where(
                    KnowledgeBaseRow.id == kb_id,
                    KnowledgeBaseRow.owner_id == owner_id,
                )
            )
            if row is None:
                raise NotFoundError(KB_NOT_FOUND_MESSAGE)
            row.name = name
            row.description = description
            session.flush()
            doc_count = session.scalar(
                select(func.count(DocumentRow.id)).where(DocumentRow.kb_id == kb_id)
            )
            return _kb_to_dict(row, doc_count or 0)

    def delete_knowledge_base(self, kb_id: str, owner_id: str) -> bool:
        """删除知识库（级联删元数据），仅限本人名下。

        注意：Milvus 中的向量需由 service 层同步清理。

        Returns:
            True 表示确实删掉了一行；False 表示不存在**或不属于该账号**。
            「不是你的」返回 False 而不是 True，是为了让调用方无法通过
            返回值区分两种情况——它本来就不该知道别人的库是否存在。
        """
        with self._db.session() as session:
            # 先确认归属再删文档：反过来的话，即使最后知识库那行没删掉，
            # 别人的文档元数据也已经没了（不可逆）。
            owned = session.scalar(
                select(KnowledgeBaseRow.id).where(
                    KnowledgeBaseRow.id == kb_id,
                    KnowledgeBaseRow.owner_id == owner_id,
                )
            )
            if owned is None:
                return False
            # 显式删除文档行，不依赖数据库级 CASCADE——
            # 将来若换 Postgres / 关闭外键 PRAGMA，行为仍一致。
            session.execute(delete(DocumentRow).where(DocumentRow.kb_id == kb_id))
            result = session.execute(
                delete(KnowledgeBaseRow).where(
                    KnowledgeBaseRow.id == kb_id,
                    KnowledgeBaseRow.owner_id == owner_id,
                )
            )
            return bool(result.rowcount)

    # ---------- 文档 ----------

    def list_documents(self, kb_id: str) -> list[Document]:
        """列出某知识库下的文档。"""
        stmt = (
            select(DocumentRow)
            .where(DocumentRow.kb_id == kb_id)
            .order_by(DocumentRow.created_at, DocumentRow.id)
        )
        with self._db.session() as session:
            return [_doc_to_domain(row) for row in session.scalars(stmt).all()]

    def get_document(self, doc_id: str) -> Document | None:
        """按 ID 取单个文档；不存在返回 None。

        重试链路要用它读回 `kb_id` 与 `filename` —— 原始文件的落盘路径由
        这三者拼出来（`uploads/{kb_id}/{doc_id}_{filename}`），
        所以**不能**只传 doc_id 就去磁盘上猜文件名。
        """
        with self._db.session() as session:
            row = session.get(DocumentRow, doc_id)
            return _doc_to_domain(row) if row is not None else None

    def find_document_by_hash(self, kb_id: str, content_hash: str) -> Document | None:
        """按正文 hash 找一个**已成功入库**的文档；没有则 None。

        只认 `status=ready`：失败的文档虽然也算"传过这份文件"，
        但用户重传的动机往往正是上一次失败——拦住它等于把人堵在门外。
        """
        if not content_hash:
            return None
        stmt = (
            select(DocumentRow)
            .where(
                DocumentRow.kb_id == kb_id,
                DocumentRow.content_hash == content_hash,
                DocumentRow.status == DOC_STATUS_READY,
            )
            .limit(1)
        )
        with self._db.session() as session:
            row = session.scalars(stmt).first()
            return _doc_to_domain(row) if row is not None else None

    def create_document(self, kb_id: str, filename: str, content_hash: str = "") -> Document:
        """登记文档，status=processing。

        Raises:
            NotFoundError: kb_id 不存在
            ConflictError: 同一知识库里已有相同正文 hash 的文档（并发兜底）
        """
        row = DocumentRow(
            id=str(uuid.uuid4()),
            kb_id=kb_id,
            filename=filename,
            content_hash=content_hash[:64],
            status=DOC_STATUS_PROCESSING,
            chunk_count=0,
            error="",
        )
        with self._db.session() as session:
            if session.get(KnowledgeBaseRow, kb_id) is None:
                raise NotFoundError(KB_NOT_FOUND_MESSAGE)
            session.add(row)
            try:
                session.flush()
            except IntegrityError as exc:
                # 走到这里说明两个请求几乎同时上传了同一份正文：
                # 应用层的"先查后插"双双扑空，只有数据库能给出一致性。
                # 回查已有记录的 id 一并返回，前端才能提示"已存在，无需重传"。
                session.rollback()
                existing = session.scalars(
                    select(DocumentRow).where(
                        DocumentRow.kb_id == kb_id,
                        DocumentRow.content_hash == content_hash[:64],
                    )
                ).first()
                raise ConflictError(
                    "该文件内容已在此知识库中（按正文哈希判定），无需重复上传",
                    detail={"doc_id": existing.id if existing else None},
                ) from exc
        return _doc_to_domain(row)

    def mark_document_ready(self, doc_id: str, chunk_count: int) -> None:
        """标记文档处理成功。"""
        with self._db.session() as session:
            session.execute(
                update(DocumentRow)
                .where(DocumentRow.id == doc_id)
                .values(status=DOC_STATUS_READY, chunk_count=chunk_count, error="")
            )

    def mark_document_failed(self, doc_id: str, error: str) -> None:
        """标记文档处理失败并记录原因。

        这一步很关键：否则会留下永远 processing 的僵尸记录。
        """
        with self._db.session() as session:
            session.execute(
                update(DocumentRow)
                .where(DocumentRow.id == doc_id)
                .values(status=DOC_STATUS_FAILED, chunk_count=0, error=error[:2000])
            )

    def mark_document_processing(self, doc_id: str) -> None:
        """把文档重新置为处理中（重试入口用）。

        必须同时**清空 error 与 chunk_count**：只改 status 会留下
        「status=processing 但 error 还写着上次的失败原因」这种自相矛盾的行，
        前端看到的就是「正在处理，附一条旧报错」。
        """
        with self._db.session() as session:
            session.execute(
                update(DocumentRow)
                .where(DocumentRow.id == doc_id)
                .values(status=DOC_STATUS_PROCESSING, chunk_count=0, error="")
            )

    def delete_document(self, doc_id: str) -> str | None:
        """删除文档元数据，返回所属 kb_id（不存在则 None）。"""
        with self._db.session() as session:
            row = session.get(DocumentRow, doc_id)
            if row is None:
                return None
            kb_id = row.kb_id
            session.delete(row)
            return kb_id


class SqlAlchemyUserRepository:
    """基于 SQLAlchemy 的账号仓储。

    实现 UserRepositoryPort 协议。
    与知识库仓储分开的理由见 domain/ports.py 的 UserRepositoryPort 文档。
    """

    def __init__(self, database: Database) -> None:
        """注入 Database（提供 session 上下文）。

        Args:
            database: app.db.database.Database 实例
        """
        self._db = database

    def create_user(
        self, username: str, password_hash: str, display_name: str = ""
    ) -> User:
        """新建账号。

        Raises:
            ConflictError: 登录名已被占用
        """
        row = UserRow(
            id=str(uuid.uuid4()),
            username=username,
            password_hash=password_hash,
            display_name=display_name,
            is_active=True,
        )
        try:
            with self._db.session() as session:
                session.add(row)
                session.flush()
        except IntegrityError as exc:
            # 唯一约束是并发下唯一可靠的判据——不靠"先查再写"
            raise ConflictError(f"登录名已被占用：{username}") from exc
        return _user_to_domain(row)

    def get_user_by_username(self, username: str) -> User | None:
        """按登录名取账号；不存在返回 None。"""
        with self._db.session() as session:
            row = session.scalar(select(UserRow).where(UserRow.username == username))
            return _user_to_domain(row) if row is not None else None

    def get_user_by_id(self, user_id: str) -> User | None:
        """按 ID 取账号；不存在返回 None。"""
        with self._db.session() as session:
            row = session.get(UserRow, user_id)
            return _user_to_domain(row) if row is not None else None

    def list_users(self) -> list[User]:
        """列出全部账号（按创建时间）。"""
        stmt = select(UserRow).order_by(UserRow.created_at, UserRow.id)
        with self._db.session() as session:
            return [_user_to_domain(row) for row in session.scalars(stmt).all()]

    def update_password_hash(self, user_id: str, password_hash: str) -> bool:
        """覆盖某账号的口令哈希，返回是否命中。"""
        with self._db.session() as session:
            result = session.execute(
                update(UserRow)
                .where(UserRow.id == user_id)
                .values(password_hash=password_hash)
            )
            return bool(result.rowcount)
