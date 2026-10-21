"""ORM 模型：SQLite 中的表结构。

三张表，刻意保持极简——检索相关的数据不在关系库里。

    knowledge_bases   知识库
    documents         文档元数据（外键指向 knowledge_bases）
    users             账号（登录用）

实现说明：
    - 主键用 VARCHAR(36) 存 UUID 字符串，与 Milvus 中的 kb_id / doc_id 对齐，
      这样跨存储清理时不需要额外的映射表。
    - `created_at` 存 **ISO8601 UTC 字符串**而非 DATETIME：
      SQLite 没有原生时间类型，字符串可直接排序、可读、无时区歧义。
    - `knowledge_bases` 上**没有** doc_count 列：文档数是派生值，
      用实时 COUNT 算，避免冗余字段与 documents 表不一致。
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models import DOC_STATUS_PROCESSING

from .database import Base


def utc_now_iso() -> str:
    """当前 UTC 时间的 ISO8601 字符串（秒级精度）。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class KnowledgeBaseRow(Base):
    """知识库表。

    关于 `owner_id`（多租户隔离的落点）：
        可空是**刻意**的——它是后加的列，老库里已有的行没有归属人。
        空串/None 统一解释为「无主知识库」：
          - 普通用户看不到、也改不动它（`owner_id = <uid>` 的查询天然排除掉）；
          - 只有明确传 `owner_id=None` 的运维路径（如 scripts/ 下的清理脚本）
            才能触及。
        没有做成 NOT NULL + 外键，是因为 SQLite 给已有表加 NOT NULL 外键列
        必须整表重建（`ALTER TABLE` 的限制），而重建会牵动 documents 的外键。
        代价是"归属不能缺失"这条约束由服务层保证——建库时必定写入当前用户。
    """

    __tablename__ = "knowledge_bases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # 归属账号。空串 = 历史遗留的无主数据（见类文档）。
    owner_id: Mapped[str] = mapped_column(String(36), nullable=False, default="")
    created_at: Mapped[str] = mapped_column(String(32), nullable=False, default=utc_now_iso)

    __table_args__ = (
        # 列表查询恒按 owner_id 过滤，是全站最高频的查询之一
        Index("ix_knowledge_bases_owner_id", "owner_id"),
    )


class DocumentRow(Base):
    """文档元数据表。

    注意：片段正文与向量**不在此表**。
    chunk_count 只是计数，正文的权威副本在 uploads 原始文件与 Milvus。
    """

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kb_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    # 正文的 SHA-256（十六进制，64 字符）。用途只有一个：**阻止同一份正文
    # 在同一个库里入库两次**。此前实测同一份 txt 攒到过 4 条 ready 记录，
    # 向量分逐位相同、在 top_k 里互相挤占名额——那是"混合检索召回异常"的根源。
    # 空串表示"没算过"（历史数据、或调用方没传），不参与唯一约束。
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DOC_STATUS_PROCESSING
    )
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[str] = mapped_column(String(32), nullable=False, default=utc_now_iso)

    __table_args__ = (
        # 列表查询与删除都按 kb_id 过滤，建索引
        Index("ix_documents_kb_id", "kb_id"),
        # 部分唯一索引：只约束"算过 hash"的行。
        # 用 WHERE content_hash != '' 而不是普通唯一索引，是因为历史数据与
        # 不传 hash 的调用方都会写入空串——那些行必须允许多条存在。
        # **并发也靠它兜底**：两个请求同时上传同一份文件时，应用层的查重
        # 会双双扑空，最终由这条索引让第二个插入失败（IntegrityError），
        # 仓储把它翻译成 409 并回传已有 doc_id。
        Index(
            "ux_documents_kb_content_hash",
            "kb_id",
            "content_hash",
            unique=True,
            sqlite_where=text("content_hash != ''"),
        ),
    )


class UserRow(Base):
    """账号表。

    约束说明（都是有意为之）：
      - `username` 带 **UNIQUE**：唯一性必须由数据库保证。靠"先查再写"挡不住
        并发注册（两个请求同时查到"不存在"，然后都插进去）。
        写入前统一小写规范化，所以 `Admin` 与 `admin` 是同一个账号。
      - `password_hash` 只存哈希，永不存明文，也永不返回给前端。
      - `is_active` 用停用代替删号：删号会让历史数据失去责任人，
        而"停用"是可逆的、可追溯的。
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False, default=utc_now_iso)
