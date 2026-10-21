"""数据库连接与会话管理。

设计：
    - Database 封装 Engine 与 sessionmaker，由 container 装配为单例。
    - get_session 作为上下文管理器，负责 commit / rollback / close，
      业务代码不必手写 try-except-finally。
    - SQLite 在多线程下需 check_same_thread=False，
      但 FastAPI 的同步依赖会在线程池中执行，务必确保每次请求用独立 session。

实现说明：
    - 连接串用**绝对路径**拼装（`sqlite:///H:/.../data/rag.db`），
      避免同一个库因为 cwd 不同而落成两个文件。
    - 打开 `foreign_keys=ON`：SQLite 默认**不**执行外键约束，
      不显式打开的话 `documents.kb_id` 上的 CASCADE 形同虚设。
    - 打开 WAL：读写并发明显更好，代价是目录里多出 `-wal` / `-shm` 文件。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


class Database:
    """SQLite 数据库封装（Engine + Session 工厂）。"""

    def __init__(self, sqlite_path: str) -> None:
        """建立引擎与会话工厂。

        Args:
            sqlite_path: SQLite 文件路径，如 ./data/rag.db
        """
        db_file = Path(sqlite_path).expanduser().resolve()
        # 目录不存在时 sqlite 无法建库文件，这里主动补齐
        db_file.parent.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = db_file.as_posix()

        self.engine: Engine = create_engine(
            f"sqlite:///{self.sqlite_path}",
            connect_args={"check_same_thread": False},
            future=True,
        )
        self._install_pragmas()
        self._session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            autoflush=False,
            future=True,
        )
        logger.debug("SQLite 已就绪：%s", self.sqlite_path)

    def _install_pragmas(self) -> None:
        """为每条新连接设置 PRAGMA。

        注意：journal_mode 是**持久化**设置（写在库文件头上），
        foreign_keys / synchronous / busy_timeout 是**连接级**设置，
        故每次连接都要重设。

        `busy_timeout` 是为多人并发准备的：WAL 只允许「多读一写」，
        两个写事务重叠时后到的那个默认**立刻**抛 `database is locked`
        （不是等待）。5 秒的等待足够覆盖本项目所有写事务（最长的
        `mark_document_ready` 也是毫秒级），既不会无限挂住请求，
        也不会把一次正常的并发上传变成 500。
        """

        @event.listens_for(self.engine, "connect")
        def _on_connect(dbapi_connection, _connection_record) -> None:  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA busy_timeout=5000")
            finally:
                cursor.close()

    def init_schema(self) -> None:
        """建表（幂等）。应用启动时调用，无需迁移工具。

        顺序不能反：**先补列，再 create_all**。
        `create_all` 不会给已存在的表加列，但它会尝试建索引——
        老库里没有 `content_hash` 列时直接建那个唯一索引会报
        "no such column"，启动就挂在这里。
        """
        # 必须先导入 models，否则 Base.metadata 里没有表定义，
        # create_all 会「成功」地什么都不建。
        from . import models  # noqa: F401

        self._migrate_documents()
        self._migrate_knowledge_bases()
        Base.metadata.create_all(self.engine)

    def _migrate_knowledge_bases(self) -> None:
        """老库补 `owner_id` 列与归属索引（多租户隔离的前置条件）。

        为什么不设 NOT NULL：
            `ALTER TABLE ... ADD COLUMN ... NOT NULL` 在 SQLite 上要求给出
            DEFAULT，而给了 DEFAULT ''（无主）与给 NULL 在语义上没差别，
            真正拦住"必须写 owner"的地方在服务层（建库必定写入当前用户）。
            用 NOT NULL DEFAULT '' 是为了让 `owner_id = ?` 这类查询不必担心
            NULL 的三值逻辑（`NULL = ?` 永远不成立，会让"自己的库"也查不到，
            那是最典型的静默失效）。

        为什么必须显式建索引：
            同 `_migrate_documents` 的坑②——`create_all` 不会给已存在的表补建索引。
            而 `ix_knowledge_bases_owner_id` 是列表查询的唯一过滤条件，
            缺了它在库变大后表现为"知识库列表越来越慢"，却不会有任何报错。
        """
        with self.engine.begin() as conn:
            columns = {
                row[1]
                for row in conn.exec_driver_sql("PRAGMA table_info(knowledge_bases)")
            }
            if not columns:  # 表还不存在，create_all 会连同索引一起建
                return
            if "owner_id" not in columns:
                conn.exec_driver_sql(
                    "ALTER TABLE knowledge_bases ADD COLUMN owner_id VARCHAR(36) "
                    "NOT NULL DEFAULT ''"
                )
                logger.info(
                    "已为 knowledge_bases 表补充 owner_id 列（多租户隔离）；"
                    "存量知识库保持「无主」，仅运维脚本可访问"
                )

            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_knowledge_bases_owner_id "
                "ON knowledge_bases (owner_id)"
            )

    def _migrate_documents(self) -> None:
        """老库补 `content_hash` 列与去重唯一索引（轻量迁移，只做加法）。

        项目刻意不引 Alembic：只三张表、只加不删，一个 20 行的探测
        足够；引迁移框架会让"启动即能跑"变成"先 alembic upgrade"。
        **只加列不删列**——删列是不可逆操作，真要删就写显式脚本。

        两个坑都在这里（实测踩过）：
          ① `create_all` **不会**给已存在的表加列，所以 ALTER 必须自己做；
          ② `create_all` **也不会**给已存在的表补建索引（它只在建表那次
             建索引），所以那条唯一索引必须显式 `CREATE ... IF NOT EXISTS`。
             只加列不建索引的后果很隐蔽：应用层的查重看起来在工作，
             并发时却没有任何兜底，重复入库照样发生。
        """
        with self.engine.begin() as conn:
            columns = {
                row[1] for row in conn.exec_driver_sql("PRAGMA table_info(documents)")
            }
            if not columns:  # 表还不存在，create_all 会连同索引一起建
                return
            if "content_hash" not in columns:
                conn.exec_driver_sql(
                    "ALTER TABLE documents ADD COLUMN content_hash VARCHAR(64) "
                    "NOT NULL DEFAULT ''"
                )
                logger.info("已为 documents 表补充 content_hash 列（用于阻止重复入库）")

            try:
                conn.exec_driver_sql(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_documents_kb_content_hash "
                    "ON documents (kb_id, content_hash) WHERE content_hash != ''"
                )
            except Exception:  # noqa: BLE001
                # 老库里若已有重复 hash（例如手工导入过），建索引会失败。
                # 不因为"数据脏"让服务起不来：降级为无索引，并在日志里说清
                # 后果——去重仍然生效（应用层查），但并发窗口不再有数据库兜底。
                logger.exception(
                    "创建文档去重唯一索引失败（库内已有重复 content_hash）；"
                    "去重仍由应用层保证，但并发去重失去数据库兜底"
                )

    @contextmanager
    def session(self) -> Iterator[Session]:
        """提供自动提交的事务型会话。

        Yields:
            Session: 数据库会话；正常退出自动 commit，异常自动 rollback。
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        """释放连接池。应用关闭时调用。"""
        self.engine.dispose()


def get_session() -> Iterator[Session]:
    """FastAPI 依赖：每请求一个会话。

    容器在 app.core.container 中，此处延迟导入以避免
    「container → db → container」的循环依赖。
    """
    from app.core.container import get_database

    with get_database().session() as session:
        yield session
