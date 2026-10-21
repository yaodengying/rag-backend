# -*- coding: utf-8 -*-
"""会话存储的 Redis 实现（实现 domain.ports.ConversationStorePort）。

键设计（全部带 `rag:` 前缀，与同一实例上其它业务互不干扰）：
    rag:conv:{id}        hash   —— 会话元数据（title / kb_id / 计数 / 时间戳）
    rag:conv:{id}:msgs   list   —— 消息序列，每条一个 JSON（按时间正序追加）
    rag:conv:index       zset   —— 全部会话 ID → score=最近更新时间戳（列表页排序）

为什么用 zset 而不是 SCAN：
    列表页要「按最近更新倒序」，zset 天然有序；SCAN 需要全量遍历后再排序，
    会话数一多就是无谓的开销。

TTL 语义：
    「会话过期」是**整组键一起过期**。每次 append_turn 用 pipeline 对三个键
    统一 refresh TTL——续期必须三键同步，只续 meta 会出现「列表里还有、
    点进去消息没了」的半截会话。

错误语义（只在实现层出现一次，见 Port 的约定）：
    - Redis 连不上 / 命令失败 → UpstreamError（存储是上游基础设施）
    - 会话不存在 / 已过期     → NotFoundError
    - 入参不合法              → ValidationError
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from app.core.exceptions import NotFoundError, UpstreamError, ValidationError
from app.domain.models import Citation, Conversation, ConversationSummary, StoredMessage

if TYPE_CHECKING:
    from redis import Redis

logger = logging.getLogger(__name__)

# 时间格式与前端历史页「按前 10 位（YYYY-MM-DD）分组」的口径绑死，是契约不是巧合
_TIME_FMT = "%Y-%m-%d %H:%M:%S"

# 会话标题的最大长度（与前端旧版 deriveTitle 的口径一致：24 字 + 省略号）
_TITLE_MAX = 24

# zset 索引键（不带会话 ID）
_INDEX_KEY = "rag:conv:index"

# 消息角色的合法取值（写入侧把关，读取侧不猜）
_ROLES = ("user", "assistant")


def _now() -> str:
    return time.strftime(_TIME_FMT)


def _to_str(raw: Any) -> str:
    """Redis 返回的字段统一转 str。

    decode_responses=False 时拿到的是 bytes，None 表示字段缺失；
    两处（列表页与详情页）都要做这件事，收口成一个函数免得口径走偏。
    """
    if raw is None:
        return ""
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _derive_title(first_question: str) -> str:
    """用第一条用户提问做会话标题（折叠空白、截断到 24 字）。"""
    one_line = " ".join((first_question or "").split())
    if not one_line:
        return "新会话"
    return one_line if len(one_line) <= _TITLE_MAX else f"{one_line[:_TITLE_MAX]}…"


def _citation_to_dict(citation: Citation) -> dict[str, Any]:
    return {
        "doc_id": citation.doc_id,
        "filename": citation.filename,
        "page": citation.page,
        "snippet": citation.snippet,
    }


def _citation_from_dict(raw: dict[str, Any]) -> Citation:
    return Citation(
        doc_id=str(raw.get("doc_id") or ""),
        filename=str(raw.get("filename") or ""),
        page=raw.get("page") if isinstance(raw.get("page"), str) else None,
        snippet=str(raw.get("snippet") or ""),
    )


def _message_to_json(message: StoredMessage) -> str:
    if message.role not in _ROLES:
        raise ValidationError(f"消息 role 非法：{message.role!r}（允许 {_ROLES}）")
    payload = {
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at,
        "citations": [_citation_to_dict(c) for c in message.citations],
    }
    return json.dumps(payload, ensure_ascii=False)


def _message_from_json(raw: str) -> StoredMessage:
    data = json.loads(raw)
    return StoredMessage(
        role=str(data.get("role") or "user"),
        content=str(data.get("content") or ""),
        created_at=str(data.get("created_at") or ""),
        citations=[_citation_from_dict(c) for c in data.get("citations") or [] if isinstance(c, dict)],
    )


class RedisConversationStore:
    """ConversationStorePort 的 Redis 实现（同步客户端，阻塞调用丢线程池）。"""

    def __init__(self, client: "Redis", ttl_seconds: int) -> None:
        self._client = client
        if ttl_seconds <= 0:
            raise ValidationError(f"会话 TTL 必须为正整数 — 当前 {ttl_seconds}")
        self._ttl = ttl_seconds

    # ---------- 键 ----------

    @staticmethod
    def _meta_key(conversation_id: str) -> str:
        return f"rag:conv:{conversation_id}"

    @staticmethod
    def _msgs_key(conversation_id: str) -> str:
        return f"rag:conv:{conversation_id}:msgs"

    # ---------- 错误包装 ----------

    def _run(self, description: str, fn):
        """把 redis 异常统一包成 UpstreamError——错误语义只在实现层出现一次。"""
        try:
            return fn()
        except NotFoundError:
            raise
        except Exception as exc:  # redis.exceptions.RedisError 及连接层异常
            raise UpstreamError(
                f"会话存储（Redis）不可用：{description}失败 — {exc}",
                detail={"op": description},
            ) from exc

    # ---------- Port 实现 ----------

    def append_turn(
        self,
        conversation_id: str,
        kb_id: str,
        user_message: StoredMessage,
        assistant_message: StoredMessage,
        user_id: str = "",
    ) -> ConversationSummary:
        """追加一轮问答。

        Args:
            user_id: 会话归属账号。**首轮写入时定下，之后不变**——
                续聊时即使传了别的值也不会覆盖（否则一次越权请求就能
                把别人的会话"过继"给自己）。空串表示历史遗留的无主会话，
                只有明确传空串的运维路径能读到。
        """
        conversation_id = (conversation_id or "").strip()
        if not conversation_id:
            raise ValidationError("会话 ID 不能为空")

        for message in (user_message, assistant_message):
            if not (message.content or "").strip():
                raise ValidationError("消息正文不能为空")

        def _do() -> ConversationSummary:
            meta_key = self._meta_key(conversation_id)
            msgs_key = self._msgs_key(conversation_id)
            now = _now()
            created = self._client.hget(meta_key, "created_at")
            created_at = created.decode() if isinstance(created, bytes) else (created or now)
            # 标题只取第一条用户提问：会话中途换问题不改标题（与旧版前端口径一致）
            existing_title = self._client.hget(meta_key, "title")
            title = (
                existing_title.decode()
                if isinstance(existing_title, bytes) and existing_title
                else _derive_title(user_message.content)
            )
            # 归属只在首次写入时确定。用 HSETNX 语义（先读后写、只在空缺时填）
            # 而不是直接 hset：续聊可能是并发请求，后来的那个不该改归属。
            existing_owner = _to_str(self._client.hget(meta_key, "user_id"))

            old_count = int(self._client.hget(meta_key, "message_count") or 0)
            old_citations = int(self._client.hget(meta_key, "citation_count") or 0)
            new_count = old_count + 2
            new_citations = old_citations + len(assistant_message.citations)
            score = time.time()

            pipe = self._client.pipeline()
            pipe.hset(
                meta_key,
                mapping={
                    "id": conversation_id,
                    "kb_id": kb_id or "",
                    "title": title,
                    "created_at": created_at,
                    "updated_at": now,
                    "message_count": new_count,
                    "citation_count": new_citations,
                },
            )
            # 归属：已有值就保留，空缺才填。单独一条 HSETNX 而不是塞进上面的
            # mapping——mapping 里的 hset 会无条件覆盖，续聊时传入的 user_id
            # 只要有一次不对，整个会话就被改姓了。
            if not existing_owner and user_id:
                pipe.hsetnx(meta_key, "user_id", user_id)
            pipe.rpush(msgs_key, _message_to_json(user_message))
            pipe.rpush(msgs_key, _message_to_json(assistant_message))
            pipe.zadd(_INDEX_KEY, {conversation_id: score})
            pipe.expire(meta_key, self._ttl)
            pipe.expire(msgs_key, self._ttl)
            # 索引也续期：zset 是共用键，无法按成员单独 TTL，
            # 只能整键续期（每次写入都往后推）。代价是它永远不过期，
            # 但成员会在 meta 消失后由 list_conversations 摘除，
            # 所以"永不过期"不等于"永远变脏"。
            # 不设 expire 的话，索引键从第一次写入起就永久驻留：
            # 即使所有会话都过期，它仍占着内存且每次列表页都要全量扫描。
            pipe.expire(_INDEX_KEY, self._ttl)
            pipe.execute()

            return ConversationSummary(
                id=conversation_id,
                title=title,
                kb_id=kb_id or "",
                created_at=created_at,
                updated_at=now,
                message_count=new_count,
                citation_count=new_citations,
            )
        return self._run("追加会话消息", _do)

    def list_conversations(
        self,
        kb_id: str | None = None,
        keyword: str | None = None,
        user_id: str | None = None,
    ) -> list[ConversationSummary]:
        """列出会话。

        Args:
            user_id: 会话归属过滤。**None 表示不按归属过滤**（仅供运维脚本）；
                HTTP 层永远传当前账号，因为会话正文就是用户问过的内容。
                空串只匹配"无主"的历史会话。
        """
        def _do() -> list[ConversationSummary]:
            # zrevrange：score（最近更新）倒序
            ids = self._client.zrevrange(_INDEX_KEY, 0, -1)
            result: list[ConversationSummary] = []
            kw = (keyword or "").strip().lower()
            stale: list[str] = []

            # 一次 pipeline 取回全部 meta，避免「N 个会话 = N 次往返」。
            # 之前逐条 HGETALL，会话上千时会明显拖慢列表页（实测 15 条没事，
            # 但这是 O(n) 次 RTT 的结构性开销，不是量小就无害）。
            keys = [raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id) for raw_id in ids]
            metas: list[dict] = []
            if keys:
                pipe = self._client.pipeline(transaction=False)
                for cid in keys:
                    pipe.hgetall(self._meta_key(cid))
                metas = pipe.execute()

            for cid, meta in zip(keys, metas):
                if not meta:
                    # 索引有、元数据没了：过期窗口内的半截状态，清掉索引即可
                    stale.append(cid)
                    continue
                item_kb = _to_str(meta.get(b"kb_id"))
                if kb_id and item_kb != kb_id:
                    continue
                # 归属过滤发生在 title 匹配之前：先按归属判掉，
                # 避免用别人的会话标题去验证"某个关键词是否命中"。
                if user_id is not None and _to_str(meta.get(b"user_id")) != user_id:
                    continue
                title = _to_str(meta.get(b"title"))
                if kw and kw not in title.lower():
                    continue
                result.append(
                    ConversationSummary(
                        id=cid,
                        kb_id=item_kb,
                        title=title,
                        created_at=_to_str(meta.get(b"created_at")),
                        updated_at=_to_str(meta.get(b"updated_at")),
                        message_count=int(meta.get(b"message_count") or 0),
                        citation_count=int(meta.get(b"citation_count") or 0),
                    )
                )

            if stale:
                # 批量摘除，而不是边遍历边 zrem：一次往返收尾
                self._client.zrem(_INDEX_KEY, *stale)
            return result

        return self._run("列出会话", _do)

    def get_conversation(
        self, conversation_id: str, user_id: str | None = None
    ) -> Conversation:
        """读取会话（含全部消息）。

        Args:
            user_id: 归属过滤。None 表示不校验（仅运维脚本）；
                HTTP 层永远传当前账号。归属不符与"不存在"抛**同一个**
                NotFoundError——拿到 403 就等于确认了这个 conversation_id 存在。
        """
        conversation_id = (conversation_id or "").strip()
        if not conversation_id:
            raise ValidationError("会话 ID 不能为空")

        def _do() -> Conversation:
            meta_key = self._meta_key(conversation_id)
            meta = self._client.hgetall(meta_key)
            if not meta:
                raise NotFoundError(f"会话不存在或已过期：{conversation_id}")
            if user_id is not None and _to_str(meta.get(b"user_id")) != user_id:
                raise NotFoundError(f"会话不存在或已过期：{conversation_id}")
            raw_msgs = self._client.lrange(self._msgs_key(conversation_id), 0, -1)
            return Conversation(
                id=conversation_id,
                kb_id=(meta.get(b"kb_id") or b"").decode(),
                title=(meta.get(b"title") or b"").decode(),
                created_at=(meta.get(b"created_at") or b"").decode(),
                updated_at=(meta.get(b"updated_at") or b"").decode(),
                message_count=int(meta.get(b"message_count") or 0),
                citation_count=int(meta.get(b"citation_count") or 0),
                messages=[_message_from_json(m) for m in raw_msgs],
            )

        return self._run("读取会话", _do)

    def conversation_exists(self, conversation_id: str) -> bool:
        """会话是否存在。**刻意不做归属校验**，调用方必须清楚这一点。

        为什么与 `get_conversation` 区别对待：
            后者会把会话内容（问题、回答、引用来源）读出来，越权读等于
            数据泄漏，所以必须校验归属。而本函数只回答"这个键还在不在"，
            不返回任何内容——它的用途是"续聊前确认会话还没过期"。

        风险与边界：
            如果将来把它暴露成 HTTP 接口（比如 `HEAD /conversations/{id}`），
            它就变成了**会话 ID 存在性探针**：知道 ID 的人能试出别人的会话
            是否还在。要暴露时必须先加归属校验，或改成不区分"不存在"与
            "不是你的"的 404。当前它只在服务层内部使用。
        """
        conversation_id = (conversation_id or "").strip()
        if not conversation_id:
            return False

        def _do() -> bool:
            return bool(self._client.exists(self._meta_key(conversation_id)))

        return self._run("检查会话存在", _do)

    def delete_conversation(self, conversation_id: str, user_id: str | None = None) -> None:
        """删除会话。

        Args:
            user_id: 归属过滤。None 表示不校验（运维与"删知识库"级联清理用）；
                HTTP 层传当前账号，归属不符抛 NotFoundError。

        为什么级联清理可以不传：那条路径的调用方是删知识库，**知识库本身
        已经做过归属校验**——能删这个库就已经证明了所有权，再按会话归属
        过滤反而会漏删（历史遗留的无主会话会留下来变成孤儿）。
        """
        conversation_id = (conversation_id or "").strip()
        if not conversation_id:
            raise ValidationError("会话 ID 不能为空")

        def _do() -> None:
            if user_id is not None:
                meta = self._client.hgetall(self._meta_key(conversation_id))
                # 不存在与不属于同一个人都抛同一句话，避免枚举
                if not meta or _to_str(meta.get(b"user_id")) != user_id:
                    raise NotFoundError(f"会话不存在或已过期：{conversation_id}")
            pipe = self._client.pipeline()
            pipe.delete(self._meta_key(conversation_id))
            pipe.delete(self._msgs_key(conversation_id))
            pipe.zrem(_INDEX_KEY, conversation_id)
            pipe.execute()

        self._run("删除会话", _do)

    def delete_by_kb(self, kb_id: str) -> int:
        """删除某知识库下的全部会话。

        实现取舍（遍历索引匹配 kb_id，不建反向索引）：
            反向索引（rag:conv:kb:{kb_id} 集合）能让删除变成 O(1)，
            但每次 append_turn 都要多写一个 set 并保证它与 zset 同步，
            多一处「必须一起维护」的状态就多一处能不一致的地方。
            会话总量以万计之前，遍历 + pipeline 批量 HGETALL 的代价可以接受，
            换来的是 Redis 里只有一份真相（meta 的 kb_id 字段）。

        分三个动作，全部走 pipeline 批量化：
            1. 从 zset 按 kb_id 挑出目标 ID（一次 pipeline 拉全部 meta）
            2. 批量删 meta 与 msgs（每会话两条 DEL，压进一个 pipeline）
            3. 从 zset 批量摘除
        中途失败由 _run 统一包成 UpstreamError；调用方（删知识库）是
        "尽力而为"语义，失败不回滚已删的知识库，只记日志。
        """
        kb_id = (kb_id or "").strip()
        if not kb_id:
            # 空 kb_id 会匹配到所有 kb_id 为空的会话，属于危险调用
            raise ValidationError("kb_id 不能为空")

        def _do() -> int:
            raw_ids = self._client.zrevrange(_INDEX_KEY, 0, -1)
            if not raw_ids:
                return 0
            ids = [r.decode() if isinstance(r, bytes) else str(r) for r in raw_ids]

            pipe = self._client.pipeline(transaction=False)
            for cid in ids:
                pipe.hget(self._meta_key(cid), "kb_id")
            owners = pipe.execute()

            targets = [
                cid
                for cid, owner in zip(ids, owners)
                if _to_str(owner) == kb_id
            ]
            if not targets:
                return 0

            pipe = self._client.pipeline(transaction=False)
            for cid in targets:
                pipe.delete(self._meta_key(cid))
                pipe.delete(self._msgs_key(cid))
            pipe.zrem(_INDEX_KEY, *targets)
            pipe.execute()

            logger.info(
                "已清理知识库关联会话：kb_id=%s，删除 %s 个会话（扫描 %s 个）",
                kb_id,
                len(targets),
                len(ids),
            )
            return len(targets)

        return self._run("按知识库删除会话", _do)
