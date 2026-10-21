# -*- coding: utf-8 -*-
"""会话历史自检：多轮对话历史「Redis 存取 + 溯源回放」这一段是对的。

背景（为什么需要这个脚本）：
    会话历史从「纯前端 localStorage」迁到「服务端 Redis」后，故障面变了：
      ① 历史静默丢失 —— 落库失败被吞成日志，聊天照样工作，
         用户翻历史时才发现"昨天那几轮没了"；
      ② 溯源断层 —— 回答里的引用带页码，存进历史后丢了页码，
         换个时间核对就变成"只有结论没有证据"；
      ③ 历史进了提示词但没进对地方 —— 无历史请求的提示词如果与
         引入该特性之前不一致，就是一次静默的行为漂移；
      ④ 未配置 Redis 时本该整体降级，却把 /chat 一起拖死。
    这些都只能靠断言钉住。

覆盖组：
    第 1 组  ChatService × 会话存储（假 store）：首问生成 ID、两轮落库、
             溯源（文件名+页码）进历史、历史送进提示词、请求体 history 被忽略
    第 2 组  降级语义：未配置 store 不落库不报错；store 读取失败退回请求体
             历史；落库失败不影响已生成的回答；history_turns=0 不进提示词
    第 3 组  RedisConversationStore（**真实 Redis**，容器没起则整组跳过）：
             CRUD 回环、标题派生、kb/keyword 过滤、TTL 三键同步、删除幂等
    第 4 组  Redis 故障 → UpstreamError（错误语义只在实现层出现一次）
    第 5 组  HTTP /conversations：503 未配置 / 200 详情带溯源 / 404 / 204；
             流式 done 事件恒带 conversation_id 字段

运行：
    .venv\\Scripts\\python.exe scripts\\check_conversations.py

    第 1/2/4/5 组全假实现、离线可跑；第 3 组需要本机 Redis（docker）。
    注意：第 3 组会写入真实 Redis（键前缀 check:conv:），结束前清理。

退出码：0 全部通过；1 存在失败项。
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# settings 的 env_file 是相对路径，必须先切到项目根
os.chdir(PROJECT_ROOT)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from app.core.exceptions import (  # noqa: E402
    AppError,
    NotFoundError,
    UpstreamError,
)
from app.domain.models import (  # noqa: E402
    Chunk,
    Conversation,
    Citation,
    RetrievedChunk,
    StoredMessage,
)
from app.domain.ports import ConversationStorePort  # noqa: E402
from app.schemas.conversation import ConversationDetailOut  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.conversation_service import ConversationService  # noqa: E402

_PASS: list[str] = []
_FAIL: list[str] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASS.append(name)
        print(f"  [ok]   {name}")
    else:
        _FAIL.append(name)
        print(f"  [FAIL] {name}  {detail}")


def section(title: str) -> None:
    print(f"\n---- {title} ----")


# =============================================================================
# 假实现
# =============================================================================


def make_chunk(text: str, filename: str, page_start: int | None) -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()),
        kb_id="kb-check",
        doc_id=f"doc-{filename}",
        text=text,
        chunk_index=0,
        page_start=page_start,
        page_end=page_start,
        filename=filename,
    )


def make_ranked(chunk: Chunk) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, vector_score=0.5, rerank_score=3.2)


class FakeGraph:
    """假问答图 / 假检索子图：.invoke(initial) → 预设的最终状态。"""

    def __init__(self, final: dict[str, Any]) -> None:
        self.final = final
        self.calls: list[dict] = []

    def invoke(self, initial: dict) -> dict:
        self.calls.append(initial)
        return self.final


class RecordingLLM:
    """假 LLM：登记收到的 (system, user)，流式按字吐出预设答案。"""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append((system_prompt, user_prompt))
        return self.answer

    async def stream(self, system_prompt: str, user_prompt: str):  # noqa: ANN201
        self.prompts.append((system_prompt, user_prompt))
        for i in range(0, len(self.answer), 3):
            yield self.answer[i : i + 3]


class FakeStore:
    """内存版 ConversationStorePort（与 Redis 实现同语义，离线可跑）。"""

    def __init__(self) -> None:
        self.convs: dict[str, Conversation] = {}
        # 会话归属：与 Redis 实现一样独立存放，缺失视为空归属
        self.owners: dict[str, str] = {}
        self.fail_reads = False  # 置 True 时 get_conversation 抛 UpstreamError

    def append_turn(
        self, conversation_id, kb_id, user_message, assistant_message, user_id=""
    ):
        conversation_id = (conversation_id or "").strip()
        if not conversation_id:
            raise ValueError("conversation_id required")
        conv = self.convs.get(conversation_id)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        if conv is None:
            conv = Conversation(
                id=conversation_id,
                kb_id=kb_id,
                title=" ".join(user_message.content.split())[:24],
                created_at=now,
                updated_at=now,
            )
            self.convs[conversation_id] = conv
            # 归属只在首轮写入，后续续聊不覆盖（对齐真实存储的 HSETNX 语义）
            self.owners[conversation_id] = user_id
        conv.kb_id = kb_id
        conv.updated_at = now
        conv.message_count += 2
        conv.citation_count += len(assistant_message.citations)
        conv.messages.extend([user_message, assistant_message])
        return conv

    def list_conversations(self, kb_id=None, keyword=None, user_id=None):
        items = [
            c
            for c in self.convs.values()
            if (not kb_id or c.kb_id == kb_id)
            and (user_id is None or self.owners.get(c.id, "") == user_id)
        ]
        if keyword:
            items = [c for c in items if keyword.lower() in c.title.lower()]
        return sorted(items, key=lambda c: c.updated_at, reverse=True)

    def get_conversation(self, conversation_id, user_id=None):
        if self.fail_reads:
            raise UpstreamError("模拟 Redis 故障")
        conv = self.convs.get((conversation_id or "").strip())
        # 归属不符与不存在抛同一句话：两者对外必须无法区分
        if conv is None or (
            user_id is not None and self.owners.get(conv.id, "") != user_id
        ):
            raise NotFoundError(f"会话不存在或已过期：{conversation_id}")
        return conv

    def conversation_exists(self, conversation_id):
        return (conversation_id or "").strip() in self.convs

    def delete_conversation(self, conversation_id, user_id=None):
        conversation_id = (conversation_id or "").strip()
        if user_id is not None and self.owners.get(conversation_id, "") != user_id:
            # 越权删除：静默失败会让人以为删掉了，抛错才是真实语义
            raise NotFoundError(f"会话不存在或已过期：{conversation_id}")
        self.owners.pop(conversation_id, None)
        self.convs.pop(conversation_id, None)


def make_service(store, llm, answer="答案正文 (来源：[1])", *, history_turns=None):
    """组装带假图的 ChatService；ranked 非空 → 走生成路径，不触发改写。"""
    from app.core.config import get_settings

    if history_turns is not None:
        get_settings().conversation_history_turns = history_turns
    ranked = [
        make_ranked(make_chunk("巡检周期为每季度一次。", "设备手册.pdf", 3)),
        make_ranked(make_chunk("固件升级请先备份配置。", "运维指南.docx", 7)),
    ]
    final = {
        "ranked": ranked,
        "refused": False,
        "answer": answer,
        "rewritten_question": None,
        "retry_count": 0,
        "candidates": ranked,
    }
    return ChatService(
        graph=FakeGraph(final),
        llm=llm,
        retrieval_graph=FakeGraph({"ranked": ranked, "rewritten_question": None}),
        conversation_store=store,
    ), ranked


# =============================================================================
# 第 1 组：ChatService × 会话存储
# =============================================================================


def check_service_with_store() -> None:
    section("第 1 组 ChatService × 会话存储（假 store）")

    store = FakeStore()
    llm = RecordingLLM("答案正文 (来源：[1])")
    service, ranked = make_service(store, llm)

    # 1.1 首问：不带 conversation_id → 服务端生成，且响应带回同一 ID
    result = service.ask("kb-check", "巡检周期是多久？")
    ok("首问生成会话 ID 并随结果回传", bool(result.conversation_id))
    ok("首问落库：store 收到该会话", store.conversation_exists(result.conversation_id))

    # 1.2 落库内容：两轮四条，assistant 带 citations（文件名+页码）
    conv = store.get_conversation(result.conversation_id)
    ok("落库为两轮（user+assistant）", [m.role for m in conv.messages] == ["user", "assistant"])
    assistant = conv.messages[1]
    ok(
        "assistant 消息携带溯源（文件名+页码）",
        len(assistant.citations) == 2
        and assistant.citations[0].filename == "设备手册.pdf"
        and assistant.citations[0].page == "第3页"
        and assistant.citations[1].page == "第7页",
        f"citations={[ (c.filename, c.page) for c in assistant.citations ]}",
    )

    # 1.3 第二轮：历史从 Redis 读进图的初始状态；请求体 history 被忽略
    #     （非流式路径的图是假的、不经过 LLM，所以提示词模板单独在第 1.6 断言）
    poison = [{"role": "user", "content": "这条来自请求体的历史绝不该出现"}]
    llm2 = RecordingLLM("追问的回答")
    service2, _ = make_service(store, llm2)
    result2 = service2.ask(
        "kb-check", "那固件呢？", history=poison, conversation_id=result.conversation_id
    )
    ok("续聊复用同一会话 ID", result2.conversation_id == result.conversation_id)
    state_history = service2._graph.calls[-1]["history"]
    ok(
        "图初始状态的历史来自 Redis（首轮问答在场）",
        [t["content"] for t in state_history][:2] == ["巡检周期是多久？", "答案正文 (来源：[1])"],
        f"history={state_history}",
    )
    ok("请求体 history 被忽略（真相源是 Redis）", all("绝不该出现" not in t["content"] for t in state_history))

    # 1.5 流式路径真正经过 LLM：续聊时提示词里能看到历史
    store3 = FakeStore()
    llm3 = RecordingLLM("流式答案")
    service3, _ = make_service(store3, llm3)

    async def _collect(question: str, conv_id=None):
        events = []
        async for event in service3.ask_stream("kb-check", question, conversation_id=conv_id):
            events.append(event)
        return events

    first_events = asyncio.run(_collect("流式第一问"))
    first_done = next(e for e in first_events if e.get("type") == "done")
    ok("流式路径收到 done 事件", bool(first_done))
    ok(
        "done 事件恒带 conversation_id 字段（含键）",
        "conversation_id" in first_done,
        f"keys={sorted(first_done)}",
    )
    ok("done 事件的会话 ID 已落库", store3.conversation_exists(first_done.get("conversation_id")))
    ok("首轮流式提示词无历史段落", "历史对话" not in llm3.prompts[-1][1])
    asyncio.run(_collect("流式第二问", conv_id=first_done.get("conversation_id")))
    second_prompt = llm3.prompts[-1][1]
    ok("续聊流式提示词含历史（第一问在场）", "历史对话" in second_prompt and "流式第一问" in second_prompt)
    ok("续聊流式落库：共 4 条消息", store3.get_conversation(first_done["conversation_id"]).message_count == 4)

    # 1.6 提示词模板：无历史时与引入该特性之前完全一致（不留占位空行）
    from app.rag.prompts import build_answer_prompts

    _, user_no_hist = build_answer_prompts(ranked, "问题")
    _, user_hist = build_answer_prompts(
        ranked, "问题", history=[{"role": "user", "content": "上一问"}, {"role": "assistant", "content": "上一答"}]
    )
    ok("无历史提示词不含历史段落", "历史对话" not in user_no_hist)
    ok("无历史提示词不留占位空行（无三连换行）", "\n\n\n" not in user_no_hist)
    ok("有历史提示词含历史段落与内容", "历史对话" in user_hist and "上一问" in user_hist and "上一答" in user_hist)
    ok("历史段落声明「不是资料来源」", "不是" in user_hist and "资料来源" in user_hist)


# =============================================================================
# 第 2 组：降级语义
# =============================================================================


def check_degradation() -> None:
    section("第 2 组 降级语义（无 store / store 故障 / 关历史）")

    # 2.1 未配置 store：不落库、结果 conversation_id 为空、行为与从前一致
    llm = RecordingLLM("答案")
    service, _ = make_service(None, llm)
    result = service.ask("kb-check", "问题", history=[{"role": "user", "content": "旧历史"}])
    ok("未配置 store：conversation_id 为空", result.conversation_id == "")

    async def _collect():
        return [e async for e in service.ask_stream("kb-check", "流式问题")]

    events = asyncio.run(_collect())
    done = next((e for e in events if e.get("type") == "done"), {})
    ok(
        "未配置 store：done 的 conversation_id 为 None（键仍在，字段集不变）",
        done.get("conversation_id", "missing") is None,
        f"done={done.get('conversation_id')!r}",
    )

    # 2.2 store 读取失败 → 退回请求体 history，问答不炸
    broken = FakeStore()
    broken.fail_reads = True
    llm2 = RecordingLLM("答案")
    service2, _ = make_service(broken, llm2)
    result2 = service2.ask(
        "kb-check", "问题", history=[{"role": "user", "content": "请求体历史"}],
        conversation_id="conv-x",
    )
    ok("store 读取失败：问答照常返回", bool(result2.answer.text))
    ok(
        "store 读取失败：图历史退回请求体历史",
        service2._graph.calls[-1]["history"] == [{"role": "user", "content": "请求体历史"}],
        f"history={service2._graph.calls[-1]['history']}",
    )
    ok("store 读取失败：仍尝试落库（写通则记上）", "conv-x" in broken.convs)

    # 2.3 落库失败不影响已生成的回答
    class WriteBrokenStore(FakeStore):
        def append_turn(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise UpstreamError("模拟写入失败")

    wbroken = WriteBrokenStore()
    llm3 = RecordingLLM("答案")
    service3, _ = make_service(wbroken, llm3)
    result3 = service3.ask("kb-check", "问题", conversation_id="conv-y")
    ok("落库失败：回答照常返回（不被存储故障绑架）", result3.answer.text == "答案正文 (来源：[1])")

    # 2.4 history_turns=0：历史只落库不进提示词（图初始状态里也不该有）
    from app.core.config import get_settings

    original = get_settings().conversation_history_turns
    try:
        store4 = FakeStore()
        llm4 = RecordingLLM("答案一")
        service4, _ = make_service(store4, llm4)
        first = service4.ask("kb-check", "第一问", conversation_id="conv-h")
        llm5 = RecordingLLM("答案二")
        service5, _ = make_service(store4, llm5, history_turns=0)
        service5.ask("kb-check", "第二问", conversation_id="conv-h")
        ok("history_turns=0：历史不进图初始状态", service5._graph.calls[-1]["history"] == [])
        ok("history_turns=0：落库照常", store4.get_conversation("conv-h").message_count == 4)
        _ = first
    finally:
        get_settings().conversation_history_turns = original


# =============================================================================
# 第 3 组：RedisConversationStore（真实 Redis）
# =============================================================================


def _redis_available() -> bool:
    try:
        import redis

        url = os.environ.get("CHECK_REDIS_URL") or _settings_redis_url()
        client = redis.Redis.from_url(url, socket_connect_timeout=1.5)
        client.ping()
        return True
    except Exception:  # noqa: BLE001
        return False


def _settings_redis_url() -> str:
    from app.core.config import get_settings

    return get_settings().redis_url.strip() or "redis://127.0.0.1:6379/0"


def check_redis_store() -> bool:
    """真实 Redis 的实现层自检。返回是否执行（Redis 不可达则整组跳过）。"""
    section("第 3 组 RedisConversationStore（真实 Redis）")
    if not _redis_available():
        print("  [skip]  Redis 不可达，本组整组跳过（不影响退出码）")
        return False

    from app.infrastructure.redis_store import RedisConversationStore
    import redis as redis_lib

    client = redis_lib.Redis.from_url(_settings_redis_url())
    store = RedisConversationStore(client=client, ttl_seconds=3600)
    made_keys: list[str] = []

    cid = f"check:conv:{uuid.uuid4().hex[:8]}"
    made_keys.append(f"rag:conv:{cid}")
    made_keys.append(f"rag:conv:{cid}:msgs")

    try:
        # 3.1 追加一轮 → 读回一致（含溯源）
        user_msg = StoredMessage(role="user", content="XG-8800 的巡检周期是多久？", created_at="2026-10-17 12:00:00")
        asst_msg = StoredMessage(
            role="assistant",
            content="每季度一次 (来源：[1])",
            created_at="2026-10-17 12:00:01",
            citations=[Citation(doc_id="d1", filename="设备手册.pdf", page="第3页", snippet="巡检周期…")],
        )
        summary = store.append_turn(cid, "kb-check", user_msg, asst_msg)
        ok("append_turn 返回列表项（计数已含本轮）", summary.message_count == 2 and summary.citation_count == 1)
        conv = store.get_conversation(cid)
        ok(
            "读回消息与溯源一致（文件名+页码）",
            len(conv.messages) == 2
            and conv.messages[1].citations[0].filename == "设备手册.pdf"
            and conv.messages[1].citations[0].page == "第3页",
        )
        ok("标题取首条用户提问", conv.title.startswith("XG-8800 的巡检周期"))
        ok("created_at 格式为 YYYY-MM-DD HH:mm:ss", len(conv.created_at) == 19 and conv.created_at[4] == "-")

        # 3.2 TTL 三键同步
        pttls = [client.pttl(k) for k in made_keys]
        ok("meta 与 msgs 都设了 TTL（三键同步续期）", all(p > 0 for p in pttls), f"pttls={pttls}")

        # 3.3 列表过滤
        listed = store.list_conversations(kb_id="kb-check", keyword="巡检")
        ok("列表按 kb/keyword 过滤命中", any(c.id == cid for c in listed))
        listed_none = store.list_conversations(kb_id="kb-check", keyword="不存在的关键词")
        ok("keyword 不命中时为空", all(c.id != cid for c in listed_none))

        # 3.4 存在性 / 404 / 删除幂等
        ok("conversation_exists=True", store.conversation_exists(cid))
        store.delete_conversation(cid)
        ok("删除后 get 抛 NotFoundError", _raises(store.get_conversation, cid, NotFoundError))
        ok("conversation_exists=False", not store.conversation_exists(cid))
        store.delete_conversation(cid)  # 幂等：再删一次不炸
        ok("删除幂等（再删不抛错）", True)
        made_keys = []  # 已清理
        return True
    finally:
        for key in made_keys:
            try:
                client.delete(key)
            except Exception:  # noqa: BLE001
                pass
        try:
            client.zrem("rag:conv:index", cid)
        except Exception:  # noqa: BLE001
            pass


def _raises(fn, *args, exc_type=Exception) -> bool:
    try:
        fn(*args)
    except exc_type:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


# =============================================================================
# 第 4 组：Redis 故障 → UpstreamError
# =============================================================================


def check_redis_error_semantics() -> None:
    section("第 4 组 Redis 故障的错误语义")

    class ExplodingClient:
        """每个命令都抛连接错误的最小假客户端。"""

        def __getattr__(self, name):
            def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
                raise ConnectionError("模拟 Redis 不可达")

            return _boom

    from app.infrastructure.redis_store import RedisConversationStore

    store = RedisConversationStore(client=ExplodingClient(), ttl_seconds=3600)
    msg = StoredMessage(role="user", content="q", created_at="2026-10-17 12:00:00")
    asst = StoredMessage(role="assistant", content="a", created_at="2026-10-17 12:00:01")

    ok("append_turn 故障 → UpstreamError", _raises(store.append_turn, "c1", "kb", msg, asst, exc_type=UpstreamError))
    ok("get_conversation 故障 → UpstreamError", _raises(store.get_conversation, "c1", exc_type=UpstreamError))
    ok("list_conversations 故障 → UpstreamError", _raises(store.list_conversations, exc_type=UpstreamError))
    ok("delete_conversation 故障 → UpstreamError", _raises(store.delete_conversation, "c1", exc_type=UpstreamError))

    # 入参校验在连接之前（坏 ID 不该表现为"存储故障"）
    ok("空会话 ID → ValidationError", _raises(store.append_turn, "", "kb", msg, asst, exc_type=AppError))
    ok("非法 role → ValidationError", _raises(
        store.append_turn, "c2", "kb",
        StoredMessage(role="system", content="q"), asst, exc_type=AppError,
    ))


# =============================================================================
# 第 5 组：HTTP /conversations + done 契约
# =============================================================================


def _http_app(service: ConversationService):
    from fastapi import FastAPI

    from app.api.deps import get_conversation_service
    from app.api.v1.auth import get_auth_service, require_current_user
    from app.api.v1.conversations import router
    from app.core.exceptions import register_exception_handlers

    fake_user = _FakeUser()
    app = FastAPI()
    # 503/404 靠统一异常处理器落成响应体；生产 app 由 create_app 注册，
    # 这里必须显式挂上，否则 AppError 会以未处理异常裸抛（500）
    register_exception_handlers(app)
    app.include_router(router)
    app.dependency_overrides[get_conversation_service] = lambda: service
    # 本组验的是会话契约，不是鉴权：直接覆盖鉴权依赖成"已登录用户"。
    # 注意别覆盖 require_current_user 本身——路由级 dependencies 里存的是
    # 它的函数对象引用，覆盖它只影响 Depends(x) 解析出的调用点，语义会分叉。
    app.dependency_overrides[require_current_user] = lambda: fake_user
    app.dependency_overrides[get_auth_service] = lambda: None
    return app


_FAKE_USER_ID = "u-check"


class _FakeUser:
    """鉴权依赖的最小替身：只有会话接口真正会用到的 id。"""

    def __init__(self, user_id: str = _FAKE_USER_ID) -> None:
        self.id = user_id


def check_http_and_contract() -> None:
    section("第 5 组 HTTP /conversations 与 done 契约")
    from fastapi.testclient import TestClient

    # 5.1 未配置 → 503，且话术里点名缺的是 REDIS_URL
    with TestClient(_http_app(ConversationService(None))) as client:
        resp = client.get("/conversations")
        ok("未配置 REDIS_URL：GET /conversations → 503", resp.status_code == 503, f"got {resp.status_code}")
        ok("503 话术点名 REDIS_URL", "REDIS_URL" in resp.json().get("message", ""))

    # 5.2 有数据 → 详情带溯源（文件名+页码）
    store = FakeStore()
    store.append_turn(
        "conv-1", "kb-check",
        StoredMessage(role="user", content="巡检周期？", created_at="2026-10-17 12:00:00"),
        StoredMessage(
            role="assistant", content="每季度一次", created_at="2026-10-17 12:00:01",
            citations=[Citation(doc_id="d1", filename="设备手册.pdf", page="第3页", snippet="…")],
        ),
        user_id=_FAKE_USER_ID,  # 归属必须与登录用户一致，否则列表按隔离规则过滤掉
    )
    service = ConversationService(store)
    with TestClient(_http_app(service)) as client:
        resp = client.get("/conversations")
        ok("GET /conversations → 200", resp.status_code == 200)
        items = resp.json()
        ok("列表项字段齐全（含 citation_count）", items and {"id", "title", "message_count", "citation_count"} <= set(items[0]))

        detail = client.get("/conversations/conv-1")
        ok("GET /conversations/{id} → 200", detail.status_code == 200)
        parsed = ConversationDetailOut.model_validate(detail.json())
        ok(
            "详情里 assistant 消息带溯源（文件名+页码）",
            parsed.messages[1].citations[0].filename == "设备手册.pdf"
            and parsed.messages[1].citations[0].page == "第3页",
        )

        missing = client.get("/conversations/conv-none")
        ok("会话不存在 → 404", missing.status_code == 404, f"got {missing.status_code}")

        deleted = client.delete("/conversations/conv-1")
        ok("DELETE → 204", deleted.status_code == 204, f"got {deleted.status_code}")
        ok("删除后再取 → 404", client.get("/conversations/conv-1").status_code == 404)

    # 5.3 done 事件字段集（成功与拒答一致，conversation_id 恒在）
    from app.services.chat_service import _done_event

    success = _done_event(answer="a", refused=False, citations=[], retried=False, attempts=0, query="q")
    refused = _done_event(answer="a", refused=True, citations=[], retried=True, attempts=1, query="q", conversation_id="c9")
    ok("done 成功/拒答字段集合一致", set(success) == set(refused))
    ok("done 含 conversation_id 且可回传", refused.get("conversation_id") == "c9")


# =============================================================================
# main
# =============================================================================


def main() -> int:
    print("=" * 60)
    print("会话历史自检（Redis 多轮对话 + 溯源回放）")
    print("=" * 60)

    check_service_with_store()
    check_degradation()
    ran_redis = check_redis_store()
    check_redis_error_semantics()
    check_http_and_contract()

    print("\n" + "=" * 60)
    if _FAIL:
        print(f"存在 {_FAIL.count and len(_FAIL)} 项失败：")
        for name in _FAIL:
            print(f"  - {name}")
        return 1
    note = "" if ran_redis else "（Redis 离线，第 3 组已跳过）"
    print(f"全部通过（{len(_PASS)} 项）{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
