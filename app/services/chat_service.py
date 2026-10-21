# -*- coding: utf-8 -*-
"""问答服务：驱动 LangGraph 图，产出带引用的回答。

与 rag/graph.py 的分工：
    graph 定义「图长什么样」；
    ChatService 负责「怎么调图」——组装初始 state、区分一次性/流式输出、
    把图的结果（含各节点中间态）转成对外的 DTO。

一次性 vs 流式（两条路径的实现方式不同，取舍写在下面，不藏着）：
    ask()        跑**完整图**：含改写重试与短路拒答，返回带引用的 Answer。
    ask_stream() 跑**检索子图**，拿到 ranked 后直接 `llm.stream()` 逐 token 推。
                 这样才有真正的增量（图节点返回的是最终字段值，不是增量）。
                 改写重试是否执行，**由调用方（前端 → 前端用户）决定**：
                 一次改写要等 LLM 生成一轮查询（本机 7B 约 2~4 秒），
                 而用户此时盯着一个空白窗口，所以它不该由服务端替用户默认开启。
                 请求字段 `enable_retry` 是三态：显式的 true/false 一定照办，
                 省略/null 才落回配置默认值（`CHAT_STREAM_RETRY_DEFAULT`）。
                 重试的**次数与判据都与完整图同源**（共用
                 `route_after_rerank` 与 `MAX_RETRY`），不会出现
                 "流式重试 3 次、非流式 1 次"这种最难排的偏差。
                 此外 `done` 事件会回填 `retried` / `can_retry`，于是前端还有
                 第二条路：**先按默认（不重试）出结果，只有真的没找到时才问
                 用户要不要再找一次**——比事先盲开开关更省等待，见
                 `_done_event` 的注释。

为什么 ask() 返回 ChatResult 而不是直接 Answer：
    接口响应里 sources（检索到了什么，含四个阶段的分数）与
    citations（模型依据了什么）是两份信息，用途不同：
        sources  —— 检索质量问题：是不是没召回到？
        citations—— 依据问题：模型到底参考了哪几页？页码对不对？
    只回 citations 会让"检索召回了但被重排丢了"这种情况彻底不可见。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from app.core.config import get_settings
from app.core.exceptions import AppError, NotFoundError, UpstreamError
from app.core.tracing import tracing_scope
from app.domain.models import Answer, Citation, RetrievedChunk, StoredMessage
from app.rag.grounding import STATUS_LOW, GroundingReport, audit_answer
from app.rag.graph import NODE_REWRITE, route_after_rerank
from app.rag.prompts import build_answer_prompts
from app.rag.rewrite import rewrite_query
from app.rag.state import MAX_RETRY, NO_ANSWER_TEXT
from app.services.mappers import chunk_preview
from app.services.ownership import ensure_kb_owned

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from app.domain.ports import ChunkRepositoryPort, ConversationStorePort, LLMPort

logger = logging.getLogger(__name__)

# 引用摘要的截断长度。引用是给人核对的，太长会淹没回答本身；
# 400 字足够认出"是不是这一段"，完整正文仍可用 chunk_id 回查。
_CITATION_SNIPPET_LIMIT = 400


@dataclass(slots=True)
class ChatResult:
    """一次问答的完整产出。

    Attributes:
        answer: 领域层回答（text / citations / refused）
        sources: 送入模型的全部片段（含检索与重排的各阶段分数）
        query: 实际用于检索的查询（发生改写时与原始问题不同）
        retried: 是否触发过查询改写重试
        grounding: 接地性核验报告；`None` = 本轮没有可核验的回答（拒答路径）。
            被 block 拦下时**不是** None——那份报告要留下来（`blocked=True`），
            否则运维分不清「被拦」与「真的没检索到」
    """

    answer: Answer
    sources: list[RetrievedChunk] = field(default_factory=list)
    query: str = ""
    retried: bool = False
    # 本轮归属的会话（未配置会话存储时为空串）。首问由服务端生成，
    # 经响应/`done` 事件回传给前端，此后每轮带上它，历史才归得了堆。
    conversation_id: str = ""
    # 事后核对出来的"回答与资料对不上"的迹象（空列表 = 没发现问题）。
    # 只记日志与回传，**不改写回答**：改文字会把"模型没遵守提示词"
    # 这件事洗掉，而它正是要观测的对象。
    # 保留这个平铺字段是为了兼容已有调用方；明细看 `grounding`。
    grounding_warnings: list[str] = field(default_factory=list)
    # 接地性核验的完整报告（status / 未支撑的硬事实 / 覆盖度）。
    # 与 grounding_warnings 同源，但不为 None 时调用方可以据此渲染"低置信"提示。
    grounding: GroundingReport | None = None


class ChatService:
    """RAG 问答服务。"""

    def __init__(
        self,
        graph: "CompiledStateGraph",
        llm: "LLMPort",
        retrieval_graph: "CompiledStateGraph",
        retry_default: bool | None = None,
        conversation_store: "ConversationStorePort | None" = None,
        repository: "ChunkRepositoryPort | None" = None,
    ) -> None:
        """注入编译后的问答图、检索子图与 LLM。

        Args:
            graph: build_rag_graph 的编译产物（非流式路径）
            llm: 大模型客户端（流式路径直接用它）
            retrieval_graph: build_retrieval_graph 的编译产物（流式路径）
            retry_default: 流式路径的「是否允许改写重试」默认值。传 None 表示
                读配置（`CHAT_STREAM_RETRY_DEFAULT`）。做成参数是为了自检能
                把两种默认值都试一遍，不必改环境变量再重启进程。
            conversation_store: 会话存储（Redis 实现）。None 表示本进程
                未配置会话能力：不落库、历史退回请求体、行为与从前一致。
            repository: 元数据仓储，**仅用于归属校验**（判断 kb_id 是否属于
                当前账号）。行为与 RetrievalService 一致：None 时只允许
                owner_id 也为 None 的内部路径。
        """
        self._graph = graph
        self._llm = llm
        self._retrieval_graph = retrieval_graph
        self._store = conversation_store
        self._repository = repository
        self._history_turns = int(get_settings().conversation_history_turns)
        self._default_top_k = int(get_settings().retrieve_top_k)
        self._default_rerank_k = int(get_settings().rerank_top_n)
        self._retry_default = (
            bool(get_settings().chat_stream_retry_default)
            if retry_default is None
            else bool(retry_default)
        )
        # 生成后核验的动作：off（不查）/ flag（只告警）/ block（低置信降级为拒答）。
        # 每次构造时读一次配置——它是部署策略，不该在进程内被改来改去；
        # 自检要切模式请直接构造两个 service 实例。
        self._grounding_action = str(get_settings().grounding_action or "flag").strip().lower()

    # ---------- 只读画像（供接口层做能力发现，避免默认值抄两份） ----------

    @property
    def retry_default(self) -> bool:
        """请求未显式传 `enable_retry` 时，流式路径是否重试。"""
        return self._retry_default

    @property
    def max_retry(self) -> int:
        """改写重试上限（与完整图共用 `state.MAX_RETRY` 这一份，不是另定的数）。"""
        return MAX_RETRY

    @property
    def grounding_action(self) -> str:
        """当前生效的接地性核验动作（off / flag / block）。"""
        return self._grounding_action

    # ---------- 非流式 ----------

    def ask(
        self,
        kb_id: str,
        question: str,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        rerank_k: int | None = None,
        conversation_id: str | None = None,
        owner_id: str | None = None,
    ) -> ChatResult:
        """非流式问答：跑完整图，返回答案与引用来源。

        Args:
            kb_id: 知识库 ID
            question: 用户问题
            history: 多轮历史（配置了会话存储且带 conversation_id 时**被忽略**，
                改由 Redis 提供——服务端才是历史的唯一真相，前端抄一份必然漂移）
            top_k: 每路召回条数；None 用配置默认值
            rerank_k: 重排保留条数；None 用配置默认值
            conversation_id: 会话 ID。配置了会话存储时：传了就续聊该会话，
                没传就生成新 ID（**首问也落库**）；未配置存储时忽略。
            owner_id: 当前登录账号 ID，用于知识库归属校验。
                None 表示跳过校验（仅内部脚本/自检用）。

        Returns:
            ChatResult；资料不足时 answer.refused 为 True 且 citations 为空。

        Raises:
            ValidationError: 知识库或问题为空
            NotFoundError: 知识库不存在或不属于该账号
            UpstreamError: LLM 不可用（检索侧故障抛 RetrievalError）
        """
        question = (question or "").strip()
        if not question:
            # 接口层有 min_length=1 兜着，这里是为了让直接调用服务的路径
            # （如 scripts/ask.py）也拿到一致的、明确的失败语义
            from app.core.exceptions import ValidationError

            raise ValidationError("问题不能为空")

        # 归属校验必须在**任何事之前**：建会话、读历史都依赖 kb_id，
        # 越权者不该因为一次"顺手校验一下"就让服务端为它生成会话 ID
        # 并在 Redis 里落下记录。
        self._ensure_owned(kb_id, owner_id)

        conversation_id = self._resolve_conversation(conversation_id)
        history = self._history_from_store(conversation_id, history)

        initial = self._initial_state(kb_id, question, history, top_k, rerank_k)
        final = self._graph.invoke(initial)

        ranked = list(final.get("ranked") or [])
        refused = bool(final.get("refused"))
        text = str(final.get("answer") or "")

        if not text:
            # 图跑完却没有答案：说明某个节点被改坏了（正常路径要么生成、
            # 要么拒答，都会写 answer）。宁可显式报错，也不要返回空回答。
            raise UpstreamError(
                "问答图执行结束但没有产出回答文本，请检查 graph.py 的分支是否完整",
                detail={"refused": refused, "ranked": len(ranked)},
            )

        citations = [] if refused else [_to_citation(item) for item in ranked]

        # 生成之后的最后一道闸门（见 rag/grounding.py 的模块文档）：
        # 核验回答里的数字/型号/日期是否真在参考资料里出现过。
        #
        # 拒答路径保持 `None` 而不是一个空报告：流式 done 事件里拒答也是
        # `grounding: null`，两条路径必须同一个形状——否则前端要写
        # "可能为 null、也可能是 ok 的空报告"这种两分支，而分支一多必然有
        # 一条没人测（本项目已在这类"字段集合不一致"上栽过）。
        report: GroundingReport | None = None
        if not refused:
            report, should_block = self._ground(text, ranked)
            if should_block:
                # block 是**部署策略**而不是默认行为：它把"看起来正常、其实
                # 没依据"的回答换成明确拒答。代价是拒答率上升（用户会以为
                # 库里没有这条），所以出厂为 flag，只在对外可信度要求极高的
                # 场景打开，并同时盯住评估集的 answer_relevancy。
                logger.warning(
                    "接地性核验未通过，降级为拒答（kb=%s 问题=%r）：%s",
                    kb_id,
                    question,
                    "；".join(report.warnings),
                )
                text = NO_ANSWER_TEXT
                refused = True
                citations = []
                report = replace(report, blocked=True)
            elif report.warnings:
                # 只告警、不改写：改文字会把"模型哪里编了"这个证据洗掉
                logger.warning(
                    "回答接地性告警（kb=%s 问题=%r）：%s",
                    kb_id,
                    question,
                    "；".join(report.warnings),
                )

        warnings = list(report.warnings) if report is not None else []
        result = ChatResult(
            answer=Answer(text=text, citations=citations, refused=refused),
            sources=ranked,
            query=str(final.get("rewritten_question") or question),
            retried=int(final.get("retry_count") or 0) > 0,
            conversation_id=conversation_id or "",
            grounding_warnings=warnings,
            grounding=report,
        )
        self._persist_turn(
            conversation_id, kb_id, question, text, refused,
            [] if refused else [c for c in citations],
            owner_id,
        )
        logger.info(
            "问答完成：kb=%s 会话=%s 问题=%r 查询=%r 候选=%s 上下文=%s 回答=%s 字 "
            "拒答=%s 改写重试=%s 引用=%s",
            kb_id,
            conversation_id or "-",
            question,
            result.query,
            len(final.get("candidates") or []),
            len(ranked),
            len(text),
            refused,
            result.retried,
            len(citations),
        )
        return result

    # ---------- 流式 ----------

    async def ask_stream(
        self,
        kb_id: str,
        question: str,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        rerank_k: int | None = None,
        enable_retry: bool | None = None,
        conversation_id: str | None = None,
        owner_id: str | None = None,
    ) -> AsyncIterator[dict]:
        """流式问答，产出结构化事件。

        Args:
            enable_retry: 首轮检索无可用片段时，是否允许一次改写 + 再检索。
                **三态**：None 表示"调用方没有表态" → 用实例的 `retry_default`
                （默认来自配置 `CHAT_STREAM_RETRY_DEFAULT`，出厂为 False）；
                True / False 是调用方的明确选择，**一定压过默认值**——
                否则"前端用户关掉开关"这件事会被服务端默认值悄悄推翻。
                打开后一次改写在本机 7B 上约多 2~4 秒，期间只推一个 "retry"
                事件，让前端能显式提示用户；重试上限取自 `MAX_RETRY`
                （与完整图同一份），不是这里另定的数。
            conversation_id: 会话 ID。配置了会话存储时：传了就续聊（历史从
                Redis 读并送进提示词），没传就生成新 ID（首问也落库）；
                会话 ID 经 `done` 事件的 `conversation_id` 回传。未配置
                存储时忽略，历史退回请求体 `history`（与从前一致）。

        Yields:
            dict: 事件字典，type 取值
                - "retry":   将要改写查询 {"attempt", "max_attempts", "reason"}
                             （只在 enable_retry 且首轮无可用片段时出现，
                              位置在 sources **之前**）
                - "sources": 检索到的来源列表（生成开始前先推，前端可先展示）
                - "delta":   增量文本 {"content": ...}
                - "done":    结束。成功与拒答两条路径字段**完全一致**
                             （同一个 helper 产出）：{"answer", "refused",
                             "citations", "retried", "attempts", "can_retry",
                             "query", "conversation_id"}。
                            其中 `can_retry` = 拒答且重试额度还没用满，是
                            前端「当场询问用户要不要再试一次」的**唯一信号**：
                            首次请求通常拿到 can_retry=true，用户点了「再找
                            一次」就把 enable_retry=true 连同原问题重发一遍。
                            注意 `retried` ≠ "调过改写"：改写无产出时不会重跑
                            检索，此时 attempts=1 而 retried=false。
                - "error":   出错 {"detail": ...}

        实现要点：
            1. 检索子图是**同步**的（嵌入前向是 CPU 密集的），
               用 `asyncio.to_thread` 丢到线程池，否则会卡住整个事件循环，
               SSE 连接在检索期间无法响应心跳或断开事件。
               改写同理——它也是一次阻塞的 LLM 调用；会话的读写同属此类。
            2. 检索为空 → 同步收到的东西也照常以 delta 形式推出去，
               让前端只需处理一种文本通道，不必为拒答写分支。
            3. 上游出错 → 推 "error" 事件并**正常结束**生成器。
               不抛异常是因为 SSE 一旦开始发送就无法改 HTTP 状态码，
               在流里报错只能靠事件体；抛出去只会让客户端看到一个截断的流。
            4. 会话落库失败**不影响**已经生成的回答——答案先到用户手上，
               历史少记一轮只是可用性损失；反过来（先落库再回答）则是
               用存储故障绑架问答主链路，得不偿失。
        """
        question = (question or "").strip()
        if not question:
            yield {"type": "error", "detail": "问题不能为空"}
            return

        # 归属校验。**必须在流开始之前**，且不能指望前端只对按钮做判断：
        # 一旦开始推 SSE 就只能靠事件体报错，HTTP 状态码已经发出去了。
        # 所以越权在这里直接以 error 事件收尾，并且**不生成会话 ID、
        # 不读历史、不落库**——越权请求不该在任何存储里留下痕迹。
        try:
            self._ensure_owned(kb_id, owner_id)
        except AppError as exc:
            yield {"type": "error", "detail": exc.message, "code": exc.code}
            return

        conversation_id = self._resolve_conversation(conversation_id)
        history = self._history_from_store(conversation_id, history)

        initial = self._initial_state(kb_id, question, history, top_k, rerank_k)

        try:
            final = await asyncio.to_thread(self._retrieval_graph.invoke, initial)
        except AppError as exc:
            yield {"type": "error", "detail": exc.message, "code": exc.code}
            return

        ranked = list(final.get("ranked") or [])

        # 三态解析：None = 调用方没表态 → 用服务端默认值；True / False 是明确的
        # 选择，**一定压过默认值**。这条"显式选择优先于默认"必须有断言钉住
        # （check_chat.py 第 13 组）：写成 `if not enable_retry: 用默认值`
        # 就会让"前端用户关掉开关"被服务端默认值悄悄推翻，而界面上看起来
        # 一切正常——这是最容易漏掉的一种反向 bug。
        allow_retry = self._retry_default if enable_retry is None else bool(enable_retry)

        # 可选的改写重试（是否允许由请求或默认值决定，见 allow_retry）。
        #
        # 这里刻意**复用完整图的判据与上限**：`route_after_rerank` 内部含
        # 「retry_count < MAX_RETRY」的边界判断，所以流式重试跑几次、
        # 何时收手，与 `/chat` 完全同源。若在服务层另写一个 `if not ranked`
        # 加硬编码次数，就会出现「流式重试 3 次、非流式 1 次」这类双向
        # 不一致——两边都会动，是最难定位的一种偏差。
        attempts = 0  # 改写尝试次数（含"改写了但无产出"那一次）
        retried = False  # 是否**真的**用改写后的查询重新检索过，与 attempts 不同
        while (
            allow_retry
            and route_after_rerank({"ranked": ranked, "retry_count": attempts})
            == NODE_REWRITE
        ):
            attempts += 1
            # 先推事件、再改写：本事件的唯一作用是填住那 2~4 秒静默，
            # 它不是「重试成功」的承诺——改写可能无产出（此时随即走拒答）。
            yield {
                "type": "retry",
                "attempt": attempts,
                "max_attempts": MAX_RETRY,
                "reason": "首轮检索无可用片段，正在改写查询",
            }
            # 改写同样是阻塞的 LLM 调用，与检索一样要丢线程池
            rewritten = await asyncio.to_thread(
                rewrite_query, self._llm, question, attempt=attempts
            )
            if rewritten == question:
                # 改写无产出 → 第二轮结果必然与第一轮相同，别再白跑一次检索
                logger.info("流式问答：改写无有效产出，放弃重试（kb=%s）", kb_id)
                break
            try:
                final = await asyncio.to_thread(
                    self._retrieval_graph.invoke,
                    self._initial_state(
                        kb_id,
                        question,
                        history,
                        top_k,
                        rerank_k,
                        rewritten_question=rewritten,
                        retry_count=attempts,
                    ),
                )
            except AppError as exc:
                yield {"type": "error", "detail": exc.message, "code": exc.code}
                return
            retried = True
            ranked = list(final.get("ranked") or [])
            logger.info(
                "流式问答：改写重试第 %s 次后取到 %s 条可用片段（kb=%s）",
                attempts,
                len(ranked),
                kb_id,
            )

        # 实际用于检索的查询：`final` 此时要么是首轮结果，要么是重试那一轮。
        # 重试那一轮的初始状态里带了 rewritten_question，所以这里读到的就是
        # 真正生效的查询。放在拒答分支**之前**算，是为了让 done 事件在
        # 「改了写但没救回来」时也能如实回填 query——排查时最想知道的就是
        # "到底拿什么去查的"。
        query = str(final.get("rewritten_question") or question)

        yield {
            "type": "sources",
            "sources": [_source_dict(item) for item in ranked],
        }

        if not ranked:
            logger.info(
                "流式问答：无可用片段，直接拒答（kb=%s 改写尝试=%s 已重检索=%s）",
                kb_id,
                attempts,
                retried,
            )
            yield {"type": "delta", "content": NO_ANSWER_TEXT}
            if self._store is not None:
                await asyncio.to_thread(
                    self._persist_turn,
                    conversation_id, kb_id, question, NO_ANSWER_TEXT, True, [],
                    owner_id,
                )
            yield _done_event(
                answer=NO_ANSWER_TEXT,
                refused=True,
                citations=[],
                retried=retried,
                attempts=attempts,
                query=query,
                conversation_id=conversation_id,
            )
            return

        # 生成用**原始问题**，不是检索用的改写查询：改写是为了让索引更容易命中，
        # 而回答要面对的是用户真正问的那件事。两者混用会出现
        # "用户问巡检周期，回答却在讲设备巡检查询字符串"这类错位。
        # 历史与非流式路径同源：都进 build_answer_prompts（check_conversations
        # 第 1 组钉过"流式漏传历史"这个 bug，漏传时续聊变成单轮且无报错）。
        system, user = build_answer_prompts(ranked, question, history=history)
        if query != question:
            logger.info("流式问答：检索用了改写查询 %r（原始问题 %r）", query, question)

        pieces: list[str] = []
        try:
            # LangSmith 追踪（未启用时 tracing_scope 返回 nullcontext，零开销）。
            #
            # 为什么要在这里单独包一层：`/chat/stream` 只跑**检索子图**，
            # 生成阶段就在这个循环里直接调 `llm.stream()`，不在图里。
            # 只配环境变量的话，LangSmith 里这条 trace 只有 retrieve / rerank
            # 两个节点——"谁生成的、提示词多长、吐了多少 token、首字延迟多少"
            # 全都看不到，等于监控缺了一半。见 app/core/tracing.py 的模块文档。
            with tracing_scope("stream_generate", run_type="llm", tags=["chat", "stream"]):
                async for delta in self._llm.stream(system, user):
                    pieces.append(delta)
                    yield {"type": "delta", "content": delta}
        except AppError as exc:
            logger.warning("流式生成失败：%s", exc)
            yield {"type": "error", "detail": exc.message, "code": exc.code}
            return

        answer = "".join(pieces).strip()
        refused = NO_ANSWER_TEXT in answer
        report = GroundingReport()
        if not refused:
            # 两条路径用**同一个** `_ground`，判据与文案不会分叉。
            #
            # 但流式**永远只 flag**：token 已经逐段推给前端了，事后判定收不回来
            # ——这是 SSE 的硬约束，不是保守选择。`grounding_action=block` 在
            # 流式下退化为 flag：done 事件照样带 `status=low`，由前端提示
            # "本回答含资料未支撑的数字，请核对引用"。想真正拦下，就用非流式 `/chat`。
            report, _ = self._ground(answer, ranked)
            if report.warnings:
                logger.warning(
                    "流式回答接地性告警（kb=%s 问题=%r）：%s",
                    kb_id,
                    question,
                    "；".join(report.warnings),
                )
        citations = [] if refused else [_citation_dict(item) for item in ranked]
        if self._store is not None:
            await asyncio.to_thread(
                self._persist_turn,
                conversation_id,
                kb_id,
                question,
                answer,
                refused,
                # 存储层要的是领域 Citation（含页码），不是接口 dict
                [] if refused else [_to_citation(item) for item in ranked],
                owner_id,
            )
        logger.info(
            "流式问答完成：kb=%s 会话=%s 查询=%r 上下文=%s 回答=%s 字 拒答=%s "
            "改写尝试=%s 已重检索=%s",
            kb_id,
            conversation_id or "-",
            query,
            len(ranked),
            len(answer),
            refused,
            attempts,
            retried,
        )
        yield _done_event(
            answer=answer,
            refused=refused,
            citations=citations,
            retried=retried,
            attempts=attempts,
            query=query,
            conversation_id=conversation_id,
            grounding=_grounding_dict(report),
        )

    # ---------- 内部 ----------

    def _ground(self, answer: str, ranked: list[RetrievedChunk]) -> tuple[GroundingReport, bool]:
        """跑一次接地性核验，返回 (报告, 是否应当拦下)。

        三个 action：
            off   —— 不做核验（只用于对照实验；核验本身是微秒级字符串运算，
                     关掉它省不出可观的开销）。
            flag  —— 只产告警，回答照常返回。**出厂值**。
            block —— `status=low` 时由调用方把回答换成拒答话术。
                     只对非流式生效，流式改为在 done 里标记（见 ask_stream 注释）。

        为什么 block 的判据是 `low` 而不是"有任何未支撑的数字"：
            `warn` 这一档专门留给"单个孤立数字对不上"——中文回答里的
            "3 个步骤"这类量词、以及"48 小时"写成"两天"的同义改写都会
            落在这里，拦下来就是误杀。`low` 对应的是型号/日期对不上、
            或半数以上数字对不上，那已经不是措辞问题。
        """
        if self._grounding_action == "off" or not answer:
            return GroundingReport(), False
        report = audit_answer(answer, ranked)
        return report, self._grounding_action == "block" and report.status == STATUS_LOW

    def _ensure_owned(self, kb_id: str, owner_id: str | None) -> None:
        """归属校验的转发口（实现见 `services/ownership.py`）。

        保留这个私有方法而不是在调用点直接调 `ensure_kb_owned`：
        调用点有 3 处，都写着 `self._repository` 反而更容易在新增
        调用时漏传。方法名与参数保持不变，也让既有自检不必改动。

        Raises:
            NotFoundError: 知识库不存在、不属于该账号，或未装配仓储
        """
        ensure_kb_owned(self._repository, kb_id, owner_id)

    def _resolve_conversation(self, conversation_id: str | None) -> str | None:
        """会话 ID 解析。

        未配置存储 → 恒 None（完全不进入会话语义，行为与从前一致）；
        配置了存储 → 传了就用传的，没传就生成新 ID。
        **首问也生成 ID** 是刻意为之：首问不落库的话，第二轮从 Redis 读出的
        历史就缺了第一轮，多轮上下文从第一天起就是残的。
        """
        if self._store is None:
            return None
        cid = (conversation_id or "").strip()
        return cid or str(uuid.uuid4())

    def _history_from_store(
        self,
        conversation_id: str | None,
        fallback: list[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        """从会话存储读历史（最近 N 条），读不到再退回请求体里的 history。

        退级链：Redis 正常 → 服务端历史（真相源）；
                会话过期/不存在 → 空历史（它本来就是新会话）；
                Redis 故障 → 请求体 history（问答不能因存储故障不可用）。

        `conversation_history_turns=0` 表示「落库与溯源照常，但不把历史送进
        提示词」——有些语料下历史会稀释参考资料的存在感。
        """
        if conversation_id and self._store is not None:
            try:
                conversation = self._store.get_conversation(conversation_id)
            except NotFoundError:
                return []
            except AppError as exc:
                logger.warning("会话历史读取失败，本轮退回请求体历史：%s", exc)
                return list(fallback or [])
            recent = (
                conversation.messages[-self._history_turns:]
                if self._history_turns > 0
                else []
            )
            return [{"role": m.role, "content": m.content} for m in recent]
        return list(fallback or [])

    def _persist_turn(
        self,
        conversation_id: str | None,
        kb_id: str,
        question: str,
        answer: str,
        refused: bool,
        citations: list[Citation],
        user_id: str | None = None,
    ) -> None:
        """把一轮问答写进会话存储。调用方保证 store 已配置且 conversation_id 非空。

        Args:
            user_id: 会话归属。**只在首轮写入时生效**（存储层用 HSETNX），
                续聊不会覆盖已有归属。None / 空串表示无法确定归属
                （内部脚本路径），此时会话变成"无主"，HTTP 层读不到它。

        任何存储侧失败都只记日志：答案已经生成（甚至已经推给用户了），
        此时抛错只会让一次成功回答被 500/SSE error 收尾。
        """
        if self._store is None or not conversation_id:
            return
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            self._store.append_turn(
                conversation_id,
                kb_id,
                StoredMessage(role="user", content=question, created_at=now),
                StoredMessage(
                    role="assistant",
                    content=answer,
                    created_at=now,
                    citations=list(citations),
                ),
                user_id=user_id or "",
            )
        except AppError as exc:
            logger.warning("会话落库失败（不影响本次回答）：%s", exc)

    def _initial_state(
        self,
        kb_id: str,
        question: str,
        history: list[dict[str, str]] | None,
        top_k: int | None,
        rerank_k: int | None,
        rewritten_question: str | None = None,
        retry_count: int = 0,
    ) -> dict:
        """组装图的初始状态。

        每个字段都显式给值：`RAGState` 是 `total=False` 的 TypedDict，
        字段缺失时节点里 `.get()` 拿到的是 None，而不是默认值——
        `top_k=None` 与"不传 top_k"在节点里看起来一样，但
        显式写入配置默认值后，日志与 trace 里能看到本次实际用的深度。

        Args:
            rewritten_question: 改写重试时传入改写后的查询。
                **不传**（None）时故意不写这个键：检索节点会把
                `rewritten_question` 初始化成本次查询，这是"检索口径统一"
                的前提（见 rag/nodes.py 第 1 条约定）。
            retry_count: 已发生的改写次数。检索子图里没有条件边，
                这个字段不影响执行，但回填后日志与 trace 能看到"这次是重试"。
        """
        state = {
            "kb_id": kb_id,
            "question": question,
            "history": list(history or []),
            "top_k": int(self._default_top_k if top_k is None else top_k),
            "rerank_k": int(self._default_rerank_k if rerank_k is None else rerank_k),
            "retry_count": int(retry_count),
        }
        if rewritten_question is not None:
            state["rewritten_question"] = rewritten_question
        return state


# -----------------------------------------------------------------------------
# 领域模型 → DTO 片段
# -----------------------------------------------------------------------------


def _to_citation(item: RetrievedChunk) -> Citation:
    """检索片段 → 领域 Citation（含页码与重排分）。"""
    chunk = item.chunk
    return Citation(
        doc_id=chunk.doc_id,
        filename=chunk.filename,
        page=chunk.page_label,
        snippet=chunk_preview(chunk, _CITATION_SNIPPET_LIMIT),
        rerank_score=item.rerank_score,
    )


def _citation_dict(item: RetrievedChunk) -> dict:
    """检索片段 → 与 schemas.CitationOut 对齐的字典。"""
    chunk = item.chunk
    return {
        "doc_id": chunk.doc_id,
        "filename": chunk.filename,
        "page": chunk.page_label,
        "snippet": chunk_preview(chunk, _CITATION_SNIPPET_LIMIT),
    }


def _done_event(
    *,
    answer: str,
    refused: bool,
    citations: list[dict],
    retried: bool,
    attempts: int,
    query: str,
    conversation_id: str | None = None,
    grounding: dict | None = None,
) -> dict:
    """构造收尾的 `done` 事件。

    成功路径与拒答路径都从这一个地方出。此前两处各写一个字段不同的字面
    dict（拒答那条连 `citations` 都缺），前端只能靠"有没有某个键"分两个
    解析分支——而漏字段这种事不会有任何报错，只会在前端表现为某个标签
    永远空着。所以字段集合由这里唯一决定。

    `can_retry` 由 `refused` 与剩余额度**现算**，不单独传参：否则调用方可以
    写出"说自己还能重试、额度其实早就用满"这种自相矛盾的事件，而前端
    会照着它弹一个点了没用的按钮。

    `conversation_id` 恒在字段集里（未配置会话存储时为 None）：前端只认
    "有这个键且非空" 一种判定，不写 "键可能不存在" 的第二分支。

    `grounding` 同理恒在字段集里（未核验/拒答路径为 None）。流式下它是
    **唯一能告诉前端"这条回答不太可信"的通道**——token 已经推出去了，
    服务端没法把答案换成拒答话术，只能在这里标记。
    """
    return {
        "type": "done",
        "answer": answer,
        "refused": refused,
        "citations": citations,
        "retried": retried,
        "attempts": attempts,
        "can_retry": refused and attempts < MAX_RETRY,
        "query": query,
        "conversation_id": conversation_id,
        "grounding": grounding,
    }


def _grounding_dict(report: GroundingReport | None) -> dict | None:
    """接地性报告 → 可 JSON 序列化的字典（None 表示本轮没有可核验的回答）。"""
    if report is None:
        return None
    return {
        "status": report.status,
        "checked": report.checked,
        "unsupported": list(report.unsupported),
        "invalid_refs": list(report.invalid_refs),
        "coverage": round(report.coverage, 4),
        "blocked": report.blocked,
        "warnings": list(report.warnings),
    }


def _source_dict(item: RetrievedChunk) -> dict:
    """检索片段 → 与 schemas.SourceOut 对齐的字典。

    四个分数全部带出（允许 None），与检索接口的数据口径一致——
    同一个片段在 `/search` 与 `/chat` 里显示的分数必须是同一套，
    否则调参会得到互相矛盾的结论。
    """
    chunk = item.chunk
    return {
        "doc_id": chunk.doc_id,
        "filename": chunk.filename,
        "page": chunk.page_label,
        "chunk_index": chunk.chunk_index,
        "text": chunk_preview(chunk, _CITATION_SNIPPET_LIMIT),
        "vector_score": item.vector_score,
        "rerank_score": item.rerank_score,
    }
