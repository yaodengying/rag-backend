# -*- coding: utf-8 -*-
"""问答路由：一次性与流式（SSE）。

两个端点 + 一个能力发现端点：

    POST /chat          一次性返回完整答案与来源（恒含一次改写重试）
    POST /chat/stream   以 SSE 推送，事件顺序为：
                            event: retry     可选：正在改写查询。仅当本次请求
                                             决定重试（显式 enable_retry=true，
                                             或未表态而服务端默认为 true）
                                             且首轮检索无可用片段时出现，
                                             位置在 sources **之前**
                            event: sources   检索到的来源（先推，前端可先渲染引用）
                            event: delta     回答增量
                            event: done      结束，附完整回答与重试事实
                            event: error     出错
    GET  /chat/options  服务端默认值与事件名，供前端初始化"是否允许重试"的开关

「是否重试」由谁决定（这是本模块最需要说清的一件事）：
    请求体 `enable_retry` 是**三态**——true / false 是前端用户的选择，服务端
    照办；省略或 null 才落回服务端默认值。所以前端有两条路都能满足"让用户
    自己选"：
      (a) 预置开关：把用户的偏好随每次请求传上来；
      (b) 当场询问：先按默认出结果，若 `done.can_retry` 为 true（首轮没找到
          且额度还没用满），再问用户要不要「再找一次」，用户点了就用
          enable_retry=true 把同一个问题重发一遍。
    (b) 的好处是不必让用户盲签：事先谁也不知道这一问用不用得上改写。
    两条路都只是同一个请求字段的不同用法，服务端没有额外分支。

SSE 实现要点：
    1. 响应类用 StreamingResponse，media_type="text/event-stream"。
    2. 必须设 headers：Cache-Control: no-cache、X-Accel-Buffering: no，
       否则 Nginx 等反代会缓冲整个响应，流式形同虚设。
    3. 每条消息格式为 "event: <name>\\ndata: <json>\\n\\n"，双换行结尾。
    4. 客户端断开时（GeneratorExit）要及时释放，避免继续占用 LLM 连接。
    5. 生成阶段可能长达数十秒，需在反向代理上放宽读超时。

为什么 `data` 用 `json.dumps(..., ensure_ascii=False)`：
    SSE 的 data 段以换行结束，任何**裸换行**都会把一条消息截成两条。
    json.dumps 会把换行转义成 \\n，天然安全。
    不用 ensure_ascii=True 是因为中文答案被转成 \\uXXXX 后，
    报文体积膨胀数倍、抓包时完全不可读——SSE 本身是 UTF-8 传输，没有必要。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_chat_service, get_collection_service
from app.api.v1.auth import require_current_user, router_dependencies
from app.schemas.auth import UserOut
from app.schemas.chat import (
    ChatOptionsOut,
    ChatRequest,
    ChatResponse,
    CitationOut,
    ContextUsageOut,
    GroundingOut,
    SourceOut,
)
from app.services.chat_service import ChatResult, ChatService
from app.services.collection_service import CollectionService

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=router_dependencies)

# SSE 响应头。X-Accel-Buffering 是给 Nginx 看的，Cache-Control 是给
# 浏览器与中间缓存看的——缺任何一个都可能让"流式"退化成"等全部生成完再一次吐出"。
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.post("/chat", response_model=ChatResponse, summary="RAG 问答（非流式）")
async def chat(
    payload: ChatRequest,
    current_user: UserOut = Depends(require_current_user),
    service: ChatService = Depends(get_chat_service),
    collections: CollectionService = Depends(get_collection_service),
) -> ChatResponse:
    """检索 + 重排 + 生成，一次性返回答案与引用来源。

    资料不足时返回固定拒答话术且 refused=True，不会调用大模型。

    Raises:
        NotFoundError: 知识库不存在或不属于当前账号（404）
        ValidationError: 入参非法（400）
        UpstreamError: 大模型不可用（502）
    """
    # 先校验知识库存在**且属于当前账号**：ID 打错时给 404，比返回一条
    # "未找到相关内容"有用得多——后者会让人去调 CHUNK_SIZE，而问题其实在 URL 上。
    # 归属校验放在这里与放在 service 里是**两道**，不是重复：这一道是为了
    # 在流式路径开始之前就能给出 404（一旦推了 SSE，状态码就改不了了）。
    collections.ensure_exists(payload.kb_id, current_user.id)

    # 图是同步的（嵌入与交叉编码器都在 CPU 上跑），丢线程池避免阻塞事件循环：
    # 阻塞期间同一 worker 上的健康检查、SSE 心跳、其他请求全都会被卡住。
    result: ChatResult = await asyncio.to_thread(
        service.ask,
        payload.kb_id,
        payload.question,
        payload.history,
        payload.top_k,
        payload.rerank_k,
        payload.conversation_id,
        current_user.id,
    )
    return _to_response(result)


@router.post("/chat/stream", summary="RAG 问答（SSE 流式）")
async def chat_stream(
    payload: ChatRequest,
    current_user: UserOut = Depends(require_current_user),
    service: ChatService = Depends(get_chat_service),
    collections: CollectionService = Depends(get_collection_service),
) -> StreamingResponse:
    """以 SSE 流式返回答案。

    事件：retry? → sources → delta* → done（或 error）

    知识库不存在的检查放在**建流之前**：一旦开始流式响应，
    HTTP 状态码就已经发出去了，此时再报 404 只能塞进事件体里，
    客户端得写两套错误处理。能放在流前的错误就放在流前。
    """
    collections.ensure_exists(payload.kb_id, current_user.id)

    return StreamingResponse(
        _sse_generator(service, payload, current_user.id),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.get(
    "/chat/options",
    response_model=ChatOptionsOut,
    summary="问答能力与默认值（供前端初始化开关）",
)
async def chat_options(service: ChatService = Depends(get_chat_service)) -> ChatOptionsOut:
    """返回流式问答的前端可选项。

    为什么需要这个端点：前端要渲染「是否允许改写重试」的开关，就必须知道
    服务端当前的默认值才能给出正确的初始状态。让前端自己抄一份默认值，
    等于把同一个决定放在两个地方——改一处忘一处时的表现是「界面上开着、
    后端却没重试」，没有任何报错，只有用户觉得"这开关是假的"。
    所以默认值只由服务端持有（配置 `CHAT_STREAM_RETRY_DEFAULT`），前端来读。

    值取自 `ChatService` 而**不是**直接读配置：默认值与上限在服务里已经
    解析过一次，从这里读能保证"端点报出来的"与"真正执行的"是同一份
    （`check_chat.py` 第 13 组有交叉断言钉住这件事，防止端点变成一个
    谁也不校验的装饰品）。
    """
    return ChatOptionsOut(
        stream_retry_default=service.retry_default,
        stream_retry_max_attempts=service.max_retry,
        stream_events=["retry", "sources", "delta", "done", "error"],
        non_stream_always_retries=True,
        retrieve_top_k=service.default_top_k,
        rerank_top_n=service.default_rerank_k,
        retrieve_k_mode=service.retrieve_k_mode,
        grounding_action=service.grounding_action,
    )


# -----------------------------------------------------------------------------
# 内部
# -----------------------------------------------------------------------------


def _to_response(result: ChatResult) -> ChatResponse:
    """ChatResult → 接口响应。"""
    return ChatResponse(
        answer=result.answer.text,
        grounding=(
            GroundingOut(
                status=result.grounding.status,
                checked=result.grounding.checked,
                unsupported=list(result.grounding.unsupported),
                invalid_refs=list(result.grounding.invalid_refs),
                coverage=round(result.grounding.coverage, 4),
                blocked=result.grounding.blocked,
                warnings=list(result.grounding.warnings),
            )
            if result.grounding is not None
            else None
        ),
        sources=[
            SourceOut(
                doc_id=item.chunk.doc_id,
                filename=item.chunk.filename,
                page=item.chunk.page_label,
                chunk_index=item.chunk.chunk_index,
                text=item.chunk.text,
                vector_score=item.vector_score,
                rerank_score=item.rerank_score,
            )
            for item in result.sources
        ],
        citations=[
            CitationOut(
                doc_id=citation.doc_id,
                filename=citation.filename,
                page=citation.page,
                snippet=citation.snippet,
            )
            for citation in result.answer.citations
        ],
        refused=result.answer.refused,
        # retried / query 是非流式路径原先漏掉的观测字段：客户端看不到
        # "到底拿什么去查的、有没有真的重试过"，出问题时只能看服务端日志。
        # ChatResult 里本来就有这两项，之前只是没往外给。
        retried=result.retried,
        query=result.query,
        conversation_id=result.conversation_id or None,
        usage=ContextUsageOut(**result.usage) if result.usage else None,
        truncated=result.truncated,
    )


def _sse_format(event: str, data: str) -> str:
    """把事件编码为 SSE 报文。

    Args:
        event: 事件名，如 sources / delta / done / error
        data: 已序列化的 JSON 字符串

    Returns:
        形如 "event: delta\\ndata: {...}\\n\\n" 的报文。
    """
    return f"event: {event}\ndata: {data}\n\n"


def _dumps(payload: dict) -> str:
    """序列化事件体（中文不转义，见模块文档）。"""
    return json.dumps(payload, ensure_ascii=False)


async def _sse_generator(
    service: ChatService, payload: ChatRequest, owner_id: str | None
) -> AsyncIterator[str]:
    """把 ChatService.ask_stream 的事件流转成 SSE 文本流。

    事件名直接取自事件体的 `type` 字段（`retry` / `sources` / `delta` /
    `done` / `error`），服务层加一种事件这里不用改——这也是为什么
    事件体里同时保留 `type`（自描述）而不是靠 event 名猜。

    客户端的断开语义：
        浏览器关掉页面 / 用户点取消 → Starlette **取消**跑在流上的那个任务，
        这里收到 `asyncio.CancelledError`（手动 `aclose()` 时则是 `GeneratorExit`）。
        两者**都是 `BaseException` 的子类**，下面的 `except Exception` 拦不到它们，
        信号会一路钻到最底层的 `client.astream`，把 httpx 连接释放掉。
        所以这里只放行、不做任何"兜底回复"——一旦把它们转成一条 `error` 事件，
        CPython 还会抛 `RuntimeError: async generator ignored GeneratorExit`。

    注意（实测，别再按"直觉"想当然）：
        `async for` **不会**在退出时自动 `aclose()` 内层生成器。手动 `aclose()` 时
        信号也**不是**同步到底的：要等中间几层生成器被回收、asyncio 的 async-gen
        finalizer 把它们的 `aclose()` 排进事件循环之后才走到底（实测 `aclose()`
        返回时底层 `astream` 的 `finally` 还没跑）。反过来，消费者既不 close 也不
        取消任务的话，一层都不会被清理。
        断言见 `scripts/check_chat.py` 第 11 组。
    """
    try:
        async for event in service.ask_stream(
            payload.kb_id,
            payload.question,
            payload.history,
            payload.top_k,
            payload.rerank_k,
            payload.enable_retry,
            payload.conversation_id,
            owner_id,
        ):
            kind = str(event.get("type") or "message")
            yield _sse_format(kind, _dumps(event))
    except asyncio.CancelledError:
        # 客户端断开：放行，不做任何"兜底回复"
        logger.info("SSE 客户端断开，终止本次生成（kb=%s）", payload.kb_id)
        raise
    except Exception as exc:  # noqa: BLE001
        # 流已开始，状态码改不了，只能以 error 事件收尾
        logger.exception("SSE 生成异常：%s", exc)
        yield _sse_format(
            "error",
            _dumps({"type": "error", "detail": "服务内部错误", "code": "internal_error"}),
        )


__all__ = ["router"]
