"""FastAPI 应用入口：装配中间件、路由与生命周期钩子。

启动顺序（lifespan）：
    1. setup_logging()
    2. settings.validate_runtime()      —— 配置与模型目录自检，早失败早发现
    3. database.init_schema()           —— 建表（幂等）
    4. vector_store.ensure_collection() —— 建/校验 Milvus collection 与维度
       维度取自 **embedder**（最终追溯到权重 config.json），不是配置值：
       MILVUS_DIMENSION 只是声明，真正的维度由模型决定。用前者建表、
       后者写入，就会出现"建表 1024、写入 768"这种只能等上传才炸的错配。
    5. 预热 embedder（可选）             —— 首次推理较慢，启动时预热可避免首个请求超时

关闭顺序：
    container.shutdown()  —— 释放 Milvus 连接与线程池

中间件：
    - CORS：来源由 settings.cors_origin_list 提供
    - 请求日志/耗时（可选，建议加，便于定位慢请求）
    - 全局异常处理器：见 core/exceptions.register_exception_handlers

关于预热：默认**关闭**（EMBEDDING_PRELOAD 未配置时不预热）。
    加载 BGE 模型要十几秒，开发期每次重启都等会很难受；
    但容器部署时打开它可以避免第一个上传请求超时。
    这里做成可配置，不替使用者做决定。
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app import __version__
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期：启动时初始化资源，关闭时释放。"""
    settings = get_settings()

    # 1. 日志
    setup_logging(settings.log_level)

    # 2. 启动期自检：配置缺项、模型目录、数值区间
    settings.validate_runtime()

    # 2.5 链路追踪（可选）。必须在业务模块**之前**装配：
    #     configure_tracing 把配置写进环境变量，而 langsmith SDK 在
    #     第一次建 tracer 时读环境变量——晚于第一次问答就晚了。
    #     未配置 LANGSMITH_API_KEY 时这里返回 False，不做任何事。
    from app.core.tracing import configure_tracing

    configure_tracing()

    # 3. 建表（幂等）
    from app.core import container

    database = container.get_database()
    database.init_schema()

    # 4. 建/校验 Milvus collection
    #    维度以向量化器为准（它从权重 config.json 解析，并与 MILVUS_DIMENSION
    #    交叉校验过，不一致会在 get_embedder() 里直接抛错）。
    vector_store = container.get_vector_store()
    vector_store.ensure_collection(container.get_embedder().dimension)

    # 5. 可选预热：让首个上传请求不必等模型加载
    if os.environ.get("EMBEDDING_PRELOAD", "").strip().lower() in {"1", "true", "yes"}:
        container.get_embedder().embed_query("预热")

    logger.info(
        "%s v%s 已就绪 | LLM=%s | collection=%s | 上传白名单=%s",
        settings.app_name,
        __version__,
        settings.llm_model,
        settings.milvus_collection,
        ",".join(settings.upload_extension_list),
    )
    try:
        yield
    finally:
        # 先刷 trace 再释放资源：资源释放会关掉 HTTP 客户端池，而 trace
        # 上报走的是另一个连接；顺序反了不至于出错，但"最后几条 trace
        # 查不到"恰恰发生在进程正在退出的时刻，是本项目最想避免的观测盲区。
        from app.core.tracing import flush_tracing

        flush_tracing()
        container.shutdown()
        logger.info("资源已释放，服务退出")


def _install_upload_guard(app: FastAPI) -> None:
    """按 `Content-Length` 提前拒绝超大上传（413）。

    为什么不能只靠路由里的"读时封顶"：
        Starlette 解析 multipart 时，**先把整个请求体写进临时文件**
        （超过 1MB 就落盘），然后才交给我们。一个 5GB 的请求会在
        `_read_upload` 之前就把磁盘写满——那时再报"超限"已经晚了。

    `Content-Length` 可以被伪造或缺失（chunked），所以这只是**第一道**；
    真正的保证仍是 `_read_upload` 的读取封顶与服务层的体积校验。
    三道闸门的分工：
        Content-Length  —— 挡掉明显离谱的请求，省掉落盘
        读取封顶        —— 挡掉伪造 Content-Length 的情况，省掉内存
        服务层校验      —— 唯一可信边界（脚本/内部调用也会走到）

    这里只处理上传路径，其它请求不受影响；放行时**不读 body**，
    否则会把后面的 multipart 解析弄坏。
    """
    settings = get_settings()

    @app.middleware("http")
    async def _limit_upload_size(request: Request, call_next):
        if request.method != "POST":
            return await call_next(request)

        path = request.url.path
        if not path.endswith("/documents") and not path.endswith("/documents/batch"):
            return await call_next(request)

        raw = request.headers.get("content-length")
        if not raw:
            return await call_next(request)

        try:
            length = int(raw)
        except ValueError:
            return await call_next(request)

        limit = (
            settings.max_batch_total_bytes
            if path.endswith("/documents/batch")
            else settings.max_upload_bytes
        )
        # multipart 的边界、字段名、文件名都有开销，给 1MB 余量；
        # 否则一个正好 50MB 的合法文件会因为边界字节被 413 挡掉。
        if length > limit + 1024 * 1024:
            logger.warning("拒绝超大上传：path=%s content-length=%s", path, length)
            return JSONResponse(
                status_code=413,
                content={
                    "code": "payload_too_large",
                    "message": (
                        f"请求体 {length / 1024 / 1024:.1f} MB 超过上限 "
                        f"{limit / 1024 / 1024:.0f} MB；"
                        + (
                            "请减少单次上传的文件数或分批上传"
                            if path.endswith("/batch")
                            else "请压缩或拆分后再上传"
                        )
                    ),
                    "detail": {"content_length": length, "limit_bytes": limit},
                },
            )
        return await call_next(request)


def create_app() -> FastAPI:
    """应用工厂。

    用工厂而不是模块级单例的好处：测试可多次创建互不干扰的应用实例。

    Returns:
        已装配完成的 FastAPI 实例。
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        debug=settings.debug,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    _install_upload_guard(app)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/", include_in_schema=False)
    async def _root() -> RedirectResponse:
        """根路径跳到 Swagger，方便本地调试。"""
        return RedirectResponse(url="/docs")

    return app


# 供 `uvicorn app.main:app` 使用
app = create_app()
