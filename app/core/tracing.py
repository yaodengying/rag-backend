# -*- coding: utf-8 -*-
"""LangSmith 链路追踪的统一出口。

为什么单独一个模块，而不是各处 import langsmith：
    全项目只有**这一个**文件知道 LangSmith 存在。业务代码调用的是
    `traced_run()`，启用与不启用走**同一个调用点**。这样"关掉追踪"
    不需要改任何业务代码，也不会出现「某个模块忘了加判断、于是关不掉」
    的半开状态。

三个入口：
    configure_tracing()  启动期调一次：把 Settings 的值灌进环境变量
    traced_run()         业务代码用：把一段**调用**包成 trace（未启用时零开销直通）
    flush_tracing()      进程退出前调一次：等在途上报落地，别丢最后几条

---

为什么是「灌环境变量」而不是把配置传给 SDK：

    langsmith SDK 的配置入口**只认环境变量**（`LANGSMITH_API_KEY` 等）。
    `langchain_core.tracers` 在模块导入期就会读一次环境变量来决定行为，
    公开的 client 构造参数只覆盖其中一部分。强行用代码传参，会得到
    "一部分配置生效、一部分不生效"的结果——这种半生效比全不生效更难查。

    所以方向是反的：**由我们读 .env，再把值写回环境变量**。配额的唯一真相
    仍是 .env（与项目其它配置一致），SDK 拿到的是它认识的形式。

为什么不用 `LANGCHAIN_TRACING_V2` 这个旧名字：
    它是 LangChain v1 之前的写法，langsmith 0.14 仍兼容但已标记为旧称。
    混用两个名字会得到"两个都设了、谁生效看版本"的结果，所以统一用
    `LANGSMITH_*` 这一套。

为什么没有「装饰器」入口（这是设计取舍，不是没写）：
    装饰器在**模块导入期**就被应用，而 `configure_tracing()` 在 lifespan 里
    执行——导入永远先于配置。所以装饰器只能在导入期决定"包不包"，那一刻
    追踪状态还没确定，只能二选一：
        (a) 总是包一层、内部再判断 → 关掉追踪也永远带着一层 wrapper，
            额外开销、异常栈变形、签名退化成 (*args, **kwargs)；
        (b) 从不包 → 追踪永远不生效。
    两者都不是想要的结果。真正需要追踪的位置是**流式生成**（一个调用点），
    用 `traced_run()` 在调用期判断即可，代价是调用处要显式写一层。
    这是本项目唯一愿意接受的形状：状态不确定的地方不装饰。

---

关闭时必须**完全一样**（本模块最重要的约束）：

    未启用时 `traced_run()` 直接执行函数并原样返回结果，不 import langsmith、
    不产生额外对象。少一层 wrapper 意味着没有额外调用开销、不改变异常栈。
    这条由 `scripts/check_tracing.py` 钉住：关掉追踪时，问答链路的行为
    必须与完全没有本模块时一致。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# 进程级状态。`_enabled` 是**唯一**的启用判据，`traced_run()` 每次调用都读它。
_configured: bool = False
_enabled: bool = False

# Settings 字段 → SDK 认识的环境变量名。
# 键名必须逐字正确（SDK 按名字读）。这里统一写大写，避免 Windows（环境变量
# 大小写不敏感）与 Linux（敏感）之间出现"本机好、容器坏"的分歧。
_ENV_MAP: dict[str, str] = {
    "langsmith_api_key": "LANGSMITH_API_KEY",
    "langsmith_project": "LANGSMITH_PROJECT",
    "langsmith_endpoint": "LANGSMITH_ENDPOINT",
    "langsmith_hide_inputs": "LANGSMITH_HIDE_INPUTS",
    "langsmith_hide_outputs": "LANGSMITH_HIDE_OUTPUTS",
    "langsmith_log_errors": "LANGSMITH_LOG_ERRORS",
}


def configure_tracing() -> bool:
    """启动期装配：把 Settings 里的追踪配置灌进环境变量。幂等。

    Returns:
        True 表示追踪已启用；False 表示未配置 API Key（正常状态，非错误）。
    """
    global _configured, _enabled
    if _configured:
        return _enabled

    from app.core.config import get_settings

    settings = get_settings()

    # 未配置 Key：什么都不做，且**不写任何环境变量**。
    # 刻意如此——若这里顺手写了 LANGSMITH_PROJECT，而使用者 shell profile 里
    # 另有 LANGSMITH_API_KEY，就会得到"我们以为关了、SDK 却在记录"的意外上报。
    # 不碰环境变量 = 关掉就是真的关掉。
    if not settings.langsmith_enabled:
        _configured = True
        _enabled = False
        logger.debug("LangSmith 追踪未启用（LANGSMITH_API_KEY 未配置）")
        return False

    for field, env_key in _ENV_MAP.items():
        value = getattr(settings, field, "")
        if isinstance(value, bool):
            os.environ[env_key] = "true" if value else "false"
        else:
            text = str(value or "").strip()
            if text:
                os.environ[env_key] = text

    # 显式置 true 而不是依赖 SDK 默认值：默认值随版本变过，而"有没有 key"
    # 才是我们这边的判据。走到这里说明 key 一定存在。
    os.environ.setdefault("LANGSMITH_TRACING", "true")

    _configured = True
    _enabled = True
    logger.info(
        "LangSmith 追踪已启用：project=%s endpoint=%s 隐藏输入=%s 隐藏输出=%s",
        settings.langsmith_project,
        settings.langsmith_endpoint or "默认（https://api.smith.langchain.com）",
        settings.langsmith_hide_inputs,
        settings.langsmith_hide_outputs,
    )
    if not settings.langsmith_hide_inputs or not settings.langsmith_hide_outputs:
        # warning 而非 info：这意味着用户问题与上传文档正文会离开本机，
        # 必须在日志里一眼看见，而不是埋在启动横幅里。
        logger.warning(
            "LangSmith 追踪将上报%s——若库中含敏感文档，请把 "
            "LANGSMITH_HIDE_INPUTS / LANGSMITH_HIDE_OUTPUTS 设为 true",
            "输入" if not settings.langsmith_hide_inputs else "输出",
        )
    return True


def traced_run(
    name: str,
    func: F,
    *args: Any,
    run_type: str = "chain",
    tags: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """把一次调用包成 LangSmith run；未启用追踪时**直接调用**，无 wrapper。

    用法（同步）::

        result = traced_run("stream_generate", _consume, system, user, run_type="llm")

    用法（异步生成器）则不适合本函数——异步迭代要在调用方展开，
    见 `ChatService.ask_stream` 里的写法：启用时用 `tracing_v2_enabled()`
    上下文管理器把生成阶段包起来，未启用时原样迭代。

    Args:
        name: 在 LangSmith 里显示的 run 名称。用**稳定**的名字（别把 kb_id
            之类的动态值拼进去）——LangSmith 按 name 聚合延迟与错误率，
            动态名会把统计拆成几百条各 1 样本的曲线。
        func: 要执行的函数。
        *args / **kwargs: 原样传给 func。
        run_type: run 类型，影响 LangSmith 的图标与聚合口径。
            "chain" —— 组合步骤；"llm" —— 真正调用模型（口径带 token 数）。
        tags: 附加标签，可在 LangSmith 里按标签过滤。

    Returns:
        func 的返回值，原样透传。
    """
    if not _enabled:
        # 直通：不 import langsmith、不建对象、不改异常栈。
        return func(*args, **kwargs)

    from langsmith.run_helpers import traceable

    wrapped = traceable(run_type=run_type, name=name, tags=tags or [])(func)
    return wrapped(*args, **kwargs)


def tracing_scope(
    name: str,
    *,
    run_type: str = "chain",
    tags: list[str] | None = None,
):
    """上下文管理器形式的 trace 包装（供无法用函数包裹的写法使用）。

    典型场景：`async for delta in llm.stream(...)` —— 要追踪的是**整个迭代
    过程**而不是单次取值，用函数包裹会变成一个不消费的生成器对象（LangSmith
    里看到 0ms、0 token，比没有 trace 更误导）。

    用法::

        with tracing_scope("stream_generate", run_type="llm"):
            async for delta in llm.stream(system, user):
                ...

    未启用追踪时返回一个**空上下文**（`contextlib.nullcontext`），
    零开销。
    """
    if not _enabled:
        from contextlib import nullcontext

        return nullcontext()

    # tracing_v2_enabled 会为当前上下文的 runnable 挂上 tracer；
    # 本项目的生成阶段直接调 ChatOpenAI（本身是 Runnable），所以能捕获到
    # 真实的 LLM run（含 token 数与首字延迟）。
    from langchain_core.tracers.context import tracing_v2_enabled

    from app.core.config import get_settings

    settings = get_settings()
    return tracing_v2_enabled(
        project_name=settings.langsmith_project,
        tags=tags or [],
    )


def flush_tracing() -> None:
    """等待在途 trace 上报完成。

    LangSmith 是**异步**上报的（后台线程），进程退出时尚未发出的 trace 会
    直接丢失。表现为"最后几条请求在 LangSmith 里查不到"——这类丢失在排查
    "为什么这次没记录"时极具误导性。

    调用位置：main.py 的 lifespan `finally`，以及命令行脚本的 `finally`。
    **失败不抛**：追踪是观测能力，它的收尾失败不该改变进程退出码。
    """
    if not _enabled:
        return
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
        logger.debug("LangSmith trace 已全部发出")
    except Exception:  # noqa: BLE001 - 见 docstring：观测层失败不影响退出
        logger.warning("等待 LangSmith trace 上报超时或失败（不影响业务）", exc_info=True)


def is_enabled() -> bool:
    """追踪当前是否启用（供自检与日志读取，不要用它做业务分支）。"""
    return _enabled


def reset_for_tests() -> None:
    """重置进程标记（**仅供测试**）。

    生产代码不要调用：重置会让后续 configure_tracing() 重新写一遍环境变量，
    而环境变量一旦写进去就删不干净（SDK 已导入时更是如此）。
    """
    global _configured, _enabled
    _configured = False
    _enabled = False


__all__ = [
    "configure_tracing",
    "flush_tracing",
    "is_enabled",
    "reset_for_tests",
    "traced_run",
    "tracing_scope",
]
