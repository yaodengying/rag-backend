# -*- coding: utf-8 -*-
"""大模型适配器：OpenAI 兼容协议（默认指向本机 Ollama /v1）。

实现目标端口：app.domain.ports.LLMPort

为什么用 OpenAI 兼容协议而不是厂商 SDK：
    Ollama、vLLM、LM Studio、以及各家云端服务都提供 /v1/chat/completions。
    统一走这个协议，换后端只改 LLM_BASE_URL 与 LLM_MODEL，代码零改动，
    也不会被某个厂商 SDK 的版本节奏绑架。

为什么用 langchain-openai 的 ChatOpenAI 而不是裸 httpx：
    ChatOpenAI 是 LangChain 的 Runnable，天然能被 LangGraph 节点持有、
    被回调系统观测（将来要做 token 级 SSE 或 LangSmith 追踪时不用重写）。
    代价是它接受**消息对象**而不是裸字符串，所以在 `_messages()` 里
    统一做一次转换，业务侧仍然只传两个字符串。

提示词约束（防幻觉，核心业务约束）：
    提示词本身在 `app/rag/prompts.py`，本模块只负责"把两条消息发出去"。
    这样调整话术不用碰传输层，替换后端也不用碰提示词。

错误语义（重要，别退化成裸异常）：
    连接拒绝 / 鉴权失败 / 响应格式异常 → `UpstreamError`（502）；
    **超时单独一类** → `UpstreamTimeoutError`（504）。
    理由：502 的动作是"去启动/重启上游"，504 的动作是"调大 LLM_TIMEOUT、
    换更小的模型、或把问题拆短"——把它们混成一个 502，运维只会去重启
    Ollama，而 Ollama 明明在跑、只是这题太长。客户端也可以安全地对 504
    重试（超时的请求通常没被处理完），对 502 则未必。

    捕获时用 `except Exception` 而不是细分异常类型——openai SDK 的异常层次
    跨版本变动过，而这里的兜底语义本来就是"上游出问题了"，与具体异常类无关。
    超时靠**类名/消息特征**识别（`TimeoutError` / `APITimeoutError` /
    `httpx.ReadTimeout` / "timed out"），不 import httpx 的异常类，
    避免把自己绑死在传输库版本上。

    **但 `AppError` 必须先原样放行**：`_build_client()` 在配置缺失时抛
    `ConfigurationError`，若被兜底包成 UpstreamError，运维会去重启 Ollama，
    而真正该改的是 .env 里的 LLM_BASE_URL。这条是自检脚本抓出来的真 bug
    （见 scripts/check_chat.py 第 8 组）。

空回复视为上游异常：
    模型返回空串不是"没有答案"，而是异常（截断、内容过滤、后端故障）。
    返回空串会让上层生成一条空回答，用户看到的是"页面卡住了"；
    显式抛 UpstreamError 才能在日志和响应里说清到底发生了什么。

本地 Ollama 注意事项：
    - 需先 `ollama serve` 并 `ollama pull <model>`；
    - api_key 任意非空字符串即可（见 Settings.llm_api_key_or_placeholder）；
    - 若本机有全局 http_proxy，实测 httpx 会把 127.0.0.1 的连接交给本机代理
      再转发回来，可以通；但把 localhost 加进 NO_PROXY 更稳妥
      （.env 里已有这一项）。
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import (
    AppError,
    ConfigurationError,
    UpstreamError,
    UpstreamTimeoutError,
)

logger = logging.getLogger(__name__)

# 超时的识别特征（类名小写 / 消息小写）。
# 不 import httpx / openai 的异常类：那会把适配层绑死在传输库版本上，
# 而"是不是超时"这件事靠特征判断已经足够可靠。
_TIMEOUT_HINTS: tuple[str, ...] = (
    "timeout",
    "timedout",
    "timed out",
    "time out",
)

# 不走代理的主机。本机后端（127.0.0.1:11434）必须放进 NO_PROXY，
# 否则全局代理会参与握手，表现为随机超时或 502。
#
# 只**追加**到已有的 NO_PROXY，不覆盖、更不删除 HTTP_PROXY：
# 删除代理变量会让同一进程里其他出网请求（如 HuggingFace 下载）静默失去代理，
# 那是一种"改一个模块、坏另一个模块"的副作用。云端端点不受影响——
# NO_PROXY 只列出本机地址。
_NO_PROXY_HOSTS = ("localhost", "127.0.0.1", "::1")


class OpenAICompatibleLLM:
    """OpenAI 兼容协议大模型客户端。

    实现 LLMPort 协议。

    延迟构造底层 client：`get_llm()` 被容器 lru_cache 持有，
    如果构造期就建 client，进程启动时会多一次无谓的 SDK 初始化；
    更重要的是——**服务启动不应依赖 Ollama 是否在跑**。
    把连接动作推迟到第一次生成，启动期就不会因为模型服务没起而失败。
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
        client: Any = None,
    ) -> None:
        """记录配置，不建立连接。

        Args:
            base_url: 形如 http://127.0.0.1:11434/v1
            model: 模型名，如 qwen2:7b
            api_key: 本机 Ollama 传任意非空值
            temperature: 采样温度，事实型问答建议 0.1~0.3
            timeout: 单次请求超时秒数
            client: **测试注入点**——传入实现了 `.invoke(messages)` /
                `.astream(messages)` 的假 client，即可在不起 Ollama 的情况下
                验证消息组装、内容抽取与错误映射（见 scripts/check_chat.py）。
                生产代码不要传这个参数。
        """
        settings = get_settings()
        self._base_url = (base_url or settings.llm_base_url or "").strip().rstrip("/")
        self._model = (model or settings.llm_model or "").strip()
        self._api_key = api_key or settings.llm_api_key_or_placeholder
        self._temperature = (
            float(settings.llm_temperature) if temperature is None else float(temperature)
        )
        self._timeout = int(settings.llm_timeout if timeout is None else timeout)
        self._client = client

    # ---------- 只读属性（供自检与日志读取）----------

    @property
    def model(self) -> str:
        """当前模型名。"""
        return self._model

    @property
    def base_url(self) -> str:
        """当前端点。"""
        return self._base_url

    # ---------- LLMPort ----------

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """一次性生成完整回答。

        Args:
            system_prompt: 系统提示（含防幻觉约束与拒答话术）
            user_prompt: 用户提示（参考资料 + 问题）

        Returns:
            模型输出的完整文本（已去掉首尾空白）。

        Raises:
            ConfigurationError: 未配置 LLM_BASE_URL / LLM_MODEL
            UpstreamError: 上游不可用、超时、响应异常或返回空内容
        """
        messages = self._messages(system_prompt, user_prompt)
        try:
            response = self._build_client().invoke(messages)
        except AppError:
            # ConfigurationError 等"我们自己配错了"必须原样抛出。
            # 被下面的兜底包成 UpstreamError 会让运维去重启 Ollama，
            # 而真正的问题是 .env 里少了一行——这是自检脚本抓到的真 bug。
            raise
        except Exception as exc:  # noqa: BLE001 - 见模块文档「错误语义」
            raise self._translate(exc, streaming=False) from exc

        return self._extract(response)

    async def stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        """流式生成，逐段产出文本增量。

        Yields:
            str: 每次增量文本片段（可能为空串，调用方需自行跳过）。

        Raises:
            ConfigurationError: 未配置 LLM_BASE_URL / LLM_MODEL
            UpstreamError: 上游不可用或中途断开
        """
        messages = self._messages(system_prompt, user_prompt)
        try:
            async for chunk in self._build_client().astream(messages):
                delta = self._chunk_text(chunk)
                if delta:
                    yield delta
        except AppError:
            # 配置错误不能被下面的兜底改写成 UpstreamError（同 generate）
            raise
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc, streaming=True) from exc

    # ---------- 内部 ----------

    @staticmethod
    def _is_timeout(exc: Exception) -> bool:
        """判断异常是否是超时。

        同时看类名与消息：`httpx.ReadTimeout`、`openai.APITimeoutError`、
        内置 `TimeoutError` 的类名都含 timeout；但有些后端（如 vLLM 反代）
        返回的是普通 `Exception("request timed out")`，只能靠消息认。
        """
        if isinstance(exc, TimeoutError):  # 3.11 起 asyncio.TimeoutError 即此类
            return True
        name = type(exc).__name__.lower()
        if any(hint.replace(" ", "") in name for hint in _TIMEOUT_HINTS):
            return True
        text = str(exc).lower()
        return any(hint in text for hint in _TIMEOUT_HINTS)

    def _translate(self, exc: Exception, *, streaming: bool) -> UpstreamError:
        """把上游异常翻译成带语义的 AppError（504 超时 / 502 其他）。"""
        stage = "流式调用" if streaming else "调用"
        if self._is_timeout(exc):
            return UpstreamTimeoutError(
                f"{stage}大模型超时：{self._timeout}s 内未完成"
                f"（{self._base_url} / {self._model}）。"
                "本机 7B 在 CPU/低显存下生成较慢，可调大 LLM_TIMEOUT、"
                "减小 RERANK_TOP_N（上下文更短）或换更小的模型",
                detail={
                    "base_url": self._base_url,
                    "model": self._model,
                    "timeout_s": self._timeout,
                    "streaming": streaming,
                    "exception": type(exc).__name__,
                },
            )
        return UpstreamError(
            f"{stage}大模型失败（{self._base_url} / {self._model}）：{exc}",
            detail={
                "base_url": self._base_url,
                "model": self._model,
                "streaming": streaming,
                "exception": type(exc).__name__,
            },
        )

    @staticmethod
    def _allow_localhost_bypass() -> None:
        """把本机地址并入 NO_PROXY（幂等，保留原有条目）。"""
        current = os.environ.get("NO_PROXY", "") or os.environ.get("no_proxy", "")
        entries = [item.strip() for item in current.split(",") if item.strip()]
        for host in _NO_PROXY_HOSTS:
            if host not in entries:
                entries.append(host)
        merged = ",".join(entries)
        os.environ["NO_PROXY"] = merged
        os.environ["no_proxy"] = merged

    def _build_client(self):
        """懒构造 ChatOpenAI（首次调用时建立，之后复用）。"""
        if self._client is not None:
            return self._client

        if not self._base_url:
            raise ConfigurationError(
                "LLM_BASE_URL 未配置；请在 .env 中指向 OpenAI 兼容端点，"
                "例如 http://127.0.0.1:11434/v1"
            )
        if not self._model:
            raise ConfigurationError("LLM_MODEL 未配置；例如 qwen2:7b")

        # 本机后端不该走代理。必须在 import langchain_openai 之前设置：
        # httpx 的代理配置在 ChatOpenAI 构造内部建 client 时读取。
        self._allow_localhost_bypass()

        from langchain_openai import ChatOpenAI

        self._client = ChatOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            model=self._model,
            temperature=self._temperature,
            timeout=self._timeout,
            # 重试交给上层：LLM 调用失败通常是上游没起来或模型被卸载，
            # SDK 层静默重试 3 次只会让用户多等一分钟才看到 502。
            max_retries=0,
        )
        logger.info(
            "LLM 客户端就绪：%s（model=%s temperature=%s timeout=%ss）",
            self._base_url,
            self._model,
            self._temperature,
            self._timeout,
        )
        return self._client

    @staticmethod
    def _messages(system_prompt: str, user_prompt: str) -> list:
        """把两条字符串转成 ChatOpenAI 需要的消息对象列表。

        系统提示为空时不构造 SystemMessage：有些后端对空 system 消息
        会报 400，而"没有系统提示"是完全合法的用法。
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages: list = []
        if (system_prompt or "").strip():
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))
        return messages

    @staticmethod
    def _extract(response: Any) -> str:
        """从响应对象里取出文本内容。

        注意 `content` 的类型在不同后端下不一样：
            - 大多数返回纯字符串；
            - 部分多模态后端返回 [{"type": "text", "text": ...}, ...]。
        统一在这里收敛，上层只面对 str。

        Raises:
            UpstreamError: 内容为空（见模块文档「空回复视为上游异常」）
        """
        content = getattr(response, "content", response)
        if isinstance(content, list):
            # 列表形态：只取 text 分片，忽略图片等非文本块
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            content = "".join(parts)

        text = str(content or "").strip()
        if not text:
            raise UpstreamError(
                "大模型返回了空内容。常见原因：请求被内容过滤、达到输出上限后被截断、"
                "或后端模型被卸载。请检查 Ollama 日志与 LLM_TIMEOUT 设置。"
            )
        return text

    @staticmethod
    def _chunk_text(chunk: Any) -> str:
        """从流式分片里取出增量文本（空分片返回空串，不报错）。"""
        content = getattr(chunk, "content", chunk)
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts)
        return str(content or "")
