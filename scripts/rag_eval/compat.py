# -*- coding: utf-8 -*-
"""RAGAS 0.4.3 与本机依赖栈之间的兼容垫片。

问题（实测）：
    `ragas/llms/base.py` 顶层执行
        from langchain_community.chat_models.vertexai import ChatVertexAI
    而 langchain-community 0.4 已把 Vertex 的聊天模型移出（该目录只剩
    `llms/vertexai.py`），于是 `import ragas` 直接
    `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`。

为什么不按 ragas 的解析结果装依赖：
    pip 会把 openai 3.28 降到 1.109、langchain-openai 1.7 降到 1.1.9 才能满足
    instructor 的约束。那是**评估工具**倒逼**生产链路**降级，代价远大于收益：
    `app/infrastructure/llm.py` 走的正是 langchain-openai。

为什么不改 site-packages：
    换机器、重建 venv 就丢，属于"本机能跑"的隐性依赖。

所以在这里注入一个空壳模块。安全性依据：`ChatVertexAI` 在 ragas 里只出现在
`MULTIPLE_COMPLETION_SUPPORTED` 这个 isinstance 候选列表中（用于判断后端是否
支持 n>1），本项目的裁判模型走 OpenAI 兼容协议，永远不会构造它。

已知代价：垫片只掩盖了"模块不存在"这一件事。若将来 ragas 真的要用 Vertex
能力，这里会以 AttributeError 而不是静默错误暴露——那正是期望的行为。
"""

from __future__ import annotations

import sys
import types

_VERTEX_CHAT_MODULE = "langchain_community.chat_models.vertexai"

_installed = False


def _build_stub() -> types.ModuleType:
    """构造空壳模块。实例化即报错，避免被当成可用实现误用。"""
    module = types.ModuleType(_VERTEX_CHAT_MODULE)

    class ChatVertexAI:  # noqa: D401 - 仅占位
        """占位类：只用于 isinstance 判断，不可实例化。"""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError(
                "ChatVertexAI 是兼容垫片里的占位类；本项目走 OpenAI 兼容协议，"
                "不应构造它。若确实需要 Vertex，请安装 langchain-google-vertexai "
                "并移除 scripts/rag_eval/compat.py 的垫片。"
            )

    module.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
    return module


def ensure_ragas_importable() -> None:
    """在 `import ragas` **之前**调用；幂等，可重复调用。"""
    global _installed
    if _installed:
        return

    try:
        __import__(_VERTEX_CHAT_MODULE)
    except ModuleNotFoundError:
        stub = _build_stub()
        sys.modules[_VERTEX_CHAT_MODULE] = stub
        # 让 `from langchain_community.chat_models import vertexai` 这种写法
        # 也能命中同一个对象，避免出现"两次导入得到两个模块"。
        package = sys.modules.get("langchain_community.chat_models")
        if package is None:
            package = __import__("langchain_community.chat_models", fromlist=["*"])
        setattr(package, "vertexai", stub)

    _installed = True
