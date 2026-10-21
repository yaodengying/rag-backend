"""文本切分适配器：基于 langchain-text-splitters 的递归字符切分。

实现目标端口：app.domain.ports.SplitterPort

为什么用 LangChain 的切分器而不是自己写：
    递归字符切分的难点不在"按长度切断"，而在**切断优先级**——
    先段落、再句子、最后兜底，且要保证相邻块有重叠。
    LangChain 的 ``RecursiveCharacterTextSplitter`` 已经把这件事做对了，
    而且 ``create_documents(texts, metadatas)`` 会把**每页的元数据原样复制到
    该页切出的每一块上**，页码溯源因此由库负责，我们不需要自己算字符偏移。

本项目的做法（LangChain 原生路径）：
    ``Page``（1 起页码）→ 每个非空页作为一条独立文本 + ``{"page_no": n}`` 元数据
    → ``create_documents()`` → 每块自带 ``page_no`` → 映射成领域 ``Chunk``。

为什么**不**把所有页拼成一整串再切（曾经的做法，已废弃）：
    拼接后统一的字符区间需要手工反查"这块落在哪几页"（区间相交），
    逻辑绕、易错，且会产生**跨页块**——一块同时属于第 1、2 页，
    引用溯源只能显示"第1-2页"，精度反而下降。
    按页独立切分后，每块**必然且只属于一页**，``page_start == page_end``，
    引用能精确到页；也天然避免上一页末句与下一页首句被凑成同一个块。

    代价：极短的页（如 PPT 单页只有几十字）不会被邻居"拼大"，
    而是各自成为一个小块。对本项目是**可接受甚至更好**的取舍——
    幻灯片本就是独立语义单元；而项目实测的调优结论是块宁小勿大
    （中文块过大会同时稀释向量相似度与重排得分）。

关键点：
    1. 切分单位是**字符**，不是 token。中文一个汉字约等于一个字符，
       所以一块 300 字符约对应 300 个汉字。
       中文场景 ``CHUNK_SIZE`` 建议 150~300。
    2. 页码溯源由 LangChain 的元数据传递保证，见上。
    3. separators 要兼顾中英文标点与换行，优先在语义边界切断。
    4. 空页（``text`` 只有空白）必须剔除：留着会切出空块，
       白白占一条 Milvus 记录，还会把 ``chunk_index`` 打出空洞。

实测注意（langchain-text-splitters 1.1.3，勿凭印象改）：
    - ``strip_whitespace`` 默认是 **True**，且是**在分隔符处**生效：
      ``'AAA\\n\\nBBB'`` 会切成 ``['AAA', 'BBB']``，那个 ``\\n\\n`` 谁都不属于、
      直接消失。所以**块文本不能按字节还原原文**——拼接所有块 ≠ 原文，
      差值就是这些边界空白。这对语义与页码溯源都没有影响，
      但任何"逐字相等"的断言都必须先做空白归一化
      （见 ``scripts/check_splitting.py`` 的 ``compact()``）。
    - 因此**不传** ``add_start_index=True``：既然块文本已被归一化，
      那个"块在原文中的字符偏移"拿去做精确切片会对不上，
      留着一个用不上的、还会误导人的元数据键不如不要。
"""

from __future__ import annotations

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.domain.models import Chunk, Page

logger = logging.getLogger(__name__)

# 中英文友好的分隔符优先级：先按段落，再按句子，最后兜底按字符
DEFAULT_SEPARATORS: list[str] = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "…",
    ". ",
    "! ",
    "? ",
    "; ",
    "，",
    ", ",
    " ",
    "",
]

# 传给 LangChain 的页码元数据键。
# 用 ``page_no`` 而不是 ``page``：后者的语义在 LangChain 各 Loader 里是
# **0 起**（对齐数组下标），而本项目对外一律 1 起，同名会被误读。
_PAGE_KEY = "page_no"


class LangChainSplitter:
    """递归字符切分器，页码溯源交给 LangChain 的元数据传递。

    实现 SplitterPort 协议。
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ) -> None:
        """构造切分器。

        Args:
            chunk_size: 单块目标字符数；None 时取 settings.chunk_size
            chunk_overlap: 相邻块重叠字符数；None 时取 settings.chunk_overlap
            separators: 分隔符优先级列表；None 时用中英文友好的默认值
        """
        settings = get_settings()
        self._chunk_size = int(chunk_size if chunk_size is not None else settings.chunk_size)
        self._chunk_overlap = int(
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        )
        if self._chunk_overlap >= self._chunk_size:
            # 配置校验会拦，但这里再兜一层：构造器可能被直接调用
            raise ValueError(
                f"chunk_overlap({self._chunk_overlap}) 必须小于 chunk_size({self._chunk_size})"
            )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=separators or DEFAULT_SEPARATORS,
        )

    @property
    def chunk_size(self) -> int:
        """当前块大小（字符）。"""
        return self._chunk_size

    def split(self, pages: list[Page]) -> list[Chunk]:
        """切分并回填页码。

        Args:
            pages: 解析得到的页列表，page_no 为 1-based

        Returns:
            Chunk 列表；chunk_index 已按原文顺序填充（连续无空洞）。
            embedding 字段此处保持 None，由 Embedder 后续填充。

        注意：
            本方法只负责文本层面，id / kb_id / doc_id 由调用方（ingest 流程）赋值。
        """
        if not pages:
            return []

        texts, metadatas = self._as_langchain_inputs(pages)
        if not texts:
            logger.debug("全部页都没有可用文本，切分结果为空")
            return []

        # LangChain 原生路径：文本与页码元数据一起交给切分器，
        # 切出的每一块都自动带上所属页的页码（见模块 docstring）。
        documents = self._splitter.create_documents(texts, metadatas=metadatas)

        chunks: list[Chunk] = []
        for document in documents:
            text = document.page_content
            if not text.strip():
                continue  # 理论上不会出现，兜一层以防切分器行为变化
            # 页码可能是 None（解析层判不出来）：那就让 page_start 保持 None，
            # 前端据此不渲染页码。**不要**用 `or 1` 兜底——那会把
            # 「不知道」变成「第1页」，正是要修的那类静默错误。
            page_no = document.metadata.get(_PAGE_KEY)
            page = int(page_no) if isinstance(page_no, int) and not isinstance(page_no, bool) else None
            chunks.append(
                Chunk(
                    id="",
                    kb_id="",
                    doc_id="",
                    text=text,
                    chunk_index=len(chunks),
                    page_start=page,
                    page_end=page,
                )
            )
        logger.debug(
            "切分完成：%s 页 → %s 块（块大小 %s，重叠 %s）",
            len(pages),
            len(chunks),
            self._chunk_size,
            self._chunk_overlap,
        )
        return chunks

    @staticmethod
    def _as_langchain_inputs(pages: list[Page]) -> tuple[list[str], list[dict]]:
        """把页列表转成 LangChain 需要的「文本列表 + 元数据列表」。

        空页在这里剔除，保证两者**长度一致且一一对应**——
        ``create_documents`` 对长度不一致只会抛一个难懂的 ``IndexError``。

        Returns:
            (每页文本, 每页元数据)；全为空页时返回两个空列表。
        """
        texts: list[str] = []
        metadatas: list[dict] = []
        for page in pages:
            text = (page.text or "").strip()
            if not text:
                continue
            texts.append(text)
            # 页码为 None 时照样写进元数据（值就是 None）：LangChain 会原样
            # 复制到该页的每一块，下游据此判"未知"，不会与"第 1 页"混淆。
            metadatas.append({_PAGE_KEY: page.page_no})
        return texts, metadatas
