# -*- coding: utf-8 -*-
"""文档解析适配器：LangChain ``Document`` → 领域 ``Page`` 列表。

实现目标端口：app.domain.ports.DocumentParserPort

分工（与 app/infrastructure/loading.py 的边界）：
    loading.py   磁盘文件 → ``list[Document]``   —— 框架相关，用 LangChain Loader
    parsing.py   ``list[Document]`` → ``list[Page]`` —— 框架无关的纯映射

    所以本文件里不该出现 pypdf / docx2txt / python-pptx 的任何调用；
    也不该出现文件 IO（除了把路径交给加载层）。
    这样页码规则可以脱离磁盘与模型直接单测：
    ``document_to_page(Document(page_content="x", metadata={"page": 0}))``。

页码规则（唯一的语义转换点）：

    | metadata 里的 page | 来源                    | 映射结果        |
    |--------------------|-------------------------|-----------------|
    | 0 起整数           | PyPDFLoader / PptxLoader | ``page + 1``    |
    | 缺失               | Docx2txtLoader / TextLoader | 恒为第 1 页  |
    | 存在但无法解析     | 少数 loader / 异常 PDF   | ``None``（未知） |

    LangChain 的 loader 用 **0 起**的 ``page``（对齐数组下标），
    而面向用户的页面显示与引用跳转都是 **1 起**，转换只在这里做一次。

为什么「无法解析」不再回退第 1 页（修过的 bug）：
    旧实现是 ``try: int(raw) + 1 except: 1``。于是 ``page="3 of 10"``
    （pypdf 某些版本对带页码标签的 PDF 会吐这种值）或 ``page="iv"``
    （罗马数字）会被**静默**当成第 1 页——引用溯源于是显示「第1页」，
    而那段正文其实在第 3 页。**编造一个错误页码比承认不知道更糟**：
    用户按页码去翻会翻空，然后失去对整个溯源的信任。
    现在的规则是：
        1. ``page`` 能当整数读 → ``+1``（0 起契约不变）；
        2. 读不出来 → 再看 ``page_label``（它是 **1 起**的字符串，
           取开头的十进制数字，``"3 of 10"`` → 3、``"P3"`` → 3）；
        3. 两者都读不出来 → ``None``，接口里页码为 null，前端不渲染页码标签。
    ``page`` 缺失（txt/md/docx 没有页概念）仍然记为第 1 页——那是如实
    「整份文档就是一个定位单位」，不是编造。

关于页码的诚实说明：
    - PDF / PPTX 有天然的页边界（page / slide），页码是**准确**的。
    - TXT / MD / DOCX 是流式文本，没有"页"的概念：
      TXT/MD 整体记为第 1 页；DOCX 不做"按固定行数估算页码"——
      那是**编造**的页码，会让引用溯源显示假信息，比诚实地写"第 1 页"更糟。
      需要精确页码时应先用 Office/LibreOffice 转 PDF 再上传。
    - 因此 page_start / page_end 在本项目里的语义是
      「可溯源的最小定位单位」，不是排版意义上的页。

关于 .doc：
    旧版二进制 .doc **不在**支持列表中。它的文本抽取依赖
    ``unstructured``（要额外 25 个包）或 Word COM 接口（只在装了 Office 的
    Windows 上可用），两条路都会让依赖清单和部署方式变脏。
    直接在扩展名白名单外拒绝更清楚：用户看到的提示是
    「不支持的扩展名，请另存为 .docx」。

错误语义由上一层的 loading.py 保证：
    空文件、编码异常、加密文件、损坏文件、扫描件无文本
    → 抛 AppError（或 ValidationError），**绝不返回空列表**。
    空列表会被上层当作"文档没有文本"从而掩盖真正的问题。
"""

from __future__ import annotations

import logging
import re

from langchain_core.documents import Document

from app.domain.models import Page
from app.infrastructure.loading import LangChainDocumentLoader

logger = logging.getLogger(__name__)

# 从「3」「3 of 10」「P3」「Page 3」这类页码标签里取第一个十进制数字。
# 只认阿拉伯数字：罗马数字（i/ii/iv）在语义上无法映射到物理页序，
# 硬猜一个数字出来就又回到「编造页码」的老路上。
_LEADING_INT = re.compile(r"(\d+)")


def _as_int(value: object) -> int | None:
    """把「像整数」的值收敛成 int，不像就返回 None。

    接受的形态（都是在真实 loader 元数据里见过的）：
        int / 形如 3.0 的 float / 纯数字字符串 "3"（前后可有空白）
    拒绝：bool（`True` 是 int 子类，读成页码纯属事故）、"3 of 10"、
    "iv"、2.5 这种非整数 float——它们要么该走 page_label，要么就是坏数据。
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("+-").isdigit():
            return int(text)
    return None


def _from_label(value: object) -> int | None:
    """从 page_label 里取页码（**1 起**，不再 +1）。"""
    if not isinstance(value, str):
        return None
    match = _LEADING_INT.search(value)
    return int(match.group(1)) if match else None


def parse_page_no(metadata: dict) -> int | None:
    """按模块文档的三条规则确定 1 起的页码；无法判定返回 None。"""
    if "page" in metadata:
        raw = metadata.get("page")
        number = _as_int(raw)
        if number is not None:
            page_no = number + 1
            # 负页码（某些 loader 用 -1 表示"前置页"）没有对应的展示语义，
            # 直接当未知处理，避免出现「第0页」「第-1页」这种荒谬标签。
            return page_no if page_no >= 1 else None
        if raw is None:
            # 显式的 None 与"缺失"同义：视为该格式没有页概念
            return _from_label(metadata.get("page_label")) or 1
        labelled = _from_label(metadata.get("page_label"))
        if labelled is not None:
            return labelled
        logger.warning(
            "页码元数据无法解析（page=%r page_label=%r），按「未知页码」处理——"
            "不猜数字，避免给引用溯源标错页",
            raw,
            metadata.get("page_label"),
        )
        return None

    labelled = _from_label(metadata.get("page_label"))
    if labelled is not None:
        return labelled
    # 完全没有页概念（txt/md/docx）：整份文档就是第 1 页
    return 1


def document_to_page(document: Document) -> Page:
    """把 LangChain Document 映射成领域 Page。

    Args:
        document: 加载层产出的文档对象

    Returns:
        Page；``text`` 已去掉首尾空白，``page_no`` 为 1 起，
        无法判定时为 ``None``（未知，而不是猜一个）。

    注意：
        ``page`` 可能是 ``None``、字符串、浮点或缺失（不同 Loader 不一致），
        规则集中在 ``parse_page_no`` 里，并附有一张判定表。
    """
    return Page(
        page_no=parse_page_no(document.metadata),
        text=document.page_content.strip(),
    )


class CompositeDocumentParser:
    """文档解析器：加载 + 映射，实现 DocumentParserPort 协议。

    名字沿用骨架里的 ``Composite``（按扩展名分发到多个实现），
    分发逻辑现在归加载层，本类只负责把结果映射成领域模型。
    """

    def __init__(self, loader: LangChainDocumentLoader | None = None) -> None:
        """注入加载器。

        Args:
            loader: 加载器；None 时用默认的 LangChain 实现。
                留出这个参数是为了让上游能替换加载策略（如换成
                从对象存储拉取的实现）而不改本类。
        """
        self._loader = loader or LangChainDocumentLoader()

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """全部受支持的扩展名（小写带点）。"""
        return self._loader.supported_extensions

    def parse(self, file_path: str) -> list[Page]:
        """加载并解析文件为页列表。

        Args:
            file_path: 本地文件绝对路径

        Returns:
            Page 列表；加载不到文本时抛错而不是返回空列表。

        Raises:
            ValidationError: 扩展名不在受支持列表中
            AppError: 文件不存在、内容为空、加载出空文本或加载过程异常
        """
        documents = self._loader.load(file_path)
        pages = [document_to_page(document) for document in documents]
        logger.debug("解析完成：%s → %s 页", file_path, len(pages))
        return pages
