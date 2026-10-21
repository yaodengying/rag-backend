# -*- coding: utf-8 -*-
"""文档加载适配器：磁盘文件 → LangChain ``Document`` 列表。

对应数据流里的**第 3 步：加载**（见 docs/data-flow.md 链路一）。
上游是「原始文件落盘」，下游是 parsing.py 的「Document → 领域 Page 映射」。

为什么单独一层，而不是继续塞在 parsing.py 里：
    加载是**框架相关**的——PDF/DOCX/TXT/MD 用 langchain-community 现成的 Loader，
    PPTX 用 langchain-core 的 BaseLoader 自定义。换框架、加格式只动本文件。
    而「Document 的 metadata → 领域 Page」是**框架无关**的纯映射，
    留在 parsing.py 后可以脱离 LangChain、脱离磁盘直接单测。
    这与 splitting.py（框架）与领域 Chunk（模型）的分工完全一致。

统一入口：
    ``LangChainDocumentLoader().load(path) -> list[Document]``
    只做四件事：按扩展名分发、解决文本编码、拦截空文件/空文本、包装异常。
    **不**做切分，**不**做页码语义转换（0-based → 1-based 属于 parsing.py 的职责）。

各 Loader 的元数据真相（langchain-community 0.4.2 实测，勿凭印象改）：

    | Loader                    | metadata 键                                        | page 语义   |
    |---------------------------|----------------------------------------------------|-------------|
    | PyPDFLoader(mode="page")  | source / page / page_label / total_pages / ...     | **0 起整数** |
    | Docx2txtLoader            | source                                             | 无          |
    | TextLoader                | source                                             | 无          |
    | PptxLoader（本文件）       | source / page / total_pages                        | 0 起（对齐） |

    注意 PyPDFLoader 还有 ``page_label``（1 起的**字符串**，尊重 PDF 自身的页码标签）。
    本项目的 ``Page.page_no`` 是 int，所以取 ``page + 1``，不取 ``page_label``——
    否则遇到罗马数字页码（i、ii、iii）会塞不进 int 字段。

三个必须自己兜的点（都是实测踩出来的，别删）：

    1. **文本编码不能交给 TextLoader 自动判定**，两条路都不通：
       - ``encoding=None`` 用的是**系统默认编码**，结果随 ``-X utf8`` /
         ``PYTHONUTF8`` / 系统区域设置漂移。同一份 GB18030 文件，
         加不加 ``-X utf8`` 得到的文本不一样，一条能读一条直接报错。
       - ``autodetect_encoding=True`` 在 chardet>=5.2 上会直接抛
         ``TypeError: FileEncoding.__new__() got an unexpected keyword argument 'mime_type'``
         （chardet 的 ``detect_all()`` 返回了 langchain 的 NamedTuple 未声明的键）。
       所以这里用固定候选表逐个试解，命中后把 encoding **显式**传给 TextLoader。
       候选里 ``utf-8-sig`` 排第一：它同时覆盖「有 BOM」和「无 BOM」的 UTF-8。

    2. **空文件 / 文件不存在**在各 Loader 上表现不一致：
       pypdf 抛 ``EmptyFileError``，BasePDFLoader 对不存在的路径抛 ``ValueError``，
       TextLoader 则对空文件安安静静返回一段空串。
       统一在 ``_ensure_readable()`` 里前置拦截，报错文案才跟格式无关。

    3. **加载出空文本不算成功**：``load()`` 不报错不等于读到了内容。
       若放任空文本继续走，用户看到的是「上传成功但 0 个片段」，
       真正的原因（扫描件、纯图片 PPT）被掩盖。
       因此最后一定要过 ``_ensure_not_blank()``。

关于 PPTX 为什么自己写 Loader：
    ``UnstructuredPowerPointLoader`` 依赖 ``unstructured``，会连带拉进
    nltk / numba / llvmlite / langdetect / python-magic 等 25 个包，
    在 Python 3.14 上有兼容风险；而它相对 python-pptx 并不提供更准确的页码
    （两者都是「每张幻灯片一页」）。
    所以这里用 ``langchain_core.document_loaders.BaseLoader`` 实现一个加载器：
    ``load()`` / ``lazy_load()`` / ``aload()`` 全部由 BaseLoader 提供，行为与内置 Loader 一致。
    将来若要做元素级（标题/正文/表格）切分，再换成 UnstructuredPowerPointLoader。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterator

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from app.core.exceptions import (
    CorruptedDocumentError,
    DocumentError,
    EmptyDocumentError,
    EncryptedDocumentError,
    NotFoundError,
    ValidationError,
)

logger = logging.getLogger(__name__)

# .txt / .md 的解码候选，按顺序尝试；全部失败才降级到 errors="replace"
#
# 顺序的取舍：utf-8-sig → utf-8 优先，保证现代文件零歧义；
# gb18030 是 GBK/GB2312 的超集，放在 big5 前面（大陆环境更常见）。
# 已知局限：gb18030 几乎能解码任意字节序列，所以**繁体 Big5 文件可能被
# 误判成 gb18030 并解出乱码，而不是抛错**。要根治得引入字符集探测，
# 但那会带来新的不确定性，当前保持确定性优先；需要时把候选顺序调成 big5 在前。
TEXT_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "gb18030", "big5")


def _resolve_text_encoding(path: Path) -> str | None:
    """逐个候选编码试解整个文件，返回第一个不报错的编码。

    Args:
        path: 文本文件路径

    Returns:
        可完整解码该文件的编码名；全部失败返回 None。
    """
    raw = path.read_bytes()
    for encoding in TEXT_ENCODINGS:
        try:
            raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        return encoding
    return None


class PptxLoader(BaseLoader):
    """PPTX 加载器：一张幻灯片一个 ``Document``（页码 0 起，与 PyPDFLoader 对齐）。

    实现 ``lazy_load()`` 即可，``load()`` / ``aload()`` / ``load_and_split()``
    由 ``BaseLoader`` 提供。

    注意：只取幻灯片正文的可见形状文本与表格文本，**不取演讲者备注**。
    备注常是讲稿，混进来会稀释正文向量；需要时应显式加开关再取。
    """

    def __init__(self, file_path: str) -> None:
        """记录文件路径。

        Args:
            file_path: PPTX 文件路径
        """
        self.file_path = str(file_path)

    def lazy_load(self) -> Iterator[Document]:
        """逐张幻灯片产出 Document。"""
        from pptx import Presentation

        presentation = Presentation(self.file_path)
        total_pages = len(presentation.slides)
        for index, slide in enumerate(presentation.slides):
            pieces: list[str] = []
            for shape in slide.shapes:
                text = getattr(shape, "text", None)
                if text and text.strip():
                    pieces.append(text.strip())
                # 表格文本不在 shape.text 里，必须单独取，否则参数表会整段丢失
                if getattr(shape, "has_table", False):
                    for row in shape.table.rows:
                        cells = [cell.text.strip() for cell in row.cells]
                        if any(cells):
                            pieces.append(" | ".join(cells))
            yield Document(
                page_content="\n".join(pieces),
                metadata={
                    "source": self.file_path,
                    "page": index,
                    "total_pages": total_pages,
                },
            )


# ---------------------------------------------------------------------------
# 各格式的加载函数：统一签名 (path) -> list[Document]
# ---------------------------------------------------------------------------


def _load_pdf(path: str) -> list[Document]:
    """PDF：PyPDFLoader 逐页加载。

    ``mode="page"`` 是显式写出来的——虽然当前默认值就是它，
    但默认值可能随版本变（``mode="single"`` 会把整份文档合成一个 Document，
    页码溯源会整体失效），不能依赖默认。
    """
    return PyPDFLoader(path, mode="page").load()


def _load_docx(path: str) -> list[Document]:
    """DOCX：Docx2txtLoader。

    它把段落与表格单元格都抽出来（实测确认），段落间以空行分隔。
    """
    return Docx2txtLoader(path).load()


def _load_text(path: str) -> list[Document]:
    """TXT / MD：TextLoader，但编码由本模块判定后显式传入。

    Markdown 不做语法解析，当纯文本处理：RAG 要的是原始文字，
    去掉 ``#``、``|`` 只会丢掉结构线索（标题层级、表格边框常常是有用的信号）。
    """
    text_path = Path(path)
    encoding = _resolve_text_encoding(text_path)
    if encoding is None:
        # 所有候选都失败：不静默丢文件，用替换字符保证"能读"并留下警告。
        # 这条分支无法再走 TextLoader（它是严格解码，会抛 UnicodeDecodeError），
        # 所以直接构造 Document —— 仍是 LangChain 的数据模型，不破坏下游契约。
        logger.warning("文本编码无法识别，已按 utf-8 + errors=replace 解码：%s", path)
        text = text_path.read_bytes().decode("utf-8", errors="replace")
        return [Document(page_content=text, metadata={"source": str(path)})]
    return TextLoader(path, encoding=encoding).load()


def _load_pptx(path: str) -> list[Document]:
    """PPTX：本文件自带的 PptxLoader（理由见模块 docstring）。"""
    return PptxLoader(path).load()


# 扩展名 → (展示名, 加载函数)
#
# 这是**唯一**的格式清单：supported_extensions 由它派生，
# 新增格式只需在这里加一行（外加 .env 的 UPLOAD_ALLOWED_EXTENSIONS）。
_LOADERS: dict[str, tuple[str, Callable[[str], list[Document]]]] = {
    ".pdf": ("PDF", _load_pdf),
    ".docx": ("DOCX", _load_docx),
    ".pptx": ("PPTX", _load_pptx),
    ".txt": ("TXT", _load_text),
    ".md": ("Markdown", _load_text),
}

# 加载出空文本时的补充提示，按展示名索引
_BLANK_HINTS: dict[str, str] = {
    "PDF": "；若为扫描件（图片型 PDF），需先做 OCR",
    "PPTX": "；幻灯片里可能只有图片，没有可提取的文字",
}

# 「文件坏了」的异常特征（**类名**，小写比对）。
# 这些都是真实会遇到的：截断的 PDF（download 没下完）、把 .docx 改名为 .pdf、
# 用旧版工具导出的畸形 pip 包结构。按类名认比按消息文本认稳——
# pypdf / python-pptx / zipfile 的文案会随版本变，类名基本不变。
_CORRUPT_CLASS_HINTS: tuple[str, ...] = (
    "badzip",  # zipfile.BadZipFile：docx/pptx 不是合法 zip
    "pdfread",  # pypdf.errors.PdfReadError
    "pdfstream",  # pypdf.errors.PdfStreamError
    "emptyfile",  # pypdf.errors.EmptyFileError（0 字节，正常会先被前置拦截）
    "packagefound",  # docx/pptx: PackageNotFoundError（缺 [Content_Types].xml）
    "unsupported",  # 某些版本的 UnsupportedFormatError
)

# 消息文本特征：补类名认不出的情况（不同库版本抛的类不一样）
_CORRUPT_TEXT_HINTS: tuple[str, ...] = (
    "eof marker",
    "xref",
    "not a zip file",
    "truncated",
    "corrupt",
    "invalid header",
    "unexpected end",
    "stream has ended",
)


class LangChainDocumentLoader:
    """按扩展名分发到 LangChain Loader 的文档加载器。"""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """全部受支持的扩展名（小写带点，已排序）。"""
        return tuple(sorted(_LOADERS))

    def load(self, file_path: str) -> list[Document]:
        """加载磁盘文件为 LangChain Document 列表。

        Args:
            file_path: 本地文件路径

        Returns:
            按文档自身顺序排列的 Document 列表。多页格式（PDF/PPTX）
            每页一个，无页码格式（DOCX/TXT/MD）整体一个。

        Raises:
            ValidationError: 扩展名不在受支持列表中
            NotFoundError: 文件不存在
            DocumentError: 内容为空、加载出空文本、文件损坏/加密或加载过程异常
                （code 分别为 empty_document / corrupted_document /
                encrypted_document / document_error，前端可据此分支）
        """
        path = Path(file_path)
        entry = _LOADERS.get(path.suffix.lower())
        if entry is None:
            supported = "、".join(self.supported_extensions)
            raise ValidationError(
                f"不支持的扩展名：{path.suffix or '(无扩展名)'}；当前支持 {supported}"
            )

        label, factory = entry
        self._ensure_readable(path)

        try:
            documents = factory(str(path))
        except Exception as exc:  # noqa: BLE001 - 兜住一切第三方异常，统一转成 DocumentError
            raise self._wrap(label, exc) from exc

        self._ensure_not_blank(label, documents)
        logger.debug("加载完成：%s %s → %s 个 Document", label, path.name, len(documents))
        return documents

    # ---------- 内部 ----------

    @staticmethod
    def _ensure_readable(path: Path) -> None:
        """前置拦截「文件不存在」与「空文件」，让报错文案与格式无关。"""
        if not path.is_file():
            # 404 而不是 500：这是"你给的路径不存在"，不是服务坏了
            raise NotFoundError(f"文件不存在：{path}")
        if path.stat().st_size == 0:
            raise EmptyDocumentError(
                "文件内容为空（0 字节）；上传可能中断，请重新上传",
                detail={"filename": path.name, "size_bytes": 0},
            )

    @staticmethod
    def _ensure_not_blank(label: str, documents: list[Document]) -> None:
        """确认确实加载到了文本。

        判据是「**至少一个** Document 有非空白文本」，不是「列表非空」：
        PyPDFLoader 对纯图片页会产出 ``page_content=""`` 的 Document，
        列表非空但内容为空。
        """
        if any(document.page_content.strip() for document in documents):
            return
        raise EmptyDocumentError(
            f"{label} 未加载到任何文本{_BLANK_HINTS.get(label, '')}",
            detail={"format": label, "documents": len(documents)},
        )

    @staticmethod
    def _wrap(label: str, exc: Exception) -> DocumentError:
        """把第三方异常翻译成**可分支**的 DocumentError。

        分类的价值在于「下一步做什么」不同：
            加密   → 让用户解除密码保护
            损坏   → 让用户重新导出（常见于下载中断、扩展名被改）
            其他   → 保守地报"加载失败"并带上原始异常名，便于搜issue
        三类都是 400（客户端问题），`code` 字段供前端/CLI 分支。
        """
        class_name = type(exc).__name__.lower()
        text = str(exc).lower()

        if "decrypt" in class_name or "password" in text or "encrypted" in text:
            return EncryptedDocumentError(
                f"{label} 已加密，请先解除密码保护",
                detail={"format": label, "exception": type(exc).__name__},
            )

        if any(hint in class_name for hint in _CORRUPT_CLASS_HINTS) or any(
            hint in text for hint in _CORRUPT_TEXT_HINTS
        ):
            return CorruptedDocumentError(
                f"{label} 文件已损坏或格式不合法（{type(exc).__name__}）；"
                "常见原因是下载中断、扩展名被改名，或导出工具不规范——"
                "请用原程序重新导出后再上传",
                detail={
                    "format": label,
                    "exception": type(exc).__name__,
                    "reason": str(exc)[:300],
                },
            )

        return DocumentError(
            f"{label} 加载失败：{type(exc).__name__}: {exc}",
            detail={"format": label, "exception": type(exc).__name__},
        )
