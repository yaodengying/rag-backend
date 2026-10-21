"""上传内容的静态安全检查（恶意 / 畸形文件的第一道闸）。

**它挡的不是"格式不支持"，而是"格式对但内容是攻击载荷"**：

    - 扩展名白名单只看名字，把 `evil.exe` 改名成 `evil.pdf` 成本为零；
    - 一个 50MB 的 zip 炸弹解压后可以是 50GB，docx/pptx 就是 zip；
    - 一个 3 万页的 PDF 能让解析阶段把内存吃干净（不是崩溃，是 OOM 后被
      systemd / 容器 OOM killer 杀掉，日志里什么都没有）；
    - 文本文件里塞 NUL 字节，会让某些解析器把二进制当文本读进去。

因此这里做四件事（顺序有意：先看名字再看内容，最便宜的检查放前面）：

1. **文件名本身**：控制字符（`\\x00`、换行）会把日志与响应头切开；
2. **魔数（magic bytes）**：`.pdf` 必须以 `%PDF-` 开头（允许 1KB 前导垃圾——
   部分导出工具会写一串注释），`.docx`/`.pptx` 必须是 zip 容器；
3. **容器膨胀**：zip 条目数、解压后总大小、单条压缩比；
4. **文本形态**：NUL 字节与控制字符占比。

**能力边界要写清楚**：这里只看结构，不解压、不执行。真正的
"文档里藏宏/外链/JS"不归本模块管——PyPDF / docx2txt / python-pptx
都只抽文本，不执行嵌入对象，这是选它们的主要原因。反面是 pptx 的
`UnstructuredPowerPointLoader`，它会额外拉 25 个包且并无更强隔离。

失败一律抛 `UnsafeFileError`（400 + code=`unsafe_file`）：
前端据此提示"这个文件不能传"，与"参数填错了"分开。
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import UnsafeFileError

logger = logging.getLogger(__name__)

# PDF 规范允许文件头前有少量垃圾数据（读者应在前 1024 字节内寻找 %PDF-）
_PDF_MAGIC = b"%PDF-"
_PDF_MAGIC_WINDOW = 1024
_ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")

# 文本类：只有这几种扩展名走"看内容像不像文本"这条分支
_TEXT_EXTENSIONS = frozenset({".txt", ".md"})

# 控制字符（排除 \t \n \r）。NUL 单独判：它几乎只出现在二进制里。
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
# Windows 保留字符 + 路径分隔符，落盘前必须清掉（ingest 还会再清一次，
# 这里管的是"别把带换行的名字写进日志/响应头"）
_BAD_NAME_CHARS = re.compile(r"[\x00-\x1f\x7f/\\]")

# 文本文件里控制字符的容忍比例：中文正文里 0 个；超过 1% 基本可判定为二进制。
_TEXT_CONTROL_RATIO = 0.01

# 压缩比判定的**下限体积**：单条目解压后小于 1MB 时，比例再高也无害。
# 这条下限不是宽容，而是纠正一个错误的直觉——危险的是**绝对体积**，
# 不是比例。实测一份 36KB 的正常 docx（`<w:p>` 反复出现）压缩比就有 346:1，
# 若只看比例会把正常文档判成炸弹。真正的炸弹是"比例高**且**解压后很大"。
_MIN_BOMB_ENTRY_BYTES = 1024 * 1024


@dataclass(frozen=True)
class SafetyLimits:
    """静态检查的阈值。默认值与 Settings 保持一致（由 from_settings 覆盖）。"""

    max_pdf_pages: int = 3000
    max_zip_entries: int = 5000
    max_zip_uncompressed_bytes: int = 300 * 1024 * 1024
    max_zip_ratio: int = 200

    @classmethod
    def from_settings(cls) -> SafetyLimits:
        """从全局配置构造。

        **不在模块顶层 import settings**：那会让本模块在 import 时就把
        `.env` 读一遍，测试里想换阈值只能改环境变量。函数内延迟导入是
        本项目处理"配置 → 适配器"依赖的既定做法。
        """
        from app.core.config import get_settings

        settings = get_settings()
        return cls(
            max_pdf_pages=settings.max_pdf_pages,
            max_zip_entries=settings.max_zip_entries,
            max_zip_uncompressed_bytes=settings.max_zip_uncompressed_bytes,
            max_zip_ratio=settings.max_zip_ratio,
        )


def inspect_upload(
    filename: str,
    content: bytes,
    *,
    limits: SafetyLimits | None = None,
) -> None:
    """检查上传内容是否安全/可解析；不安全时抛 `UnsafeFileError`。

    Args:
        filename: 已归一化的文件名（只取最后一段）
        content: 文件字节
        limits: 阈值；None 时取全局配置

    Raises:
        UnsafeFileError: 文件名非法、魔数不符、容器膨胀、文本形态异常
    """
    _ensure_safe_name(filename)

    if not content:
        raise UnsafeFileError(f"文件内容为空：{filename}")

    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        _check_pdf(filename, content, limits or SafetyLimits.from_settings())
    elif suffix in (".docx", ".pptx"):
        _check_zip_container(filename, content, limits or SafetyLimits.from_settings())
    elif suffix in _TEXT_EXTENSIONS:
        _check_text(filename, content)


# ---------------------------------------------------------------------------
# 1. 文件名
# ---------------------------------------------------------------------------


def _ensure_safe_name(filename: str) -> None:
    if _BAD_NAME_CHARS.search(filename):
        raise UnsafeFileError(
            "文件名包含控制字符或路径分隔符，已拒绝；请重命名后再上传",
            detail={"filename_repr": repr(filename)[:200]},
        )
    if len(filename) > 200:
        # 落盘名是 `{doc_id}_{filename}`，doc_id 36 字符 + 下划线，
        # 太长的名字会在部分文件系统上直接 ENAMETOOLONG（Win32 260 上限）。
        raise UnsafeFileError(
            f"文件名过长（{len(filename)} 字符 > 200），请缩短后再上传"
        )


# ---------------------------------------------------------------------------
# 2. PDF：魔数 + 页数
# ---------------------------------------------------------------------------


def _check_pdf(filename: str, content: bytes, limits: SafetyLimits) -> None:
    head = content[:_PDF_MAGIC_WINDOW]
    if _PDF_MAGIC not in head:
        raise UnsafeFileError(
            f"{filename} 的扩展名是 .pdf，但文件头不是 PDF——"
            "可能是改了后缀的其它格式，或下载不完整",
            detail={
                "expected_magic": _PDF_MAGIC.decode("ascii"),
                "head_hex": head[:16].hex(),
                "size": len(content),
            },
        )

    pages = _pdf_page_count(filename, content)
    if pages is not None and pages > limits.max_pdf_pages:
        raise UnsafeFileError(
            f"{filename} 共 {pages} 页，超过上限 {limits.max_pdf_pages} 页；"
            "请拆分后再上传（页数过多的文档会在解析阶段耗尽内存）",
            detail={"pages": pages, "limit": limits.max_pdf_pages},
        )


def _pdf_page_count(filename: str, content: bytes) -> int | None:
    """读取页数；读不出来返回 None（交给加载器报更精确的损坏原因）。

    **为什么不用 len(reader.pages) 之外的判断**：pypdf 对损坏文件的报错
    有十几种形态，在这里穷举会把"文件坏了"变成"文件不安全"，而两者的
    用户动作不同（前者重新导出，后者别传）。
    """
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - pypdf 是 PyPDFLoader 的硬依赖
        return None

    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            # 有密码：不在这里判，加载器会抛加密专属错误（提示解除保护）
            try:
                reader.decrypt("")
            except Exception:  # noqa: BLE001
                return None
        return len(reader.pages)
    except Exception as exc:  # noqa: BLE001
        logger.debug("读取 PDF 页数失败（不拦截，交给加载器分类）：%s: %s", filename, exc)
        return None


# ---------------------------------------------------------------------------
# 3. zip 容器（docx / pptx）：条目数、解压总量、压缩比
# ---------------------------------------------------------------------------


def _check_zip_container(filename: str, content: bytes, limits: SafetyLimits) -> None:
    if not content.startswith(_ZIP_MAGICS):
        raise UnsafeFileError(
            f"{filename} 的扩展名是 {Path(filename).suffix.lower()}，"
            "但文件头不是 zip 容器——可能是改了后缀的其它格式",
            detail={"head_hex": content[:16].hex(), "size": len(content)},
        )

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            infos = archive.infolist()
            _check_zip_entries(filename, infos, limits)
            _check_zip_expansion(filename, infos, limits)
    except zipfile.BadZipFile as exc:
        raise UnsafeFileError(
            f"{filename} 的 zip 结构损坏，无法读取：{exc}；请重新导出该文件",
            detail={"exception": type(exc).__name__},
        ) from exc


def _check_zip_entries(filename: str, infos: list[zipfile.ZipInfo], limits: SafetyLimits) -> None:
    if len(infos) > limits.max_zip_entries:
        raise UnsafeFileError(
            f"{filename} 内包含 {len(infos)} 个条目，超过上限 {limits.max_zip_entries}；"
            "疑似压缩炸弹或异常文件，已拒绝",
            detail={"entries": len(infos), "limit": limits.max_zip_entries},
        )


def _check_zip_expansion(
    filename: str, infos: list[zipfile.ZipInfo], limits: SafetyLimits
) -> None:
    """按**中央目录声明的**大小估算解压总量与压缩比。

    局限（必须知道）：`file_size` 是压缩包自己声明的，可以被构造得比
    真实解压结果小。真正兜底的是解压侧——docx2txt / python-pptx 解压时
    若真的膨胀到 GB 级，进程会被 OOM 杀掉。所以本函数的价值是拦住
    **绝大多数**现成的炸弹样本（它们通常连声明值都懒得伪造），而不是
    提供严格保证。这一点写在这里，避免后来者以为加了这个就万事大吉。
    """
    total = 0
    for info in infos:
        total += max(info.file_size, 0)
        if (
            limits.max_zip_ratio > 0
            and info.compress_size > 0
            and info.file_size >= _MIN_BOMB_ENTRY_BYTES
        ):
            ratio = info.file_size / info.compress_size
            if ratio > limits.max_zip_ratio:
                raise UnsafeFileError(
                    f"{filename} 中的 {info.filename} 压缩比 {ratio:.0f}:1，"
                    f"超过上限 {limits.max_zip_ratio}:1；疑似压缩炸弹，已拒绝",
                    detail={
                        "entry": info.filename[:200],
                        "compressed": info.compress_size,
                        "uncompressed": info.file_size,
                        "ratio": round(ratio, 1),
                    },
                )
        if total > limits.max_zip_uncompressed_bytes:
            raise UnsafeFileError(
                f"{filename} 解压后总大小超过 "
                f"{limits.max_zip_uncompressed_bytes // 1024 // 1024} MB；"
                "疑似压缩炸弹，已拒绝",
                detail={
                    "declared_uncompressed": total,
                    "limit": limits.max_zip_uncompressed_bytes,
                },
            )


# ---------------------------------------------------------------------------
# 4. 纯文本：NUL 与控制字符占比
# ---------------------------------------------------------------------------


def _check_text(filename: str, content: bytes) -> None:
    if b"\x00" in content[:65536]:
        raise UnsafeFileError(
            f"{filename} 是文本扩展名但内容包含 NUL 字节——"
            "多半是二进制文件改了后缀，已拒绝",
            detail={"size": len(content)},
        )

    sample = content[:65536]
    # 只统计能被当作 UTF-8/Latin-1 解码的部分：中文正文里控制字节占比是 0，
    # 真二进制几乎必然超过 1%。
    controls = len(_CONTROL_CHARS.findall(sample.decode("latin-1")))
    if sample and controls / len(sample) > _TEXT_CONTROL_RATIO:
        raise UnsafeFileError(
            f"{filename} 中控制字符占比 {controls / len(sample):.1%}，"
            "超过 1%——疑似二进制文件伪装成文本，已拒绝",
            detail={"controls": controls, "sample_size": len(sample)},
        )
