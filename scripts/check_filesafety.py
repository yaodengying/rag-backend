# -*- coding: utf-8 -*-
"""上传内容静态检查自检（恶意 / 畸形文件的第一道闸）。

为什么需要这个脚本：
    扩展名白名单是**唯一**的既有防线，而它只看名字——把 `evil.exe` 改名成
    `evil.pdf` 成本为零。这不是"理论上可能"：一个 50MB 的 zip 炸弹解压后
    可以是 50GB，而 .docx / .pptx 本来就是 zip 容器；一个 3 万页的 PDF
    能让解析阶段 OOM，进程被系统直接杀掉，日志里连一行报错都没有。

判据：

    | 分组         | 期望                                                        |
    |-------------|-------------------------------------------------------------|
    | 文件名       | 控制字符/换行/超长 → 拒；正常中文名 → 放行                   |
    | PDF 魔数     | 必须在前 1KB 内出现 `%PDF-`；伪装成 .pdf 的可执行文件 → 拒   |
    | PDF 页数     | 超过 MAX_PDF_PAGES → 拒；读不出页数（结构损坏）时**不拦截**  |
    | zip 容器     | 魔数不是 PK → 拒；BadZipFile → 拒；条目数/总量/压缩比超限 → 拒 |
    | 文本形态     | NUL 字节 → 拒；控制字符占比 > 1% → 拒；正常中英文 → 放行     |
    | 错误语义     | UnsafeFileError 是 ValidationError 子类、400、code=unsafe_file |
    | 接线         | IngestService 上传伪装文件时真的调用了检查，且**不产生元数据行** |

运行：
    .venv\\Scripts\\python.exe scripts\\check_filesafety.py

    纯内存构造样本，不读磁盘、不连 Milvus、不加载 BGE，秒级完成。

退出码：0 全部通过；1 存在失败项。
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from app.core.exceptions import (  # noqa: E402
    UnsafeFileError,
    ValidationError,
)
from app.infrastructure.filesafety import SafetyLimits, inspect_upload  # noqa: E402
from app.services.ingest_service import IngestService  # noqa: E402

# ---------------------------------------------------------------------------
# 极简断言器
# ---------------------------------------------------------------------------

_failures: list[str] = []
_checks = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _checks
    _checks += 1
    if condition:
        print(f"  [ok]   {label}")
    else:
        print(f"  [FAIL] {label}" + (f"  -> {detail}" if detail else ""))
        _failures.append(label)


def expect_reject(label: str, keyword: str, filename: str, content: bytes, limits: SafetyLimits | None = None) -> None:
    """断言 inspect_upload 拒绝该内容，且消息里含关键词。"""
    global _checks
    _checks += 1
    try:
        inspect_upload(filename, content, limits=limits)
    except UnsafeFileError as exc:
        if keyword in str(exc):
            print(f"  [ok]   {label}")
        else:
            print(f"  [FAIL] {label}  -> 消息里没有「{keyword}」：{exc}")
            _failures.append(label)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {label}  -> 抛的是 {type(exc).__name__}：{exc}")
        _failures.append(label)
    else:
        print(f"  [FAIL] {label}  -> 没有抛异常")
        _failures.append(label)


def expect_pass(label: str, filename: str, content: bytes, limits: SafetyLimits | None = None) -> None:
    """断言 inspect_upload 放行该内容。"""
    global _checks
    _checks += 1
    try:
        inspect_upload(filename, content, limits=limits)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {label}  -> 误报 {type(exc).__name__}：{exc}")
        _failures.append(label)
    else:
        print(f"  [ok]   {label}")


# ---------------------------------------------------------------------------
# 样本构造
# ---------------------------------------------------------------------------

DEFAULT = SafetyLimits()


def make_pdf(pages: int = 1) -> bytes:
    """用 pypdf 造一份合法 PDF（空白页）。"""
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def make_zip(entries: dict[str, bytes], compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    """造一个 zip 容器（.docx/.pptx 的真实结构是一堆 XML，这里只关心容器层）。"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------


def case_filename() -> None:
    print("\n[1] 文件名")
    expect_reject(
        "含换行 → 拒（会把日志与响应头切开）",
        "控制字符",
        "a\nb.txt",
        b"hello",
    )
    expect_reject("含 NUL → 拒", "控制字符", "a\x00b.txt", b"hello")
    expect_reject("含路径分隔符 → 拒", "路径分隔符", "a/b.txt", b"hello")
    expect_reject("超过 200 字符 → 拒", "文件名过长", "长" * 201 + ".txt", b"hello")
    expect_pass("正常中文名 → 放行", "售后服务制度.txt", "客户须在 48 小时内响应".encode("utf-8"))
    expect_pass("200 字符整 → 放行（边界）", "长" * 196 + ".txt", b"hello")
    expect_reject("空内容 → 拒", "内容为空", "empty.txt", b"")


def case_pdf() -> None:
    print("\n[2] PDF")
    expect_pass("合法 PDF → 放行", "a.pdf", make_pdf(1))
    expect_pass(
        "前 1KB 内有前导垃圾的 PDF → 放行（部分导出工具会写注释）",
        "b.pdf",
        b"% comment\n" + make_pdf(1),
    )
    expect_reject(
        "伪装成 .pdf 的 MZ 可执行文件 → 拒",
        "文件头不是 PDF",
        "evil.pdf",
        b"MZ\x90\x00" + b"\x00" * 200,
    )
    expect_reject(
        "伪造 .pdf 的纯文本 → 拒",
        "文件头不是 PDF",
        "fake.pdf",
        b"this is definitely not a pdf",
    )
    expect_reject(
        "页数超限 → 拒",
        "超过上限",
        "long.pdf",
        make_pdf(5),
        SafetyLimits(max_pdf_pages=3),
    )
    expect_pass(
        "页数恰好等于上限 → 放行（边界）",
        "edge.pdf",
        make_pdf(3),
        SafetyLimits(max_pdf_pages=3),
    )
    # 结构损坏但魔数正确：**不**在这里判死，交给加载器报 corrupted_document
    expect_pass(
        "魔数正确但结构损坏 → 放行（交给加载器分类）",
        "broken.pdf",
        b"%PDF-1.4\n garbage after header",
    )


def case_zip_container() -> None:
    print("\n[3] zip 容器（docx / pptx）")
    normal = make_zip({"word/document.xml": b"<w:document/>", "docProps/app.xml": b"<x/>"})
    expect_pass("正常 docx → 放行", "a.docx", normal)
    expect_pass("正常 pptx → 放行", "a.pptx", normal)

    expect_reject(
        "伪装成 .docx 的非 zip 内容 → 拒",
        "文件头不是 zip",
        "evil.docx",
        b"\x7fELF" + b"\x00" * 100,
    )
    expect_reject(
        "PK 开头但结构损坏 → 拒",
        "zip 结构损坏",
        "broken.docx",
        b"PK\x03\x04" + b"\x00" * 64,
    )
    expect_reject(
        "条目数超限 → 拒",
        "超过上限",
        "many.docx",
        make_zip({f"f{i}.xml": b"x" for i in range(6)}),
        SafetyLimits(max_zip_entries=3),
    )
    expect_reject(
        "压缩比炸弹（2MB 零字节）→ 拒",
        "压缩比",
        "bomb.docx",
        make_zip({"word/document.xml": b"\x00" * (2 * 1024 * 1024)}),
        SafetyLimits(max_zip_ratio=20),
    )
    expect_reject(
        "声明的解压总量超限 → 拒",
        "解压后总大小",
        "big.docx",
        make_zip({f"f{i}.xml": os.urandom(1024 * 1024) for i in range(3)}),
        SafetyLimits(max_zip_uncompressed_bytes=2 * 1024 * 1024, max_zip_ratio=0),
    )
    expect_pass(
        "max_zip_ratio=0 → 关闭压缩比检查（总量仍受控）",
        "ratio-off.docx",
        make_zip({"word/document.xml": b"\x00" * (2 * 1024 * 1024)}),
        SafetyLimits(max_zip_ratio=0),
    )
    # 大字典序、低压缩比的正常 docx：压缩比约 3~10 倍，不能被误杀
    expect_pass(
        "正常 docx 的压缩比不触发误报",
        "real.docx",
        make_zip({"word/document.xml": ("段落内容" * 3000).encode("utf-8")}),
        DEFAULT,
    )
    # 真实测过的坑：重复度高的正常 docx 压缩比可达 346:1。
    # 只看比例会把正常文档判成炸弹，所以比例判定带 1MB 下限。
    expect_pass(
        "小体积高压缩比（346:1，<1MB）→ 放行",
        "repeat.docx",
        make_zip({"word/document.xml": ("段落内容" * 3000).encode("utf-8")}),
        DEFAULT,
    )


def case_text() -> None:
    print("\n[4] 文本形态")
    expect_pass("中文正文 → 放行", "a.txt", "第一章 总则\n客户须在 48 小时内响应。".encode("utf-8"))
    expect_pass("英文 markdown → 放行", "a.md", b"# Title\n\n- item one\n- item two\n")
    expect_pass("含制表符与换行 → 放行（不算控制字符）", "tab.txt", b"a\tb\r\nc\nd\n")
    expect_reject(
        "含 NUL 字节 → 拒（二进制伪装成文本）",
        "NUL",
        "bin.txt",
        b"hello\x00world",
    )
    expect_reject(
        "控制字符占比 > 1% → 拒",
        "控制字符占比",
        "ctrl.txt",
        b"abc\x01\x02" * 2000,
    )
    # 二进制但恰好没有 NUL（例如全是 0x01）：靠占比兜住
    expect_reject(
        "无 NUL 的二进制（控制字符密集）→ 拒",
        "控制字符占比",
        "dense.txt",
        b"\x01\x02\x03\x04" * 4096,
    )


def case_error_semantics() -> None:
    print("\n[5] 错误语义")
    check("UnsafeFileError 是 ValidationError 的子类", issubclass(UnsafeFileError, ValidationError))
    check("http_status = 400（客户端可修）", UnsafeFileError.http_status == 400)
    check("code = unsafe_file（与普通 validation_error 分开）", UnsafeFileError.code == "unsafe_file")
    exc = UnsafeFileError("x", detail={"k": 1})
    check("detail 原样保留", exc.detail == {"k": 1})
    try:
        inspect_upload("evil.pdf", b"MZ" + b"\x00" * 10)
    except UnsafeFileError as caught:
        check("拒绝时带上 head_hex 便于排查", isinstance(caught.detail, dict) and "head_hex" in caught.detail)
    else:
        check("拒绝时带上 head_hex 便于排查", False, "没有抛异常")


class _Repo:
    """最小仓储：只实现接线用例会用到的方法。"""

    def __init__(self) -> None:
        self.created: list[str] = []

    def get_knowledge_base(self, kb_id: str) -> dict | None:
        return {"id": kb_id}

    def find_document_by_hash(self, kb_id: str, content_hash: str):  # noqa: ANN201
        return None

    def create_document(self, kb_id: str, filename: str, content_hash: str = ""):  # noqa: ANN201
        self.created.append(filename)
        raise AssertionError("被检查拦下的文件不应该走到 create_document")


def case_service_wiring() -> None:
    print("\n[6] 与入库服务的接线")
    import asyncio

    repo = _Repo()
    service = IngestService(
        pipeline=None,  # 被拦在上传前，pipeline 不会被调用
        repository=repo,
        allowed_extensions=(".pdf", ".txt"),
        max_upload_bytes=1024 * 1024,
    )
    check("默认 inspector 已装配（不是 None）", service._inspect is not None)  # noqa: SLF001

    async def upload() -> None:
        await service.ingest_file("kb-1", "evil.pdf", b"MZ" + b"\x00" * 100, owner_id=None)

    try:
        asyncio.run(upload())
    except UnsafeFileError as exc:
        check("伪装文件被服务层拦下（不只是接口层）", "文件头不是 PDF" in str(exc))
    except Exception as exc:  # noqa: BLE001
        check("伪装文件被服务层拦下（不只是接口层）", False, f"抛的是 {type(exc).__name__}: {exc}")
    else:
        check("伪装文件被服务层拦下（不只是接口层）", False, "没有抛异常")

    check("被拦下时没有产生元数据行", repo.created == [], str(repo.created))

    # 体积检查在内容检查之前：500MB 的假 PDF 不该被读魔数（这里用小阈值验证顺序）
    small = IngestService(
        pipeline=None,
        repository=_Repo(),
        allowed_extensions=(".pdf",),
        max_upload_bytes=10,
    )

    async def too_big() -> None:
        await small.ingest_file("kb-1", "big.pdf", b"MZ" + b"\x00" * 100, owner_id=None)

    try:
        asyncio.run(too_big())
    except ValidationError as exc:
        check("体积超限先于内容检查触发", "超出体积上限" in str(exc))
    else:
        check("体积超限先于内容检查触发", False, "没有抛异常")


def main() -> int:
    print("=" * 72)
    print("上传内容静态检查自检：魔数 / 容器膨胀 / 文本形态 / 服务层接线")
    print("=" * 72)

    case_filename()
    case_pdf()
    case_zip_container()
    case_text()
    case_error_semantics()
    case_service_wiring()

    print("\n" + "=" * 72)
    if _failures:
        print(f"失败 {len(_failures)} / {_checks} 项：")
        for label in _failures:
            print(f"  - {label}")
        return 1
    print(f"全部通过：{_checks} 项断言")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
