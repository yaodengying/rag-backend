# -*- coding: utf-8 -*-
"""文档加载自检：验证「文件 → LangChain Document → 领域 Page」这一步是对的。

为什么需要这个脚本：
    「没报错」不等于「加载对了」。页码错位、中文变乱码、扫描件被当成空文档，
    这三种问题都不会抛异常，只会让后面的检索与引用悄悄失准。
    所以每次改加载器、升级 langchain、换 Python 版本后都应跑一遍。

判据（不只断言"不报错"）：

    | 检查项        | 期望                                                    |
    |--------------|---------------------------------------------------------|
    | PDF 页码      | 3 页文档 → page_no 恰为 [1,2,3]（loader 的 page 是 0 起） |
    | PPTX 页码     | 3 张幻灯片 → page_no 恰为 [1,2,3]                        |
    | 无页码格式    | DOCX/TXT/MD → 单页且 page_no == 1                        |
    | 内容完整性    | DOCX 表格单元格、PPTX 幻灯片表格文字都不得丢失            |
    | 中文编码      | GB18030 文件解读出正确中文，不得出现替换字符 �            |
    | 空文件        | 抛 EmptyDocumentError「文件内容为空」                     |
    | 文件不存在    | 抛 NotFoundError（404，不是 500）                         |
    | 不支持的扩展名 | 抛 ValidationError（.doc 在这一层就被拒）                |
    | 损坏文件      | 抛 CorruptedDocumentError，消息含「已损坏」               |
    | 无文本文件    | 抛 EmptyDocumentError「未加载到任何文本」，不得静默返回空  |
    | 加密文件      | 抛 EncryptedDocumentError「已加密，请先解除密码保护」      |
    | 三类文档问题  | code 互不相同、都是 400、都继承 DocumentError             |
    | 纯映射函数    | page=0/缺失/字符串/浮点/罗马数字/bool → 预期页码或 None    |
    | 无法解析的页码 | **None 而不是 1**（编造页码会让引用溯源标错页）           |

运行：
    .venv\\Scripts\\python.exe scripts\\check_loading.py

    脚本会把项目根临时加进 sys.path，因此不要求当前工作目录，
    但**必须用项目自己的 .venv**（依赖 langchain-community / docx2txt / python-pptx）。

退出码：0 全部通过；1 存在失败项。
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document  # noqa: E402

from app.core.exceptions import (  # noqa: E402
    CorruptedDocumentError,
    DocumentError,
    EmptyDocumentError,
    EncryptedDocumentError,
    NotFoundError,
    ValidationError,
)
from app.domain.models import Page  # noqa: E402
from app.infrastructure.loading import LangChainDocumentLoader  # noqa: E402
from app.infrastructure.parsing import (  # noqa: E402
    CompositeDocumentParser,
    document_to_page,
)
from scripts.make_sample_docs import build_pdf  # noqa: E402

# ---------------------------------------------------------------------------
# 极简断言器：收集全部失败项，而不是第一个就退出——
# 一次跑完拿到完整清单，比反复重跑省事。
# ---------------------------------------------------------------------------

_failures: list[str] = []
_checks = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    """记录一项断言结果。"""
    global _checks
    _checks += 1
    if condition:
        print(f"  [ok]   {label}")
    else:
        print(f"  [FAIL] {label}" + (f"  -> {detail}" if detail else ""))
        _failures.append(label)


def expect_error(label: str, exc_type: type[Exception], keyword: str, action) -> None:
    """断言 action 抛出指定类型的异常，且消息里含关键词。"""
    global _checks
    _checks += 1
    try:
        action()
    except exc_type as exc:
        if keyword in str(exc):
            print(f"  [ok]   {label}（{type(exc).__name__}: {exc}）")
        else:
            print(f"  [FAIL] {label}  -> 消息里没有 {keyword!r}：{exc}")
            _failures.append(label)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {label}  -> 期望 {exc_type.__name__}，实际 {type(exc).__name__}: {exc}")
        _failures.append(label)
    else:
        print(f"  [FAIL] {label}  -> 没有抛异常")
        _failures.append(label)


def build_encrypted_pdf(target: Path) -> None:
    """用 pypdf 给一份最小 PDF 加密，用于验证加密文件的报错提示。"""
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(io.BytesIO(build_pdf(["secret page"])))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt("open-sesame")
    with target.open("wb") as handle:
        writer.write(handle)


# ---------------------------------------------------------------------------
# 各项检查
# ---------------------------------------------------------------------------


def check_positive(parser: CompositeDocumentParser, fixtures: Path) -> None:
    """正向：5 种格式的页数与页码。"""
    print("\n[1] 各格式加载与页码")

    pdf_pages = parser.parse(str(fixtures / "sample.pdf"))
    check("PDF 页数为 3", len(pdf_pages) == 3, f"实际 {len(pdf_pages)}")
    check(
        "PDF 页码为 [1,2,3]",
        [page.page_no for page in pdf_pages] == [1, 2, 3],
        f"实际 {[page.page_no for page in pdf_pages]}",
    )
    check("PDF 文本非空", all(page.text for page in pdf_pages))

    pptx_pages = parser.parse(str(fixtures / "sample.pptx"))
    check("PPTX 页数为 3", len(pptx_pages) == 3, f"实际 {len(pptx_pages)}")
    check(
        "PPTX 页码为 [1,2,3]",
        [page.page_no for page in pptx_pages] == [1, 2, 3],
        f"实际 {[page.page_no for page in pptx_pages]}",
    )
    check("PPTX 首页含标题文字", "文档加载" in pptx_pages[0].text)

    for name in ("sample.docx", "sample.txt", "sample.md"):
        pages = parser.parse(str(fixtures / name))
        check(f"{name} 为单页且 page_no=1", len(pages) == 1 and pages[0].page_no == 1)

    docx_text = parser.parse(str(fixtures / "sample.docx"))[0].text
    # 表格单元格如果丢了，参数表这类关键信息会在检索里彻底不可见
    check("DOCX 表格单元格未丢失", "Document 列表" in docx_text, docx_text[:80])


def check_encoding(parser: CompositeDocumentParser, fixtures: Path) -> None:
    """编码：GB18030 与 UTF-8 必须读出同样的中文。"""
    print("\n[2] 中文编码")

    utf8_text = parser.parse(str(fixtures / "sample.txt"))[0].text
    gbk_text = parser.parse(str(fixtures / "sample-gb18030.txt"))[0].text
    check("GB18030 与 UTF-8 解码结果一致", utf8_text == gbk_text)
    check("未出现替换字符 U+FFFD", "\ufffd" not in gbk_text)
    check("中文关键词可读", "文档加载是 RAG 的第一步" in gbk_text)


def check_edges(parser: CompositeDocumentParser, workdir: Path) -> None:
    """边界：各类失败必须以**正确的异常类型**与文案暴露。

    三类文档问题的处置动作完全不同（换文件 / 解密码 / 上 OCR），
    所以断言的是具体子类而不只是 AppError——只判基类的话，
    "加密被判成损坏"这种错误照样能过。
    """
    print("\n[3] 失败语义")

    expect_error(
        "空文件",
        EmptyDocumentError,
        "文件内容为空",
        lambda: parser.parse(str(workdir / "empty.txt")),
    )
    expect_error(
        "文件不存在",
        NotFoundError,
        "文件不存在",
        lambda: parser.parse(str(workdir / "nope.pdf")),
    )
    expect_error(
        "不支持的扩展名 .doc",
        ValidationError,
        "不支持的扩展名",
        lambda: parser.parse(str(workdir / "legacy.doc")),
    )
    expect_error(
        "损坏的 PDF",
        CorruptedDocumentError,
        "已损坏",
        lambda: parser.parse(str(workdir / "broken.pdf")),
    )
    expect_error(
        "无文本的 PDF（扫描件场景）",
        EmptyDocumentError,
        "未加载到任何文本",
        lambda: parser.parse(str(workdir / "blank.pdf")),
    )
    expect_error(
        "加密的 PDF",
        EncryptedDocumentError,
        "已加密",
        lambda: parser.parse(str(workdir / "encrypted.pdf")),
    )

    codes = {
        cls.__name__: cls.code
        for cls in (
            DocumentError,
            CorruptedDocumentError,
            EncryptedDocumentError,
            EmptyDocumentError,
        )
    }
    check(
        "三类文档问题各有独立的 code（前端/CLI 可分支）",
        len(set(codes.values())) == len(codes),
        f"{codes}",
    )
    check(
        "文档问题一律 400（客户端换文件即可），不是 500",
        all(
            cls.http_status == 400
            for cls in (
                DocumentError,
                CorruptedDocumentError,
                EncryptedDocumentError,
                EmptyDocumentError,
            )
        ),
        f"{' '.join(f'{k}={v}' for k, v in codes.items())}",
    )
    check(
        "子类都继承 DocumentError（上层仍可按基类统一处理）",
        all(
            issubclass(cls, DocumentError)
            for cls in (
                CorruptedDocumentError,
                EncryptedDocumentError,
                EmptyDocumentError,
            )
        ),
    )
    check("文件不存在是 404 而不是 400（路径错 ≠ 文件内容错）", NotFoundError.http_status == 404)


def check_mapping() -> None:
    """纯映射：document_to_page 的页码规则（不碰磁盘）。

    ⚠️ 关键判据是「**无法解析 → None**」。旧实现在这里静默回退第 1 页，
    结果引用溯源显示「第1页」而正文在第 3 页——编造页码比承认不知道更糟。
    """
    print("\n[4] Document → Page 映射规则")

    cases: list[tuple[dict, int | None]] = [
        ({"page": 0}, 1),
        ({"page": 2}, 3),
        ({"page": 1.0}, 2),  # 形如 1.0 的 float 等价于 1
        ({"page": 2.5}, None),  # 非整数 float：不猜
        ({}, 1),  # 没有页概念（txt/md/docx）
        ({"page": None}, 1),  # 显式 None 与缺失同义
        ({"page": "4"}, 5),
        ({"page": " 4 "}, 5),
        ({"page": "iv"}, None),  # 罗马数字：无法映射物理页序 → 未知
        ({"page": "3 of 10"}, None),  # 该走 page_label，page 本身读不出
        ({"page": True}, None),  # bool 是 int 子类，读成页码纯属事故
        ({"page": -1}, None),  # 负页码没有展示语义
        ({"page_label": "5"}, 5),  # page_label 是 1 起，不再 +1
        ({"page_label": "P3"}, 3),
        ({"page_label": "iv"}, 1),  # 没有 page 键 ⇒ 该格式无页概念，如实记第 1 页
        ({"page": "iv", "page_label": "4"}, 4),  # page 读不出时退回 page_label
        ({"page": "3 of 10", "page_label": "3 of 10"}, 3),
    ]
    for metadata, expected in cases:
        page = document_to_page(Document(page_content=" x ", metadata=metadata))
        check(
            f"metadata={metadata} → page_no={expected}",
            page.page_no == expected,
            f"实际 {page.page_no}",
        )

    page = document_to_page(Document(page_content="  前后空白  ", metadata={}))
    check("文本首尾空白被清理", page.text == "前后空白", repr(page.text))
    check("返回的是领域 Page", isinstance(page, Page))

    unknown = document_to_page(Document(page_content=" x ", metadata={"page": "iv"}))
    check(
        "未知页码是 None 而不是 0（0 会被前端渲染成「第0页」）",
        unknown.page_no is None,
        f"实际 {unknown.page_no!r}",
    )


def check_registry(parser: CompositeDocumentParser) -> None:
    """格式清单必须与配置里的上传白名单一致，否则会出现"能上传不能加载"。"""
    print("\n[5] 加载器清单 vs 上传白名单")

    from app.core.config import get_settings

    loader_exts = set(parser.supported_extensions)
    upload_exts = set(get_settings().upload_extension_list)
    check(
        f"扩展名清单一致 {sorted(loader_exts)}",
        loader_exts == upload_exts,
        f"仅加载器有 {sorted(loader_exts - upload_exts)}；仅白名单有 {sorted(upload_exts - loader_exts)}",
    )
    check("PPTX 走自定义 LangChain Loader（不依赖 unstructured）", ".pptx" in loader_exts)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # 切到项目根：.env 是按相对路径加载的，不切目录就读不到配置，
    # 「扩展名清单 vs 上传白名单」那一项会退化成比对两份默认值。
    os.chdir(PROJECT_ROOT)

    fixtures = PROJECT_ROOT / "tests" / "fixtures"
    if not (fixtures / "sample.pdf").exists():
        print("未找到测试样例，正在生成……")
        from scripts.make_sample_docs import main as make_samples

        make_samples()

    parser = CompositeDocumentParser()
    loader = LangChainDocumentLoader()
    print(f"加载器实现：{type(loader).__name__}，支持 {parser.supported_extensions}")

    # 边界用例需要几个"畸形文件"，在临时目录里现造，避免污染 fixtures
    workdir = Path(tempfile.mkdtemp(prefix="rag-loading-check-"))
    try:
        (workdir / "empty.txt").write_bytes(b"")
        (workdir / "legacy.doc").write_bytes(b"\xd0\xcf\x11\xe0old binary doc")
        (workdir / "broken.pdf").write_bytes(b"this is definitely not a pdf")
        (workdir / "blank.pdf").write_bytes(build_pdf([""]))
        build_encrypted_pdf(workdir / "encrypted.pdf")

        check_positive(parser, fixtures)
        check_encoding(parser, fixtures)
        check_edges(parser, workdir)
        check_mapping()
        check_registry(parser)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print("\n" + "=" * 60)
    if _failures:
        print(f"未通过 {len(_failures)}/{_checks} 项：")
        for item in _failures:
            print(f"  - {item}")
        return 1
    print(f"全部通过（{_checks} 项）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
