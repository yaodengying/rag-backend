# -*- coding: utf-8 -*-
"""生成文档加载步骤的测试样例。

用途：
    给「文档加载」这一步造出各格式的最小样例，便于本地验证
    （PDF 多页、PPTX 多页、DOCX 表格、TXT 多编码、MD、空文件）。

用法：
    .venv\\Scripts\\python.exe scripts\\make_sample_docs.py

输出：
    tests/fixtures/ 下的样例文件，可重复生成（会覆盖）。

为什么 PDF 用英文：
    最小 PDF 用的是 base-14 内置字体（Helvetica），只支持 WinAnsi 编码，
    塞中文需要嵌入 CID 字体，会让这个「只为造样例」的脚本变得很重。
    页码验证不依赖文字内容，中文字符串放在 DOCX / PPTX / TXT / MD 里验证即可。
"""

from __future__ import annotations

import sys
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# ---------------------------------------------------------------------------
# 最小 PDF
# ---------------------------------------------------------------------------


def build_pdf(pages: list[str]) -> bytes:
    """把每页一行 ASCII 文本拼成一个多页 PDF（base-14 字体，无需额外依赖）。"""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    # 1: Catalog（先占位，稍后回填页对象号）
    catalog_no = add(b"")
    pages_no = add(b"")
    font_no = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_nos: list[int] = []
    for text in pages:
        content = f"BT /F1 18 Tf 72 700 Td ({_escape(text)}) Tj ET".encode("ascii")
        content_no = add(
            b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream"
        )
        page_no = add(
            b"<< /Type /Page /Parent "
            + str(pages_no).encode()
            + b" 0 R /MediaBox [0 0 612 792] /Contents "
            + str(content_no).encode()
            + b" 0 R /Resources << /Font << /F1 "
            + str(font_no).encode()
            + b" 0 R >> >> >>"
        )
        page_nos.append(page_no)

    objects[catalog_no - 1] = b"<< /Type /Catalog /Pages " + str(pages_no).encode() + b" 0 R >>"
    objects[pages_no - 1] = (
        b"<< /Type /Pages /Count "
        + str(len(page_nos)).encode()
        + b" /Kids ["
        + b" ".join(str(no).encode() + b" 0 R" for no in page_nos)
        + b"] >>"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(index).encode() + b" 0 obj\n" + body + b"\nendobj\n"

    xref_at = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root "
        + str(catalog_no).encode()
        + b" 0 R >>\nstartxref\n"
        + str(xref_at).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


def _escape(text: str) -> str:
    """转义 PDF 字面量字符串里的括号与反斜杠。"""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


# ---------------------------------------------------------------------------
# 各格式样例
# ---------------------------------------------------------------------------

_PDF_PAGES = [
    "Page 1 - Document loading with LangChain. The loader turns a file into Document objects.",
    "Page 2 - Chunking keeps page provenance so citations can point back to a page.",
    "Page 3 - Embedding and vector search follow after loading and chunking.",
]

_DOCX_PARAGRAPHS = [
    "文档加载步骤说明",
    "加载器负责把磁盘上的文件读成 LangChain 的 Document 对象，"
    "再由解析适配器映射成领域模型 Page。",
    "DOCX 属于流式文本，没有天然的页边界，因此整份文档统一记为第 1 页。",
]

_PPTX_SLIDES = [
    ("文档加载", "PDF / DOCX / TXT / MD / PPTX 五类格式统一入口"),
    ("切分", "按页独立切分，每一块都能追溯到所属页码"),
    ("向量化", "本地 BGE 权重，CPU 推理，块向量做 L2 归一化"),
]

# ---------------------------------------------------------------------------
# 问答溯源夹具（handbook.pptx）
#
# 为什么单独再造一份 PPTX 而不是复用 sample.pptx：
#     sample.pptx 的三页讲的是"流水线本身"，彼此共享大量词元
#     （加载/切分/向量化 都出现在同一句话里），用它验证页码会踩到一个坑：
#     回答正确不代表**引用页**正确——三个片段都可能被召回，
#     于是"引用了第 2 页"这句话本身没有信息量。
#
#     这份手册的每一页承载**一个互不重叠的事实**，且关键数字只出现一次：
#         第 2 页 九点整 / 半小时      第 3 页 五千元
#         第 4 页 每季度一次           第 5 页 十天
#     于是"问报销上限 → 引用第 3 页"成为一个可证伪的断言：
#     引用对了说明页码溯源整条链路（loader → parser → splitter → Milvus → 检索）都对；
#     引用错了就能立刻定位到是哪一环把页码丢了。
#
#     另外：中文 PDF 需要嵌入 CID 字体才能渲染（见本文件顶部说明），
#     而 PPTX 的每一页就是一张幻灯片，天然带页边界且支持中文——
#     它是本项目里"中文 + 多页 + 可自动生成"的唯一组合。
# ---------------------------------------------------------------------------

_HANDBOOK_SLIDES: list[tuple[str, str]] = [
    ("员工手册 · 总则", "本手册适用于公司全体正式员工。手册自二零二六年一月一日起施行。"),
    ("考勤制度", "公司上班时间为每日九点整。员工须在到岗后半小时内完成打卡。"),
    ("费用报销", "单笔报销金额不得超过五千元；超出部分须由部门总监审批。"),
    ("设备巡检", "生产设备的巡检周期为每季度一次，由设备科负责执行。"),
    ("假期管理", "员工入职满一年后，每年可享受带薪年假十天。"),
]

# 每页事实 → 期望被引用的页码（用于 ask.py --demo 的自动校验）
HANDBOOK_FACTS: list[tuple[str, int]] = [
    ("报销金额上限是多少？", 3),
    ("设备多久巡检一次？", 4),
    ("上班时间是几点？", 2),
    ("一年有几天年假？", 5),
]

_TXT = """# 纯文本样例（UTF-8）

文档加载是 RAG 的第一步。
这一步只做一件事：把各种格式的文件读成统一的文本表示。

它不做切分，也不做向量化 —— 那些是后面的步骤。
"""

_MD = """# Markdown 样例

## 二级标题

- 列表项一：加载器按扩展名分发
- 列表项二：加载失败必须抛错，而不是返回空列表

| 格式 | 页码来源 |
| --- | --- |
| PDF | 每页 1 页 |
| PPTX | 每张幻灯片 1 页 |
"""


def build_handbook(target: Path) -> Path:
    """生成用于问答溯源验证的中文多页 PPTX（每页一个独立事实）。

    单独抽成函数（而不是塞进 main）是为了让 ask.py / check_chat.py
    能在自己的临时目录里按需生成，不必依赖 fixtures 是否已经跑过。
    """
    from pptx import Presentation
    from pptx.util import Pt

    presentation = Presentation()
    layout = presentation.slide_layouts[1]
    for title, body in _HANDBOOK_SLIDES:
        slide = presentation.slides.add_slide(layout)
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
        slide.placeholders[1].text_frame.paragraphs[0].font.size = Pt(18)
    target.parent.mkdir(parents=True, exist_ok=True)
    presentation.save(str(target))
    return target


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    # --- PDF ---
    (FIXTURE_DIR / "sample.pdf").write_bytes(build_pdf(_PDF_PAGES))

    # --- DOCX ---
    import docx

    document = docx.Document()
    for paragraph in _DOCX_PARAGRAPHS:
        document.add_paragraph(paragraph)
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "步骤"
    table.cell(0, 1).text = "产出"
    table.cell(1, 0).text = "加载"
    table.cell(1, 1).text = "Document 列表"
    document.save(str(FIXTURE_DIR / "sample.docx"))

    # --- PPTX ---
    from pptx import Presentation
    from pptx.util import Pt

    presentation = Presentation()
    layout = presentation.slide_layouts[1]
    for title, body in _PPTX_SLIDES:
        slide = presentation.slides.add_slide(layout)
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
        slide.placeholders[1].text_frame.paragraphs[0].font.size = Pt(18)
    presentation.save(str(FIXTURE_DIR / "sample.pptx"))

    # --- PPTX（问答溯源用：中文多页，每页一个独立事实）---
    build_handbook(FIXTURE_DIR / "handbook.pptx")

    # --- TXT / MD ---
    (FIXTURE_DIR / "sample.txt").write_text(_TXT, encoding="utf-8")
    (FIXTURE_DIR / "sample-gb18030.txt").write_text(_TXT, encoding="gb18030")
    (FIXTURE_DIR / "sample.md").write_text(_MD, encoding="utf-8")
    (FIXTURE_DIR / "empty.txt").write_bytes(b"")

    for path in sorted(FIXTURE_DIR.iterdir()):
        print(f"{path.name:26} {path.stat().st_size:>8} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
