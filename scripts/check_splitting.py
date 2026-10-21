# -*- coding: utf-8 -*-
"""解析 + 切分自检：验证「文件 → Page（带页码）→ Chunk（带页码溯源）」这一步是对的。

为什么需要这个脚本：
    加载层已有 `check_loading.py` 把关（页码、编码、表格、空文件…），
    但从 Page 到 Chunk 这一段此前**没有任何自动检查**。它的失效是静默的：
    页码串位、块整体超出块大小、切分把内容吃掉、空页挤出空洞的 chunk_index，
    这四种都不会抛异常，只会让引用溯源显示错页、或让检索质量悄悄变差。

判据（不只断言"不报错"）：

    | 检查项            | 期望                                                     |
    |------------------|----------------------------------------------------------|
    | 页码溯源          | page_start == page_end，且落在该文档实际页号内             |
    | 单页格式          | TXT / MD / DOCX 切出的每一块 page_start == 1             |
    | 逐页独立           | 块**不得跨页**（这正是改为按页切分的原因）                |
    | 顺序              | chunk_index 从 0 连续无空洞；page_start 单调不减          |
    | 覆盖性            | 原文每 10 字都能在某一块里找到（切分不许吃掉内容）          |
    | 无捏造            | 每块文本必须是**本页原文**的子串（不得混入别的页）           |
    | 块大小            | 每块长度 ≤ CHUNK_SIZE                                     |
    | 重叠              | 同一页的相邻块之间至少要有一处重叠                         |
    | 空页              | 空页不产出块，且**不影响其后页的页码**                     |
    | 全空              | 全部页为空时返回空列表（不抛错，报错是加载层的职责）         |
    | 越权              | 切分器不填 id / kb_id / doc_id / embedding，不碰 Milvus    |
    | 参数校验          | chunk_overlap >= chunk_size 抛 ValueError                |

运行（必须用项目自己的 .venv）：
    .venv\\Scripts\\python.exe scripts\\check_splitting.py

    脚本会自建临时样例目录，不依赖 tests/fixtures 是否已生成，
    但 PDF / PPTX / TXT / MD 需要 pypdf / python-pptx；
    DOCX 用例在 python-docx 缺失时自动跳过（它只是开发期依赖）。

退出码：0 全部通过；1 存在失败项。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# settings 的 env_file 是相对路径，必须先切到项目根
os.chdir(PROJECT_ROOT)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from app.core.config import get_settings  # noqa: E402
from app.domain.models import Chunk, Page  # noqa: E402
from app.infrastructure.parsing import CompositeDocumentParser  # noqa: E402
from app.infrastructure.splitting import LangChainSplitter  # noqa: E402
from scripts.make_sample_docs import build_pdf  # noqa: E402

# ---------------------------------------------------------------------------
# 极简断言器：收集全部失败项，而不是第一个就退出
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


# ---------------------------------------------------------------------------
# 自检专用的通用断言：任一页列表 + 其切分结果必须满足的契约
# ---------------------------------------------------------------------------


def overlap_len(left: str, right: str) -> int:
    """返回 left 的后缀与 right 的前缀的最大重合长度（0 表示不重叠）。

    比较前先去掉全部空白：LangChain 默认 ``strip_whitespace=True``，
    分隔符处的空白（如 ``\\n\\n``）在切分后**会被丢掉**，
    见模块 docstring 的说明。不做归一化会误判成"没有重叠"。
    """
    left, right = compact(left), compact(right)
    for length in range(min(len(left), len(right)), 0, -1):
        if left.endswith(right[:length]):
            return length
    return 0


def compact(text: str) -> str:
    """去掉全部空白字符，用于比较切分前后是否丢失内容。

    不能直接用原串比较：LangChain 的切分器默认会 strip 掉块首尾空白，
    落在分隔符上的 ``\\n``/空格因此不会出现在任何块里。
    这些空白对语义没有影响，但对"逐字相等"的断言是致命的误报来源。
    """
    return "".join(text.split())


def check_contract(tag: str, pages: list[Page], chunks: list[Chunk], chunk_size: int) -> None:
    """校验一份「页列表 → 块列表」是否满足全部通用契约。"""
    usable = [(page.page_no, page.text.strip()) for page in pages if page.text.strip()]
    valid_pages = {no for no, _ in usable}

    check(f"{tag}：产出了块", bool(chunks), f"实际 {len(chunks)} 块")
    if not chunks:
        return

    check(
        f"{tag}：chunk_index 从 0 连续无空洞",
        [c.chunk_index for c in chunks] == list(range(len(chunks))),
        f"实际 {[c.chunk_index for c in chunks]}",
    )

    # 页码溯源：按页切分 ⇒ 每块必然且只属于一页
    unpaired = [c for c in chunks if c.page_start != c.page_end]
    check(f"{tag}：page_start == page_end（不跨页）", not unpaired,
          f"{len(unpaired)} 块跨页，例如 {unpaired[0].page_start}-{unpaired[0].page_end}"
          if unpaired else "")

    unknown = [c for c in chunks if c.page_start not in valid_pages]
    check(f"{tag}：页码落在实际存在的页内", not unknown,
          f"越界页码 {sorted({c.page_start for c in unknown})}")

    starts = [c.page_start for c in chunks]
    check(f"{tag}：页码单调不减（顺序保持）", starts == sorted(starts), f"实际 {starts}")

    # 覆盖性 + 无捏造：每块必须是本页原文的子串，且原文内容都要出现
    # （按去空白后的字符串比较，理由见 compact() 的说明）
    page_compact = {no: compact(text) for no, text in usable}
    fabricated = [
        c for c in chunks
        if c.page_start in page_compact and compact(c.text) not in page_compact[c.page_start]
    ]
    check(f"{tag}：每块都是本页原文的子串（不混入别页）", not fabricated,
          f"{len(fabricated)} 块不是子串，首例 {fabricated[0].text[:30]!r}" if fabricated else "")

    missing: list[str] = []
    for page_no, text in usable:
        joined = compact("".join(c.text for c in chunks if c.page_start == page_no))
        whole = page_compact[page_no]
        for index in range(0, len(whole), 10):
            segment = whole[index : index + 10]
            if segment not in joined:
                missing.append(f"第{page_no}页 {segment!r}")
                break
    check(f"{tag}：原文内容无丢失（逐 10 字覆盖）", not missing, "；".join(missing))

    oversized = [c for c in chunks if len(c.text) > chunk_size]
    check(f"{tag}：每块长度 ≤ CHUNK_SIZE({chunk_size})", not oversized,
          f"{len(oversized)} 块超长，最长 {max(len(c.text) for c in oversized)}" if oversized else "")

    blank = [c for c in chunks if not c.text.strip()]
    check(f"{tag}：没有空块", not blank, f"实际 {len(blank)} 块为空")

    # 越权检查：切分器只负责文本层面
    check(f"{tag}：未填 id / kb_id / doc_id",
          all(not c.id and not c.kb_id and not c.doc_id for c in chunks))
    check(f"{tag}：embedding 保持 None", all(c.embedding is None for c in chunks))

    # 同一页的相邻块之间应存在重叠
    for page_no, text in usable:
        same_page = [c.text for c in chunks if c.page_start == page_no]
        if len(same_page) < 2:
            continue
        gaps = [overlap_len(same_page[i], same_page[i + 1]) for i in range(len(same_page) - 1)]
        check(f"{tag}：第{page_no}页相邻块有重叠", any(g > 0 for g in gaps),
              f"该页 {len(same_page)} 块之间完全没有重叠（{gaps}），CHUNK_OVERLAP 可能没生效")


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------

_LONG_PAGE = (
    "向量检索是双塔结构，查询与文档各自编码，二者在编码阶段从不见面，精度天然受限。"
    "交叉编码器把查询与文档拼成一个序列一起过模型，能捕捉细粒度交互，精度显著更高，代价是慢。"
    "因此只对初检召回的少量结果做精排，而不是对全库做重排。"
) * 6


def case_pdf(parser: CompositeDocumentParser, splitter: LangChainSplitter, workdir: Path) -> None:
    """多页 PDF：逐页溯源 + 空页不串位。"""
    print("\n[1] PDF：多页页码溯源 + 空页处理")

    # 第 2 页故意留空（模拟纯图片页），验证空页不产出块、也不影响第 3 页页码
    target = workdir / "three_pages.pdf"
    target.write_bytes(build_pdf(["page one text about loading", "", "page three text about chunking"]))

    pages = parser.parse(str(target))
    check("解析出 3 页", len(pages) == 3, f"实际 {len(pages)}")
    check("页码为 [1,2,3]", [p.page_no for p in pages] == [1, 2, 3],
          f"实际 {[p.page_no for p in pages]}")
    check("第 2 页文本为空（纯图片页）", pages[1].text == "", f"实际 {pages[1].text!r}")

    chunks = splitter.split(pages)
    hit = sorted({c.page_start for c in chunks})
    check("空页不产出块", 2 not in hit, f"实际出现的页码 {hit}")
    check("空页之后的页页码不串位（仍是第 3 页）", hit == [1, 3], f"实际 {hit}")
    check_contract("PDF", pages, chunks, splitter.chunk_size)


def case_long_page(splitter: LangChainSplitter) -> None:
    """单页长文本：块大小、重叠、覆盖性。"""
    print("\n[2] 长文本：块大小与重叠")

    pages = [Page(page_no=1, text=_LONG_PAGE)]
    chunks = splitter.split(pages)
    check("长文本切出多个块", len(chunks) > 1, f"实际 {len(chunks)} 块")
    check_contract("长文本", pages, chunks, splitter.chunk_size)

    if len(chunks) > 1:
        gaps = [
            overlap_len(chunks[i].text, chunks[i + 1].text) for i in range(len(chunks) - 1)
        ]
        print(f"         块长 {[len(c.text) for c in chunks]}  相邻重叠 {gaps}")
        check("存在跨块重叠（CHUNK_OVERLAP 生效）", any(g > 0 for g in gaps), f"重叠 {gaps}")

    # 重叠设置过大时不该静默出错
    expect_error(
        "overlap >= chunk_size 抛 ValueError",
        ValueError,
        "必须小于",
        lambda: LangChainSplitter(chunk_size=100, chunk_overlap=100),
    )


def case_single_page(
    parser: CompositeDocumentParser, splitter: LangChainSplitter, workdir: Path
) -> None:
    """无页码格式：TXT / MD / DOCX 一律记第 1 页。"""
    print("\n[3] 无页码格式：TXT / MD / DOCX")

    txt = workdir / "single.txt"
    txt.write_text(_LONG_PAGE, encoding="utf-8")
    md = workdir / "single.md"
    md.write_text("# 标题\n\n" + _LONG_PAGE + "\n\n- 列表项\n", encoding="utf-8")

    for path in (txt, md):
        pages = parser.parse(str(path))
        check(f"{path.name} 解析为 1 页", len(pages) == 1, f"实际 {len(pages)} 页")
        chunks = splitter.split(pages)
        check(f"{path.name} 的块全部记第 1 页",
              all(c.page_start == 1 and c.page_end == 1 for c in chunks),
              f"实际页码 {sorted({c.page_start for c in chunks})}")
        check_contract(path.suffix[1:].upper(), pages, chunks, splitter.chunk_size)

    docx = workdir / "single.docx"
    try:
        import docx as docx_module
    except Exception:  # noqa: BLE001 - python-docx 是开发期依赖，缺了就跳过
        print("  [skip] python-docx 未安装，跳过 DOCX 用例")
        return
    document = docx_module.Document()
    document.add_paragraph("第一段：解析把文档变成带页码的页列表。")
    document.add_paragraph(_LONG_PAGE)
    document.save(str(docx))

    pages = parser.parse(str(docx))
    check("DOCX 解析为 1 页", len(pages) == 1, f"实际 {len(pages)} 页")
    chunks = splitter.split(pages)
    check("DOCX 的块全部记第 1 页",
          all(c.page_start == 1 for c in chunks), f"实际 {sorted({c.page_start for c in chunks})}")
    check_contract("DOCX", pages, chunks, splitter.chunk_size)


def case_pptx(parser: CompositeDocumentParser, splitter: LangChainSplitter, workdir: Path) -> None:
    """PPTX：每张幻灯片一页，块按页归位。"""
    print("\n[4] PPTX：每张幻灯片独立成页")

    from pptx import Presentation

    target = workdir / "slides.pptx"
    presentation = Presentation()
    layout = presentation.slide_layouts[1]
    for index in range(1, 4):
        slide = presentation.slides.add_slide(layout)
        slide.shapes.title.text = f"第 {index} 张幻灯片"
        slide.placeholders[1].text = f"这是第 {index} 页的内容。" + _LONG_PAGE[:120]
    presentation.save(str(target))

    pages = parser.parse(str(target))
    check("解析出 3 页（3 张幻灯片）", len(pages) == 3, f"实际 {len(pages)}")
    check("页码为 [1,2,3]", [p.page_no for p in pages] == [1, 2, 3],
          f"实际 {[p.page_no for p in pages]}")

    chunks = splitter.split(pages)
    check("每张幻灯片都切出了块", sorted({c.page_start for c in chunks}) == [1, 2, 3],
          f"实际 {sorted({c.page_start for c in chunks})}")
    check_contract("PPTX", pages, chunks, splitter.chunk_size)


def case_edges(splitter: LangChainSplitter) -> None:
    """边界：全空页、单页为空、页码为 0 起之外的异常输入。"""
    print("\n[5] 边界")

    check("全空页返回空列表", splitter.split([Page(page_no=1, text="   \n\t ")]) == [])
    check("空页列表返回空列表", splitter.split([]) == [])

    pages = [Page(page_no=1, text=""), Page(page_no=2, text="有内容的一页。"), Page(page_no=3, text="  ")]
    chunks = splitter.split(pages)
    check("夹在空页之间的有效页仍能切出块", len(chunks) >= 1, f"实际 {len(chunks)}")
    check("有效页的页码正确", all(c.page_start == 2 for c in chunks),
          f"实际 {sorted({c.page_start for c in chunks})}")

    # 未知页码（parsing 判定为 None）必须**原样透传**。
    # 这里最容易写成 `page_no or 1`——0 与 None 都是 falsy，一兜底就变成
    # 「引用显示第 1 页而正文在第 3 页」，而且不会报错，只能靠这类断言抓。
    unknown = splitter.split([Page(page_no=None, text="未知页码的一页，内容足够切出一块。")])
    check("page_no=None 的页仍能切分", len(unknown) == 1, f"实际 {len(unknown)}")
    check(
        "未知页码原样透传为 None（不写 `or 1` 兜底）",
        unknown[0].page_start is None and unknown[0].page_end is None,
        f"page_start={unknown[0].page_start} page_end={unknown[0].page_end}",
    )
    zero = splitter.split([Page(page_no=0, text="页码为 0 的页（0 是 falsy，但绝不能变成 1）。")])
    check(
        "page_no=0 不被 falsy 兜底改成 1（`or 1` 的典型翻车点）",
        zero[0].page_start == 0,
        f"实际 {zero[0].page_start}",
    )


# ---------------------------------------------------------------------------


def main() -> int:
    settings = get_settings()
    print("=" * 64)
    print("解析 + 切分自检")
    print(f"  CHUNK_SIZE={settings.chunk_size}  CHUNK_OVERLAP={settings.chunk_overlap}")
    print(f"  受支持格式：{'、'.join(CompositeDocumentParser().supported_extensions)}")
    print("=" * 64)

    parser = CompositeDocumentParser()
    splitter = LangChainSplitter()

    workdir = Path(tempfile.mkdtemp(prefix="rag-split-check-"))
    try:
        case_pdf(parser, splitter, workdir)
        case_long_page(splitter)
        case_single_page(parser, splitter, workdir)
        case_pptx(parser, splitter, workdir)
        case_edges(splitter)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print("\n" + "=" * 64)
    if _failures:
        print(f"未通过 {len(_failures)}/{_checks} 项：")
        for item in _failures:
            print(f"  - {item}")
        return 1
    print(f"全部通过（{_checks} 项）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
