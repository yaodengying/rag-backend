"""生成中文多页 PDF 夹具，用于验证「上传 → 召回 → 重排 → 问答溯源」全链路。

为什么单独一个脚本，而不是并进 make_sample_docs.py：
    make_sample_docs.py 用 base-14 Helvetica 手搓 PDF，只能写 ASCII；
    中文必须先嵌入 TTF 字体（fpdf2 + 系统字体），依赖不同，单独放更清楚。

夹具的设计原则（这是能否验证出问题的关键，不是随手写几段话）：

1. **一页一事实**。每页只讲一件事，且该页的关键数字在**整个语料里只出现一次**
   （三份 PDF 合计 13 页，13 个互不重叠的数字）。否则「引用了第 3 页」这句话
   没有信息量——两页都写着 5000，召回哪一页都算对，页码溯源就退化成运气。

2. **页内长度控制在 CHUNK_SIZE 以内**，保证一页 = 一个片段，
   页号与片段是一一对应的，不会出现「跨页块该算第几页」的歧义。

3. **刻意制造跨文档的词面重叠**。IT 规范第 3 页讲「故障响应时限」，
   售后制度第 3 页讲「工单响应时效」——同一个「响应」出现在两份文档里，
   这正是重排该发挥作用的场景（词面看不出该选哪份，得看语义）。

4. **问题不照抄原文**。验证的是检索而不是字符串匹配，
   所以探针问题用口语化说法（原文写「工作时间为 9:00—18:00」，
   问题问「上班时间是几点」）。

用法：
    python scripts/make_pdf_fixtures.py            # 写入 tests/fixtures/pdf/
    python scripts/make_pdf_fixtures.py --out DIR  # 自定义输出目录
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Probe:
    """一条验收探针：问什么、答案在哪份文档的哪一页、该页的独有关键词。"""

    question: str
    filename: str
    page: int  # 1 起，与接口返回的「第N页」一致
    keyword: str


# 每份文档 = (文件名, 标题, [每页正文])
# 页号即列表下标 + 1，也就是解析后的 page 元数据。
DOCUMENTS: list[tuple[str, str, list[str]]] = [
    (
        "员工手册.pdf",
        "员工手册",
        [
            "第一章 公司概况\n"
            "本公司成立于二零一一年，主营业务为企业级软件与数据服务。"
            "总部设在北京，在成都、深圳各设有研发中心。",
            "第二章 工作时间\n"
            "公司实行标准工时制。办公区开放时间为 9:00 至 18:00，"
            "午休时段为 12:00 至 13:30。员工应在开放时间内完成当日工作安排。",
            "第三章 报销制度\n"
            "行政与差旅费用凭合规发票报销。单笔报销金额不得超过 5000 元，"
            "超出部分须由部门总监审批后方可提交财务。",
            "第四章 设备巡检\n"
            "服务器机房与弱电间实行定期巡检。巡检频率为每周一次，"
            "通常安排在周五下午，巡检记录由运维专员归档。",
            "第五章 年假制度\n"
            "员工入职满一年后享有年假 10 天。此后每满两年增加 1 天，"
            "年假须在当年度内使用，逾期不结转。",
        ],
    ),
    (
        "IT运维规范.pdf",
        "IT 运维规范",
        [
            "第 1 节 数据备份\n"
            "核心数据库每日凌晨 2:00 执行一次全量备份，备份文件异地保存，"
            "保留时长为 30 天，超期自动清理。",
            "第 2 节 账号与密码\n"
            "系统账号密码须包含大小写字母、数字与符号。密码有效期 90 天，"
            "到期强制更换，不得与前三次使用过的密码重复。",
            "第 3 节 故障分级响应\n"
            "故障按影响面分为三级。一级（P1）故障响应时限 15 分钟，"
            "二级（P2）为 2 小时，三级（P3）为 24 小时。",
            "第 4 节 网络与容量\n"
            "生产环境数据库实例的最大连接数配置为 200，"
            "连接池上限须低于该值，预留 20 个连接供运维排查使用。",
        ],
    ),
    (
        "售后服务制度.pdf",
        "售后服务制度",
        [
            "第一条 保修范围\n"
            "产品自客户签收之日起提供保修 24 个月。人为损坏、"
            "私自拆机及不可抗力导致的故障不在保修范围内。",
            "第二条 退换货规则\n"
            "客户享有 7 天无理由退货权利，15 天内可申请换货。"
            "退货商品需保持外观完好并附全部随机配件。",
            "第三条 工单响应时效\n"
            "客户提交的售后工单，客服须在 48 小时内首次响应并给出处理方案，"
            "复杂问题应同步告知预计解决时间。",
            "第四条 备件供应\n"
            "确认需要更换备件的，常用备件应在 5 个工作日内发出；"
            "非常用备件需向厂商订货，发出时间另行通知。",
        ],
    ),
]

# 探针问题：用于验证召回命中与页码溯源是否正确。
#
# keyword 是该页**独有**的关键字符串（全库只在这一页出现），有两个用途：
#   1. 自洽检查——证明「第 N 页」确实装着这条事实，排除期望页写错的可能；
#   2. 唯一性检查——若某关键词在多个页面出现，"引用了第 N 页"就不再是
#      可证伪的判据（引对了也可能是撞上的）。
# 注意 keyword 用于**归一化后**的子串匹配（去掉空白），因为 PDF 抽取会在
# 数字两侧插入空格（原文「年假 10 天」抽出来是「年假  10  天」）。
PROBES: list[Probe] = [
    Probe("上班时间是几点？", "员工手册.pdf", 2, "9:00"),
    Probe("报销金额上限是多少？", "员工手册.pdf", 3, "5000"),
    Probe("设备多久巡检一次？", "员工手册.pdf", 4, "每周"),
    Probe("一年有几天年假？", "员工手册.pdf", 5, "10天"),
    Probe("数据库备份保留多少天？", "IT运维规范.pdf", 1, "30天"),
    Probe("密码多久需要更换一次？", "IT运维规范.pdf", 2, "90天"),
    Probe("P1 级故障的响应时限是多久？", "IT运维规范.pdf", 3, "15分钟"),
    Probe("数据库允许的最大连接数是多少？", "IT运维规范.pdf", 4, "200"),
    Probe("产品保修期是多长时间？", "售后服务制度.pdf", 1, "24个月"),
    Probe("无理由退货是几天？", "售后服务制度.pdf", 2, "7天"),
    Probe("售后工单多久响应？", "售后服务制度.pdf", 3, "48小时"),
    Probe("备件多久能发出？", "售后服务制度.pdf", 4, "5个工作日"),
]

# Windows 中文字体候选（从前往后取第一个存在的）。
# 用 TTF 而不是 TTC：fpdf2 对字体集合（.ttc）需要额外指定索引，simhei 是单文件更省事。
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\Deng.ttf",
)


def _norm(text: str) -> str:
    """去掉全部空白，用于子串匹配。

    PDF 抽取会在数字两侧补空格、在长行处断行，直接做子串匹配会漏，
    而这些空格的位置取决于字体度量与页面宽度，不是稳定的。
    """
    return "".join(text.split())


def find_font() -> Path:
    """返回一个可用的中文字体路径。

    Raises:
        FileNotFoundError: 候选字体都不存在。
    """
    for candidate in _FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(
        "找不到中文字体，候选：" + "、".join(_FONT_CANDIDATES)
    )


def build_pdf(title: str, pages: list[str], font: Path, out_path: Path) -> None:
    """把每页文字写成一个 PDF 文件。"""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_font("cjk", "", str(font))
    pdf.set_auto_page_break(auto=True, margin=15)

    for text in pages:
        pdf.add_page()
        pdf.set_font("cjk", size=16)
        pdf.multi_cell(0, 10, title)
        pdf.set_font("cjk", size=12)
        pdf.ln(2)
        pdf.multi_cell(0, 8, text)

    # 不写页脚。踩过一次：页脚用 set_y(-20) 定位后，y + 行高刚好越过
    # 自动分页线，每页的页脚都被挤成独立的一页（5 页变 10 页），
    # 页码与内容的对应关系整体错位。而且页脚会往正文里注入
    # 「第 N 页」这种本不存在的字符串——它恰好是页码溯源测试的敏感词，
    # 留在正文里会让测试变得不干净。
    pdf.output(str(out_path))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="生成中文多页 PDF 夹具")
    parser.add_argument("--out", default="tests/fixtures/pdf", help="输出目录")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    font = find_font()
    print(f"字体：{font}")

    for filename, title, pages in DOCUMENTS:
        target = out_dir / filename
        build_pdf(title, pages, font, target)
        print(f"  {filename}  {len(pages)} 页  {target.stat().st_size / 1024:.1f} KB")

    print(f"\n共 {len(DOCUMENTS)} 份 / {sum(len(p) for _, _, p in DOCUMENTS)} 页 → {out_dir}")

    # 生成时就把「探针写错页」和「关键词不唯一」挡掉。
    # 这两类错误不会让脚本报错，只会让后续验收静默变成假阴性
    # （明明检索正确却判 FAIL），是最难排查的一种。
    problems: list[str] = []
    for probe in PROBES:
        pages = next((pages for name, _, pages in DOCUMENTS if name == probe.filename), None)
        if pages is None:
            problems.append(f"探针指向不存在的文档：{probe.filename}")
            continue
        if not 1 <= probe.page <= len(pages):
            problems.append(f"{probe.filename} 只有 {len(pages)} 页，探针要求第 {probe.page} 页")
            continue
        if _norm(probe.keyword) not in _norm(pages[probe.page - 1]):
            problems.append(
                f"{probe.filename} 第{probe.page}页 不含关键词「{probe.keyword}」"
            )
        occurrences = [
            (name, index + 1)
            for name, _, pages in DOCUMENTS
            for index, text in enumerate(pages)
            if _norm(probe.keyword) in _norm(text)
        ]
        if len(occurrences) > 1:
            problems.append(
                f"关键词「{probe.keyword}」在 {occurrences} 多处出现，页码判据不可证伪"
            )

    if problems:
        print("\n夹具自检未通过：")
        for item in problems:
            print(f"  ✗ {item}")
        return 1

    print(f"夹具自检通过：{len(PROBES)} 条探针的期望页均含其独有关键词（全库唯一）")
    print("探针：" + "；".join(f"{p.question}→{p.filename}第{p.page}页" for p in PROBES[:3]) + " …")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
