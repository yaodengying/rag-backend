# -*- coding: utf-8 -*-
"""回答接地性核验（grounding）：生成**之后**、返回**之前**的一道事后闸门。

为什么需要这一层（它和前面几道闸门的区别）：

    检索侧的闸门（rerank_min_score、`ranked` 为空就短路拒答）拦的是
    「**没找到**资料」。但检索幻觉最常见的形态不是"没找到还硬答"，而是
    「**找到了、但找错了**」——召回到一份主题相近的文档，模型照着它答得
    通顺、信誓旦旦，数字和型号却是那份文档里的。
    这类错误：
      · 提示词挡不住（模型确实遵守了"只依据参考资料"，只是资料错了）；
      · 拒答判据挡不住（`ranked` 非空，refused=False）；
      · 人工抽检也很难发现（句子读起来毫无破绽）。

    唯一可自动化的抓手是**可证伪的硬事实**：数字、金额、百分比、日期、
    型号。它们要么在参考资料里逐字出现，要么就是模型自己编的——
    没有第三种解释。这也是 `prompts.py` 里第 2 条禁令想约束的东西，
    只是提示词只能"劝"，这一层负责"查"。

为什么只查硬事实，不查「语义是否被支持」：
    语义级校验（NLI / LLM-as-judge）能抓住"改写式幻觉"（把"压力前放量区
    下轨"说成"箱体上轨的下方"），但要额外一次模型推理——本机 7B 一次
    ~2 秒，而 `/chat/stream` 的答案是逐 token 推给用户的，事后判定的
    结果**追不回流里已经发出去的字**。硬事实校验是纯字符串运算，微秒级，
    可以在两条路径上都跑，且零假阴性代价可控。两者是互补关系，
    不是替代关系（见 docs/anti-hallucination.md 的分层表）。

判据的分寸（宁漏勿误杀，这条很重要）：
    误报的代价是运维从此不看这个告警，等于这一层不存在。所以：
      · 引用序号 `[1]` 不算事实（它是我们自己的标注，不是模型编的内容）；
      · `1,000` 与 `1000`、全角与半角、`XG-8800` 与 `xg 8800` 视为同一个；
      · 单个孤立数字对不上 → 只 `warn`（可能是"3 个步骤"这类量词）；
      · 型号或日期对不上 → `low`（这类标识编不出巧合，写错就是错的）；
      · 两个以上数字对不上且占比过半 → `low`。

    反过来，中文数字（"四十八小时"）、同义改写（"48 小时"→"两天"）
    查不出来，这是本层**已知的漏检面**，写在这里而不是假装没有。

接入方式：
    纯函数、无 I/O、不依赖 LLM——`chat_service` 的两条路径（`ask` /
    `ask_stream`）都调同一个 `audit_answer`，口径不可能分叉。
    `_audit_answer`（只查引用越界）保留不动：它是本模块的一个特例，
    `check_chat.py` 第 6 组按旧签名钉着它。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from app.domain.models import RetrievedChunk

# 状态：ok（没有可疑之处）/ warn（有迹象，但不影响返回）/ low（回答不可信）
STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_LOW = "low"

# 引用序号，如「(来源：[1])」「[2]」。核验前先剥掉：
# 它是系统加给模型的标注，不是模型生成的事实，混进数字里会造成大量误报。
_CITATION_MARK = re.compile(r"\[\s*\d{1,3}\s*\]")

# 列表序号，如「1. 」「2、」「3) 」。核验前必须剥掉：
# 它是排版，不是事实。实测误报就出在这里——q02「以下情况不在保修范围内：
# 1. 人为损坏。2. 私自拆机。3. 不可抗力」被判成"3 个数字都是编的"，
# 整题 status=low。序号只可能出现在行首或句读之后，且后面不能紧跟数字
# （否则「5.5 元」会被截成「5 元」）。
_LIST_MARK = re.compile(r"(^|[\n；;。！？：:])\s*\d{1,2}\s*[.、)．](?![0-9])", re.MULTILINE)

# 日期：2026-10-17 / 2026/10/17 / 2026年10月17日
_DATE = re.compile(r"\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}\s*日?")

# 型号：字母开头 + 数字（XG-8800、AB100、HX_2000、XG 8800）。最少 2 个字母 +
# 2 位数字，避免把 "第3" 这种中文序号里的单个数字误当成型号。
# 分隔符允许空格：模型常把 `XG-8800` 写成 `XG 8800`，那是同一个东西；
# 不允许的话这个 token 会被拆成裸数字 8800，反过来匹配失败（实测踩过）。
_SPEC = re.compile(r"(?<![0-9A-Za-z])[A-Za-z]{2,}[-_ ]?\d{2,}[0-9A-Za-z-]*")

# 数字（含小数与百分号）。千分位由归一化处理，这里不写成 `1,000` 的形式。
_NUMBER = re.compile(r"(?<![0-9A-Za-z.])\d+(?:\.\d+)?%?")

# 归一化：把数字之间的千分位逗号去掉（必须在去空白之前做）。
# ⚠️ 必须限定"逗号后正好三位且后面不再跟数字"，不能写成 `(?<=\d),(?=\d)`：
# NFKC 会把中文全角逗号「，」转成半角，而中文句子里的逗号后面紧跟数字极其常见
# （「型号 XG-8800，2026年10月17日生效」）。用宽松写法会把这个逗号吃掉，
# 让 `XG-8800` 的右边界变成数字、`2026年` 的左边界变成数字——
# 结果**资料里明明写着的型号和日期被判成"编造的"**（实测踩过，全线误报）。
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
# 归一化：空白、连字符、下划线一律去掉。
# 为什么连字符也要去：`XG-8800` 与 `XG 8800` 在参考资料里都可能出现，
# 而回答里只会有一种写法；不去掉就会把"同一个型号"判成"编造的型号"。
_SEPARATORS = re.compile(r"[\s\-_·]+")


def _normalize(text: str) -> str:
    """把文本压成可比较的形态：全角→半角、去千分位、去空白与连字符、转小写。

    NFKC 负责全角数字/字母/百分号（本机中文语料里混用很常见）。
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = _THOUSANDS.sub("", text)
    text = _SEPARATORS.sub("", text)
    return text.lower()


@dataclass(frozen=True, slots=True)
class GroundingReport:
    """一次接地性核验的结果。

    Attributes:
        status: `ok` / `warn` / `low`，语义见模块文档「判据的分寸」
        checked: 核验过的硬事实条数（去重后）
        unsupported: 在参考资料里找不到的硬事实（去重、按出现顺序）
        invalid_refs: 越界的来源序号
        warnings: 人类可读告警（进日志，也回传给调用方）
        blocked: 是否已被服务层按 `grounding_action=block` 拦下
    """

    status: str = STATUS_OK
    checked: int = 0
    unsupported: tuple[str, ...] = ()
    invalid_refs: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    blocked: bool = False

    @property
    def coverage(self) -> float:
        """硬事实的支持率。没有可核验的事实时为 1.0（不是 0 —— 不适用不该算差）。"""
        if self.checked <= 0:
            return 1.0
        return (self.checked - len(self.unsupported)) / self.checked

    @property
    def suspicious(self) -> bool:
        """回答是否**值得用户警惕**：`warn` 与 `low` 都算。

        为什么不写成 `bool(self.unsupported)`：
            没有任何硬事实可核验时（回答里全是"通常""建议先确认"这类
            不含数字与型号的表述），`unsupported` 是空的——但那种情况是
            **不适用**，不是"全部都对"。

        三种 status 的含义见 `audit_answer`：
            ok    → 该核验的都核验过且对得上（或没有可核验项）
            warn  → 有少量数字不在引用里，可能是模型自己的表述
            low   → 支持率过低，多半是编的
        只有 ok 才不算可疑；warn 也要如实提示，否则用户没法分辨
        "文档里写的"和"模型顺口说的"。
        """
        return self.status != STATUS_OK


def _extract_facts(answer: str) -> list[tuple[str, str]]:
    """按出现顺序抽出回答里的硬事实，返回 [(kind, 原样 token)]。

    先抽日期与型号，再把它们占过的区间从数字扫描里排除——否则
    `2026-10-17` 会被拆成 2026 / 10 / 17 三个数字，一条日期错报成三条。

    抽取前先做两件事，都是实测踩出来的：
      · NFKC 归一（全角数字/标点与半角混排）；
      · 剥掉引用序号与列表序号——它们是**我们加的标注**和**排版**，
        不是模型生成的事实。
    """
    text = unicodedata.normalize("NFKC", answer or "")
    text = _CITATION_MARK.sub(" ", text)
    # 保留分隔符（行首/标点），否则「。1. 」「。2. 」会粘成一句，后续断句全乱
    text = _LIST_MARK.sub(lambda m: m.group(1) + " ", text)
    claimed: list[tuple[int, int]] = []
    found: list[tuple[int, str, str]] = []

    for kind, pattern in (("date", _DATE), ("spec", _SPEC)):
        for match in pattern.finditer(text):
            found.append((match.start(), kind, match.group(0)))
            claimed.append(match.span())

    for match in _NUMBER.finditer(text):
        start = match.start()
        if any(begin <= start < end for begin, end in claimed):
            continue
        found.append((start, "number", match.group(0)))

    found.sort(key=lambda item: item[0])
    return [(kind, raw) for _, kind, raw in found]


def _is_supported(token: str, haystack: str) -> bool:
    """判断一个硬事实是否出现在（已归一化的）参考资料里。

    用前后边界而不是裸子串：`48` 不能靠 `1480` 蒙混过关，
    `XG-8800` 也不能靠 `XG-88001` 蒙混过关。
    中文左右两侧天然满足边界（`[0-9a-z]` 排除），所以不需要额外处理。
    """
    normalized = _normalize(token)
    if not normalized:
        return True
    pattern = rf"(?<![0-9a-z]){re.escape(normalized)}(?![0-9a-z])"
    return re.search(pattern, haystack) is not None


def _context_text(sources: Sequence[RetrievedChunk]) -> str:
    """把本轮送进模型的全部片段拼成一段（已归一化）的可检索文本。"""
    parts: list[str] = []
    for item in sources:
        chunk = getattr(item, "chunk", item)
        text = getattr(chunk, "text", "") or ""
        if text:
            parts.append(text)
    return _normalize("\n".join(parts))


def audit_answer(answer: str, sources: Sequence[RetrievedChunk]) -> GroundingReport:
    """核验一条回答是否踩在参考资料上。

    Args:
        answer: 模型生成的回答（已经 strip）
        sources: 本轮实际送进模型的片段（即 `ranked`，不是全部 candidates）

    Returns:
        GroundingReport。**不修改 answer**：告警需要能看到模型的原始输出，
        改写会把"它哪里编了"这个证据洗掉。
    """
    if not answer:
        return GroundingReport()

    invalid_refs = tuple(sorted(_invalid_refs(answer, len(sources))))
    facts = _extract_facts(answer)
    haystack = _context_text(sources)

    seen: set[str] = set()
    unsupported: list[str] = []
    kinds: dict[str, str] = {}
    for kind, raw in facts:
        key = _normalize(raw)
        if not key or key in seen:
            continue
        seen.add(key)
        if not _is_supported(raw, haystack):
            unsupported.append(raw)
            kinds[key] = kind

    checked = len(seen)
    warnings: list[str] = []

    if invalid_refs:
        warnings.append(
            f"引用了不存在的来源序号 {list(invalid_refs)}（本轮资料共 {len(sources)} 条）"
        )
    hard = [raw for raw in unsupported if kinds.get(_normalize(raw)) in {"spec", "date"}]
    if hard:
        warnings.append(f"回答里出现资料中不存在的型号/日期：{hard}")
    numbers = [raw for raw in unsupported if kinds.get(_normalize(raw)) == "number"]
    if numbers:
        warnings.append(
            f"回答里有 {len(numbers)} 个数字在资料中找不到：{numbers}（共核验 {checked} 项）"
        )

    status = STATUS_OK
    if invalid_refs or hard:
        # 引用越界与型号/日期对不上都是"编出来的"，没有巧合可解释
        status = STATUS_LOW
    elif numbers:
        ratio = len(numbers) / checked if checked else 0.0
        status = STATUS_LOW if (len(numbers) >= 2 and ratio >= 0.5) else STATUS_WARN

    return GroundingReport(
        status=status,
        checked=checked,
        unsupported=tuple(unsupported),
        invalid_refs=invalid_refs,
        warnings=tuple(warnings),
    )


def _invalid_refs(answer: str, total: int) -> set[int]:
    """回答里越界的来源序号（引用 `[5]` 而当轮只有 2 条资料）。"""
    indices = {int(m.group(0).strip("[] ")) for m in _CITATION_MARK.finditer(answer or "")}
    return {index for index in indices if index < 1 or index > total}


__all__ = [
    "STATUS_LOW",
    "STATUS_OK",
    "STATUS_WARN",
    "GroundingReport",
    "audit_answer",
]
