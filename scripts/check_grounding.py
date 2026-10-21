# -*- coding: utf-8 -*-
"""防幻觉兜底自检：接地性核验（grounding）+ 整批最高分闸门。

为什么单独成一个脚本（而不是塞进 check_chat.py）：
    check_chat 测的是「链路分支对不对」，本脚本测的是「**判据**对不对」——
    核验器的难点全在边角：列表序号算不算事实、千分位与全角怎么归一、
    `48` 会不会被 `1480` 蒙混、流式下 block 该不该生效。
    这些用例几乎全是纯函数调用，与图/检索无关，混在一起会让
    check_chat 的 2000 行再长一截，而失败原因反而更难定位。

判据（每条都对应一个真实踩过的坑或一个明确的产品决定）：

    | 检查项              | 期望                                                    |
    |--------------------|--------------------------------------------------------|
    | 数字归一            | 1,000 ≡ 1000；２４ ≡ 24；全角/半角不误判               |
    | 型号归一            | XG-8800 ≡ xg 8800（大小写、连字符、空格都不敏感）       |
    | 数字边界            | 48 不能靠 1480 通过（否则漏报一半）                     |
    | 列表序号            | 「1. 2. 3.」不算事实（q02/q04 曾整题误判为 low）        |
    | 引用序号            | 「[1]」不算事实（那是我们加的标注）                     |
    | 中文数字            | 「四十八小时」不抽（已知漏检面，写文档而不是假装没漏）  |
    | 分级·单个数字       | → warn（量词、同义改写都落这里，拦下来是误杀）          |
    | 分级·两个以上数字   | 占比过半 → low                                          |
    | 分级·型号/日期      | → low（编不出巧合）                                     |
    | 分级·引用越界       | → low                                                   |
    | 三态动作            | off 不核验 / flag 只告警 / block 仅非流式拦下           |
    | 流式                | block 退化为 flag；done 恒带 grounding 键               |
    | 整批闸门            | 最高分 < 阈值 → 视为无可用片段；reranker=None 时不判    |
    | 配置校验            | 拼错 action 必须拒绝启动（否则静默不生效）              |

运行（**不需要 Milvus / BGE / Ollama**）：
    .venv\\Scripts\\python.exe scripts\\check_grounding.py

退出码：0 全部通过；1 存在失败项。
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from app.core.config import Settings, get_settings  # noqa: E402
from app.core.exceptions import ConfigurationError  # noqa: E402
from app.domain.models import Chunk, RetrievedChunk  # noqa: E402
from app.rag.grounding import STATUS_LOW, STATUS_OK, STATUS_WARN, audit_answer  # noqa: E402
from app.rag.nodes import make_rerank_node  # noqa: E402
from app.rag.state import NO_ANSWER_TEXT  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402

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


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

CONTEXT = (
    "第一条 保修范围：产品自客户签收之日起提供保修 24 个月，"
    "型号 XG-8800 与 XG 9000 均在范围内。人为损坏、私自拆机不在保修范围内。"
    "单笔报销上限 1,000 元。生效日期 2026 年 10 月 17 日。"
)


def _sources(*texts: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(
                id=f"c{index}",
                kb_id="kb",
                doc_id="doc",
                text=text,
                page_start=1,
                page_end=1,
                filename="售后服务制度.pdf",
            ),
            rerank_score=5.0,
        )
        for index, text in enumerate(texts or (CONTEXT,), start=1)
    ]


class _StaticGraph:
    """只回放固定结果的图：ChatService 只用 `.invoke`，不需要真跑节点。"""

    def __init__(self, **payload) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def invoke(self, state):  # noqa: ANN001, ANN201
        self.calls.append(dict(state))
        return dict(self.payload)


class _StaticLLM:
    def __init__(self, answer: str = "") -> None:
        self._answer = answer

    def generate(self, system_prompt: str, user_prompt: str) -> str:  # noqa: ANN001
        return self._answer

    async def stream(self, system_prompt: str, user_prompt: str):  # noqa: ANN001
        # 逐字推，才能验证"流式下 answer 不被替换、delta 拼回原文"
        for char in self._answer:
            yield char


def _service(answer: str, ranked: list[RetrievedChunk], *, refused: bool = False) -> ChatService:
    graph = _StaticGraph(ranked=ranked, answer=answer, refused=refused, retry_count=0)
    return ChatService(graph=graph, llm=_StaticLLM(answer), retrieval_graph=graph)


@contextmanager
def _action(value: str):
    """临时改 grounding_action。

    直接改 Settings 单例的属性：nodes/service 每次调用都 `get_settings()`，
    拿到的是同一个对象，所以赋值即可生效，不必改 .env 或重启进程。
    `lru_cache` 保证它确实是单例——这一点由本脚本第 6 组顺带断言。
    """
    settings = get_settings()
    original = settings.grounding_action
    settings.grounding_action = value
    try:
        yield settings
    finally:
        settings.grounding_action = original


@contextmanager
def _floor(value: float):
    """临时改 rerank_min_top_score（整批最高分闸门）。"""
    settings = get_settings()
    original = settings.rerank_min_top_score
    settings.rerank_min_top_score = value
    try:
        yield settings
    finally:
        settings.rerank_min_top_score = original


# ---------------------------------------------------------------------------
# 1. 抽取与归一：决定"什么算事实"
# ---------------------------------------------------------------------------


def check_extract() -> None:
    section("1. 抽取与归一（什么算事实）")
    src = _sources()

    plain = audit_answer("保修 24 个月。", src)
    check("原文照抄 → 全部支持", plain.status == STATUS_OK and plain.checked == 1, str(plain))

    thousands = audit_answer("单笔报销上限 1000 元。", src)
    check(
        "「1,000」与「1000」视为同一个（千分位归一）",
        thousands.status == STATUS_OK and not thousands.unsupported,
        f"{thousands.unsupported}",
    )

    fullwidth = audit_answer("保修 ２４ 个月。", src)
    check(
        "全角「２４」≡ 半角「24」（NFKC 归一）",
        fullwidth.status == STATUS_OK and not fullwidth.unsupported,
        f"{fullwidth.unsupported}",
    )

    spec_case = audit_answer("型号 xg-8800 在保修范围内。", src)
    check(
        "型号大小写不敏感（xg-8800 ≡ XG-8800）",
        spec_case.status == STATUS_OK and not spec_case.unsupported,
        f"{spec_case.unsupported}",
    )

    spec_space = audit_answer("型号 XG 8800 在保修范围内。", src)
    check(
        "型号空格不敏感（XG 8800 ≡ XG-8800）",
        spec_space.status == STATUS_OK and not spec_space.unsupported,
        f"{spec_space.unsupported}",
    )

    boundary = audit_answer("保修 1480 个月。", src)
    check(
        "「48」不能靠「1480」通过（前后边界）",
        "1480" in boundary.unsupported,
        f"{boundary.unsupported}",
    )

    comma = audit_answer("型号 XG-8800 生效日期 2026 年 10 月 17 日。", src)
    check(
        "中文逗号后的数字不被当千分位吃掉（曾导致型号/日期全线误报）",
        comma.status == STATUS_OK and not comma.unsupported,
        f"{comma.unsupported}",
    )

    listed = audit_answer("不在保修范围内的情形：\n1. 人为损坏。\n2. 私自拆机。", src)
    check(
        "列表序号「1. 2.」不算事实（q02/q04 曾整题误判 low）",
        listed.status == STATUS_OK and listed.checked == 0,
        f"status={listed.status} unsupported={listed.unsupported}",
    )

    cited = audit_answer("保修 24 个月。（来源：[1]）", src)
    check(
        "引用序号「[1]」不算事实（系统标注，不是模型生成）",
        cited.checked == 1 and not cited.unsupported,
        f"checked={cited.checked} unsupported={cited.unsupported}",
    )

    chinese = audit_answer("保修四十八个月。", src)
    check(
        "中文数字不抽（已知漏检面：宁漏不误杀）",
        chinese.checked == 0 and chinese.status == STATUS_OK,
        f"checked={chinese.checked}",
    )


# ---------------------------------------------------------------------------
# 2. 分级：决定"多可疑才算不可信"
# ---------------------------------------------------------------------------


def check_status() -> None:
    section("2. 分级（ok / warn / low）")
    src = _sources()

    one_number = audit_answer("按 3 个步骤操作即可。", src)
    check(
        "单个孤立数字对不上 → warn（量词/同义改写都落这里，拦下即误杀）",
        one_number.status == STATUS_WARN,
        f"status={one_number.status}",
    )

    two_numbers = audit_answer("分别在 72 小时和 96 小时内处理。", src)
    check(
        "两个数字全对不上 → low",
        two_numbers.status == STATUS_LOW,
        f"status={two_numbers.status} unsupported={two_numbers.unsupported}",
    )

    spec = audit_answer("型号 XG-9999 不在保修范围内。", src)
    check(
        "型号对不上 → low（这类标识编不出巧合）",
        spec.status == STATUS_LOW,
        f"status={spec.status}",
    )

    date = audit_answer("2026 年 12 月 1 日起生效。", src)
    check("日期对不上 → low", date.status == STATUS_LOW, f"status={date.status}")

    ref = audit_answer("保修 24 个月。（来源：[5]）", src)
    check(
        "引用越界 → low（引用了本轮不存在的资料）",
        ref.status == STATUS_LOW and ref.invalid_refs == (5,),
        f"status={ref.status} refs={ref.invalid_refs}",
    )

    none_at_all = audit_answer("人为损坏不在保修范围内。", src)
    check(
        "没有可核验的硬事实 → ok，coverage=1.0（不适用不等于差）",
        none_at_all.status == STATUS_OK
        and none_at_all.checked == 0
        and none_at_all.coverage == 1.0,
        f"checked={none_at_all.checked} cov={none_at_all.coverage}",
    )

    empty = audit_answer("", src)
    check("空回答 → ok、不核验（拒答路径走 refused 判定）", empty.status == STATUS_OK and empty.checked == 0)

    full = audit_answer("保修 24 个月，上限 1000 元。", src)
    check(
        "全部命中 → coverage=1.0、warnings 为空",
        full.coverage == 1.0 and not full.warnings,
        f"{full.warnings}",
    )

    # 同一 token 出现多次只算一项、只报一次
    dup = audit_answer("72 小时。72 小时内处理。", src)
    check(
        "同一未支撑数字重复出现只报一次（check 按去重后计数）",
        len(dup.unsupported) == 1,
        f"{dup.unsupported}",
    )


# ---------------------------------------------------------------------------
# 3. 接入：非流式的三态动作
# ---------------------------------------------------------------------------


def check_non_stream() -> None:
    section("3. 接入·非流式（off / flag / block）")
    bad_answer = "型号 XG-9999 在保修范围内。"
    src = _sources()

    with _action("flag"):
        result = _service(bad_answer, src).ask("kb", "XG-9999 保修吗？")
        check(
            "flag：回答原样返回，只带标记（改文字会洗掉证据）",
            result.answer.text == bad_answer
            and result.answer.refused is False
            and result.grounding is not None
            and result.grounding.status == STATUS_LOW,
            f"{result.grounding}",
        )
        check(
            "flag：grounding_warnings 与 grounding.warnings 同源",
            result.grounding is not None
            and result.grounding_warnings == list(result.grounding.warnings),
            f"{result.grounding_warnings}",
        )
        check("flag：citations 照常返回（用户仍能看到依据）", len(result.answer.citations) == 1)

    with _action("off"):
        result = _service(bad_answer, src).ask("kb", "XG-9999 保修吗？")
        check(
            "off：完全不核验（checked=0、status=ok），用于对照实验",
            result.grounding is not None
            and result.grounding.checked == 0
            and result.grounding.status == STATUS_OK,
            f"{result.grounding}",
        )

    with _action("block"):
        result = _service(bad_answer, src).ask("kb", "XG-9999 保修吗？")
        check(
            "block + low：回答降级为拒答话术（宁可说没有，也不给编的）",
            result.answer.text == NO_ANSWER_TEXT and result.answer.refused is True,
            f"{result.answer.text!r}",
        )
        check("block：citations 清空（拒答不带引用）", result.answer.citations == [])
        check(
            "block：报告里 blocked=True（否则运维分不清「被拦」与「真没检索到」）",
            result.grounding is not None and result.grounding.blocked is True,
            f"{result.grounding}",
        )
        check(
            "block：告警仍然保留（诊断信息不能随降级一起丢）",
            bool(result.grounding and result.grounding.warnings),
        )

    with _action("block"):
        warn_answer = "按 3 个步骤操作即可。"
        result = _service(warn_answer, src).ask("kb", "怎么操作？")
        check(
            "block + warn：不拦（单个数字对不上属措辞问题，不是编造）",
            result.answer.text == warn_answer and result.answer.refused is False,
            f"{result.answer.text!r}",
        )

    with _action("flag"):
        result = _service(NO_ANSWER_TEXT, src, refused=True).ask("kb", "无关问题？")
        check(
            "拒答路径：grounding 为 None、无告警（没有回答可查，也不伪造一个 ok）",
            result.answer.refused is True
            and result.grounding is None
            and not result.grounding_warnings,
            f"{result.grounding}",
        )


# ---------------------------------------------------------------------------
# 4. 接入：流式（block 为什么退化为 flag）
# ---------------------------------------------------------------------------


def _collect(service: ChatService, **kwargs) -> list[dict]:
    async def _run() -> list[dict]:
        events: list[dict] = []
        async for event in service.ask_stream("kb", kwargs.pop("question", "XG-9999 保修吗？"), **kwargs):
            events.append(event)
        return events

    return asyncio.run(_run())


def check_stream() -> None:
    section("4. 接入·流式（done 事件是唯一的事后通道）")
    bad_answer = "型号 XG-9999 在保修范围内。"
    src = _sources()

    with _action("block"):
        events = _collect(_service(bad_answer, src), enable_retry=False)
        done = [event for event in events if event["type"] == "done"]
        check("流式：done 事件恒带 grounding 键", bool(done) and "grounding" in done[0], f"{done[:1]}")
        check(
            "流式：block 退化为 flag —— 答案**不被替换**（token 已发出，收不回来）",
            done and done[0]["answer"] == bad_answer and done[0]["refused"] is False,
            f"{done[0]['answer']!r}" if done else "无 done",
        )
        check(
            "流式：status=low 仍要如实上报（前端据此提示核对引用）",
            done and done[0]["grounding"]["status"] == STATUS_LOW,
            f"{done[0]['grounding']}" if done else "无 done",
        )
        deltas = "".join(event["content"] for event in events if event["type"] == "delta")
        check("流式：delta 拼起来 == done.answer", deltas == bad_answer, f"{deltas!r}")

    with _action("flag"):
        events = _collect(_service(NO_ANSWER_TEXT, [], refused=True), enable_retry=False)
        done = [event for event in events if event["type"] == "done"]
        check(
            "流式拒答：grounding 为 None（没有可核验的内容，不伪造一个 ok）",
            done and done[0]["grounding"] is None,
            f"{done[0]['grounding']}" if done else "无 done",
        )


# ---------------------------------------------------------------------------
# 5. 整批最高分闸门（检索侧兜底）
# ---------------------------------------------------------------------------


class _FixedReranker:
    """按登记表给分的假重排器：分数写死，便于精确压阈值线。"""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls = 0

    def rerank(self, query, candidates, top_n=None, min_score=None):  # noqa: ANN001, ANN201
        self.calls += 1
        out = []
        for item, score in zip(candidates, self._scores, strict=False):
            out.append(
                RetrievedChunk(
                    chunk=item.chunk,
                    vector_score=item.vector_score,
                    bm25_score=item.bm25_score,
                    fusion_score=item.fusion_score,
                    rerank_score=score,
                )
            )
        return out[:top_n] if top_n else out


def check_floor() -> None:
    section("5. 整批最高分闸门（rerank_min_top_score）")
    candidates = _sources(CONTEXT, CONTEXT)  # 两条候选，分数分别 0.8 / 0.3
    state = {"question": "备件能加急吗？", "candidates": candidates, "rerank_k": 4}

    with _floor(0.0):
        node = make_rerank_node(_FixedReranker([0.8, 0.3]))
        out = node(state)
        check(
            "阈值 0（默认关闭）：低分片段照常保留",
            len(out["ranked"]) == 2,
            f"{len(out['ranked'])} 条",
        )

    with _floor(1.5):
        node = make_rerank_node(_FixedReranker([0.8, 0.3]))
        out = node(state)
        check(
            "最高分 0.8 < 1.5 → 整批视为不可用（ranked 清空，走重试/拒答）",
            out["ranked"] == [],
            f"{out['ranked']}",
        )

    with _floor(1.5):
        node = make_rerank_node(_FixedReranker([1.8, 0.3]))
        out = node(state)
        check(
            "最高分 1.8 ≥ 1.5 → 保留（单条低分不影响整批）",
            len(out["ranked"]) == 2,
            f"{len(out['ranked'])} 条",
        )

    with _floor(1.5):
        node = make_rerank_node(None)  # type: ignore[arg-type]
        out = node(state)
        check(
            "reranker=None（分数全为 None）→ 闸门不生效，按融合分截断",
            len(out["ranked"]) == 2,
            f"{len(out['ranked'])} 条",
        )

    with _floor(1.5):
        node = make_rerank_node(_FixedReranker([0.8, 0.3]))
        out = node(state)
        check(
            "清空后 rerank 节点仍然只返回 ranked（不改写候选，便于排查）",
            "candidates" not in out,
            f"{list(out)}",
        )


# ---------------------------------------------------------------------------
# 6. 配置
# ---------------------------------------------------------------------------


def check_config() -> None:
    section("6. 配置校验与单例")

    check(
        "默认动作是 flag（默认只观测，不擅自把回答换成拒答）",
        Settings.model_fields["grounding_action"].default == "flag",
        str(Settings.model_fields["grounding_action"].default),
    )
    check(
        "整批闸门默认关闭（0.0），要用必须在自己的语料上标定",
        float(Settings.model_fields["rerank_min_top_score"].default) == 0.0,
    )

    settings = get_settings()
    check(
        "get_settings 是单例（自检靠它临时改配置，不是巧合）",
        get_settings() is settings,
    )

    # 拼错动作名必须拒绝启动：否则"功能开了却什么都没发生"，最难查的静默失效
    broken = Settings(grounding_action="blcok")
    try:
        broken.validate_runtime()
    except ConfigurationError as exc:
        check(
            "GROUNDING_ACTION 拼错 → 启动被拒，消息点名该配置",
            "GROUNDING_ACTION" in str(exc),
            str(exc)[:120],
        )
    else:
        check("GROUNDING_ACTION 拼错 → 启动被拒", False, "没有抛 ConfigurationError")

    negative = Settings(rerank_min_top_score=-1.0)
    try:
        negative.validate_runtime()
    except ConfigurationError as exc:
        check(
            "RERANK_MIN_TOP_SCORE 为负 → 启动被拒（负阈值等于永不触发）",
            "RERANK_MIN_TOP_SCORE" in str(exc),
            str(exc)[:120],
        )
    else:
        check("RERANK_MIN_TOP_SCORE 为负 → 启动被拒", False, "没有抛 ConfigurationError")

    with _action("block"):
        service = _service("x", [])
        check(
            "ChatService 每次构造时读配置（动作不是写死的）",
            service.grounding_action == "block",
            service.grounding_action,
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    print("防幻觉兜底自检：接地性核验 + 整批最高分闸门")
    check_extract()
    check_status()
    check_non_stream()
    check_stream()
    check_floor()
    check_config()

    print()
    if _failures:
        print(f"失败 {len(_failures)} / {_checks} 项：")
        for item in _failures:
            print(f"  - {item}")
        return 1
    print(f"全部通过：{_checks} 项")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
