# -*- coding: utf-8 -*-
"""评估器自检：数据集校验、页码解析、命中判定、聚合与落盘。

为什么需要：
    评估脚本本身也要被评估。这套逻辑里三处最容易"静默算错"：
      1. 页码标签解析（「第2-4页」被当成单页 → 召回率虚高）
      2. 命中判定（文件名对、页码不对也算命中 → 溯源能力被高估）
      3. 聚合（把"算不出"当成 0 分 → 均值被拉低，或反过来被隐藏）

    这些错误不会报错，只会让报告好看或难看，正是最需要钉住的。

全部断言不依赖 Milvus / BGE / Ollama：只跑纯函数与临时目录，秒级完成。

运行：
    .venv\\Scripts\\python.exe scripts\\check_eval.py

退出码：0 全部通过；1 存在失败项。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from app.domain.models import Citation  # noqa: E402
from scripts.rag_eval.dataset import (  # noqa: E402
    DatasetError,
    EvalSample,
    ExpectedSource,
    load_dataset,
    select,
)
from scripts.rag_eval.pipeline import Prediction  # noqa: E402
from scripts.rag_eval.report import build_payload, write_reports  # noqa: E402
from scripts.rag_eval.scoring import (  # noqa: E402
    RetrievalScore,
    _clean,
    aggregate,
    parse_pages,
    retrieval_score,
)

DATASET = PROJECT_ROOT / "eval" / "dataset" / "qa_dataset.jsonl"

_failures: list[str] = []
_checks = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    """记录一项断言结果（收集全部失败项，不只停在第一个）。"""
    global _checks
    _checks += 1
    if condition:
        print(f"  [ok]   {label}")
    else:
        print(f"  [FAIL] {label}" + (f"  -> {detail}" if detail else ""))
        _failures.append(label)


def raises(func, exc_type: type[BaseException]) -> bool:
    """断言调用会抛指定异常。"""
    try:
        func()
    except exc_type:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

def make_sample(
    sample_id: str = "s1",
    *,
    answerable: bool = True,
    expected: tuple[tuple[str, int | None], ...] = (("a.pdf", 1),),
) -> EvalSample:
    return EvalSample(
        id=sample_id,
        kb_id="kb-1",
        question="问题",
        reference="标准答案",
        reference_contexts=("原文",),
        expected_sources=tuple(ExpectedSource(filename=f, page=p) for f, p in expected),
        answerable=answerable,
    )


def make_prediction(
    sample: EvalSample,
    *,
    citations: tuple[tuple[str, str | None], ...] = (),
    refused: bool = False,
    error: str = "",
) -> Prediction:
    return Prediction(
        sample=sample,
        answer="" if error else ("拒答" if refused else "回答"),
        contexts=["上下文"],
        citations=[
            Citation(doc_id="d", filename=name, page=page, snippet="片段")
            for name, page in citations
        ],
        refused=refused,
        latency_ms=1200,
        error=error,
    )


# ---------------------------------------------------------------------------
# 第 1 组：数据集加载与校验
# ---------------------------------------------------------------------------

def check_dataset() -> None:
    print("\n[1] 数据集加载与校验")
    samples = load_dataset(DATASET)

    check("默认数据集可加载", bool(samples), f"得到 {len(samples)} 条")
    check("题数 18", len(samples) == 18, f"实际 {len(samples)}")
    check(
        "可答题 16 条",
        sum(1 for s in samples if s.answerable) == 16,
        f"实际 {sum(1 for s in samples if s.answerable)}",
    )
    check("id 唯一", len({s.id for s in samples}) == len(samples))
    check("每条都有 kb_id", all(s.kb_id for s in samples))
    check(
        "可答题都有 reference",
        all(s.reference for s in samples if s.answerable),
    )
    check(
        "可答题都有 expected_sources（否则召回率无从计算）",
        all(s.expected_sources for s in samples if s.answerable),
    )
    check(
        "应拒答题不设 expected_sources",
        all(not s.expected_sources for s in samples if not s.answerable),
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        def write(name: str, lines: list[str]) -> Path:
            path = tmp_dir / name
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return path

        dup = write(
            "dup.jsonl",
            [
                json.dumps({"id": "x", "kb_id": "k", "question": "q", "reference": "r"}),
                json.dumps({"id": "x", "kb_id": "k", "question": "q2", "reference": "r"}),
            ],
        )
        check("id 重复被拒", raises(lambda: load_dataset(dup), DatasetError))

        no_kb = write("nokb.jsonl", [json.dumps({"id": "x", "question": "q"})])
        check("缺 kb_id 被拒", raises(lambda: load_dataset(no_kb), DatasetError))

        no_ref = write(
            "noref.jsonl",
            [json.dumps({"id": "x", "kb_id": "k", "question": "q", "answerable": True})],
        )
        check("可答但无 reference 被拒", raises(lambda: load_dataset(no_ref), DatasetError))

        bad_page = write(
            "badpage.jsonl",
            [
                json.dumps(
                    {
                        "id": "x",
                        "kb_id": "k",
                        "question": "q",
                        "reference": "r",
                        "expected_sources": [{"filename": "a.pdf", "page": "1"}],
                    }
                )
            ],
        )
        check("page 不是整数被拒", raises(lambda: load_dataset(bad_page), DatasetError))

        broken = write("broken.jsonl", ["{不是 json"])
        check("非法 JSON 被拒", raises(lambda: load_dataset(broken), DatasetError))

        empty = tmp_dir / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        check("空数据集被拒", raises(lambda: load_dataset(empty), DatasetError))

    check("缺文件抛 FileNotFoundError", raises(lambda: load_dataset(Path("不存在.jsonl")), FileNotFoundError))
    check("未知 id 被拒", raises(lambda: select(samples, ids=["不存在"]), DatasetError))
    check("limit 生效", len(select(samples, limit=3)) == 3)


# ---------------------------------------------------------------------------
# 第 2 组：页码解析
# ---------------------------------------------------------------------------

def check_pages() -> None:
    print("\n[2] 页码标签解析")
    check("单页", parse_pages("第3页") == {3}, str(parse_pages("第3页")))
    check("跨页区间", parse_pages("第2-4页") == {2, 3, 4}, str(parse_pages("第2-4页")))
    check("破折号变体（en dash）", parse_pages("第2–4页") == {2, 3, 4}, str(parse_pages("第2–4页")))
    check("带空格", parse_pages("第 5 页") == {5}, str(parse_pages("第 5 页")))
    check("None 返回空集", parse_pages(None) == frozenset())
    check("空串返回空集", parse_pages("") == frozenset())
    check("无法解析返回空集而不是抛错", parse_pages("封面") == frozenset())


# ---------------------------------------------------------------------------
# 第 3 组：命中判定（页码级召回）
# ---------------------------------------------------------------------------

def check_retrieval() -> None:
    print("\n[3] 命中判定")

    def score(expected, citations) -> RetrievalScore:
        return retrieval_score(make_prediction(make_sample(expected=expected), citations=citations))

    equal = score((("a.pdf", 1),), (("a.pdf", "第1页"),))
    check("文件名+页码都对 → 命中", equal.hit == 1.0 and equal.page_recall == 1.0, str(equal))
    check("首个命中 → MRR=1", equal.mrr == 1.0, str(equal.mrr))

    wrong_page = score((("a.pdf", 2),), (("a.pdf", "第1页"),))
    check(
        "文件名对、页码错 → 不命中（溯源能力的最小区分度）",
        wrong_page.hit == 0.0,
        str(wrong_page),
    )

    wrong_file = score((("a.pdf", 1),), (("b.pdf", "第1页"),))
    check("文件名不同 → 不命中", wrong_file.hit == 0.0, str(wrong_file))

    second = score((("a.pdf", 3),), (("b.pdf", "第1页"), ("a.pdf", "第3页")))
    check("第 2 名命中 → MRR=0.5", second.mrr == 0.5, str(second.mrr))

    span = score((("a.pdf", 2),), (("a.pdf", "第1-3页"),))
    check("跨页块覆盖期望页 → 命中", span.hit == 1.0, str(span))

    partial = score((("a.pdf", 1), ("a.pdf", 2)), (("a.pdf", "第1页"),))
    check("多期望来源只中一个 → page_recall=0.5", partial.page_recall == 0.5, str(partial.page_recall))

    no_expected = retrieval_score(make_prediction(make_sample(expected=())))
    check("没有期望来源 → None（不是 0，不参与均值）", no_expected.hit is None)

    failed = retrieval_score(make_prediction(make_sample(), error="UpstreamError"))
    check("样本失败 → None", failed.hit is None)

    refused = retrieval_score(
        make_prediction(make_sample(), citations=(), refused=True)
    )
    check("拒答样本不产生引用 → 不命中", refused.hit == 0.0)

    no_page_expected = score((("a.pdf", None),), (("a.pdf", "第9页"),))
    check("期望只校验文件名时，页码无关", no_page_expected.hit == 1.0)


# ---------------------------------------------------------------------------
# 第 4 组：聚合口径
# ---------------------------------------------------------------------------

def check_aggregate() -> None:
    print("\n[4] 聚合口径")
    good = make_prediction(
        make_sample("s1"), citations=(("a.pdf", "第1页"),)
    )
    refused_ok = make_prediction(make_sample("s2", answerable=False), refused=True)
    refused_wrong = make_prediction(make_sample("s3", answerable=False), refused=False)
    broken = make_prediction(make_sample("s4"), error="UpstreamError: 连不上")

    summary = aggregate([good, refused_ok, refused_wrong, broken], {}, [])

    check("总数", summary["counts"]["total"] == 4, str(summary["counts"]))
    check("可答数（含一题失败的）", summary["counts"]["answerable"] == 2, str(summary["counts"]))
    check("应拒答数", summary["counts"]["refusal_cases"] == 2, str(summary["counts"]))
    check("失败数", summary["counts"]["failed"] == 1, str(summary["counts"]))
    check(
        "拒答正确率 0.5（两题里对了一题）",
        summary["generation"]["refusal_accuracy"] == 0.5,
        str(summary["generation"]["refusal_accuracy"]),
    )
    check(
        "答出率分母剔除失败样本（1 答 / 1 跑通）",
        summary["generation"]["answer_rate"] == 1.0,
        str(summary["generation"]["answer_rate"]),
    )
    check(
        "失败样本进 errors 列表",
        [item["id"] for item in summary["errors"]] == ["s4"],
        str(summary["errors"]),
    )
    check("无 RAGAS 指标时 ragas 段为空", summary["ragas"] == [])

    check("_clean 把 NaN 收敛成 None", _clean(float("nan")) is None)
    check("_clean 保留正常值并四舍五入", _clean(0.1234567) == 0.123457)
    check("_clean 处理 None", _clean(None) is None)


# ---------------------------------------------------------------------------
# 第 5 组：结果落盘
# ---------------------------------------------------------------------------

def check_report() -> None:
    print("\n[5] 结果落盘")
    pred = make_prediction(
        make_sample("s1"), citations=(("售后服务制度.pdf", "第1页"),)
    )
    refusal = make_prediction(make_sample("s2", answerable=False), refused=True)
    predictions = [pred, refusal]

    summary = aggregate(predictions, {"s1": {"faithfulness": 0.75}}, [])
    payload = build_payload(
        dataset_path=DATASET,
        predictions=predictions,
        ragas_scores={"s1": {"faithfulness": 0.75}},
        metrics=[],
        summary=summary,
        config={"llm_model": "qwen2:7b", "llm_base_url": "http://127.0.0.1:11434/v1"},
        duration_s=1.5,
    )

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run"
        written = write_reports(out, payload)

        check("生成四份文件", len(written) == 4, str(sorted(written)))
        for name in ("summary.json", "samples.jsonl", "scores.csv", "report.md"):
            check(f"{name} 存在且非空", (out / name).exists() and (out / name).stat().st_size > 0)

        reloaded = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        check("summary.json 可解析", "summary" in reloaded and "samples" in reloaded)
        check("summary.json 保留逐题记录", len(reloaded["samples"]) == 2)

        rows = (out / "scores.csv").read_text(encoding="utf-8-sig").strip().splitlines()
        check("scores.csv 行数 = 样本数 + 表头", len(rows) == 3, f"实际 {len(rows)} 行")

        markdown = (out / "report.md").read_text(encoding="utf-8")
        check("report.md 有口径章节", "## 六、指标口径与局限" in markdown)
        check("report.md 有逐题明细", "| q" in markdown or "| s1 |" in markdown, "缺逐题表")
        check("report.md 标注应拒答题", "拒答" in markdown)


def main() -> int:
    print("=" * 64)
    print("评估器自检（纯函数 + 临时目录，不依赖 Milvus / BGE / Ollama）")
    print("=" * 64)

    check_dataset()
    check_pages()
    check_retrieval()
    check_aggregate()
    check_report()

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
