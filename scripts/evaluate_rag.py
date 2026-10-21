# -*- coding: utf-8 -*-
"""RAG 评估入口：跑测试问答集，用 RAGAS 打分，落盘结果文件。

用法（在项目根目录执行）：

    # 全量评估（进程内直跑问答链路，不需要先起服务）
    .venv\\Scripts\\python.exe scripts\\evaluate_rag.py

    # 先小样本试跑，确认指标还算得动
    .venv\\Scripts\\python.exe scripts\\evaluate_rag.py --limit 3

    # 只算确定性检索指标（不调裁判模型，几秒钟出结果）
    .venv\\Scripts\\python.exe scripts\\evaluate_rag.py --no-ragas

    # 指定指标 / 数据集 / 输出目录
    .venv\\Scripts\\python.exe scripts\\evaluate_rag.py ^
        --metrics faithfulness,answer_relevancy --out eval\\results\\baseline

    # 只看数据集的题，不跑模型
    .venv\\Scripts\\python.exe scripts\\evaluate_rag.py --dry-run

产物（`--out` 目录下）：
    summary.json   聚合指标（机器读）
    samples.jsonl  逐题明细（含送进模型的上下文原文）
    scores.csv     评分表（Excel 可直接打开）
    report.md      报告（结论、口径、局限）

退出码：
    0  评估跑完且无失败样本
    1  有样本失败，或有指标全部算不出
    2  环境/输入问题（缺数据集、ragas 未安装、裁判模型未配置）
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# .env、SQLite、uploads 都是相对路径 → 必须在项目根下运行。
# 与其在加载失败时让用户猜，不如这里直接切过去（脚本专用，不影响被 import）。
os.chdir(PROJECT_ROOT)

# 本机有全局 http_proxy：裁判模型在 127.0.0.1:11434，走代理会多一跳甚至超时。
for _host in ("localhost", "127.0.0.1", "::1"):
    _current = os.environ.get("NO_PROXY", "")
    if _host not in _current:
        _current = f"{_current},{_host}".strip(",")
    os.environ["NO_PROXY"] = _current
os.environ["no_proxy"] = os.environ["NO_PROXY"]

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# Ollama 的 OpenAI 兼容响应不带 usage，langchain-openai 每次都会对 None 发警告。
# 这是本机后端的能力缺口，不是评估结果的一部分，刷屏会盖住真正的告警。
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message="Unexpected type for token usage")

DEFAULT_DATASET = PROJECT_ROOT / "eval" / "dataset" / "qa_dataset.jsonl"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "eval" / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 RAGAS 评估 RAG 问答链路（检索召回 + 答案质量）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET, help="评估数据集 JSONL"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="结果目录；默认 eval/results/<时间戳>",
    )
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（试跑用）")
    parser.add_argument("--ids", default="", help="只跑指定题号，逗号分隔，如 q01,q05")
    parser.add_argument(
        "--metrics",
        default="faithfulness,answer_relevancy,context_recall,context_precision",
        help="RAGAS 指标，逗号分隔；--no-ragas 时忽略",
    )
    parser.add_argument(
        "--no-ragas", action="store_true", help="只算确定性检索指标，不调裁判模型"
    )
    parser.add_argument(
        "--max-workers", type=int, default=2, help="RAGAS 并发数；本机 7B 建议 1~2"
    )
    parser.add_argument(
        "--timeout", type=int, default=180, help="单次裁判调用的超时秒数"
    )
    parser.add_argument("--no-progress", action="store_true", help="隐藏 RAGAS 进度条")
    parser.add_argument(
        "--dry-run", action="store_true", help="只校验数据集并打印计划，不跑模型"
    )
    return parser.parse_args(argv)


def _config_snapshot() -> dict:
    from app.core.config import get_settings

    settings = get_settings()
    return {
        "llm_model": settings.llm_model,
        "llm_base_url": settings.llm_base_url,
        "embedding_model": Path(settings.embedding_model_path).name,
        "embedding_device": settings.embedding_device,
        "rerank_enabled": bool(settings.rerank_enabled),
        "rerank_model": Path(settings.rerank_model_path).name if settings.rerank_model_path else "",
        "retrieve_top_k": settings.retrieve_top_k,
        "rerank_top_n": settings.rerank_top_n,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "milvus_collection": settings.milvus_collection,
        "answerable_only_for_ragas": True,
    }


def _require_ragas() -> None:
    """提前确认 ragas 可用，别等模型都跑完了才在打分阶段炸。"""
    from scripts.rag_eval.compat import ensure_ragas_importable

    ensure_ragas_importable()
    try:
        import ragas  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 环境问题
        raise SystemExit(
            "ragas 未安装。请在项目 venv 内执行：\n"
            "  .venv\\Scripts\\python.exe -m pip install --no-deps \\\n"
            "      ragas instructor appdirs diskcache nest-asyncio docstring_parser "
            "-i https://pypi.tuna.tsinghua.edu.cn/simple\n"
            "（--no-deps 是刻意的：默认安装会把 openai / langchain-openai 一起降级）\n"
            f"原始错误：{exc}"
        ) from exc


def _print_plan(samples, metric_keys: list[str], out_dir: Path) -> None:
    by_kb: dict[str, int] = {}
    for sample in samples:
        by_kb[sample.kb_id] = by_kb.get(sample.kb_id, 0) + 1
    answerable = sum(1 for s in samples if s.answerable)
    print(f"数据集：{len(samples)} 题（可答 {answerable} / 应拒答 {len(samples) - answerable}）")
    for kb_id, count in by_kb.items():
        print(f"  知识库 {kb_id}：{count} 题")
    print(f"RAGAS 指标：{', '.join(metric_keys) if metric_keys else '（未启用）'}")
    print(f"结果目录：{out_dir}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from scripts.rag_eval.dataset import DatasetError, load_dataset, select

    try:
        samples = load_dataset(args.dataset)
        ids = [item for item in args.ids.split(",") if item.strip()]
        samples = select(samples, ids=ids or None, limit=args.limit)
    except (DatasetError, FileNotFoundError) as exc:
        print(f"[数据集错误] {exc}", file=sys.stderr)
        return 2

    metric_keys = (
        [] if args.no_ragas else [k.strip() for k in args.metrics.split(",") if k.strip()]
    )
    out_dir = args.out or DEFAULT_OUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")

    if metric_keys:
        try:
            _require_ragas()
        except SystemExit as exc:
            print(str(exc), file=sys.stderr)
            return 2
        # 提前校验指标名，避免跑到打分阶段才发现拼错
        from scripts.rag_eval.scoring import available_metric_keys

        unknown = [k for k in metric_keys if k not in available_metric_keys()]
        if unknown:
            print(
                f"[参数错误] 未知指标：{', '.join(unknown)}；"
                f"可选：{', '.join(available_metric_keys())}",
                file=sys.stderr,
            )
            return 2

    _print_plan(samples, metric_keys, out_dir)
    if args.dry_run:
        print("（--dry-run：未执行模型调用）")
        return 0

    started = time.perf_counter()

    # ---- 1. 跑问答链路，收集预测 ----
    from scripts.rag_eval.pipeline import RagPipeline

    pipeline = RagPipeline()
    predictions = []
    print("\n开始逐题提问：")
    for index, sample in enumerate(samples, start=1):
        pred = pipeline.run(sample)
        predictions.append(pred)
        if pred.ok:
            mark = "拒答" if pred.refused else f"引用{len(pred.citations)}"
            print(
                f"  [{index:>2}/{len(samples)}] {sample.id} ✓ {mark} "
                f"{pred.latency_ms}ms"
            )
        else:
            print(f"  [{index:>2}/{len(samples)}] {sample.id} ✗ {pred.error[:90]}")

    # ---- 2. RAGAS 打分 ----
    from scripts.rag_eval import scoring

    ragas_scores: dict[str, dict[str, float | None]] = {}
    metrics = []
    if metric_keys:
        from scripts.rag_eval.adapters import build_judge_embeddings, build_judge_llm

        print("\n装配裁判模型与指标...")
        judge_llm = build_judge_llm()
        judge_emb = build_judge_embeddings()
        metrics = scoring.build_metrics(metric_keys, judge_llm, judge_emb)

        from scripts.rag_eval.compat import ensure_ragas_importable

        ensure_ragas_importable()
        from ragas import RunConfig

        answerable = sum(1 for p in predictions if p.ok and p.sample.answerable)
        print(f"RAGAS 打分：{answerable} 题 × {len(metrics)} 指标（并发 {args.max_workers}）")
        ragas_scores = scoring.score_with_ragas(
            predictions,
            metrics,
            llm=judge_llm,
            embeddings=judge_emb,
            run_config=RunConfig(
                max_workers=max(1, args.max_workers),
                timeout=args.timeout,
                max_retries=2,
                max_wait=30,
            ),
            show_progress=not args.no_progress,
        )

    duration = time.perf_counter() - started

    # ---- 3. 汇总与落盘 ----
    from scripts.rag_eval.report import build_payload, render_console, write_reports

    summary = scoring.aggregate(predictions, ragas_scores, metrics)
    payload = build_payload(
        dataset_path=args.dataset,
        predictions=predictions,
        ragas_scores=ragas_scores,
        metrics=metrics,
        summary=summary,
        config=_config_snapshot(),
        duration_s=duration,
    )
    written = write_reports(out_dir, payload)

    print(render_console(payload))
    print("结果文件：")
    for purpose, path in written.items():
        print(f"  {purpose:<8}{path}")

    # ---- 4. 退出码 ----
    failures = summary["counts"]["failed"]
    empty_metrics = [item["key"] for item in summary["ragas"] if item["valid"] == 0]
    if empty_metrics:
        print(
            f"\n[警告] 这些指标一题都没算出来：{', '.join(empty_metrics)}"
            "（模型输出无法解析或裁判调用失败）",
            file=sys.stderr,
        )
        return 1
    if failures:
        print(f"\n[警告] {failures} 题未跑通，详见 report.md 第五节", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
