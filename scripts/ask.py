# -*- coding: utf-8 -*-
"""单知识库问答脚本：向一个知识库提问，打印答案与**溯源页码**。

为什么做成 HTTP 客户端而不是"直接 import 容器跑一遍"：
    这是这个项目**唯一**会同时走通接口层、服务层、LangGraph 图、
    混合检索、重排与本地大模型的路径。进程内直调会绕过 FastAPI 的
    依赖注入、异常处理器与 SSE 编码，那些恰恰是最容易在重构中悄悄坏掉、
    而单元自检又覆盖不到的部分。

用法（服务需先起来：见 README「本地运行」）：

    # 问一句（库里只有一个知识库时可省略 -k）
    .venv\\Scripts\\python.exe scripts\\ask.py "报销金额上限是多少？"

    # 指定知识库、显示检索明细（逐路分数 + 最终上下文）
    .venv\\Scripts\\python.exe scripts\\ask.py -k <kb_id> --debug "设备多久巡检一次？"

    # 交互模式（连续提问，输入 q 退出）
    .venv\\Scripts\\python.exe scripts\\ask.py -k <kb_id> -i

    # 列出知识库
    .venv\\Scripts\\python.exe scripts\\ask.py --list

    # 一键演示：造中文多页 PPTX → 建库 → 入库 → 提问 → **自动校验溯源页码**
    .venv\\Scripts\\python.exe scripts\\ask.py --demo
    .venv\\Scripts\\python.exe scripts\\ask.py --demo --keep   # 保留演示库

    # 机器可读输出
    .venv\\Scripts\\python.exe scripts\\ask.py --json "..."

退出码：
    0  全部正常
    1  问答失败，或 --demo 的页码校验不通过
    2  环境问题（服务未启动 / 没有可用知识库）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 本机有全局 http_proxy。调 127.0.0.1 的 FastAPI 也走代理会多一跳，
# 且在代理不稳定时表现为"服务没起来"的假象。
for _host in ("localhost", "127.0.0.1", "::1"):
    _current = os.environ.get("NO_PROXY", "")
    if _host not in _current:
        _current = f"{_current},{_host}".strip(",")
    os.environ["NO_PROXY"] = _current
os.environ["no_proxy"] = os.environ["NO_PROXY"]

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

import httpx  # noqa: E402

DEFAULT_API = os.environ.get("RAG_API_BASE", "http://127.0.0.1:8000/api/v1")
DEMO_KB_NAME = "问答演示库 · 自动生成"
TIMEOUT = httpx.Timeout(180.0, connect=5.0)

# 演示用的英文 PDF 页（与 make_sample_docs 的 _PDF_PAGES 对应），
# 用来证明 PDF 的页码溯源同样成立——中文页靠 PPTX，英文页靠 PDF，
# 两条不同 loader 的路径都被覆盖到。
PDF_QUESTION = "What does the document say comes after loading and chunking?"
PDF_EXPECTED_PAGE = 3


# ---------------------------------------------------------------------------
# HTTP 封装
# ---------------------------------------------------------------------------


class ApiError(RuntimeError):
    """接口返回非 2xx。"""


class Client:
    """极简 API 客户端（只包住本脚本用到的几个端点）。"""

    def __init__(self, base: str) -> None:
        self._base = base.rstrip("/")
        self._http = httpx.Client(base_url=self._base, timeout=TIMEOUT)

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *_exc) -> None:
        self._http.close()

    def _call(self, method: str, path: str, **kwargs):
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise SystemExit(
                f"无法连接后端 {self._base}：{exc}\n"
                "请先启动服务（在项目根的 backend 目录下）：\n"
                "  .venv\\Scripts\\python.exe -m uvicorn app.main:app "
                "--host 127.0.0.1 --port 8000"
            ) from exc
        if response.status_code >= 400:
            detail = response.text[:400]
            raise ApiError(f"{method} {path} → {response.status_code}：{detail}")
        return response.json() if response.content else None

    def list_kbs(self) -> list[dict]:
        return self._call("GET", "/knowledge-bases")

    def create_kb(self, name: str, description: str = "") -> dict:
        return self._call(
            "POST", "/knowledge-bases", json={"name": name, "description": description}
        )

    def delete_kb(self, kb_id: str) -> None:
        self._call("DELETE", f"/knowledge-bases/{kb_id}")

    def upload(self, kb_id: str, path: Path) -> dict:
        with path.open("rb") as handle:
            return self._call(
                "POST",
                f"/knowledge-bases/{kb_id}/documents",
                files={"file": (path.name, handle, "application/octet-stream")},
            )

    def search(self, kb_id: str, query: str, top_k: int | None, rerank_k: int | None) -> dict:
        body: dict = {"query": query, "debug": True}
        if top_k:
            body["top_k"] = top_k
        if rerank_k:
            body["rerank_k"] = rerank_k
        return self._call("POST", f"/knowledge-bases/{kb_id}/search", json=body)

    def chat(
        self,
        kb_id: str,
        question: str,
        top_k: int | None = None,
        rerank_k: int | None = None,
    ) -> dict:
        body: dict = {"kb_id": kb_id, "question": question}
        if top_k:
            body["top_k"] = top_k
        if rerank_k:
            body["rerank_k"] = rerank_k
        return self._call("POST", "/chat", json=body)


# ---------------------------------------------------------------------------
# 展示
# ---------------------------------------------------------------------------

_SEPARATOR = "─" * 68


def _pick_kb(api: Client, requested: str | None) -> str:
    """确定要查询的知识库 ID。"""
    if requested:
        return requested
    kbs = api.list_kbs()
    if not kbs:
        raise SystemExit(
            "知识库里还没有任何知识库。可以先跑一键演示：\n"
            "  .venv\\Scripts\\python.exe scripts\\ask.py --demo"
        )
    if len(kbs) > 1:
        print("存在多个知识库，请用 -k 指定（或先看 --list）：")
        for item in kbs:
            print(f"  - {item['id']}  {item['name']}（{item.get('doc_count', 0)} 篇文档）")
        raise SystemExit(2)
    return kbs[0]["id"]


def _print_sources(sources: list[dict]) -> None:
    """打印送入模型的上下文片段（含页码）。"""
    if not sources:
        print("  上下文：空（未召回到可用片段）")
        return
    print(f"  上下文（{len(sources)} 条，已重排）：")
    for index, item in enumerate(sources, start=1):
        page = item.get("page") or "页码未知"
        score = item.get("rerank_score")
        score_text = f" 重排分={score:.3f}" if isinstance(score, float) else ""
        text = (item.get("text") or "").replace("\n", " ")[:60]
        print(f"    [{index}] {item.get('filename')} {page}{score_text}  {text}…")


def _print_answer(payload: dict, debug_payload: dict | None) -> None:
    """打印一次问答结果。"""
    answer = (payload.get("answer") or "").strip()
    print(_SEPARATOR)
    if payload.get("refused"):
        print("【拒答】资料不足，未调用大模型生成自由内容")
    print(answer or "（空回答）")

    citations = payload.get("citations") or []
    if citations:
        print("\n引用来源：")
        for index, item in enumerate(citations, start=1):
            page = item.get("page") or "页码未知"
            print(f"  [{index}] {item.get('filename')} · {page} · doc={item.get('doc_id', '')[:8]}")
            snippet = (item.get("snippet") or "").replace("\n", " ")[:70]
            if snippet:
                print(f"       {snippet}…")
    elif not payload.get("refused"):
        print("\n（模型未给出结构化引用；下面列出的上下文仍是本次回答的依据）")

    if debug_payload is not None:
        print()
        _print_sources(debug_payload.get("results") or [])
        dense = (debug_payload.get("debug") or {}).get("dense") or {}
        bm25 = (debug_payload.get("debug") or {}).get("bm25") or {}
        print(
            f"  召回明细：向量路 {dense.get('count', 0)} 条 / "
            f"BM25 路 {bm25.get('count', 0)} 条 / "
            f"融合后 {len((debug_payload.get('debug') or {}).get('fused') or [])} 条"
        )
    print(_SEPARATOR)


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------


def cmd_list(api: Client) -> int:
    kbs = api.list_kbs()
    if not kbs:
        print("（没有知识库）")
        return 0
    print(f"{'kb_id':38} {'文档':>4}  名称")
    for item in kbs:
        print(f"{item['id']:38} {item.get('doc_count', 0):>4}  {item['name']}")
    return 0


def cmd_ask(api: Client, args: argparse.Namespace, question: str) -> int:
    """问一个问题并打印结果。返回 0/1。"""
    kb_id = _pick_kb(api, args.kb)
    result = api.chat(kb_id, question, args.top_k, args.rerank_k)
    if args.json:
        # 机器可读模式：只输出 JSON，不夹带人读的排版，
        # 否则 `ask.py --json ... | jq` 会因为前面的分隔线直接失败
        print(json.dumps({"kb_id": kb_id, "question": question, **result}, ensure_ascii=False))
        return 0
    debug_payload = None
    if args.debug:
        debug_payload = api.search(kb_id, question, args.top_k, args.rerank_k)
    _print_answer(result, debug_payload)
    return 0


def cmd_interactive(api: Client, args: argparse.Namespace) -> int:
    kb_id = _pick_kb(api, args.kb)
    print(f"已连接到知识库 {kb_id}。输入问题回车提问，输入 q 退出。")
    while True:
        try:
            question = input("\n问> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"q", "quit", "exit"}:
            return 0
        if not question:
            continue
        try:
            cmd_ask(api, args, question)
        except ApiError as exc:
            print(f"请求失败：{exc}")
    return 0


# ---------------------------------------------------------------------------
# 一键演示（含溯源页码校验）
# ---------------------------------------------------------------------------


def cmd_demo(api: Client, args: argparse.Namespace) -> int:
    """造样例 → 建库 → 入库 → 逐个提问 → 校验引用的页码是否准确。"""
    from scripts.make_sample_docs import HANDBOOK_FACTS, build_handbook

    fixtures = PROJECT_ROOT / "tests" / "fixtures"
    handbook = build_handbook(fixtures / "handbook.pptx")
    pdf = fixtures / "sample.pdf"
    if not pdf.exists():
        from scripts.make_sample_docs import main as make_samples

        make_samples()

    print(f"演示文档：{handbook.name}（{len(HANDBOOK_FACTS) + 1} 页中文手册） + {pdf.name}（3 页英文）")

    kb = api.create_kb(DEMO_KB_NAME, "由 ask.py --demo 生成，用于验证溯源页码")
    kb_id = kb["id"]
    print(f"知识库已创建：{kb_id}")

    exit_code = 0
    try:
        for path in (handbook, pdf):
            doc = api.upload(kb_id, path)
            flag = "ok" if doc.get("status") == "ready" else "FAIL"
            print(
                f"  入库 {path.name:16} 状态={doc.get('status'):8} "
                f"片段={doc.get('chunk_count')}  {flag}"
            )
            if doc.get("status") != "ready":
                print(f"    错误：{doc.get('error')}")
                return 1

        cases = [(question, page, handbook.name) for question, page in HANDBOOK_FACTS]
        cases.append((PDF_QUESTION, PDF_EXPECTED_PAGE, pdf.name))

        print("\n" + "=" * 68)
        print("逐题验证：答案是否引用了正确的页")
        print("=" * 68)

        failures: list[str] = []
        for question, expected, source_file in cases:
            result = api.chat(kb_id, question, args.top_k, args.rerank_k)
            _print_answer(result, None)
            cited = [item.get("page") for item in (result.get("citations") or [])]
            hit = f"第{expected}页" in cited
            first_ok = bool(cited) and cited[0] == f"第{expected}页"

            verdict = "PASS" if hit else "FAIL"
            rank = "首选" if first_ok else ("在列表中" if hit else "未出现")
            print(f"  [{verdict}] {question}")
            print(f"         期望 {source_file} 第{expected}页 → 实际引用 {cited}（{rank}）")
            if not hit:
                failures.append(f"{question}（期望第{expected}页，实际 {cited}）")

        print("\n" + "=" * 68)
        if failures:
            print(f"页码溯源校验未通过 {len(failures)}/{len(cases)} 项：")
            for item in failures:
                print(f"  - {item}")
            exit_code = 1
        else:
            print(f"页码溯源校验全部通过（{len(cases)} 项）")
    finally:
        if args.keep:
            print(f"\n演示知识库已保留：{kb_id}")
        else:
            try:
                api.delete_kb(kb_id)
                print(f"\n演示知识库已清理：{kb_id}")
            except ApiError as exc:  # noqa: BLE001
                print(f"\n清理演示知识库失败（可手动删除 {kb_id}）：{exc}")

    return exit_code


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ask.py",
        description="单知识库问答（走 HTTP 接口，含溯源页码）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("question", nargs="*", help="要问的问题；不传则需 -i / --list / --demo")
    parser.add_argument("-k", "--kb", help="知识库 ID；不传时自动选择（仅当库里只有一个）")
    parser.add_argument("-i", "--interactive", action="store_true", help="交互模式")
    parser.add_argument("--list", action="store_true", help="列出全部知识库后退出")
    parser.add_argument("--demo", action="store_true", help="一键演示：造样例、入库、验证页码溯源")
    parser.add_argument("--keep", action="store_true", help="演示结束后保留知识库（默认删除）")
    parser.add_argument("--debug", action="store_true", help="额外打印检索明细（逐路召回与分数）")
    parser.add_argument("--json", action="store_true", help="额外输出原始 JSON")
    parser.add_argument("-t", "--top-k", type=int, default=None, help="每路召回条数")
    parser.add_argument("-r", "--rerank-k", type=int, default=None, help="重排后保留条数")
    parser.add_argument("--api", default=DEFAULT_API, help=f"接口前缀（默认 {DEFAULT_API}）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    with Client(args.api) as api:
        if args.list:
            return cmd_list(api)
        if args.demo:
            return cmd_demo(api, args)
        if args.interactive:
            return cmd_interactive(api, args)

        question = " ".join(args.question).strip()
        if not question:
            build_parser().print_help()
            return 2
        try:
            return cmd_ask(api, args, question)
        except ApiError as exc:
            print(f"请求失败：{exc}")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
