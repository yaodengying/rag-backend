"""从「上传接口」入口验证全链路：上传 PDF → 召回 → 重排 → 问答溯源。

为什么要单独一个脚本，而不并进 check_*.py：
    check_*.py 系列是**离线/单组件**自检，import 服务端模块直接调函数，
    绕过了 HTTP 层。本脚本相反——全程只走 HTTP 接口
    （/knowledge-bases → /documents → /search → /chat），
    验证的是「接口层 + 服务层 + 检索链路」作为一个整体的真实行为，
    连服务的启动与关闭都由它自己管。

    两者的分工：check_*.py 保证"每个零件对"，本脚本保证"装起来也对"。

关键设计：**用「可证伪的页码」当验收标准**
    光看答案文本对不对是不够的——模型可能答对但引用错页，
    也可能引用了对的页但答案是幻觉。所以夹具的每一页只放一个事实、
    且关键数字全库唯一，于是"首位引用是否等于期望页"是一个
    不会偶然成立的判据（详见 make_pdf_fixtures.py 的模块文档）。

三类指标分开报，就是为了在答案不对时能定位到具体环节：
    召回（dense / bm25 是否命中）→ 重排（命中块排到第几）→ 生成（答案与引用）

用法：
    python scripts/verify_upload_flow.py                 # 全流程，结束后关掉服务
    python scripts/verify_upload_flow.py --keep-server   # 保留服务，便于手工 curl
    python scripts/verify_upload_flow.py --cleanup       # 验完删掉知识库
    python scripts/verify_upload_flow.py --skip-chat     # 只验召回与重排（不依赖 Ollama）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "pdf"
API = "http://127.0.0.1:8000/api/v1"
READY_TIMEOUT = 240  # 冷启动要加载 BGE + 重排权重，本机 CPU 上给足 4 分钟

# 本脚本创建的知识库带这个描述，作为"可以安全清理"的标记。
# 用它做判断而不是用名字：名字人类可能随手起一个一样的，
# 描述则只由本脚本写入，不会误删用户自己建的库。
KB_MARKER = "由 verify_upload_flow.py 创建"


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def section(text: str) -> None:
    print(f"\n--- {text} ---")


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:+.3f}"


# ---------------------------------------------------------------------------
# 服务生命周期
# ---------------------------------------------------------------------------


def is_ready(client: httpx.Client) -> bool:
    try:
        return client.get(f"{API}/ready", timeout=3).status_code == 200
    except Exception:
        return False


def start_server(log_path: Path) -> subprocess.Popen[bytes]:
    """后台拉起 uvicorn，返回进程句柄。

    环境要点：
        - cwd 必须是 backend 根目录：Settings 按相对路径读 .env，
          换个目录启动会直接因为缺 LLM_BASE_URL 之类的必填项而拒绝启动；
        - NO_PROXY 必须带上 127.0.0.1，否则全局 http_proxy 会参与本机请求。
    """
    env = os.environ.copy()
    env["NO_PROXY"] = "localhost,127.0.0.1"
    env["no_proxy"] = "localhost,127.0.0.1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("wb")
    return subprocess.Popen(  # noqa: S603
        [
            str(ROOT / ".venv" / "Scripts" / "python.exe"),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )


def wait_ready(client: httpx.Client, proc: subprocess.Popen[bytes] | None, log: Path) -> bool:
    deadline = time.time() + READY_TIMEOUT
    while time.time() < deadline:
        if is_ready(client):
            return True
        if proc is not None and proc.poll() is not None:
            print(f"  服务进程已退出（code={proc.returncode}），日志尾部：")
            print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]))
            return False
        time.sleep(2)
    print(f"  等待就绪超时（{READY_TIMEOUT}s）")
    return False


# ---------------------------------------------------------------------------
# 探针
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    """一条探针在三个阶段的表现。"""

    question: str
    expect_file: str
    expect_page: str
    keyword: str = ""
    dense_hit: bool = False
    bm25_hit: bool = False
    fused_rank: int | None = None
    fused_total: int = 0
    final_rank: int | None = None
    final_total: int = 0
    # 无重排基线：`RetrievalService.retrieve_with_trace` 在 reranker 为 None 时
    # 就是 `trace.fused[:rerank_k]`，所以基线可以用**同一个响应**里的
    # debug.fused 前 N 条精确算出来，不必再起一个关掉重排的服务。
    baseline_total: int = 0
    baseline_rank: int | None = None
    rerank_score: float | None = None
    top1_rerank: float | None = None
    answer: str = ""
    refused: bool | None = None
    cited: list[tuple[str, str | None]] = field(default_factory=list)
    # 命中片段的正文里确实含该页的独有关键词——证明"页码指向的内容"没错位，
    # 而不是"页码对但内容不对"（loader 的页码偏移、切分跨页都会造成后者）
    fact_ok: bool = False

    @property
    def recall_ok(self) -> bool:
        return self.fused_rank is not None

    @property
    def rerank_ok(self) -> bool:
        return self.final_rank == 1 and self.fact_ok

    @property
    def cite_ok(self) -> bool:
        return bool(self.cited) and self.cited[0] == (self.expect_file, self.expect_page)


def norm(text: str) -> str:
    """去空白后比较：PDF 抽取会在数字两侧补空格，直接子串匹配会漏。"""
    return "".join(text.split())


def page_of(chunk: dict) -> str:
    return chunk.get("page") or "无页码"


def matches(chunk: dict, filename: str, page_label: str) -> bool:
    return chunk.get("filename") == filename and page_of(chunk) == page_label


def find_rank(chunks: list[dict], filename: str, page_label: str) -> int | None:
    for index, chunk in enumerate(chunks, start=1):
        if matches(chunk, filename, page_label):
            return index
    return None


def find_chunk(chunks: list[dict], filename: str, page_label: str) -> dict | None:
    for chunk in chunks:
        if matches(chunk, filename, page_label):
            return chunk
    return None


def iter_sse(lines: list[str]):
    """把 SSE 原始行解析成 (事件名, 数据) 序列。

    必须**保留空行**：空行是事件之间的分隔符，丢了它相邻两个事件会被
    粘成一个块，"event: delta" 后面跟着两条 data，json.loads 直接报
    Extra data（第一版就是这么挂的）。
    """
    name: str | None = None
    payload: list[str] = []
    for line in lines:
        if not line.strip():
            if name is not None:
                yield name, json.loads("\n".join(payload))
                name, payload = None, []
            continue
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            payload.append(line[5:].strip())
    if name is not None:
        yield name, json.loads("\n".join(payload))


# ---------------------------------------------------------------------------
# 各阶段
# ---------------------------------------------------------------------------


def do_upload(client: httpx.Client, kb_id: str, files: list[Path]) -> list[dict]:
    """上传：第一份走单文件接口，其余走批量接口，两条路径都覆盖到。"""
    uploaded: list[dict] = []

    first = files[0]
    section(f"单文件接口  POST /knowledge-bases/{{kb_id}}/documents   ← {first.name}")
    response = client.post(
        f"{API}/knowledge-bases/{kb_id}/documents",
        files={"file": (first.name, first.read_bytes(), "application/pdf")},
        timeout=300,
    )
    print(f"  HTTP {response.status_code}  {response.text[:200]}")
    response.raise_for_status()
    uploaded.append(response.json())

    if len(files) > 1:
        section(f"批量接口  POST /knowledge-bases/{{kb_id}}/documents/batch  ← {len(files) - 1} 份")
        response = client.post(
            f"{API}/knowledge-bases/{kb_id}/documents/batch",
            files=[
                ("files", (path.name, path.read_bytes(), "application/pdf"))
                for path in files[1:]
            ],
            timeout=300,
        )
        print(f"  HTTP {response.status_code}")
        response.raise_for_status()
        uploaded.extend(response.json())

    return uploaded


def do_recall_and_rerank(
    client: httpx.Client, kb_id: str, probes: list, top_k: int, rerank_k: int
) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    for probe in probes:
        expect_page = f"第{probe.page}页"
        response = client.post(
            f"{API}/knowledge-bases/{kb_id}/search",
            json={"query": probe.question, "debug": True, "top_k": top_k, "rerank_k": rerank_k},
            timeout=180,
        )
        response.raise_for_status()
        body = response.json()
        debug = body["debug"]

        result = ProbeResult(
            question=probe.question,
            expect_file=probe.filename,
            expect_page=expect_page,
            keyword=probe.keyword,
        )
        result.dense_hit = find_chunk(debug["dense"]["results"], probe.filename, expect_page) is not None
        result.bm25_hit = find_chunk(debug["bm25"]["results"], probe.filename, expect_page) is not None
        result.fused_total = len(debug["fused"])
        result.fused_rank = find_rank(debug["fused"], probe.filename, expect_page)
        result.final_total = len(body["results"])
        result.final_rank = find_rank(body["results"], probe.filename, expect_page)
        baseline = debug["fused"][:rerank_k]
        result.baseline_total = len(baseline)
        result.baseline_rank = find_rank(baseline, probe.filename, expect_page)

        hit = find_chunk(body["results"], probe.filename, expect_page) or find_chunk(
            debug["fused"], probe.filename, expect_page
        )
        if hit is not None:
            result.rerank_score = hit.get("rerank_score")
            result.fact_ok = norm(probe.keyword) in norm(hit.get("text", ""))
        result.top1_rerank = body["results"][0].get("rerank_score") if body["results"] else None
        results.append(result)

        print(
            f"  {probe.question[:22]:<24} 期望 {probe.filename} {expect_page}  "
            f"dense={'○' if result.dense_hit else '×'} "
            f"bm25={'○' if result.bm25_hit else '×'} "
            f"融合第{result.fused_rank or '-'}/{result.fused_total} → "
            f"重排第{result.final_rank or '-'}/{result.final_total} "
            f"(分 {fmt(result.rerank_score)}) "
            f"内容含「{probe.keyword}」={'是' if result.fact_ok else '否'}"
        )
    return results


def do_chat(
    client: httpx.Client, kb_id: str, results: list[ProbeResult], top_k: int, rerank_k: int
) -> None:
    for result in results:
        response = client.post(
            f"{API}/chat",
            json={"kb_id": kb_id, "question": result.question, "top_k": top_k, "rerank_k": rerank_k},
            timeout=300,
        )
        response.raise_for_status()
        body = response.json()
        result.answer = body["answer"]
        result.refused = body["refused"]
        result.cited = [
            (c.get("filename") or "", c.get("page")) for c in body.get("citations") or []
        ]
        mark = "✓" if result.cite_ok else "✗"
        print(
            f"  {mark} {result.question[:20]:<22} "
            f"引用={result.cited[:2]}  拒答={result.refused}  "
            f"答案={result.answer.replace(chr(10), ' ')[:44]}"
        )


# ---------------------------------------------------------------------------
# 重排深挖
# ---------------------------------------------------------------------------

# 这两条查询是刻意挑的「重排才有用」的场景，不是随机抽样：
#   ① 词面歧义：「响应」在 IT 运维规范第 3 页（P1 故障 15 分钟）与
#      售后服务制度第 3 页（工单 48 小时）都出现，BM25 分不开，
#      该选哪份得看语义；
#   ② 口语化：原文写「故障响应时限」，问题问「多久有人处理」，
#      词面几乎不重合，考验的是稠密路的语义召回 + 重排的语义打分。
RERANK_DEMO = [
    "售后工单多久响应？",
    "系统出故障了多久有人处理？",
]


def do_rerank_detail(
    client: httpx.Client, kb_id: str, queries: list[str], top_k: int, rerank_k: int
) -> None:
    """把「融合序 → 重排序」的完整位移打出来。

    查两次，是刻意的——因为重排返回的结果里藏着两个不同的"消失"原因：

        rerank.rerank() 是先按 `score >= threshold` 过滤、再取 top_n。
        所以一条候选没出现在结果里，可能是
            (a) 重排分低于 RERANK_MIN_SCORE 被当成无关项丢弃，或
            (b) 分数够高但被 rerank_k 截断。
        这两种情况对排查的意义完全相反——(a) 说明重排认为它无关，
        (b) 说明它其实是相关的、只是名额不够。

    接口只在 results 里回填存活候选的 rerank_score，被丢的拿不到分数，
    单看一次请求无法区分 (a)(b)。解法不是改接口，而是再查一次把
    rerank_k 拉到上限：此时 top_n 大到不可能截断，凡是还没出现的，
    必定是被阈值丢掉的 (a)；出现了就说明分数拿到了，能直接读出多少分。
    """
    for query in queries:
        payload = {"query": query, "debug": True, "top_k": top_k, "rerank_k": rerank_k}
        body = client.post(f"{API}/knowledge-bases/{kb_id}/search", json=payload).json()
        wide = client.post(
            f"{API}/knowledge-bases/{kb_id}/search",
            json={**payload, "rerank_k": 50},
            timeout=180,
        ).json()

        fused = body["debug"]["fused"]
        produced = {item["chunk_id"]: index for index, item in enumerate(body["results"], start=1)}
        all_scored = {item["chunk_id"]: item.get("rerank_score") for item in wide["results"]}

        print(f"\n  查询：{query}")
        print(f"    稠密路 {body['debug']['dense']['count']} 条 / BM25 路 "
              f"{body['debug']['bm25']['count']} 条 → 融合 {len(fused)} 条 → "
              f"过重排阈值 {len(wide['results'])} 条 → 实际喂给 LLM {len(body['results'])} 条")
        print(f"    {'融合序':<7}{'来源':<23}{'余弦':>7}{'BM25':>7}{'RRF':>9}"
              f"{'重排分':>9}   生产位次")
        for index, chunk in enumerate(fused, start=1):
            score = all_scored.get(chunk["chunk_id"])
            if score is None:
                score_text, note = "  被丢", "重排判为无关（logits < 0）"
            else:
                score_text = f"{score:+.3f}"
                note = "喂给 LLM" if chunk["chunk_id"] in produced else "过阈值但被 rerank_k 截断"
                if chunk["chunk_id"] in produced:
                    position = produced[chunk["chunk_id"]]
                    if position != index:
                        note += f"（融合第{index} → 重排第{position}）"
            cosine = chunk.get("vector_score")
            print(
                f"    #{index:<6}{chunk['filename'] + ' ' + page_of(chunk):<23}"
                f"{(cosine if cosine is not None else float('nan')):>7.3f}"
                f"{(chunk.get('bm25_score') if chunk.get('bm25_score') is not None else float('nan')):>7.2f}"
                f"{(chunk.get('fusion_score') if chunk.get('fusion_score') is not None else float('nan')):>9.5f}"
                f"{score_text:>9}   {note}"
            )

        if len(body["results"]) >= 2:
            first, second = body["results"][0], body["results"][1]
            top, runner = first.get("rerank_score"), second.get("rerank_score")
            if top is not None and runner is not None:
                print(
                    f"    → 首选 {first['filename']} {page_of(first)}（{top:+.3f}）"
                    f" 领先次选 {second['filename']} {page_of(second)}（{runner:+.3f}）"
                    f" {top - runner:+.3f}"
                )


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def report(results: list[ProbeResult], with_chat: bool) -> int:
    banner("汇总")

    recalled = [item for item in results if item.recall_ok]
    promoted = [
        item
        for item in results
        if item.fused_rank not in (None, 1) and item.final_rank == 1
    ]
    demoted = [
        item for item in results if item.fused_rank == 1 and item.final_rank != 1
    ]

    print(f"  探针数                    {len(results)}")
    print(f"  召回命中（融合列表内）    {len(recalled)}/{len(results)}")
    print(f"  稠密路命中                {sum(1 for i in results if i.dense_hit)}/{len(results)}")
    print(f"  BM25 路命中               {sum(1 for i in results if i.bm25_hit)}/{len(results)}")
    print(f"  重排后居首                {sum(1 for i in results if i.final_rank == 1)}/{len(results)}")
    print(f"  命中片段正文含该页关键词  {sum(1 for i in results if i.fact_ok)}/{len(results)}")
    print(f"  重排把它从第 N 提到第 1   {len(promoted)} 条")
    if demoted:
        print(f"  ⚠ 重排把原本第 1 的挤下去  {len(demoted)} 条（需人工看是否错排）")

    if with_chat:
        cited = sum(1 for item in results if item.cite_ok)
        refused = [item for item in results if item.refused]
        print(f"  首位引用 = 期望页         {cited}/{len(results)}")
        if refused:
            print(f"  ⚠ 被拒答的                {len(refused)} 条："
                  + "、".join(item.question for item in refused))

    # ---- 重排 A/B ----
    # 无重排基线是「按融合分截断取前 rerank_k 条」，与有重排用的是同一批候选，
    # 所以这是干净的单变量对比：唯一变量就是"有没有过交叉编码器"。
    with_total = sum(item.final_total for item in results)
    without_total = sum(item.baseline_total for item in results)
    without_hit = sum(1 for item in results if item.baseline_rank)
    print("\n  重排 A/B（无重排 = 按融合分截断取前 rerank_k 条，候选集完全相同）")
    print(f"    {'指标':<24}{'有重排':>10}{'无重排':>10}")
    print(f"    {'送入 LLM 的片段总数':<24}{with_total:>10}{without_total:>10}")
    print(f"    {'其中答案页数':<24}{len(results):>10}{without_hit:>10}")
    print(
        f"    {'上下文纯度（答案页占比）':<22}"
        f"{len(results) / max(with_total, 1):>9.0%}{without_hit / max(without_total, 1):>10.0%}"
    )
    print(
        f"    {'答案页居首的探针':<24}"
        f"{sum(1 for i in results if i.final_rank == 1):>7}/{len(results)}"
        f"{sum(1 for i in results if i.baseline_rank == 1):>7}/{len(results)}"
    )

    missed = [item for item in results if not item.recall_ok]
    if missed:
        print("\n  未召回的探针（问题在检索侧，不是生成侧）：")
        for item in missed:
            print(f"    - {item.question}（期望 {item.expect_file} {item.expect_page}）")

    print()
    ok = len(recalled) == len(results) and all(item.rerank_ok for item in results)
    if with_chat:
        ok = ok and all(item.cite_ok for item in results)
    print("  结论：" + ("PASS —— 召回、重排、生成三段都对得上" if ok else "FAIL —— 见上方明细"))
    return 0 if ok else 1


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="从上传接口入口验证全链路")
    parser.add_argument("--keep-server", action="store_true", help="结束时保留服务")
    parser.add_argument("--cleanup", action="store_true", help="结束时删除验证用知识库")
    parser.add_argument("--skip-chat", action="store_true", help="跳过问答（不依赖 Ollama）")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rerank-k", type=int, default=4)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from scripts.make_pdf_fixtures import PROBES  # 探针与夹具同源，避免两处各写一遍

    banner("0. 准备夹具与服务")

    files = sorted(FIXTURE_DIR.glob("*.pdf"))
    if not files:
        print(f"  夹具目录为空：{FIXTURE_DIR}\n  先跑：python scripts/make_pdf_fixtures.py")
        return 2
    # 按 DOCUMENTS 的顺序上传，读起来与文档顺序一致
    order = ["员工手册.pdf", "IT运维规范.pdf", "售后服务制度.pdf"]
    files.sort(key=lambda path: order.index(path.name) if path.name in order else 99)
    for path in files:
        print(f"  夹具 {path.name}  {path.stat().st_size / 1024:.1f} KB")

    log_path = ROOT / "data" / "verify_server.log"
    proc: subprocess.Popen[bytes] | None = None
    started = False

    client = httpx.Client(trust_env=False, timeout=60)
    try:
        if is_ready(client):
            print("  服务已在 127.0.0.1:8000 运行，复用")
        else:
            print("  服务未运行，拉起 uvicorn …")
            proc = start_server(log_path)
            started = True
            if not wait_ready(client, proc, log_path):
                return 1
        print(f"  就绪  {client.get(f'{API}/ready').json()}")

        banner("1. 建库")
        # 先清掉本脚本上一轮留下的库，保证重复运行不会越攒越多。
        # 只删描述匹配 KB_MARKER 的，用户自己建的库不动。
        stale = [
            kb for kb in client.get(f"{API}/knowledge-bases").json()
            if kb.get("description") == KB_MARKER
        ]
        for item in stale:
            client.delete(f"{API}/knowledge-bases/{item['id']}")
            print(f"  清理上一轮的验证库 {item['id']}（{item['name']}）")

        response = client.post(
            f"{API}/knowledge-bases",
            json={"name": "PDF 导入验证库", "description": KB_MARKER},
        )
        response.raise_for_status()
        kb = response.json()
        kb_id = kb["id"]
        print(f"  kb_id = {kb_id}\n  name  = {kb['name']}")

        banner("2. 从上传接口导入 PDF")
        uploaded = do_upload(client, kb_id, files)
        for item in uploaded:
            flag = "✓" if item["status"] == "ready" else "✗"
            print(f"  {flag} {item['filename']:<20} status={item['status']:<8} "
                  f"chunks={item['chunk_count']}  {item['error']}")

        section("文档列表  GET /knowledge-bases/{kb_id}/documents")
        docs = client.get(f"{API}/knowledge-bases/{kb_id}/documents").json()
        for item in docs:
            print(f"  {item['filename']:<20} chunks={item['chunk_count']}")
        print(f"  合计 {len(docs)} 份 / {sum(d['chunk_count'] for d in docs)} 个片段")

        banner("3. 召回与重排  POST /knowledge-bases/{kb_id}/search (debug=true)")
        results = do_recall_and_rerank(client, kb_id, PROBES, args.top_k, args.rerank_k)

        section("重排深挖：融合序 → 重排序（选的是重排真正该起作用的场景）")
        do_rerank_detail(client, kb_id, RERANK_DEMO, args.top_k, args.rerank_k)

        if not args.skip_chat:
            banner("4. 问答溯源  POST /chat")
            do_chat(client, kb_id, results, args.top_k, args.rerank_k)

            banner("5. 流式问答  POST /chat/stream")
            question = PROBES[0].question
            started_at = time.time()
            raw: list[str] = []
            stamps: list[float] = []
            # 逐行读而不是一次性 iter_text：要拿到"每个 data 帧到达的时刻"，
            # 否则无法区分"真流式"与"其实一次性返回、只是被切成多个事件"。
            with client.stream(
                "POST",
                f"{API}/chat/stream",
                json={"kb_id": kb_id, "question": question, "top_k": args.top_k,
                      "rerank_k": args.rerank_k},
                timeout=300,
            ) as response:
                for line in response.iter_lines():
                    raw.append(line)
                    if line.strip().startswith("data:"):
                        stamps.append(time.time() - started_at)
            events = list(iter_sse(raw))
            kinds = [name for name, _ in events]
            deltas = [payload["content"] for name, payload in events if name == "delta"]
            print(f"  问题：{question}")
            print(f"  事件序列 {kinds[:3]} … {kinds[-1]}  共 {len(kinds)} 个事件 / {len(deltas)} 个增量块")
            print(
                "  首个 data 帧 %.1fs → 末个 data 帧 %.1fs（跨度 %.1fs，逐 token 边生成边推）"
                % (stamps[0], stamps[-1], stamps[-1] - stamps[0])
            )
            print(f"  sources 事件：{[(s['filename'], s['page']) for s in events[0][1]['sources'][:2]]}")
            print(f"  delta 拼接 = {' '.join(deltas)[:70]}")
            done = events[-1][1]
            print(f"  done 事件：refused={done.get('refused')}  "
                  f"citations={[(c['filename'], c['page']) for c in done.get('citations') or []]}")

            banner("6. 拒答路径")
            response = client.post(
                f"{API}/chat",
                json={"kb_id": kb_id, "question": "今天天气怎么样？", "top_k": args.top_k,
                      "rerank_k": args.rerank_k},
                timeout=120,
            )
            body = response.json()
            print(f"  refused={body['refused']}  citations={len(body['citations'])}  "
                  f"sources={len(body['sources'])}")
            print(f"  answer={body['answer'][:80]}")

        code = report(results, with_chat=not args.skip_chat)

        if args.cleanup:
            banner("7. 清理")
            response = client.delete(f"{API}/knowledge-bases/{kb_id}")
            print(f"  DELETE /knowledge-bases/{{kb_id}} → HTTP {response.status_code}")
            remaining = client.get(f"{API}/knowledge-bases").json()
            print(f"  剩余知识库 {len(remaining)} 个；uploads 文件 "
                  f"{len(list((ROOT / 'data' / 'uploads').glob('*')))} 个")
        else:
            print(f"\n  知识库已保留：{kb_id}")
            print(f"  手工复验：python scripts/ask.py -k {kb_id} \"上班时间是几点？\" --debug")

        return code
    finally:
        client.close()
        if started and not args.keep_server and proc is not None:
            banner("关闭服务")
            proc.terminate()
            try:
                proc.wait(timeout=15)
                print("  已关闭（数据仍留在 SQLite / Milvus / uploads）")
            except subprocess.TimeoutExpired:
                proc.kill()
                print("  已强制结束")


if __name__ == "__main__":
    raise SystemExit(main())
