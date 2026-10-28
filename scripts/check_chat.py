# -*- coding: utf-8 -*-
"""问答链路自检：验证「混合检索 → 重排 → 生成 / 改写重试 / 短路拒答」这一段是对的。

为什么需要这个脚本：
    问答链路是本项目里**失败最安静**的一段。三类典型故障都不会抛异常：
      ① 拒答判定失效 —— 模型说了"找不到答案"，系统却标成"已作答"；
      ② 改写回路失控 —— 要么永不触发（检索不足直接放弃），
         要么停不下来（无限改写，一次请求打十几次模型）；
      ③ 引用页码错位 —— 答案对、引用的页却是另一页（用户核对时才发现）。
    这三类都只能靠断言钉住。

    覆盖率还有一个现实理由：真实跑一遍问答要 ~15 秒（交叉编码器 + 7B 模型），
    开发期不可能每次都跑。本脚本用**全假的上游**，几秒内覆盖全部分支。

判据（不只断言"不报错"）：

    | 检查项          | 期望                                                          |
    |----------------|---------------------------------------------------------------|
    | 页码标签        | Chunk.page_label：未知 → None；单页 → 第N页；多页 → 第N-M页      |
    | 参考资料标注    | format_context 带 `[序号] 来源：文件名 第N页`；页码未知时如实写明 |
    | 标注顺序        | 序号按传入顺序递增，正文顺序不被重排                            |
    | 提示词一致性    | NO_ANSWER_TEXT 必须同时出现在 system 与 user 模板里             |
    | 路由分支        | ranked 非空→generate；空且未超限→rewrite；空且超限→refuse        |
    | 图·正常路径     | 检索有候选 → 生成被调用一次，answer 非空，refused=False          |
    | 图·短路拒答     | 检索为空 → **生成节点一次都不调用**，answer=固定话术             |
    | 图·改写重试     | 首轮重排全丢 → 改写一次 → 二轮命中 → 生成；重排被调用两次         |
    | 图·重试上限     | 重排始终全丢 → 只改写一次即拒答（回路不会无限转）                |
    | 图·无重排器     | reranker=None → 按融合分截断，rerank_score 全为 None            |
    | ChatService     | 引用页码/文件名来自片段；拒答时 citations 空而 sources 非空       |
    | 服务·空问题     | 抛 ValidationError（不是返回空答案）                            |
    | 服务·图无产出   | 抛 UpstreamError（宁可报错，也不要返回空回答）                   |
    | LLM·消息组装    | system 为空白时不构造 SystemMessage（部分后端对空 system 报 400） |
    | LLM·内容抽取    | 字符串 / 多模态列表两种 content 都能抽出文本                     |
    | LLM·空回复      | 抛 UpstreamError（空串不是"没有答案"，是上游故障）               |
    | LLM·上游异常    | 连接失败 / 超时 → UpstreamError，不是裸异常                      |
    | LLM·配置缺失    | base_url / model 为空 → ConfigurationError                      |
    | LLM·流式        | stream 逐段产出；中途异常 → UpstreamError                       |
    | SSE 报文        | 单条消息以空行结尾；data 段内**不得有裸换行**                    |
    | SSE 事件顺序    | retry? → sources → delta* → done；出错时以 error 收尾           |
    | SSE 断开        | aclose()/取消 → 关断信号穿透到最底层 astream；不得被 except Exception 吞掉 |
    | 流式·改写重试   | enable_retry=true 且首轮无片段 → retry 事件 → 再检索 → 正常生成  |
    | 流式·重试上限   | 重试次数与完整图**同源**（共用 route_after_rerank + MAX_RETRY）  |
    | 流式·生成不变量 | 生成永远用原始问题；改写只影响检索命中                          |
    | 流式·降级       | 改写无产出/报错 → 不白跑第二次检索，按拒答收尾（不升级成 500）    |
    | 重试·三态       | 显式 true/false **压过**服务端默认值；省略/传 null 才用默认值     |
    | 重试·done 契约  | done 带 retried/attempts/can_retry/query；成功与拒答字段集合一致 |
    | 重试·不变式     | can_retry == (refused and attempts < MAX_RETRY)，逐场景核对      |
    | 重试·当场询问   | 默认拒答且 can_retry → 同问题带 enable_retry=true 重发 → 救回     |
    | 重试·能力发现   | GET /chat/options 与实际行为同源（换实例端点跟着变）             |
    | HTTP /chat      | 200 + answer/sources/citations/refused/retried/query 字段齐全    |
    | HTTP 404        | 知识库不存在 → 404（且响应体是统一错误结构）                     |

运行（**不需要 Milvus，不需要 BGE 权重，不需要 Ollama**）：
    .venv\\Scripts\\python.exe scripts\\check_chat.py

    全程使用内存假实现：假 embedder（查表）、假向量库（余弦 + 2-gram 关键词）、
    假重排器（登记表给分，可切换"永远丢光"）、假 LLM（登记表应答，可切换为空/报错）。

    第 11 / 12 组是仅有的两组**直接驱动 async generator**——11 组用
    `anext` / `aclose` / `athrow` / 任务取消测"半流断开"（`TestClient` 会同步消费
    完整条响应，走不到那条路径），12 组直接驱动 `_sse_generator`。
    12 组必须从 `_sse_generator` 进而不是直接调 `ask_stream`：请求字段就是在这
    一层被丢掉的，从服务层进就测不到"死字段"。

    第 13 组测的不是"重试逻辑对不对"（那是 12 组的事），而是**这个决定由谁做**：
    功能写对了、开关照样可能是假的——默认值盖掉用户的显式选择，
    或者前端根本拿不到判断依据。两处要特别注意：
      · 先断言 done 的字段齐全、再读字段（否则少一个键会让整节以 KeyError 崩掉，
        输出里既没有 FAIL 也没有汇总行，比"断言不通过"更难发现）；
      · "默认值真的接了配置"必须**临时把配置翻过来**才有无可证伪力，
        因为本机 .env 里这一项恰好等于类默认值。

退出码：0 全部通过；1 存在失败项。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# settings 的 env_file 是相对路径，必须先切到项目根
os.chdir(PROJECT_ROOT)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from app.core.exceptions import (  # noqa: E402
    ConfigurationError,
    NotFoundError,
    UpstreamError,
    UpstreamTimeoutError,
    ValidationError,
)
from app.domain.models import Chunk, RetrievedChunk  # noqa: E402
from app.infrastructure.llm import OpenAICompatibleLLM  # noqa: E402
from app.rag.graph import (  # noqa: E402
    NODE_GENERATE,
    NODE_REFUSE,
    NODE_REWRITE,
    build_rag_graph,
    build_retrieval_graph,
    route_after_rerank,
)
from app.rag.prompts import (  # noqa: E402
    ANSWER_SYSTEM,
    ANSWER_USER,
    REWRITE_SYSTEM,
    build_answer_prompts,
    format_context,
)
from app.rag.retrieval import HybridRetriever  # noqa: E402
from app.rag.state import MAX_RETRY, NO_ANSWER_TEXT  # noqa: E402
from app.schemas.chat import ChatRequest  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402

DIM = 8

# HTTP 用例用的知识库 ID：FakeCollections 只放行它，数据也必须落在这个 id 下
HTTP_KB = "kb-ok"

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
            print(f"  [ok]   {label}（{type(exc).__name__}: {str(exc)[:60]}…）")
        else:
            print(f"  [FAIL] {label}  -> 消息里没有 {keyword!r}：{exc}")
            _failures.append(label)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {label}  -> 期望 {exc_type.__name__}，实际 {type(exc).__name__}: {exc}")
        _failures.append(label)
    else:
        print(f"  [FAIL] {label}  -> 没有抛异常")
        _failures.append(label)


def section(title: str) -> None:
    """打印分组标题。"""
    print()
    print(f"=== {title} ===")


# ---------------------------------------------------------------------------
# 假的上游（全部在内存里，不碰 Milvus / BGE / Ollama）
# ---------------------------------------------------------------------------


def _onehot(axis: int) -> list[float]:
    """单位基向量：不同轴之间余弦恰好为 0，可精确构造"召不回"。"""
    return [1.0 if i == axis else 0.0 for i in range(DIM)]


def _cosine(left: list[float], right: list[float]) -> float:
    """余弦相似度（零向量返回 0，避免除零）。"""
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


def _bigrams(text: str) -> set[str]:
    """字符二元组集合，用来模拟中文关键词命中。"""
    cleaned = "".join(ch for ch in text if not ch.isspace())
    return {cleaned[i : i + 2] for i in range(max(len(cleaned) - 1, 0))}


class FakeEmbedder:
    """按文本查表返回向量的假向量化器（未登记 → 与查询正交）。"""

    dimension = DIM

    def __init__(self, table: dict[str, list[float]] | None = None) -> None:
        self._table = dict(table or {})
        self.query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._vec(text)

    def _vec(self, text: str) -> list[float]:
        return list(self._table.get(text, _onehot(DIM - 1)))


class FakeStore:
    """内存向量库：稠密走余弦，稀疏走 2-gram 重合度。

    刻意实现成一个**真的会排序**的存储，而不是"永远返回预置列表"：
    否则融合与重排的输入顺序就不是真实产出的顺序，自检会在
    `fusion._assert_sorted` 之外的层面失去意义。
    """

    def __init__(self) -> None:
        self._rows: list[Chunk] = []
        self.dense_calls = 0
        self.bm25_calls = 0

    def ensure_collection(self, dimension: int) -> None:  # noqa: ARG002
        return None

    def upsert(self, chunks: list[Chunk]) -> int:
        self._rows.extend(chunks)
        return len(chunks)

    def search(self, kb_id, query_embedding, top_k, min_score=None):  # noqa: ANN001, ANN201
        self.dense_calls += 1
        threshold = 0.0 if min_score is None else float(min_score)
        scored = [
            (row, _cosine(query_embedding, row.embedding or []))
            for row in self._rows
            if row.kb_id == kb_id
        ]
        scored = [pair for pair in scored if pair[1] >= threshold]
        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        return [RetrievedChunk(chunk=row, vector_score=score) for row, score in scored[:top_k]]

    def search_bm25(self, kb_id, query, top_k, min_score=None):  # noqa: ANN001, ANN201
        self.bm25_calls += 1
        query_grams = _bigrams(query)
        scored = []
        for row in self._rows:
            if row.kb_id != kb_id:
                continue
            hits = len(query_grams & _bigrams(row.text))
            if hits:
                scored.append((row, float(hits)))
        threshold = 0.0 if min_score is None else float(min_score)
        scored = [pair for pair in scored if pair[1] >= threshold]
        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        return [RetrievedChunk(chunk=row, bm25_score=score) for row, score in scored[:top_k]]

    def delete_by_doc(self, doc_id: str) -> int:
        before = len(self._rows)
        self._rows = [row for row in self._rows if row.doc_id != doc_id]
        return before - len(self._rows)

    def delete_by_kb(self, kb_id: str) -> int:
        before = len(self._rows)
        self._rows = [row for row in self._rows if row.kb_id != kb_id]
        return before - len(self._rows)


class FakeReranker:
    """假重排器：按登记表给分；`drop_first` 次调用故意全丢。

    `drop_first` 是构造"改写重试"场景的关键：它让第一次重排返回空、
    第二次正常返回，从而精确地把图推上重试回路。
    """

    def __init__(
        self,
        table: dict[str, float] | None = None,
        min_score: float = 0.0,
        drop_first: int = 0,
    ) -> None:
        self._table = dict(table or {})
        self._min_score = min_score
        self._drop_first = drop_first
        self.calls: list[tuple[str, int, int]] = []

    def rerank(self, query, candidates, top_n, min_score=None):  # noqa: ANN001, ANN201
        self.calls.append((query, len(candidates), top_n))
        if not candidates or top_n <= 0:
            return []
        if len(self.calls) <= self._drop_first:
            return []
        threshold = self._min_score if min_score is None else float(min_score)
        scored = [
            (item, float(self._table.get(item.chunk.id, 1.0))) for item in candidates
        ]
        kept = [pair for pair in scored if pair[1] >= threshold]
        kept.sort(key=lambda pair: (-pair[1], pair[0].chunk.id))
        out: list[RetrievedChunk] = []
        for item, score in kept[:top_n]:
            item.rerank_score = score
            out.append(item)
        return out


class FakeLLMClient:
    """假 ChatOpenAI：`.invoke()` 与 `.astream()` 都按登记表应答。

    `error` 与 `empty` 两个开关用来验证错误映射——
    没有它们，"空回复视为上游异常"这类分支就只能靠读代码来确认。
    """

    def __init__(
        self,
        answers: dict[str, str] | None = None,
        default: str = "（假模型的默认回答）",
        *,
        error: Exception | None = None,
        empty: bool = False,
        stream_pieces: list[str] | None = None,
    ) -> None:
        self._answers = dict(answers or {})
        self._default = default
        self._error = error
        self._empty = empty
        self._stream_pieces = list(stream_pieces or ["假", "流式", "回答"])
        self.messages: list[list] = []

    def invoke(self, messages):  # noqa: ANN001, ANN201
        self.messages.append(messages)
        if self._error is not None:
            raise self._error
        if self._empty:
            return _Response("")
        joined = "\n".join(str(getattr(m, "content", "")) for m in messages)
        for key, answer in self._answers.items():
            if key in joined:
                return _Response(answer)
        return _Response(self._default)

    async def astream(self, messages):  # noqa: ANN001, ANN201
        self.messages.append(messages)
        if self._error is not None:
            raise self._error
        for piece in self._stream_pieces:
            yield _Response(piece)


class _Response:
    """模拟 LangChain AIMessage（只需要 .content）。"""

    def __init__(self, content) -> None:  # noqa: ANN001
        self.content = content


def _is_rewrite_prompt(system_prompt: str) -> bool:
    """区分「改写请求」与「生成请求」。

    必须用**恒等比较**，不能用 `"改写" in system_prompt`：
    防幻觉提示词里有「不得改写、四舍五入或换算单位」，子串匹配会把生成请求
    误判成改写请求——假 LLM 于是把改写结果当回答返回，而断言仍然"通过"。
    这类"测试跟着实现一起错"的坑，只有靠精确匹配才关得住。
    """
    return system_prompt == REWRITE_SYSTEM


class CountingLLM:
    """实现 LLMPort 的假 LLM，用于图节点自检（不经过 ChatOpenAI）。"""

    def __init__(self, answer: str = "（假回答）", rewrite: str = "改写后的查询") -> None:
        self._answer = answer
        self._rewrite = rewrite
        self.generate_calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.generate_calls.append((system_prompt, user_prompt))
        # 改写请求与生成请求用 system 提示词区分（改写节点的 system 就是 REWRITE_SYSTEM）
        if _is_rewrite_prompt(system_prompt):
            return self._rewrite
        return self._answer

    async def stream(self, system_prompt, user_prompt):  # noqa: ANN001, ANN201
        yield self._answer


# ---------------------------------------------------------------------------
# 构造被测对象
# ---------------------------------------------------------------------------


def _chunk(
    text: str,
    page: int,
    *,
    kb_id: str,
    vector: list[float],
    filename: str = "handbook.pptx",
    doc_id: str = "doc-1",
) -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()),
        kb_id=kb_id,
        doc_id=doc_id,
        text=text,
        chunk_index=page - 1,
        page_start=page,
        page_end=page,
        filename=filename,
        embedding=list(vector),
    )


def _retriever(store: FakeStore, embedder: FakeEmbedder, **overrides) -> HybridRetriever:
    params = {"top_k": 5, "hybrid_enabled": True, "min_similarity": 0.0}
    params.update(overrides)
    return HybridRetriever(embedder=embedder, vector_store=store, **params)


# ---------------------------------------------------------------------------
# 1. 页码标签
# ---------------------------------------------------------------------------


def check_page_label() -> None:
    section("1. 页码标签（引用溯源的基础）")

    def label(start, end):  # noqa: ANN001, ANN202
        return Chunk(id="c", kb_id="k", doc_id="d", text="t", page_start=start, page_end=end).page_label

    check("page_start=None → None（未知，不显示页码）", label(None, None) is None)
    check("单页 → 第3页", label(3, 3) == "第3页", f"实际 {label(3, 3)!r}")
    check("page_end=None 视为单页 → 第3页", label(3, None) == "第3页")
    check("跨页 → 第3-5页", label(3, 5) == "第3-5页", f"实际 {label(3, 5)!r}")

    # DTO 出口也必须带上同一个页码（曾经这里比对的是 mappers 里的同名包装函数，
    # 那个包装已删除——现在只有 Chunk.page_label 一份实现，改测真正的消费点）。
    from app.services.mappers import retrieved_to_dict

    dto = retrieved_to_dict(
        RetrievedChunk(
            chunk=Chunk(id="c", kb_id="k", doc_id="d", text="t", page_start=4, page_end=4, filename="手册.pptx")
        )
    )
    check("mappers 出口的 page 与领域方法口径一致", dto["page"] == "第4页", f"实际 {dto['page']!r}")
    check("mappers 出口同时给出文件名", dto["filename"] == "手册.pptx", f"实际 {dto['filename']!r}")


# ---------------------------------------------------------------------------
# 2. 参考资料标注
# ---------------------------------------------------------------------------


def check_format_context() -> None:
    section("2. 参考资料标注（文件名 + 页码）")

    def hit(text: str, page: int | None, filename: str = "手册.pptx") -> RetrievedChunk:
        chunk = Chunk(
            id=str(uuid.uuid4()),
            kb_id="kb",
            doc_id="doc",
            text=text,
            page_start=page,
            page_end=page,
            filename=filename,
        )
        return RetrievedChunk(chunk=chunk, rerank_score=1.0)

    check("空列表 → 占位而不是空串", format_context([]) == "（无）", repr(format_context([])))

    ctx = format_context([hit("报销上限五千元。", 3)])
    check("含序号与文件名", "[1] 来源：手册.pptx 第3页" in ctx, ctx)
    check("正文紧随标注之后", "报销上限五千元。" in ctx.splitlines()[1], ctx)

    ctx = format_context([hit("正文甲", 1), hit("正文乙", 2), hit("正文丙", None)])
    indexes = [line.split("]")[0] + "]" for line in ctx.splitlines() if line.startswith("[")]
    check("序号按顺序递增", indexes == ["[1]", "[2]", "[3]"], f"实际 {indexes}")
    check("页码未知时如实标注", "第3页" not in ctx.splitlines()[-2] and "页码未知" in ctx)
    check("多条之间用空行分隔", "\n\n" in ctx)

    ctx = format_context([hit("x" * 50, 1, filename="")], snippet_limit=10)
    check("超长正文被截断", "x" * 11 not in ctx and "…" in ctx)
    check("文件名为空时给占位", "未知文件" in ctx)

    system, user = build_answer_prompts([hit("报销上限五千元。", 3)], "报销上限是多少？")
    check("问题被填进 user 提示词", "报销上限是多少？" in user)
    check("参考资料被填进 user 提示词", "第3页" in user)
    check("system 提示词不被误改", system == ANSWER_SYSTEM)


def check_prompt_consistency() -> None:
    section("3. 提示词与拒答判定的口径一致性")

    check(
        "NO_ANSWER_TEXT 出现在 system 模板里",
        NO_ANSWER_TEXT in ANSWER_SYSTEM,
        "话术一改，refused 判定立刻失效",
    )
    check(
        "user 模板保留 {no_answer} 占位（话术只在 state.py 定义一处，不复制）",
        "{no_answer}" in ANSWER_USER,
    )

    _, formatted = build_answer_prompts([], "问题")
    check("格式化后 user 提示词里出现拒答话术（供模型原样回复）", NO_ANSWER_TEXT in formatted)
    check("格式化后 {no_answer} 占位已被填充", "{no_answer}" not in formatted)
    check("空参考资料时用「（无）」占位而不是留空", "（无）" in formatted)

    from app.rag.nodes import _is_refusal

    check("拒答判定：原样 → True", _is_refusal(NO_ANSWER_TEXT) is True)
    check(
        "拒答判定：加了前后缀也算拒答（模型常见行为）",
        _is_refusal(f"抱歉，{NO_ANSWER_TEXT}") is True,
    )
    check("拒答判定：正常回答 → False", _is_refusal("报销上限是五千元。") is False)


def check_prompt_hardening() -> None:
    """防幻觉提示词的**可核对性**。

    为什么断言字符串而不是断言模型行为：模型行为要跑真 LLM、每次结果不同，
    没法进 CI。但「禁令还在不在」是可以钉死的——它最容易被"顺手优化措辞"
    改掉，而改掉之后回答质量下降是渐进的、没有任何报错。
    抽象要求（"不要编造"）对 7B 无效，所以这里要求的是**编号的逐条禁令**。
    """
    section("3b. 防幻觉提示词（逐条禁令，不是抽象要求）")

    clauses = {
        "只依据资料": "只使用参考资料中明确写出的信息",
        "禁止外部知识": "禁止使用常识、经验、推测或任何外部知识",
        "数字型号逐字一致": "逐字一致",
        "禁止跨段拼接": "不得把不同段落、不同文件的信息拼接成一个新结论",
        "资料没提的不写": "一律不要写",
        "允许部分回答": "只写资料覆盖的那部分",
        "拒答话术逐字": "必须逐字回复",
        "拒答限定为完全无关": "完全无关",
    }
    for label, fragment in clauses.items():
        check(f"含禁令「{label}」", fragment in ANSWER_SYSTEM, fragment)

    markers = [f"{index}." for index in range(1, 7)]
    check(
        "6 条禁令以编号列表给出（7B 对逐条编号的遵守度显著高于散文）",
        all(marker in ANSWER_SYSTEM for marker in markers),
        f"缺少 {[m for m in markers if m not in ANSWER_SYSTEM]}",
    )
    check(
        "user 提示词重复了「数字与型号逐字引用」（离问题最近的约束最有效）",
        "逐字引用" in ANSWER_USER,
    )
    check(
        "明确写了「资料简短/只覆盖一部分不是拒答的理由」（防过度拒答）",
        "不是拒答的理由" in ANSWER_SYSTEM,
        "q02 就是这么被误拒的：上下文正确，模型仍照抄拒答话术",
    )
    check(
        "不再要求行内「(来源：[n])」：模型实测不遵守，溯源走结构化 citations",
        "不需要写文件名、页码或序号" in ANSWER_SYSTEM
        and "来源标注（必须做）" not in ANSWER_SYSTEM,
        "留一条必然被违反的指令，只会让提示词遵守度这个信号永久噪音化",
    )
    check(
        "提示词里没有「能被直接照抄的占位话术」（不给模型一句可以当答案交上来的句子）",
        not any(
            phrase in ANSWER_SYSTEM + ANSWER_USER
            for phrase in ("资料未说明", "资料未提及", "暂未提及")
        ),
        "实测 qwen2:7b：同一句占位话术在 system 与 user 各出现一次 → q05 整题回「资料未说明。」"
        "（5/5），而资料里明明写着答案；删掉任意一半即 0/5。留白可以，留句式不行。",
    )
    check(
        "提示词里没有留下外部知识的许可口（如「可结合常识」「酌情补充」）",
        not any(
            phrase in ANSWER_SYSTEM
            for phrase in ("结合常识", "酌情补充", "可以补充", "自行判断")
        ),
    )


def check_grounding_audit() -> None:
    """接地性审计：只告警、不改写。

    改写回答会把「模型没遵守提示词」这件事洗掉，而那正是要观测的对象。
    """
    section("3c. 回答接地性审计")

    from app.rag.grounding import audit_answer

    def _audit_answer(answer: str, sources: list[RetrievedChunk]) -> list[str]:
        """只取告警列表——引用越界是完整核验（grounding）的一个子集。

        实现于 `app/rag/grounding.py`，完整分级（ok/warn/low）与
        数字/型号核验在 `scripts/check_grounding.py` 里测；
        本节只钉住"引用序号"这一项的契约。
        """
        return list(audit_answer(answer, sources).warnings)

    def _source(index: int) -> RetrievedChunk:
        chunk = Chunk(
            id=f"c{index}",
            kb_id="kb",
            doc_id="doc",
            text=f"正文{index}",
            page_start=index,
            page_end=index,
            filename="手册.pptx",
        )
        return RetrievedChunk(chunk=chunk, rerank_score=1.0)

    sources = [_source(1), _source(2)]

    check(
        "回答没标序号 → **不**告警（模型实测不遵守，别让噪音淹没真问题）",
        _audit_answer("报销上限是五千元。", sources) == [],
        f"{_audit_answer('报销上限是五千元。', sources)}",
    )
    check(
        "序号都在范围内 → 无告警",
        _audit_answer("报销上限是五千元。(来源：[1])", sources) == [],
        f"{_audit_answer('报销上限是五千元。(来源：[1])', sources)}",
    )
    out_of_range = _audit_answer("报销上限是五千元。(来源：[5])", sources)
    check(
        "序号越界 → 告警里点名越界值与资料条数",
        any("不存在的来源序号" in w and "2 条" in w for w in out_of_range),
        f"{out_of_range}",
    )
    check(
        "序号 [0] 也算越界（序号从 1 起）",
        bool(_audit_answer("答。(来源：[0])", sources)),
    )
    check(
        "同一越界序号引用多次只报一次",
        len(_audit_answer("答。(来源：[9]) 又说 [9]", sources)) == 1,
        f"{_audit_answer('答。(来源：[9]) 又说 [9]', sources)}",
    )
    check("空回答不审计（拒答路径由 refused 判定负责）", _audit_answer("", sources) == [])
    check(
        "多个合法序号各自识别",
        _audit_answer("答。(来源：[1][2])", sources) == [],
        f"{_audit_answer('答。(来源：[1][2])', sources)}",
    )
    check(
        "3 位数序号也不会被截断成正则误判（[123] 越界应被抓到）",
        bool(_audit_answer("答。(来源：[123])", sources)),
    )


# ---------------------------------------------------------------------------
# 4. 路由分支
# ---------------------------------------------------------------------------


def check_route() -> None:
    section("4. 重排后的路由分支")

    check("有可用片段 → generate", route_after_rerank({"ranked": [object()]}) == NODE_GENERATE)
    check(
        "无可用片段且未超限 → rewrite",
        route_after_rerank({"ranked": [], "retry_count": 0}) == NODE_REWRITE,
    )
    check(
        f"无可用片段且已达上限({MAX_RETRY}) → refuse",
        route_after_rerank({"ranked": [], "retry_count": MAX_RETRY}) == NODE_REFUSE,
    )
    check(
        "retry_count 缺失视为 0 → rewrite",
        route_after_rerank({"ranked": []}) == NODE_REWRITE,
    )
    check(
        "刻意用 ranked 而不是 candidates 作判据（召回到了但全被判为无关 → 不该硬生成）",
        route_after_rerank({"ranked": [], "candidates": [object()], "retry_count": 9})
        == NODE_REFUSE,
    )


# ---------------------------------------------------------------------------
# 5. 图的三条路径
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """一套完整的测试场景：真实现编出来的图 + 全程可控的假上游。"""

    kb_id: str
    question: str
    text: str
    store: FakeStore
    embedder: FakeEmbedder
    reranker: object | None
    llm: CountingLLM
    retriever: HybridRetriever
    graph: object

    def state(self, **overrides) -> dict:  # noqa: ANN003
        """图的初始状态（与 ChatService._initial_state 同构）。"""
        base = {
            "kb_id": self.kb_id,
            "question": self.question,
            "top_k": 5,
            "rerank_k": 4,
            "retry_count": 0,
        }
        base.update(overrides)
        return base


def _scenario(
    *,
    drop_first: int = 0,
    empty_kb: bool = False,
    with_reranker: bool = True,
    kb_id: str | None = None,
) -> Scenario:
    """搭一个「内存向量库 + 假 embedder + 假重排器 + 假 LLM」的完整场景。

    Args:
        kb_id: 指定知识库 ID；HTTP 用例必须传固定值（FakeCollections 只放行
            "kb-ok"），否则会出现"请求打的是 kb-ok、数据却存在随机 kb 下"
            的假失败——本节就踩过一次。
    """
    kb_id = kb_id or f"kb-{uuid.uuid4().hex[:6]}"
    query = "报销金额上限是多少？"
    text = "单笔报销金额不得超过五千元；超出部分须由部门总监审批。"
    store = FakeStore()
    if not empty_kb:
        store.upsert([_chunk(text, 3, kb_id=kb_id, vector=_onehot(0))])
    embedder = FakeEmbedder({query: _onehot(0), text: _onehot(0)})
    reranker = FakeReranker(drop_first=drop_first) if with_reranker else None
    llm = CountingLLM(answer="报销上限为五千元。(来源：[1])")
    retriever = _retriever(store, embedder)
    return Scenario(
        kb_id=kb_id,
        question=query,
        text=text,
        store=store,
        embedder=embedder,
        reranker=reranker,
        llm=llm,
        retriever=retriever,
        graph=build_rag_graph(retriever, reranker, llm),
    )


def check_graph_paths() -> None:
    section("5. 图的四条执行路径")

    # --- 正常 ---
    case = _scenario()
    final = case.graph.invoke(case.state())
    check("正常路径：answer 非空", bool(final.get("answer")), repr(final.get("answer")))
    check("正常路径：refused=False", final.get("refused") is False)
    check("正常路径：LLM 只被调用一次（全部用于生成）", len(case.llm.generate_calls) == 1,
          f"实际 {len(case.llm.generate_calls)} 次")
    check("正常路径：重排只跑一遍", len(case.reranker.calls) == 1,
          f"实际 {len(case.reranker.calls)}")
    check("正常路径：ranked 1 条且带重排分", len(final.get("ranked") or []) == 1
          and final["ranked"][0].rerank_score is not None)
    check("正常路径：生成用的提示词里带页码来源", "第3页" in case.llm.generate_calls[0][1])

    # --- 短路拒答：检索为空 ---
    case = _scenario(empty_kb=True)
    final = case.graph.invoke(case.state())
    check("空库路径：answer 为固定拒答话术", final.get("answer") == NO_ANSWER_TEXT,
          repr(final.get("answer")))
    check("空库路径：refused=True", final.get("refused") is True)
    check(
        "空库路径：**生成节点一次都没被调用**（LLM 只被用于改写）",
        len(case.llm.generate_calls) == 1
        and _is_rewrite_prompt(case.llm.generate_calls[0][0]),
        f"实际调用 {len(case.llm.generate_calls)} 次",
    )
    check("空库路径：ranked 为空", not final.get("ranked"))
    check(f"空库路径：retry_count 停在 {MAX_RETRY}（回路已终止）",
          int(final.get("retry_count") or 0) == MAX_RETRY, f"实际 {final.get('retry_count')}")
    check(
        "空库路径：重排器**一次都没被调用**（候选为空时节点直接短路，"
        "不做一次毫无意义的交叉编码器前向）",
        len(case.reranker.calls) == 0,
        f"实际 {len(case.reranker.calls)}",
    )

    # --- 改写重试后成功 ---
    case = _scenario(drop_first=1)
    final = case.graph.invoke(case.state())
    check("重试路径：最终 answer 非空（第二轮救回来了）", bool(final.get("answer")))
    check("重试路径：refused=False", final.get("refused") is False)
    check("重试路径：重排被调用两次", len(case.reranker.calls) == 2,
          f"实际 {len(case.reranker.calls)}")
    check("重试路径：LLM 调用两次（1 次改写 + 1 次生成）", len(case.llm.generate_calls) == 2,
          f"实际 {len(case.llm.generate_calls)}")
    rewrite_calls = [c for c in case.llm.generate_calls if _is_rewrite_prompt(c[0])]
    check("重试路径：改写确实发生了一次", len(rewrite_calls) == 1)
    check("重试路径：第二次检索用的是改写后的查询",
          case.embedder.query_calls.count("改写后的查询") == 1,
          f"实际 query_calls={case.embedder.query_calls}")
    check("重试路径：retry_count=1", int(final.get("retry_count") or 0) == 1)

    # --- 重排始终全丢 → 达上限后拒答 ---
    case = _scenario(drop_first=99)
    final = case.graph.invoke(case.state())
    check("全丢路径：answer 为固定拒答话术", final.get("answer") == NO_ANSWER_TEXT)
    check("全丢路径：refused=True", final.get("refused") is True)
    check(
        "全丢路径：**最多改写一次**，回路不会无限转",
        len([c for c in case.llm.generate_calls if _is_rewrite_prompt(c[0])]) == 1,
        f"实际 {len(case.llm.generate_calls)} 次 LLM 调用",
    )
    check("全丢路径：重排被调用两次", len(case.reranker.calls) == 2,
          f"实际 {len(case.reranker.calls)}")

    # --- 无重排器 → 按融合分截断 ---
    case = _scenario(with_reranker=False)
    final = case.graph.invoke(case.state())
    ranked = final.get("ranked") or []
    check("无重排器：ranked 非空（退化为按融合分截断）", len(ranked) == 1)
    check("无重排器：rerank_score 全为 None", all(item.rerank_score is None for item in ranked))


def check_retrieval_graph() -> None:
    section("6. 流式用的检索子图")

    case = _scenario()
    sub = build_retrieval_graph(case.retriever, case.reranker)
    final = sub.invoke(case.state())
    check("子图产出 ranked", len(final.get("ranked") or []) == 1)
    check(
        "子图**不**产出回答（answer 为空；生成由服务层流式产出）",
        not final.get("answer"),
        f"实际 answer={final.get('answer')!r}",
    )
    check("子图不会自己置 refused（拒答判定在服务层做）", not final.get("refused"))
    check("子图产出 candidates", len(final.get("candidates") or []) >= 1)
    check("子图检索到的正文正确", case.text in final["ranked"][0].chunk.text)
    check("子图命中片段的页码可用（供 SSE sources 事件使用）",
          final["ranked"][0].chunk.page_label == "第3页")

    empty = _scenario(empty_kb=True)
    sub = build_retrieval_graph(empty.retriever, empty.reranker)
    final = sub.invoke(empty.state())
    check("子图在空库上产出空 ranked（服务层据此走拒答）", not final.get("ranked"))


# ---------------------------------------------------------------------------
# 7. ChatService
# ---------------------------------------------------------------------------


def _service(**scenario_kwargs) -> tuple[ChatService, Scenario]:
    """装配一个跑在内存假上游上的 ChatService（非流式图 + 检索子图）。

    `repository` 可选用途：需要走"带 owner_id"的链路（如 HTTP 组）时，
    服务里必须真有仓储，否则会落到"无仓储即拒绝"的分支——那是防漏装的设计，
    不是 bug。
    """
    repository = scenario_kwargs.pop("repository", None)
    case = _scenario(**scenario_kwargs)
    service = ChatService(
        graph=case.graph,  # type: ignore[arg-type]
        llm=case.llm,
        retrieval_graph=build_retrieval_graph(case.retriever, case.reranker),
        repository=repository,
    )
    return service, case


def check_chat_service() -> None:
    section("7. ChatService：引用映射与失败语义")

    service, case = _service()
    result = service.ask(case.kb_id, case.question, top_k=5, rerank_k=4)
    check("ask：answer 非空", bool(result.answer.text))
    check("ask：refused=False", result.answer.refused is False)
    check("ask：sources 非空", len(result.sources) == 1)
    check("ask：citations 条数与 sources 一致", len(result.answer.citations) == 1)
    citation = result.answer.citations[0]
    check("ask：引用页码来自片段（第3页）", citation.page == "第3页", f"实际 {citation.page!r}")
    check("ask：引用文件名来自片段", citation.filename == "handbook.pptx", citation.filename)
    check("ask：引用 doc_id 来自片段", citation.doc_id == "doc-1")
    check("ask：引用带摘要", "报销" in citation.snippet)
    check("ask：query 回填为实际检索用的查询", result.query == case.question)
    check("ask：未触发改写", result.retried is False)
    check("ask：带回上下文用量（输入 token / 利用率）",
          bool(result.usage) and result.usage.get("prompt_tokens", 0) > 0,
          str(result.usage))
    check("ask：正常回答不会被标成截断", result.truncated is False)

    service, case = _service(empty_kb=True)
    refused = service.ask(case.kb_id, case.question)
    check("拒答时 refused=True", refused.answer.refused is True)
    check("拒答时 citations 为空（不作无依据的引用）", refused.answer.citations == [])
    check("拒答时 sources 为空（库里确实什么都没有）", refused.sources == [])
    check("拒答时 text 为固定话术", refused.answer.text == NO_ANSWER_TEXT)
    check("拒答路径没有调用模型 → usage 为 None", refused.usage is None, str(refused.usage))

    service, case = _service(drop_first=1)
    retried = service.ask(case.kb_id, case.question)
    check("触发改写重试时 retried=True", retried.retried is True)
    check("触发改写重试时 query 是改写后的查询", retried.query == "改写后的查询",
          retried.query)

    expect_error(
        "空问题 → ValidationError（不是返回空答案）",
        ValidationError,
        "问题不能为空",
        lambda: service.ask(case.kb_id, "   "),
    )

    # 图跑完却没有 answer：模拟 graph 被改坏
    class BrokenGraph:
        def invoke(self, _state):  # noqa: ANN001, ANN202
            return {"refused": False, "ranked": []}

    broken = ChatService(
        graph=BrokenGraph(),  # type: ignore[arg-type]
        llm=CountingLLM(),
        retrieval_graph=build_retrieval_graph(_retriever(FakeStore(), FakeEmbedder()), None),
    )
    expect_error(
        "图无产出 → UpstreamError（宁可报错也不返回空回答）",
        UpstreamError,
        "没有产出回答文本",
        lambda: broken.ask("kb-x", "问题"),
    )


# ---------------------------------------------------------------------------
# 8. LLM 适配器
# ---------------------------------------------------------------------------


def check_llm_adapter() -> None:
    section("8. LLM 适配器：消息组装、内容抽取与错误映射")

    client = FakeLLMClient(default="你好")
    llm = OpenAICompatibleLLM(client=client)
    check("generate 返回文本", llm.generate("s", "u") == "你好")
    messages = client.messages[-1]
    check("消息顺序为 [system, human]", len(messages) == 2, f"实际 {len(messages)}")
    check("system 内容正确", messages[0].content == "s")
    check("human 内容正确", messages[1].content == "u")

    client = FakeLLMClient(default="x")
    OpenAICompatibleLLM(client=client).generate("   ", "u")
    check(
        "system 为空白时不构造 SystemMessage（部分后端对空 system 报 400）",
        len(client.messages[-1]) == 1,
        f"实际 {len(client.messages[-1])}",
    )

    llm = OpenAICompatibleLLM(client=FakeLLMClient(default=[{"type": "text", "text": "列表内容"}]))
    check("多模态列表 content 能抽出文本", llm.generate("s", "u") == "列表内容")

    llm = OpenAICompatibleLLM(client=FakeLLMClient(default=[{"image": "x"}, "尾巴"]))
    check("列表里的非文本块被跳过", llm.generate("s", "u") == "尾巴")

    llm = OpenAICompatibleLLM(client=FakeLLMClient(empty=True))
    expect_error("空回复 → UpstreamError", UpstreamError, "空内容", lambda: llm.generate("s", "u"))

    llm = OpenAICompatibleLLM(client=FakeLLMClient(error=ConnectionError("connection refused")))
    expect_error(
        "连接失败 → UpstreamError（不是裸 ConnectionError）",
        UpstreamError,
        "调用大模型失败",
        lambda: llm.generate("s", "u"),
    )

    llm = OpenAICompatibleLLM(base_url=" ", model=" ")
    expect_error(
        "base_url 未配置 → ConfigurationError",
        ConfigurationError,
        "LLM_BASE_URL",
        lambda: llm.generate("s", "u"),
    )

    llm = OpenAICompatibleLLM(base_url="http://127.0.0.1:1/v1", model=" ")
    expect_error(
        "model 未配置 → ConfigurationError",
        ConfigurationError,
        "LLM_MODEL",
        lambda: llm.generate("s", "u"),
    )

    async def collect(gen) -> list[str]:  # noqa: ANN001
        return [piece async for piece in gen]

    llm = OpenAICompatibleLLM(client=FakeLLMClient(stream_pieces=["甲", "", "乙"]))
    pieces = asyncio.run(collect(llm.stream("s", "u")))
    check("流式产出增量（空分片被跳过）", pieces == ["甲", "乙"], f"实际 {pieces}")

    llm = OpenAICompatibleLLM(client=FakeLLMClient(error=TimeoutError("timed out")))
    expect_error(
        "流式超时 → UpstreamTimeoutError(504)，而不是笼统的 502",
        UpstreamTimeoutError,
        "超时",
        lambda: asyncio.run(collect(llm.stream("s", "u"))),
    )


class _ReadTimeout(Exception):
    """模拟 httpx.ReadTimeout：类名里带 timeout，消息里没有。"""


def check_llm_timeout() -> None:
    """超时识别与 504 语义。

    为什么要单独验：**超时是唯一"客户端可以安全重试"的上游故障**。
    把它误映射成 502，前端就只会显示"稍后再试"而不给重试按钮；
    反过来把连接失败映射成 504，又会让前端无脑重试一个已经挂掉的服务。
    两种错都在日志里看不出来——它们都只是一行 warning。
    """
    section("8b. LLM 超时：识别方式与 504 语义")

    cases: list[tuple[str, Exception]] = [
        ("内置 TimeoutError", TimeoutError("timed out")),
        ("asyncio.TimeoutError", asyncio.TimeoutError()),
        ("类名带 timeout（httpx.ReadTimeout）", _ReadTimeout("read operation failed")),
        ("消息带 timed out（反代常见）", Exception("request timed out after 60s")),
        ("消息带 time out（带空格）", Exception("Connection time out")),
    ]
    for label, exc in cases:
        llm = OpenAICompatibleLLM(client=FakeLLMClient(error=exc))
        expect_error(
            f"{label} → UpstreamTimeoutError",
            UpstreamTimeoutError,
            "超时",
            lambda llm=llm: llm.generate("s", "u"),
        )

    llm = OpenAICompatibleLLM(client=FakeLLMClient(error=_ReadTimeout("boom")))
    try:
        llm.generate("s", "u")
    except UpstreamTimeoutError as exc:
        timeout_exc = exc
    else:
        timeout_exc = None
    check(
        "504 的 detail 带上 timeout_s 与 streaming（便于判断该调哪个配置）",
        timeout_exc is not None
        and timeout_exc.detail.get("timeout_s") == llm._timeout  # noqa: SLF001
        and timeout_exc.detail.get("streaming") is False,
        str(getattr(timeout_exc, "detail", {})),
    )
    check(
        "超时提示里点名 LLM_TIMEOUT 与 RERANK_TOP_N（可执行的处置动作）",
        timeout_exc is not None
        and "LLM_TIMEOUT" in timeout_exc.message
        and "RERANK_TOP_N" in timeout_exc.message,
        getattr(timeout_exc, "message", ""),
    )
    check(
        "UpstreamTimeoutError 是 504 且 code=upstream_timeout",
        UpstreamTimeoutError.http_status == 504
        and UpstreamTimeoutError.code == "upstream_timeout",
        f"{UpstreamTimeoutError.http_status} / {UpstreamTimeoutError.code}",
    )
    check(
        "超时是 UpstreamError 的子类（旧调用方按 502 处理也不会漏）",
        issubclass(UpstreamTimeoutError, UpstreamError),
    )

    for label, exc in (
        ("连接被拒", ConnectionError("connection refused")),
        ("500 响应", RuntimeError("upstream returned 500")),
    ):
        llm = OpenAICompatibleLLM(client=FakeLLMClient(error=exc))
        expect_error(
            f"{label} → UpstreamError（502，不可盲目重试）",
            UpstreamError,
            "失败",
            lambda llm=llm: llm.generate("s", "u"),
        )
        try:
            llm.generate("s", "u")
        except UpstreamError as exc:
            check(
                f"{label} 不被误判为超时",
                not isinstance(exc, UpstreamTimeoutError),
                type(exc).__name__,
            )


# ---------------------------------------------------------------------------
# 9. SSE 报文与 HTTP 接口
# ---------------------------------------------------------------------------


def check_sse_format() -> None:
    section("9. SSE 报文格式")

    from app.api.v1.chat import _sse_format

    frame = _sse_format("delta", '{"type":"delta","content":"你好"}')
    check("以 event: 开头", frame.startswith("event: delta\n"))
    check("以空行结尾（SSE 的消息分隔符）", frame.endswith("\n\n"))
    check("data 段占且只占一行", len(frame.split("\n\n")[0].splitlines()) == 2, repr(frame))

    payload = {"type": "done", "answer": "第一行\n第二行"}
    frame = _sse_format("done", json.dumps(payload, ensure_ascii=False))
    body = frame.split("\n\n")[0]
    check(
        "正文里的换行被 JSON 转义，不产生裸换行（否则一条消息会被截成两条）",
        len(body.splitlines()) == 2,
        repr(body),
    )
    check("中文未被转义成 \\uXXXX（SSE 本身是 UTF-8）", "\\u" not in body)


class FakeCollections:
    """假知识库服务：只放行 kb-ok，其余一律 404。

    放行名单**写死**而不是"永远放行"：这样"请求打的是 kb-ok、
    数据却存在随机 kb 下"这类假失败会立刻暴露（本节踩过一次）。
    """

    def ensure_exists(self, kb_id: str, owner_id: str | None = None) -> None:  # noqa: ANN001
        if kb_id != HTTP_KB:
            raise NotFoundError(f"知识库不存在：{kb_id}")


class _FakeUser:
    """鉴权依赖的最小替身：本节验的是问答契约，不是鉴权。"""

    id = "u-chat"


class _OwnershipRepo:
    """只回答"这个库是不是这个人的"，别的什么都不做。

    生产里 ChatService 通过它做归属校验；不装它的话，
    "带了 owner_id 却没有仓储"会走宁可错杀的拒绝分支（这是有意的设计），
    本节要验的是问答契约，所以得把这一环补上。
    """

    def __init__(self, kb_id: str, owner_id: str) -> None:
        self._kb_id = kb_id
        self._owner_id = owner_id

    def get_knowledge_base(self, kb_id: str, owner_id: str):  # noqa: ANN201
        if kb_id == self._kb_id and owner_id == self._owner_id:
            return {"id": kb_id, "name": "fake"}
        return None


def _http_app(service: ChatService):
    """最小可用的问答应用：只挂 chat 路由 + 统一异常处理 + 依赖覆盖。"""
    from fastapi import FastAPI

    from app.api.deps import get_chat_service, get_collection_service
    from app.api.v1.auth import get_auth_service, require_current_user
    from app.api.v1.chat import router
    from app.core.exceptions import register_exception_handlers

    fake_user = _FakeUser()
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_chat_service] = lambda: service
    app.dependency_overrides[get_collection_service] = FakeCollections
    # 路由级 router_dependencies 也走 Depends(require_current_user)，覆盖它即生效
    app.dependency_overrides[require_current_user] = lambda: fake_user
    app.dependency_overrides[get_auth_service] = lambda: None
    return app


def check_http_routes() -> None:
    section("10. HTTP 接口：/chat 与 /chat/stream")

    from fastapi.testclient import TestClient

    service, case = _service(
        kb_id=HTTP_KB, repository=_OwnershipRepo(HTTP_KB, _FakeUser.id)
    )
    with TestClient(_http_app(service)) as client:
        response = client.post(
            "/api/v1/chat", json={"kb_id": HTTP_KB, "question": case.question}
        )
        check("POST /chat → 200", response.status_code == 200, response.text[:200])
        body = response.json()
        check("响应含 answer", bool(body.get("answer")))
        check("响应含 sources 且带页码",
              bool(body["sources"]) and body["sources"][0]["page"] == "第3页",
              json.dumps(body.get("sources"), ensure_ascii=False)[:160])
        check("响应含 citations 且带页码",
              bool(body["citations"]) and body["citations"][0]["page"] == "第3页")
        check("响应含 refused=False", body.get("refused") is False)
        check("sources 带四个阶段的分数口径（vector_score 可为 None，键必须在）",
              "vector_score" in body["sources"][0] and "rerank_score" in body["sources"][0])

        response = client.post("/api/v1/chat", json={"kb_id": "kb-missing", "question": "任意"})
        check("知识库不存在 → 404", response.status_code == 404, str(response.status_code))
        check(
            "404 响应体是统一错误结构",
            response.json().get("code") == "not_found",
            response.text[:160],
        )

        response = client.post("/api/v1/chat", json={"kb_id": "kb-ok", "question": ""})
        check("空问题 → 422", response.status_code == 422, str(response.status_code))

    # --- 流式：正常 ---
    stream_service, _case = _service(
        kb_id=HTTP_KB, repository=_OwnershipRepo(HTTP_KB, _FakeUser.id)
    )
    _patch_stream_llm(stream_service, ["五千", "元", "。"])
    with TestClient(_http_app(stream_service)) as client:
        response = client.post(
            "/api/v1/chat/stream", json={"kb_id": HTTP_KB, "question": "报销金额上限是多少？"}
        )
        check("POST /chat/stream → 200", response.status_code == 200, str(response.status_code))
        check(
            "Content-Type 为 text/event-stream",
            response.headers.get("content-type", "").startswith("text/event-stream"),
            response.headers.get("content-type", ""),
        )
        check("响应头禁用了缓冲（否则流式形同虚设）",
              response.headers.get("x-accel-buffering") == "no")
        events = _parse_sse(response.text)
        kinds = [name for name, _ in events]
        check("事件类型序列为 sources → delta* → done",
              kinds == ["sources", "delta", "delta", "delta", "done"],
              f"实际 {kinds}")
        check("sources 事件带页码", events[0][1]["sources"][0]["page"] == "第3页")
        check("delta 内容拼接等于完整回答",
              "".join(e["content"] for name, e in events if name == "delta") == "五千元。")
        check("done 事件带完整回答与引用",
              events[-1][1]["answer"] == "五千元。"
              and events[-1][1]["citations"][0]["page"] == "第3页")
        check("done 事件 refused=False", events[-1][1]["refused"] is False)

    # --- 流式：无可用片段 → 拒答 ---
    stream_service, _case = _service(
        kb_id=HTTP_KB, empty_kb=True, repository=_OwnershipRepo(HTTP_KB, _FakeUser.id)
    )
    _patch_stream_llm(stream_service, ["不该出现"])
    with TestClient(_http_app(stream_service)) as client:
        response = client.post(
            "/api/v1/chat/stream", json={"kb_id": HTTP_KB, "question": "报销金额上限是多少？"}
        )
        events = _parse_sse(response.text)
        kinds = [name for name, _ in events]
        check("拒答流：事件序列仍完整（sources → delta → done）",
              kinds == ["sources", "delta", "done"], f"实际 {kinds}")
        check("拒答流：sources 为空列表", events[0][1]["sources"] == [])
        check("拒答流：delta 就是固定话术", events[1][1]["content"] == NO_ANSWER_TEXT)
        check("拒答流：done.refused=True 且 citations 为空",
              events[-1][1]["refused"] is True and events[-1][1]["citations"] == [])
        check("拒答流：**未调用 LLM 生成**（流式片段没有出现）",
              "不该出现" not in response.text)

    # --- 流式：上游报错 → error 事件收尾 ---
    stream_service, _case = _service(
        kb_id=HTTP_KB, repository=_OwnershipRepo(HTTP_KB, _FakeUser.id)
    )
    _patch_stream_llm(stream_service, [], error=UpstreamError("上游挂了"))
    with TestClient(_http_app(stream_service)) as client:
        response = client.post(
            "/api/v1/chat/stream", json={"kb_id": HTTP_KB, "question": "报销金额上限是多少？"}
        )
        events = _parse_sse(response.text)
        kinds = [name for name, _ in events]
        check("出错流以 error 事件收尾", kinds[-1] == "error", f"实际 {kinds}")
        check("出错流仍先推了 sources（前端已渲染的引用不会突然消失）",
              kinds[0] == "sources")
        check("error 事件带 detail", bool(events[-1][1].get("detail")))

    # --- 流式：知识库不存在 → 建流之前就 404 ---
    stream_service, _case = _service(kb_id=HTTP_KB)
    with TestClient(_http_app(stream_service)) as client:
        response = client.post(
            "/api/v1/chat/stream", json={"kb_id": "kb-missing", "question": "任意"}
        )
        check("流式接口对不存在的知识库返回 404（状态码还能改的时刻）",
              response.status_code == 404, str(response.status_code))


def _patch_stream_llm(service: ChatService, pieces: list[str], error: Exception | None = None) -> None:
    """把 ChatService 里的 LLM 换成受控的假流式实现。

    ChatService 的 `_llm` 是私有属性，这里直接替换——自检脚本本就是
    白盒工具，比给生产代码开一个 setter 更划算。
    """
    service._llm = _StreamLLM(pieces, error)  # noqa: SLF001


class _StreamLLM:
    """只实现 stream 的假 LLM（流式路径不用 generate）。"""

    def __init__(self, pieces: list[str], error: Exception | None = None) -> None:
        self._pieces = list(pieces)
        self._error = error

    def generate(self, system_prompt, user_prompt):  # noqa: ANN001, ANN201
        return "".join(self._pieces)

    async def stream(self, system_prompt, user_prompt):  # noqa: ANN001, ANN201
        for piece in self._pieces:
            yield piece
        if self._error is not None:
            raise self._error


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """把 SSE 报文解析成 [(event, payload), ...]。"""
    events: list[tuple[str, dict]] = []
    for raw in text.split("\n\n"):
        if not raw.strip():
            continue
        name = ""
        data = ""
        for line in raw.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        events.append((name, json.loads(data)))
    return events


# ---------------------------------------------------------------------------
# 11. SSE 客户端断开：关断信号必须穿透到最底层，且不得被 except Exception 吞掉
# ---------------------------------------------------------------------------
#
# 为什么必须单独测这一段：
#     `TestClient` 会同步消费完整条响应，**永远走不到"半流断开"**，
#     所以 `check_http_routes` 里那些 HTTP 用例覆盖不到这条路径。
#     而这条路径错了不会报错、只会**悄悄泄露**：客户端早就走了，
#     服务端还在往下生成、占着一条 LLM 连接继续烧 token。
#
# 真实的断开是怎么发生的（不是 aclose，而是 cancel）：
#     Starlette 1.7 的 `StreamingResponse` **从不调用** `body_iterator.aclose()`。
#     断开时它 cancel 掉跑 `stream_response` 的那个任务，
#     于是 `CancelledError` 被投进 `_sse_generator` 停住的地方，
#     沿 `ask_stream` → `llm.stream` → client.astream 一路向下钻到底。
#     正因此 11.3 才是"生产路径"，11.1 的 `aclose()` 是"手动关"/兜底路径。
#
# 关键事实（本机实测，CPython 3.14.7 + asyncio）：
#     ① `GeneratorExit` 与 `asyncio.CancelledError` 都是 **BaseException** 的
#        子类，`except Exception` 根本拦不到——这正是本组要钉住的点。
#     ② `aclose()` 之后信号**不是同步**到达最底层的：`aclose()` 只把
#        `GeneratorExit` 抛进 `_sse_generator`，中间那几层生成器要等自己被回收，
#        asyncio 的 async-gen finalizer 才会把它们的 `aclose()` 排进循环。
#        ⇒ 断言前必须先 `_drain_loop()`，否则会得到"信号还没走到底"的**假失败**。
#     ③ 断开时收到的异常类型：11.3 是 `CancelledError`（同步可见）；
#        11.1 的 `aclose()` 在本机是 `GeneratorExit`。两者都合法，故断言
#        "是这两者之一"而不是写死某一个（换 asyncio 实现可能变）。


async def _drain_loop(times: int = 10) -> None:
    """让事件循环把手头排队的回调（含 async-generator 终结器）跑完。

    见本组注释 ②：不 drain 就断言底层是否收到关闭信号，会假失败。
    """
    for _ in range(times):
        await asyncio.sleep(0)


class _ProbeClient:
    """假 ChatOpenAI：`astream` 记录**有没有异常被抛进生成器**。

    `seen` 是本组断言的核心——只有关闭/取消信号真的穿透到这一层，
    底层那条 httpx 连接才会被释放；`unwound`（finally 是否执行）
    与 `completed`（是否自然跑完）一起用来区分"被中断"和"正常结束"。
    """

    def __init__(self, pieces=("甲", "乙", "丙", "丁", "戊"), delay: float = 0.0) -> None:
        self._pieces = list(pieces)
        self._delay = delay
        self.started = 0
        self.completed = 0
        self.unwound = 0
        self.seen: list[str] = []

    def invoke(self, messages):  # noqa: ANN001, ANN201
        return _Response("".join(self._pieces))

    async def astream(self, messages):  # noqa: ANN001, ANN201
        self.started += 1
        try:
            for piece in self._pieces:
                if self._delay:
                    await asyncio.sleep(self._delay)  # 停在这里，好让取消落进来
                yield _Response(piece)
        except BaseException as exc:  # noqa: BLE001
            # 必须用 BaseException：GeneratorExit / CancelledError 都不是 Exception，
            # 写成 `except Exception` 就永远记不到东西——这正是被测代码的风险点。
            self.seen.append(type(exc).__name__)
            raise
        finally:
            self.unwound += 1
        self.completed += 1


def _stream_service(client: _ProbeClient) -> tuple[ChatService, ChatRequest]:
    """装一个"真 ChatService + 假 ChatOpenAI client"的流式被测对象。

    链路与生产一致：`_sse_generator → ask_stream → OpenAICompatibleLLM.stream
    → client.astream`，四层都在，才能测"信号能不能钻到底"。
    """
    case = _scenario()
    service = ChatService(
        graph=case.graph,  # type: ignore[arg-type]
        llm=OpenAICompatibleLLM(client=client, base_url="http://fake", model="fake"),
        retrieval_graph=build_retrieval_graph(case.retriever, case.reranker),
    )
    return service, ChatRequest(kb_id=case.kb_id, question=case.question)


async def _sse_disconnect_cases() -> None:
    from app.api.v1.chat import _sse_generator

    # --- 11.1 aclose()：手动关闭（兜底路径） ---
    client = _ProbeClient()
    service, payload = _stream_service(client)
    agen = _sse_generator(service, payload, None)

    first = await agen.__anext__()
    check("断开前：首帧是 sources（检索结果先推）", first.startswith("event: sources\n"), first[:40])
    second = await agen.__anext__()
    check("断开前：第 2 帧是 delta", second.startswith("event: delta\n"), second[:40])
    check("断开前：底层 astream 已被调用一次（流已真的开起来）", client.started == 1,
          str(client.started))
    check("断开前：astream 还停在 yield 上（未结束）",
          client.unwound == 0 and client.completed == 0,
          f"unwound={client.unwound} completed={client.completed}")

    clean = True
    try:
        await agen.aclose()
    except BaseException:  # noqa: BLE001
        clean = False
    # 若 except Exception 吞掉关闭信号还接着 yield，CPython 会抛
    # RuntimeError: async generator ignored GeneratorExit。
    check("aclose() 正常返回（无 RuntimeError：没有吞掉 GeneratorExit 再 yield）", clean)

    await _drain_loop()
    # 实测：本机（CPython 3.14.7 + asyncio）`aclose()` 一路送到最底层的是 GeneratorExit。
    # 这里**刻意不写死类名**——换成别的 asyncio 实现可能是 CancelledError，
    # 两者语义等价（都不是 Exception 子类），写死只会让升级 Python 时出现假失败。
    check("aclose() 后：关断信号穿透到最底层 astream（收到 GeneratorExit 或 CancelledError）",
          client.seen in (["GeneratorExit"], ["CancelledError"]),
          f"实际 {client.seen}")
    check("aclose() 后：astream 的 finally 已执行（连接释放路径被走到）", client.unwound == 1,
          str(client.unwound))
    check("aclose() 后：astream 未跑完（中断在 yield 上，不是自然结束）", client.completed == 0,
          str(client.completed))

    # --- 11.2 用 athrow 精确投递：关断信号不许被 except Exception 吞掉 ---
    service, payload = _stream_service(_ProbeClient())
    agen = _sse_generator(service, payload, None)
    await agen.__anext__()
    raised: BaseException | None = None
    frame = None
    try:
        frame = await agen.athrow(GeneratorExit())
    except BaseException as exc:  # noqa: BLE001
        raised = exc
    check("athrow(GeneratorExit) → 原样抛出（重抛），没有被转成一条 error 帧",
          isinstance(raised, GeneratorExit) and frame is None,
          f"raised={raised!r} frame={frame!r}")

    service, payload = _stream_service(_ProbeClient())
    agen = _sse_generator(service, payload, None)
    await agen.__anext__()
    raised = None
    frame = None
    try:
        frame = await agen.athrow(asyncio.CancelledError())
    except BaseException as exc:  # noqa: BLE001
        raised = exc
    check("athrow(CancelledError) → 原样抛出（`except asyncio.CancelledError: raise` 生效）",
          isinstance(raised, asyncio.CancelledError) and frame is None,
          f"raised={raised!r} frame={frame!r}")

    # 对照组：普通 Exception 必须被拦下转成 error 帧。
    # 没有这条，上面两个断言可能只是"因为什么都没拦"而通过——那样测试是空的。
    service, payload = _stream_service(_ProbeClient())
    agen = _sse_generator(service, payload, None)
    await agen.__anext__()
    raised = None
    frame = None
    try:
        frame = await agen.athrow(ValueError("模拟业务炸了"))
    except BaseException as exc:  # noqa: BLE001
        raised = exc
    check("对照组：athrow(ValueError) → 返回 error 帧（普通异常确实被兜住，前两条才不是空断言）",
          raised is None and frame is not None and frame.startswith("event: error\n"),
          f"raised={raised!r} frame={frame!r}")
    with contextlib.suppress(BaseException):
        await agen.aclose()

    # --- 11.3 任务取消：uvicorn/Starlette 断开时真正发生的事（生产路径） ---
    client = _ProbeClient(delay=5.0)  # 每段之间 await，好让取消精确落在 await 上
    service, payload = _stream_service(client)
    agen = _sse_generator(service, payload, None)
    await agen.__anext__()  # 消费掉 sources

    pending = asyncio.create_task(agen.__anext__())
    await asyncio.sleep(0.1)  # 让它跑到 astream 的 await 上停住
    check("取消前：astream 已启动并停在 await 上",
          client.started == 1 and client.unwound == 0,
          f"started={client.started} unwound={client.unwound}")

    pending.cancel()
    raised = None
    try:
        await pending
    except BaseException as exc:  # noqa: BLE001
        raised = exc
    check("取消 __anext__ → CancelledError 重抛（未被 except Exception 变成 error 帧）",
          isinstance(raised, asyncio.CancelledError), repr(raised))
    # 注意：这里**不 drain**，断言的就是"立刻"——取消是同步钻到底的（与 aclose 不同）
    check("取消后（同步，无需等 GC）：CancelledError 已穿透到最底层 astream",
          client.seen == ["CancelledError"], f"实际 {client.seen}")
    check("取消后：astream 的 finally 已执行（连接立即释放，不拖到 GC）", client.unwound == 1,
          str(client.unwound))
    check("取消后：astream 未跑完（客户端走了就不再继续烧 token）", client.completed == 0,
          str(client.completed))


def check_sse_disconnect() -> None:
    section("11. SSE 客户端断开：关断信号穿透与重抛")
    asyncio.run(_sse_disconnect_cases())


# ---------------------------------------------------------------------------
# 12. 流式改写重试（ask_stream 的 enable_retry）
# ---------------------------------------------------------------------------
#
# 为什么单独测这一段：
#     这是本项目里唯一一个**默认关闭**的功能，而默认关闭的功能最容易被
#     写坏却无人察觉——接手时它就是个半成品，三处各缺一块：
#       (a) `ask_stream` 里调的 `self._rewrite_query` 根本不存在；
#       (b) `_initial_state` 不接受 `rewritten_question`；
#       (c) `_sse_generator` 压根没把 `payload.enable_retry` 传给服务。
#     后果是：要么 AttributeError / TypeError，要么**完全静默失效**
#     （请求字段成了死字段，那段代码永远执行不到）。所以本组除了测行为，
#     还专门有 12.8 一条钉住 (c)。
#
# 与第 11 组的分工：第 11 组测"关断信号能不能钻到底"，
# 本组测"重试这条业务分支对不对"，两者互不覆盖。


class _RewriteStreamLLM:
    """流式路径的假 LLM：`generate` 只用于查询改写，`stream` 用于生成。

    与 `_StreamLLM` 的区别是**两个方法分别记录提示词**。
    本组最要紧的一条不变量是"生成用原始问题、改写只影响检索命中"，
    要断言它就必须能分别看到改写提示词与生成提示词的内容。
    """

    def __init__(
        self,
        pieces=("五千", "元", "。"),
        rewrite: str | Exception = "改写后的查询",
    ) -> None:
        self._pieces = list(pieces)
        self._rewrite = rewrite
        self.rewrite_calls: list[str] = []
        self.stream_calls: list[str] = []

    def generate(self, system_prompt, user_prompt):  # noqa: ANN001, ANN201
        self.rewrite_calls.append(user_prompt)
        if isinstance(self._rewrite, Exception):
            raise self._rewrite
        return self._rewrite

    async def stream(self, system_prompt, user_prompt):  # noqa: ANN001, ANN201
        self.stream_calls.append(user_prompt)
        for piece in self._pieces:
            yield piece


class _OpenRepo:
    """放行一切的假仓储：本节验的是重试语义，不是归属。

    为什么不能省：HTTP 路径必然带 owner_id，服务里没有仓储就会落到
    "宁可错杀"的拒绝分支（那是防漏装的设计）。所以走 HTTP 的用例
    一律装它，好让断言失败时指向重试逻辑本身，而不是被 404 挡住。
    """

    def get_knowledge_base(self, kb_id: str, owner_id: str):  # noqa: ANN201
        return {"id": kb_id, "name": "fake"}


def _service_with_llm(
    case: Scenario,
    llm: _RewriteStreamLLM,
    retry_default: bool | None = None,
    repository: object | None = None,
) -> ChatService:
    """用给定的假 LLM 装配流式服务（图与检索子图都跑在场景的假上游上）。

    `retry_default` 显式注入：三态开关里的"未表态"那一态要落回服务端默认值，
    自检需要把两种默认值都试一遍，不能靠改环境变量再重启进程。

    `repository` 不传则默认装 `_OpenRepo`（放行一切）：本节的用例大半要走
    HTTP，而 HTTP 必带 owner_id，没仓储会被归属校验直接挡成 404。
    """
    return ChatService(
        graph=case.graph,  # type: ignore[arg-type]
        llm=llm,
        retrieval_graph=build_retrieval_graph(case.retriever, case.reranker),
        repository=_OpenRepo() if repository is None else repository,  # type: ignore[arg-type]
        retry_default=retry_default,
    )


def _retry_service(
    *, retry_default: bool | None = None, **scenario_kwargs
) -> tuple[ChatService, Scenario, _RewriteStreamLLM]:
    """装一个跑在内存假上游上的流式被测对象，改写与生成可分别控制。"""
    case = _scenario(**scenario_kwargs)
    llm = _RewriteStreamLLM()
    return _service_with_llm(case, llm, retry_default), case, llm


async def _collect_sse(service: ChatService, payload: ChatRequest) -> list[tuple[str, dict]]:
    """走真实的 `_sse_generator` 把事件收全，解析成 [(event, payload), ...]。

    刻意从 `_sse_generator` 进而不是直接调 `ask_stream`：
    请求字段就是在这一层被丢掉的，从服务层进就测不到那个 bug。
    """
    from app.api.v1.chat import _sse_generator

    text = ""
    async for frame in _sse_generator(service, payload, None):
        text += frame
    return _parse_sse(text)


def _payload(case: Scenario, *, enable_retry: bool | None = None) -> ChatRequest:
    """构造请求体。

    `enable_retry=None` 就是"未表态"那一态（HTTP 上省略字段与显式传 null
    在 pydantic 里落到同一个 None，服务端无从区分，也不需要区分）。
    """
    return ChatRequest(kb_id=case.kb_id, question=case.question, enable_retry=enable_retry)


async def _stream_retry_cases() -> None:
    # --- 12.1 默认关闭：首轮无可用片段 → 直接拒答，一次改写都不做 ---
    service, case, llm = _retry_service(drop_first=1)
    events = await _collect_sse(service, _payload(case, enable_retry=False))
    kinds = [name for name, _ in events]
    check("默认（enable_retry=false）：事件里没有 retry", "retry" not in kinds, f"实际 {kinds}")
    check(
        "默认：首轮无可用片段 → sources 为空 + 固定拒答话术",
        kinds == ["sources", "delta", "done"]
        and events[0][1]["sources"] == []
        and events[1][1]["content"] == NO_ANSWER_TEXT,
        f"实际 {kinds}",
    )
    check(
        "默认：改写提示词一次都没发出去（不是「改了没用」，是根本没改）",
        llm.rewrite_calls == [],
        f"实际 {llm.rewrite_calls}",
    )
    check("默认：重排只跑一次（没有发生第二次检索）", len(case.reranker.calls) == 1,
          f"实际 {len(case.reranker.calls)}")

    # --- 12.2 打开：retry 事件在最前，重试后 sources 来自第二轮 ---
    service, case, llm = _retry_service(drop_first=1)
    events = await _collect_sse(service, _payload(case, enable_retry=True))
    kinds = [name for name, _ in events]
    check(
        "打开 enable_retry：事件序列为 retry → sources → delta* → done",
        kinds == ["retry", "sources", "delta", "delta", "delta", "done"],
        f"实际 {kinds}",
    )
    check("retry 事件排在 sources **之前**（前端先知道「在等」，再拿到引用）",
          kinds.index("retry") < kinds.index("sources"))
    retry_event = events[0][1]
    check(
        "retry 事件自描述：带 attempt / max_attempts / reason",
        retry_event.get("attempt") == 1
        and retry_event.get("max_attempts") == MAX_RETRY
        and bool(retry_event.get("reason")),
        json.dumps(retry_event, ensure_ascii=False),
    )
    sources = events[1][1]["sources"]
    check(
        "重试后 sources 非空且带页码（第二轮确实救回来了）",
        len(sources) == 1 and sources[0]["page"] == "第3页",
        json.dumps(sources, ensure_ascii=False)[:160],
    )
    check(
        "第二次检索用的是改写后的查询",
        case.embedder.query_calls.count("改写后的查询") == 1,
        f"实际 {case.embedder.query_calls}",
    )
    check("重排共跑两次（首轮全丢 + 重试救回）", len(case.reranker.calls) == 2,
          f"实际 {len(case.reranker.calls)}")
    check("重试成功后 done.refused=False 且 answer 非空",
          events[-1][1]["refused"] is False and bool(events[-1][1]["answer"]))

    # --- 12.3 不变量：生成永远用**原始问题** ---
    check(
        "生成用的是原始问题，不是改写查询（改写只影响检索命中）",
        len(llm.stream_calls) == 1
        and case.question in llm.stream_calls[0]
        and "改写后的查询" not in llm.stream_calls[0],
        llm.stream_calls[0][:100] if llm.stream_calls else "<生成根本没被调用>",
    )
    check("改写请求里带的是原始问题",
          len(llm.rewrite_calls) == 1 and case.question in llm.rewrite_calls[0])

    # --- 12.4 重试次数与完整图的 MAX_RETRY 同源 ---
    service, case, llm = _retry_service(drop_first=99)
    events = await _collect_sse(service, _payload(case, enable_retry=True))
    kinds = [name for name, _ in events]
    retries = [name for name in kinds if name == "retry"]
    check(
        f"重试次数恰好 MAX_RETRY({MAX_RETRY}) —— 与 /chat 同一份上限，不是另定的数",
        len(retries) == MAX_RETRY,
        f"实际 {len(retries)} 次",
    )
    check(
        "重试后仍无可用片段 → 以正常拒答收尾（不是 error 事件）",
        kinds[-1] == "done" and events[-1][1]["refused"] is True and "error" not in kinds,
        f"实际 {kinds}",
    )
    check(
        "重试只多跑一次检索，且全程没有调生成",
        len(case.reranker.calls) == 1 + MAX_RETRY and llm.stream_calls == [],
        f"reranker={len(case.reranker.calls)} stream={len(llm.stream_calls)}",
    )

    # --- 12.4b 「与图同源」的可证伪验证：把上限抬到 3，流式重试必须跟着变 ---
    #
    # 没有这条，「重试次数恰好 MAX_RETRY」在 MAX_RETRY==1 时与「服务层硬编码
    # 重试一次」完全无法区分——两边都得 1。只有临时改写两处常量，才能证明
    # 服务层读的是**与完整图同一份**上限，而不是自己写死的数字。
    import app.rag.graph as graph_module  # noqa: PLC0415
    import app.services.chat_service as chat_module  # noqa: PLC0415

    originals = (graph_module.MAX_RETRY, chat_module.MAX_RETRY)
    graph_module.MAX_RETRY = 3
    chat_module.MAX_RETRY = 3
    try:
        service, case, llm = _retry_service(drop_first=99)
        events = await _collect_sse(service, _payload(case, enable_retry=True))
        kinds = [name for name, _ in events]
        check(
            "把 MAX_RETRY 抬到 3 → 流式重试确实跟着跑 3 次（判据与上限都来自图那一份）",
            kinds.count("retry") == 3,
            f"实际 {kinds}",
        )
        check("抬到 3 后：首轮 + 3 次重试 → 重排共 4 次", len(case.reranker.calls) == 4,
              f"实际 {len(case.reranker.calls)}")
        check("retry 事件的 max_attempts 跟着变成 3", events[0][1]["max_attempts"] == 3,
              json.dumps(events[0][1], ensure_ascii=False))
    finally:
        graph_module.MAX_RETRY, chat_module.MAX_RETRY = originals
    check("常量已还原（不污染后续用例）", graph_module.MAX_RETRY == originals[0])

    # --- 12.5 改写无产出 → 提前收网，不白跑第二次检索 ---
    case = _scenario(drop_first=1)
    llm = _RewriteStreamLLM(rewrite=case.question)  # 模型原样抄回问题
    service = _service_with_llm(case, llm)
    events = await _collect_sse(service, _payload(case, enable_retry=True))
    kinds = [name for name, _ in events]
    check("改写与原问题相同时仍推 retry 事件（如实告知「在改写」，不假装重试成功了）",
          kinds.count("retry") == 1, f"实际 {kinds}")
    check(
        "改写无产出 → **不再重跑检索**（第二轮结果必然相同，白跑一次纯浪费）",
        len(case.reranker.calls) == 1,
        f"实际 {len(case.reranker.calls)}",
    )
    check("改写无产出 → 正常拒答收尾", kinds[-1] == "done" and events[-1][1]["refused"] is True)

    # --- 12.6 改写层报错 → 降级为原问题，不升级成 500 ---
    case = _scenario(drop_first=1)
    llm = _RewriteStreamLLM(rewrite=UpstreamError("改写时上游挂了"))
    service = _service_with_llm(case, llm)
    events = await _collect_sse(service, _payload(case, enable_retry=True))
    kinds = [name for name, _ in events]
    check(
        "改写报错 → 降级为原问题，以 done 收尾（不把一次改写失败升级成 500 / error 事件）",
        "error" not in kinds and kinds[-1] == "done" and events[-1][1]["refused"] is True,
        f"实际 {kinds}",
    )
    check("改写报错 → 同样不重跑检索", len(case.reranker.calls) == 1,
          f"实际 {len(case.reranker.calls)}")

    # --- 12.7 首轮就取到片段时，打开开关也不多做 ---
    service, case, llm = _retry_service()
    events = await _collect_sse(service, _payload(case, enable_retry=True))
    kinds = [name for name, _ in events]
    check(
        "首轮就取到片段时，即使打开 enable_retry 也不改写、不多跑检索",
        kinds == ["sources", "delta", "delta", "delta", "done"]
        and llm.rewrite_calls == []
        and len(case.reranker.calls) == 1,
        f"实际 {kinds}；改写 {len(llm.rewrite_calls)} 次",
    )


def check_stream_retry() -> None:
    section("12. 流式改写重试（ask_stream 的 enable_retry）")
    asyncio.run(_stream_retry_cases())

    # --- 12.8 HTTP：JSON 里的 enable_retry 必须真的送达服务层 ---
    # 这条是**死字段回归**：本次接手时 `_sse_generator` 没把
    # `payload.enable_retry` 传下去，于是无论请求里写什么，服务层收到的
    # 永远是默认的 False——功能写好了也永远不会被执行，且毫无报错。
    from fastapi.testclient import TestClient

    service, case, _llm = _retry_service(kb_id=HTTP_KB, drop_first=1)
    with TestClient(_http_app(service)) as client:
        response = client.post(
            "/api/v1/chat/stream",
            json={"kb_id": HTTP_KB, "question": case.question, "enable_retry": True},
        )
        events = _parse_sse(response.text)
        kinds = [name for name, _ in events]
        check("HTTP：enable_retry=true 送达服务层（事件里出现 retry，不再是死字段）",
              "retry" in kinds, f"实际 {kinds}")
        sources = next((e.get("sources") for name, e in events if name == "sources"), None)
        check(
            "HTTP：改写 → 再检索 → 生成 全链路走通（sources 非空且带页码）",
            bool(sources) and sources[0]["page"] == "第3页",
            f"sources={sources}",
        )

    service2, case2, _llm2 = _retry_service(kb_id=HTTP_KB, drop_first=1)
    with TestClient(_http_app(service2)) as client:
        response = client.post(
            "/api/v1/chat/stream",
            json={"kb_id": HTTP_KB, "question": case2.question, "enable_retry": False},
        )
        kinds = [name for name, _ in _parse_sse(response.text)]
        check("HTTP：enable_retry=false（默认）→ 不重试", "retry" not in kinds, f"实际 {kinds}")


# ---------------------------------------------------------------------------
# 13. 重试决策归前端用户（三态开关 + 当场询问）
# ---------------------------------------------------------------------------
#
# 第 12 组测的是「重试这条分支本身对不对」，本组测的是「这个决定由谁做」。
# 两件事必须分开测，因为功能写对了、开关照样可能是假的：
#   (a) **默认值盖掉用户的显式选择**：写成 `if not enable_retry: 用默认值`，
#       于是前端关掉开关后服务端照旧重试——界面上一切正常，只是开关是假的。
#       13.1c 就是钉这一条：显式 false 必须压过默认 true。
#   (b) **前端拿不到判断依据**：done 事件不回填 retried / can_retry，
#       "让用户选择"就只能退化成事先盲开开关——而事先谁也不知道这一问
#       用不用得上改写。13.2/13.3 钉的是这份契约。
# 13.5 再把两条路串成一个闭环：默认出结果 → 拒答且 can_retry → 用户点
# 「再找一次」→ 带 enable_retry=true 重发同一个问题 → 真的救回来。


async def _retry_choice_cases() -> None:
    # --- 13.1 三态：显式选择一定压过服务端默认值 ---
    service, case, llm = _retry_service(drop_first=1, retry_default=False)
    events = await _collect_sse(service, _payload(case))
    kinds = [name for name, _ in events]
    check("未表态 + 默认关 → 不重试", "retry" not in kinds, f"实际 {kinds}")

    service, case, llm = _retry_service(drop_first=1, retry_default=True)
    events = await _collect_sse(service, _payload(case))
    kinds = [name for name, _ in events]
    check(
        "未表态 + 默认开 → 重试（证明默认值真的被读到，不是摆设）",
        "retry" in kinds,
        f"实际 {kinds}",
    )

    service, case, llm = _retry_service(drop_first=1, retry_default=True)
    events = await _collect_sse(service, _payload(case, enable_retry=False))
    kinds = [name for name, _ in events]
    check(
        "显式 enable_retry=false **压过**服务端默认 true → 不重试"
        "（写成「if not enable_retry 就用默认值」会在这里翻车）",
        "retry" not in kinds and llm.rewrite_calls == [],
        f"实际 {kinds}；改写 {len(llm.rewrite_calls)} 次",
    )

    service, case, llm = _retry_service(drop_first=1, retry_default=False)
    events = await _collect_sse(service, _payload(case, enable_retry=True))
    kinds = [name for name, _ in events]
    check("显式 enable_retry=true 压过服务端默认 false → 重试", "retry" in kinds, f"实际 {kinds}")

    # --- 13.2 done 事件回填"重试事实"：前端唯一的判断依据 ---
    samples: dict[str, dict] = {}

    def _sample(label: str, done: dict) -> dict:
        """收下一个 done 样本，并**先**钉住四个判断字段齐全。

        顺序很重要：先断言存在性、再读字段。否则少一个键会让本节以 KeyError
        崩掉，整套自检就从"跑失败"退化成"跑不到"——后者更难发现，因为输出里
        既没有 FAIL 也没有汇总行。变异实验第一版正是被这个假象骗过。
        """
        missing = sorted(
            {"retried", "attempts", "can_retry", "query", "usage", "truncated"} - set(done)
        )
        check(
            f"done 字段齐全（{label}）：retried / attempts / can_retry / query / usage / truncated",
            not missing,
            f"缺少 {missing}",
        )
        samples[label] = done
        return done

    service, case, llm = _retry_service(retry_default=False)
    hit = _sample("首轮命中", (await _collect_sse(service, _payload(case)))[-1][1])
    check(
        "首轮命中 → refused=false / attempts=0 / can_retry=false（不该邀请重试）",
        hit["refused"] is False and hit["attempts"] == 0 and hit["can_retry"] is False,
        json.dumps(hit, ensure_ascii=False),
    )
    check(
        "未改写时 done.query 就是原始问题（前端能用它核对检索口径）",
        hit["query"] == case.question,
        hit["query"],
    )

    service, case, llm = _retry_service(drop_first=1, retry_default=False)
    refused_sample = _sample(
        "未重试即拒答", (await _collect_sse(service, _payload(case)))[-1][1]
    )
    check(
        "拒答且没人重试过 → can_retry=true（前端据此弹「再找一次」）",
        refused_sample["can_retry"] is True,
        json.dumps(refused_sample, ensure_ascii=False),
    )
    check(
        "拒答时 done.query 照样回填（排查「到底拿什么去查的」）",
        refused_sample["query"] == case.question,
        refused_sample["query"],
    )

    service, case, llm = _retry_service(drop_first=1, retry_default=True)
    rescued = _sample("重试救回", (await _collect_sse(service, _payload(case)))[-1][1])
    check(
        "重试救回 → retried=true / attempts=1 / can_retry=false / query 是改写后的查询",
        rescued["retried"] is True
        and rescued["attempts"] == 1
        and rescued["can_retry"] is False
        and rescued["query"] == "改写后的查询",
        json.dumps(rescued, ensure_ascii=False),
    )

    service, case, llm = _retry_service(drop_first=99, retry_default=True)
    exhausted = _sample("重试后仍拒答", (await _collect_sse(service, _payload(case)))[-1][1])
    check(
        "额度用满后仍拒答 → can_retry=false（别弹一个点了没用的按钮）",
        exhausted["can_retry"] is False and exhausted["retried"] is True,
        json.dumps(exhausted, ensure_ascii=False),
    )

    # 改写无产出时 attempts 记 1，但 retried 必须仍是 false —— 「调过改写」与
    # 「真的重新检索过」是两件事，混了前端会显示「已重试」却没有新结果。
    case_x = _scenario(drop_first=1)
    llm_x = _RewriteStreamLLM(rewrite=case_x.question)
    service_x = _service_with_llm(case_x, llm_x, retry_default=True)
    barren = _sample("改写无产出", (await _collect_sse(service_x, _payload(case_x)))[-1][1])
    check(
        "改写无产出 → attempts=1 但 retried=false（如实区分「试过改写」与「真的重检索了」）",
        barren["attempts"] == 1 and barren["retried"] is False and barren["can_retry"] is False,
        json.dumps(barren, ensure_ascii=False),
    )

    # --- 13.3 契约一致性：成功与拒答两条路径的 done 键集合必须一模一样 ---
    key_sets = {label: sorted(done) for label, done in samples.items()}
    check(
        "成功路径与拒答路径的 done 字段集合完全一致（前端只需一个解析分支）",
        len({tuple(keys) for keys in key_sets.values()}) == 1,
        json.dumps(key_sets, ensure_ascii=False),
    )
    for label, done in samples.items():
        expected = bool(done["refused"]) and int(done["attempts"]) < MAX_RETRY
        check(
            f"不变式 can_retry == (refused and attempts < MAX_RETRY) 在「{label}」成立",
            done["can_retry"] == expected,
            f"实际 can_retry={done['can_retry']} refused={done['refused']} attempts={done['attempts']}",
        )

    # --- 13.4 retried 不能是凭空填的：与"重排跑了几次"交叉核对 ---
    service, case, llm = _retry_service(drop_first=1, retry_default=True)
    events = await _collect_sse(service, _payload(case))
    done = events[-1][1]
    check(
        "retried=true 与「重排确实跑了两次」对得上",
        done["retried"] is True and len(case.reranker.calls) == 2,
        f"retried={done['retried']} reranker={len(case.reranker.calls)}",
    )
    service, case, llm = _retry_service(drop_first=1, retry_default=False)
    events = await _collect_sse(service, _payload(case))
    done = events[-1][1]
    check(
        "retried=false 与「重排只跑了一次」对得上",
        done["retried"] is False and len(case.reranker.calls) == 1,
        f"retried={done['retried']} reranker={len(case.reranker.calls)}",
    )

    # --- 13.5 当场询问闭环：先用默认出结果，再按用户点击重发 ---
    first_service, first_case, _ = _retry_service(drop_first=1, retry_default=False)
    first_done = (await _collect_sse(first_service, _payload(first_case)))[-1][1]
    check(
        "闭环第一步：默认出结果 → 拒答且 can_retry=true（前端这时才需要问用户）",
        first_done["refused"] is True and first_done["can_retry"] is True,
        json.dumps(first_done, ensure_ascii=False),
    )
    second_service, second_case, _ = _retry_service(drop_first=1, retry_default=False)
    second_events = await _collect_sse(second_service, _payload(second_case, enable_retry=True))
    second_kinds = [name for name, _ in second_events]
    second_sources = next(
        (e.get("sources") for name, e in second_events if name == "sources"), None
    )
    second_done = second_events[-1][1]
    check(
        "闭环第二步：同一个问题带 enable_retry=true 重发 → retry → sources 非空 → 成功回答",
        second_kinds[0] == "retry"
        and bool(second_sources)
        and second_done["refused"] is False,
        f"实际 {second_kinds}",
    )
    check(
        "闭环第二步：额度用掉后 can_retry 收缩为 false（不会诱导用户反复点）",
        second_done["can_retry"] is False and second_done["retried"] is True,
        json.dumps(second_done, ensure_ascii=False),
    )


def check_retry_choice() -> None:
    section("13. 重试决策归前端用户（三态开关 + 当场询问）")

    # 省略字段与显式传 null 解析后必须**完全相同**：服务端只有"未表态"这一
    # 种状态。若哪天给二者分出不同语义，前端传 null 就会得到一个没人定义过
    # 的行为——这类差异不会有任何报错，只会在某个客户端上悄悄表现不同。
    parsed_omitted = ChatRequest(kb_id="k", question="q")
    parsed_null = ChatRequest(kb_id="k", question="q", enable_retry=None)
    check(
        "省略 enable_retry 与显式 null 解析后完全一致（都落到 None）",
        parsed_omitted.enable_retry is None
        and parsed_null.enable_retry is None
        and parsed_omitted.model_dump() == parsed_null.model_dump(),
        f"省略={parsed_omitted.enable_retry!r} null={parsed_null.enable_retry!r}",
    )

    # 「未表态」那一态的取值来自配置项 CHAT_STREAM_RETRY_DEFAULT，所以要证明
    # 它真的接了配置，就必须**临时把配置翻过来**：本机 .env 里这一项恰好
    # 等于类默认值（都是 false），不翻的话"读到配置"与"硬编码了 False"
    # 得到完全相同的结果——这正是最容易被当成"已验证"的假绿。
    from app.core.config import get_settings

    settings = get_settings()
    original_default = bool(settings.chat_stream_retry_default)
    try:
        settings.chat_stream_retry_default = True
        wired = _service_with_llm(_scenario(), _RewriteStreamLLM())
        check(
            "把配置改成 true → 未注入默认值的服务跟着变（默认值那一态真的接了配置）",
            wired.retry_default is True,
            f"实际 {wired.retry_default}",
        )
    finally:
        settings.chat_stream_retry_default = original_default
    check(
        "配置已还原（不污染后续用例）",
        bool(get_settings().chat_stream_retry_default) == original_default,
        f"实际 {get_settings().chat_stream_retry_default}",
    )

    asyncio.run(_retry_choice_cases())

    from fastapi.testclient import TestClient

    # --- 13.6 HTTP：省略 / 显式 null 都落回服务端默认值 ---
    #
    # 注意每个请求都用**独立**的 service：FakeReranker 的 drop_first 是
    # "前 N 次调用返回空"，两个请求共用一个实例时，第二个请求的重排已经是
    # 第 3 次调用、不再丢分，于是它按逻辑就不该重试——会得到一条看似
    # "null 与省略行为不同"的假失败（本组第一版就踩了这个坑）。
    service, case, _llm = _retry_service(kb_id=HTTP_KB, drop_first=1, retry_default=True)
    with TestClient(_http_app(service)) as client:
        omitted = client.post(
            "/api/v1/chat/stream", json={"kb_id": HTTP_KB, "question": case.question}
        )
        kinds_omitted = [name for name, _ in _parse_sse(omitted.text)]
        check(
            "HTTP：省略 enable_retry → 走服务端默认（true）→ 出现 retry",
            "retry" in kinds_omitted,
            f"实际 {kinds_omitted}",
        )

    service_null, case_null, _llm_null = _retry_service(
        kb_id=HTTP_KB, drop_first=1, retry_default=True
    )
    with TestClient(_http_app(service_null)) as client:
        explicit_null = client.post(
            "/api/v1/chat/stream",
            json={"kb_id": HTTP_KB, "question": case_null.question, "enable_retry": None},
        )
        kinds_null = [name for name, _ in _parse_sse(explicit_null.text)]
        check(
            "HTTP：显式传 null 与省略等价（都属于「未表态」，服务端无从区分也无需区分）",
            kinds_null == kinds_omitted,
            f"省略 {kinds_omitted} vs null {kinds_null}",
        )

        # --- 13.7 HTTP：/chat/options 必须与实际行为同一份来源 ---
        options = client.get("/api/v1/chat/options")
        check("HTTP：GET /chat/options 返回 200", options.status_code == 200, str(options.status_code))
        data = options.json()
        check(
            "options 报出的默认值与实际行为一致（默认 true 的实例 → 端点也报 true）",
            data.get("stream_retry_default") is True,
            json.dumps(data, ensure_ascii=False),
        )
        check(
            "options 的重试上限与 MAX_RETRY 同源（前端不要假定为 1）",
            data.get("stream_retry_max_attempts") == MAX_RETRY,
            json.dumps(data, ensure_ascii=False),
        )
        check(
            "options 的事件名齐全（retry/sources/delta/done/error）",
            set(data.get("stream_events") or [])
            == {"retry", "sources", "delta", "done", "error"},
            json.dumps(data.get("stream_events"), ensure_ascii=False),
        )
        check(
            "options 如实说明非流式恒重试（避免前端以为开关对它也有效）",
            data.get("non_stream_always_retries") is True,
            json.dumps(data, ensure_ascii=False),
        )

        bad = client.post(
            "/api/v1/chat/stream",
            json={"kb_id": HTTP_KB, "question": case.question, "enable_retry": "maybe"},
        )
        check(
            "HTTP：enable_retry 传了非法值 → 422（而不是静默退化成默认值）",
            bad.status_code == 422,
            f"实际 {bad.status_code}",
        )

    # 换一个"默认关"的实例：端点必须跟着变，否则说明它在读别的东西
    service2, _case2, _llm2 = _retry_service(kb_id=HTTP_KB, drop_first=1, retry_default=False)
    with TestClient(_http_app(service2)) as client:
        data2 = client.get("/api/v1/chat/options").json()
        check(
            "默认关的实例 → 端点跟着报 false（确实读的是同一个来源，不是写死的常量）",
            data2.get("stream_retry_default") is False,
            json.dumps(data2, ensure_ascii=False),
        )

    # --- 13.8 非流式 /chat：开关对它是空操作，但观测字段必须回填 ---
    #
    # 这是一条**行为特征断言**（characterization）：非流式路径**故意**恒重试一次，
    # 因为它本来就在等完整答案，重试不额外增加用户可感的等待。写下它不是为了
    # 认可这个行为，而是为了让"哪天有人顺手改成对称"这件事必须显式改测试，
    # 而不是悄悄改变线上行为。
    service, case, _llm = _retry_service(kb_id=HTTP_KB, drop_first=1)
    with TestClient(_http_app(service)) as client:
        payload = client.post(
            "/api/v1/chat",
            json={"kb_id": HTTP_KB, "question": case.question, "enable_retry": False},
        ).json()
        check(
            "非流式：传 enable_retry=false 仍会重试一次（恒重试是有意为之，非 bug）",
            payload.get("retried") is True,
            json.dumps(payload, ensure_ascii=False)[:200],
        )
        check(
            "非流式：query 回填为实际检索用的查询（这里是改写后的）",
            payload.get("query") == "改写后的查询",
            payload.get("query"),
        )

    service, case, _llm = _retry_service(kb_id=HTTP_KB)
    with TestClient(_http_app(service)) as client:
        payload = client.post(
            "/api/v1/chat", json={"kb_id": HTTP_KB, "question": case.question}
        ).json()
        check(
            "非流式：首轮就命中 → retried=false 且 query 就是原始问题",
            payload.get("retried") is False and payload.get("query") == case.question,
            json.dumps(payload, ensure_ascii=False)[:200],
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    check_page_label()
    check_format_context()
    check_prompt_consistency()
    check_prompt_hardening()
    check_grounding_audit()
    check_route()
    check_graph_paths()
    check_retrieval_graph()
    check_chat_service()
    check_llm_adapter()
    check_llm_timeout()
    check_sse_format()
    check_http_routes()
    check_sse_disconnect()
    check_stream_retry()
    check_retry_choice()

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
