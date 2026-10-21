# -*- coding: utf-8 -*-
"""模型自检：验证 embedding / rerank 权重可用且行为正确（FlagEmbedding 推理）。

为什么需要这个脚本：
    「模型能加载」和「模型是好的」是两回事。加载成功但输出全零、
    维度不对、query/passage 走错编码路径，都会让检索静默失效。
    所以每次换权重、换机器、换推理库后都应跑一遍语义自检。

判据（不只看「不报错」）：

    | 检查项        | 期望                                                  |
    |--------------|-------------------------------------------------------|
    | 向量维度      | 等于配置的 MILVUS_DIMENSION（bge-large-zh = 1024）      |
    | L2 范数       | 1.0（bge + 余弦检索必须归一化）                          |
    | 确定性        | 同一输入两次编码结果逐位一致                             |
    | 区分度        | sim(query, 相关句) 明显 > sim(query, 无关句)            |
    | **指令只在 query 侧** | embed_query(q) 必须 != embed_documents([q])，|
    |              | 否则说明两条编码路径被合并，相似度会整体偏移               |
    | 重排尺度      | 必须是**原始 logits**（FlagReranker 的 normalize       |
    |              | 默认为 False 即此意）；normalize=True 才套 Sigmoid 到 [0,1] |
    | 重排排序      | 相关对分数 > 无关对分数                                 |
    | 重排回填/截断 | rerank() 按分数降序、rerank_score 已回填、top_n 生效      |
    | 阈值过滤      | 全部低于 min_score 时返回空列表（上层据此拒答）            |

运行（必须用项目自己的 .venv）：
    .venv\\Scripts\\python.exe scripts\\check_models.py

    脚本会把项目根临时加进 sys.path 并切到项目根（pydantic-settings 的
    env_file=".env" 是相对路径，需要以项目根为工作目录）。

退出码：0 全部通过；1 存在失败项。

注意：
    权重已在本机时必须设 HF_HUB_OFFLINE=1，否则 transformers 会尝试连
    huggingface.co，国内会长时间卡住（适配器内部已 setdefault）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# .env 按相对路径加载，必须先切到项目根
os.chdir(PROJECT_ROOT)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.domain.models import Chunk, RetrievedChunk  # noqa: E402
from app.infrastructure.embedding import BgeEmbedder  # noqa: E402
from app.infrastructure.rerank import BgeReranker  # noqa: E402

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


# ---------------------------------------------------------------------------
# 测试语料：三句里只有两句与问题相关，能同时检验排序与阈值过滤
# ---------------------------------------------------------------------------

QUERY = "哪个是向量数据库？"
RELEVANT = [
    "Milvus 是一个开源的向量数据库，支持十亿级向量的相似性检索。",
    "FAISS 是 Facebook 开源的向量相似度搜索库。",
]
IRRELEVANT = "今天中午食堂的红烧肉味道不错，就是有点咸。"
CORPUS = RELEVANT + [IRRELEVANT]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _chunk(text: str, index: int) -> Chunk:
    return Chunk(
        id=f"chunk-{index}",
        kb_id="kb-check",
        doc_id="doc-check",
        text=text,
        chunk_index=index,
        page_start=1,
        page_end=1,
        filename="check.txt",
    )


# ---------------------------------------------------------------------------
# 向量化
# ---------------------------------------------------------------------------


def check_embedding(embedder: BgeEmbedder) -> None:
    print("\n[1] 向量化（FlagEmbedding）")

    dim = embedder.dimension
    check(
        f"维度等于 MILVUS_DIMENSION（{dim}）",
        dim == get_settings().milvus_dimension,
        f"实际 {dim}",
    )

    corpus = np.asarray(embedder.embed_documents(CORPUS), dtype=np.float32)
    query = np.asarray(embedder.embed_query(QUERY), dtype=np.float32)

    check("输出形状正确", corpus.shape == (len(CORPUS), dim), f"实际 {corpus.shape}")
    check("向量 dtype 为 float32", corpus.dtype == np.float32, f"实际 {corpus.dtype}")

    norms = np.linalg.norm(corpus, axis=1)
    check(
        "corpus L2 范数全为 1.0",
        bool(np.allclose(norms, 1.0, atol=1e-5)),
        f"实际 {norms.tolist()}",
    )
    check(
        "query L2 范数为 1.0",
        bool(np.isclose(np.linalg.norm(query), 1.0, atol=1e-5)),
        f"实际 {np.linalg.norm(query)}",
    )

    again = np.asarray(embedder.embed_documents(CORPUS), dtype=np.float32)
    check("两次编码逐位一致（可复现）", bool(np.array_equal(corpus, again)))

    sims = [_cosine(corpus[i], query) for i in range(len(CORPUS))]
    print(f"         相似度：Milvus={sims[0]:.4f}  FAISS={sims[1]:.4f}  食堂={sims[2]:.4f}")
    check(
        "相关句相似度 > 无关句（区分度）",
        sims[0] > sims[2] and sims[1] > sims[2],
        f"实际 {[round(s, 4) for s in sims]}",
    )

    # 关键回归：检索指令只能加在 query 侧。若两条路径输出相同，
    # 说明指令没生效（或反过来给文档块也加了），语义空间会整体偏移。
    instruction = get_settings().embedding_query_instruction
    as_document = np.asarray(embedder.embed_documents([QUERY])[0], dtype=np.float32)
    same = bool(np.allclose(as_document, query, atol=1e-6))
    if instruction:
        check(
            "query 侧加了检索指令（与不加指令的编码结果不同）",
            not same,
            "embed_query 与 embed_documents 输出一致，指令未生效",
        )
        check(
            "指令前缀与配置一致",
            embedder._query_instruction == instruction,  # noqa: SLF001 - 自检脚本读内部值
            f"实际 {embedder._query_instruction!r}",
        )
    else:
        check("未配置检索指令时两条路径应当一致", same)

    check("空输入返回空列表", embedder.embed_documents([]) == [])


# ---------------------------------------------------------------------------
# 重排
# ---------------------------------------------------------------------------


def check_rerank(reranker: BgeReranker) -> None:
    print("\n[2] 重排（FlagEmbedding 交叉编码器）")

    candidates = [RetrievedChunk(chunk=_chunk(text, i)) for i, text in enumerate(CORPUS)]

    # 先看底层分数尺度：必须拿到未经 Sigmoid 的原始 logits
    raw = reranker._load().compute_score(  # noqa: SLF001 - 自检脚本
        [(QUERY, text) for text in CORPUS], batch_size=8, normalize=False
    )
    sigmoid = reranker._load().compute_score(  # noqa: SLF001 - 自检脚本
        [(QUERY, text) for text in CORPUS], batch_size=8, normalize=True
    )
    print(
        f"         原始 logits={[round(float(x), 3) for x in raw]}  "
        f"normalize=True={[round(float(x), 5) for x in sigmoid]}"
    )
    check(
        "取到原始 logits（非 Sigmoid 概率）",
        max(float(x) for x in raw) > 1.0 or min(float(x) for x in raw) < 0.0,
        f"全部落在 [0,1] 内，疑似误用了套 Sigmoid 的输出：{raw}",
    )
    check(
        "normalize=True 时才落到 [0,1]",
        all(0.0 <= float(x) <= 1.0 for x in sigmoid),
        f"实际 {sigmoid}",
    )

    top_n = len(CORPUS)
    result = reranker.rerank(QUERY, candidates, top_n=top_n, min_score=None)
    scores = [candidate.rerank_score for candidate in result]
    print(f"         重排结果：{[round(s, 3) for s in scores]}")
    check("返回条数不超过 top_n", len(result) <= top_n, f"实际 {len(result)}")
    check("rerank_score 已回填", all(score is not None for score in scores))
    check(
        "按 rerank_score 降序",
        scores == sorted(scores, reverse=True),
        f"实际 {scores}",
    )
    check(
        "相关项排在无关项之前",
        result[0].chunk.text in RELEVANT,
        f"首位是 {result[0].chunk.text!r}",
    )

    truncated = reranker.rerank(QUERY, candidates, top_n=1, min_score=None)
    check("top_n=1 只返回 1 条", len(truncated) == 1, f"实际 {len(truncated)}")

    # 阈值过滤：min_score 高于所有得分 → 空列表（上层据此拒答）
    highest = max(float(x) for x in raw)
    strict = reranker.rerank(QUERY, candidates, top_n=5, min_score=highest + 1.0)
    check("全部低于 min_score 时返回空列表", strict == [], f"实际 {len(strict)} 条")

    # 阈值恰好挡掉无关项：logits 为负的都不该留下
    filtered = reranker.rerank(QUERY, candidates, top_n=5, min_score=0.0)
    check(
        "min_score=0.0 挡掉负分（无关）项",
        all(float(candidate.rerank_score) >= 0.0 for candidate in filtered),
        f"实际 {[round(float(c.rerank_score), 3) for c in filtered]}",
    )
    check(
        "无关句被过滤后仍保留相关句",
        len(filtered) == len(RELEVANT),
        f"实际保留 {len(filtered)} 条，期望 {len(RELEVANT)}",
    )

    check("候选为空返回空列表", reranker.rerank(QUERY, [], top_n=5) == [])
    check("top_n<=0 返回空列表", reranker.rerank(QUERY, candidates, top_n=0) == [])


# ---------------------------------------------------------------------------
# 架构契约（纯静态，不加载任何权重）
# ---------------------------------------------------------------------------


def check_layering() -> None:
    """domain 层必须与框架无关——依赖方向只能单向指向 domain。

    这条约束原本写在 `tests/unit/test_placeholder.py`，那是骨架期留下的
    pytest 占位文件，没有任何入口会去跑它（requirements 里的 pytest 也是
    注释掉的）。约束本身不能丢：domain 一旦 import fastapi / langchain，
    端口就绑死在框架上，换框架等于重写领域层。

    用 AST 取 import 比字符串搜 "import fastapi" 可靠——后者对
    `from fastapi import Depends`、括号换行、`importlib` 动态导入都会漏。
    这里只做静态可见的检查，动态导入由代码评审兜底。
    """
    print("\n[0] 架构契约")

    import ast

    forbidden = ("fastapi", "starlette", "langchain", "langgraph", "pymilvus",
                 "sqlalchemy", "redis", "openai", "flagembedding", "pydantic_settings")
    domain_dir = PROJECT_ROOT / "app" / "domain"
    offenders: list[str] = []
    modules = sorted(domain_dir.glob("*.py"))
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                if root in forbidden:
                    offenders.append(f"{path.name}:{node.lineno} → {name}")

    check(
        f"domain 层不 import 框架（检查 {len(modules)} 个模块）",
        not offenders,
        "；".join(offenders),
    )


# ---------------------------------------------------------------------------


def main() -> int:
    settings = get_settings()
    print("=" * 64)
    print("模型自检（FlagEmbedding）")
    print(f"  向量权重：{settings.embedding_model_path}   device={settings.embedding_device}")
    print(f"  重排权重：{settings.rerank_model_path}      device={settings.rerank_device}")
    print("=" * 64)

    check_layering()

    print("\n加载向量模型 ...", flush=True)
    embedder = BgeEmbedder()
    check_embedding(embedder)

    if settings.rerank_enabled:
        print("\n加载重排模型 ...", flush=True)
        check_rerank(BgeReranker())
    else:
        print("\n[RERANK_ENABLED=false] 跳过重排检查")

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
