"""多路召回融合：加权 RRF（Reciprocal Rank Fusion）。

为什么用 RRF 而不是"把分数归一化后加权求和"：
    两路召回的分数**不可比**——余弦相似度落在 [-1,1] 且可比区间很窄
    （本机 BGE 实测相关约 0.66、无关约 0.23），BM25 分却是无界的
    （实测同一语料里 2.55 与 0.45 并存，换一份语料整体尺度又全变）。
    想把它们归一化后再加权，就得先决定"怎么归一化"，而这恰恰依赖语料，
    是个换数据集就失效的隐式假设。

    RRF 只吃**排名**，对分数的绝对尺度免疫：
        score(d) = Σ_path  w_path / (k + rank_path(d))
    这是 Cormack 等人 2009 年提出 RRF 时的原始动机，也是它成为
    混合检索默认融合方式的原因。

关于"加权 RRF"：
    标准 RRF 不带权重。本模块支持给每路乘一个权重（`w/(k+rank)`），
    好处是既能调"哪一路更重要"，又保留了对分数尺度的免疫。
    注意权重的**绝对大小无意义，只有比值有意义**：
        0.5/0.5 与 1.0/1.0 的排序结果完全相同，只是 fusion_score 差一倍。
    所以 fusion_score 只可用于**相对比较**，不要当概率或阈值用。

与 Milvus 服务端 `hybrid_search(RRFRanker)` 的关系：
    Milvus 3.0.2 也能在服务端做 RRF，但那样拿不到逐路的 vector_score /
    bm25_score，调参时是黑盒；且权重只能表达成另一套 ranker。
    本模块选择在应用侧融合：多花一次网络往返，换来每一路分数的可见性
    ——"检不出结果时能看清是哪一路没召回"，这个信息对调参远比省一次 RTT 值钱。
    服务端融合的实现在 `check_retrieval.py` 里有对照验证（两者排序一致）。

本模块依赖：
    只用标准库 + `app.domain.models`，**不 import pymilvus / langchain**，
    因此可以脱离数据库与模型做纯逻辑单测。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.domain.models import Chunk, RetrievedChunk

# RRF 的平滑常数默认值。
# 60 是原论文与业界通行取值：它决定了"排名靠前的优势有多大"。
#   k 越大 → 各排名之间的差距被压平，更依赖"两路都出现"这个信号；
#   k 越小 → 单路排名靠前的片段优势更大。
# 本项目量级（召回 8~50 条）下 60 偏平滑，属安全默认。
DEFAULT_RRF_K = 60


def _text_key(chunk: Chunk) -> str | None:
    """同一份内容在"跨文档副本"之间对齐用的键。

    为什么按 (文件名, 空白归一化后的正文) 而不是只按正文：
        两篇不同文档引用同一段公告、或同一模板的样板段，是**合法**的
        重复内容，合并会把一个真实来源抹掉。加上文件名后，只有
        "同名文件的完全相同正文"才会被合并——那基本只有一种成因：
        同一个文件被重复上传了。

    正文为空时返回 None（不参与去重）：空正文无法判断内容是否相同，
    按 id 各自保留更安全。
    """
    text = " ".join((chunk.text or "").split())
    if not text:
        return None
    return f"{chunk.filename or ''}\x00{text}"


@dataclass(slots=True)
class FusionResult:
    """融合结果 + 去重计数（计数进 trace，调参时要看得见）。"""

    items: list[RetrievedChunk] = field(default_factory=list)
    duplicates_removed: int = 0
    """因「同名文件 + 相同正文」被合并掉的条数。

    这个数长期不为 0 说明库里有重复上传的文档：它们会**挤占召回名额**，
    让真正不同的片段排不进来。暴露出来才能发现，而不是等用户抱怨
    "怎么同一个文件出现四次"。
    """
    near_duplicates_removed: int = 0
    """因「近重复」（字符三元组 Jaccard ≥ 阈值，且数字完全一致）被合并的条数。

    与 `duplicates_removed` 分开计数：两者成因不同——
    精确重复是"同一份文件传了多次"，近重复是"同一段内容被重新保存/重新切分
    （如另存过的 PDF）"。混成一个数，就分不清该治上传流程还是该治解析流程。
    """


@dataclass(slots=True)
class _Accumulator:
    """融合过程中的单条累加器。

    同一个 chunk id 在两路里都可能出现，需要把两路的分数与排名
    归并到一条上——这正是 RRF 捕捉"两路都认可"的机制。
    """

    chunk: Chunk
    vector_score: float | None = None
    bm25_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None

    @property
    def best_rank(self) -> int:
        """两路中最好（最小）的排名，未出现的路不参与比较。

        用于排序的次级键：融合分相同时，某一路排得更靠前的胜出。
        """
        ranks = [r for r in (self.vector_rank, self.bm25_rank) if r is not None]
        return min(ranks) if ranks else 0

    def rrf(self, k: int, vector_weight: float, bm25_weight: float) -> float:
        """加权 RRF 分：Σ w/(k+rank)。未出现的路贡献 0。"""
        score = 0.0
        if self.vector_rank is not None:
            score += vector_weight / (k + self.vector_rank)
        if self.bm25_rank is not None:
            score += bm25_weight / (k + self.bm25_rank)
        return score


def rrf_score(rank: int, k: int = DEFAULT_RRF_K, weight: float = 1.0) -> float:
    """单路的 RRF 贡献：w / (k + rank)。

    Args:
        rank: 1-based 排名
        k: 平滑常数
        weight: 该路权重

    Returns:
        该路对融合分的贡献。
    """
    return weight / (k + rank)


def fuse(
    vector_hits: list[RetrievedChunk],
    bm25_hits: list[RetrievedChunk],
    *,
    k: int = DEFAULT_RRF_K,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    top_k: int | None = None,
    dedupe: bool = True,
    near_dupe_ratio: float = 0.0,
    near_dupe_min_chars: int = 60,
) -> list[RetrievedChunk]:
    """把两路召回融合成一份按融合分降序的候选列表（去重后）。"""
    return fuse_detailed(
        vector_hits,
        bm25_hits,
        k=k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        top_k=top_k,
        dedupe=dedupe,
        near_dupe_ratio=near_dupe_ratio,
        near_dupe_min_chars=near_dupe_min_chars,
    ).items


def fuse_detailed(
    vector_hits: list[RetrievedChunk],
    bm25_hits: list[RetrievedChunk],
    *,
    k: int = DEFAULT_RRF_K,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    top_k: int | None = None,
    dedupe: bool = True,
    near_dupe_ratio: float = 0.0,
    near_dupe_min_chars: int = 60,
) -> FusionResult:
    """加权 RRF 融合 + **跨副本去重**，返回结果与去重计数。

    Args:
        vector_hits: 向量召回结果，**必须已按相似度降序**（本函数按位置取排名）
        bm25_hits: BM25 召回结果，**必须已按 BM25 分降序**
        k: RRF 平滑常数，须 >= 1
        vector_weight: 稠密路权重，须 >= 0
        bm25_weight: 稀疏路权重，须 >= 0
        top_k: 融合后保留条数；None 表示全保留
        dedupe: 是否合并「同名文件 + 相同正文」的副本（默认开）。
        near_dupe_ratio: 近重复判据（字符三元组 Jaccard），0 = 关闭。
            HybridRetriever 从 `HYBRID_NEAR_DUPE_RATIO` 传入；默认 0.95。
            **数字/型号、否定词、情态词必须完全一致**才算重复——「48 小时」与
            「24 小时」、「可以」与「不可以」、「应当」与「可以」相似度都很高，
            但它们不是重复，合并会删掉一条关键条款。
        near_dupe_min_chars: 参与近重复合并的最小正文长度（折叠空白后）。
            短文本不合并：两篇文档引用同一句公告是合法的，只有成段复制
            才值得当作副本。

    Returns:
        FusionResult：items 为最终列表，duplicates_removed 为被合并的条数。
        **输入为空时不报错，返回空结果。**

    Raises:
        ValueError: 参数越界，或某一路的结果未按其分数降序排列

    排序规则（确定性，便于测试）：
        ① fusion_score 降序；
        ② 两路中较好的排名（best_rank）升序——分数打平时，谁在任一路排得更前面谁靠前；
        ③ chunk.id 升序——纯粹为了消除不确定性（同分同排名时结果不该随字典顺序漂移）。

    为什么必须去重（实测故障，不是洁癖）：
        同一个 txt 被重复上传 4 次后，库里出现 4 个 doc_id、正文完全相同。
        向量路对这 4 条给出**逐位相同**的相似度（实测 0.18244785070419312 ×4），
        BM25 路也是同分，于是融合后有 4 条并列，把 top_k=8 里的名额占掉一半，
        真正不同的片段被挤到截断线之外。用户看到的是「引用里同一个文件出现四次」，
        而检索看起来"正常工作"——这正是最难查的一类召回异常。
        去重放在**截断之前**，名额才会真正释放出来。

        去重键见 `_text_key()`：只合并同名文件的相同正文，不误伤
        "两篇文档引用同一段公告"这种合法重复。
    """
    if k < 1:
        raise ValueError(f"RRF 的 k 必须 >= 1，当前 {k}")
    if vector_weight < 0 or bm25_weight < 0:
        raise ValueError(
            f"权重不能为负：vector_weight={vector_weight}, bm25_weight={bm25_weight}"
        )

    merged: dict[str, _Accumulator] = {}

    # 两路的处理完全对称，用同一段逻辑跑，避免"改了一路忘了另一路"
    for hits, weight, score_field, rank_field in (
        (vector_hits, vector_weight, "vector_score", "vector_rank"),
        (bm25_hits, bm25_weight, "bm25_score", "bm25_rank"),
    ):
        if weight == 0.0 or not hits:
            # 权重为 0 的路整条跳过：连排名都不记，省掉无意义的比较
            continue
        _assert_sorted(hits, score_field)
        for rank, hit in enumerate(hits, start=1):
            chunk = hit.chunk
            if not chunk.id:
                # 没有 id 就无法去重，也无法在两路之间对齐——直接拒绝，
                # 否则会出现"同一条片段在结果里出现两次"这种难查的错
                raise ValueError("检索结果里存在 chunk.id 为空的片段，无法做融合")
            acc = merged.get(chunk.id)
            if acc is None:
                acc = _Accumulator(chunk=chunk)
                merged[chunk.id] = acc
            elif acc.chunk.text == "" and chunk.text:
                # 两路返回的是同一条片段，正文理应一致；保留有正文的那份
                acc.chunk = chunk

            score = getattr(hit, score_field)
            # 同一路里出现重复 id：只认最好（最靠前）的那次排名
            if getattr(acc, rank_field) is None:
                setattr(acc, rank_field, rank)
                setattr(acc, score_field, score)

    results: list[_Accumulator] = list(merged.values())

    # 排序规则见 docstring；确定性排序对测试与"结果可复现"都必要
    results.sort(
        key=lambda acc: (
            -acc.rrf(k, vector_weight, bm25_weight),
            acc.best_rank,
            acc.chunk.id,
        )
    )

    duplicates = 0
    if dedupe:
        # 排序后再去重：保留下来的自然是同内容里分最高的那条，
        # 且不需要二次排序（顺序仍是单调递减的）。
        kept: list[_Accumulator] = []
        seen: set[str] = set()
        for acc in results:
            key = _text_key(acc.chunk)
            if key is not None and key in seen:
                duplicates += 1
                continue
            if key is not None:
                seen.add(key)
            kept.append(acc)
        results = kept

    near_removed = 0
    if near_dupe_ratio and near_dupe_ratio > 0 and len(results) > 1:
        # 近重复放在精确去重**之后**：先按字符串键把"逐字相同"的副本合并掉，
        # 剩下的才需要算相似度（O(n²) 的比较，先做廉价的那一步）。
        results, near_removed = _merge_near_duplicates(
            results, near_dupe_ratio, near_dupe_min_chars
        )

    fused = [
        RetrievedChunk(
            chunk=acc.chunk,
            vector_score=acc.vector_score,
            bm25_score=acc.bm25_score,
            fusion_score=acc.rrf(k, vector_weight, bm25_weight),
        )
        for acc in results
    ]
    if top_k is not None and top_k >= 0:
        fused = fused[:top_k]
    return FusionResult(
        items=fused,
        duplicates_removed=duplicates,
        near_duplicates_removed=near_removed,
    )


def _assert_sorted(hits: list[RetrievedChunk], score_field: str) -> None:
    """断言一路结果按其分数降序，否则排名无意义。

    这不是多余的防御：排名取的是**列表位置**，若上游忘了排序，
    RRF 会安静地算出完全错误的融合分——而且结果看上去"也像那么回事"，
    极难排查。宁可在入口把这种错误顶回去。
    """
    previous: float | None = None
    for index, hit in enumerate(hits):
        score = getattr(hit, score_field)
        if score is None:
            raise ValueError(
                f"第 {index + 1} 条结果的 {score_field} 为空——"
                "送进融合的每一路都必须由对应的检索路径产出（稠密路给 vector_score，"
                "稀疏路给 bm25_score）"
            )
        if previous is not None and score > previous + 1e-9:
            raise ValueError(
                f"{score_field} 未按降序排列（第 {index} 条 {previous} → "
                f"第 {index + 1} 条 {score}）；排名取自列表位置，"
                "顺序错了融合分就没有意义"
            )
        previous = score


def _merge_near_duplicates(
    results: list[_Accumulator], ratio: float, min_chars: int
) -> tuple[list[_Accumulator], int]:
    """合并近重复片段（保留分更高、也就是更靠前的那条）。

    判据（四条同时满足）：
        1. 折叠空白后的正文长度 ≥ min_chars；
        2. 字符三元组 Jaccard ≥ ratio；
        3. 数字/型号里的数字串**完全相同**（见 `_numeric_tokens`）；
        4. 否定词与情态词集合完全相同（见 `_negation_markers` / `_modality_markers`）。

    第 1 条保住"短公告被两篇文档引用"的合法性；第 3、4 条是安全阀：
    「客服须在 48 小时内响应」与「24 小时内响应」、「可以加急」与
    「不可以加急」、「应当提交」与「可以提交」相似度都很高，但它们
    **不是重复**；没有这几道闸门，去重会静默删掉一条真实条款
    （skill 明确警告过）。文件名不参与判断：另存过的 PDF 正文相同但
    名字不同，正是这里要合的场景。
    """
    kept: list[_Accumulator] = []
    seen: list[tuple[set[str], tuple[str, ...], frozenset[str], frozenset[str]]] = []
    removed = 0
    for acc in results:
        text = acc.chunk.text or ""
        if len("".join(text.split())) < max(0, int(min_chars)):
            kept.append(acc)
            continue
        signature = (
            _numeric_tokens(text),
            _negation_markers(text),
            _modality_markers(text),
        )
        grams = _ngrams(text)
        duplicate = False
        for prev_grams, prev_numbers, prev_negations, prev_modalities in seen:
            if signature != (prev_numbers, prev_negations, prev_modalities):
                continue
            if _jaccard(grams, prev_grams) >= ratio:
                duplicate = True
                break
        if duplicate:
            removed += 1
            continue
        kept.append(acc)
        seen.append((grams, *signature))
    return kept, removed


def _ngrams(text: str, size: int = 3) -> set[str]:
    """字符 n-gram 集合（先折叠空白）。中文不需要分词即可比较相似度。"""
    compact = "".join((text or "").split())
    if len(compact) <= size:
        return {compact} if compact else set()
    return {compact[i : i + size] for i in range(len(compact) - size + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _numeric_tokens(text: str) -> tuple[str, ...]:
    """抽取数字串（含小数、版本号、型号里的数字），排序后用于可比性判断。"""
    return tuple(sorted(re.findall(r"\d+(?:[.\-:/]\d+)*", text or "")))


# 否定含义的字。不要求分词：宁可多拦一次（不合并），也不要把
# 「不可以」和「可以」当成重复合并掉——那等于删掉一条约束。
_NEGATION_RE = re.compile(r"[不无未非禁勿]")

# 情态/强制力标记：应当 vs 可以、必须 vs 建议，是**不同条款**而不是重复。
# 长词在前，避免「必须」被拆成「须」。
_MODALITY_RE = re.compile(r"必须|不得|严禁|禁止|应当|应该|应|须|可以|可|建议|允许")


def _negation_markers(text: str) -> frozenset[str]:
    """文本里出现过的否定字集合；集合不同 → 不是重复。"""
    return frozenset(_NEGATION_RE.findall(text or ""))


def _modality_markers(text: str) -> frozenset[str]:
    """文本里出现过的情态词集合；集合不同 → 不是重复。

    例：「员工应当提交申请」与「员工可以提交申请」只差一个词，
    相似度极高，但强制力完全不同——合并会删掉"应当"这条约束。
    """
    return frozenset(_MODALITY_RE.findall(text or ""))
