"""混合检索：稠密向量召回 + BM25 关键词召回 → RRF 融合。

放在 `rag/` 而不是 `services/` 的理由：
    这条链路有两个消费者——对外的 `RetrievalService`（只检索不生成，用于调试）
    和对内的 LangGraph `retrieve` 节点。逻辑写在这里，两边都调它，
    才不会出现"接口查到的和问答用的不是同一套检索"这种最难查的偏差。

只做「召回 + 融合」，不做重排：
    重排是交叉编码器的活（`infrastructure/rerank.py`），
    属于「精度」而非「召回」阶段。把两者分开，才能单独回答
    "是没召回到，还是召回了但被重排丢了"。

    为什么两路都要：
        向量检索擅长语义相近但用词不同的情况，对**专有名词、编号、型号**却很弱
        ——这类词在训练语料里没见过，"BGE-1024" 与 "BGE-1024" 的向量未必最近；
        BM25 恰好相反：词表命中就得分，对语义改写毫无办法。
        两者互补，这是混合检索的标准动机。

    为什么混合模式下**不给稠密路设相似度阈值**（实测结论，不是偏好）：
        RRF 只吃排名，所以两路的召回深度必须等价；给其中一路加硬阈值
        等于人为把这路的排名截短。这个错误还特别隐蔽——
        BM25 路照常出结果，从返回内容上看不出"向量路已经一条都没召回"，
        只有把逐路明细打出来才会发现。
        实测（CHUNK_SIZE=300，bge-large-zh-v1.5，MIN_SIMILARITY=0.35）：

            查询                    片段0     片段1     过 0.35？
            多久需要做一次检查？     0.2804    0.2936    ✗ 全被滤掉
            巡检                    0.3081    0.1043    ✗
            XG-8800                0.3135    0.0007    ✗
            设备的固件怎么升级        0.5721    0.2554    ✓

        中文短查询对长块，余弦相似度本就落在 0.28~0.35，0.35 的门槛是致命的。
        因此混合模式下稠密路默认**按排名取 top_k、不做阈值过滤**，
        精度交给交叉编码器重排（它有自己的 logits 阈值）。
        纯向量模式下没有重排这个下游闸门，阈值仍是必需的——
        同一个 `HybridRetriever` 按模式取不同的默认值，见 `__init__`。

    动态 k（`RETRIEVE_K_MODE=adaptive`）：
        默认 fixed —— 每路固定取调用方要求的 top_k，融合后同样截这么多条，
        与历史行为逐位一致。adaptive 时先按上限取（**不放大** top_k），再按
        每路自己的分数找断层，截到 `[RETRIEVE_K_MIN, 上限]`，算法见
        `rag/topk.py`。为什么不按融合分截：融合分只由名次决定，任何语料都会
        切出同一条曲线，那个"动态"只是换了个写法的固定值。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.exceptions import AppError, RetrievalError
from app.domain.models import RetrievedChunk
from app.domain.ports import EmbedderPort, VectorStorePort
from app.rag.fusion import fuse_detailed
from app.rag.topk import TOP_K_HARD_LIMIT, adaptive_k, resolve_k

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RecallTrace:
    """一次召回的全过程快照，用于调参与排查。

    存在的意义：检索不出结果时，第一个要回答的问题是
    「向量路召回了什么？BM25 路召回了什么？融合后排第几？」
    只给最终结果的话，这三个问题都得靠猜。
    """

    query: str
    hybrid_enabled: bool
    dense_min_similarity: float = 0.0
    """稠密路生效的相似度下限。0 表示不设限（混合模式的默认）。
    **放在 trace 里是刻意的**：这个值设错时检索不会报错、只会变差，
    必须能在一行日志里看到它到底是多少。"""
    vector_hits: list[RetrievedChunk] = field(default_factory=list)
    bm25_hits: list[RetrievedChunk] = field(default_factory=list)
    fused: list[RetrievedChunk] = field(default_factory=list)
    duplicates_removed: int = 0
    """融合时被「同名文件 + 相同正文」合并掉的条数。

    长期不为 0 说明库里有重复上传的副本——它们会挤占召回名额，
    让真正不同的片段排不进来（详见 `rag/fusion.py` 的说明）。
    """
    near_duplicates_removed: int = 0
    """融合时被「近重复」合并掉的条数（另存过的 PDF、重新解析出的同一段文本）。

    与 `duplicates_removed` 分开：一个治上传流程，一个治解析/另存流程。
    """

    k_mode: str = "fixed"
    """本次生效的深度模式（fixed / adaptive）。

    放进 trace 是刻意的：adaptive 打开后"这次为什么只返回 3 条"必须能在一行
    日志里读到，否则只能靠猜是阈值、去重还是动态截断干的。"""
    k_selected: int = 0
    """融合后实际保留的条数。adaptive 下这是拐点算出来的值，
    fixed 下等于调用方要求的深度（不是实际命中数——后者看 `len(fused)`）。"""
    k_dense: int = 0
    k_bm25: int = 0
    """两路各自生效的深度。fixed 模式下两者都等于 k_selected。"""

    def describe(self) -> str:
        """一行摘要，适合直接打日志。"""
        dedup = f" 去重={self.duplicates_removed}" if self.duplicates_removed else ""
        if self.near_duplicates_removed:
            dedup += f" 近重复={self.near_duplicates_removed}"
        # k 必须一眼可读：adaptive 下"这次为什么只留了 3 条"是排查的第一问，
        # 而 fixed 模式下这个值等于调用方要求，写出来能直接看出请求有没有生效。
        k_desc = f"k={self.k_mode}:{self.k_selected}"
        if self.k_mode == "adaptive":
            k_desc += f"(dense={self.k_dense} bm25={self.k_bm25})"
        return (
            f"query={self.query!r} hybrid={self.hybrid_enabled} "
            f"稠密下限={self.dense_min_similarity} {k_desc} "
            f"向量召回={len(self.vector_hits)} BM25召回={len(self.bm25_hits)} "
            f"融合后={len(self.fused)}{dedup}"
        )


class HybridRetriever:
    """两路召回 + RRF 融合。

    依赖只有 `EmbedderPort` 与 `VectorStorePort`——不 import pymilvus，
    测试时用 Fake 即可跑通全部融合逻辑（见 `scripts/check_retrieval.py`）。
    """

    def __init__(
        self,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        *,
        top_k: int | None = None,
        k_mode: str | None = None,
        k_min: int | None = None,
        k_max: int | None = None,
        k_drop_ratio: float | None = None,
        hybrid_enabled: bool | None = None,
        rrf_k: int | None = None,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
        min_similarity: float | None = None,
        bm25_min_score: float | None = None,
        near_dupe_ratio: float | None = None,
        near_dupe_min_chars: int | None = None,
    ) -> None:
        """注入端口与检索参数。

        Args:
            embedder: 向量化器（编码查询）
            vector_store: 向量库（同时提供稠密与 BM25 两路检索）
            top_k: 每路召回深度，None 时取 settings.retrieve_top_k。**这是上限**：
                调用方仍可在单次 `recall(top_k=...)` 里改小；adaptive 模式只会在
                这个值之内收缩，不会把它放大。
            k_mode: 深度模式 `fixed` / `adaptive`，None 时取
                `settings.retrieve_k_mode`（默认 fixed，与历史行为一致）。
            k_min: adaptive 的下限，None 时取 `settings.retrieve_k_min`
            k_max: adaptive 的上限（同时是过采样条数），None 时取
                `settings.retrieve_k_max`。实际取 `min(top_k, k_max)`。
            k_drop_ratio: 断层判据，None 时取 `settings.retrieve_k_drop_ratio`
            hybrid_enabled: 是否启用 BM25 路，None 时取 settings.hybrid_enabled
            rrf_k: RRF 平滑常数，None 时取 settings.hybrid_rrf_k
            vector_weight: 稠密路权重，None 时取 settings.hybrid_vector_weight
            bm25_weight: 稀疏路权重，None 时取 settings.hybrid_bm25_weight
            min_similarity: 稠密路的相似度下限。**None 时按模式取不同的默认值**：
                混合模式 → `settings.hybrid_dense_min_similarity`（默认 0.0，不设限）
                纯向量模式 → `settings.min_similarity`
                理由见 `Settings.min_similarity` 的注释（附实测分数表）与
                模块文档「为什么混合模式下不给稠密路设阈值」。
            bm25_min_score: 稀疏路的 BM25 分下限，None 时取
                `settings.hybrid_bm25_min_score`（默认 0.0 = 不过滤）。
                BM25 分无界且随语料漂移，所以**默认不设绝对阈值**；
                需要压掉"只共享一个「的」"的噪声时，用 `tune_retrieval.py`
                打印的真实分数来决定，别凭感觉写。
            near_dupe_ratio: 近重复合并阈值（字符三元组 Jaccard），None 时取
                `settings.hybrid_near_dupe_ratio`（默认 0.95）。0 = 关闭。
                数字/型号不同、否定词不同的片段**不会**被合并
                （见 `fusion._numeric_tokens` / `_negation_markers`）。
            near_dupe_min_chars: 参与近重复合并的最小正文长度，None 时取
                `settings.hybrid_near_dupe_min_chars`（默认 60）。短文本不合并，
                保住"两篇文档引用同一句公告"的合法性。
        """
        settings = get_settings()
        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = int(settings.retrieve_top_k if top_k is None else top_k)
        self._k_mode = str(
            settings.retrieve_k_mode if k_mode is None else k_mode
        ).strip().lower()
        if self._k_mode not in {"fixed", "adaptive"}:
            # 拼成 "adaptve" 时若不拦，会静默走 fixed ——
            # 表现是"这个开关明明开了却什么都没发生"。
            raise ValueError(
                f"k_mode 只能是 fixed / adaptive，当前 {self._k_mode!r}"
            )
        self._k_min = int(settings.retrieve_k_min if k_min is None else k_min)
        self._k_max = int(settings.retrieve_k_max if k_max is None else k_max)
        self._k_drop_ratio = float(
            settings.retrieve_k_drop_ratio if k_drop_ratio is None else k_drop_ratio
        )
        if self._k_min < 1:
            raise ValueError(f"k_min 必须 >= 1，当前 {self._k_min}")
        if self._k_max < self._k_min:
            # 写反时 adaptive 每次都会切到 0 条，表现为"突然什么都搜不到"，
            # 且不会报错——只能在这里拦。
            raise ValueError(
                f"k_max({self._k_max}) 不能小于 k_min({self._k_min})"
            )
        if not 0.0 <= self._k_drop_ratio <= 1.0:
            raise ValueError(
                f"k_drop_ratio 应落在 [0,1]，当前 {self._k_drop_ratio}"
            )
        self._hybrid_enabled = (
            bool(settings.hybrid_enabled) if hybrid_enabled is None else bool(hybrid_enabled)
        )
        self._rrf_k = int(settings.hybrid_rrf_k if rrf_k is None else rrf_k)
        self._vector_weight = float(
            settings.hybrid_vector_weight if vector_weight is None else vector_weight
        )
        self._bm25_weight = float(
            settings.hybrid_bm25_weight if bm25_weight is None else bm25_weight
        )
        if min_similarity is not None:
            self._min_similarity = float(min_similarity)
        elif self._hybrid_enabled:
            self._min_similarity = float(settings.hybrid_dense_min_similarity)
        else:
            self._min_similarity = float(settings.min_similarity)
        self._bm25_min_score = float(
            settings.hybrid_bm25_min_score if bm25_min_score is None else bm25_min_score
        )
        self._near_dupe_ratio = float(
            settings.hybrid_near_dupe_ratio
            if near_dupe_ratio is None
            else near_dupe_ratio
        )
        if not 0.0 <= self._near_dupe_ratio <= 1.0:
            raise ValueError(
                f"near_dupe_ratio 应落在 [0,1]，当前 {self._near_dupe_ratio}"
            )
        self._near_dupe_min_chars = int(
            settings.hybrid_near_dupe_min_chars
            if near_dupe_min_chars is None
            else near_dupe_min_chars
        )
        if self._near_dupe_min_chars < 0:
            raise ValueError(
                f"near_dupe_min_chars 不能为负，当前 {self._near_dupe_min_chars}"
            )

        if self._hybrid_enabled and self._vector_weight + self._bm25_weight <= 0:
            # 两路权重都是 0 时融合分会全为 0，排序退化成按 id 排——结果看似正常
            # 但完全没有相关性，是最难发现的一类配置错误。这里直接拦下。
            raise ValueError(
                "HYBRID_VECTOR_WEIGHT 与 HYBRID_BM25_WEIGHT 不能同时为 0，"
                "否则融合分恒为 0、排序失去意义"
            )

    # ---------- 只读属性（供日志与自检读取）----------

    @property
    def top_k(self) -> int:
        """每路召回深度的**默认上限**（单次调用可用 top_k 参数改小）。"""
        return self._top_k

    @property
    def k_mode(self) -> str:
        """当前生效的深度模式（fixed / adaptive）。

        暴露出来是为了让日志与自检能直接回答"动态 k 到底开没开"——
        这个开关配错时的症状是"结果条数不对"，而不是报错。
        """
        return self._k_mode

    @property
    def hybrid_enabled(self) -> bool:
        """BM25 路是否启用。"""
        return self._hybrid_enabled

    @property
    def dense_min_similarity(self) -> float:
        """稠密路当前生效的相似度下限（0 = 不设限）。

        暴露出来是为了让日志与自检能直接看到"这一路到底有没有被卡"——
        这个值一旦设错，症状是"检索变差了"，而不是"报错了"。
        """
        return self._min_similarity

    @property
    def bm25_min_score(self) -> float:
        """稀疏路当前生效的 BM25 分下限（0 = 不过滤）。"""
        return self._bm25_min_score

    # ---------- 主入口 ----------

    def retrieve(
        self,
        kb_id: str,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """召回并融合，返回按融合分降序的候选片段。

        Args:
            kb_id: 知识库 ID
            query: 查询原文
            top_k: 覆盖每路召回深度与本方法的返回条数；None 用构造时的值。
                adaptive 模式下它是**上限**，不保证一定返回这么多条
                （断层可能切得更短，见 `_adaptive_keep`）。

        Returns:
            融合后的候选列表；查询为空或无命中时返回空列表。
        """
        return self.recall(kb_id, query, top_k=top_k).fused

    def recall(
        self,
        kb_id: str,
        query: str,
        top_k: int | None = None,
    ) -> RecallTrace:
        """同 `retrieve()`，但把两路原始结果一并返回，便于调参。

        `top_k` 是**天花板**：None 用构造期配置；给了就按它来（再受服务端
        硬上限约束）。fixed 模式下它同时也是融合后的保留条数；adaptive 模式
        下先取 `min(top_k, K_MAX)` 条，再按每路分数的断层收缩。
        """
        text = (query or "").strip()
        depth = resolve_k(top_k, default=self._top_k, hard_limit=TOP_K_HARD_LIMIT)
        trace = RecallTrace(
            query=text,
            hybrid_enabled=self._hybrid_enabled,
            dense_min_similarity=self._min_similarity,
            k_mode=self._k_mode,
        )

        if not text or depth <= 0:
            logger.debug("检索跳过：query=%r depth=%s", text, depth)
            return trace

        # adaptive 只**收缩**深度，不放大：取 min(depth, K_MAX)。
        # fixed 模式下 fetch == depth，与历史行为逐位一致。
        fetch = min(depth, self._k_max) if self._k_mode == "adaptive" else depth

        if not self._hybrid_enabled:
            trace.vector_hits = self._dense(kb_id, text, fetch)
        else:
            # 权重为 0 的路整条跳过：省一次网络往返，也让日志如实反映实际用了哪条路
            if self._vector_weight > 0:
                trace.vector_hits = self._dense(kb_id, text, fetch)
            if self._bm25_weight > 0:
                trace.bm25_hits = self._sparse(kb_id, text, fetch)

        keep = self._adaptive_keep(trace, upper=fetch)
        trace.k_selected = keep
        trace.fused = self._fuse(trace, keep)
        logger.info("检索完成：%s", trace.describe())
        return trace

    def _adaptive_keep(self, trace: RecallTrace, *, upper: int) -> int:
        """按断层收缩两路深度，返回融合后要保留的条数。

        fixed 模式：两路深度与融合保留数都等于 `upper`（调用方要求的深度），
        与改造前的行为逐位一致。
        adaptive：分别对两路算断层，融合后保留 `max(k_dense, k_bm25)`。
        为什么取 max 而不是两路之和：fixed 模式下两路同深度、融合后截同值，
        取 max 正好复现它；取和会让 adaptive 反而比 fixed 留得更多，与"动态
        收缩"的初衷相反。
        """
        if self._k_mode != "adaptive":
            trace.k_dense = upper
            trace.k_bm25 = upper
            return upper

        trace.k_dense = adaptive_k(
            [_native_score(item, "vector_score") for item in trace.vector_hits],
            k_min=self._k_min,
            k_max=upper,
            drop_ratio=self._k_drop_ratio,
        )
        trace.k_bm25 = adaptive_k(
            [_native_score(item, "bm25_score") for item in trace.bm25_hits],
            k_min=self._k_min,
            k_max=upper,
            drop_ratio=self._k_drop_ratio,
        )
        # 截断放在融合之前：融合只吃排名，先按断层裁掉"明显不相关"的尾部，
        # 它们本来也抢不进 RRF 前列，却会占掉去重与截断的名额。
        trace.vector_hits = trace.vector_hits[: trace.k_dense]
        trace.bm25_hits = trace.bm25_hits[: trace.k_bm25]
        return max(trace.k_dense, trace.k_bm25)

    def _fuse(self, trace: RecallTrace, depth: int) -> list[RetrievedChunk]:
        """融合两路结果，并把"上游给了乱序结果"这种故障翻译成 502。

        `fuse_detailed` 用 ValueError 表达"这路结果没按分数降序"——那是
        **上游违约**（Milvus 返回了未排序结果），不是调用方传错参数。
        裸 ValueError 会冒到兜底处理器变成 500「服务内部错误」，
        运维看不出该去查哪里；包成 RetrievalError 才对得上它的语义。
        """
        try:
            result = fuse_detailed(
                trace.vector_hits,
                trace.bm25_hits,
                k=self._rrf_k,
                vector_weight=self._vector_weight,
                bm25_weight=self._bm25_weight,
                top_k=depth,
                near_dupe_ratio=self._near_dupe_ratio,
                near_dupe_min_chars=self._near_dupe_min_chars,
            )
        except ValueError as exc:
            raise RetrievalError(f"混合检索融合失败：{exc}") from exc

        trace.duplicates_removed = result.duplicates_removed
        if result.duplicates_removed:
            logger.info(
                "融合时合并了 %s 条重复副本（同名文件 + 相同正文）——"
                "库中可能存在重复上传；去重发生在截断之前，名额已释放给其它片段",
                result.duplicates_removed,
            )
        trace.near_duplicates_removed = result.near_duplicates_removed
        if result.near_duplicates_removed:
            logger.info(
                "融合时合并了 %s 条近重复片段（相似度 ≥ %.2f，数字/型号一致）——"
                "常见成因是同一份文档被另存/重新解析后以不同文件名入库",
                result.near_duplicates_removed,
                self._near_dupe_ratio,
            )
        return result.items

    # ---------- 两路召回 ----------

    def _dense(self, kb_id: str, query: str, top_k: int) -> list[RetrievedChunk]:
        """稠密路：编码查询后做余弦检索。

        阈值由构造期按模式定好（见模块文档）。混合模式下通常为 0.0，
        即"按排名取 top_k、不提前丢"——精度交给下游重排。
        这里**不做**任何默认值兜底：阈值一旦设错，症状是"检索变差了"
        而不是报错，所以它必须是构造期显式确定、可被日志观测的值。
        """
        embedding = self._embedder.embed_query(query)
        return self._vector_store.search(
            kb_id, embedding, top_k=top_k, min_score=self._min_similarity
        )

    def _sparse(self, kb_id: str, query: str, top_k: int) -> list[RetrievedChunk]:
        """稀疏路：BM25 全文检索。

        失败时**降级为"这一路没有结果"**而不是让整个检索失败：
        稠密路照常可用，用户依然能拿到答案，只是少了关键词召回。
        这类失败通常是底层不支持全文检索或 schema 未重建，
        启动期的 `ensure_collection()` 已经把关过，运行期再出问题属异常，
        所以记 exception 级别日志交给运维，不打断请求。
        """
        try:
            return self._vector_store.search_bm25(
                kb_id,
                query,
                top_k=top_k,
                # 默认 0.0 = 不过滤（BM25 分无界、随语料漂移，写死阈值必失效）。
                # 0 显式转成 None 而不是把 0 传下去：BM25 分永远 >= 0，
                # 传 0 等于"过滤负分"，看着像设了阈值其实什么都没做——
                # 这种"以为设了"的错觉比不设更麻烦。
                # 需要压噪声时由 HYBRID_BM25_MIN_SCORE 显式给出，取值靠
                # scripts/tune_retrieval.py 打印的真实分布，不靠感觉。
                min_score=self._bm25_min_score if self._bm25_min_score > 0 else None,
            )
        except AppError:
            logger.exception(
                "BM25 检索失败，本次降级为纯向量检索（kb_id=%s）——"
                "若持续出现，检查 collection 是否按带 BM25 函数的 schema 重建",
                kb_id,
            )
            return []


def _native_score(item: RetrievedChunk, field: str) -> float:
    """取某一路自己的原始分数（自适应只看这个，不看融合分）。

    缺失按 0 处理：正常路径不会缺（`search()` 填 vector_score，
    `search_bm25()` 填 bm25_score），真缺了也只影响本次截断，不该让一次
    检索因为这个辅助函数而失败。
    """
    value = getattr(item, field)
    return float(value) if value is not None else 0.0
