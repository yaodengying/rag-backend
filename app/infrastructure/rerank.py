"""重排适配器：BGE 交叉编码器（cross-encoder），FlagEmbedding 推理。

实现目标端口：app.domain.ports.RerankerPort

为什么必须用交叉编码器：
    向量检索是「双塔」结构——query 与 passage 各自编码后算相似度，
    二者在编码阶段从不见面，精度天然受限。交叉编码器把 (query, passage)
    拼成一个序列一起过模型，能捕捉细粒度交互，精度显著更高，代价是慢，
    所以只用于对少量初检结果做精排。

实现要点：
    1. 只对向量检索召回的 top_k（约 8~50 条）做重排，不要对全库做。
    2. **分数尺度：统一用原始 logits**（本机 bge-reranker-base 实测，
       FlagEmbedding CPU）：
         - `FlagReranker.compute_score(pairs)` 的 `normalize` 默认就是 **False**，
           返回未经 Sigmoid 的原始 logits；
         - 传 `normalize=True` 才套 Sigmoid 得到 [0,1] 概率。
       本项目统一按**原始 logits** 处理，因此 min_score 按 logits 尺度设。
       **注意 min_score 不能想当然设 0**：相关对的 logits 并不保证为正——
       简历语料实测相关对 [-4.5, +4.3]、无关对 [-10.2, -6.2]（2026-10-28，
       "面试人姓名/年龄" 这类短查询的相关片段只有 -1 左右），用 0.0 会把
       全部相关片段滤掉，表现为「库里有资料却拒答」。当前取 -5.0（.env），
       换语料或换模型后必须用真实查询重新校准。
    3. CPU 上同样禁止 fp16（同 embedding.py 的坑）。
       FlagReranker 的 use_fp16 默认已是 False，本项目仍按设备显式传入并加 dtype 兜底。
    4. 与 LLM 不同，重排必须本地加载权重——Ollama 不提供重排接口
       （POST /api/rerank 返回 404），任何「用 Ollama 做重排」的方案都不成立。
    5. rerank_enabled=False 时本类不装配，由 container 返回 None，
       上层需按「无重排」分支处理（直接用融合分排序）。

    6. **失败必须收敛成 `RerankError`(502)**，与 `EmbeddingError` 对称。
       重排是"锦上添花"的一步，但它炸了不代表可以静默降级：
       没有重排就只剩融合分排序，而 `RERANK_MIN_SCORE` 那道闸门也随之失效，
       回答质量会**无声地**退化。带 code 的 502 让运维知道该看重排模型，
       而不是去看 Milvus 或 LLM。
       三类必须拦下：前向异常（OOM/权重损坏）、分数条数与候选数不一致
       （会让候选与分数张冠李戴）、分数含 NaN（NaN 与任何阈值比较都是 False，
       会把**全部**候选静默丢光，最终表现为"库里明明有资料却拒答"）。

    7. **过滤是「绝对下限 + 相对落差」双条件**（`RERANK_RELATIVE_MARGIN`）：
       绝对下限 `RERANK_MIN_SCORE` 回答"这一批够不够格回答"——全批低于它就
       返回空、上层拒答；相对落差 `max(下限, 最高分 - margin)` 回答"谁能跟
       最高分一起进上下文"——把明显陪跑的片段去掉，避免 top_n 名额被平庸项
       占满。只看绝对值的坏处是：换模型/换语料就得重新标一个数，而"差多少"
       这个口径能跟着本批分数自动平移。
       两条约束不能互相替代：只留相对值时，整批都很差仍会挑出"相对最好"的
       一条硬答；只留绝对下限时，一批平庸分（但都高过下限）会全量进上下文。
"""

from __future__ import annotations

import logging
import math
import os
import threading

from app.core.config import get_settings
from app.core.exceptions import AppError, ConfigurationError, RerankError
from app.domain.models import RetrievedChunk

logger = logging.getLogger(__name__)

# 单次打分的条数上限。重排只作用于初检结果（远小于此值），
# 这里主要是给 batch_size 一个固定值：FlagEmbedding 在 条数 < batch_size
# 时不打 tqdm 进度条，固定 64 可让常规调用保持日志干净。
_MAX_RERANK_BATCH = 64

# FlagAutoReranker 的编码器型类别名：模型名不在官方映射表时用它兜底
_ENCODER_ONLY = "encoder-only-base"


def _is_cpu(device: str) -> bool:
    """判断设备描述是否指向 CPU。"""
    return device.lower().startswith("cpu")


class BgeReranker:
    """BGE 交叉编码器重排器（FlagEmbedding）。

    实现 RerankerPort 协议。
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        """记录配置，不做实际加载（懒加载以避免拖慢启动）。

        Args:
            model_path: 本地权重目录；None 时取 settings.rerank_model_path
            device: "cpu" / "cuda" / "cuda:0"；None 时取 settings.rerank_device
            max_length: 拼接序列最大长度，超长截断（query 单独限长，见下）
        """
        settings = get_settings()
        self._model_path = model_path or settings.rerank_model_path
        self._device = (device or settings.rerank_device or "cpu").lower()
        self._max_length = int(max_length)
        self._min_score = float(settings.rerank_min_score)
        # 相对落差（0 = 关闭），见模块文档第 7 条
        self._relative_margin = float(settings.rerank_relative_margin)
        # CPU 必须 fp32（见模块文档第 3 条）
        self._use_fp16 = not _is_cpu(self._device)

        self._model = None
        # 模型加载可能被并发触发（多个问答请求），用锁保证只加载一次
        self._lock = threading.Lock()

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """按交叉编码器得分重排并截断。

        Args:
            query: 用户问题原文
            candidates: 初检结果（通常已按融合分降序）
            top_n: 返回条数上限
            min_score: logits 下限；低于此值视为与问题无关而丢弃。
                None 时取 settings.rerank_min_score

        Returns:
            重排后的结果，按 rerank_score 降序，rerank_score 已回填。
            全部低于阈值时返回空列表（上层据此触发拒答）。
        """
        if not candidates or top_n <= 0:
            return []

        threshold = self._min_score if min_score is None else float(min_score)
        model = self._load()

        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        scores = self._score(model, pairs)

        scored = [
            (candidate, float(score))
            for candidate, score in zip(candidates, scores)
        ]
        # NaN 与任何阈值比较都是 False → 会把全部候选静默丢光，
        # 表现为"库里明明有资料却拒答"。在离开适配器之前就拦下。
        if any(not math.isfinite(score) for _, score in scored):
            raise RerankError(
                "重排输出包含 NaN/Inf——NaN 与阈值的比较恒为 False，"
                "会把全部候选静默过滤掉（表现为「有资料却拒答」）；"
                "请检查权重完整性与输入文本是否含异常字符"
            )
        # 双条件门槛（详见模块文档第 7 条）：
        #   绝对下限 threshold 决定"够不够格回答"（全批不够格 → 空 → 拒答）；
        #   相对落差 margin 决定"够不够格跟最高分并列进上下文"（去掉陪跑项）。
        # margin<=0 时退化为纯绝对阈值，行为与旧版完全一致。
        effective = threshold
        if self._relative_margin > 0:
            top = max(score for _, score in scored)
            effective = max(threshold, top - self._relative_margin)
            logger.debug(
                "重排门槛：绝对 %.3f / 最高 %.3f / 落差 %.3f → 生效 %.3f",
                threshold,
                top,
                self._relative_margin,
                effective,
            )
        # 先按分数过滤无关项，再截断——否则低分项会占掉 top_n 名额
        kept = [item for item in scored if item[1] >= effective]
        kept.sort(key=lambda item: item[1], reverse=True)

        result: list[RetrievedChunk] = []
        for candidate, score in kept[:top_n]:
            candidate.rerank_score = score
            result.append(candidate)
        return result

    def _score(self, model, pairs: list[tuple[str, str]]) -> list[float]:
        """调用模型打分，把底层异常与形状问题收敛成 `RerankError`。

        与 `embedding._forward()` 同一套规则：**`AppError` 原样放行**
        （`ConfigurationError` 是"该改 .env"，包成 502 会让运维去查显存）。
        """
        try:
            scores = model.compute_score(
                pairs,
                batch_size=_MAX_RERANK_BATCH,
                max_length=self._max_length,
                # 关键：不套 Sigmoid，拿原始 logits（见模块文档第 2 条）
                normalize=False,
            )
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001 - 见模块文档第 6 条
            hint = ""
            if "out of memory" in str(exc).lower() or "cuda" in str(exc).lower():
                hint = "；显存不足的可能性较大，可调小 RERANK_TOP_N / RETRIEVE_TOP_K 或改用 CPU"
            raise RerankError(
                f"重排失败（共 {len(pairs)} 对候选，max_length={self._max_length}）："
                f"{type(exc).__name__}: {exc}{hint}",
                detail={
                    "pairs": len(pairs),
                    "max_length": self._max_length,
                    "exception": type(exc).__name__,
                },
            ) from exc

        values = list(scores)
        if len(values) != len(pairs):
            # 条数对不上会让候选与分数张冠李戴：检索"能跑"，但引用的是别人的原文
            raise RerankError(
                f"重排输出条数与候选数不一致：输入 {len(pairs)} 对，"
                f"输出 {len(values)} 个分数"
            )
        return values

    # ---------- 内部 ----------

    def _load(self):
        """懒加载交叉编码器，CPU 场景强制关闭 fp16。"""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:  # 双检锁
                return self._model

            if not self._model_path:
                raise ConfigurationError(
                    "RERANK_MODEL_PATH 未配置；请在 .env 中指向本地 BGE 重排权重目录，"
                    "或设 RERANK_ENABLED=false 关闭重排"
                )
            from pathlib import Path

            if not Path(self._model_path).is_dir():
                raise ConfigurationError(
                    f"RERANK_MODEL_PATH 不是有效目录：{self._model_path}"
                )

            # 权重已在本机，禁止联网（国内访问 huggingface.co 会长时间卡住）
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

            logger.info(
                "加载重排模型：%s（device=%s fp16=%s）",
                self._model_path,
                self._device,
                self._use_fp16,
            )
            model = self._load_flag_reranker()
            self._guard_dtype(model)
            self._model = model
            logger.info("重排模型就绪：max_length=%s", self._max_length)
            return self._model

    def _load_flag_reranker(self):
        """构造 FlagReranker；模型名不在官方映射表时退回编码器型。"""
        options = {
            "devices": self._device,
            "use_fp16": self._use_fp16,
            "max_length": self._max_length,
        }
        from FlagEmbedding import FlagAutoReranker

        try:
            return FlagAutoReranker.from_finetuned(self._model_path, **options)
        except ValueError as exc:
            # FlagAutoReranker 按模型名查官方映射表来区分「编码器型」与
            # 「LLM 型（bge-reranker-v2-gemma 等）」。自定义目录名查不到时，
            # 按最常见的编码器型 BGE reranker 加载（bge-reranker-base/large）。
            logger.warning(
                "模型目录名不在 FlagEmbedding 官方映射表内，按编码器型 BGE 重排器加载：%s（%s）",
                self._model_path,
                type(exc).__name__,
            )
            return FlagAutoReranker.from_finetuned(
                self._model_path, model_class=_ENCODER_ONLY, **options
            )

    def _guard_dtype(self, model) -> None:
        """CPU 上禁止半精度权重（同 embedding.py 的坑）。"""
        try:
            import torch

            parameter = next(model.model.parameters())
        except Exception:  # noqa: BLE001 - 拿不到参数就不拦，不影响主流程
            return
        if _is_cpu(self._device) and parameter.dtype not in (
            torch.float32,
            torch.float64,
        ):
            logger.warning(
                "检测到 CPU 上的非 float32 重排权重（%s），已强制转回 float32",
                parameter.dtype,
            )
            model.model.float()
