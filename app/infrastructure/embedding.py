"""向量化适配器：本地 BGE 权重，FlagEmbedding 推理。

实现目标端口：app.domain.ports.EmbedderPort

为什么用 FlagEmbedding 而不是 sentence-transformers：
    BGE 系列是 BAAI 自家模型，FlagEmbedding 是其官方推理库。它直接基于
    transformers 的 AutoModel/AutoTokenizer，少一层抽象：
      - 检索指令由 `query_instruction_for_retrieval` 参数声明，
        encode_queries / encode_corpus 自动区分加不加前缀（正是我们要的语义）；
      - 换模型（bge-m3 / bge 系列微调）只需改路径，由 FlagAutoModel
        按目录名自动选对应的池化方式与指令模板；
      - 不再需要 sentence-transformers 这一层，依赖更薄。
    注意 LangChain 侧的 `HuggingFaceBgeEmbeddings` / `HuggingFaceEmbeddings`
    **用不了**：两者在 `__init__` 里直接 `import sentence_transformers`，
    而本项目已按需求卸载它。它们也已被官方标记 deprecated（removed in 1.0）。

向量维度的唯一真相来源是**权重**，不是配置：
    维度取自权重目录 ``config.json`` 的 ``hidden_size``（BERT 系约定），
    构造时读取，毫秒级、不加载 torch、不占显存。
    ``MILVUS_DIMENSION`` 退化为**声明值**，必须与权重一致——不一致就
    在启动时报错，而不是等第一次上传才暴露。
    这样不会出现「collection 按旧维度建好、向量按新维度写入」这种
    只能在真实上传时才炸、且报错信息与根因相距很远的局面。
    读不到 config.json 时降级为声明值并告警，最终仍由 ``_load()``
    的实际编码结果兜底核对。

关键约束（踩过的坑，务必遵守）：

    1. query 与 passage 必须分开编码。
       BGE 中文模型要求检索指令（如「为这个句子生成表示以用于检索相关文章：」）
       只加在 **query** 前，文档块不加。混用会让两者落到不同语义空间，
       相似度整体偏移。本类用 embed_query / embed_documents 两个方法强制区分，
       底层分别走 encode_queries / encode_corpus。

    2. bge + 余弦检索必须 L2 归一化。
       否则内积不等于余弦相似度，Milvus 的 COSINE 度量会失真。
       本项目在 from_finetuned(normalize_embeddings=True) 处强制打开。

    3. CPU 上绝不能开 fp16。
       FlagEmbedding 的 use_fp16 默认是 **True**（为 GPU 优化），
       在 CPU 上会抛 NotImplementedError: "LayerNormKernelImpl" not implemented
       for 'Half'。本项目按设备自动决定：CPU → False，CUDA → True；
       并在加载后**检查权重 dtype** 兜底——比"指望调用方记得改"可靠。

    4. 权重已在本机时必须设 HF_HUB_OFFLINE=1，
       否则 transformers 会尝试连 huggingface.co，国内会长时间卡住。

    5. 模型是重量级对象，必须进程内单例复用（由 container 保证）。

    6. **空列表不能直接交给 FlagEmbedding**：
       ``encode_corpus([])`` 会抛 ``AttributeError: 'list' object has no
       attribute 'keys'``——它内部按 dict 取字段。必须在适配器里短路。

    7. **空白文本必须拒绝**：
       BGE 对空串/纯空白不报错，会老老实实返回一条 L2=1 的
       "纯 padding" 向量——形状合法、语义为空。这种向量一旦进索引，
       会对任意查询给出一段不高不低的相似度，属于**静默污染**。
       宁可让入库失败，也不要往索引里塞无意义向量。

分批与进度：
    一次送进模型的条数由 ``settings.embedding_batch_size`` 决定
    （上限 ``_MAX_ENCODE_BATCH``），**由本适配器自己分批**，调用方
    直接传整份列表即可，不必也不该再持有一个独立的分组常量
    （两处各写一个 64 只会随时间互相漂移）。
    条数较多时按累计量打进度日志：CPU 上编码上万块要几分钟，
    日志里不该是一片空白。

维度自检（本机实测 H:\\models\\bge-large-zh-v1.5，CPU，FlagEmbedding 1.4.2）：
    config.json 声明 hidden_size=1024，与 MILVUS_DIMENSION 一致；
    输出 1024 维；normalize_embeddings=True 后 L2 范数 = 1.0；
    「哪个是向量数据库？」对 Milvus 段落 0.591 / FAISS 段落 0.463 / 食堂段落 0.032
    （排序正确）；两次编码结果逐位一致（可复现）。
更换模型时改 EMBEDDING_MODEL_PATH 即可，维度会自动跟随；
若新维度与 MILVUS_DIMENSION 不符，启动时会明确报错。

实测注意（别拿它当 bug 修）：
    **批形状不同，结果会有 float32 末位差异**。同一批 67 条按 batch=32 编码，
    与逐条编码相比最大绝对差 1.19e-7（= 2^-23，正好是 float32 在 1.0 处的 eps），
    余弦相似度 0.99999985；而**同样批形状**重跑是逐位一致的。
    原因是矩阵乘的归约顺序随 batch 维度变化。所以：
      - "可复现"只在固定批形状下成立，别拿不同批大小的向量做逐位比较；
      - 任何 golden 断言都要留容差（见 scripts/check_embedding.py 的用例 4）。
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import AppError, ConfigurationError, EmbeddingError, ValidationError

logger = logging.getLogger(__name__)

# 单次编码的条数上限（与 settings.embedding_batch_size 取小值）。
# 存在的意义是兜住"把 batch 配成 10000"这种会直接 OOM 的写法。
_MAX_ENCODE_BATCH = 64

# 每累计这么多条打一行进度日志（条数本身很少时不打，避免刷屏）
_PROGRESS_EVERY = 512

# FlagAutoModel 的编码器型类别名：目录名不在官方映射表时用它兜底
_ENCODER_ONLY = "encoder-only-base"

# config.json 里表示隐藏层维度的键：BERT 系用 hidden_size，
# 部分架构（如 T5/部分自定义模型）用 d_model
_DIMENSION_KEYS = ("hidden_size", "d_model")


def _is_cpu(device: str) -> bool:
    """判断设备描述是否指向 CPU。"""
    return device.lower().startswith("cpu")


def read_declared_dimension(model_path: str) -> int | None:
    """从权重目录的 config.json 读出向量维度。

    纯文件读取：毫秒级，不需要 torch / transformers，也不会占显存。
    读不到（目录不存在、无 config.json、字段缺失、JSON 损坏）返回 None，
    由调用方降级处理，不在这一层抛错。

    Args:
        model_path: 权重目录

    Returns:
        维度；无法判定时 None。
    """
    try:
        with (Path(model_path) / "config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    for key in _DIMENSION_KEYS:
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _probe_dimension(model) -> int | None:
    """实测向量维度：真正编码一条短文本，量输出的长度。

    这是**最终**的核对手段——config.json 可能被人手改过、或模型结构特殊。
    取不到返回 None，由调用方报错。
    """
    try:
        # 用 encode_corpus 而不是 encode_queries：探测只看形状，
        # 没必要为它额外带上检索指令前缀
        vector = model.encode_corpus(["维度探测"], batch_size=1, convert_to_numpy=True)
    except Exception:  # noqa: BLE001 - 探测失败不掩盖真实错误，交给上层报配置错
        return None
    try:
        return len(vector[0])
    except (IndexError, TypeError):
        return None


class BgeEmbedder:
    """本地 BGE 向量化器（FlagEmbedding）。

    实现 EmbedderPort 协议。
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        query_instruction: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        """记录配置并解析维度，不做模型加载（懒加载以避免拖慢启动）。

        Args:
            model_path: 本地权重目录；None 时取 settings.embedding_model_path
            device: "cpu" / "cuda" / "cuda:0"；None 时取 settings.embedding_device
            query_instruction: 检索指令前缀；None 时取 settings
            batch_size: 编码批大小；None 时取 settings.embedding_batch_size

        Raises:
            ConfigurationError: 权重声明的维度与 MILVUS_DIMENSION 不一致
        """
        settings = get_settings()
        self._model_path = model_path or settings.embedding_model_path
        self._device = (device or settings.embedding_device or "cpu").lower()
        self._query_instruction = (
            settings.embedding_query_instruction
            if query_instruction is None
            else query_instruction
        )
        self._batch_size = max(
            1, min(int(batch_size or settings.embedding_batch_size), _MAX_ENCODE_BATCH)
        )
        self._declared_dimension = int(settings.milvus_dimension)
        # CPU 必须 fp32（见模块文档第 3 条）。FlagEmbedding 默认 True，故显式覆盖。
        self._use_fp16 = not _is_cpu(self._device)

        self._model = None
        self._dimension = self._resolve_dimension()
        # 模型加载可能被并发触发（多个上传请求），用锁保证只加载一次
        self._lock = threading.Lock()

    @property
    def dimension(self) -> int:
        """向量维度。以权重声明为准，与 MILVUS_DIMENSION 交叉校验后返回。"""
        return self._dimension

    @property
    def device(self) -> str:
        """推理设备。"""
        return self._device

    @property
    def batch_size(self) -> int:
        """单次送入模型的条数。"""
        return self._batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """编码文档块，不加检索指令。

        Args:
            texts: 片段正文列表

        Returns:
            与输入等长的 L2 归一化向量列表；输入为空时返回空列表。

        Raises:
            ValidationError: 存在空/纯空白文本（见模块文档第 7 条）
            ConfigurationError: 权重缺失或维度不符
        """
        if not texts:
            # 约束 6：encode_corpus([]) 会抛 AttributeError，这里短路
            return []
        self._reject_blank(texts)
        model = self._load()
        return self._encode_in_batches(model, texts)

    def embed_query(self, text: str) -> list[float]:
        """编码查询，加检索指令前缀。

        Args:
            text: 用户问题原文

        Returns:
            单条 L2 归一化向量。

        Raises:
            ValidationError: 查询为空
            EmbeddingError: 模型前向失败或输出形状非法（维度/NaN）
        """
        if not text or not text.strip():
            raise ValidationError("查询文本为空，无法编码")
        model = self._load()
        # batch_size 用配置值而非 1：FlagEmbedding 在 条数 < batch_size 时不打进度条，
        # 单条查询用 batch_size=1 反而会在日志里刷一行 tqdm。
        vector = self._forward(
            model.encode_queries,
            [text],
            stage="查询编码",
        )[0]
        return self._as_floats(vector, stage="查询编码")

    # ---------- 内部 ----------

    def _resolve_dimension(self) -> int:
        """确定向量维度：以权重为准，并与声明值交叉校验。

        Raises:
            ConfigurationError: 权重声明维度与 MILVUS_DIMENSION 不一致
        """
        if not self._model_path:
            # 路径缺失由 settings.validate_runtime() 在启动时拦截；
            # 这里只做降级，不在构造期抛错以免掩盖那条更清楚的提示
            return self._declared_dimension

        actual = read_declared_dimension(self._model_path)
        if actual is None:
            logger.warning(
                "无法从权重目录读出向量维度（缺 config.json 或字段），"
                "暂按 MILVUS_DIMENSION=%s 处理；加载模型时会再核对一次",
                self._declared_dimension,
            )
            return self._declared_dimension

        if actual != self._declared_dimension:
            raise ConfigurationError(
                f"向量维度不一致：权重 {self._model_path} 的 config.json 声明 "
                f"{actual} 维，但 MILVUS_DIMENSION 配置为 {self._declared_dimension}。"
                f"请把 MILVUS_DIMENSION 改成 {actual} 并重启；"
                "若 collection 已按旧维度建好，还需更换 MILVUS_COLLECTION，"
                "或删除该 collection 后重新入库（维度无法原地变更）。"
            )
        return actual

    def _encode_in_batches(self, model, texts: list[str]) -> list[list[float]]:
        """按 _batch_size 分批编码，并在长任务中打进度日志。

        FlagEmbedding 的 encode_corpus 内部本身也按 batch 切，
        这里显式分批是为了：① 让单次调用的峰值内存可控；
        ② 有条数可报的进度（内部批是黑盒，拿不到回调）。

        Raises:
            EmbeddingError: 任一前向失败，或输出条数/维度/数值非法
        """
        total = len(texts)
        vectors: list[list[float]] = []
        next_log = _PROGRESS_EVERY

        for start in range(0, total, self._batch_size):
            window = texts[start : start + self._batch_size]
            encoded = self._forward(model.encode_corpus, window, stage="文档编码")
            if len(encoded) != len(window):
                # 条数对不上会让上层 `zip(strict=True)` 或调用方静默错位——
                # 向量与正文张冠李戴，检索结果看起来"能跑但答非所问"。
                raise EmbeddingError(
                    f"向量化输出条数与输入不一致：输入 {len(window)} 条，"
                    f"输出 {len(encoded)} 条（batch={self._batch_size}）"
                )
            vectors.extend(self._as_floats(vector, stage="文档编码") for vector in encoded)

            done = start + len(window)
            if done >= next_log and total > _PROGRESS_EVERY:
                logger.info("向量化进度 %s/%s", done, total)
                next_log = done + _PROGRESS_EVERY

        return vectors

    def _forward(self, encode, texts: list[str], *, stage: str) -> list:
        """调用模型前向，把一切底层异常收敛成 EmbeddingError。

        为什么不在调用点裸调：torch 的异常（OOM、shape 不匹配、权重加载到
        一半的 `RuntimeError`）会一路冒到 FastAPI 的兜底处理器，用户只看到
        「服务内部错误」，日志里是一段没人读的栈。而这类故障的处置动作
        很明确——降批大小 / 换 CPU / 检查权重——所以必须带着这些线索抛出去。
        """
        try:
            return encode(
                texts,
                batch_size=self._batch_size,
                convert_to_numpy=True,
            )
        except AppError:
            # ConfigurationError（权重缺失/维度不符）等必须原样上抛：
            # 那是配置问题，包成 EmbeddingError 会让运维去查显存。
            raise
        except Exception as exc:  # noqa: BLE001 - 见方法文档
            hint = ""
            if "out of memory" in str(exc).lower() or "cuda" in str(exc).lower():
                hint = "；显存不足的可能性较大，可调小 EMBEDDING_BATCH_SIZE 或改用 CPU"
            raise EmbeddingError(
                f"向量化失败（{stage}，device={self._device}，"
                f"batch={self._batch_size}）：{type(exc).__name__}: {exc}{hint}",
                detail={
                    "stage": stage,
                    "device": self._device,
                    "batch_size": self._batch_size,
                    "exception": type(exc).__name__,
                },
            ) from exc

    def _as_floats(self, vector, *, stage: str) -> list[float]:
        """转成 float 列表，并校验维度与数值有效性。

        维度不符会让 Milvus 报一句"dim not equal"（看不出是谁的错），
        NaN/Inf 则会被写进索引、对**任意**查询都算出一个可疑的相似度——
        两者都必须在离开适配器之前拦下。
        """
        try:
            values = [float(x) for x in vector]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError(f"{stage} 输出了无法转成浮点的向量：{exc}") from exc

        if len(values) != self._dimension:
            raise EmbeddingError(
                f"{stage} 输出维度 {len(values)} 与预期 {self._dimension} 不符；"
                "通常意味着权重与 MILVUS_DIMENSION 不是同一套模型"
            )
        if not all(math.isfinite(value) for value in values):
            raise EmbeddingError(
                f"{stage} 输出包含 NaN/Inf——这类向量会污染索引（对任意查询都返回"
                "可疑相似度）；请检查权重完整性与输入文本是否含异常字符"
            )
        return values

    @staticmethod
    def _reject_blank(texts: list[str]) -> None:
        """拒绝空/纯空白文本（模块文档第 7 条）。

        切分器已保证不产出空块，所以这里是**防御性**的：
        万一上游规则变了，应该在入库这一步就失败，而不是把
        无语义的 padding 向量写进索引。
        """
        for index, text in enumerate(texts):
            if not text or not text.strip():
                raise ValidationError(
                    f"第 {index} 条待编码文本为空：BGE 会为它生成一条无语义的 "
                    "padding 向量，宁可不入库也不要污染索引；"
                    "请检查上游切分是否漏掉了空块过滤"
                )

    def _load(self):
        """懒加载 FlagEmbedding 模型；维度与权重声明不符时抛 ConfigurationError。"""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:  # 双检锁：等锁期间可能已被别的线程加载
                return self._model

            if not self._model_path:
                raise ConfigurationError(
                    "EMBEDDING_MODEL_PATH 未配置；请在 .env 中指向本地 BGE 权重目录"
                )
            if not Path(self._model_path).is_dir():
                raise ConfigurationError(
                    f"EMBEDDING_MODEL_PATH 不是有效目录：{self._model_path}"
                )

            # 权重已在本机，禁止联网（国内访问 huggingface.co 会长时间卡住）
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

            logger.info(
                "加载向量模型：%s（device=%s fp16=%s batch=%s）",
                self._model_path,
                self._device,
                self._use_fp16,
                self._batch_size,
            )
            try:
                model = self._load_flag_embedder()
            except AppError:
                raise
            except Exception as exc:  # noqa: BLE001
                # 权重目录在、但加载不起来（文件不完整、transformers 版本不兼容、
                # 内存不够）：这不是配置错误，而是运行时故障，但同样该带线索。
                raise EmbeddingError(
                    f"加载向量模型失败（{self._model_path}）："
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            self._guard_dtype(model)

            # 最终核对：真实编码一次，量出实际维度
            actual = _probe_dimension(model)
            if actual is None:
                raise ConfigurationError(f"无法获取向量维度：{self._model_path}")
            if int(actual) != self._dimension:
                raise ConfigurationError(
                    f"向量维度与权重声明不符：实际输出 {actual}，"
                    f"此前按 {self._dimension} 处理（来自 config.json / "
                    "MILVUS_DIMENSION）。请检查权重目录是否完整，"
                    "以及 MILVUS_DIMENSION 是否与之一致。"
                )

            self._model = model
            logger.info(
                "向量模型就绪：维度 %s，batch %s，device %s",
                self._dimension,
                self._batch_size,
                self._device,
            )
            return self._model

    def _load_flag_embedder(self):
        """构造 FlagAutoModel，目录名不在官方映射表时退回编码器型。"""
        from FlagEmbedding import FlagAutoModel

        options = {
            "devices": self._device,
            "use_fp16": self._use_fp16,
            "normalize_embeddings": True,  # 约束 2：余弦检索必需
            "convert_to_numpy": True,
            "query_instruction_for_retrieval": self._query_instruction or None,
        }
        try:
            return FlagAutoModel.from_finetuned(self._model_path, **options)
        except ValueError as exc:
            # FlagAutoModel 靠**目录名**查官方映射表来决定池化方式与指令模板；
            # 微调后另存为自定义目录名（如 H:\models\my-bge-ft）就查不到。
            # 此时显式声明为编码器型 BGE，行为与原 sentence-transformers 一致。
            logger.warning(
                "模型目录名不在 FlagEmbedding 官方映射表内，按编码器型 BGE 加载：%s（%s）",
                self._model_path,
                type(exc).__name__,
            )
            return FlagAutoModel.from_finetuned(
                self._model_path, model_class=_ENCODER_ONLY, **options
            )

    def _guard_dtype(self, model) -> None:
        """CPU 上禁止半精度权重（会抛 LayerNormKernelImpl not implemented）。"""
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
                "检测到 CPU 上的非 float32 权重（%s），已强制转回 float32——"
                "半精度在 CPU 上会直接抛 NotImplementedError",
                parameter.dtype,
            )
            model.model.float()
