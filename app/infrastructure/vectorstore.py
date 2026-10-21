"""向量库适配器：Milvus（pymilvus 的 MilvusClient）。

实现目标端口：app.domain.ports.VectorStorePort

职能分工：
    - 本模块负责「向量 + 片段」的存取与相似度检索。
    - 知识库/文档的元数据不在这里，走 app/db（SQLite）。

Schema：
    id           VARCHAR(64)      主键（片段 UUID）
    kb_id        VARCHAR(64)      知识库 ID，检索过滤字段
    doc_id       VARCHAR(64)      文档 ID，删除用
    text         VARCHAR(65535)   片段正文（enable_analyzer=true，中文分析器）
    filename     VARCHAR(512)     来源文件名
    chunk_index  INT64            片段序号
    page_start   INT64            起始页（-1 表示未知）
    page_end     INT64            结束页（-1 表示未知）
    embedding    FLOAT_VECTOR(dim) 稠密向量（调用方写入）
    sparse       SPARSE_FLOAT_VECTOR 稀疏向量（**由 BM25 函数自动生成**）

索引：
    embedding  metric_type=COSINE, index_type=AUTOINDEX
    sparse     metric_type=BM25,   index_type=AUTOINDEX

函数：
    Function(name="bm25_text", type=BM25, in=["text"], out=["sparse"])

混合检索的两路召回都落在**同一个 collection**上：
    稠密路 = search()（对 embedding 做余弦）
    稀疏路 = search_bm25()（对 sparse 做 BM25 全文匹配）
好处是 kb_id 过滤、删除、可见性语义对两条路完全一致，
不会出现"向量删了但倒排索引还留着"这类两套存储才有的不一致。

实现说明（都是踩过的坑）：
    1. 维度必须与 embedding 模型一致（bge-large-zh-v1.5 = 1024）。
       维度不符时**显式报错并提示重建 collection**，绝不静默重建丢数据。
    2. Milvus 走 gRPC。若本机配置了全局 http_proxy，gRPC 会被代理
       拦成 502（表现为 "HTTP proxy returned response code 502"）。
       本适配器在连接前会为 localhost 补 NO_PROXY，见 _relax_proxy_for_local()。
    3. page_start / page_end 用 **-1 哨兵**表示"未知"，不用 NULL：
       可空字段在不同 pymilvus 版本上的行为不完全一致，哨兵值最稳。
    4. 写入后 flush：Milvus 的 segment 在 sealed 之前对检索不可见，
       不 flush 会出现「刚上传完立刻搜不到」的迷惑现象。
       代价是每次写入多一次同步等待，本项目的上传频率完全可以接受。
    5. **VARCHAR 的 max_length 按 UTF-8 字节计，不是字符**（实测 Milvus 3.0.2）：
       20000 汉字（60000 B）通过、21845 汉字（65535 B）刚好通过、
       30000 汉字（90000 B）被拒并报 `length of varchar field text exceeds
       max length, length: 90000, max length: 65535`。
       所以截断必须按字节做（_clip_utf8），按字符截断在中文下等于没截：
       `text[:65535]` 是 196605 字节，照样超限。
    6. 写入按 `settings.milvus_upsert_batch_size` **分批**，理由有两条：
       ① 单次请求体随文档规模线性膨胀（1024 维 float32 每条约 4 KB，
          三万块就是 120 MB），分批把峰值内存和请求体都压住；
       ② 有条数可报的进度——写一万块要十几秒，没有日志会让人以为卡死。
       **不在每批后 flush**：flush 是同步等待，每批都做会把耗时乘上批数，
       末尾统一 flush 一次即可满足「写完立刻可见」。
    7. 写入前校验 id 的字符集（**不只是**过滤表达式要校验）：
       带引号或空格的 kb_id 能被写进 Milvus，但 `delete_by_kb` 的过滤表达式
       拼不出合法表达式 → 那些向量**永远删不掉**。校验前置到写入，
       让脏 id 在入口就被拒绝，而不是留下一批删不掉的残留。
    8. **稀疏字段由 Milvus 侧的函数生成，调用方不能写**：
       schema 里声明 `Function(BM25, in=[text], out=[sparse])` 之后，
       insert/upsert 只要给 text，Milvus 会自动算好 sparse（实测 3.0.2 通过）。
       反过来，若在 data 里手填 sparse，会与函数输出冲突而报错。
       所以 `_to_row()` **刻意不包含 sparse**，并在 `_SCALAR_FIELDS` 里
       也不回传它——稀疏向量动辄上千维，回传给上层纯属浪费。
    9. **判断"已存在的 collection 是否具备全文检索能力"不能靠猜字段名**：
       实测 `describe_collection()` 的返回结构有两处反直觉：
         a. 字段级配置都不在字段顶层——`field["enable_analyzer"]` 恒为 None，
            真正的值压在 `field["params"]` 里，且**都是字符串**
            （`enable_analyzer='true'`、`analyzer_params='{"type":"chinese"}'`）；
         b. 稀疏字段带着 `is_function_output=True` 出现在 fields 里，
            但它究竟是"函数输出"还是"用户写入"，只有配合顶层
            `functions` 列表才能分辨。
       因此本适配器按「fields 里有没有 sparse + text 的 params.enable_analyzer
       是不是 true + 有没有 metric_type=BM25 的索引」三条一起判，
       逐条给出人话提示。任一条不满足 → `ConfigurationError` 要求重建，
       **绝不静默重建**（同维度不符的处理原则）。
   10. **中文分析器不过滤停用词，BM25 召回的噪声比英文场景大**：
       实测 Milvus 3.0.2 的 chinese 分析器（jieba）把"的/了/是"这类助词
       也当普通词元。后果是「完全不相关的词」仅因为与正文共享一个「的」字
       就能拿到 BM25 正分并被召回（实测：真命中 2.40 vs 停用词噪声 1.01）。
       这不是 bug，而是 BM25 的固有特性——它的定位是**高召回的关键词匹配**，
       精确性交给下游的交叉编码器重排（`rerank.py` 的 min_score）。
       本适配器因此**不设默认 BM25 阈值**：BM25 分无界且随语料变化，
       硬编码阈值在换语料后必然失效；需要时由调用方显式传 `min_score`。
       调参时用 `check_retrieval.py` 打印的真分数，别凭感觉设。
   11. **有了两个向量字段之后，稠密检索必须显式指定 anns_field**：
       加 sparse 字段前，collection 里只有一个向量字段，`search()` 不传
       anns_field 也能跑；加完之后 Milvus 会直接报
       `multiple anns_fields exist, please specify a anns_field in search_params`。
       这是加混合检索时**必然会踩到**的一处向后不兼容——不是配置问题、
       也不是数据问题，报错信息还不指向"你新加了字段"。
       所以 `search()` 现在显式传 `anns_field=_DENSE_FIELD`，
       两路各自认领自己的字段，谁也不依赖默认值。
"""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlparse

from app.core.config import get_settings
from app.core.exceptions import ConfigurationError, RetrievalError, ValidationError
from app.domain.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

# Milvus 的 VARCHAR 上限（**按 UTF-8 字节计**，见模块文档第 5 条）
_MAX_VARCHAR = 65535
# filename 字段的 max_length
_MAX_FILENAME = 512
# 写入批大小的兜底上限：配置写错（比如 100000）时不让单次请求失控
_MAX_UPSERT_BATCH = 2000
# 每累计这么多条打一行入库进度日志
_PROGRESS_EVERY = 2000
# page_start / page_end 的"未知"哨兵
_PAGE_UNKNOWN = -1
# 过滤表达式的安全字符集：kb_id / doc_id 都是 UUID，不应出现引号等字符
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# 需要读取的标量字段
# 注意：**不含 sparse**。稀疏向量动辄上千维，回传给上层纯属浪费（见模块文档第 8 条）
_SCALAR_FIELDS = (
    "id",
    "kb_id",
    "doc_id",
    "text",
    "filename",
    "chunk_index",
    "page_start",
    "page_end",
)

# ---------- 全文检索（BM25）----------
# 稠密向量字段名。加 sparse 字段之后，Milvus 不再能自行推断用哪个向量字段，
# 稠密检索必须显式指定 anns_field（见模块文档第 11 条）
_DENSE_FIELD = "embedding"
# 正文/全文检索用字段名
_TEXT_FIELD = "text"
# 稀疏向量字段名；由 BM25 函数从 text 自动生成
_SPARSE_FIELD = "sparse"
# BM25 函数名（出现在 schema 的 functions 列表里）
_BM25_FUNCTION = "bm25_text"
# 分词器：Milvus 内置 jieba 中文分析器（见模块文档第 10 条的噪声说明）
_ANALYZER_PARAMS = {"type": "chinese"}
# 全文检索的度量类型
_BM25_METRIC = "BM25"
# 稀疏倒排索引类型
_SPARSE_INDEX_TYPE = "AUTOINDEX"


def _relax_proxy_for_local(uri: str) -> None:
    """本机地址补进 NO_PROXY。

    pymilvus 的 gRPC 通道会读取 http_proxy；一旦代理拒绝 CONNECT，
    报错是不可读的 502。本机 Milvus 不该走代理，这里直接放行。
    """
    host = urlparse(uri).hostname or ""
    if host not in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}:
        return
    current = os.environ.get("NO_PROXY", "") or os.environ.get("no_proxy", "")
    entries = {item.strip() for item in current.split(",") if item.strip()}
    missing = {"127.0.0.1", "localhost"} - entries
    if not missing:
        return
    merged = ",".join(sorted(entries | {"127.0.0.1", "localhost"}))
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged
    logger.warning(
        "检测到全局代理但 NO_PROXY 未放行本机，已自动补上：NO_PROXY=%s（"
        "否则 pymilvus 的 gRPC 会被代理拦成 502）",
        merged,
    )


def _checked_id(value: str, field: str) -> str:
    """校验用于过滤表达式的 ID，避免表达式注入。"""
    if not _SAFE_ID.match(value or ""):
        raise ValidationError(f"{field} 格式非法，拒绝用于过滤表达式：{value!r}")
    return value


def _clip_utf8(text: str, max_bytes: int) -> str:
    """按 UTF-8 字节数截断，且保证不切断多字节字符。

    Milvus 的 VARCHAR max_length 是**字节**（见模块文档第 5 条），
    所以不能按字符切。切成 `bytes[:max]` 后末端可能是一个不完整的多字节
    序列，用 `errors="ignore"` 丢掉那几个残字节——即最多少 3 个字节，
    不影响语义，也不会写出半个汉字。
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _hit_to_chunk(hit: dict, kb_id: str) -> Chunk:
    """Milvus 命中 → 领域 Chunk。

    稠密路与稀疏路的返回结构一致（都是 `{distance, entity}`），
    转换规则只此一份——避免两条路在页码哨兵、缺省值上慢慢走偏。

    `distance` 的语义随检索方式而变（COSINE 相似度 / BM25 分），
    所以这里**不碰分数**，由调用方决定填进 vector_score 还是 bm25_score。
    """
    entity = hit.get("entity") or {}
    page_start = entity.get("page_start")
    page_end = entity.get("page_end")
    return Chunk(
        id=str(entity.get("id", "")),
        kb_id=str(entity.get("kb_id", kb_id)),
        doc_id=str(entity.get("doc_id", "")),
        text=str(entity.get("text", "")),
        chunk_index=int(entity.get("chunk_index") or 0),
        page_start=None if page_start in (None, _PAGE_UNKNOWN) else int(page_start),
        page_end=None if page_end in (None, _PAGE_UNKNOWN) else int(page_end),
        filename=str(entity.get("filename", "")),
    )


class MilvusVectorStore:
    """Milvus 向量库适配器。

    实现 VectorStorePort 协议。
    """

    def __init__(
        self,
        uri: str | None = None,
        collection: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        """建立连接并记录 collection 名。

        Args:
            uri: 形如 http://127.0.0.1:19530；None 时由 settings 拼装
            collection: collection 名；None 时取 settings.milvus_collection
            user / password: 开启鉴权时传入

        Raises:
            RetrievalError: 连接失败或服务不可达
        """
        settings = get_settings()
        self._uri = uri or settings.milvus_uri
        self._collection = collection or settings.milvus_collection
        self._metric_type = settings.milvus_metric_type
        self._index_type = settings.milvus_index_type
        self._upsert_batch_size = max(
            1, min(int(settings.milvus_upsert_batch_size), _MAX_UPSERT_BATCH)
        )
        # 由 ensure_collection() 记录；只用于给 search 一个可读的维度报错
        self._dimension: int | None = None

        _relax_proxy_for_local(self._uri)

        from pymilvus import MilvusClient

        kwargs: dict = {"uri": self._uri, "timeout": 10.0}
        if user:
            kwargs["user"] = user
        if password:
            kwargs["password"] = password

        try:
            self._client = MilvusClient(**kwargs)
            # 真正探活：构造 MilvusClient 不会立刻建连，
            # 不主动发一次请求的话错误会拖到首次写入才暴露
            self._client.list_collections()
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(
                f"无法连接 Milvus（{self._uri}）：{exc}；"
                "请确认容器已启动（docker ps），且 NO_PROXY 已放行 localhost"
            ) from exc
        logger.info("Milvus 已连接：%s，collection=%s", self._uri, self._collection)

    # ---------- 只读属性 ----------

    @property
    def upsert_batch_size(self) -> int:
        """单次写入的批大小（已按 _MAX_UPSERT_BATCH 收敛）。"""
        return self._upsert_batch_size

    @property
    def collection_name(self) -> str:
        """当前 collection 名（日志与运维脚本用）。"""
        return self._collection

    # ---------- 建表 ----------

    def ensure_collection(self, dimension: int) -> None:
        """确保 collection 与索引存在、维度一致、且具备全文检索能力。

        对**已存在**的 collection 做三项校验，任一不符都显式报错而**不重建**：
            ① 向量维度与当前 embedding 权重一致；
            ② 有 `sparse` 字段且其索引的 metric_type 是 BM25；
            ③ `text` 字段开启了 analyzer（否则 BM25 函数拿不到词元）。

        Args:
            dimension: 期望的向量维度

        Raises:
            ConfigurationError: 已存在的 collection 与当前配置/schema 不符
        """
        if self._client.has_collection(self._collection):
            existing = self._existing_dimension()
            if existing is not None and int(existing) != int(dimension):
                raise ConfigurationError(
                    f"collection `{self._collection}` 的向量维度是 {existing}，"
                    f"但当前配置为 {dimension}。"
                    "维度不符不能静默重建（会丢数据）：请更换 MILVUS_COLLECTION，"
                    f"或删掉该 collection 后重新入库。"
                )

            problems = self._bm25_schema_problems()
            if problems:
                raise ConfigurationError(
                    f"collection `{self._collection}` 缺少混合检索所需的 schema"
                    f"（共 {len(problems)} 项）：\n  - "
                    + "\n  - ".join(problems)
                    + "\n这些是建表期属性，无法原地补齐。请在确认可丢弃现有向量后重建："
                    "\n  .venv\\Scripts\\python.exe scripts\\init_milvus.py --recreate --yes"
                )

            self._client.load_collection(self._collection)
            self._dimension = int(dimension)
            logger.info(
                "collection 已存在：%s（维度 %s，稠密+BM25 双路就绪）",
                self._collection,
                dimension,
            )
            return

        from pymilvus import DataType, Function, FunctionType

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("kb_id", DataType.VARCHAR, max_length=64)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        # enable_analyzer 是 BM25 函数的**硬性前置**：不开就报
        # "BM25 function input field must set enable_analyzer to true"
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=_MAX_VARCHAR,
            enable_analyzer=True,
            analyzer_params=dict(_ANALYZER_PARAMS),
        )
        schema.add_field("filename", DataType.VARCHAR, max_length=_MAX_FILENAME)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("page_start", DataType.INT64)
        schema.add_field("page_end", DataType.INT64)
        schema.add_field(_DENSE_FIELD, DataType.FLOAT_VECTOR, dim=int(dimension))
        # 稀疏字段：只声明，值由下面的 BM25 函数在写入时自动生成
        schema.add_field(_SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name=_BM25_FUNCTION,
                function_type=FunctionType.BM25,
                input_field_names=[_TEXT_FIELD],
                output_field_names=[_SPARSE_FIELD],
            )
        )

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name=_DENSE_FIELD,
            index_type=self._index_type,
            metric_type=self._metric_type,
        )
        index_params.add_index(
            field_name=_SPARSE_FIELD,
            index_type=_SPARSE_INDEX_TYPE,
            metric_type=_BM25_METRIC,
        )

        self._client.create_collection(
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
        )
        self._client.load_collection(self._collection)
        self._dimension = int(dimension)
        logger.info(
            "collection 已创建：%s（维度 %s，稠密 COSINE + 稀疏 BM25）",
            self._collection,
            dimension,
        )

    def _existing_dimension(self) -> int | None:
        """读取已存在 collection 的向量维度。"""
        try:
            description = self._client.describe_collection(self._collection)
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取 collection 结构失败：%s", exc)
            return None
        for field in description.get("fields", []):
            if field.get("name") == _DENSE_FIELD:
                params = field.get("params") or {}
                dim = params.get("dim")
                return int(dim) if dim is not None else None
        return None

    def _bm25_schema_problems(self) -> list[str]:
        """检查已存在 collection 的全文检索能力，返回问题清单（空 = 就绪）。

        判据的来源见模块文档第 9 条：analyzer 配置压在 `field["params"]` 下
        且是字符串，不能读字段顶层；稀疏索引按 metric_type 认，比按索引名猜稳。

        读取失败时返回空清单（放行）：与 `_existing_dimension` 的口径一致
        ——"读不到"不等于"不具备"，不该因此拒绝启动；
        真正的失败会在 `search_bm25` 处带着完整上下文暴露。
        """
        try:
            description = self._client.describe_collection(self._collection)
            indexes = list(self._client.list_indexes(self._collection))
            index_infos = [
                self._client.describe_index(self._collection, name) for name in indexes
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取 collection 全文检索能力失败，跳过该项校验：%s", exc)
            return []

        fields = {field.get("name"): field for field in description.get("fields", [])}
        problems: list[str] = []

        sparse_field = fields.get(_SPARSE_FIELD)
        if sparse_field is None:
            problems.append(f"缺少稀疏字段 `{_SPARSE_FIELD}`（BM25 函数的输出目标）")
        elif not sparse_field.get("is_function_output"):
            problems.append(
                f"字段 `{_SPARSE_FIELD}` 不是函数输出——它应是 BM25 的产物，"
                "而不是被写入的普通字段"
            )

        text_field = fields.get(_TEXT_FIELD) or {}
        # 注意：值是字符串 'true'，不是布尔 True
        if str((text_field.get("params") or {}).get("enable_analyzer", "")).lower() != "true":
            problems.append(
                '字段 `text` 未开启 analyzer（params.enable_analyzer 不是 "true"），'
                "BM25 函数取不到词元"
            )

        if not any(
            str(info.get("metric_type", "")).upper() == _BM25_METRIC
            for info in index_infos
        ):
            problems.append(
                f"没有 metric_type={_BM25_METRIC} 的稀疏索引"
                f"（当前索引：{[(i.get('field_name'), i.get('metric_type')) for i in index_infos]}）"
            )

        return problems

    # ---------- 写入 ----------

    def upsert(self, chunks: list[Chunk]) -> int:
        """批量写入片段（要求 embedding 非空）。

        按 `milvus_upsert_batch_size` 分批发送；**全部批写完后只 flush 一次**
        （flush 是同步等待，每批都做会把耗时乘上批数，见模块文档第 6 条）。

        Returns:
            写入条数（各批 upsert_count 之和）。
        """
        if not chunks:
            return 0

        rows = [self._to_row(chunk) for chunk in chunks]
        total = len(rows)
        batch_size = self._upsert_batch_size
        written = 0
        next_log = _PROGRESS_EVERY

        for start in range(0, total, batch_size):
            window = rows[start : start + batch_size]
            try:
                result = self._client.upsert(
                    collection_name=self._collection, data=window
                )
            except Exception as exc:  # noqa: BLE001
                # 包装成 RetrievalError：pymilvus 的 MilvusException 不在
                # 异常处理器的覆盖范围里，直接抛出去会变成 500「服务内部错误」，
                # 用户看不到真正原因（例如字段超长）。
                raise RetrievalError(
                    f"写入 Milvus 失败（collection={self._collection}，"
                    f"第 {start + 1}~{start + len(window)} 条）：{exc}"
                ) from exc

            written += int((result or {}).get("upsert_count", len(window)))

            done = start + len(window)
            if done >= next_log and total > _PROGRESS_EVERY:
                logger.info("入库进度 %s/%s（collection=%s）", done, total, self._collection)
                next_log = done + _PROGRESS_EVERY

        self._client.flush(self._collection)
        logger.info(
            "写入向量 %s 条（collection=%s，分 %s 批，批大小 %s）",
            written,
            self._collection,
            (total + batch_size - 1) // batch_size,
            batch_size,
        )
        return written

    def _to_row(self, chunk: Chunk) -> dict:
        """领域片段 → Milvus 行，并在这里做全部字段级护栏。

        放在写入前的理由：字段超长会在 Milvus 侧被拒，而报错信息里
        只有一个 `row number`（是**这一批内的序号**，不是全局序号），
        拿到手也定位不到是哪一块。先在本地收口，能截断的截断、能拒绝的拒绝。
        """
        if not chunk.embedding:
            raise ValidationError(f"片段缺少向量，无法入库：{chunk.id}")

        # 三个 ID 都要能拼进过滤表达式，否则这批向量将来删不掉
        _checked_id(chunk.id, "id")
        _checked_id(chunk.kb_id, "kb_id")
        _checked_id(chunk.doc_id, "doc_id")

        text = chunk.text
        if len(text.encode("utf-8")) > _MAX_VARCHAR:
            # 正常切分不会到这一步（chunk_size 只有几百），但别让写入静默失败
            text = _clip_utf8(text, _MAX_VARCHAR)
            logger.warning(
                "片段 %s 正文超长，已截断到 %s 字节（Milvus VARCHAR 上限）",
                chunk.id,
                _MAX_VARCHAR,
            )

        filename = chunk.filename or ""
        if len(filename.encode("utf-8")) > _MAX_FILENAME:
            # 文件名来自上传方，超长是**用户可构造**的：不截断就会让整篇文档入库失败
            filename = _clip_utf8(filename, _MAX_FILENAME)
            logger.warning("文件名超长，已截断到 %s 字节：%s", _MAX_FILENAME, filename)

        return {
            "id": chunk.id,
            "kb_id": chunk.kb_id,
            "doc_id": chunk.doc_id,
            _TEXT_FIELD: text,
            "filename": filename,
            "chunk_index": int(chunk.chunk_index),
            "page_start": chunk.page_start if chunk.page_start is not None else _PAGE_UNKNOWN,
            "page_end": chunk.page_end if chunk.page_end is not None else _PAGE_UNKNOWN,
            _DENSE_FIELD: chunk.embedding,
        }

    # ---------- 检索 ----------

    def search(
        self,
        kb_id: str,
        query_embedding: list[float],
        top_k: int,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """向量检索，按 kb_id 过滤。"""
        settings = get_settings()
        threshold = settings.min_similarity if min_score is None else float(min_score)

        # 维度不对时 Milvus 的报错只有一句 "the dim of the vector is not equal to
        # the dim of the collection"，看不出是谁配错了，这里提前拦下。
        if self._dimension is not None and len(query_embedding) != self._dimension:
            raise ValidationError(
                f"查询向量维度 {len(query_embedding)} 与 collection `{self._collection}` "
                f"的 {self._dimension} 不一致——通常意味着查询侧与入库侧用了不同的 "
                "embedding 模型"
            )

        # kb_id 的校验必须在 try **之外**：它抛的是 ValidationError（400，属于
        # 调用方传错了），放进 try 会被下面的 except 一起包成 RetrievalError（502，
        # 意思是"上游坏了"）——语义完全反了。
        expression = f'kb_id == "{_checked_id(kb_id, "kb_id")}"'

        try:
            response = self._client.search(
                collection_name=self._collection,
                data=[query_embedding],
                # 必须显式指定：collection 里现在有 embedding 与 sparse 两个
                # 向量字段，不指定会报 multiple anns_fields exist（见文档第 11 条）
                anns_field=_DENSE_FIELD,
                filter=expression,
                limit=int(top_k),
                output_fields=list(_SCALAR_FIELDS),
                search_params={"metric_type": self._metric_type},
            )
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"Milvus 检索失败（collection={self._collection}）：{exc}") from exc

        hits = response[0] if response else []
        results: list[RetrievedChunk] = []
        for hit in hits:
            score = float(hit.get("distance", 0.0))
            if score < threshold:
                continue
            results.append(
                RetrievedChunk(chunk=_hit_to_chunk(hit, kb_id), vector_score=score)
            )
        return results

    # ---------- BM25 全文检索 ----------

    def search_bm25(
        self,
        kb_id: str,
        query: str,
        top_k: int,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """BM25 全文检索（走稀疏向量的倒排索引），按 kb_id 过滤。

        与 `search()` 的关系：两者是**同一 collection 上的两路独立召回**，
        结果各自按自己的分数排序，由上层（`rag/fusion.py`）用 RRF 融合。
        本方法只负责"把关键词能命中的片段捞出来"。

        Args:
            kb_id: 知识库 ID，作为过滤条件
            query: 查询原文；稀疏检索直接在服务端分词，本地不向量化
            top_k: 返回条数上限
            min_score: BM25 分下限。**默认 None = 不过滤**：
                BM25 分无界且随语料（文档数、平均长度）漂移，
                写死阈值换一份语料就失效；精度交给下游交叉编码器重排。
                需要时由调用方按实测分数显式给出。

        Returns:
            按 BM25 分降序的结果列表。空查询、top_k<=0 时返回空列表。

        Raises:
            ValidationError: kb_id 格式非法
            RetrievalError: Milvus 侧检索失败
        """
        text = (query or "").strip()
        if not text or top_k <= 0:
            return []

        threshold = None if min_score is None else float(min_score)

        # 同 search()：kb_id 校验必须在 try 之外，
        # 否则 400（调用方传错）会被下面的 except 包成 502（上游坏了）
        expression = f'kb_id == "{_checked_id(kb_id, "kb_id")}"'

        try:
            response = self._client.search(
                collection_name=self._collection,
                data=[text],
                anns_field=_SPARSE_FIELD,
                search_params={"metric_type": _BM25_METRIC},
                filter=expression,
                limit=int(top_k),
                output_fields=list(_SCALAR_FIELDS),
            )
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(
                f"Milvus BM25 检索失败（collection={self._collection}）：{exc}；"
                f"若提示字段不存在，说明该 collection 不是按当前 schema 建的，"
                "需用 scripts/init_milvus.py --recreate --yes 重建"
            ) from exc

        hits = response[0] if response else []
        results: list[RetrievedChunk] = []
        for hit in hits:
            score = float(hit.get("distance", 0.0))
            if threshold is not None and score < threshold:
                continue
            results.append(
                RetrievedChunk(chunk=_hit_to_chunk(hit, kb_id), bm25_score=score)
            )
        return results

    # ---------- 删除 ----------

    def delete_by_doc(self, doc_id: str) -> int:
        """删除某文档的全部片段。"""
        return self._delete_where(f'doc_id == "{_checked_id(doc_id, "doc_id")}"')

    def delete_by_kb(self, kb_id: str) -> int:
        """删除某知识库的全部片段。"""
        return self._delete_where(f'kb_id == "{_checked_id(kb_id, "kb_id")}"')

    def _delete_where(self, expression: str) -> int:
        """按表达式删除并 flush。"""
        try:
            result = self._client.delete(collection_name=self._collection, filter=expression)
            self._client.flush(self._collection)
        except Exception as exc:  # noqa: BLE001
            # 删除失败不应让整个业务操作回滚：向量残留是"脏数据"而非"错数据"，
            # 记日志让运维介入，比抛异常把用户的删除操作顶回去更合适。
            logger.error("按表达式删除向量失败（%s）：%s", expression, exc)
            return 0
        return int((result or {}).get("delete_count", 0))

    # ---------- 探活 ----------

    def ping(self) -> bool:
        """连通性探测，供 /ready 使用。"""
        try:
            self._client.list_collections()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Milvus 探活失败：%s", exc)
            return False

    def close(self) -> None:
        """释放客户端连接。"""
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass

    # ---------- 运维接口（供 scripts/ 下的运维脚本使用，**不属于** VectorStorePort）----------

    @property
    def client(self):
        """底层 MilvusClient，仅供运维脚本做 schema / 统计类查询。

        业务代码不要用：用它会绕过端口抽象，换向量库时这些调用会全部失效。
        """
        return self._client

    def drop_collection(self) -> bool:
        """删除整个 collection 及其全部向量（**不可逆**）。

        只为运维脚本的重建流程提供（换 embedding 模型后维度无法原地变更，
        本质上就是换 collection）。业务代码一律走 delete_by_doc / delete_by_kb：
        那两个是"删数据"，这个是"删表"，误用一次整库就没了。

        Returns:
            是否真的删掉了；原本不存在返回 False。
        """
        if not self._client.has_collection(self._collection):
            return False
        self._client.drop_collection(self._collection)
        logger.warning("已删除 collection：%s（不可逆操作）", self._collection)
        return True
