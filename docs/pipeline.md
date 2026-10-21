# 全链路手册：加载 → 切分 → 向量化 → 入库 → 混合检索 → 重排 → 问答

一份「从磁盘上的文件到一句带页码引用的回答」的完整走查。每个阶段给出：  
**入口文件 → 数据形状变化 → 关键参数 → 实测坑 → 自检命令**。

想快速看架构分层去 [`architecture.md`](architecture.md)，  
想看两条链路的时序去 [`data-flow.md`](data-flow.md)，本文是**按阶段串起来的一条线**。

---

## 0. 全景

```
                    ┌──────────── 入库链路（离线，写）────────────┐
 文件                │                                            │
  │                 │  ①加载        ②解析/切分       ③向量化       ④入库
  ▼                 │  Loader   →   Page → Chunk  →  Chunk+向量 → Milvus
┌──────────┐        │   │            │               │            │
│ .pdf     │        │   │            │               │            │
│ .docx    │───────►│   │            │               │            │
│ .pptx    │        │   │            │               │            │
│ .txt/.md │        │   └─页码溯源贯穿始终（page_start/page_end）─┘
└──────────┘        └────────────────────────────────────────────┘
                                        │ 同一份 collection
                    ┌──────────── 问答链路（在线，读）────────────┐
 问题                │                                            │
  │                 │  ⑤混合召回          ⑥重排          ⑦生成      │
  ▼                 │  稠密+BM25 → RRF →  交叉编码器 →  提示词+LLM  │
┌──────────┐        │   │                  │             │        │
│ "报销上限"│───────►│   │                  │             ▼        │
└──────────┘        │   └──────────────► 带页码引用的回答 ◄────────┘
                    └────────────────────────────────────────────┘
```

一句话概括每步的**责任边界**（越界是这类项目最常见的设计问题）：

| 阶段      | 只做这件事                               | 不做                  |
| ------- | ----------------------------------- | ------------------- |
| ① 加载    | 文件 → `list[Document]`（带页码元数据）       | 不切分、不做领域映射          |
| ② 解析+切分 | `Document` → `Page` → `Chunk`（逐页独立） | 不碰向量、不碰存储           |
| ③ 向量化   | 文本 → L2 归一化的稠密向量                    | 不做检索、不知道 Milvus 存在  |
| ④ 入库    | `Chunk` → Milvus 行（稠密 + 稀疏双字段）      | 不生成、不重排             |
| ⑤ 混合召回  | 两路召回 → RRF 融合成一份候选                  | **不重排**（那是精度阶段）     |
| ⑥ 重排    | 候选 → 交叉编码器精排、截断                     | 不召回、不改查询            |
| ⑦ 生成    | 上下文 + 问题 → 回答                       | 不判断"能不能答"（拒答由条件边决定） |

---

## 1. 文档加载

**入口**：`app/infrastructure/loading.py` → `LangChainDocumentLoader.load(path)`

用 LangChain 的 Loader 把文件读成 `Document`，页码元数据由 Loader 提供：

| 格式             | 加载器                         | `metadata["page"]` | 映射后 `page_no`  |
| -------------- | --------------------------- | ------------------ | -------------- |
| `.pdf`         | `PyPDFLoader(mode="page")`  | 0 起整数              | `page + 1`（准确） |
| `.pptx`        | 自研 `PptxLoader(BaseLoader)` | 0 起整数              | `page + 1`（准确） |
| `.docx`        | `Docx2txtLoader`            | 无                  | 恒为 1           |
| `.txt` / `.md` | `TextLoader(encoding=显式)`   | 无                  | 恒为 1           |

**关键取舍**

- **为什么自研 `PptxLoader` 而不用 `UnstructuredPowerPointLoader`**：后者要拖进  
  `unstructured` 的 25 个依赖包；而 PPTX 的页边界就是幻灯片，`python-pptx` 直接能拿。
- **为什么 `.txt` 显式传编码**：`autodetect_encoding=True` 在本机崩过  
  （chardet ≥ 5.2 的 `FileEncoding` 装不下 `mime_type`），改成固定候选编码表。
- **为什么 `.doc` 直接拒绝**：抽取要么靠 `unstructured`（依赖变脏），  
  要么靠 Word COM（只在装了 Office 的 Windows 可用）。拒了并提示"另存为 .docx"。

**失败语义**：空文件、编码异常、加密、损坏、扫描件无文本 → 抛 `AppError`，  
**绝不返回空列表**（空列表会被上层当成"文档没内容"，掩盖真正的问题）。

**自检**：`scripts/check_loading.py`（45 项）

---

## 2. 解析与切分

### 2.1 解析：`Document` → `Page`

**入口**：`app/infrastructure/parsing.py` → `document_to_page(document)`

页码的唯一转换点，规则是三级判定（`parse_page_no()`）：

| metadata 里的 `page` | 来源 | 结果 |
|---|---|---|
| 0 起整数（含 `"4"`、`1.0`） | PyPDFLoader / PptxLoader | `page + 1` |
| 缺失 / `None` | Docx2txtLoader / TextLoader | 第 1 页 |
| 存在但读不出（`"iv"`、`"3 of 10"`、`2.5`、`True`、`-1`） | 少数 loader / 异常 PDF | 先试 `page_label`，仍不行 → **`None`（未知）** |

⚠️ **无法解析时不再回退第 1 页**（旧的 `try: int(raw)+1 except: 1` 已删）。旧实现会把
`"3 of 10"`、`"iv"` **静默**当成第 1 页，于是引用溯源显示「第1页」而正文在第 3 页——
**编造一个错误页码比承认不知道更糟**：用户按页码去翻会翻空，然后失去对整个溯源的信任。
`Page.page_no` 因此是 `int | None`，`None` 在接口里是 `null`，前端不渲染页码标签。

`page_label` 是 **1 起**的字符串，取开头十进制数字（`"3 of 10"` → 3、`"P3"` → 3）；
罗马数字不猜（`"iv"` 无法映射到物理页序）。
`parsing.py` 只 import `langchain_core.documents.Document`，不做文件 IO、不调 pypdf，
所以页码规则可以脱离磁盘与模型直接单测（`check_loading.py` 第 4 组 17 个用例）。

**顺带的连锁**：`splitting.py` 里判 `isinstance(page_no, int)`（**排除 bool**）不用
`page_no or 1` —— `0` 与 `None` 都是 falsy，一兜底就把「未知」变成「第 1 页」。

### 2.2 切分：`Page` → `Chunk`

**入口**：`app/infrastructure/splitting.py` → `LangChainSplitter.split(pages)`

走 LangChain 原生路径：

```python
create_documents(逐页文本, metadatas=[{"page_no": n}])
```

**按页独立切分，不产生跨页块**，因此每块 `page_start == page_end`，引用能精确到页。

> 已废弃的做法：先拼接所有页再切，然后用字符偏移反算"这块落在哪几页"。  
> 区间相交的逻辑绕且易错，还会产生跨页块（引用只能显示"第1-2页"，精度反而下降）。

**默认参数**：`CHUNK_SIZE=300` / `CHUNK_OVERLAP=50`（中文字符数，不是 token）。

**实测坑**

- `RecursiveCharacterTextSplitter` 的 `strip_whitespace` **默认 True** 且在  
  **分隔符处**生效：`'AAA\n\nBBB'` → `['AAA','BBB']`，那个 `\n\n` 谁都不属于、直接消失。  
  所以拼接所有块 ≠ 原文，任何"逐字相等"的断言必须先做空白归一化。
- 因此**不传** `add_start_index=True`：块文本已归一化，那个字符偏移拿去精确切片会对不上，  
  留一个误导人的元数据键不如不要。
- **空页必须剔除**：留着会切出空块，白占一条 Milvus 记录，还把 `chunk_index` 打出空洞。

**自检**：`scripts/check_splitting.py`（94 项）

---

## 3. 向量化

**入口**：`app/infrastructure/embedding.py` → `BgeEmbedder`

FlagEmbedding 加载本地 BGE 权重，CPU 推理，块向量做 L2 归一化（余弦检索的前提）。

**为什么不用 sentence-transformers**：LangChain 的 HF 封装（`HuggingFaceBgeEmbeddings`）  
硬依赖它且已 deprecated，套一层只有命名收益。FlagEmbedding 是 BAAI 官方推理库，  
`FlagAutoModel` 直接吃权重目录。

**dimension 的来源是权重，不是配置**：

```
读 <权重目录>/config.json 的 hidden_size  →  embedder.dimension
                                          →  ensure_collection(dimension)
                                          →  写入时再校验一次
MILVUS_DIMENSION 只是「声明值」，与权重不一致 → 启动即失败
```

用配置值建表、用权重值写入，会出现"建表 1024、写入 768"这种**只能等上传才炸**的错配。

**query 与 passage 分开编码**：BGE 的检索指令（`EMBEDDING_QUERY_INSTRUCTION`）  
**只加在 query 侧**（`encode_queries`），文档块不加（`encode_corpus`）。  
两者走同一条编码路径会落到不同语义空间，相似度整体偏移。

**实测坑**

- `FlagEmbedding.encode_corpus([])` 抛 `AttributeError` → 空列表必须短路。
- 空串/纯空白**不报错**，而是返回一条无语义的 padding 向量 → 必须显式拒绝，  
  宁可失败也不污染索引。
- `use_fp16` 默认 True → **CPU 上会崩**，必须按设备显式关掉。
- 批形状不同会有 float32 末位差异（实测最大 `1.19e-7`，余弦 `0.99999985`）——  
  golden 断言要留容差，不能要求逐位相等。

**自检**：`scripts/check_embedding.py`（63 项）、`scripts/check_models.py`（22 项）

---

## 4. Milvus 入库

**入口**：`app/infrastructure/vectorstore.py` → `MilvusVectorStore.upsert(chunks)`

一个 collection 装两路召回所需的一切 —— 这是混合检索能"零额外基础设施"的前提：

| 字段                        | 类型                                  | 用途                                   |
| ------------------------- | ----------------------------------- | ------------------------------------ |
| `id`                      | VARCHAR(64) PK                      | 片段 UUID                              |
| `kb_id` / `doc_id`        | VARCHAR(64)                         | 过滤与级联删除                              |
| `text`                    | VARCHAR(65535)，开 `chinese` analyzer | 正文 + **BM25 的输入**                    |
| `filename`                | VARCHAR(512)                        | 引用展示                                 |
| `chunk_index`             | INT64                               | 展示顺序                                 |
| `page_start` / `page_end` | INT64                               | **页码溯源**（-1 = 未知）                    |
| `embedding`               | FLOAT_VECTOR(dim)                   | 稠密路，COSINE + AUTOINDEX               |
| `sparse`                  | SPARSE_FLOAT_VECTOR                 | 稀疏路，由 `Function(BM25)` 从 `text` 自动生成 |

**实测坑**

- ⚠️ **`VARCHAR max_length` 按 UTF-8 字节计，不是字符**。实测 21845 个汉字  
  （正好 65535 字节）通过、30000 个汉字（90000 字节）被拒。所以截断按**字节**做，  
  且不切断多字节序列（`_clip_utf8`）。
- ⚠️ **ID 的字符集校验必须前置到写入**：带引号的 `kb_id` 能写进 Milvus，  
  却拼不出 `delete_by_kb` 的过滤表达式 → 那批向量**永远删不掉**。
- ⚠️ `filename` 来自上传方，用户可构造超长名字让整篇文档入库失败 → 截断而不是交给 Milvus 拒绝。
- 分批写入（`MILVUS_UPSERT_BATCH_SIZE`，默认 500）+ 末尾只 flush 一次：  
  写完即可检索，不需要重启或等 compaction。

**7 步流水线**（`app/rag/ingest.py`）：建文档行 → 落盘 → 加载 → 解析 → 切分 → 向量化 →  
写 Milvus → 标记 ready；任一步失败则标记 `failed` 并**清理残留**（半成品向量比没有更糟）。

### 4.0 进流水线之前：体积与内容检查

流水线本身只负责"把字节变成向量"，它假设输入是安全且可解析的。这层假设由
`IngestService._validate` 守住（顺序即成本从低到高）：

| 顺序 | 检查 | 失败 | 为什么放这个位置 |
|---:|---|---|---|
| 1 | 空内容 / 扩展名白名单 / 体积 | 400 | 最便宜，一个 500MB 的假 PDF 不该先去读魔数 |
| 2 | 文件名（控制字符、超长） | 400 `unsafe_file` | 换行会切开日志与响应头；>200 字符会在 Win32 上 `ENAMETOOLONG` |
| 3 | 魔数（`.pdf` 前 1KB 内 `%PDF-`；docx/pptx 必须是 zip） | 400 `unsafe_file` | 改后缀的成本是零，只认名字等于没校验 |
| 4 | PDF 页数 / zip 膨胀（条目数、解压总量、压缩比） | 400 `unsafe_file` | 这两类能把进程 OOM 掉，而 OOM 时日志里什么都没有 |
| 5 | 文本形态（NUL、控制字符占比 > 1%） | 400 `unsafe_file` | 二进制伪装成 `.txt` 会让下游按文本解析 |

通过之后才查重（正文 SHA-256）→ 排队（`INGEST_MAX_CONCURRENCY`）→ 进流水线。

⚠️ **压缩比阈值必须带绝对体积下限**：实测一份 36KB 的正常 docx（`<w:p>` 反复出现）
压缩比 346:1，只看比例会把正常文档判成炸弹。规则是"单条目解压后 ≥ 1MB **且** 比例超限"。

⚠️ 这里的检查只看**结构**，不解压、不执行。PDF 里的 JS、docx 的宏都不在射程内——
但 PyPDF / docx2txt / python-pptx 都只抽文本、不执行嵌入对象，这一层风险由工具选型消掉。

**去重与并发**（`app/rag/concurrency.py`）：
- `(kb_id, 正文 hash)` 命中已有 ready 文档 → 409 + `detail.doc_id`，**不新建记录**。
  数据库侧有部分唯一索引兜底（并发时应用层查重会双双扑空）。
- 同一 `(kb_id, hash)` 的入库串行（`DocumentLockPool`），重试按 `doc_id` 串行；
- 全局并发受 `INGEST_MAX_CONCURRENCY` 限制（默认 1）。**都是进程内的**，
  多 worker 需要 Redis 锁 → 部署上入库保持单写。

### 4.x 失败之后：原地重试（`reprocess`）

失败时**元数据行与原始文件都保留**，因此重试不需要重新上传：

```
POST /api/v1/documents/{doc_id}/reprocess
```

- **doc_id 不变、不新建记录**。这一点是重点：重传会生成新的 doc_id，留下「一条 failed + 一条 ready」，
  并把同一份正文的向量写两遍（实测库里同一份 txt 攒到过 4 条 ready，检索时互相挤占名额）。
- 原始文件路径由 `kb_id + doc_id + filename` 拼出（`uploads/{kb_id}/{doc_id}_{filename}`），
  所以 `_original_path()` 与 `_save_original()` 共用同一套命名规则——两处各写一遍，
  改一处忘一处就会变成"重试永远报文件丢失"。
- 重跑前再调一次 `delete_by_doc`：上次失败时的清理**自己也可能失败**（它的异常是被吞掉的），
  这次重试要兜住那个残留。幂等，没写进去时返回 0。
- 只允许 `failed` → `processing`：前者正在跑（重复触发会写两遍向量），后者已成功
  （那是"重建索引"而不是"重试"，需要先清空再写，重建窗口内检索不到这份文档，语义不同）。
- 原始文件丢失 → 404，`error` 里直接提示"请重新上传"——没有文件就真的没法重试。

`error` 开头是稳定的 code，决定"重试有没有意义"：`embedding_error` / `upstream_timeout` / OOM 类
`RuntimeError` 重试有用；`corrupted_document` / `encrypted_document` / `empty_document`
重试只会复现同样的报错。

**自检**：`scripts/check_ingest.py`（51 项，全部假上游，不连 Milvus）

**自检**：`scripts/check_vectorstore.py`（84 项）

---

## 5. 混合召回：向量 + BM25 → RRF

**入口**：`app/rag/retrieval.py` → `HybridRetriever.retrieve()`

```
query ─┬─► embed_query ─► Milvus  稠密检索（embedding, COSINE）  ─┐
       │                                                          ├─► RRF 融合 ─► candidates
       └─► 原文 ───────► Milvus  BM25 检索（sparse，服务端分词）  ─┘
```

| 路  | 擅长        | 短板                      |
| -- | --------- | ----------------------- |
| 稠密 | 语义相近但用词不同 | 专有名词 / 编号 / 型号（训练语料没见过） |
| 稀疏 | 关键词精确命中   | 语义改写毫无办法                |

**融合公式**（`app/rag/fusion.py`，纯函数、只依赖标准库）：

```
fusion_score(d) = Σ  w_path / (k + rank_path(d))        k = HYBRID_RRF_K（默认 60）
```

**为什么是 RRF 而不是"归一化后加权求和"**：两路分数**不可比**——余弦相似度落在  
[-1,1] 且可比区间很窄，BM25 分却无界。要加权求和就得先决定"怎么归一化"，  
而这是个换语料就失效的隐式假设。RRF 只吃**排名**，对分数尺度免疫。

**权重只有比值有意义**：`0.5/0.5` 与 `1.0/1.0` 排序完全相同，只是 `fusion_score` 差一倍。

**为什么在应用侧融合而不是用 Milvus 的 `hybrid_search(RRFRanker)`**：  
服务端融合拿不到逐路的 `vector_score` / `bm25_score`，调参是黑盒。  
多一次网络往返，换来「检不出结果时能看清是哪一路没召回」。

**实测坑**

- ⚠️ **混合模式下不能给稠密路设相似度阈值**。实测中文短查询对 300 字长块的余弦：
  | 查询         | 片段0        | 片段1    | 过 0.35？ |
  | ---------- | ---------- | ------ | ------- |
  | 多久需要做一次检查？ | 0.2804     | 0.2936 | ✗ 全滤掉   |
  | 巡检         | 0.3081     | 0.1043 | ✗       |
  | XG-8800    | 0.3135     | 0.0007 | ✗       |
  | 设备的固件怎么升级  | **0.5721** | 0.2554 | ✓       |
  `MIN_SIMILARITY=0.35` 会把整条稠密路**静默清空**，而混合检索还会**掩盖**它——  
  BM25 照常出结果，从返回内容上看不出向量路一条都没召回。  
  → 混合模式用 `HYBRID_DENSE_MIN_SIMILARITY`（默认 **0.0**，不设限），精度交给重排。  
  给两路只设一边阈值本身也不对称：BM25 分无界，根本没法设阈值。
- ⚠️ **加 `sparse` 字段后，稠密检索必须显式传 `anns_field`**，否则 Milvus 报  
  `multiple anns_fields exist`——报错完全不指向"你新加了字段"。
- ⚠️ **`describe_collection()` 的 analyzer 配置不在字段顶层**，压在 `field["params"]`  
  里且值是**字符串**（`enable_analyzer='true'`）。按顶层读会恒为 `None`，  
  schema 校验变成**假绿**。
- **中文分析器不过滤停用词**：jieba 把"的/了/是"也当普通词元，无关文本仅因共享一个  
  「的」字就能拿到 BM25 正分（实测真命中 2.40 vs 噪声 1.01）。这不是 bug 而是 BM25 的  
  固有特性——它负责**高召回**，精确性交给下游重排。
- **BM25 分无界且随语料漂移 → 默认不设阈值**。要设就按实测分数显式传，并接受它只对当前语料有效。
  阈值配成 `0` 时**显式转 `None`** 再下发：BM25 分永远 ≥ 0，传 0 等于"过滤负分"，
  看着像设了阈值其实什么都没做——这种"以为设了"的错觉比不设更麻烦。
  （`HybridRetriever._sparse()` 里有一行 `if self._bm25_min_score > 0` 就是这个原因。）
- ⚠️ **跨副本去重：重复上传的同名文件会挤占召回名额**（实测过的真实故障）。
  同一个 txt 传 4 次后，库里 4 个 `doc_id`、正文逐字相同，向量路给出
  **逐位相同**的相似度（`0.18244785070419312 ×4`），融合后 4 条并列，
  把 `top_k=8` 的一半名额占掉，真正不同的片段被挤到截断线外。
  用户看到「引用里同一个文件出现四次」，而检索"看起来正常工作"。
  → `fuse_detailed()` 按 `(filename, 空白归一化正文)` 去重（保留融合分最高者），
  **去重必须发生在 `top_k` 截断之前**，名额才真的释放；计数进
  `RecallTrace.duplicates_removed`，否则"召回变好了"无从归因。
  只合并**同名文件**：两篇文档引用同一段公告是合法重复，按正文合并会抹掉一个真实来源。
- **错误语义**：`fuse_detailed` 用 `ValueError` 表达"某一路结果没按分数降序"。
  那是**上游违约**（Milvus 返回未排序结果），不是调用方传错参数，
  所以 `HybridRetriever._fuse()` 把它包成 `RetrievalError`（502）——
  裸 `ValueError` 会冒到兜底处理器变成 500「服务内部错误」，运维看不出该查哪里。

**自检**：`scripts/check_retrieval.py`（128 项，核心断言是  
「关键词文档在纯向量下召不回、混合检索下召得回」）

---

## 6. 重排

**入口**：`app/infrastructure/rerank.py` → `BgeReranker.rerank()`

把 `(query, passage)` 拼成一个序列一起过交叉编码器，能捕捉细粒度交互，精度显著高于双塔，  
代价是慢，所以**只对初检的几条候选做**。

**分数尺度：统一用原始 logits**（`normalize=False`，不要套 Sigmoid）：

| 模式                        | 相关对   | 无关对     |
| ------------------------- | ----- | ------- |
| 原始 logits（本项目采用）          | +5.16 | -10.19  |
| `normalize=True`（Sigmoid） | 0.994 | 0.00004 |

`RERANK_MIN_SCORE` 因此按 logits 尺度设（默认 `0.0`，正负样本相差约 15~18）。  
把 `normalize=True` 的输出当 logits 用会把阈值语义完全弄反。

**为什么必须本地加载权重**：Ollama 不提供重排接口（`POST /api/rerank` 返回 404），  
任何"用 Ollama 做重排"的方案都不成立。

**处理顺序有讲究**：先按阈值过滤 → 再降序 → 最后截断 `top_n`。  
反过来（先截断再过滤）会让低分项占掉 `top_n` 名额，把真正相关的挤掉。

`RERANK_ENABLED=false` 时链路**降级为按融合分截断**，不是报错——  
重排是精度增强，不是链路必需件。

---

## 7. 问答：LangGraph 状态图

**入口**：`app/rag/graph.py`（图形状）+ `app/services/chat_service.py`（怎么调图）

```
START → retrieve → rerank ─┬─ 有可用片段 ──────────────► generate → END
                           │
                           └─ 无可用片段 ─┬─ 未超重试上限 ─► rewrite_query ─┐
                                          │                                 │
                                          └─ 已超上限 ────► refuse → END     │
                                                                            │
                          ◄─────────────────────────────────────────────────┘
```

**条件边是这张图存在的意义**：「检索不足 → 改写重试」这个回路用直线链表达不了。

**两个刻意的判定口径**

1. 路由判据是 `ranked` 而不是 `candidates`。  
   「召回到了」与「召回的能用」是两件事。重排可能把全部候选判为不相关  
   （实测：问巡检周期，召回的长块里混着报销与固件内容，logits 直接是负的）。  
   此时 `candidates` 非空、`ranked` 为空——盲目拿去生成，模型就会在一堆无关文字里  
   "找"答案，这正是幻觉的高发场景。
2. 检索为空时**不调用 LLM 生成**，直接返回固定拒答话术（`refuse` 节点）。  
   省一次推理是次要的，主要是不给模型"自由发挥"的机会。

**查询口径统一**：所有节点读 `rewritten_question or question`。  
只有 `rewrite_query` 写 `rewritten_question`，所以"检索用的查询"与"重排/生成用的查询"  
必然是同一个值——分开传会出现"重排用原问题、生成用改写问题"这类错位。

**生成用原始问题、检索用改写查询**：改写是为了让索引更容易命中，  
回答要面对的是用户真正问的那件事。

**流式的实现取舍**（`ChatService.ask_stream`）：  
token 只能来自 `llm.stream()`，图节点返回的是最终字段值而不是增量。  
所以流式走「检索子图 + 服务层直接流式生成」：

|       | 非流式 `ask()`   | 流式 `ask_stream()`                          |
| ----- | ------------- | ------------------------------------------ |
| 图     | 完整图           | `build_retrieval_graph`（retrieve → rerank） |
| 含改写重试 | ✅ 恒开（客户端本来就在等完整答案，重试不额外增加可感等待） | 由请求/默认值决定，见下 |
| 输出    | `Answer` + 引用 | SSE 事件 `retry? → sources → delta* → done`   |

**「是否重试」由前端用户决定**，服务端只提供三态解析，不替用户拍板：

| 请求体 `enable_retry` | 行为 |
| --- | --- |
| `true` / `false` | 前端用户的显式选择，服务端**一定照办**（显式 `false` 会压过服务端默认值） |
| 省略 / `null` | 落回服务端默认值 `CHAT_STREAM_RETRY_DEFAULT`（出厂 `false`） |

前端有两条路都能做到"让用户自己选"，两者只是同一个请求字段的不同用法：

- **预置开关**：把用户偏好随每次请求传上来。缺点是**盲签**——事先谁也不知道这一问用不用得上改写。
- **当场询问**：先按默认出结果，`done.can_retry=true`（首轮没找到且重试额度还没用满）时
  才问用户要不要「再找一次」，用户点了就用 `enable_retry=true` 把**同一个问题**重发一遍。

`GET /api/v1/chat/options` 把服务端默认值、重试上限与事件名暴露给前端，
这样前端**不必抄一份默认值**——抄的那份迟早与后端漂移，而表现是
「开关显示开着、后端却没重试」，没有任何报错。非流式 `POST /chat` 恒重试一次，
与本字段无关（`options.non_stream_always_retries` 亦如实说明）。

为什么会有人想打开它：一次改写要等 LLM 生成一轮查询（本机 7B 约 2~4 秒），
默认关是为了首帧时间；但长文档、慢问答这类场景宁可多等一次，也不想切成非流式。
打开时先推一个 `event: retry` 把这段静默填住，用户看到的是"正在改写查询"
而不是一个静止的界面。判据与次数**与完整图同源**（共用 `route_after_rerank`
与 `MAX_RETRY`），不会出现"流式重试 3 次、非流式 1 次"这种两边各说各话的偏差。
改写本身（提示词 + 失败降级 + 无效判定）也只有一份实现，在 `rag/rewrite.py`，
节点与流式重试都调它。

`done` 事件回填 `retried` / `attempts` / `can_retry` / `query`（成功与拒答两条路径
字段完全一致，由 `_done_event` 唯一产出）。注意 `retried` ≠ "调过改写"：
改写无产出（模型原样抄回问题）时不会重跑检索，此时 `attempts=1` 而 `retried=false`。

**提示词约束**（`app/rag/prompts.py`）：只能依据参考资料作答；资料不足必须**原样回复**  
`NO_ANSWER_TEXT`。上层用**字符串包含**判定 `refused`——所以改话术时  
`prompts.py` 与 `state.py` **必须一起改**，`check_chat.py` 有断言盯着这一点。


**客户端断开（半流取消）的语义**：`_sse_generator` 只**放行**关断信号、不做任何兜底回复。
`GeneratorExit` 与 `asyncio.CancelledError` 都是 `BaseException` 的子类（**不是** `Exception`），
所以 `except Exception` 拦不到它们，信号会一路钻到最底层的 `client.astream`，
把 httpx 连接释放掉；反过来，若把它们吞掉转成一条 `error` 事件，
CPython 会抛 `RuntimeError: async generator ignored GeneratorExit`。

**自检**：`scripts/check_chat.py`（246 项，全假上游，不需要 Milvus / BGE / Ollama）

---

## 8. 溯源页码：整条链路的验收

页码是本项目里**最容易"看起来对、实际错"**&#x7684;一环：答案正确不代表引用页正确。  
所以造了一份专门用来证伪的样例。

**夹具**：`tests/fixtures/handbook.pptx`（`make_sample_docs.build_handbook`）  
—— 5 页中文手册，**每页一个互不重叠的事实，关键数字只出现一次**：

| 页 | 标题        | 唯一事实          |
| - | --------- | ------------- |
| 1 | 员工手册 · 总则 | 施行日期          |
| 2 | 考勤制度      | **九点整** / 半小时 |
| 3 | 费用报销      | **五千元**       |
| 4 | 设备巡检      | **每季度一次**     |
| 5 | 假期管理      | **十天**        |

这样"问报销上限 → 应引用第 3 页"就成为一个**可证伪**的断言：  
引用对了说明 `loader → parser → splitter → Milvus → 检索 → 重排 → citations`  
整条链路的页码都没丢；引用错了能立刻定位到是哪一环把页码丢了。

> 为什么不复用 `sample.pptx`：那三页讲的是流水线本身，彼此共享大量词元，  
> 三个片段都可能被召回 → "引用了第 2 页"这句话本身没有信息量。
>
> 为什么中文页码验证用 PPTX 而不是 PDF：中文 PDF 需要嵌入 CID 字体才能渲染，  
> 而 PPTX 的每页就是一张幻灯片，天然带页边界且支持中文——  
> 它是本项目里"中文 + 多页 + 可自动生成"的唯一组合。  
> 英文 PDF 页（`sample.pdf`）单独覆盖 `PyPDFLoader` 那条路径。

**跑法**：

```bash
.venv\Scripts\python.exe scripts\ask.py --demo
```

它会：造夹具 → 建库 → 入库 handbook.pptx + sample.pdf → 逐题提问 →  
**自动校验每题的引用页** → 打印 PASS/FAIL → 清理知识库。

实测结果（5/5，且每题引用的都是**首选**页）：

| 问题                                     | 期望                | 实际引用  |
| -------------------------------------- | ----------------- | ----- |
| 报销金额上限是多少？                             | handbook.pptx 第3页 | 第3页 ✓ |
| 设备多久巡检一次？                              | handbook.pptx 第4页 | 第4页 ✓ |
| 上班时间是几点？                               | handbook.pptx 第2页 | 第2页 ✓ |
| 一年有几天年假？                               | handbook.pptx 第5页 | 第5页 ✓ |
| What comes after loading and chunking? | sample.pdf 第3页    | 第3页 ✓ |

**页码的诚实说明**：`page_start` / `page_end` 的语义是  
「**可溯源的最小定位单位**」，不是排版意义上的页。  
PDF / PPTX 有天然页边界，页码准确；TXT / MD / DOCX 是流式文本，  
整体记为第 1 页——**不按行数估算页码**，那是编造的页码，会让引用显示假信息。

---

## 9. 一页速查

| 阶段    | 文件                                   | 关键配置                                       | 自检                        |
| ----- | ------------------------------------ | ------------------------------------------ | ------------------------- |
| ① 加载  | `infrastructure/loading.py`          | `UPLOAD_ALLOWED_EXTENSIONS`                | `check_loading.py` 29     |
| ② 解析  | `infrastructure/parsing.py`          | —                                          | 同上                        |
| ② 切分  | `infrastructure/splitting.py`        | `CHUNK_SIZE` / `CHUNK_OVERLAP`             | `check_splitting.py` 91   |
| ③ 向量化 | `infrastructure/embedding.py`        | `EMBEDDING_MODEL_PATH` / `_BATCH_SIZE`     | `check_embedding.py` 52   |
| ④ 入库  | `infrastructure/vectorstore.py`      | `MILVUS_COLLECTION` / `_UPSERT_BATCH_SIZE` | `check_vectorstore.py` 83 |
| ⑤ 召回  | `rag/retrieval.py` + `rag/fusion.py` | `RETRIEVE_TOP_K` / `HYBRID_*`              | `check_retrieval.py` 104  |
| ⑥ 重排  | `infrastructure/rerank.py`           | `RERANK_ENABLED` / `_TOP_N` / `_MIN_SCORE` | 同上                        |
| ⑦ 问答  | `rag/graph.py` + `rag/nodes.py` + `rag/rewrite.py` | `LLM_*` / `CHAT_STREAM_RETRY_DEFAULT` | `check_chat.py` 211       |
| ⑧ 认证  | `infrastructure/security.py` + `services/auth_service.py` | `AUTH_*`                | `check_auth.py` 130       |
| 运维    | `scripts/init_milvus.py`             | —                                          | `check_models.py` 22      |

**合计 775 项断言**（29 + 91 + 22 + 52 + 83 + 104 + 211 + 130 + 53）。  
全部可在本地跑，其中 `check_vectorstore` / `check_retrieval` 需要 Milvus 容器；
`check_conversations` 的第 3 组需要 Redis（不可达时整组跳过）。
`check_chat` / `check_auth` 完全离线（不需要 Milvus、BGE 权重或 Ollama）。

```bash
# 一次跑完（check_vectorstore / check_retrieval 需 Milvus 在跑）
for s in loading splitting models embedding vectorstore retrieval chat auth; do
  .venv\Scripts\python.exe scripts\check_$s.py
done
```

**端到端**：

```bash
.venv\Scripts\python.exe scripts\ask.py --demo            # 问答 + 页码溯源校验
.venv\Scripts\python.exe scripts\ask.py "报销上限是多少？"  # 单库提问
.venv\Scripts\python.exe scripts\ask.py -k <kb_id> --debug -i   # 交互 + 检索明细
```

---

## 10. 已知限制 / 下一步

| 限制       | 现状                                                                | 修法                                                                          |
| -------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------- |
| 切分按字符数硬切 | Markdown 的多级标题会被揉进同一块，膨胀后重排会打负分（实测同一句单独成块 +0.63、混进 272 字长块 -3.99） | 让切分器**按标题切**（`MarkdownHeaderTextSplitter`），不要靠调小 `CHUNK_SIZE`——272 字已在推荐区间内 |
| 多轮对话未生效  | `history` 字段一路透传，但提示词里还没有插入历史消息的位置                                | 在 `build_answer_prompts` 里加上历史消息                                            |
| SSE 断开的清理依赖消费者 | `_sse_generator` **没有** `finally` 兜底：消费者既不 `aclose()` 也不取消任务时，生成器停在 `yield` 上，中间几层只能等 GC 回收（实测：`astream` 的 `finally` 不执行） | 生产路径没问题——Starlette 1.7 的 `StreamingResponse` 断开时取消跑 `stream_response` 的任务（它**并不调用** `body_iterator.aclose()`），信号因此同步钻到底层。自己写 `async for` 消费这个生成器时必须显式 `aclose()` |
| 流式重试的默认值在服务端 | 客户端不传 `enable_retry` 时用 `CHAT_STREAM_RETRY_DEFAULT`（出厂 `false`）：**同一份 `.env` 换到别的部署就是另一种默认行为**，而客户端代码一个字都没改 | 前端用 `GET /chat/options` 读默认值来初始化开关，不要自己写死；要改默认行为只改 `.env`，不必发前端版本 |
| 非流式不接受重试开关 | `POST /chat` **恒**重试一次，传 `enable_retry=false` 也不会关掉（`check_chat.py` 第 13.8 组是行为特征断言） | 有意为之：非流式调用方本来就在等完整答案，重试不额外增加用户可感的等待。真要能关，得给 `route_after_rerank` 加一个 `allow_retry` 状态位——目前不值这个改动 |
| 单库检索     | 每次请求带一个 `kb_id`                                                   | `RAGState.kb_id` → `kb_ids: list[str]`，过滤表达式改成 `in`                         |
