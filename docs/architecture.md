# 架构说明

## 1. 总览

`rag-backend` 是一个检索增强生成（RAG）后端服务，技术栈为
**FastAPI + LangChain 1.2 + LangGraph + Milvus**。

一句话概括数据流：

> 文档进来被**加载 → 解析 → 切分 → 向量化**存进 Milvus；
> 问题进来被**向量化 → 召回 → 重排**，再连同片段交给大模型生成带引用的回答。

## 2. 分层与依赖方向

```
                    ┌─────────────────────────────────────────┐
    HTTP 请求 ─────►│  app/api/v1/       接口层               │
                    │  路由、参数校验、SSE 编码                │
                    └────────────────┬────────────────────────┘
                                     │ Depends
                    ┌────────────────▼────────────────────────┐
                    │  app/services/     应用层               │
                    │  用例编排、事务边界、跨存储一致性        │
                    └────┬──────────────────────────┬─────────┘
                         │                          │
        ┌────────────────▼───────────┐   ┌──────────▼─────────────┐
        │  app/rag/      编排层       │   │  app/db/    持久化层    │
        │  LangGraph 状态图与入库链    │   │  SQLite 元数据          │
        └────────────────┬───────────┘   └────────────────────────┘
                         │ 依赖 Protocol
        ┌────────────────▼───────────────────────────────────────┐
        │  app/infrastructure/   适配层                          │
        │  loading / parsing / splitting / embedding /           │
        │  vectorstore / rerank / llm                            │
        └────────────────┬───────────────────────────────────────┘
                         │ implements
        ┌────────────────▼───────────────────────────────────────┐
        │  app/domain/   领域层（ports.py 定义端口）              │
        │  只依赖标准库，不依赖任何框架                            │
        └────────────────────────────────────────────────────────┘

        横切：app/core/（配置、日志、异常、依赖容器）
        对外：app/schemas/（Pydantic DTO）
```

**依赖只能向下，不能反向。** 这条规则带来三个实际好处：

1. **可替换**：把 Milvus 换成 pgvector，只改 `infrastructure/vectorstore.py`
   与 `container.py` 的一行装配，`rag/`、`services/` 零改动。
2. **可测试**：`rag/nodes.py` 依赖的是 `EmbedderPort` 协议，
   测试时传一个返回固定向量的 Fake 即可，无需加载模型、无需数据库。
3. **可读**：想看「检索怎么做」去 `rag/nodes.py`，想看「Milvus 怎么连」去
   `infrastructure/vectorstore.py`，职责不会糊在一起。

## 3. 各层职责

| 层 | 目录 | 该做的 | 不该做的 |
|---|---|---|---|
| 接口层 | `app/api/` | 路由、DTO 校验、SSE 编码、依赖注入 | 业务逻辑、直接操作数据库 |
| 应用层 | `app/services/` | 用例编排、事务边界、跨存储一致性、领域→DTO 转换 | 算法实现、HTTP 细节 |
| 编排层 | `app/rag/` | 状态流转、条件分支、提示词 | 具体驱动调用 |
| 持久化层 | `app/db/` | SQLite 元数据 CRUD、仓储接口实现 | 存向量或片段正文 |
| 适配层 | `app/infrastructure/` | 实现端口、唯一允许直接依赖第三方驱动 | 业务判断（如阈值过滤策略） |
| 领域层 | `app/domain/` | 领域模型、端口协议 | 依赖任何框架 |
| 横切 | `app/core/` | 配置、日志、异常、依赖容器 | 业务逻辑 |

## 4. 为什么用 LangGraph 而不是一条链

RAG 的真实控制流不是直线，至少有三处分支：

1. **首轮召回为空** → 改写查询重试一次（而不是直接放弃）；
2. **重排后全部低于阈值** → 拒答（而不是把无关片段硬塞给模型）；
3. **资料不足** → 短路拒答，**不调用大模型**。

用 `prompt | llm | parser` 这样的直线链表达不了上述分支；
硬塞进 `if/else` 虽然能跑，但控制流散落在服务代码里，难观测、难调试、
无法在中间环节插入人工审核。

LangGraph 把这些控制流显式建模成节点与条件边：

```
START → retrieve → rerank ─┬─ 有片段 ────────────► generate → END
                           │
                           └─ 无片段 ─┬─ 未超重试 ─► rewrite ─┐
                                      │                        │
                                      └─ 已超重试 ─► refuse → END
                                      ▲                        │
                                      └────────────────────────┘
                                             （回到 retrieve）
```

节点是**纯函数**：`(state) -> dict`。依赖通过 `make_*_node()` 闭包注入，
而不是在节点里 `import container` —— 这样单测可以直接传 Fake。

## 5. 组件选型理由

### 5.1 向量库：Milvus

- 元数据放 SQLite、向量放 Milvus，各用所长：
  Milvus 擅长高维相似度检索但事务性 CRUD 弱；元数据量小且需要强一致。
- 直接对接原生 `pymilvus`（`MilvusClient`），**不经** `langchain-milvus`：
  混合检索要用 Milvus 原生全文检索与 schema 里的 BM25 `Function`，
  服务端封装既建不出这种 schema，也拿不到逐路分数（调参变黑盒）。

**维度必须与 embedding 模型一致**（bge-large-zh-v1.5 = 1024）。
维度不符时应显式报错并提示重建 collection，**不要静默重建**——那会丢数据。

#### 入库写入的四条约定（都是实测出来的）

**1. `max_length` 按 UTF-8 字节计，不是字符。**
Milvus 报错时给的是字节数（`length: 90000, max length: 65535`），实测边界：

| `text` 内容 | 字节数 | 结果 |
|---|---|---|
| 20000 汉字 | 60000 | 通过 |
| 21845 汉字 | 65535 | **刚好通过** |
| 30000 汉字 | 90000 | 拒绝 |
| `filename` = 600 个 ASCII 字符 | 600 | 拒绝（上限 512）|
| `filename` = 600 个汉字 | 1800 | 拒绝 |

所以截断必须**按字节**做（`_clip_utf8`）。按字符截断在中文下等于没截：
`text[:65535]` 是 196605 字节，照样超限。`filename` 尤其要注意——
它来自上传方，**用户可构造出一个让整篇文档入库失败的文件名**，
所以这里必须截断而不是任由 Milvus 拒绝。

**2. 写入分批，末尾只 flush 一次。**
批大小取 `MILVUS_UPSERT_BATCH_SIZE`（默认 500，代码侧硬上限 2000）。
两个动机：① 请求体与峰值内存随文档规模线性增长（1024 维 float32 每条约 4 KB，
三万块一次性发出就是 ~120 MB）；② 写一万块要十几秒，分批才有条数可报的进度。
**不在每批后 flush**——flush 是同步等待，每批都做会把耗时乘上批数，
末尾统一 flush 一次就够满足"写完立刻可见"。

**3. ID 的字符集校验要前置到写入，不能只在校验过滤表达式时做。**
带引号或空格的 `kb_id` 能被写进 Milvus，但 `delete_by_kb` 拼不出合法过滤表达式
→ 那批向量**永远删不掉**。在 `upsert` 里就校验 `id`/`kb_id`/`doc_id`，
让脏 ID 在入口被拒绝，而不是留下一批删不掉的残留。

**4. `ValidationError` 与 `RetrievalError` 不能混。**
`search()` 里 `_checked_id(kb_id)` 必须放在 `try` **之外**：
它抛的是"调用方传错了"（400），一旦被 `except` 一起包成 `RetrievalError`
就变成了"上游坏了"（502），语义完全反了。
（这条是 `check_vectorstore.py` 抓出来的真实缺陷。）


### 5.2 向量模型：本地 BGE 权重（FlagEmbedding）

- 用 **FlagEmbedding**（BAAI 官方推理库，BGE 的「主场」）直接加载本地权重目录
  （如 `<models-dir>\bge-large-zh-v1.5`），不走 Ollama：Ollama 里的 bge 通常是量化版，
  且 manifest 目录不是可加载的权重目录。
  它直接基于 transformers 的 `AutoModel`/`AutoTokenizer`，**不经过 sentence-transformers**。
- 本机实测（`<models-dir>\bge-large-zh-v1.5`，FlagEmbedding 1.4.2，CPU）：结构为 BERT + CLS pooling，
  输出 **1024 维**；`normalize_embeddings=True` 后 L2 范数 = 1.0；
  「哪个是向量数据库？」对 Milvus 段落余弦 **0.591** / FAISS 段落 **0.463** / 无关的食堂段落 **0.032**
  （排序正确）；两次编码结果逐位一致。
- **query 与 passage 必须分开编码**：BGE 中文指令只加在 query 前。
  实现上分别走 `encode_queries`（自动加 `query_instruction_for_retrieval`）与
  `encode_corpus`（不加）；混用会让两者落到不同语义空间，相似度整体偏移。
  自检脚本里有一条专门断言「`embed_query(q)` 必须 ≠ `embed_documents([q])`」。
- **CPU 上禁止 fp16**：`use_fp16=True` 会抛
  `NotImplementedError: "LayerNormKernelImpl" not implemented for 'Half'`。
  注意 `FlagAutoModel.from_finetuned` 的 `use_fp16` **默认是 True**（为 GPU 优化），
  必须按设备显式传入；本项目在适配器里按设备自动决定并在加载后检查权重 dtype 兜底，
  而不是指望调用方记得改。
- **目录名会决定加载方式**：`FlagAutoModel` 靠**目录名**查官方映射表来选池化方式与指令模板。
  微调后另存为自定义目录名（如 `<models-dir>\my-bge-ft`）会抛 `ValueError`；
  适配器捕获后退回 `model_class="encoder-only-base"`，行为与原来的编码器型 BGE 一致。
- 权重已在本机时设 `HF_HUB_OFFLINE=1`，否则 transformers 会尝试连
  huggingface.co，国内会长时间卡住。
- **不用 LangChain 自带的 HF 封装**：`HuggingFaceBgeEmbeddings` /
  `HuggingFaceEmbeddings`（`langchain-community`）与 `langchain-huggingface` 那一路，
  在 `__init__` 里直接 `import sentence_transformers`，而本项目按需求已卸载它；
  它们也已被官方标 deprecated（removed in 1.0）。所以向量化**不套**
  LangChain 的 `Embeddings` 抽象，由本适配器直接对接 FlagEmbedding。
- **向量维度的唯一真相来源是权重**，不是配置：维度取自权重目录
  `config.json` 的 `hidden_size`（纯文件读取，毫秒级、不加载 torch），
  `MILVUS_DIMENSION` 退化为**声明值**并做交叉校验，不一致时
  `BgeEmbedder.__init__` 直接抛 `ConfigurationError`，服务**启动即失败**。
  `main.py` 建 collection 用的是 `embedder.dimension`。
  这样做是因为原先"建表用配置值、写入用模型输出"是两个真相来源，
  换模型（如 bge-base-zh = 768 维）时的错配只能等第一次上传才炸，
  且 Milvus 的报错与根因相距很远。读不到 `config.json` 时降级为声明值并告警，
  最终由 `_load()` 里的实编码核对兜底。
- **分批由适配器自理**：调用方（`rag/ingest.py`）直接传整份文本列表，
  批大小是 `EMBEDDING_BATCH_SIZE`（上限 64）。原先流水线里另有一个
  `_EMBED_GROUP_SIZE = 64` 的手工分组循环，属于重复定义——两处各写一个
  常量只会随时间漂移，还会让人误以为改流水线能影响编码批大小。现已删除。
  条数较多时按累计量打进度日志（CPU 上编码上万块要几分钟，日志不该空白）。
- **两条 FlagEmbedding 边界必须自己拦**（实测）：
  `encode_corpus([])` 抛 `AttributeError: 'list' object has no attribute 'keys'`
  （内部按 dict 取字段），故空列表在适配器短路返回 `[]`；
  空串 / 纯空白**不报错**，会返回一条 L2=1 的"纯 padding"向量——形状合法、
  语义为空，进索引后会对任意查询给出一段不高不低的相似度，属于静默污染，
  故显式抛 `ValidationError`。
  注意切分器已保证不产出空块，所以这条是**防御性**的。
- **批形状不同会导致浮点末位差异**（实测，别当 bug 修）：同一批 67 条按
  batch=32 编码，与逐条编码相比最大绝对差 `1.19e-7`（= 2^-23，正好是
  float32 在 1.0 处的 eps），余弦相似度 `0.99999985`；**同批形状**重跑则逐位一致。
  所以 "可复现" 只在固定批形状下成立，任何 golden 断言都要留容差
  （见 `scripts/check_embedding.py` 用例 4）。

### 5.3 重排模型：本地 BGE 交叉编码器

- 向量检索是双塔结构，query 与 passage 在编码阶段从不见面，精度天然受限。
  交叉编码器把 `(query, passage)` 拼成一个序列一起过模型，精度显著更高。
- **只对初检的少数结果（8~50 条）做重排**，不要对全库做——代价是慢。
- **分数尺度有两种，必须明确用哪一种**（本机 `bge-reranker-base` 实测，
  FlagEmbedding 1.4.2 / CPU）：
  `FlagReranker.compute_score()` 的 `normalize` **默认就是 False**，
  返回未经 Sigmoid 的**原始 logits**（实测相关对 `+5.16`、无关对 `-10.19`）；
  传 `normalize=True` 才套 Sigmoid 落到 `[0,1]`（实测 `0.99426` / `0.00004`）。
  本项目统一按原始 logits 处理，阈值 `0.0` 即可干净区分正负样本
  （实测相差约 15.3）。若误把 `normalize=True` 的输出当 logits，
  阈值语义会完全弄反。
- 模型名同样会影响加载方式：`FlagAutoReranker` 按模型名区分「编码器型」与
  「LLM 型（bge-reranker-v2-gemma 等）」，自定义目录名会抛 `ValueError`；
  适配器捕获后退回 `model_class="encoder-only-base"`。
- **Ollama 不提供重排接口**（`POST /api/rerank` 返回 404），
  交叉编码器必须本地加载权重。
- 重排分同时承担**两道闸门**：逐条过滤（`RERANK_MIN_SCORE`，默认 0.0 = 挡负分）
  与整批最高分（`RERANK_MIN_TOP_SCORE`，默认关）。后者拦的是"主题相邻、
  答案没有"这类**每条都拿正分**的错误召回——逐条闸门对它无能为力。
  阈值标定方法与实测数据见 [`anti-hallucination.md`](anti-hallucination.md)。

### 5.4 大模型：OpenAI 兼容协议

- 走 `/v1/chat/completions` 而非厂商 SDK：Ollama、vLLM、LM Studio
  以及各家云服务都支持这个协议，换后端只改 `LLM_BASE_URL` 与 `LLM_MODEL`。
- **端点与模型名不在代码里设默认值**，必须由 `.env` 提供（理由见第 7 节）；
  缺失时启动期即报错，不会带病运行。
- 用 `langchain-openai` 的 `ChatOpenAI`：它本身就是 Runnable，
  能直接作为 LangGraph 节点，且原生支持 `astream` 流式。

### 5.5 文档加载：LangChain Loader

**加载（loading.py）与解析（parsing.py）分开**，界线是「是否依赖框架」：

| 层 | 输入 → 输出 | 依赖 | 为什么这样切 |
|---|---|---|---|
| `loading.py` | 磁盘文件 → `list[Document]` | langchain-community / langchain-core | 格式驱动、编码探测都是框架相关的；换框架只动这里 |
| `parsing.py` | `list[Document]` → `list[Page]` | 仅 langchain-core 的 `Document` 类型 | 纯映射，可脱离磁盘与框架直接单测页码规则 |

各格式的加载器选型：

| 格式 | 加载器 | 理由 |
|---|---|---|
| `.pdf` | `PyPDFLoader(mode="page")` | pypdf 逐页提取，页码天然准确；`mode` 显式写出，不依赖默认值 |
| `.docx` | `Docx2txtLoader` | 段落与**表格单元格**都抽取（实测不丢），比手写 python-docx 遍历更省事 |
| `.txt` / `.md` | `TextLoader` | 编码由我们判定后**显式传入**（理由见第 8 节陷阱表） |
| `.pptx` | 自研 `PptxLoader`（继承 `langchain_core.document_loaders.BaseLoader`） | 见下 |

**PPTX 为什么不用 `UnstructuredPowerPointLoader`**：
它会连带安装 `unstructured` 及其 `nltk` / `numba` / `llvmlite` / `langdetect` /
`python-magic` 等 **25 个**传递依赖，在 Python 3.14 上有兼容风险；
而它相对 python-pptx 并不提供更准确的页码（两者都是「一张幻灯片一页」）。
自己继承 `BaseLoader` 实现约 30 行即可，`load()` / `lazy_load()` / `aload()`
全部由基类提供，对外行为与内置 Loader 完全一致。
将来若要做**元素级**切分（按标题/正文/表格分块），再换 Unstructured 那一路。

**页码的 0 起 / 1 起转换只做一次**：
LangChain 的 loader 用 0 起的 `page`（对齐数组下标），面向用户的引用是 1 起，
转换点唯一地放在 `parsing.document_to_page()`。
`metadata` 里拿不到 `page` 的格式（DOCX/TXT/MD）一律记为第 1 页。

**关于 `langchain-community` 已进入 sunset**：
导入时会打一条 `DeprecationWarning`，官方建议迁往独立集成包。
但截至本次实测，PDF/Text 还没有可用的独立包（`langchain-pypdf` 不存在，
`langchain-unstructured` 1.0.1 只包 unstructured 那一路），
且 `langchain-community` 0.4.2 依赖 `langchain-core>=1.4`，与本项目锁定的
core 1.6.9 兼容，安装只新增 4 个包、不动 langgraph。故暂留，后续再跟迁移。

### 5.6 文本切分：LangChain 递归字符切分

用 `langchain-text-splitters` 的 `RecursiveCharacterTextSplitter`。
难点不在"按长度切断"，而在**切断优先级**（先段落、再句子、最后兜底按字符）
与相邻块重叠，这两件事 LangChain 已经做对了，没必要自己写。

**页码溯源由 LangChain 的元数据传递负责**，不是我们算出来的：

```
Page(1 起页码) ──► create_documents(每页文本, metadatas=[{"page_no": n}, ...])
                ──► 每个块自带该页的 page_no（库会把元数据复制到该页切出的每一块）
                ──► Chunk(page_start = page_end = page_no)
```

**按页独立切分，不做跨页块**（早期实现是把所有页拼成一整串再切，已废弃）。
拼接后要自己维护字符区间、再按区间相交反查"这块落在哪几页"，
逻辑绕且易错；更关键的是会产生**跨页块**——一块同时属于第 1、2 页，
引用溯源只能显示"第1-2页"，精度反而下降。按页切分后每块**必然且只属于一页**，
引用能精确到页，也天然避免上一页末句与下一页首句被凑进同一块。

代价：极短的页（如 PPT 单页只有几十字）不会被邻居拼大。
对本项目是可接受甚至更好的取舍——幻灯片本就是独立语义单元，
而项目的调优结论是块宁小勿大（长块会同时稀释向量相似度与重排得分）。

**两个实测行为，别凭印象改**（langchain-text-splitters 1.1.3）：

| 行为 | 说明 |
|---|---|
| `strip_whitespace` 默认 True，且在**分隔符处**生效 | `'AAA\n\nBBB'` → `['AAA', 'BBB']`，那个 `\n\n` 谁都不属于、直接消失。**块文本不能按字节还原原文**：拼接所有块 ≠ 原文，差值就是这些边界空白。对语义与页码溯源无影响，但任何"逐字相等"的断言都必须先做空白归一化 |
| 刻意**不传** `add_start_index=True` | 既然块文本已被归一化，"块在原文中的字符偏移"拿去做精确切片会对不上。留一个用不上、还会误导人的元数据键，不如不要 |
| 空页必须剔除 | 空页会切出空块，白占一条 Milvus 记录，还会把 `chunk_index` 打出空洞 |

### 5.7 混合检索：向量 + BM25 → RRF

单路向量检索有一个结构性短板：**专有名词、编号、型号**这类词在训练语料里没见过，
它们的向量未必彼此最近。BM25 恰好相反——词表命中就得分，但对语义改写毫无办法。
两者互补，这是混合检索的标准动机。

**两路召回都在同一个 collection 上**，不引入第二套存储：

| 路 | 字段 | 度量 | 索引 |
|---|---|---|---|
| 稠密 | `embedding`（调用方写入） | COSINE | AUTOINDEX |
| 稀疏 | `sparse`（**BM25 函数自动生成**） | BM25 | AUTOINDEX |

`sparse` 由 schema 里声明的 `Function(BM25, in=[text], out=[sparse])` 在写入时算出，
所以 `_to_row()` 刻意**不包含** `sparse`，`_SCALAR_FIELDS` 也不回传它
（稀疏向量动辄上千维，回传给上层纯属浪费）。

放在同一个 collection 的好处是 kb_id 过滤、删除、可见性语义对两条路完全一致，
不会出现"向量删了但倒排索引还留着"这种两套存储才有的不一致。

#### 融合：加权 RRF

```
fusion_score(d) = Σ_path  w_path / (k + rank_path(d))
```

**为什么不是"分数归一化后加权求和"**：两路分数不可比。余弦相似度落在 [-1,1]
且可比区间很窄（本机 BGE 实测相关约 0.66、无关约 0.23），BM25 分却无界
（实测同一语料里 2.40 与 1.01 并存，换一份语料整体尺度又全变）。
要加权求和就得先决定"怎么归一化"，而这是个换数据集就失效的隐式假设。
RRF 只吃**排名**，对分数尺度免疫。

**权重的绝对大小无意义，只有比值有意义**：`0.5/0.5` 与 `1.0/1.0` 排序完全相同，
只是 `fusion_score` 差一倍。所以它不能当概率或阈值用。

**排序的确定性**：`fusion_score` 降序 → `best_rank`（两路中较好的排名）升序 →
`chunk.id` 升序。第三级纯粹为了消除不确定性——RRF 在"两条片段各自只在一条路
排第 1"时天然同分，不定序的话结果会随字典顺序漂移，测试与复现都无从谈起。

#### 融合放在应用侧，不用 Milvus 的服务端 `hybrid_search`

Milvus 3.0.2 支持 `hybrid_search(RRFRanker)`，也能做对。不用它的原因只有一个：
**那样拿不到逐路的 `vector_score` / `bm25_score`**，调参时是黑盒。
本项目的取舍是多花一次网络往返，换来"检不出结果时能看清是哪一路没召回"——
这个信息对排查问题的价值远大于省一次 RTT。

#### 两个实测行为

**1. 加 `sparse` 字段后，稠密检索必须显式传 `anns_field`。**
collection 里只有一个向量字段时不传也能跑；有两个之后 Milvus 直接报
`multiple anns_fields exist, please specify a anns_field in search_params`。
报错信息完全不指向"你新加了字段"，是加混合检索时**必然踩到**的一处向后不兼容。
（这条是 `check_retrieval.py` 跑出来的真实缺陷。）

**2. 中文分析器不过滤停用词，BM25 召回带噪声。**
Milvus 的 `chinese` 分析器（jieba）把"的/了/是"也当普通词元，于是
"完全无关的词"仅因与正文共享一个「的」字就能拿到 BM25 正分并被召回
（实测：真命中 2.40 vs 停用词噪声 1.01）。

这不是 bug，而是 BM25 的固有特性——它负责**高召回**，精确性交给交叉编码器重排。
因此本项目**默认不设 BM25 阈值**：BM25 分无界且随语料漂移，写死一个
`min_bm25_score` 换一份语料立刻失效。需要时由调用方按实测分数显式传。

#### 检索逻辑的归属

`rag/retrieval.py` 的 `HybridRetriever` 是对外 `RetrievalService`（只检索不生成）
与对内 LangGraph `retrieve` 节点**共用的同一份实现**。
放在 `rag/` 而不是 `services/`，就是为了让两者不可能跑出不同的检索结果——
"接口查到的"与"问答用的"必须是一套。容器里 `get_hybrid_retriever()` 是单例，
两个消费者拿到同一个实例。

重排不在这里：它是交叉编码器的活（`infrastructure/rerank.py`），属于「精度」
而非「召回」阶段。分开之后才能单独回答"是没召回到，还是召回了但被重排丢了"。

## 6. 数据存储职责划分

| 数据 | 存放位置 | 说明 |
|---|---|---|
| 知识库/文档元数据 | SQLite (`data/rag.db`) | 名称、状态、片段数、时间戳 |
| 原始上传文件 | `data/uploads/{kb_id}/` | 正文的权威副本，便于重试与追溯 |
| 片段正文 + 向量 | Milvus collection | 检索的唯一来源 |
| 模型权重 | 项目外目录 | 体积大，不入库（见 `.gitignore`） |

**关键：SQLite 不存片段正文与向量。** 因此删除文档时必须同时清理三处
（元数据、Milvus 向量、uploads 文件），否则会留下孤儿数据——
向量还在但检索不到、文件还在但用户看不到。这个一致性由
`services/collection_service.py` 负责。

## 7. 配置管理

- 唯一入口 `app/core/config.py` 的 `Settings`（pydantic-settings）。
- 优先级：环境变量 > `.env` > 代码默认值。
- **业务代码禁止散落 `os.getenv`**，一律 `get_settings()`。
- 配置只有一个来源：`Settings` ← `.env`。本机路径、密钥一律走 `.env`
  （已在 `.gitignore` 中），仓库里只有 `.env.example` 这一份模板。

### 7.1 默认值分两类（判断标准：换台机器要不要改）

| 类别 | 例子 | 代码里的默认值 |
|---|---|---|
| 与机器无关的常量 | Milvus 主机/端口、切分参数、各类阈值 | 允许带真实值，开箱可跑 |
| 与机器/后端相关 | `EMBEDDING_MODEL_PATH`、`RERANK_MODEL_PATH`、`LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY` | **一律留空**，只从 `.env` / 环境变量读 |

理由有三：`.env` 已被 `.gitignore` 忽略，而 `config.py` 要入库；这些值本质是
「某台机器上的事实」；空默认值才能让 `validate_runtime()` 把「缺配置」
变成启动期硬失败。

**反面教材（本骨架真实发生过）**：代码里给 `llm_base_url` / `llm_model`
填了真实值，却给模型权重目录留空。结果是**失败模式不一致**——没有 `.env`
时服务照常启动、LLM 照常回答，但 embedding 以空路径加载，向量检索整体失效
且无显式报错。这种「半可用」状态比直接崩溃更难排查。

### 7.2 启动期校验

`create_app()` 启动时调用 `settings.validate_runtime()`，一次性校验：

1. 必填项是否为空（LLM 端点/模型名、embedding 目录、rerank 目录）；
2. 路径是否真实存在且为目录；
3. 数值区间（`CHUNK_OVERLAP < CHUNK_SIZE`、权重落在 `[0,1]`、
   维度与 top_k 为正等）。

**一次性列出全部问题**而非只报第一个，避免「改一处、重启、再撞下一处」
的来回折腾。校验不通过抛 `ConfigurationError`，服务拒绝启动。

## 8. 已知环境陷阱

这些问题在本机环境中真实发生过，写进文档避免重踩：

| 陷阱 | 现象 | 对策 |
|---|---|---|
| 全局 HTTP 代理 | pymilvus 走 gRPC，被代理拦成 `502`，报 `HTTP proxy returned response code 502` | 设 `NO_PROXY=localhost,127.0.0.1` |
| 分块过大 | 检索永远无结果 | 中文 `CHUNK_SIZE` 用 150~300；长块会同时稀释相似度与重排得分 |
| torch 与 transformers 版本 | `AttributeError: module 'torch' has no attribute 'accelerator'` | `transformers>=5` 需要 `torch>=2.6`。本项目不直接钉 torch——它由 FlagEmbedding 传递引入，钉了反而互相打架 |
| FlagEmbedding 靠目录名查映射表 | 自定义目录名的权重抛 `ValueError: Model name 'xxx' not found in the model mapping` | 适配器捕获后退回 `model_class="encoder-only-base"`；或把权重目录名改成官方名（如 `bge-large-zh-v1.5`） |
| FlagEmbedding 的 `use_fp16` 默认 True | CPU 上报 `LayerNormKernelImpl not implemented for 'Half'` | 按设备显式传 `use_fp16`（CPU→False），并在加载后检查 dtype 兜底 |
| CPU + fp16 | `LayerNormKernelImpl not implemented for 'Half'` | `device == "cpu"` 时强制关 fp16 |
| 目录名非 ASCII 连字符 | 路径找不到 | 引用路径时原样复制，不要手打 |
| `TextLoader` 的默认编码 | 同一份 GB18030 文本，加不加 `-X utf8` 结果不同：一种读出中文，一种直接抛 `Error loading ...` | `encoding` 必须**显式传入**。默认值取的是系统默认编码，会随 `-X utf8` / `PYTHONUTF8` / 区域设置漂移 |
| `TextLoader` 自动探测编码 | `autodetect_encoding=True` 抛 `TypeError: FileEncoding.__new__() got an unexpected keyword argument 'mime_type'` | chardet>=5.2 的 `detect_all()` 返回了 langchain 的 NamedTuple 未声明的键。本项目改用固定候选编码表（`loading.py`），不依赖自动探测 |
| pypdf 对非 PDF 文件报错 | 报错文本会直接打到 stdout（`invalid pdf header` / `EOF marker not found`），容易被误当成程序输出 | 这些是 pypdf 自己的日志。真正的失败由 `loading.py` 统一包装成 `AppError: PDF 加载失败：...` |
| 建 BM25 时 `text` 没开 analyzer | 建表直接失败：`BM25 function input field must set enable_analyzer to true` | `add_field("text", ..., enable_analyzer=True, analyzer_params={"type": "chinese"})`。这一项是 BM25 函数的硬性前置 |
| 双向量字段下稠密检索不传 `anns_field` | 加完 `sparse` 后原本正常的检索突然全挂：`multiple anns_fields exist, please specify a anns_field in search_params` | `search()` 显式传 `anns_field="embedding"`。报错信息不指向"你新加了字段"，是加混合检索**必然踩到**的向后不兼容（见 5.7） |
| 误以为 `describe_collection()` 会回传 analyzer 配置在字段顶层 | `field["enable_analyzer"]` 恒为 `None`，据此写的 schema 校验永远"通过"（假绿） | 真正的值在 `field["params"]` 里且是**字符串**：`enable_analyzer='true'`、`analyzer_params='{"type":"chinese"}'`。实测 Milvus 3.0.2 |
| 中文分析器不过滤停用词 | "完全无关的词"仅因与正文共享一个「的」字就被 BM25 召回（实测真命中 2.40 vs 噪声 1.01） | 不用修——这是 BM25 的特性。它负责高召回，精度交给交叉编码器重排；**不要**为此设 BM25 阈值（分无界且随语料漂移） |
| 给 BM25 设固定阈值 | 换一份语料后阈值失效，要么全滤掉要么全放行 | BM25 默认不设阈值。要设就按 `check_retrieval.py` 打印的实测分数显式传 `min_score`，并接受它只对当前语料有效 |
| 混合模式下沿用 `MIN_SIMILARITY=0.35` 卡稠密路 | 检索**不报错、只是变差**：中文短查询对 300 字长块的余弦只有 0.28~0.35，整条稠密路被静默清空。更隐蔽的是混合检索会**掩盖**它——BM25 还在出结果，看不出来 | 混合模式下稠密路默认不设限（`HYBRID_DENSE_MIN_SIMILARITY=0.0`），精度交给重排。`MIN_SIMILARITY` 只在纯向量模式下生效。这个值会打进 `RecallTrace` 的日志，就是为了让它可观测 |
| 检索层召回正确、最终结果却为空 | 误判成"检索坏了"，去调检索参数 | 先看 `debug=true` 的逐路明细：若两路都命中，问题在**切分**——几个无关小节被揉进同一块，重排对它打负分（实测同一句单独成块 +0.63、混进 272 字长块 -3.99）。见 data-flow 的实测表 |

## 9. 扩展点

架构预留了以下扩展位置，实现时不必改动既有分层：

| 扩展 | 落点 |
|---|---|
| 多知识库联合检索 | `RAGState.kb_ids: list[str]`，Milvus 过滤条件改为 `in` |
| 换向量库（pgvector 等） | 新增 `infrastructure/` 实现，改 `container.py` 一行 |
| 加人工审核节点 | 在 `graph.py` 的 generate 前插入节点，用 `interrupt` 暂停 |
| 多轮对话记忆 | `RAGState.history` 已有字段；接入 LangGraph checkpointer |
| ~~BM25 混合检索~~ | ✅ 已实现，见 5.7（`rag/retrieval.py` + `rag/fusion.py`） |
| 查询路由（多库/多策略） | 在 retrieve 前加一个路由节点，条件边分流 |
| 可观测性 | 在节点函数内加 LangSmith / OpenTelemetry span |

## 10. 实现进度与设计偏差

### 10.1 已完成

**入库链路**端到端可用：`core`（配置/日志/异常/容器）、`db`（表结构与仓储）、
`infrastructure`（加载/解析/切分/向量化/向量库）、`rag/ingest.py`（7 步流水线）、
`services`（collection/ingest）、`api` 与 `main.py`。

**检索链路**（只检索不生成）端到端可用：`infrastructure/vectorstore.py`
（稠密 COSINE + 稀疏 BM25 双路）、`rag/fusion.py`（加权 RRF）、
`rag/retrieval.py`（`HybridRetriever`）、`services/retrieval_service.py`
（混合召回 + 重排）、`api/v1/retrieval.py`（`POST /knowledge-bases/{kb_id}/search`）。
`nodes.py` 的 `retrieve` 节点也已实现，与检索服务共用同一个检索器。

**问答链路**端到端可用：`infrastructure/llm.py`（OpenAI 兼容协议客户端）、
`rag/prompts.py`（含 `format_context` 的来源页码标注）、`rag/nodes.py`（五个节点）、
`rag/graph.py`（完整图 + 检索子图）、`services/chat_service.py`（一次性与流式两条路径）、
`api/v1/chat.py`（`/chat` 与 SSE `/chat/stream`）。

**防幻觉兜底**：`rag/grounding.py`（生成后核验硬事实：数字/型号/日期是否真在
上下文里）+ 两处闸门（重排整批最高分、提示词 6 条禁令），分档与漏检面见
[`anti-hallucination.md`](anti-hallucination.md)。

**没有剩余的占位实现**——`app/` 下已无 `NotImplementedError`。

全链路按阶段串成一条线的走查（含实测坑与自检清单）见
[`pipeline.md`](pipeline.md)。

### 10.2 实现时对骨架注释做的调整

骨架注释是先于实现写的，落地时发现几处需要修正或明确，记录如下。

| 位置 | 骨架原意 | 实际实现与理由 |
|---|---|---|
| `parsing.py` DOCX 页码 | 按分页符或固定行数**估算**页码 | 不做估算，整体记为第 1 页。DOCX 是流式文本，没有可计算的页边界，"按固定行数估算"得到的是**编造的页码**，会让引用溯源显示假信息。诚实写第 1 页更有用；需要精确页码应转 PDF 再传 |
| `parsing.py` 扩展名 | `.docx` / `.doc` | 只保留 `.docx`。`.doc` 的文本抽取要么依赖 `unstructured`（额外 25 个包）、要么依赖 Word COM（只在装了 Office 的 Windows 可用），两种都会让依赖清单变脏；在扩展名白名单外直接拒绝、提示"另存为 .docx"更清楚 |
| `parsing.py` 的职责边界 | 单个模块内部自己调 pypdf / python-docx | 拆成两层：`loading.py` 用 LangChain Loader 负责"文件 → Document"，`parsing.py` 只做"Document → Page"的纯映射。好处是页码规则可脱离磁盘与模型单测 |
| `splitting.py` 先拼接再切 | 把所有页拼成一整串，切完再用字符区间反查页码；块可能跨页 | 改为**按页独立切分**，页码由 `create_documents(metadatas=...)` 的元数据传递给出。跨页块消失（每块必然只属于一页），手工偏移算术整体删除，引用精度反而提高 |
| `loading.py` 文本编码 | 用 `TextLoader` 的 `autodetect_encoding=True` 自动探测 | 实测在 chardet>=5.2 上直接抛 `TypeError`（见第 8 节）。改为固定候选编码表逐个试解，命中后 `encoding` 显式传入——行为确定，且与 `-X utf8` 等启动参数无关 |
| `loading.py` PPTX 加载器 | 用 `UnstructuredPowerPointLoader` | 自研 `PptxLoader`（继承 `langchain_core` 的 `BaseLoader`，基于 python-pptx）。理由见 5.5 |
| `vectorstore.py` 空页码 | 未指定 | 用 **-1 哨兵**而非 NULL。可空字段在不同 pymilvus 版本上行为不一致，哨兵值最稳 |
| `vectorstore.py` 字段超长 | `text[:65535]`（按字符截断） | Milvus 的 `max_length` 按 **UTF-8 字节**计（实测 30000 汉字=90000 字节被拒），按字符截断在中文下等于没截。改为 `_clip_utf8` 按字节截断且不切断多字节序列；同时给此前**完全没有护栏**的 `filename`（上限 512 字节）补上——它是用户可构造的，超长会让整篇文档入库失败 |
| `vectorstore.py` 写入方式 | 所有行一次性 `upsert` | 按 `MILVUS_UPSERT_BATCH_SIZE`（默认 500）分批，末尾只 flush 一次。请求体与峰值内存随文档规模线性增长（三万块 ~120 MB），且分批才有入库进度可报 |
| `vectorstore.py` ID 校验时机 | 只在校验过滤表达式时校验 | 前置到 `upsert`：带引号的 `kb_id` 能写进去却无法被 `delete_by_kb` 命中，会变成永远删不掉的残留 |
| `vectorstore.py` 异常语义 | `search()` 整体套 `try` | `_checked_id` 必须放在 `try` 外——否则"调用方传错 ID"(400) 会被包成"上游故障"(502)。`upsert`/`search` 的 `MilvusException` 统一转 `RetrievalError`，避免漏成 500 而丢掉原因 |
| `vectorstore.py` 可见性 | "注意 Milvus 的可见性延迟" | 写入后**显式 flush**，检索时用 `min_score` 过滤。不 flush 会出现"刚传完立刻搜不到"的迷惑现象 |
| `documents.py` 流式写盘 | 用 UploadFile 流式写盘 | 按 1 MB 分片累加读取，且**读取时就封顶**（最多读 `max_upload_bytes + 1`）。因为流水线签名要求完整 `bytes`，落盘只是顺带留档；多读的 1 字节让体积校验只在服务层实现一次 |
| 批量上传 HTTP 状态 | 未声明 `status_code` | 保持默认 **200**（单文件上传是 201）。批量结果是"汇总报告"而非单个新资源，与骨架声明一致未做改动 |
| `db/repositories.py` `doc_count` | "冗余列 + 重算"或"实时 COUNT" | 选**实时 COUNT**，不落冗余列，避免与 documents 表不一致 |
| 向量维度的来源 | 未指定（骨架里 `MILVUS_DIMENSION` 同时被用作建表维度） | 改为**权重为权威**：`BgeEmbedder` 从权重 `config.json` 的 `hidden_size` 解析维度，`MILVUS_DIMENSION` 降级为声明值并交叉校验，不符则启动即抛错；`main.py` 用 `embedder.dimension` 建表。理由见 5.2 |
| 编码批大小的归属 | 流水线里用 `_EMBED_GROUP_SIZE` 手工分组 | 下沉到向量化适配器（`EMBEDDING_BATCH_SIZE`）。批大小是模型自己的事，流水线持有第二个 64 只会漂移且误导 |
| 向量化是否套 LangChain `Embeddings` | 骨架按"LangChain 全家桶"来设想 | **不套**。LangChain 的 HF 封装硬依赖 sentence-transformers（已卸载）且已 deprecated，套一层只有命名收益、没有功能收益。向量化直接对接 FlagEmbedding，端口 `EmbedderPort` 保持框架无关 |
| `vectorstore.py` 的 `search()` | 只传 `data` + `filter`，不指定 `anns_field` | 现在必须显式传 `anns_field="embedding"`。加 `sparse` 字段前 collection 里只有一个向量字段，不传也能跑；有两个之后 Milvus 报 `multiple anns_fields exist`（见 5.7）。`check_retrieval.py` 抓出的真实缺陷 |
| `vectorstore.py` 的 schema | 9 个字段，无稀疏字段 | 加 `sparse`（SPARSE_FLOAT_VECTOR）+ `text` 开 analyzer + BM25 索引 + `Function`。这三项都是**建表属性、无法原地补齐**，因此 `ensure_collection()` 对旧 collection 会显式报错并逐条列出缺什么，**不静默重建**（同"维度不符"的处理原则） |
| `ensure_collection()` 的校验范围 | 只校验向量维度 | 追加"是否具备全文检索能力"三项校验。判断依据来自实测：`describe_collection()` 里 analyzer 配置**不在字段顶层**，而是压在 `field["params"]` 下且值是**字符串**（`enable_analyzer='true'`）；稀疏索引按 `metric_type == "BM25"` 认，比按索引名猜稳 |
| `retrieval_service.py` 的构造签名 | `__init__(embedder, vector_store, reranker)` | 改为 `__init__(retriever: HybridRetriever, reranker)`。收一个装配好的检索器而不是两个端口，是因为 `retrieve` 节点也要用同一份检索逻辑——两个消费者各自构造 `HybridRetriever` 会留下"融合参数悄悄不一致"的隐患 |
| `nodes.py` 的 `make_retrieve_node` | `(embedder, vector_store)` | 改为 `(retriever: HybridRetriever)`，理由同上 |
| `retrieval_service.retrieve()` 的返回 | `list[RetrievedChunk]` | 追加 `retrieve_with_trace() -> (RecallTrace, list[RetrievedChunk])`。检索接口需要中间态；若让调用方先 `recall()` 再 `retrieve()`，Embedding 前向与两路检索会**各做两遍**（自检里有回归断言盯着这一点） |
| 端口 `VectorStorePort` | 只有 `search()` | 追加 `search_bm25()`。约定：底层不支持全文检索的实现返回空列表而不是抛错，让上层退化为纯向量检索 |
| `collection_service.py` | 各处内联 `if get_knowledge_base(...) is None: raise NotFoundError` | 抽出 `ensure_exists(kb_id)`。检索路由也需要"知识库不存在就 404"，而它并不想顺带把文档列表查出来——为校验而拉全量数据是明显的浪费 |
| 检索是否走 LangGraph | 骨架里 retrieve 节点是图的一部分 | 检索逻辑放 `rag/retrieval.py`，由 `services/retrieval_service.py` 与 `nodes.py` 共用。图还没编译，但检索不必等图——先让检索可用且可单独验证，反而是更稳的顺序 |
| `graph.build_rag_graph` 的签名 | `(embedder, vector_store, reranker, llm)` | 改为 `(retriever: HybridRetriever, reranker, llm)`，理由同 `retrieval_service`：图与检索接口必须共用同一个检索器实例 |
| 流式路径的图 | 骨架建议"先在图中完成检索，再由本服务直接调用 `llm.stream()`"，并把它列为方案 (b) | 采纳 (b)，落地为**两张编译好的图**：`build_rag_graph`（完整，含改写重试）与 `build_retrieval_graph`（`retrieve → rerank`，供流式）。token 只能来自 `llm.stream()`，图节点返回的是最终字段值而不是增量，所以两条路只能这么分。流式改写重试**不放在图里**：重试时还要在改写发生的当下推一个 `event: retry` 给前端，而图跑完才返回最终状态、插不进这条时间线。所以它在服务层（`ChatService.ask_stream`，按请求 `enable_retry` 开启，默认关），但判据与上限复用图里的 `route_after_rerank` / `MAX_RETRY`，改写实现复用 `rag/rewrite.py`——**分的是编排，不是语义**。见 `chat_service.py` 的模块 docstring |
| `route_after_rerank` 的判据 | 注释写"无可用片段"，未明确以哪个字段为准 | 明确为 **`ranked`** 而不是 `candidates`。「召回到了」与「召回的能用」是两件事：重排可能把全部候选判为不相关（实测过），此时 `candidates` 非空、`ranked` 为空，拿 `candidates` 去生成等于让模型在无关文字里"找"答案 |
| `ChatService.ask()` 的返回 | `Answer` | 改为 `ChatResult`（含 `answer` / `sources` / `query` / `retried`）。接口的 `sources` 与 `citations` 是两份不同用途的信息，只回 `citations` 会让"检索召回了但被重排丢了"彻底不可见 |
| LLM 的异常语义 | 骨架只写了"连接拒绝/超时 → 抛 `UpstreamError`" | 补上一条：**`AppError` 必须先原样放行**。`_build_client()` 抛的 `ConfigurationError` 若被兜底包成 `UpstreamError`，运维会去重启 Ollama，而真正该改的是 `.env`。这条是 `check_chat.py` 抓出来的真 bug |
| LLM 的空回复 | 未指定 | 视为**上游异常**并抛 `UpstreamError`。空串不是"没有答案"，而是截断 / 内容过滤 / 后端模型被卸载；返回空串会让用户看到一条空回答，日志里也查不出所以然 |
| `ChatRequest` 的 `kb_id` 位置 | 骨架未定 | 放**请求体**而不是 URL 路径（`/chat` 而非 `/knowledge-bases/{kb_id}/chat`）。将来支持多库问答时改的是请求体字段，不必动 URL 结构 |
| `schemas/chat.py` 的 `SourceOut.score` | `score` | 改名 `vector_score`，与 `schemas/retrieval.py` 对齐。同一条片段在 `/search` 与 `/chat` 里必须显示同一套字段名，否则前端要写两套解析、调参时容易读错口径 |
| `state.py` 的拒答判定 | 注释提到"必须与 `prompts.py` 保持一致，便于上层做字符串判定" | 落地为**包含**判定（`NO_ANSWER_TEXT in answer`）而不是相等判定：模型常给固定话术加前后缀（"抱歉，无法从知识库中找到…"），要求逐字相等会把这些情况误判成"已作答" |
| 提示词里的拒答话术 | 骨架的 `ANSWER_USER` 把话术**硬编码**了一遍 | 改为 `{no_answer}` 占位、从 `state.NO_ANSWER_TEXT` 取值。话术写两处必然漂移，而漂移的后果是判定失效（模型嘴上说找不到、系统记成已作答） |
| 文档类失败的异常类型 | 一律 `AppError`（500） | 新增 `DocumentError`(400) 与三个子类：`CorruptedDocumentError` / `EncryptedDocumentError` / `EmptyDocumentError`。三者**处置动作完全不同**（重新导出 / 解密码 / 上 OCR），所以 `code` 必须能分支，不能用一个 `document_error` 糊过去。此前同一份坏 PDF 在 `/documents` 上传里是 `status=failed`（看着对），走到脚本或批处理入口就变成 500，运维去翻服务日志，而真正的动作是「让用户重新导出这个文件」。文件不存在同时改为 `NotFoundError`(404)——路径错 ≠ 文件内容错 |
| 损坏文件的识别方式 | 按 `pypdf` / `python-docx` 的具体异常类型分支 | 改为**双特征识别**：异常类名含 `badzip/pdfread/emptyfile/packagefound…`，或消息含 `eof marker/xref/not a zip file/truncated…`。理由：同一个"文件坏了"在不同加载器、不同版本里抛的类都不一样，绑死类名等于每升一次依赖就漏一类。加密单独一类（提示"先解除密码保护"），因为让用户"重新导出"是错的建议 |
| 向量化失败的错误语义 | 未包装，torch 异常直接冒到兜底处理器 | 统一包成 `EmbeddingError`(502，继承 `RetrievalError`)。三层校验：① 一次前向的所有异常（OOM/cuda 附 `EMBEDDING_BATCH_SIZE` 提示）；② 输出条数与输入不符（会让块与向量**静默错位**，检索"能跑但答非所问"）；③ 逐向量维度与 NaN/Inf（NaN 会对任意查询算出可疑相似度，污染索引）。`AppError` 仍原样放行——`ConfigurationError` 是"该改 `.env`"，不能被包成"该查显存" |
| LLM 超时的状态码 | 与 `UpstreamError` 混用 502 | 新增 `UpstreamTimeoutError`(504)。**502 与 504 的处置动作不同**：502 是"上游没起来/崩了"，动作是重启它；504 是"在跑但太慢"，动作是调大 `LLM_TIMEOUT`、减 `RERANK_TOP_N`、或换小模型，且**客户端重试是安全的**。识别不绑 SDK：先判 `TimeoutError`，再看类名/消息是否含 `timeout/timed out/time out`（反代常返回普通 `Exception("request timed out")`） |
| RRF 融合的去重 | 只按 `chunk.id` 去重（同一条片段在两路出现） | 追加**跨副本去重**：按 `(filename, 空白归一化正文)` 合并重复上传的副本，**先于 `top_k` 截断**执行。实测同一 txt 传 4 次时，向量路给出逐位相同的相似度，融合后 4 条并列占掉 `top_k=8` 的一半名额。计数进 `RecallTrace.duplicates_removed`，否则"召回变好了"无从归因。只合并同名文件——两篇文档引用同一段公告是合法重复 |
| `fuse_detailed` 抛出的 `ValueError` | 直接冒到上层 | `HybridRetriever._fuse()` 包成 `RetrievalError`(502)。`ValueError` 在这里表示"**上游**给了未排序结果"（Milvus 违约），不是调用方传错参数；裸 `ValueError` 会变成 500「服务内部错误」，看不出该查哪里 |
| BM25 阈值配成 0 时怎么下发 | 直接把 0 传给 Milvus | **显式转 `None`**。BM25 分永远 ≥ 0，传 0 等于"过滤负分"——看着像设了阈值其实什么都没做。这种"以为设了"的错觉比不设更麻烦，所以宁可显式不传 |
| 防幻觉提示词的写法 | 抽象要求（"不要编造""只依据资料"） | 改为 **6 条编号禁令**：只用明写信息 / 数字型号逐字一致 / 禁止跨段拼接成新结论 / 资料没提的原因·建议·后续不写 / **允许留白但不给句式**（"只写覆盖的那部分"）/ 无法回答时原样回 `{NO_ANSWER_TEXT}`。7B 对**可逐条核对的具体禁令**遵守度显著高于散文式要求。动机是 q14 把「压力前放量区下轨」答成「箱体上轨的下方」 |
| 「允许部分回答」怎么写 | "其余明确写「资料未说明」"（system 与 user 各写一遍） | **删掉可照抄的占位话术**，只写"只写资料覆盖的那部分、不补资料没有的内容"。7B 会把这句占位话当成**答案本身**：q05 参考资料明写"48 小时内首次响应"，模型整题只回"资料未说明。"（2×2 因子实验 5/5；删掉 system 或 user 任意一半即 0/5）。留白可以，**留句式不行** |
| 怎么知道模型的回答是否接地 | 未做 | 新增 `_audit_answer()`：正则 `\[(\d{1,3})\]` 取回答里的来源序号，核对是否落在本轮资料范围内。**只告警不改写**——改文字会把"模型没遵守提示词"这件事洗掉，而那正是要观测的对象。结果进 `ChatResult.grounding_warnings` |
| 入库失败后怎么重试 | 注释写着"原始文件也保留，便于重试"，但**没有任何入口**——用户只能重传 | 新增 `POST /documents/{doc_id}/reprocess`：读 uploads 里那份原件原地重跑，`doc_id` 不变、不新建记录。重传的代价是实的：生成第二个 doc_id（列表里一条 failed + 一条 ready），并把同一份正文的向量写两遍（实测同一份 txt 攒到 4 条 ready，检索时互相挤占名额）。`run()` 与 `reprocess()` 共用 `_process()`，否则重试路径会慢慢与首次入库漂移（忘清理残留、忘回填归属字段） |
| 重试的准入条件 | 未定义 | 只允许 `failed → processing`。`processing` 拒绝是因为"同一份文档两个线程写向量"，`ready` 拒绝是因为那是**重建索引**而非重试：重建需要先清空再写，清空到写入完成之间检索不到这份文档，而失败文档本来就没向量、不存在这个窗口。两种语义混在一个接口里，用户永远分不清自己点的是哪个。原始文件丢失 → 404 并提示"请重新上传"（没有文件就真的没法重试） |
| 上传体积怎么限 | 只有 `MAX_UPLOAD_MB`，且在路由读过整个 body 之后才判 | **三道闸门**：① HTTP 中间件按 `Content-Length` 回 413（Starlette 解析 multipart 时会先把整个 body 写进临时文件，不提前拦就等于"先写满磁盘再说超限"）；② `_read_upload` 最多读 `max+1` 字节（防伪造的 Content-Length）；③ 服务层 `_validate` 是唯一可信边界（脚本/内部调用也走它）。批量另有 `MAX_BATCH_FILES` / `MAX_BATCH_TOTAL_MB`，累计超限时**立刻停读** |
| 恶意文件怎么过滤 | 只有扩展名白名单，改个后缀即可绕过 | `infrastructure/filesafety.py` 纯函数：文件名控制字符 → 魔数（`.pdf` 前 1KB 内 `%PDF-`；docx/pptx 必须是 zip）→ PDF 页数 → zip 条目数/解压总量/单条压缩比 → 文本 NUL 与控制字符占比。失败抛 `UnsafeFileError`（400 + `code=unsafe_file`）。**压缩比判定带 1MB 绝对体积下限**：实测正常 docx 压缩比可达 346:1，只看比例会误杀 |
| 多人并发怎么保证不重复入库 | 无：同一份 txt 曾攒到 5 条 ready，向量分逐位相同，在 top_k 里互相挤占 | `documents.content_hash` + **部分唯一索引**（`WHERE content_hash != ''`，老数据 hash 为空不参与约束）。服务层先按 `(kb_id, hash)` 查重 → 409 + 已有 `doc_id`；并发时应用层查重会双双扑空，由唯一索引兜底（`IntegrityError` → 409）。去重**不看文件名**：改名不换内容算重复，同名不同内容必须放行 |
| 并发写同一个文档 / 同时向量化多份 | 状态检查与置 processing 是两次独立读写，可双双重试；多份大文件同时编码只会让尾延迟一起变差 | `DocumentLockPool`（按 key 的 `asyncio.Lock`，引用计数回收）+ `ConcurrencyGate`（`INGEST_MAX_CONCURRENCY`，默认 1 = 排队）。**都是进程内**的：多 worker 需要 Redis 锁，因此部署建议入库单 worker、检索侧多开 |
| 数据怎么备份 | 无 | `scripts/backup.py`：`VACUUM INTO` 拿一致快照（`cp .db` 会漏掉未 checkpoint 的 WAL，甚至复制出撕裂文件）+ 只打「元数据登记过」的 uploads + `manifest.json`（sha256/行数/状态分布）。**不备 Milvus 与 Redis**：向量可从原始文件重算，会话是 TTL 数据。恢复后把 ready 置 failed 并 `rebuild_vectors.py` 重建——标 ready 却没向量就是撒谎 |

### 10.3 新增的文件

- `app/services/mappers.py` — 领域模型 → DTO 的转换。
  两个服务都要把 `Document` 转成响应结构，集中一处避免重复。
- `app/infrastructure/loading.py` — 文档加载层（LangChain Loader → `Document`）。
  骨架里没有这个文件，加载与解析合在 `parsing.py`；实现时按"是否依赖框架"拆开（见 5.5）。
- `scripts/make_sample_docs.py` — 生成 5 种格式的最小样例到 `tests/fixtures/`。
  PDF 是脚本自己拼的最小 PDF（base-14 字体，不引入额外绘图库）。
- `scripts/check_loading.py` — 文档加载自检，45 项断言覆盖页码、编码、
  表格完整性、以及空文件/损坏/加密/扫描件等失败语义。`make_sample_docs.py`
  产出的样例是它的输入之一（缺样例时它会自动生成）。
- `scripts/check_splitting.py` — 解析 + 切分自检，94 项断言覆盖页码溯源、
  逐页独立（不跨页）、`chunk_index` 连续性、内容无丢失、无捏造（块必须是本页子串）、
  块大小、相邻块重叠、空页不串位、越权检查（切分器不填 id/embedding）。
  它自建临时样例目录，不依赖 `tests/fixtures` 是否已生成。
- `scripts/check_embedding.py` — 向量化自检，63 项断言。分三层：
  ① **契约层**（维度来源与交叉校验、形状、L2=1、确定性、检索指令只加在 query 侧）；
  ② **边界层**（空列表短路、空串/纯空白拒绝、跨批一致性按容差比对）；
  ③ **步骤层**——用**真 embedder + 假 parser/splitter/仓储/向量库**跑通
  `IngestionPipeline` 的第 5 步，断言跨批次块全部回填向量、向量与文本一一对应
  （不错位），以及三类失败语义（向量库写入失败 → failed + 清理残留、
  切分为空 → failed、上游漏出空白块 → failed）。
  假实现全部用 `runtime_checkable` 端口做 `isinstance` 断言，
  这样端口契约变了它也会跟着红——不需要真的连 Milvus 或写 SQLite。
- `scripts/check_ingest.py` — 入库状态机自检，51 项断言。**全假上游**，不连 Milvus、
  不读 BGE 权重、不碰真实 uploads：
  ① 首次入库（ready / chunk_count / 片段归属回填 / 落盘路径与内容逐字节一致）；
  ② 首次失败（`[corrupted_document]` 前缀、残留向量被清理、**元数据行与原始文件都保留**）；
  ③ **原地重试**（同一 doc_id 从 failed → ready、`create_document` 调用次数不变、
  parser 收到的是上次落盘那一份、重试前再清一次残留）；
  ④ 准入条件（文件丢失 → 404、processing / ready → 409、不存在 → 404，且都**不触发解析**）；
  ⑤ 重试再失败（error 换成新原因、旧原因不留、doc_id 不变）；
  ⑥ 路径安全（同名文件按 doc_id 区分、`../` 被剥离、`kb_id` 带 `..` 被清洗后仍在 upload_dir 内）
  与异常语义（`AppError` 带 code、非 `AppError` 保留类名）。
- `scripts/check_vectorstore.py` — 向量库（入库）自检，84 项断言。分三层：
  ① **契约层**（`isinstance` 端口、schema 九个字段的类型与上限、索引 metric、
  维度不符抛 `ConfigurationError` 且**不重建**）；
  ② **行为层**（写入条数、flush 后立即可见、同 id 幂等覆盖、跨批一条不丢不错位、
  字节截断与边界值、unicode/filename/页码哨兵往返、COSINE 排序、kb_id 过滤隔离、
  `min_score` 与 `top_k`、`delete_by_doc` / `delete_by_kb` 后立刻检索不到）；
  ③ **步骤层**——**真 Milvus** + 假 parser/splitter/embedder/仓储跑
  `IngestionPipeline` 第 6 步，断言向量真落库、页码与正文对应，以及
  **部分批成功后再失败时残留行被清理干净**（用错误维度的向量触发真实写入失败）。
  **全程使用独立临时 collection（`rag_backend_check_<pid>`）并在结尾删除，
  最后一条断言复核真实集合行数前后一致**——自检绝不能碰到真实数据。
- `scripts/init_milvus.py` — 骨架期只是个抛 `NotImplementedError` 的占位，
  但 README 一直把它当运维脚本宣传，且"换模型后重建 collection"的正常出路
  就落在它身上（维度无法原地变更）。现已实现：维度取自权重（不是配置字面量）、
  打印 server 版本与字段/索引/行数统计、`--recreate` 需显式确认
  （终端手输 `RECREATE`，非交互必须 `--yes`，否则拒绝并保持原状）。
  为它给 `MilvusVectorStore` 加了 `client` / `drop_collection()` 两个
  **明确标注"仅供运维脚本"** 的接口——它们不属于 `VectorStorePort`：
  `delete_by_doc` 是"删数据"，`drop_collection` 是"删表"，不该进业务端口。
- `app/rag/fusion.py` — 加权 RRF 融合，**纯函数、零外部依赖**（只用标准库 +
  `app.domain.models`，不 import pymilvus / langchain），因此可以脱离数据库与模型
  做纯逻辑单测。内含"输入必须按自身分数降序"的断言——排名取自列表位置，
  上游忘了排序时 RRF 会安静地算出完全错误的融合分，而且结果看上去"也像那么回事"，
  极难排查，宁可在入口顶回去。
- `app/rag/retrieval.py` — `HybridRetriever` 与 `RecallTrace`。
  `recall()` 返回两路原始结果 + 融合结果，是调参的入口；
  `retrieve()` 只是它的 `.fused`。BM25 路失败时**降级为纯向量检索**
  （记 exception 日志交给运维），不让关键词路的故障打断整个问答。
  权重为 0 的那一路直接跳过，连网络请求都不发。
- `app/api/v1/retrieval.py` + `app/schemas/retrieval.py` — 检索接口与 DTO。
  四个分数（`vector_score` / `bm25_score` / `fusion_score` / `rerank_score`）
  **允许缺省且不做零填充**：缺省表示"这一路没召回它"，与"召回了但得 0 分"
  是两件事——前者改召回策略，后者改模型或语料，修法完全不同。
- `scripts/check_retrieval.py` — 混合检索自检，128 项断言。分六层：
  ① **契约层**（端口含 `search_bm25`、schema 的稀疏字段/analyzer/BM25 索引、
  **旧 schema 漂移必须报错且不重建**）；
  ② **纯函数层**（RRF 公式、去重、权重比值语义、`k`、`top_k`、一路为空、
  权重为 0 跳过，以及 k<1 / 负权重 / 未排序输入 / 分数缺失 / id 为空五类护栏）；
  ③ **行为层**（BM25 中文分词命中、kb 隔离、空查询短路、脏 kb_id 抛
  `ValidationError`、停用词噪声的真分数记录、`min_score`）；
  ④ **编排层**——含本脚本的**核心断言**：构造一篇"关键词命中但向量正交"的文档，
  断言它在纯向量下召不回、在混合检索下召得回；另有单路跳过、
  BM25 失败降级、`retrieve_with_trace` 只跑一次召回的回归断言，
  以及 `make_retrieve_node` 的行为；
  ⑤ **跨副本去重**（`check_fusion_dedupe`）：4 份逐字相同的副本合并成 1 条、
  `duplicates_removed==3`、留分最高、跨文件不合并、空白归一化后才算同一篇、
  空正文不参与、融合分单调；
  ⑥ **失败语义**（`check_error_semantics` / `check_rerank_failures`）：未排序输入 →
  `RetrievalError`；`bm25_min_score=0` → 显式传 `None`；重排 OOM / 条数不符 /
  NaN → `RerankError`(502)，而 `ConfigurationError` 必须**原样透传**（不能兜成 502）。
  用确定性的假 embedder / 假重排器，**不需要 GPU 与 BGE 权重**，
  全程只用独立临时 collection 并在结尾复核真实集合行数不变。

**问答链路新增的文件**（第二轮）：

- `app/rag/prompts.py` 的 `format_context` / `build_answer_prompts` /
  `build_rewrite_prompts` — 参考资料标注与提示词组装。标注带 `[序号] 来源：文件名 第N页`：
  模型只有看到页码才可能在回答里引用，而接口的结构化 `citations` 来自同一批片段，两者对得上。
  组装函数单独抽出（而不是在节点里拼字符串）是因为流式与非流式两条路径都要用，
  两处各写一遍必然漂移（典型症状是"流式回答不带页码、非流式带"）。
- `app/infrastructure/llm.py` 的 `OpenAICompatibleLLM` — `ChatOpenAI` 封装。
  **延迟建连**：`get_llm()` 构造期不建 client，所以"服务能不能起来"不取决于 Ollama
  是否在跑。`_messages()` 统一做"两条字符串 → 消息对象"的转换，业务侧仍只传字符串。
  留了 `client` 注入点（明确标注为测试缝），`check_chat.py` 靠它覆盖错误映射全部分支。
- `app/services/chat_service.py` 的 `ChatResult` — 领域 `Answer` + `sources` 明细。
  `_source_dict` / `_citation_dict` 直接复用 `mappers.chunk_preview` 与
  `Chunk.page_label`，保证与检索接口的展示口径一致。
- `scripts/check_chat.py` — 问答链路自检，**246 项断言、完全离线**
  （假 embedder / 内存向量库 / 假重排器 / 假 LLM，不需要 Milvus、BGE 权重或 Ollama）。
  十组：页码标签、参考资料标注、提示词口径一致性、路由三分支、图的三条路径、
  检索子图、`ChatService` 引用映射与失败语义、LLM 消息组装与错误映射、
  SSE 报文与事件顺序、HTTP 接口（含 404/422 与四种流式场景）。
  内存向量库是**真的会排序**的实现（余弦 + 2-gram 关键词），不是"返回预置列表"——
  否则融合与重排的输入顺序就不是真实产出的顺序，自检会失去意义。
- `scripts/ask.py` — 单知识库问答 CLI。走 HTTP 而不是进程内直调：这是唯一会同时走通
  接口层、服务层、状态图、混合检索、重排与本地大模型的路径。`--demo` 会造夹具、
  入库并**自动校验每题的引用页**，是可执行的验收脚本。
- `scripts/make_sample_docs.py` 新增 `build_handbook()` — 5 页中文手册 PPTX，
  每页一个互不重叠的事实、关键数字只出现一次，专门用于**证伪**页码引用
  （复用 `sample.pptx` 不行：那三页共享大量词元，三个片段都可能被召回，
  "引用了第 2 页"这句话就没有信息量了）。

### 10.4 一处主动加固

`vectorstore.py` 在连接 Milvus 前会检查 URI 是否为本机地址，
是则把 `127.0.0.1` / `localhost` 补进 `NO_PROXY` 并打一条 warning。
第 8 节的代理陷阱在实际使用中太容易踩（表现是一条不可读的 502），
值得在代码里兜一层，而不是只写在文档里靠人记得设环境变量。
