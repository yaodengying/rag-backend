# 数据流

本文描述两条核心链路的完整时序，以及每步的输入、输出与失败处理。

---

## 链路一：文档入库

入口：`POST /api/v1/knowledge-bases/{kb_id}/documents`

```
客户端
  │  上传文件（multipart）
  ▼
api/v1/documents.py ── 扩展名白名单、体积校验
  │
  ▼
services/ingest_service.py
  │  校验知识库存在
  ▼
rag/ingest.py  IngestionPipeline.run()
  │
  ├─ 1. 登记元数据       db/repositories.py
  │     INSERT documents(status='processing')
  │     → 拿到 doc_id
  │
  ├─ 2. 原始文件落盘      data/uploads/{kb_id}/{doc_id}_{filename}
  │     注意：文件名带 doc_id 前缀防重名；防目录穿越
  │
  ├─ 3. 加载            infrastructure/loading.py
  │     PDF/DOCX/PPTX/TXT/MD → list[Document]（LangChain Loader）
  │     编码、非空校验、异常包装都在这一步
  │     失败 → 抛 AppError（空文件 / 加密 / 损坏 / 扫描件无文本）
  │
  ├─ 3b. 映射           infrastructure/parsing.py
  │     Document.metadata["page"] (0 起) → Page.page_no (1 起)
  │     无 page 元数据的格式（DOCX/TXT/MD）统一记为第 1 页
  │
  ├─ 4. 切分            infrastructure/splitting.py
  │     list[Page] → create_documents(逐页文本, metadatas=[{page_no}])
  │     → list[Chunk]（含 chunk_index）
  │     页码溯源由 LangChain 的元数据传递给出：每块必然且只属于一页
  │     （page_start == page_end，不产生跨页块）
  │
  ├─ 5. 向量化          infrastructure/embedding.py（FlagEmbedding）
  │     list[Chunk] → 回填 embedding（L2 归一化，维度取自权重 config.json）
  │     注意：文档块走 embed_documents()，不加检索指令；
  │     分批由适配器自理（EMBEDDING_BATCH_SIZE，上限 64），流水线只传整份列表
  │     空/纯空白文本直接失败，不写入无语义的 padding 向量
  │
  ├─ 6. 写入向量库      infrastructure/vectorstore.py
  │     Milvus upsert(chunks)，按 MILVUS_UPSERT_BATCH_SIZE 分批
  │     字段护栏：text ≤ 65535 **字节**、filename ≤ 512 **字节**（按字节截断）
  │     ID 字符集校验前置（脏 ID 能写进去就删不掉）
  │     全部批写完后 flush 一次 → 写完即可检索
  │     失败 → RetrievalError → 步骤 7 标记 failed 并清理已写入的残留行
  │
  └─ 7. 更新元数据
        UPDATE documents SET status='ready', chunk_count=N
  │
  ▼
返回文档 DTO（status / chunk_count / error）
```

### 失败处理

任何一步抛错，都必须执行「兜底」：

```
except Exception as e:
    repository.mark_document_failed(doc_id, str(e))
    vector_store.delete_by_doc(doc_id)      # 清理可能已写入的部分向量
    raise
```

两条原则：

1. **绝不留下 `processing` 僵尸记录**——前端会一直转圈，用户不知道发生了什么。
   元数据行要保留（让用户看到失败原因），原始文件也保留（便于重试）。
2. **清理半成品向量**：第 6 步可能只写了一半就失败，
   残留的向量会让后续检索召回不完整的片段。

### 元数据与向量的分工

| 数据 | 去向 | 备注 |
|---|---|---|
| 文档元数据 | SQLite `documents` 表 | 名称、状态、片段数 |
| 原始文件 | `data/uploads/` | 正文权威副本 |
| 片段正文 | Milvus `text` 字段 | **不写 SQLite** |
| 片段向量 | Milvus `embedding` 字段 | **不写 SQLite** |

---

## 链路二：RAG 问答

入口：`POST /api/v1/chat`（非流式）或 `POST /api/v1/chat/stream`（SSE）

```
客户端
  │  { kb_id, question, history, top_k, rerank_k }
  ▼
api/v1/chat.py ── Pydantic 校验
  │
  ▼
services/chat_service.py
  │  组装初始 RAGState
  ▼
rag/graph.py  (LangGraph 编译产物)
  │
  │  ┌──────────────┐
  ├─►│   retrieve   │  向量召回 + BM25 召回 → RRF 融合
  │  └──────┬───────┘  写 state.candidates
  │         ▼          内部调 rag/retrieval.py 的 HybridRetriever，
  │                    与 /search 接口是同一份实现（见「链路二·五」）
  │  ┌──────────────┐
  │  │    rerank    │  交叉编码器精排，截断 top_n
  │  └──────┬───────┘  写 state.ranked
  │         │
  │    ┌────┴─────────────────────────┐
  │    │ ranked 非空                   │ ranked 为空
  │    ▼                              ▼
  │  ┌──────────────┐         ┌───────────────┐
  │  │   generate   │         │ retry_count   │
  │  │  组装提示词   │         │  < MAX_RETRY? │
  │  │  调用 LLM    │         └───┬───────┬───┘
  │  └──────┬───────┘           是 │       │ 否
  │         │                     ▼       ▼
  │         │              ┌──────────┐ ┌────────┐
  │         │              │ rewrite  │ │ refuse │
  │         │              └────┬─────┘ └───┬────┘
  │         │                   │           │
  │         │                   └─► retrieve│
  │         ▼                               ▼
  │      state.answer                  state.answer
  │      state.refused                 = NO_ANSWER_TEXT
  │         │                          refused=True
  └─────────┴───────────────┬───────────┘
                            ▼
              组装 Answer + Citation 列表
                            │
                            ▼
                    返回 ChatResponse / SSE
```

### 为什么「无检索结果」要短路拒答

无检索结果时**不调用大模型生成**，直接返回固定话术。理由：

1. 省一次昂贵推理；
2. 更重要的是——**一旦把空上下文交给模型，它就获得了自由发挥的机会**，
   这是 RAG 幻觉的主要来源。宁可拒答，不可编造。

注意「无检索结果」有两个层次，都会走到拒答，但在日志里必须能区分：

| 情况 | 现象 | 排查方向 |
|---|---|---|
| 两路都没召回 | `candidates` 为空 | 库里确实没有；或 `RETRIEVE_TOP_K` / kb 过滤有问题 |
| 召回了但全被判为无关 | `candidates` 非空、`ranked` 为空 | 分块过大稀释了相关性（实测：同一句单独成块 +0.63、混进 272 字长块 -3.99） |

路由判据是 **`ranked`** 而不是 `candidates`：拿一堆被判为无关的文字去生成，
正是幻觉的高发场景。`ask.py -k <kb> --debug` 会把两路召回与最终上下文都打出来，
一眼就能分清是哪一种。

### 改写重试的终止条件

`rewrite_query` 只在"`ranked` 为空且 `retry_count < MAX_RETRY`"时被触发，
`retry_count` 每改写一次 +1。所以最多改写一次就会走向 `refuse`，
回路**不可能无限转**（`check_chat.py` 有断言盯着"最多改写一次"与
"全丢路径下重排恰好被调用两次"）。

这条判据是**唯一一份**：图里由 `route_after_rerank` 决定，
流式路径（`ChatService.ask_stream` 的可选重试）直接调同一个函数，
所以两条路的重试次数与收手时机必然一致，不会出现
"流式重试 3 次、非流式 1 次"这种两边各说各话的偏差。

改写用的也是 LLM，而"检索不到"的常见原因之一恰恰是上游模型不稳。
因此改写失败时**降级为用原问题再检索一次**，而不是抛错——
否则一次检索失败会升级成整个请求 500，而用户的真实处境只是"没找到"。
降级实现只有一份，在 `rag/rewrite.py`。

### 流式与非流式走了两张图

| | 非流式 `ask()` | 流式 `ask_stream()` |
|---|---|---|
| 图 | `build_rag_graph`（完整） | `build_retrieval_graph`（`retrieve → rerank`） |
| 含改写重试 | ✅ 恒开（调用方本来就在等完整答案） | 由请求/默认值决定，见下 |
| 生成方式 | 图内 `generate` 节点调 `llm.generate()` | 服务层直接 `llm.stream()` |
| 输出 | `Answer` + 引用 | SSE 事件流 |

为什么不能一张图走到底：**token 只能来自 `llm.stream()`**，
而图节点返回的是**最终字段值**（`state["answer"]` 是一个完整字符串），不是增量。
所以要么放弃增量、要么把生成挪出图。本项目选了后者，并把改写重试放到服务层，
**是否重试交给前端用户决定**：

| 请求体 `enable_retry` | 行为 |
|---|---|
| `true` / `false` | 前端用户的显式选择，服务端一定照办（显式 `false` 压过服务端默认值） |
| 省略 / `null` | 落回 `CHAT_STREAM_RETRY_DEFAULT`（出厂 `false`） |

默认关是为了首帧时间：一次改写要等 LLM 生成一轮查询（本机 7B 约 2~4 秒），
用户此时盯着一个空白窗口。开启时先推一个 `event: retry` 把这段静默填住
（用户看到的是"正在改写查询"，而不是一个静止不动的界面）。
重试的判据与上限仍与完整图同源（共用 `route_after_rerank` 与 `MAX_RETRY`）。

前端还有第二条路——**当场询问**：先按默认出结果，`done.can_retry=true` 时
才问用户要不要「再找一次」，用户点了就用 `enable_retry=true` 把同一个问题
重发一遍。这样用户不必盲签（事先谁也不知道这一问用不用得上改写）。
两条路只是同一个请求字段的不同用法，服务端没有额外分支。
`GET /api/v1/chat/options` 提供默认值/上限/事件名，前端据此初始化开关。

两张图共用同一批节点工厂、同一个 `HybridRetriever` 与 `reranker` 实例
（见 `container.get_hybrid_retriever` 的 lru_cache），
所以"流式看到的来源"与"非流式看到的来源"必然是同一批。

检索子图是**同步**的（嵌入前向是 CPU 密集的），在 `ask_stream` 里用
`asyncio.to_thread` 丢到线程池，否则会卡住整个事件循环，SSE 连接在检索期间
无法响应心跳或断开事件。

### SSE 事件序列

```
event: retry        ← 可选：仅当本次请求决定重试（显式 enable_retry=true，
data: {"type":"retry","attempt":1,"max_attempts":1,
       "reason":"首轮检索无可用片段，正在改写查询"}
                      或未表态而服务端默认为 true）且首轮检索无可用片段
                      位置在 sources **之前**，用来填住改写那 2~4 秒的静默

event: sources      ← 检索完成即推送，前端可先渲染引用卡片
data: {"type":"sources","sources":[...]}

event: delta        ← 生成增量，多次
data: {"type":"delta","content":"五千"}

event: done         ← 结束（成功与拒答两条路径字段完全一致）
data: {"type":"done","answer":"完整回答文本","refused":false,"citations":[...],
       "retried":false,"attempts":0,"can_retry":false,"query":"实际检索用的查询"}
```

`done` 的四个"重试事实"字段，是前端做「当场询问」的全部依据：

| 字段 | 含义 | 前端怎么用 |
|---|---|---|
| `retried` | 是否**真的**用改写后的查询重新检索过 | 显示"已换一种说法重查"；注意它 ≠ "调过改写"（改写无产出时 `attempts=1` 但 `retried=false`） |
| `attempts` | 改写尝试次数 | 排查/日志 |
| `can_retry` | `refused and attempts < MAX_RETRY` | **`true` 才弹「再找一次」按钮**，否则会弹一个点了没用的按钮 |
| `query` | 实际用于检索的查询 | 排查"到底拿什么去查的"；发生改写时与 `question` 不同 |

出错时以 `event: error` 结束。响应必须带：

```
Cache-Control: no-cache
X-Accel-Buffering: no     ← 否则 Nginx 会缓冲整个响应，流式失效
Content-Type: text/event-stream
```

三个实现细节：

1. **拒答也走 delta 通道**（把固定话术当一次增量推出去），
   前端只需处理一种文本通道，不必为拒答写分支。此时 `done.refused=true`、
   `citations` 为空，但 `sources` 仍会先推出去。
2. **出错时先推的 `sources` 不会撤销**。SSE 一旦开始发送就无法改 HTTP 状态码，
   在流里报错只能靠事件体——所以能放在流前的错误（如知识库不存在 → 404）
   都放在建流之前。
3. **`data` 段不得有裸换行**。`json.dumps` 会把正文里的换行转义成 `\n`，天然安全；
   但不要用 `ensure_ascii=True`——中文会被转成 `\uXXXX`，报文膨胀数倍且抓包不可读。

---

## 链路二·五：检索（只检索不生成）

入口：`POST /api/v1/knowledge-bases/{kb_id}/search`

```
客户端
  │  { query, top_k?, rerank_k?, debug? }
  ▼
api/v1/retrieval.py
  │  ① collections.ensure_exists(kb_id)   ← 库不存在 → 404（比"空结果"更容易发现打错 ID）
  ▼
services/retrieval_service.py
  │
  │  ┌─────────────────────────── rag/retrieval.py（HybridRetriever）───────────────────────────┐
  │  │                                                                                        │
  │  │  embed_query(query)  ──►  查询向量                                                       │
  │  │                                                                                        │
  │  │  ┌─────────────────────────┐          ┌──────────────────────────┐                     │
  │  ├─►│ 稠密路 vector_store      │          │ 稀疏路 vector_store       │                     │
  │  │  │ .search(anns_field=     │          │ .search_bm25(query)      │                     │
  │  │  │   "embedding", COSINE)  │          │   → BM25 倒排索引         │                     │
  │  │  │ → vector_score          │          │ → bm25_score             │                     │
  │  │  └───────────┬─────────────┘          └────────────┬─────────────┘                     │
  │  │              │  min_similarity 过滤                  │  不设阈值                          │
  │  │              └──────────────┬───────────────────────┘                                  │
  │  │                             ▼                                                          │
  │  │                  rag/fusion.py: 加权 RRF                                                │
  │  │                  score = Σ w/(k + rank)                                                │
  │  │                  → fusion_score（按 id 去重，两路分数都保留）                            │
  │  └─────────────────────────────┬──────────────────────────────────────────────────────────┘
  │                                ▼
  │                        RecallTrace（两路原始结果 + 融合结果）
  │                                │
  │  ┌─────────────────────────────▼──────────────┐
  │  │ reranker.rerank(query, fused, top_n)       │  reranker=None 时跳过，
  │  │ → rerank_score（交叉编码器原始 logits）      │  直接按 fusion_score 截断
  │  └─────────────────────────────┬──────────────┘
  ▼                                ▼
返回 RetrievalResponse        results（按相关性降序）
  ├─ results                   + debug{dense, bm25, fused}（debug=true 时）
  └─ count / reranked
```

### 为什么检索也值得单独暴露一个接口

`/chat` 把检索、重排、生成串成一条链，只回一个答案，出问题时是黑盒。
这个接口让下面三种情况一次可分：

| 现象 | 结论 |
|---|---|
| `debug.dense` 与 `debug.bm25` 都空 | 没召回到 → 查分块 / 阈值 / 向量模型 |
| `debug.fused` 有、`results` 空 | 召回到了但被重排丢掉 → 查 `RERANK_MIN_SCORE` |
| `results` 有但答案不对 | 检索没问题 → 是提示词或 LLM 的事 |

四个分数**允许缺省且不做零填充**：缺省表示"这一路没召回它"，
与"召回了但得 0 分"是两件事——前者改召回策略，后者改模型或语料。

### 调参时最该看的两件事

1. **两路各自的条数**。`向量召回=1 BM25召回=1 融合后=2` 一眼就能看出
   是哪一路在干活。若 BM25 路长期为 0，先看是不是查询词与正文完全没有共同词元。
2. **BM25 的噪声水平**。中文分析器不过滤停用词，"的/了/是"会让无关片段也拿到正分
   （`check_retrieval.py` 里有实测：真命中 2.40 vs 噪声 1.01）。
   这是 BM25 的固有特性，靠重排兜住，**不要**去设固定阈值——BM25 分无界且随语料漂移。

---

## 关键参数与调参经验

检索不出结果时，**先打印各阶段真实分数再调参**，不要凭感觉改。
按经验，问题几乎总是这三个之一：

| 症状 | 根因 | 处理 |
|---|---|---|
| 向量召回为空 | `MIN_SIMILARITY` 过高，或分块过大稀释了相似度 | 降低阈值 / 减小 `CHUNK_SIZE` |
| 召回了但被全部过滤 | 重排阈值过高 | 交叉编码器输出是 logits（可为负），阈值尺度与 [0,1] 完全不同 |
| 长文档答不出、短文档正常 | 分块过大 | 中文 `CHUNK_SIZE` 用 150~300 |
| 专有名词/编号搜不到 | 向量路对"训练语料里没见过的词"天然弱 | 靠 BM25 路兜底；确认 `HYBRID_ENABLED=true` 且 `HYBRID_BM25_WEIGHT` 不为 0 |
| BM25 路召回了明显无关的片段 | 中文分析器不过滤停用词（共享一个「的」字就能得分） | 正常现象，交给重排过滤。**不要**设 BM25 阈值——分无界且随语料漂移（见 architecture 第 8 节） |
| 融合后排序"看着不对" | 权重比值语义被误当成绝对值 | `0.5/0.5` 与 `1/1` 排序相同；`fusion_score` 只在同一次查询内可比 |
| 答案对、但引用的页不对 | **页码链路**某一环丢了 | 见下面「溯源页码」一节 |
| 回答是空字符串 | 上游模型返回空内容（截断 / 内容过滤 / 模型被卸载） | 已显式抛 `UpstreamError`(502)，去查 Ollama 日志与 `LLM_TIMEOUT` |
| 接口返回 502 但 Ollama 是好的 | `LLM_BASE_URL` / `LLM_MODEL` 没配 | `ConfigurationError` 不会被兜底成 `UpstreamError`；看响应里的 `code` 字段区分 |

**一个真实的反例**：某项目默认 `chunk_size=500` + `min_rerank_score=0.08`，
数学上永远不可达——456 字的分块约产生 209 个中文二字组 token，
而 13 token 的查询即使全部命中，Jaccard 上限也只有约 0.06 < 0.08，
于是所有候选在重排阶段被清空，表现为「任何问题都答不出来」。
把 `chunk_size` 调到 150、阈值调到 0.02 后恢复正常。

> 教训：阈值不能拍脑袋定，要看分母。分块越大，基于重叠度的分数越小。

### 实测：重排 logits 的分布（bge-reranker-base，raw logits）

用真实文档的片段测出来的尺度感（`RERANK_MIN_SCORE=0.0`）：

| 查询 | 272 字长块 | 69 字块 | 单独切出的 13 字短句 |
|---|---|---|---|
| 多久需要做一次检查？ | **-3.99** | -5.63 | **+0.63** |
| 巡检周期是多久 | +2.46 | -3.73 | **+7.33** |
| 设备的固件怎么升级 | **+3.67** | -10.20 | -10.20 |
| XG-8800 固件升级流程 | **+6.70** | -10.20 | -10.20 |
| 费用报销流程是什么 | -6.19 | **+4.60** | -5.93 |
| 今天天气怎么样 | -10.19 | -10.19 | -10.19 |

三条结论：

1. **无关对的地板是约 -10.2**，相关对能到 +7。所以 `0.0` 这个阈值是干净的：
   它把"无关"与"弱相关"分开，不需要动。看到 `-10.19` 就是模型在说"完全不相关"。
2. **重排是整条链上唯一在做"相关性判定"的环节**。同一句「巡检周期为每季度一次」，
   单独成块时 +0.63 通过、被混进 272 字长块后变成 -3.99 被拒——
   内容没变，只是被无关文字包围了。
3. 于是有个容易误判的现象：**检索层两路都召回了正确片段，最终结果却是空的**。
   这时不要去看检索——去查切分。当前 `RecursiveCharacterTextSplitter` 是按字符数
   切的，会把 Markdown 里几个 `##` 小节揉进同一块（本样例 4 个主题挤在 272 字里）。
   处理方向是让切分器**按标题切**，而不是把 `CHUNK_SIZE` 调小
   （272 字已在推荐区间 150~300 内，调小也不会拆开小节）。

---

## LangGraph 状态字段流转

`RAGState`（`app/rag/state.py`）是节点之间**唯一**的数据契约：
每个节点接收完整 state，返回需要更新的字段（增量合并语义）。

| 字段 | 写入节点 | 读取节点 | 说明 |
|---|---|---|---|
| `kb_id` / `question` | 入参 | retrieve / generate | 生成用**原始问题**（改写只是给检索用的） |
| `history` | 入参 | —（预留） | 多轮上下文；提示词里尚未插入历史消息的位置 |
| `top_k` / `rerank_k` | 入参 | retrieve / rerank | 由 `ChatService._initial_state` 显式填配置默认值（请求不传才用默认，再夹到服务端硬上限）；`retrieve` 在 `RETRIEVE_K_MODE=adaptive` 时还会按每路分数的断层再收缩，见 `rag/topk.py` |
| `rewritten_question` | retrieve（初值）/ rewrite / `ask_stream` 重试 | retrieve / rerank | 首轮由 retrieve 填成 `question`，改写后覆盖；流式重试时由 `ask_stream` 直接写入初值 |
| `candidates` | retrieve | rerank | 融合后的候选（`fusion_score` 已回填） |
| `ranked` | rerank | generate / **条件边** | 送入 LLM 的最终上下文；条件边的判据 |
| `retry_count` | rewrite / refuse / `ask_stream` 重试 | 条件边 / 出参 | 限制回路深度；拒答路径也回填，便于排查"为什么没重试"。检索子图里没有条件边，流式重试写它只为日志/trace 可读 |
| `answer` | generate / refuse | 出参 | |
| `refused` | generate / refuse | 出参 | 前端据此区分「拒答」与「答错」 |
| `error` | 任意节点 | 出参 | 链路异常信息（当前实现里异常直接向上抛，此字段留给后续的宽松降级） |

两条读法上的约定：

1. **所有节点读 `rewritten_question or question`**（`nodes._query_of`）。
   只有 `rewrite_query` 写 `rewritten_question`，所以"检索用的查询"与
   "重排用的查询"必然是同一个值。分开传会出现"重排用原问题、生成用改写问题"
   这类错位，而且是静默的。
2. **`total=False` 意味着字段缺失时 `.get()` 拿到的是 `None` 而不是默认值**。
   所以 `ChatService._initial_state` 把每个字段都显式写入——否则
   "不传 top_k"与"传 None"在节点里看起来一样，而日志与 trace 里看不出本次实际用的深度。

> LangGraph 的 `invoke()` 会**物化全部已声明的通道**：没被任何节点写过的字段
> 也会以空值出现在返回结果里（例如检索子图的 `answer` 是 `''`）。
> 所以断言"子图不产出 answer"要判 `not final.get("answer")`，
> 不能判 `"answer" not in final`——`check_chat.py` 在这里踩过一次。

---

## 溯源页码：从文件到引用的完整链路

页码是本项目里**最容易「看起来对、实际错」**的一环：答案正确不代表引用页正确。
它跨越五个模块，任何一环丢了页码都不会报错：

```
py/pdf 的 page / python-pptx 的 slide index
   │  ① loading.py       PyPDFLoader(mode="page") / PptxLoader 写入 metadata["page"]（0 起）
   ▼
metadata["page"] = 0,1,2…
   │  ② parsing.py       document_to_page()：0 起 → 1 起，**全项目唯一的转换点**
   ▼
Page.page_no = 1,2,3…
   │  ③ splitting.py     create_documents(texts, metadatas=[{"page_no": n}])
   │                     按页独立切分 → 每块必然只属于一页（page_start == page_end）
   ▼
Chunk.page_start = Chunk.page_end = n
   │  ④ vectorstore.py   写 page_start / page_end（未知用 -1 哨兵，不用 NULL）
   ▼
Milvus 行
   │  ⑤ retrieval.py     检索命中后原样带回 Chunk
   ▼
RetrievedChunk.chunk.page_start
   │  ⑥ 展示              Chunk.page_label → "第3页" / "第3-5页" / None
   ▼
prompts.format_context 的 "[1] 来源：手册.pptx 第3页"
services/chat_service  的 citations[].page
```

### 怎么验证它

答案正确不代表引用页正确，所以要有一份**可证伪**的样例：
`tests/fixtures/handbook.pptx` —— 5 页中文手册，**每页一个互不重叠的事实、
关键数字只出现一次**：

| 页 | 标题 | 唯一事实 |
|---|---|---|
| 1 | 员工手册 · 总则 | 施行日期 |
| 2 | 考勤制度 | **九点整** / 半小时 |
| 3 | 费用报销 | **五千元** |
| 4 | 设备巡检 | **每季度一次** |
| 5 | 假期管理 | **十天** |

于是"问报销上限 → 应引用第 3 页"成为一个断言：引用对了说明整条链路页码都没丢；
引用错了能立刻定位到是哪一环丢的。

```bash
.venv\Scripts\python.exe scripts\ask.py --demo
```

实测 5/5 通过，且每题引用的都是**首选**页（中文 PPTX 四题命中第 3/4/2/5 页，
英文 PDF 一题命中第 3 页 —— 两条不同的 loader 路径都被覆盖）。

> **为什么中文页码用 PPTX 验证**：中文 PDF 需要嵌入 CID 字体才能渲染，
> 而 PPTX 每页就是一张幻灯片，天然带页边界且支持中文。
>
> **为什么不用现成的 `sample.pptx`**：那三页讲的是流水线本身，彼此共享大量词元，
> 三个片段都可能被召回 → "引用了第 2 页"这句话就没有信息量了。

### 页码能力的边界（诚实说明）

`page_start` / `page_end` 的语义是「**可溯源的最小定位单位**」，不是排版意义上的页：

| 格式 | 页码 |
|---|---|
| PDF / PPTX | 准确（Loader 天然给出页边界） |
| DOCX / TXT / MD | 恒为第 1 页（流式文本，没有页的概念） |

DOCX **不做**"按固定行数估算页码"——那是**编造**的页码，会让引用溯源显示假信息，
比诚实地写"第 1 页"更糟。需要精确页码时先用 Office / LibreOffice 转 PDF 再上传。
