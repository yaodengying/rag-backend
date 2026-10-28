# rag-backend

> 面向企业内部文档的 **检索增强生成（RAG）后端服务**：上传文档即可问答，答案带**可核对的
> 文件名 + 页码**，并且整条链路（加载 → 切分 → 向量化 → 混合检索 → 重排 → 生成）都有
> **可复现的自检与离线评估**。

前端工程：[`rag-frontend`](https://github.com/yaodengying/rag-frontend)（React 19 + TypeScript + antd 6）

<p>
<img alt="python" src="https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white">
<img alt="fastapi" src="https://img.shields.io/badge/FastAPI-0.143-009688?logo=fastapi&logoColor=white">
<img alt="langgraph" src="https://img.shields.io/badge/LangGraph-1.2-1C3C3C">
<img alt="milvus" src="https://img.shields.io/badge/Milvus-3.0-00A1EA">
<img alt="tests" src="https://img.shields.io/badge/self--check-1066%20assertions-2ea44f">
<img alt="ragas" src="https://img.shields.io/badge/faithfulness-0.990-2ea44f">
</p>

**目录**

- [1. 项目介绍](#1-项目介绍)
- [2. 技术栈](#2-技术栈)
- [3. 系统架构](#3-系统架构)
- [4. 部署步骤](#4-部署步骤)
- [5. 运行演示](#5-运行演示)
- [6. RAGAS 评估结果](#6-ragas-评估结果)
- [7. 后续迭代方向](#7-后续迭代方向)
- [附录 A. 自检清单](#附录-a-自检清单1066-项断言)
- [附录 B. 接口一览](#附录-b-接口一览)
- [附录 C. 约定与踩坑速查](#附录-c-约定与踩坑速查)

> 运维相关：**上传防护与并发**见 [4.8](#48-上传防护与并发)，**备份与恢复**见 [4.9](#49-备份与恢复)
> 与 [`docs/backup.md`](docs/backup.md)。

---

## 1. 项目介绍

### 1.1 它解决什么问题

企业内部的制度、手册、运维规范散落在 PDF / Word / PPT 里，**关键词搜索找不到、通用大模型又
不知道你的私有文档**。本项目把文档收进知识库，用自然语言提问，回答时**逐句给出处**：

```
问：售后工单要求多久内首次响应？

答：客户提交的售后工单，客服须在 48 小时内首次响应并给出处理方案。

来源：售后服务制度.pdf 第 3 页   ← 可点开核对（结构化 citations，不是让模型自己写页码）
```

### 1.2 核心特性

| 特性 | 做法 | 为什么重要 |
|---|---|---|
| **页码级溯源** | 切分按页独立进行（无跨页块）→ 每块携带 `page_no` → 检索结果原样进 `citations` | 答案对不对无法自证，但"第几页"能被人一眼核对。这是知识库问答的验收底线 |
| **混合检索** | 稠密（BGE 余弦）+ 稀疏（Milvus 原生 BM25）双路召回 → 应用侧加权 RRF 融合 | 型号、编号这类"训练语料没见过"的 token 只有 BM25 捞得回；语义改写只有向量懂 |
| **交叉编码器重排** | BGE-reranker-base 对 top_k 候选逐对打分，只留 top_n | 买的是**上下文纯度**：名次通常不变，但送进提示词的条数从 `top_k` 压到 `top_n`，减少模型"在无关文字里找答案" |
| **流式 + 可重试** | SSE `retry? → sources → delta* → done`；首轮没找到时可改写查询再找一次 | 7B 模型首帧 1 秒内可见；`done.can_retry` 让前端可以"先出结果、真没有才问用户要不要再找" |
| **异常语义分层** | 502 依赖坏了 / 504 依赖太慢 / 503 能力没开 / 400 客户端可修 | 出现问题时**处置动作**不同：修依赖、调超时、补配置、改请求，混成一类就只能猜 |
| **防幻觉兜底** | 六道闸门：逐条/整批分数闸门 → 空上下文短路 → 提示词禁令 → 生成后核验硬事实（数字/型号/日期是否真在资料里） | 检索幻觉最常见的形态是「找到了、但找错了」：模型照着错文档答得通顺，提示词和拒答判据都拦不住，只能靠分数与事后核验 |
| **自带评估闭环** | 18 题问答集 + RAGAS 四指标 + 自算页码级命中/MRR + 参数调优脚本 | 改提示词、调 `top_k` 有没有变好，不靠感觉。提示词硬化就是靠它从 0.902 推到 0.990 |

### 1.3 当前状态与边界（诚实清单）

**已实现并自检覆盖**：知识库 CRUD、文档上传（单/批）、文档列表与删除、混合检索（含逐路
debug 明细）、非流式问答、SSE 流式问答、多轮会话（Redis）、登录/注册、18 题离线评估、
参数调优扫描。自检 **1066 项断言**，离线部分不需要 Milvus / BGE / Ollama。

**刻意没做（不是遗漏）**：

| 边界 | 现状 | 原因 |
|---|---|---|
| 单点登出 | 无会话表，令牌在有效期内一直可用 | 换密钥或停用账号可立即失效；要精确登出得引入令牌黑名单 |
| 单点登录 / SSO | 无 | 企业内部署常见需求，但依赖具体 IdP，不适合做进核心 |
| 团队共享 / 细粒度权限 | 知识库**按账号隔离**（`owner_id`），没有"共享给同事"这回事 | 隔离是安全底线，共享是协作功能；后者要做 ACL 模型，与前者不是一回事 |
| 文档增量更新 | 重传即新文档（按内容去重，见 3.3） | 需要文档指纹 + 分块差异比对，属于独立工作量 |

---

## 2. 技术栈

### 2.1 全景

| 层 | 选型 | 实测版本 | 说明 |
|---|---|---|---|
| 语言 | Python | 3.14.7 | 依赖清单在 3.14 下实测可完整解析 |
| Web | FastAPI + Uvicorn | 0.143.0 / 0.54.0 | 异步、自带 OpenAPI |
| 配置 | pydantic-settings | 2.15.0 | 全部配置走 `Settings`，启动期一次校验完 |
| 编排 | **LangGraph** | 1.2.14 | 显式状态图；"检索不足 → 改写重试"这种回路用直线链表达不了 |
| 框架 | LangChain / langchain-core | 1.2.11 / 1.6.9 | 只用其 Loader、TextSplitter、消息抽象 |
| 向量库 | **Milvus**（pymilvus） | 3.0.2 / 3.0.2 | 同一 collection 内做稠密 + 原生 BM25 全文检索 |
| 向量模型 | BGE-large-zh-v1.5 | FlagEmbedding 1.4.2 | 本地权重、CPU 推理、1024 维 |
| 重排模型 | BGE-reranker-base | FlagEmbedding 1.4.2 | 交叉编码器，输出**原始 logits** |
| 大模型 | OpenAI 兼容协议 | langchain-openai 1.7.0 | 本机 Ollama `qwen2:7b`；换 vLLM / 云服务只改 `.env` |
| 元数据 | SQLite + SQLAlchemy | 2.1.4 | 只存知识库/文档元数据，正文与向量不进 SQLite |
| 会话 | Redis | 7.x | 会话元数据 + 消息 + 排序索引三键，TTL 同步续期 |
| 评估 | RAGAS | 0.4.3（`--no-deps` 装） | 离线使用，不进 `app/` |

### 2.2 三个"故意不用"的选型

| 没用什么 | 用了什么 | 原因 |
|---|---|---|
| `langchain-milvus` | 直接 `pymilvus` | 混合检索要建 `sparse` 字段 + `Function(BM25)` + analyzer，且要拿**逐路分数**调参；封装层两样都给不了 |
| `langchain` 的 Embeddings 抽象 | FlagEmbedding 直连 | HF 封装硬依赖 `sentence-transformers`（本项目零引用）且已 deprecated，套一层只有命名收益 |
| PyJWT / bcrypt / passlib | 标准库 `hashlib` + `hmac` | 只需"口令哈希 + 签名令牌"，PBKDF2 与 HMAC-SHA256 足够；少三个依赖，少三处 CVE 面 |

### 2.3 依赖安装要点

```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- **torch 是 FlagEmbedding 的硬依赖**，首次安装 200MB+，装不上先看网络而不是怀疑清单写错。
- **`langchain-core` 不要单独钉版本**：`langgraph 1.2.x` 要求 `langchain-core>=1.4.7`，
  跟 `langchain 1.2` 钉在一起必然 `ResolutionImpossible`。
- **RAGAS 千万别直接 `pip install ragas`**：它的解析会把 `openai` 3.x 与 `langchain-openai`
  一起降级，而生产 LLM 适配器正依赖后者。正确姿势见 [6.1](#61-怎么跑)。

---

## 3. 系统架构

### 3.1 分层与依赖方向

```
                HTTP / SSE
                    │
                    ▼
        ┌───────────────────────┐
        │ app/api/v1/           │  接口层：路由、参数校验、SSE 编码、异常 → HTTP
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │ app/services/         │  应用层：用例编排（入库/检索/问答/会话/认证）
        └────┬──────────────┬───┘
             │              │
             ▼              ▼
   ┌──────────────────┐  ┌────────────────────────┐
   │ app/rag/         │  │ app/infrastructure/    │  适配层：加载/解析/切分/
   │ LangGraph 状态图 │  │ 向量化/向量库/重排/LLM │  Redis/SQLite
   └────────┬─────────┘  └───────────┬────────────┘
            │                        │
            └──────────┬─────────────┘
                       ▼
            ┌───────────────────────┐
            │ app/domain/           │  领域层：模型 + 端口（Protocol）
            │ 只依赖标准库          │
            └───────────────────────┘
```

**规则：依赖只能向下，`domain/` 不 import 任何三方库。** 新增一种能力走三步——
`domain/ports.py` 定义 Protocol → `infrastructure/` 写实现 → `core/container.py`（`lru_cache`
单例）装配。给 Protocol 加方法后**所有假实现都要补**，`isinstance` 断言会立刻变红（设计如此）。

### 3.2 入库链路（7 步）

```
上传 → 扩展名校验/体积/空文件 → 存盘 → SQLite(status=processing)
     → 加载(LangChain Loader) → 解析(Document→Page) → 按页切分(Chunk)
     → 向量化(BGE, L2 归一化) → Milvus 分批写入 + 一次 flush
     → SQLite(status=ready, chunk_count) → 201
```

失败语义刻意分成两层：

- **入口校验失败**（扩展名、体积、0 字节）→ `400` + 稳定 `code`（`corrupted_document` /
  `encrypted_document` / `empty_document`）；
- **处理阶段失败**（解析、向量化、写库）→ **`201` + `status=failed` + `error`**。
  这样用户在文档列表里能看见"哪一篇、为什么失败"，而不是只弹一个报错框。

### 3.3 检索链路（混合召回 → RRF → 重排）

```
query ─┬─ BGE 编码 ──► Milvus 稠密检索 (anns_field="embedding", COSINE) ─┐
       │                                                                ├─► 加权 RRF ─► 去重 ─► top_k
       └─ 原文 ──────► Milvus BM25 全文检索 (sparse 字段, chinese 分析器) ─┘                    │
                                                                                              ▼
                                                                      BGE-reranker 逐对打分 ─► top_n
```

`fusion_score(d) = Σ_path  w_path / (k + rank_path(d))`，`k=60`。

四条设计口径：

1. **RRF 而不是分数加权求和**：两路分数不可比（余弦落在 [-1,1] 且可比区间窄；BM25 无界，
   同一次查询里实测 9.84 与 1.65 并存，换一份语料整体尺度又全变）。归一化方案换语料就失效，
   RRF 只吃排名。
2. **权重只有比值有意义**：`0.5/0.5` 与 `1/1` 排序完全相同，所以不能当概率或阈值用。
3. **融合在应用侧**：Milvus 也能服务端 RRF，但那样拿不到 `vector_score` / `bm25_score`，
   调参变黑盒。多一次往返换"检不出时能看清是哪一路没召回"。
4. **跨副本去重先于 `top_k` 截断**：同一文件重复上传 4 次时，4 条候选的向量分逐位相同
   （实测 `…70419312 ×4`），会占掉 `top_k=8` 的一半名额。按 `(filename, 空白归一化正文)`
   去重、留分最高，`duplicates_removed` 进 `RetrievalTrace`（否则"召回变好"无从归因）。

### 3.4 问答链路（LangGraph）

```
START → retrieve → rerank ─┬─ ranked 非空 ─────────────────► generate → END
                           │
                           └─ ranked 为空 ─┬─ 未超上限 ─► rewrite_query ─┐
                                           │                              │
                                           └─ 已超上限 ─► refuse → END     │
                                                                          │
                          ◄───────────────────────────────────────────────┘
```

- **路由判据是 `ranked` 而不是 `candidates`**："召回到了"与"召回的能用"是两件事。重排可能把
  全部候选判为负分（问巡检周期时召回块里混着报销与固件内容），这时拿 `candidates` 去生成，
  就是让模型在无关文字里"找"答案——幻觉的高发场景。
- **无可用片段时不调 LLM**，直接返回固定拒答话术。省推理是次要的，主要是不给模型自由发挥的机会。
- **两张编译好的图**：`build_rag_graph`（含改写重试）给非流式；`build_retrieval_graph`
  （`retrieve → rerank`）给流式，生成由 `ChatService` 直接 `llm.stream()`。
  原因很硬：**token 只能来自 `llm.stream()`**，图节点返回的是最终字段值、不是增量。
  两张图必须共用**同一个**检索器/重排器实例。

| | 非流式 `POST /chat` | 流式 `POST /chat/stream` |
|---|---|---|
| 改写重试 | **恒**开一次（调用方本就在等完整答案） | 三态，由前端决定（见下） |
| 输出 | `Answer` + `citations` | SSE：`retry? → sources → delta* → done` |

`enable_retry` 是**三态**，写成 `if not enable_retry: 用默认值` 会让"用户关掉开关"被悄悄推翻：

| 传什么 | 行为 |
|---|---|
| `true` / `false` | 用户显式选择，**一定压过**服务端默认值 |
| 省略 / `null` | 落回 `CHAT_STREAM_RETRY_DEFAULT`（出厂 `false`） |

`done` 事件统一回填 `retried` / `attempts` / `can_retry` / `query`（成功与拒答字段一致，
`attempts` 从 **0** 起：0 = 首轮就答出来了），且 `can_retry = refused and attempts < MAX_RETRY`。前端可以**先按默认出结果、真没找到才问用户要不要
"再找一次"**，不必盲签也不必白等 2~4 秒。默认值/上限/事件名从 `GET /chat/options` 读，
**别在前端抄一份**（会漂移，且漂移时没有任何报错）。

### 3.5 防幻觉兜底（六道闸门）

幻觉没有单一解，只能按「哪一步出错」分层拦。完整判据与实测见
**[docs/anti-hallucination.md](docs/anti-hallucination.md)**，这里只给地图：

| # | 闸门 | 拦什么 | 默认 |
|---|------|--------|------|
| 1 | 入参 / 库校验 | kb_id 打错、空问题 | 开 |
| 2 | 逐条分数闸门（`RERANK_MIN_SCORE`） | 单条负分的无关片段 | 开（0.0） |
| 3 | 整批最高分闸门（`RERANK_MIN_TOP_SCORE`） | **整批都不对题**（"主题相邻、答案没有"） | **关**（0.0） |
| 4 | 空上下文短路 | 零候选 → 不调模型，直接拒答 | 开 |
| 5 | 提示词 6 条编号禁令 + 逐字拒答话术 | 改写数字、跨段拼接、无端补充 | 开 |
| 6 | **接地性核验**（`GROUNDING_ACTION`） | 数字/型号/日期**编造**、引用序号越界 | 开（`flag`） |

三条要点：

- **能挡在模型之前的，绝不留到模型之后**（3、4 号闸门每拦一次既省 2~10 秒，也少一次编造机会）。
- 闸门 6 查的是**可证伪的硬事实**：回答里的数字/型号/日期是否真在本轮上下文里逐字出现。
  分级 `ok / warn / low`，`flag` 档只标记不改写（改文字会把"它哪里编了"的证据洗掉），
  `block` 档在**非流式**下把 `low` 换成拒答——流式做不到，token 已经发出去了。
- **闸门 6 抓不到"过度推断"**：实测「备件能加急吗」，模型答「可以尝试沟通缩短 5 个工作日」，
  数字全对、语义凭空多推一步，核验照样给 `ok`。这类只能靠提示词 + 闸门 3 减少机会，
  语义级核验（NLI/LLM 自评）成本高且未验证，本项目**不提供该开关**——不装样子。

判决进响应：`grounding.status / unsupported / coverage / blocked`（流式在 `done` 事件里），
当前档位从 `GET /chat/options` 的 `grounding_action` 读。

### 3.6 会话历史（Redis）

| 键 | 类型 | 内容 |
|---|---|---|
| `rag:conv:{id}` | hash | `title` / `kb_id` / 消息计数 / 时间戳 |
| `rag:conv:{id}:msgs` | list | 消息序列（JSON，正序追加），assistant 消息带 citations |
| `rag:conv:index` | zset | 会话 ID → 最近更新时间（列表页排序） |

- 带 `conversation_id` → 读最近 `CONVERSATION_HISTORY_TURNS` 条进提示词，答完两轮落库；
  不带 → 服务端生成 ID（**首问也落库**，否则第二轮的历史从第一天起就是残的），经 `done` 回传。
- **TTL 三键同步续期**，只续 meta 会出现"列表里还有、点进去没消息"。
- `REDIS_URL` 留空**不拦启动**：问答照常（不落库、历史退回请求体），`/conversations` 一族报
  `503` 并点名缺什么。落库失败只记日志——不用存储故障绑架问答主链路。

### 3.7 目录结构

```
rag-backend/
├── app/
│   ├── main.py                 # 入口：中间件 / 路由 / lifespan（启动期校验）
│   ├── api/v1/                 # health auth collections documents retrieval chat conversations
│   ├── core/                   # config（含 validate_runtime）logging exceptions container
│   ├── domain/                 # models + ports（Protocol），只依赖标准库
│   ├── infrastructure/         # loading parsing splitting embedding vectorstore rerank llm
│   │                           # security filesafety redis_store
│   ├── rag/                    # ingest fusion retrieval state nodes graph prompts rewrite
│   │                           # grounding concurrency
│   ├── services/               # collection ingest retrieval chat conversation auth + mappers
│   ├── db/                     # SQLite 表结构与仓储
│   └── schemas/                # 对外 DTO（Pydantic）
├── deploy/                     # docker-compose.yml + Dockerfile
├── docs/                       # architecture data-flow pipeline evaluation tuning
│                               # anti-hallucination backup
├── eval/
│   ├── dataset/qa_dataset.jsonl
│   └── results/<时间戳>/       # report.md summary.json samples.jsonl scores.csv
├── scripts/                    # 见附录 A（13 个自检 + ask/init/evaluate/tune/probe/create_user/verify）
│                               # 与 dev.py（开发服务器）、backup.py、rebuild_vectors.py
├── tests/fixtures/             # 自检用的样例文档（由 scripts/make_sample_docs.py 生成）
├── data/                       # uploads/ + rag.db（不入库）
└── requirements.txt  .env.example  README.md
```

### 3.8 关键设计决策速查

| 问题 | 决策 | 一句话理由 |
|---|---|---|
| 页码怎么保证 | 按页独立切分，`metadata["page_no"]` 随块传递 | 跨页块会让"第几页"永远说不清 |
| 页码解析不出来 | `page_no=None`（写"页码未知"） | 猜一个第 1 页比诚实说"未知"更糟——溯源标签错了比没有更坏 |
| 切分粒度 | `CHUNK_SIZE=300 / overlap=50` | 500 会让块内塞进 4 条同参数段落，合成基准 MRR 掉到 0.8194 |
| 混合模式要不要相似度阈值 | 稠密路默认 **0.0 不设限** | 中文短查询对 300 字长块余弦仅 0.28~0.35，设 0.35 会**静默清空**整条路，而 BM25 会把它掩盖 |
| BM25 要不要阈值 | 默认不设（配置 0 → 显式传 `None`） | 中文分析器不过滤停用词，共享一个"的"就得分，分无界且随语料漂移 |
| 提示词怎么写 | 6 条编号禁令，正面要求写成**无条件祈使句** | 条件句会被 7B 当成前提不成立，连带整段不执行 |
| 允许部分回答怎么表达 | 只写"只答覆盖的那部分"，**不给占位句式** | 给了"其余写『资料未说明』"，模型会把这句话当答案交上来（实测 5/5） |
| 依赖故障报什么码 | 502 / 504 / 503 / 400 分开 | 处置动作不同：修依赖 / 调超时 / 补配置 / 改请求 |
| 闸门放在模型前还是后 | 能前就前（分数闸门、空上下文短路） | 每拦一次既省 2~10 秒，也少一次编造机会 |
| 整批最高分闸门默认开吗 | **关**（0.0），要用先量分离度 | 本机只有 1 条负样本落在正分区间，样本不足以定阈值；照抄别人的数字必翻车 |
| 核验到不可信要不要改答案 | `flag` 只标记，`block` 才换拒答（仅非流式） | 改文字会洗掉「它哪里编了」的证据；而流式 token 已发出，改不了 |

更细的取舍与实测数据见 [`docs/architecture.md`](docs/architecture.md)（决策表）、
[`docs/data-flow.md`](docs/data-flow.md)（两链路时序）、[`docs/pipeline.md`](docs/pipeline.md)（逐阶段坑）。

---

## 4. 部署步骤

### 4.1 前置依赖

| 组件 | 要求 | 本机参考 | 检查命令 |
|---|---|---|---|
| 操作系统 | Windows 10/11（Linux/macOS 同理，路径改一下） | Windows Server | — |
| Python | **3.14.x**（依赖按此实测） | `E:\Python314` | `python -V` |
| Milvus | **v3.0.2** standalone（需 etcd + minio） | `H:\Docker\milvus` | `curl http://127.0.0.1:9091/healthz` |
| Redis | 7.x（会话历史；不装也能跑，只是 `/conversations` 报 503） | `127.0.0.1:6379` | `redis-cli -a *** ping` |
| Ollama | 任意 OpenAI 兼容后端 | `127.0.0.1:11434`，模型 `qwen2:7b` | `curl http://127.0.0.1:11434/api/tags` |
| BGE 权重 | `bge-large-zh-v1.5`、`bge-reranker-base` 的**权重目录**（含 `config.json`） | `H:\models\...` | 目录里必须有 `config.json` + `pytorch_model.bin`/`.safetensors` |

> 权重目录**不是** Ollama 的 `manifests` 目录，也**不是** `models--xxx` 这种 HF 缓存符号链接目录。
> FlagEmbedding 直接按目录加载，指错会在启动期被 `validate_runtime()` 拦下。

### 4.2 起依赖服务

```powershell
# Milvus（本机已有一套完整编排：milvus v3.0.2 + etcd + minio，数据落 H:\Docker\milvus\volumes）
cd H:\Docker\milvus
docker compose up -d
docker compose ps                     # 三个容器都 healthy 再继续

# Redis（没有独立安装就起一个容器）
docker run -d --name rag-redis -p 6379:6379 redis:7-alpine --requirepass redis123456

# Ollama（装好的话确认模型在）
ollama list                           # 应能看到 qwen2:7b
```

### 4.3 安装与配置

```powershell
cd <项目根目录>
py -3.14 -m venv .venv                # 已有 .venv 可跳过（本项目现用 PyCharm 建的 3.14.7 环境）
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

copy .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"   # 生成 AUTH_SECRET_KEY，别手敲
```

`.env` 里**必须填**的 5 项（留空会被启动期校验一次性全列出来并拒绝启动）：

| 变量 | 说明 | 示例 |
|---|---|---|
| `LLM_BASE_URL` | OpenAI 兼容端点 | `http://127.0.0.1:11434/v1` |
| `LLM_MODEL` | 模型名 | `qwen2:7b` |
| `EMBEDDING_MODEL_PATH` | BGE 向量权重**目录** | `H:\models\bge-large-zh-v1.5` |
| `RERANK_MODEL_PATH` | 重排权重目录（`RERANK_ENABLED=true` 时必填） | `H:\models\bge-reranker-base` |
| `AUTH_SECRET_KEY` | 令牌签名密钥，≥32 字符 | 上面那行命令的输出 |

其余常用项（都有可用默认值，改动前建议先读 `.env.example` 里的注释）：

| 变量 | 默认 | 调到什么时候 |
|---|---|---|
| `RETRIEVE_TOP_K` | 8 | 库很大想提高召回可到 12~16；**小于 6 会明显掉 MRR**（实测 2→0.9167、4→0.9583） |
| `RETRIEVE_K_MODE` | `fixed` | `adaptive` 时先取 `min(top_k, RETRIEVE_K_MAX)` 条，再按每路分数的断层收缩到 `[RETRIEVE_K_MIN, 上限]`；不传 `top_k` 的请求现在真的走配置（旧版被 schema 写死的 8/4 挡住了）。改之前先看 `scripts/tune_retrieval.py` 的分数分布并重跑评估 |
| `RERANK_TOP_N` | 4 | 上下文纯度 vs 覆盖面的权衡；7B 下 4 条已够 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 300 / 50 | 中文建议 150~300，**500 实测掉分** |
| `HYBRID_VECTOR_WEIGHT` / `HYBRID_BM25_WEIGHT` | 0.5 / 0.5 | 只有比值有意义；0.7/0.3 实测略差 |
| `HYBRID_DENSE_MIN_SIMILARITY` | 0.0 | 不要盲目调高，理由见 3.8 |
| `CHAT_STREAM_RETRY_DEFAULT` | false | 想让"首轮没找到"自动多花 2~4 秒改写就设 true |
| `GROUNDING_ACTION` | `flag` | 接地性核验的三档动作：`off` / `flag`（只标记）/ `block`（非流式下低置信换成拒答）。`block` 会推高拒答率，属产品取舍，开之前先读 [docs/anti-hallucination.md](docs/anti-hallucination.md) |
| `RERANK_MIN_TOP_SCORE` | 0.0 | 整批最高分闸门，0 = 关。要开**必须先用 `scripts/probe_rerank_gap.py` 量分离度**，别照抄别人的数字（logits 尺度随模型/语料漂） |
| `CONVERSATION_TTL_HOURS` | 168 | 会话保留 7 天，每次问答自动续期 |

### 4.4 初始化向量库

```powershell
.\.venv\Scripts\python.exe scripts\init_milvus.py                    # 建库 / 校验维度 / 打印 schema
.\.venv\Scripts\python.exe scripts\init_milvus.py --recreate --yes   # 先删后建（会丢向量！）
```

`--recreate` 不可逆：终端下要手输 `RECREATE`，非交互环境必须显式 `--yes`。
**维度以权重为准**（读 `config.json` 的 `hidden_size`），`MILVUS_DIMENSION` 只是声明值，
不一致时启动失败并告诉你该改成多少——不会出现"按旧维度建表、按新维度写入"。

### 4.5 启动

```powershell
cd <项目根目录>                    # cwd 必须是项目根：.env 按相对路径加载
$env:NO_PROXY = "localhost,127.0.0.1,::1"
.\.venv\Scripts\python.exe scripts\dev.py --port 8000       # 推荐：自带确定性热重载
```

> **不要用 `uvicorn --reload`**。uvicorn 0.54 重载时给子进程发的是 `CTRL_C_EVENT`
> （`supervisors/basereload.py:90`），这是**控制台事件**：从 PyCharm 运行窗、Git Bash 管道
> 或 `> log` 重定向启动时投递不到 → 子进程不退出 → reloader 永久卡在 `join()`。表现是
> 「端口有响应、日志正常、改代码不生效」——旧 worker 一直在用旧代码服务，本机踩过三次。
> `scripts/dev.py` 改用 `terminate()`（Windows 上是 `TerminateProcess`，必杀），并且
> worker 崩了会**明确报退出码**，不会静默假活。
>
> 部署/压测不需要热重载时，直连 uvicorn，去掉 `--reload` 即可：
>
> ```powershell
> .\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
> ```

启动顺序 `validate_runtime()` → `init_schema()` → `ensure_collection()`，任一失败**拒绝启动**
（半可用的服务比直接崩更难查：曾经 llm 段填了真值、模型路径留空，服务能起、LLM 能答，
但向量检索整体失效且没有任何报错）。

| 地址 | 说明 |
|---|---|
| http://127.0.0.1:8000/docs | Swagger 交互文档 |
| http://127.0.0.1:8000/api/v1/health | 存活检查 |
| http://127.0.0.1:8000/api/v1/ready | 就绪检查（探测 Milvus 与权重目录） |

> `NO_PROXY` 不建议省：本机若有全局 `http_proxy`，pymilvus 的 gRPC 会被代理拦成 502。

### 4.6 容器化（可选）

```powershell
docker compose -f deploy/docker-compose.yml --env-file .env up -d --build
```

`--env-file .env` 是必须的：compose 默认在 `deploy/` 下找 `.env`，找不到根目录那份。
容器内的 `127.0.0.1` 是容器自己，访问宿主机 Ollama 要用 `LLM_BASE_URL_HOST`
（`http://host.docker.internal:11434/v1`）。变量缺失时 compose 会直接报错退出，
不会起一个检索链路已失效的容器。

### 4.7 首次部署验收（5 步，全绿再算成功）

```powershell
# 1. 就绪探针
curl -s http://127.0.0.1:8000/api/v1/ready

# 2. 离线自检（不依赖 Milvus/BGE/Ollama，秒级）
.\.venv\Scripts\python.exe scripts\check_chat.py
.\.venv\Scripts\python.exe scripts\check_splitting.py

# 3. 依赖型自检（需 Milvus / Redis）
.\.venv\Scripts\python.exe scripts\check_vectorstore.py
.\.venv\Scripts\python.exe scripts\check_retrieval.py
.\.venv\Scripts\python.exe scripts\check_conversations.py

# 4. 一键演示：造中文多页 PPTX → 建库 → 入库 → 提问 → 校验页码溯源
.\.venv\Scripts\python.exe scripts\ask.py --demo

# 5. 全程 HTTP 的端到端验证（自带服务启停，12 条探针）
.\.venv\Scripts\python.exe scripts\verify_upload_flow.py
```

### 4.8 上传防护与并发

上传链路上有**三道单请求闸门**（外加批量总闸门）和**五类内容检查**，都落在服务层而不是
接口层——接口层可以被脚本、定时任务、别的服务绕过，服务层才是可信边界。

| 闸门 | 位置 | 挡住什么 | 失败时 |
|---|---|---|---|
| `Content-Length` 预检 | HTTP 中间件 | 明显离谱的请求体（5GB），**省掉落盘** | 413 |
| 读取封顶 | `_read_upload` | 伪造 `Content-Length` 的请求，**省掉内存** | 服务层判 400 |
| 单文件上限 | `IngestService._validate` | 正常的超大文件 | 400 |
| 文件数 / 累计体积 | 批量接口 + 服务层 | "20 个 49MB 文件"绕开单文件上限 | 400（整批） |

```bash
# .env
MAX_UPLOAD_MB=50            # 单文件
MAX_BATCH_FILES=20          # 单次批量文件数
MAX_BATCH_TOTAL_MB=200      # 单次批量累计体积
```

**为什么不能只在路由里"读的时候看一眼大小"**：Starlette 解析 multipart 时
会先把整个请求体写进临时文件（>1MB 落盘），然后才交给路由函数。
一个 5GB 的请求会在你判断之前就把磁盘写满。

#### 恶意文件过滤

扩展名白名单只看名字，`evil.exe` 改名成 `evil.pdf` 零成本。所以还要看内容
（`app/infrastructure/filesafety.py`，纯函数、无 IO）：

| 检查 | 规则 | 挡住什么 |
|---|---|---|
| 文件名 | 控制字符 / 换行 / 路径分隔符 / >200 字符 | 日志注入、响应头截断、`ENAMETOOLONG` |
| 魔数 | `.pdf` 前 1KB 内必须有 `%PDF-`；`.docx`/`.pptx` 必须是 zip | 改后缀的任意二进制 |
| PDF 页数 | `MAX_PDF_PAGES`（默认 3000） | 3 万页 PDF 让解析 OOM（进程被系统杀掉，日志里什么都没有） |
| zip 容器 | 条目数、解压总量、单条压缩比 | **压缩炸弹**：.docx/.pptx 本来就是 zip |
| 文本形态 | NUL 字节；控制字符占比 > 1% | 二进制伪装成 `.txt` |

实测（`scripts/check_filesafety.py`）：

```
evil.pdf（MZ 头）      → 400 [unsafe_file] "文件头不是 PDF"
bomb.docx（5.2KB）     → 400 [unsafe_file] "压缩比 1026:1，超过上限 200:1"
52MB 请求体            → 413 payload_too_large
```

> ⚠️ **压缩比判定必须带绝对体积下限**。真实测过：一份 36KB 的正常 docx
> 压缩比高达 346:1（`<w:p>` 反复出现），只看比例会把正常文档判成炸弹。
> 现在只在"单条目解压后 ≥ 1MB **且** 比例超限"时才拒——危险的是绝对体积。

**能力边界**：这里只看结构，不解压、不执行。文档里藏宏/外链/JS 不归它管，
但也构不成风险——PyPDF / docx2txt / python-pptx 都只抽文本，不执行嵌入对象。

#### 多人并发

四件事，全部有自检钉住（`check_ingest.py` 第 6 组）：

| 机制 | 解决什么 | 实现 |
|---|---|---|
| **正文哈希去重** | 同一份文件传 4 遍 → 4 条 ready，向量分逐位相同、互相挤占 `top_k` | `documents.content_hash` + 部分唯一索引；命中返回 409 + 已有 `doc_id` |
| **按 key 互斥锁** | 两个请求同时 reprocess 同一文档 → 两份向量 | `DocumentLockPool`（引用计数，用完回收） |
| **全局并发闸门** | 多个大文件同时向量化 → 所有请求尾延迟一起变差 | `INGEST_MAX_CONCURRENCY`（默认 1 = 排队） |
| **SQLite `busy_timeout`** | 并发写报 `database is locked` | `PRAGMA busy_timeout=5000` + WAL |

去重按 **`(kb_id, 正文 SHA-256)`** 判定，不看文件名：改名不换内容算重复，
同名不同内容（用户改了文档再传）必须放行。

**作用域要说清楚**：锁与闸门都是**进程内**的。uvicorn 起 4 个 worker 就有 4 份锁，
跨进程互斥需要 Redis 锁。因此部署建议写死：**入库保持单 worker，横向扩展放到检索侧**
（只读，天然可并行）。真要上多 worker，把 SQLite 换成 PostgreSQL 再说——
单文件 SQLite 才是那时候的真瓶颈。

历史数据：`init_schema()` 会自动补 `content_hash` 列与唯一索引（幂等），
旧行的 hash 为空串，不参与唯一约束。

### 4.9 备份与恢复

```powershell
# 备份：VACUUM INTO 一致快照 + uploads（只打登记过的文件）+ manifest
.\.venv\Scripts\python.exe scripts\backup.py backup
# 校验（备份不校验等于没有备份）
.\.venv\Scripts\python.exe scripts\backup.py verify data\backups\backup-xxxx
# 恢复（破坏性，先自动备份当前数据）
.\.venv\Scripts\python.exe scripts\backup.py restore data\backups\backup-xxxx --yes
# 重建向量（恢复后文档全为 failed，Milvus 里没有对应行）
.\.venv\Scripts\python.exe scripts\rebuild_vectors.py
```

**只备 `rag.db` + `uploads/`**。向量是派生的（能从原始文件重算），
Redis 会话是 TTL 数据，都不备。完整说明（一致性原理、恢复的 6 个步骤、定时任务、
恢复演练、失败排查）见 **[docs/backup.md](docs/backup.md)**。

### 4.10 故障排查

| 现象 | 最可能的原因 | 处置 |
|---|---|---|
| `无法连接 Milvus（http://127.0.0.1:19530）` | 容器没起，或全局代理拦了 gRPC | `docker compose ps`；设 `NO_PROXY=localhost,127.0.0.1,::1` |
| 启动即失败，报维度不符 | 换了 embedding 模型但没重建 collection | `init_milvus.py --recreate --yes` |
| 启动报"collection 缺少 xxx 字段" | 旧 schema（没有 `sparse` / analyzer） | 同上重建；`ensure_collection()` **不会**静默重建 |
| `/conversations` 报 503 | 没配 `REDIS_URL` | 填 `redis://[:密码@]127.0.0.1:6379/0` |
| 问答 502 | 上游 LLM 挂了 / 地址错 | 看日志里的 `UpstreamError` detail；确认 `LLM_BASE_URL` |
| 问答 504 | 模型太慢或超时太短 | 调 `LLM_TIMEOUT`，`RERANK_TOP_N` 降一点 |
| 有资料却拒答 | 重排把候选全判负分（`RERANK_MIN_SCORE=0.0` 是硬过滤） | `/search?debug=true` 看 `rerank_score` |
| 明明有资料，回答却是空串 | 上游截断/卸载模型 | 代码已把空回复判成 `UpstreamError`，看日志 |
| curl 传中文变乱码 | Git Bash 内联中文非 UTF-8 | 把 JSON 写文件用 `-d @payload.json`，或直接用 `/docs` |
| `--reload` 卡在 `Reloading...`，改代码不生效 | Windows 上 `CTRL_C_EVENT` 投递不到控制台，reloader 卡在 `join()`，旧 worker 继续用旧代码服务 | 改用 `scripts/dev.py`；已卡死的先杀**进程树**，否则孤儿 worker 会继承监听套接字，新实例绑不上 8000 |
| 文档 `status=failed`，解析超时 / 内存不足 | 大文件一次吃满内存或耗时过长 | 修好依赖（或调大 `MAX_UPLOAD_MB` / 调小 `EMBEDDING_BATCH_SIZE`）后 `POST /api/v1/documents/{doc_id}/reprocess`，**不用重新上传** |
| 文档 failed 且 `error` 是 `[corrupted_document]` / `[encrypted_document]` | 文件本身坏了或加密，重试也没用 | 用原程序重新导出、去掉密码后再传；`reprocess` 会复现同样的失败 |
| 上传报 413 `payload_too_large` | 请求体超过 `MAX_UPLOAD_MB` (+1MB multipart 余量) | 压缩/拆分；批量请分批 |
| 上传报 400 `[unsafe_file]` | 魔数不符 / 压缩炸弹 / 二进制伪装 | 看 `detail`（`head_hex`、`ratio`）；确认文件不是改后缀来的 |
| 上传报 409 `conflict` | 同知识库已有**相同正文**的文档 | 响应 `detail.doc_id` 就是已有那条，直接用；真要重传请先删旧的 |
| 批量上传 400 "一次最多上传 N 个文件" | 文件数超 `MAX_BATCH_FILES` | 分批；单个文件的失败仍会逐条返回 |

---

## 5. 运行演示

### 5.1 一键演示（推荐先跑这个）

```powershell
.\.venv\Scripts\python.exe scripts\ask.py --demo
```

它会：生成中文多页 PPTX 夹具 → 建库 → 入库 → 提问 → **自动校验引用的页码**。实测 5/5：

| 问题 | 期望 | 实际引用 |
|---|---|---|
| 报销金额上限是多少？ | handbook.pptx 第 3 页 | 第 3 页 ✓ |
| 设备多久巡检一次？ | handbook.pptx 第 4 页 | 第 4 页 ✓ |
| 上班时间是几点？ | handbook.pptx 第 2 页 | 第 2 页 ✓ |
| 一年有几天年假？ | handbook.pptx 第 5 页 | 第 5 页 ✓ |
| What does the document say comes after loading and chunking? | sample.pdf 第 3 页 | 第 3 页 ✓ |

> 为什么答案对还要验页码：答案正确**不代表**引用正确。夹具做成"一页一事实、关键数字只出现
> 一次"，于是"问报销上限 → 必须引第 3 页"是**可证伪**的断言——引用对了说明
> `loader → parser → splitter → Milvus → 检索 → 重排 → citations` 整条链路的页码都没丢；
> 引用错了能立刻定位到哪一环。

### 5.2 手动走一遍（curl）

```powershell
# ① 建库
curl -s -X POST http://127.0.0.1:8000/api/v1/knowledge-bases `
  -H "Content-Type: application/json" -d "@kb.json"
#   kb.json: {"name":"售后制度","description":"客服与售后文档"}

# ② 上传（单文件 201；批量 200 逐文件成败）
curl -s -X POST "http://127.0.0.1:8000/api/v1/knowledge-bases/<kb_id>/documents" `
  -F "file=@D:\docs\售后服务制度.pdf"

# ③ 看文档状态：processing → ready（或 failed + error）
curl -s "http://127.0.0.1:8000/api/v1/knowledge-bases/<kb_id>/documents"

# ④ 只检索不生成，带两路明细（调参用）
curl -s -X POST "http://127.0.0.1:8000/api/v1/knowledge-bases/<kb_id>/search" `
  -H "Content-Type: application/json" -d "@search.json"
#   search.json: {"query":"售后工单多久响应","debug":true,"top_k":4}

# ⑤ 问答（非流式，带 citations）
curl -s -X POST http://127.0.0.1:8000/api/v1/chat `
  -H "Content-Type: application/json" -d "@q.json"

# ⑥ 流式问答
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/stream `
  -H "Content-Type: application/json" -d "@q.json"
#   q.json: {"kb_id":"<kb_id>","question":"售后工单要求多久内首次响应？","enable_retry":false}
```

**④ 的响应结构**（`debug=true` 时多出 `debug` 段，以下为一次实测输出、正文截断）：

```json
{
  "query": "售后工单多久响应",
  "kb_id": "…",
  "count": 1,
  "reranked": true,
  "results": [{
    "chunk_id": "…", "doc_id": "…", "filename": "售后服务制度.pdf",
    "page": "第3页", "chunk_index": 0,
    "text": "客户提交的售后工单，客服须在 48 小时内首次响应…",
    "vector_score": 0.7614, "bm25_score": 9.8423,
    "fusion_score": 0.0164, "rerank_score": 7.5308
  }],
  "debug": {
    "hybrid_enabled": true,
    "dense": {"channel": "dense", "count": 4,
              "results": [{"page": "第3页", "vector_score": 0.7614},
                          {"page": "第4页", "vector_score": 0.5697}]},
    "bm25":  {"channel": "bm25",  "count": 4,
              "results": [{"page": "第3页", "bm25_score": 9.8423},
                          {"page": "第2页", "bm25_score": 1.6685}]},
    "fused": [{"page": "第3页", "fusion_score": 0.0164}]
  }
}
```

这段输出顺带说明了两路的尺度差异：同一查询稠密分 0.76 / 0.57 / 0.45 / 0.44，BM25 分
9.84 / 1.67 / 1.66 / 1.65 ——**BM25 的绝对尺度与余弦完全不可比**，这也是融合必须走 RRF
（只吃排名）的原因。

四个分数**允许缺省**，缺省表示"这一路没召回它"——与"召回了但得 0 分"是两件事。
`page` 是 `"第3页"` / `"第3-5页"` / `null`（txt、md、docx 这类流式文本没有页边界，如实给 `null`）。

**⑥ 的 SSE 事件序**。正常作答（首轮命中）——注意 `attempts` 从 0 起：

```
event: sources
data: {"type":"sources","sources":[{"doc_id":"…","filename":"售后服务制度.pdf",
       "page":"第3页","chunk_index":0,"text":"…",
       "vector_score":0.7608,"rerank_score":7.6673}]}

event: delta
data: {"type":"delta","content":"客户提交的售后工单，"}

event: delta
data: {"type":"delta","content":"客服须在 48 小时内首次响应。"}

event: done
data: {"type":"done","answer":"客户提交的售后工单，客服须在 48 小时内首次响应。",
       "refused":false,"citations":[{"doc_id":"…","filename":"售后服务制度.pdf",
       "page":"第3页","snippet":"客户提交的售后工单…"}],
       "retried":false,"attempts":0,"can_retry":false,
       "query":"售后工单要求多久内首次响应？","conversation_id":"…",
       "grounding":{"status":"ok","checked":1,"unsupported":[],
                    "invalid_refs":[],"coverage":1.0,"blocked":false,"warnings":[]}}
```

> `grounding` 是**生成之后**的接地性核验：回答里的数字/型号/日期是否真在本轮上下文里
> 逐字出现过（`checked=1` 是那个「48 小时」）。流式下它只能事后标记——token 已经推给
> 前端了，服务端收不回来。详见 [docs/anti-hallucination.md](docs/anti-hallucination.md)。

**首轮没找到、允许重试、改写后仍然没找到**（实测报文，库是售后制度文档，问的却是餐补）：

```
event: retry
data: {"type":"retry","attempt":1,"max_attempts":1,
       "reason":"首轮检索无可用片段，正在改写查询"}

event: sources
data: {"type":"sources","sources":[]}

event: delta
data: {"type":"delta","content":"无法从知识库中找到与该问题相关的答案。"}

event: done
data: {"type":"done","answer":"无法从知识库中找到与该问题相关的答案。",
       "refused":true,"citations":[],"retried":true,"attempts":1,
       "can_retry":false,"query":"公司员工餐补标准","conversation_id":"…",
       "grounding":null}
```

拒答路径的 `grounding` 是 `null` 而不是一个空报告：没有回答就没有可核验的东西，
非流式与流式都保持同一个形状（前端只认一种判定）。

这段报文把四个"重试事实"都演示了：`retry` 事件先推（填住改写那 2~4 秒静默）、`sources` 为空、
`query` 已变成改写后的"公司员工餐补标准"、`can_retry=false`（额度 1 已用满）。
若首轮就命中，`attempts` 是 **0**、`retried` 为 `false`；若改写无产出（模型原样抄回问题），
`attempts` 会加到 1 但 `retried` 仍是 `false`——它只表示"真的用改写后的查询重检索过"。

`citations` 是**模型回答所依据的来源**，`sources` 是**送进模型的全部片段**。拒答时
`citations` 为空而 `sources` 仍保留——不作无依据的引用，但仍能看清检索到了什么。
`done` 的字段集合由 `_done_event` 唯一产出，成功与拒答两条路径完全一致（前端不用写两个解析分支）。

### 5.3 失败文档怎么重试（不用重新上传）

某个文件入库失败时，`documents` 表里留着一条 `status=failed`（`error` 里带稳定的 code），
**原始文件也留在 `data/uploads/{kb_id}/{doc_id}_{filename}`**。所以重试是原地重跑，
`doc_id` 不变、不新增记录：

```powershell
# 1) 看失败原因（error 里的方括号就是 code）
curl.exe -s "http://127.0.0.1:8000/api/v1/knowledge-bases/<kb_id>/documents"

# 2) 修好依赖（或换掉 uploads 里那份坏文件）后原地重试
curl.exe -s -X POST "http://127.0.0.1:8000/api/v1/documents/<doc_id>/reprocess"
# -> 200 {"id":"<同一个 doc_id>","status":"ready","chunk_count":4,"error":""}
```

返回语义与上传一致：**处理阶段再次失败仍是 200**，由 `status=failed` + `error` 说明原因；
只有下面三种情况才是 HTTP 错误：

| 情况 | 状态码 | 为什么 |
|---|---|---|
| 文档不存在 | 404 `not_found` | 传错 id，或已被删除 |
| 原始文件已丢失 | 404 `not_found` | 只能重新上传——`error` 里会提示"请重新上传" |
| `status=processing` | 409 `conflict` | 正在跑，重复触发会让同一份文档被写两遍向量 |
| `status=ready` | 409 `conflict` | 已成功，要重建索引请**先删除再上传**（重试 ≠ 重建） |

**哪些失败重试有意义**，看 `error` 开头的 code：

| code | 重试有用吗 | 该做什么 |
|---|---|---|
| `embedding_error` / `upstream_timeout` / `RuntimeError`（OOM 等） | ✅ 有用 | 修好依赖或调小 `EMBEDDING_BATCH_SIZE` 后重试 |
| `corrupted_document` / `encrypted_document` / `empty_document` | ❌ 没用 | 文件本身的问题，重试只会复现同样的报错——换文件重传 |
| `validation_error`（扩展名 / 体积超限） | ❌ 不产生记录 | 上传入口就被拒，压根没有 doc_id 可重试；先调 `MAX_UPLOAD_MB` 再传 |

> 为什么不让重试变成"重建索引"：重建需要**先清空再写**，清空到写入完成之间检索不到这份文档；
> 而失败文档本来就没有向量，不存在这个窗口。两种语义混在一个接口里，用户永远分不清自己点的是哪个。

### 5.4 端到端 HTTP 验证

```powershell
.\.venv\Scripts\python.exe scripts\verify_upload_flow.py
```

全程走 HTTP（自带服务启停、幂等），按 12 条探针逐一核对"问题 → 文件名 + 页码"：
3 份 PDF 夹具、每题的关键词在全库**唯一**，因此命中与否不可能模棱两可。
输出会把"未命中的探针"单列——那说明问题在检索侧，不是生成侧。

### 5.5 单库问答 CLI

```powershell
.\.venv\Scripts\python.exe scripts\ask.py "报销金额上限是多少？"              # 库里只有一个知识库可省略 -k
.\.venv\Scripts\python.exe scripts\ask.py -k <kb_id> --debug "设备多久巡检一次？"
.\.venv\Scripts\python.exe scripts\ask.py -k <kb_id> -i                       # 交互模式
.\.venv\Scripts\python.exe scripts\ask.py --list                              # 列知识库
```

它走 HTTP 而不是直接 import 容器：这是唯一同时经过接口层、服务层、状态图、混合检索、
重排与本地大模型的路径。进程内直调会绕过依赖注入、异常处理器与 SSE 编码——那些恰恰
最容易在重构中悄悄坏掉。

### 5.6 多轮会话

```powershell
# 首轮不带 conversation_id，服务端生成并回传
curl -s -X POST http://127.0.0.1:8000/api/v1/chat -H "Content-Type: application/json" `
  -d "@q1.json"        # {"kb_id":"...","question":"售后工单多久响应？"}

# 第二轮带上它，服务端自动读历史进提示词
#   q2.json: {"kb_id":"...","conversation_id":"<上一步返回的>","question":"那复杂问题呢？"}
curl -s -X POST http://127.0.0.1:8000/api/v1/chat -H "Content-Type: application/json" -d "@q2.json"

# 会话列表 / 详情（assistant 消息带 文件名+页码）/ 删除（幂等，204）
curl -s "http://127.0.0.1:8000/api/v1/conversations?kb_id=<kb_id>&keyword=工单"
curl -s "http://127.0.0.1:8000/api/v1/conversations/<id>"
curl -s -X DELETE "http://127.0.0.1:8000/api/v1/conversations/<id>"
```

### 5.7 登录（只发令牌，不校验）

```powershell
.\.venv\Scripts\python.exe scripts\create_user.py your-name --display-name 你的名字
curl -s -X POST http://127.0.0.1:8000/api/v1/auth/login -H "Content-Type: application/json" `
  -d "@login.json"     # {"username":"your-name","password":"..."}

# 正向验证：带上令牌，业务接口 200
curl -i http://127.0.0.1:8000/api/v1/knowledge-bases `
  -H "Authorization: Bearer <access_token>"

# 反向验证「鉴权已生效」：不带令牌 → 401（白名单只有 /health、/auth/login、/auth/register）
curl -i http://127.0.0.1:8000/api/v1/knowledge-bases

# 令牌是否还有效：前端启动时用它做校验
curl -i http://127.0.0.1:8000/api/v1/auth/me -H "Authorization: Bearer <access_token>"
```

---

## 6. RAGAS 评估结果

### 6.1 怎么跑

```powershell
cd <项目根目录>

# 全量（18 题 = 16 可答 + 2 应拒答；本机 7B 约 4.5 分钟）
.\.venv\Scripts\python.exe scripts\evaluate_rag.py

# 只算确定性检索指标（不调裁判模型，几秒）
.\.venv\Scripts\python.exe scripts\evaluate_rag.py --no-ragas

# 试跑 / 挑题 / 只校验数据集 / 换指标
.\.venv\Scripts\python.exe scripts\evaluate_rag.py --limit 3
.\.venv\Scripts\python.exe scripts\evaluate_rag.py --ids q02,q05,q14
.\.venv\Scripts\python.exe scripts\evaluate_rag.py --dry-run
.\.venv\Scripts\python.exe scripts\evaluate_rag.py --metrics faithfulness,context_recall --max-workers 1
```

RAGAS 的正确装法（**不要** `pip install ragas`，它会把 `openai` 与 `langchain-openai` 降级）：

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps `
    ragas==0.4.3 instructor appdirs diskcache nest-asyncio docstring_parser `
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

产物在 `eval/results/<时间戳>/`：`report.md`（结论）、`summary.json`（机器读）、
`samples.jsonl`（逐题含送进模型的上下文原文）、`scores.csv`（表格）。

### 6.2 最新结果

> 环境：`qwen2:7b`（裁判与被测同源）+ `bge-large-zh-v1.5`（CPU 编码）+ Milvus 3.0.2
> 配置：`top_k=8` / `rerank_k=4` / 权重 0.5:0.5 / `chunk_size=300`
> 结果目录：`eval/results/20261017-180621`（18 题，260 秒）

| 指标 | 值 | 有效样本 | 判据 |
|---|---:|---|---|
| 忠实度 faithfulness | **0.9896** | 16/16 | 回答里每句话能否被上下文支持（裁判 LLM） |
| 答案相关性 answer_relevancy | 0.7466 | 16/16 | 由回答反推问题，与原问题的语义相似度 |
| 上下文召回率 context_recall | 0.9375 | 16/16 | 标准答案的信息点被上下文覆盖的比例 |
| 上下文精确率 context_precision | 1.0000 | 16/16 | 上下文里有多少条真正有用 |
| **命中率 Hit@k** | **1.0000** | 16/16 | 期望的「文件名 + 页码」是否出现在引用里（纯比对） |
| **页码级召回率** | **1.0000** | 16/16 | 命中的期望来源数 / 期望来源总数 |
| **MRR** | **1.0000** | 16/16 | 首个命中来源的名次倒数 |
| 答出率 / 拒答正确率 | 1.0000 / 1.0000 | — | 16 道可答题都答了，2 道应拒答题都拒了 |
| 平均耗时 | 1.42 s | — | 最慢 10.3 s（首题含模型加载） |

**16 道可答题里 15 道忠实度满分**，唯一未满分的是 q12（0.8333）。

### 6.3 版本对比：提示词硬化把忠实度从 0.902 推到 0.990

| 版本 | 提示词要点 | 忠实度 | 相关性 | 命中率 | 答出率 |
|---|---|---:|---:|---:|---:|
| 基线 | 一句"只依据资料、答不了就拒答" | 0.9024 | 0.7868 | 1.0000 | 1.0000 |
| v1 | 6 条编号禁令 + 条件句写来源标注 | 0.8594 | 0.6301 | **0.9375** | 0.9375 |
| v2 | +「完全无关才拒答」+「不是拒答的理由」 | 0.9062 | 0.6493 | **0.9375** | 0.9375 |
| v3 | 删掉行内 `(来源：[n])` 要求 | 0.7812 | 0.6390 | 1.0000 | 1.0000 |
| v3b | 去掉「不要用『资料未说明』代替」的歧义句 | 0.8576 | 0.6468 | 1.0000 | 1.0000 |
| **最终** | 允许留白，**但不给句式** | **0.9896** | 0.7466 | 1.0000 | 1.0000 |

四轮都踩了不同的坑，值得记住的三条：

1. **正面要求必须写成无条件祈使句**。写成"若资料带有来源标注，则…"，7B 直接当作前提不成立，
   16/16 道题都不标序号，还连带整段约束不执行（q02 上下文完全正确却拒答）。
2. **不要留一条模型必然违反的指令**。行内 `(来源：[n])` 试过三种写法，遵守率始终不稳定
   （16/16、13/18 不写），于是整段删掉——结构化 `citations` 已经给了文件名 + 页码。
   留一条必然被违反的指令，只会让"提示词遵守度"这个信号永久噪音化。
3. **能照抄的句子一定会被照抄**。第 5 条曾写"其余明确写『资料未说明』"，而 system 与 user 里
   各出现一次 → 模型把这句占位话术当成答案本身：q05 资料明写"48 小时内首次响应"，
   整题只回五个字（2×2 因子实验 **5/5**；删掉任意一半即 0/5）。
   **留白可以，留句式不行。**

### 6.4 口径与局限（别把数字当对外结论）

1. **裁判就是被测模型**（同一个 7B）。自评偏宽松，绝对分数只适合**版本间横向比较**。
2. **`answer_relevancy` 惩罚简洁**。q08 现在是"需要向厂商订货，发出时间会另行通知"（忠实且正确），
   基线则编了一套 5 步流程（资料里没有）。前者被判"含糊其辞"打 0，这就是它 0.747 < 0.787 的唯一原因
   —— 在"编一套流程"和"相关性低一点"之间，本项目选后者。
3. **两套召回率不能互相替代**：`context_recall` 看语义覆盖，页码召回看溯源是否点对地方。
   后者不调模型、可复现，做版本对比优先看它。
4. **`context_recall` 1.0 → 0.9375 与提示词无关**：基线跑在跨副本去重修复之前，q10 的上下文是
   4 条逐字相同的片段；去重后只剩 1 条，语义覆盖判定随之变化。
5. **语料只有 3 篇 / 15 块**，指标偏乐观。真要做选参，得用合成硬负样本基准（见 6.5）。

### 6.5 参数调优结论（现有配置已是最优，一处未改）

真实语料**测不出区别**（3 篇 / 15 片段 / 4 事实、每事实独占一页 ⇒ top_k 4..16 × 4 组权重
**20 组全 1.0**），所以另建合成基准：**24 台设备 × 同一组 4 个参数名**（措辞逐字相同，
只有型号与数值不同）= 硬负样本。

| 维度 | 扫描结果 | 结论 |
|---|---|---|
| 向量 vs 混合 | 纯向量 MRR **0.8368** vs 混合 **0.9792** | 混合检索的价值 +0.14，**且与 top_k 无关**（差距来自 BM25 命中型号这类稀疏 token） |
| `RETRIEVE_TOP_K` | 2→0.9167 / 4→0.9583 / 6,8,12,16→0.9792 | ≥6 饱和，保留 8 |
| 两路权重 | 0.5/0.5 与 0.3/0.7 同分，0.7/0.3 略低 | 保留 0.5/0.5 |
| `CHUNK_SIZE` | 150 与 300 指标相同但块数 ×3；**500 → MRR 0.8194** | 保留 300/50 |
| `RERANK_TOP_N` / 阈值 | 12 组完全并列，且与"不重排"一致 | 基准测不出纯度收益；`RERANK_MIN_SCORE=0.0` 维持（2.0 会在歧义查询上误杀） |

复跑：`.\.venv\Scripts\python.exe scripts\tune_retrieval.py`（四 stage：real / recall / rerank / chunk）。
完整数据表见 [`docs/tuning.md`](docs/tuning.md)，评估口径见 [`docs/evaluation.md`](docs/evaluation.md)。

---

## 7. 后续迭代方向

按"影响面 × 现在做的成本"排序。**触发条件**一栏是判断"该不该现在做"的依据——
没有触发条件就先不做，避免为了完整而完整。

### P0 — 上生产前必须有

| # | 方向 | 现状 | 要做什么 | 触发条件 |
|---|---|---|---|---|
| 1 | ~~**鉴权挂载**~~ ✅ | 已完成：除 `/health`、`/auth/login`、`/auth/register` 外全部要求令牌；知识库按 `owner_id` 隔离，"不存在"与"不是你的"返回逐字节相同的 404（防枚举）。`check_auth`（136 项）断言白名单外**每条**路由都挂了鉴权 | 已落地 | 已完成 |
| 2 | **重启孤儿记录** | 进程在"处理中"被 kill，SQLite 留一条永远 `processing`，前端一直转圈 | `lifespan` 启动时扫 `status=processing` → 一律标 `failed("服务重启时仍在处理中")`；自检预置一条再重启断言 | 发生过一次非正常退出 |
| 3 | ~~**入库失败清理自检**~~ ✅ | 已有 `check_ingest.py`（51 项）：首次失败断言 `delete_by_doc` 被调用、`mark_document_failed` 被调用、原始文件保留；批次级部分失败由 `check_vectorstore` 覆盖 | 补强方向：`upsert` 自身在第 2 批抛错时，断言清理与最终状态不被二次异常改写 | 已完成 |
| 4 | **重试的异步化** | `POST /documents/{id}/reprocess` 是**同步**的，大文件重试可能几分钟（与上传同一个问题） | 改成 `BackgroundTasks` + 立即返回 `processing`，前端轮询文档状态；同时给解析/向量化加 `INGEST_TIMEOUT_S`，超时标 failed 但可重试 | 出现"重试请求被网关/浏览器超时切断，但后端仍在跑"的个案 |

### P1 — 效果与正确性的下一步

| # | 方向 | 现状 | 要做什么 | 触发条件 |
|---|---|---|---|---|
| 4 | **结构感知切分** | 按字符数切，会把几个 `##` 小节揉进一块；重排对被无关文字包围的块打负分（同一句单独成块 +0.63、混进 272 字长块 −3.99） | 按标题切（Markdown/PPTX 有天然边界），`CHUNK_SIZE` 只作上限 | 出现"两路都召回了正确片段，最终 `results` 却是空"的个案 |
| 5 | **重排阈值策略** | `RERANK_MIN_SCORE=0.0` 是硬过滤，歧义口语查询下相关页可能拿负分被整条丢弃 | 保底策略：至少保留 top-1；或用"相对分数 + 绝对下限"双条件 | 出现"有资料却拒答"，且 `/search` 里相关页 `rerank_score` 为负 |
| 6 | **检索可观测性补全** | `/search` 的 debug 只回填**存活**候选的 `rerank_score`，"被阈值丢"与"被 `rerank_k` 截断"分不开 | debug 里加 `dropped_by_threshold` / `dropped_by_rerank_k` 两个列表 | 调 `RERANK_MIN_SCORE` 时说不清影响面 |
| 7 | **增量更新** | 重传即新文档（同内容会被去重，不新增块） | 文档指纹 + 块级 diff：只重建变化的块 | 文档会被反复小改；或库里文档数 > 几百 |
| 8 | **评测集扩充** | 18 题（16 可答 + 2 应拒答），语料 3 篇 | 50+ 题，覆盖跨文档推理、多轮指代、表格数值、超长文档、应拒答样本到 10 题 | 要做版本间的显著性判断，而不只是"看起来更好" |
| 9 | **语义级核验（D 型幻觉）** | 硬事实核验抓不到"过度推断"：实测「备件能加急吗」答出"可以尝试沟通缩短 5 个工作日"，数字全对、语义多推一步，核验照样 `ok` | 加 NLI / LLM-as-judge 复核，**并且**先做 30 条人工标注的对照实验量它的召回与误报 | 有实测表明误报可接受；否则只是拿延迟换一个不可信的信号 |
| 10 | **整批最高分闸门标定** | `RERANK_MIN_TOP_SCORE` 默认关。本机 6 条负样本里只有 1 条落在正分区间（1.048 vs 正样本最低 1.797） | 补到 30+ 条"主题相邻、答案没有"的负样本，用 `scripts/probe_rerank_gap.py` 重标 | 负样本 ≥ 20 条且两组不重叠 |

### P2 — 能力扩展

| # | 方向 | 说明 |
|---|---|---|
| 9 | **多知识库路由** | 现在按 `kb_id` 显式指定；可加"跨库检索 + 库级权重"，或先让 LLM 判断该查哪个库 |
| 10 | **元数据过滤** | 按文档类型/部门/生效日期过滤（Milvus 标量字段已有基础），避免在过期制度里找答案 |
| 11 | **精确登出** | 引入令牌黑名单（Redis，TTL = 剩余有效期）或会话表；顺带支持"踢下线" |
| 12 | **前端联调** | `rag-frontend` 已按 `/chat/stream` 事件序对接；下一步是引用跳转（点页码打开原文对应页）与多轮会话列表 |
| 13 | **回答格式** | 表格类问题可要求结构化输出（Markdown 表格），并在前端渲染 |

### P3 — 工程与性能

| # | 方向 | 说明 |
|---|---|---|
| 14 | **编码性能** | 现在 BGE 走 CPU（GPU 被 7B 占满）。可改 GPU + 批处理，或加 embedding 缓存（同一段文本反复编码） |
| 15 | **并发与限流** | 单进程 + 本地 7B，并发问答会互相排队；需要队列/信号量 + 429 背压 |
| 16 | **可观测性** | 结构化日志已有；下一步是 trace id 贯穿入库/检索/问答，接入 OTel，把 `timings_ms` 打点上报 |
| 17 | **依赖可复现** | `requirements.txt` 是区间约束；建议 `pip freeze` 出 lock 文件，并加 CI 跑全部 1066 项自检 |

---

## 附录 A. 自检清单（1066 项断言）

> 为什么是**独立脚本**而不是 pytest：每个脚本都能单独跑、依赖各不相同（有的要 Milvus，
> 有的要 BGE 权重），失败时直接退出码 1，不需要装 pytest 插件、也不要求 CI。
> 仓库里没有 pytest 用例，`tests/` 只放自检用的样例文档。

```powershell
# 全部（有依赖的脚本需 Milvus / Redis 在跑）
.\.venv\Scripts\python.exe scripts\check_loading.py        #  45
.\.venv\Scripts\python.exe scripts\check_splitting.py      #  94
.\.venv\Scripts\python.exe scripts\check_models.py         #  23
.\.venv\Scripts\python.exe scripts\check_embedding.py      #  63
.\.venv\Scripts\python.exe scripts\check_vectorstore.py    #  84  ← 需 Milvus
.\.venv\Scripts\python.exe scripts\check_retrieval.py      # 128  ← 需 Milvus
.\.venv\Scripts\python.exe scripts\check_ingest.py         #  59
.\.venv\Scripts\python.exe scripts\check_filesafety.py     #  39
.\.venv\Scripts\python.exe scripts\check_grounding.py      #  45
.\.venv\Scripts\python.exe scripts\check_chat.py           # 246
.\.venv\Scripts\python.exe scripts\check_auth.py           # 130
.\.venv\Scripts\python.exe scripts\check_conversations.py  #  53  ← 第 3 组需 Redis
.\.venv\Scripts\python.exe scripts\check_eval.py           #  57
```

| 脚本 | 覆盖什么 | 依赖 |
|---|---|---|
| `check_loading` | 五种格式加载、页码、中文编码、表格完整性、空/损坏/加密/扫描件语义 | 无 |
| `check_splitting` | 页码溯源、不跨页、内容无丢失、无捏造、块大小与重叠、`page_no=None` 透传 | 无 |
| `check_models` | 端口与 schema 契约、维度来源与交叉校验、L2 归一化、确定性 | 无 |
| `check_embedding` | 维度取自权重、query 侧才加检索指令、空输入拒绝、跨批一致、三类失败语义 | BGE 权重 |
| `check_ingest` | 首次入库状态机、失败原因带 code、**原地重试**（同一 doc_id 不新建记录）、原始文件复用与丢失语义、路径安全、**正文哈希去重 + 并发闸门 + 锁回收** | 无 |
| `check_filesafety` | 文件名控制字符/超长、PDF 魔数与页数、zip 条目数/解压总量/压缩比、文本 NUL 与控制字符占比、**服务层接线**（被拦下时不产生元数据行） | 无 |
| `check_vectorstore` | schema 契约、分批写入与 flush、字节级截断、ID 前置校验、部分失败清理 | Milvus |
| `check_retrieval` | 契约 / 纯函数 / 行为 / 编排四层 + **跨副本去重** + **失败语义**；核心断言是"关键词文档纯向量召不回、混合召得回" | Milvus |
| `check_grounding` | **接地性核验的判据**：归一化（千分位/全角/大小写/连字符）、数字边界、列表序号与引用序号不误报、分级（ok/warn/low）、三档动作、流式下 block 退化为 flag、整批最高分闸门 | 无（全假上游） |
| `check_chat` | 提示词口径、路由三分支、图四条路径、引用映射、LLM 错误映射、SSE 事件序、接地性审计 | 无（全假上游） |
| `check_auth` | 哈希自描述、令牌签发/验签、防账号枚举、注册开关、**白名单外每条路由都挂了鉴权** | 无 |
| `check_conversations` | 键设计与 TTL 续期、历史进提示词、落库两轮、503 语义 | Redis（第 3 组） |
| `check_eval` | 页码标签解析、命中判定（文件名对但页码错**不算**命中）、聚合口径 | 无 |

### 附录 A.1 工具脚本（不是自检）

| 脚本 | 用途 | 依赖 |
|---|---|---|
| `scripts/dev.py` | 开发服务器 + **确定性热重载**（替代在 Windows 上会卡死的 `uvicorn --reload`，原理见 4.5 的警告） | 无 |
| `scripts/ask.py` | 命令行问答（`--demo` 走服务函数、不带 `--demo` 走 HTTP） | 服务在跑 |
| `scripts/evaluate_rag.py` | RAGAS 四指标 + 自算页码级命中/MRR | 服务 + LLM |
| `scripts/probe_rerank_gap.py` | 量正/负样本的重排分间隔，给 `RERANK_MIN_TOP_SCORE` 标定 | Milvus + BGE |
| `scripts/backup.py` / `rebuild_vectors.py` | 备份/恢复/校验、从 uploads 重算向量 | 见 `docs/backup.md` |

## 附录 B. 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/v1/health` | 存活检查 |
| GET | `/api/v1/ready` | 就绪检查（Milvus + 权重目录） |
| POST | `/api/v1/auth/register` | 注册（`AUTH_ALLOW_REGISTRATION=false` 可关） |
| POST | `/api/v1/auth/login` | 登录，换发令牌（**不校验令牌**） |
| GET / POST | `/api/v1/knowledge-bases` | 知识库列表 / 新建 |
| PUT / DELETE | `/api/v1/knowledge-bases/{kb_id}` | 更新 / 删除（级联清向量与文件） |
| GET / POST | `/api/v1/knowledge-bases/{kb_id}/documents` | 文档列表 / 上传单个（201） |
| POST | `/api/v1/knowledge-bases/{kb_id}/documents/batch` | 批量上传（200，逐文件成败） |
| POST | `/api/v1/documents/{doc_id}/reprocess` | 重试失败的文档（复用已落盘原件，doc_id 不变） |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档（连带向量与原文件） |
| POST | `/api/v1/knowledge-bases/{kb_id}/search` | 混合检索（`debug=true` 出逐路明细） |
| POST | `/api/v1/chat` | RAG 问答（非流式，带 citations + `grounding` 接地性核验报告） |
| POST | `/api/v1/chat/stream` | RAG 问答（SSE，`done` 事件带 `grounding`；token 已发，只能事后标记） |
| GET | `/api/v1/chat/options` | 流式重试默认值 / 上限 / 事件名 / 当前 `grounding_action` |
| GET | `/api/v1/conversations` | 会话列表（`?kb_id=&keyword=`） |
| GET / DELETE | `/api/v1/conversations/{id}` | 会话详情 / 删除（幂等，204） |

统一错误体：`{"code", "message", "detail"}`，`code` 为机器可读码
（`validation_error` / `unsafe_file` / `not_found` / `conflict` / `payload_too_large` /
`document_error` / `corrupted_document` / `encrypted_document` / `empty_document` /
`embedding_error` / `rerank_error` / `retrieval_error` / `upstream_error` /
`upstream_timeout` / `service_unavailable` / …）。

| 状态码 | 含义 | 客户端动作 |
|---:|---|---|
| 400 | 入参/内容不合法（含 `unsafe_file`：魔数不符、压缩炸弹） | 换文件或改参数，**不要重试** |
| 409 | 状态冲突（重复上传同一正文、文档正在处理、已 ready 又重试） | 按 `detail.doc_id` 用已有资源，或换操作 |
| 413 | 请求体超过上限（中间件按 `Content-Length` 拦，未落盘） | 压缩 / 拆分 / 分批 |
| 502 / 504 / 503 | 依赖坏了 / 依赖太慢 / 能力没开 | 修依赖 / 调超时 / 补配置 |

## 附录 C. 约定与踩坑速查

**约定**

- 配置只从 `Settings` 读，业务代码里不出现 `os.getenv`。新增数值配置顺手加区间校验。
- 机器无关常量（Milvus 地址、切分参数、阈值）可带默认值；**机器相关或涉密值一律留空**、
  只从 `.env` 读（已 gitignore）。
- 向量维度以**权重**为准；换模型必须 `init_milvus.py --recreate --yes`。
- 查询与文档分开编码：BGE 的检索指令只加在 query 前。
- 长块是检索杀手：中文 `CHUNK_SIZE` 建议 150~300。

**高频坑（每一条都真的踩过）**

| 坑 | 表现 | 处置 |
|---|---|---|
| 加 `sparse` 后没传 `anns_field` | `multiple anns_fields exist`，报错完全不提"你新加了字段" | 稠密路显式 `anns_field="embedding"` |
| 混合模式给稠密路设阈值 | 稠密路被**静默清空**，BM25 掩盖了它 | 用 `HYBRID_DENSE_MIN_SIMILARITY=0.0` |
| 中文分析器不过滤停用词 | 只共享一个「的」也能拿到 BM25 正分（一次实测：真命中 2.40 vs 噪声 1.01；本次语料是 9.84 vs 1.65） | 默认不设 BM25 阈值，精度交给重排 |
| Milvus `VARCHAR` 按字节计 | 30000 汉字（90000 B）被拒；按字符截等于没截 | 按**字节**截且不切断多字节（`text` 65535 / `filename` 512） |
| 分隔符处空白被吞 | 拼接所有块 ≠ 原文 | 逐字断言前先做空白归一化 |
| 批形状不同 | float32 末位差异（1.19e-7），golden 断言失败 | 断言留容差 |
| `use_fp16` 默认 True | CPU 上抛 `LayerNormKernelImpl not implemented for 'Half'` | 按设备显式传 |
| `TextLoader(autodetect_encoding=True)` | 直接崩（chardet≥5.2 API 变了） | 按候选表自解后显式传 `encoding` |
| 配置错误被兜底成 502 | 运维去重启 Ollama，而该改的是 `.env` | `AppError` 原样放行 |
| SSE 断开的清理 | `GeneratorExit`/`CancelledError` 是 `BaseException`，`except Exception` 拦不到 | 生产路径是 cancel 而非 `aclose()`；两层都要测 |
| 子串区分提示词类型 | `"改写" in system_prompt` 把生成请求误判成改写 | 用 `== REWRITE_SYSTEM` 恒等比较 |
| Git Bash heredoc | 反斜杠被重写成正斜杠，含 Windows 路径的替换静默失配 | 用编辑工具改，别走 heredoc |
| Windows 上临时 SQLite | `PermissionError [WinError 32]` | 先 `database.dispose()` 再删目录 |

文档索引：[`docs/architecture.md`](docs/architecture.md) 架构与决策表 ·
[`docs/data-flow.md`](docs/data-flow.md) 两条链路时序 ·
[`docs/pipeline.md`](docs/pipeline.md) 逐阶段实测 ·
[`docs/auth-and-isolation.md`](docs/auth-and-isolation.md) 鉴权与多租户隔离 ·
[`docs/evaluation.md`](docs/evaluation.md) 评估口径与提示词迭代 ·
[`docs/tuning.md`](docs/tuning.md) 调优方法论与数据表 ·
[`docs/backup.md`](docs/backup.md) 备份与恢复
