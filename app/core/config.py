"""应用配置：从环境变量与 .env 加载，供全项目统一读取。

设计要点：
    - 使用 pydantic-settings，优先级为「环境变量 > .env > 类默认值」。
    - extra="ignore"：忽略未声明字段，避免部署环境多余变量导致启动失败。
    - get_settings() 带 lru_cache，全局单例；测试中可用 cache_clear() 重置。
    - 所有业务模块一律通过 Settings 读取配置，禁止散落 os.getenv。

默认值约定（重要，勿随意放宽）：
    Settings 里的字段分两类，判断标准是「换一台机器要不要改」。

    1. **与机器无关的常量** —— 可以带真实默认值，开箱即跑。
       例如 Milvus 默认主机/端口、切分参数、各类阈值。

    2. **与机器或后端相关的值** —— 类默认值**一律留空**，只从 .env /
       环境变量读取。包括：模型权重目录、LLM 端点与模型名、API Key、
       令牌签名密钥（AUTH_SECRET_KEY —— 密钥属于「这次部署的事实」，
       写进要入库的代码里等于把它交给所有拿到仓库的人）。
       理由：
         a. `.env` 已被 .gitignore 忽略，而本文件要入库；
         b. 这些是「某台机器上的事实」，换机器该改环境而不是改代码；
         c. 空默认值 + validate_runtime() 启动期校验，能让配置缺失在
            启动第一秒就暴露。

       反面教材：曾经在代码里给 `llm_base_url` / `llm_model` 填了真实值，
       却把模型权重目录留空。后果是**失败模式不一致**——没有 .env 时
       服务照常启动、LLM 照常回答，但 embedding 以空路径加载，
       向量检索整体失效且没有显式报错。这类「半可用」状态比直接崩更难查。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# 召回/重排条数的服务端硬上限。**请求体 schema 与服务层共用这两个数**：
# 若 HTTP 允许 100、服务层却按 50 处理，同一个请求走接口与走脚本会拿到
# 不同的条数，而这种差异只会在排查"为什么少了两条"时才被发现。
# 放模块级而不是 Settings 字段，是因为它们与机器/部署无关，也不该被 .env 调。
TOP_K_HARD_LIMIT = 100
RERANK_K_HARD_LIMIT = 50


class Settings(BaseSettings):
    """全局配置。字段名对应同名环境变量（大小写不敏感）。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- 应用 ----------
    app_name: str = "rag-backend"
    app_env: Literal["local", "dev", "test", "prod"] = "local"
    # 实际挂载路由用的前缀只有这一个（main.py 的 include_router）。
    # 曾经还有一个 api_prefix="/api"，全仓库没有任何地方读它——已删。
    api_v1_prefix: str = "/api/v1"
    debug: bool = True
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # ---------- Milvus ----------
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    milvus_user: str = ""
    milvus_password: str = ""
    milvus_collection: str = "rag_backend_chunks"
    # **声明值**，不是真相来源：真实维度由 embedding 权重的 config.json 决定。
    # 两者不一致时启动即报错（见 infrastructure/embedding.py 的 _resolve_dimension）。
    milvus_dimension: int = 1024
    milvus_metric_type: str = "COSINE"
    milvus_index_type: str = "AUTOINDEX"
    # 单次写入 Milvus 的片段条数。分批是为了压住请求体大小与峰值内存
    # （1024 维 float32 每条约 4 KB，三万块一次性发出就是 ~120 MB），
    # 同时让长文档的入库有可报告的进度。上限 2000（见 vectorstore.py）。
    milvus_upsert_batch_size: int = 500

    # ---------- Embedding ----------
    # 必填：本机 BGE 权重目录，无默认值，必须由 .env / 环境变量提供
    embedding_model_path: str = ""
    embedding_device: str = "cpu"
    embedding_query_instruction: str = ""
    embedding_batch_size: int = 32

    # ---------- Rerank ----------
    rerank_enabled: bool = True
    # rerank_enabled=true 时必填，无默认值
    rerank_model_path: str = ""
    rerank_device: str = "cpu"
    rerank_top_n: int = 4
    rerank_min_score: float = 0.0
    # 「整批最高分」闸门：重排后所有片段里最高的那条都低于它，就当作
    # **没有可用资料**（清空 ranked → 走改写重试 / 拒答），而不是拿一条
    # 勉强及格的片段去生成。
    #
    # 为什么与 RERANK_MIN_SCORE 是两回事：
    #   RERANK_MIN_SCORE 是**逐条**过滤，闸门设在 0.0 时它只挡负分；
    #   但"主题相邻、答案没有"的问题，重排常给出**正分**（字面重叠多、
    #   语义不匹配），逐条闸门放它过去，模型就会拿一段周边规则硬答。
    #   实测本机语料（3 篇文档、18 题）：
    #     可答题 top1 重排分 1.797 ~ 8.563（最低 q14）
    #     硬负样本「备件发货可以加急吗？」top1 = 1.048（召回到备件发货那页，
    #       但那页没写加急）—— 7 个其它负样本直接零召回，被 0.0 逐条闸门挡住
    #   即：**负样本落在正分区间**是真实存在的，只是样本太少不足以定死阈值。
    #
    # 默认 0.0 = 关闭。要开就在自己的语料上先量分离度
    # （scripts/probe_rerank_gap.py），别照抄别人的数字：
    # 交叉编码器的 logits 尺度随模型与语料变化，本机 bge-reranker-base 的地板约 -10.2。
    rerank_min_top_score: float = 0.0

    # ---------- LLM ----------
    # 必填：端点与模型名都属于「后端相关」的值，无默认值
    llm_base_url: str = ""
    llm_model: str = ""
    # 本地后端（Ollama / vLLM）不校验密钥，可留空；云服务必须填
    llm_api_key: str = ""
    llm_temperature: float = 0.2
    llm_timeout: int = 120
    # 输出预留（skill 里的 R）：既作为 `max_tokens` 下发给模型，
    # 也从上下文预算里先扣掉。不设它，答案被后端截断时系统完全不知情——
    # 只有"空串"会被判成异常（见 infrastructure/llm.py）。
    #
    # 2048 的依据：skill 建议 R ≥ 2000；本机 7B 的答案实测在千字以内，
    # 2048 token 足够，又不会把 8k 窗口吃掉一半以上。
    llm_max_tokens: int = 2048
    # 输出上限在请求体里的**字段名**。
    #
    # 为什么必须可配：langchain-openai 1.7 会把 `max_tokens` 参数改写成
    # `max_completion_tokens`，而 Ollama / vLLM 的 OpenAI 兼容层只认
    # `max_tokens`——实测（Ollama 0.40.5）传 `max_completion_tokens=8`
    # 时限制被**完全忽略**：模型输出了 477 token，finish_reason 仍是 stop。
    # 也就是说：不换字段名，输出预留等于没设，而且截断永远不会被标记。
    #
    # 本地后端（Ollama / vLLM / LM Studio）→ max_tokens
    # OpenAI 云端较新的模型（o 系列等）→ max_completion_tokens
    llm_max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"

    # ---------- 上下文预算（见 rag/budget.py）----------
    # **实际服务窗口**，不是界面/文档里的声明值：Ollama 要填它的 num_ctx，
    # vLLM 要填 --max-model-len。填大了 → 提示词超窗，后端静默截断，
    # 预算的全套计算跟着失效；填小了 → 证据被过度裁剪。
    llm_context_window: int = 4096
    # 证据占"窗口-输出预留-固定开销"的比例。skill：小窗口 4k~16k 取 0.4~0.45，
    # 大窗口可到 0.6；0.5 是通用起点。
    context_evidence_share: float = 0.5
    # 安全余量（占剩余预算的比例）。tokenizer 近似、模板差异都靠它兜底。
    context_margin_ratio: float = 0.05
    # 每条证据的元数据开销（"[n] 来源：文件名 第N页" + 两个换行）。
    # skill：20~40 token；10 条就是 240，装箱时不能忽略。
    context_meta_tokens: int = 24
    # 单条证据参与预算的最大字符数。必须与 prompts.format_context 的截断
    # 口径一致，否则"按全文计数、按截断渲染"会让预算虚高。
    context_snippet_limit: int = 1200
    # 证据条数上限（skill 的 k_cap：单点事实 3~4、解释 6~8、对比 8~10，默认 12）。
    # 预算够**不代表**所有碎片都该进：长上下文正中间的证据会被模型忽略
    # （lost in the middle），多出来的条数还会按 token 线性推高成本。
    context_max_evidence: int = 12
    # token 计数策略：auto（本地 tokenizer → tiktoken → 字符估算）
    # / tokenizer / tiktoken / chars。字符估算误差可达 30%，只应作为兜底。
    token_estimator: Literal["auto", "tokenizer", "tiktoken", "chars"] = "auto"
    # 本地 tokenizer 目录（如 Qwen2 的 tokenizer.json 所在目录）。
    # 留空则 auto 走 tiktoken。**它应当与 LLM_MODEL 同源**，否则算的是别的模型。
    tokenizer_path: str = ""
    # 字符估算的换算系数。中文一 token 约 1.2~1.5 个汉字；
    # 取 1.2 是刻意高估 token（宁可少装一条，也不要被后端截断）。
    chars_per_token: float = 1.2

    # ---------- 检索 ----------
    # 每路召回的深度（稠密路与稀疏路各取这么多条），也是融合后送进重排的条数。
    retrieve_top_k: int = 8
    # 混合检索开关。关闭时只跑向量路——**但 schema 里仍有 BM25 字段与索引**：
    # 建表属性不该随运行期开关漂移，否则「改一行配置就得重建 collection」，
    # 这会变成一个很容易踩的坑（见 vectorstore.py 的 ensure_collection）。
    hybrid_enabled: bool = True
    # RRF 平滑常数：越大越压平排名差距，越依赖"两路都出现"这个信号
    hybrid_rrf_k: int = 60
    # 两路权重。**只有比值有意义**：0.5/0.5 与 1/1 排序完全相同，
    # 差别只是 fusion_score 的绝对大小。两者不能同时为 0。
    hybrid_vector_weight: float = 0.5
    hybrid_bm25_weight: float = 0.5
    # 稠密路的余弦相似度下限。**注意两种模式下的默认值不同，这不是笔误**：
    #
    #   纯向量模式（HYBRID_ENABLED=false）：这是唯一的精度闸门，必须设。
    #   混合模式：精度交给下游的交叉编码器重排，这里默认**不设下限**。
    #
    # 理由是实测出来的：中文短查询对 300 字长块，余弦相似度本就落在 0.28~0.35，
    # 用 0.35 当门槛会把整条稠密路**静默清空**——
    #   查询                     片段0    片段1    过 0.35？
    #   多久需要做一次检查？      0.2804   0.2936   ✗ 全被滤掉
    #   巡检                     0.3081   0.1043   ✗
    #   XG-8800                 0.3135   0.0007   ✗
    #   设备的固件怎么升级         0.5721   0.2554   ✓
    # 更隐蔽的是：混合检索会**掩盖**这个故障——BM25 路还在出结果，
    # 从返回内容上看不出"向量路已经一条都没召回"。只有 `debug=true`
    # 打印逐路明细才会暴露。
    # 而且给融合的两路只设一边的阈值本身就不对称：BM25 分无界、无法设阈值，
    # 稠密路设了阈值就变成"两路召回深度不等价"，RRF 的排名语义被破坏。
    min_similarity: float = 0.35
    # 混合模式（HYBRID_ENABLED=true）下稠密路的相似度下限。默认 0.0 = 不设限。
    # 改成 >0 请先看上面那张实测表：调高会让稠密路在短查询下静默失活。
    # 留这个开关而不是写死，是为了在"库很大、召回噪声确实过多"时仍有一条退路。
    hybrid_dense_min_similarity: float = 0.0
    # 稀疏路的 BM25 分下限。默认 **0.0 = 不过滤**，理由是 BM25 分无界且
    # 随语料（文档数、平均长度）漂移：写死一个绝对阈值，换一份语料就失效。
    #
    # 实测噪声形态（`XG-8800 的固件怎么升级`，语料 15 条）：
    #   真命中 0.68 / 只共享一个「的」的无关片段 0.55 —— 差距很小，
    #   设 0.6 会连真命中一起砍掉。所以默认交给下游交叉编码器重排。
    # 只有在"库很大、且已确认重排前候选被噪声挤爆"时才设，取值用
    # scripts/tune_retrieval.py 打印的真实分数分布来定。
    hybrid_bm25_min_score: float = 0.0
    # 近重复判据（字符三元组 Jaccard）。0 = 关闭。
    #
    # 解决的是"同名不同内容才拦得住"之外的残余：另存过的 PDF、重新解析出
    # 的同一段正文，字节不同但文本几乎一样——上传层的内容哈希拦不住它，
    # 融合层的精确去重（文件名+正文）也拦不住（文件名可能变了）。
    # 判定时**要求数字/型号完全一致**，所以「48 小时」与「24 小时」不会被误合并。
    # 语料里出现"同一条款多个版本"时先调低到 0.9 观察，别直接关掉。
    hybrid_near_dupe_ratio: float = 0.95
    # 近重复合并的最小正文长度（折叠空白后计）。短于此长度的相同文本**不合并**：
    # 「同一段公告」被两篇文档引用是合法的，只有成段复制才值得当作副本。
    # 60 的取值：一句话以上的成段复制才合并；
    # 更短的公告引用一律保留。
    hybrid_near_dupe_min_chars: int = 60

    # ---------- 动态 k（自适应召回深度）----------
    # 每路召回深度怎么定：
    #   fixed    —— 固定取调用方要求的 top_k（请求里的，或上面的 RETRIEVE_TOP_K）。
    #               **默认值**，与历史行为逐位一致，既有评估基线不用重跑。
    #   adaptive —— 先取 min(top_k, RETRIEVE_K_MAX) 条，再按**每路自己的分数**
    #               找断层，截到 [RETRIEVE_K_MIN, 上面那个上限]。
    #
    # 为什么默认 fixed：这个开关会改变"每次请求到底返回几条"，而 README 里的
    # MRR 数字是在固定 k 上量出来的。换了口径却不重跑评估，等于拿旧分数解释
    # 新行为。开启前先跑 scripts/tune_retrieval.py 看真实分数分布。
    retrieve_k_mode: Literal["fixed", "adaptive"] = "fixed"
    # adaptive 的下限：宁可多留几条，也不要因为一条断崖切到只剩 1 条。
    # 注意它**不会**把 top_k 抬上去——本来只有 3 条时就是 3 条。
    retrieve_k_min: int = 4
    # adaptive 的上限，同时是过采样的条数。它**不放大** top_k：
    # 实际取 min(top_k, RETRIEVE_K_MAX)。想让动态范围摸到 16，就把
    # RETRIEVE_TOP_K 设到 16（或请求里传 16），而不是只调大这一项；
    # 它真正的用途是拦住"请求显式传 top_k=100"把重排候选撑爆的情况。
    retrieve_k_max: int = 16
    # 断层判据：相邻两条的落差 ≥ 该比例 × 本路最高分，才算"这里换主题了"。
    # 余弦的量级随查询变化（中文短查询整体 0.28~0.35），只能看相对落差；
    # 0.25 是保守值，只把断崖式落差当断层。具体语料上应先看分数分布再定，
    # 别照抄这个数字——它和 RERANK_MIN_SCORE 一样是"语料相关"的。
    retrieve_k_drop_ratio: float = 0.25

    # ---------- 问答 ----------
    # 流式问答「首轮检索没找到时，是否允许改写查询再检索一次」的**服务端默认值**。
    #
    # 请求体里的 `enable_retry` 是**三态**的，这里只负责第三态：
    #     true / false → 前端把「用户的选择」原样传上来，服务端照办；
    #     省略 / null  → 落到这个默认值。
    # 这样拆开的好处是两件事互不干扰：产品/运维改默认行为不必动前端代码，
    # 而用户在界面上临时改主意也一定能压过默认值（见 check_chat.py 第 13 组）。
    # 若把默认值写死在 schema 里，前端要显示开关的初始状态就只能自己抄一份，
    # 抄的那份迟早与后端漂移——所以另有 GET /chat/options 供前端读取。
    #
    # 与机器无关 → 给真实默认值。**默认关**：一次改写要等 LLM 生成一轮查询
    # （本机 7B 约 2~4 秒），而用户此时盯着一个空白窗口，首帧时间优先。
    chat_stream_retry_default: bool = False

    # 生成之后的接地性核验动作。三档，语义见 app/rag/grounding.py：
    #   off   —— 不核验（只用于对照实验）
    #   flag  —— 只产告警、回答照常返回（**默认**）
    #   block —— 判定为 low 时把回答换成拒答话术（仅非流式生效）
    #
    # 为什么默认不是 block：它把"看起来正常、其实没依据"的回答换成"找不到"，
    # 用户会以为知识库里没有这条——这是**信息损失**，属于产品取舍而不是
    # 技术正确性。默认只观测（告警进日志、报告进响应），攒够数据再决定
    # 要不要在某个库上开 block。
    # 与机器无关 → 给真实默认值。
    grounding_action: str = "flag"

    # ---------- 链路追踪（LangSmith，可选） ----------
    # 整段默认**关闭**：不填 LANGSMITH_API_KEY 就什么都不发生。
    #
    # 为什么默认关而不是默认开：
    #   1. trace 会离开本机。这个项目的提示词里含**用户上传的文档正文**
    #      与**多轮历史**（见 rag/prompts.py 的 build_answer_prompts），
    #      默认上传等于替使用者做了数据外发的决定；
    #   2. 它是排查工具而不是运行依赖。关掉时链路行为必须**一模一样**，
    #      这一条由 app/core/tracing.py 的直通实现保证；
    #   3. 国内网络访问 api.smith.langchain.com 需要代理，默认开会让
    #      "一启动就有一堆上报失败日志"成为一种出厂体验。
    #
    # **留空不拦启动**（与 LLM / 权重那类必填项不同）：追踪是观测能力，
    # 缺了它问答链路照常工作，只是没有 trace。
    langsmith_api_key: str = ""
    # 在 LangSmith 里显示的 project 名。同一个 key 下用不同 project 区分
    # 「本地调参」与「对外环境」，避免两类 trace 混在一个列表里。
    langsmith_project: str = "rag-backend"
    # 自建/欧洲区端点时需要改。**注意与项目内的 LANGSMITH_ENDPOINT 环境变量同名**：
    # 本类负责把它从 .env 读出来，再由 tracing.configure_tracing() 灌回环境变量
    # 供 langsmith SDK 读取（SDK 只认环境变量，不认我们的 Settings）。
    langsmith_endpoint: str = ""
    # 隐私闸门：**默认隐藏**输入输出。隐藏后 LangSmith 里只看到
    # 「哪个节点、耗时、状态、token 数、报错」，看不到问题、资料块与回答。
    # 想看全文（本地排查幻觉/提示词问题时有用）就把对应一项设成 false。
    # 注意这是 SDK 的开关，SDK 默认值是「上报全文」——所以必须显式写 true，
    # 不能靠"不配置 = 不上报"（不配置就是上报）。
    langsmith_hide_inputs: bool = True
    langsmith_hide_outputs: bool = True
    # 上报失败时是否把异常打出来。默认**打**：追踪静默失效是最难发现的一类
    # 问题——你会以为"这几天没出问题"，其实是根本没在记录。
    langsmith_log_errors: bool = True

    # ---------- 会话（多轮对话历史，Redis） ----------
    # Redis 连接串，如 redis://127.0.0.1:6379/0。
    # 与机器/部署相关 → 类默认留空，只从 .env / 环境变量读。
    # **留空不拦启动**（与 LLM/权重那类必填项不同）：会话历史是增强能力，
    # 缺了它问答链路照常工作（历史退回请求体传入、接口层不持久化），
    # /conversations 一族接口则显式报 503 并说明缺什么——
    # 半可用状态的失败面被收敛到「明确知道自己在缺什么」的接口上。
    redis_url: str = ""
    # 会话过期时间（小时）。与机器无关 → 给真实默认值。7 天没动静的会话
    # 连同消息一起过期；每次问答都会续期，活跃会话不会被清。
    conversation_ttl_hours: int = 168
    # 问答时送进提示词的「最近 N 条消息」上限（user+assistant 合计）。
    # 太长会稀释参考资料的存在感并推高 token 成本；0 = 不把历史送进提示词
    # （仍照常落库与返回溯源）。
    conversation_history_turns: int = 6

    # ---------- 切分 ----------
    chunk_size: int = 300
    chunk_overlap: int = 50

    # ---------- 上传 ----------
    upload_dir: str = "./data/uploads"
    max_upload_mb: int = 50
    # 批量上传的两个总闸门。单文件上限挡不住"100 个 49MB 文件"：
    # 逐个校验都通过，但请求体会把磁盘与内存一起打满，且入库队列会排到下个世纪。
    max_batch_files: int = 20
    max_batch_total_mb: int = 200
    # 同时进行的入库任务数。向量化是纯 CPU 的批量前向，并行跑不会更快，
    # 只会让所有请求的尾延迟一起变差。默认 1 = 排队（后到的请求等待，
    # 而不是抢占）。真正需要横向扩展时应把**检索**多开，入库保持单写。
    ingest_max_concurrency: int = 1
    # ---------- 上传内容的静态检查（恶意/畸形文件） ----------
    # 扩展名白名单只看名字，改个后缀就能绕过（把 .exe 命名成 .pdf 成本 0）。
    # 这里按扩展名核对文件头（magic bytes），并给 zip 容器（docx/pptx）设上限：
    # 「压缩比」炸弹是唯一能用一个 50MB 上传吃光内存的形态。
    max_pdf_pages: int = 3000
    max_zip_entries: int = 5000
    max_zip_uncompressed_mb: int = 300
    # 单条目压缩比上限（解压后 / 压缩后）。正常 docx 约 3~10 倍，
    # 上千倍的是刻意构造的炸弹。设 0 关闭该检查（不推荐）。
    max_zip_ratio: int = 200
    # 逗号分隔的扩展名白名单（小写，带点）
    # 注意：不含 .doc —— 旧版二进制 .doc 的抽取会引入 unstructured 或 Word COM，
    # 两种都会让依赖/部署变脏；直接拒绝并提示"另存为 .docx"。
    # 这份白名单必须与 app/infrastructure/loading.py 的加载器清单一致
    # （scripts/check_loading.py 有断言，避免"能上传不能加载"）。
    upload_allowed_extensions: str = ".pdf,.docx,.txt,.md,.pptx"

    # ---------- 持久化 ----------
    sqlite_path: str = "./data/rag.db"

    # ---------- 认证 ----------
    # 令牌签名密钥：**与机器/部署相关的机密值** → 类默认留空，只从 .env 读。
    # 留空时服务拒绝启动（见 validate_runtime）：空密钥签出来的令牌人人可伪造，
    # "能起服务但令牌是假的"比"起不来"危险得多。
    # 生成方式：python -c "import secrets; print(secrets.token_urlsafe(48))"
    auth_secret_key: str = ""
    # 令牌有效期。与机器无关 → 给真实默认值。12 小时，覆盖一个工作日；
    # 本项目没有"续期"，过期即重新登录（登录只要一次 PBKDF2，成本可接受）。
    auth_token_ttl_minutes: int = 720
    # 是否开放注册接口。本地开发默认放开；对外环境应关闭并改用
    # scripts/create_user.py 建号（APP_ENV=prod 时 validate_runtime 会拦下）。
    auth_allow_registration: bool = True
    # 新口令的 PBKDF2 迭代次数。**只影响此后新设的口令**：
    # 校验已有口令时用的是哈希串里记的次数，改这里不会把老账号锁在门外。
    # 低于 100000 会被 validate_runtime 拦下（离线暴破的成本会显著变低）。
    auth_pbkdf2_iterations: int = 210_000

    # ---------- 派生属性 ----------
    @property
    def cors_origin_list(self) -> list[str]:
        """将逗号分隔的 CORS_ORIGINS 解析为来源列表（去空白、丢弃空项）。"""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def milvus_uri(self) -> str:
        """Milvus 连接串，形如 http://127.0.0.1:19530。"""
        return f"http://{self.milvus_host}:{self.milvus_port}"

    @property
    def llm_api_key_or_placeholder(self) -> str:
        """OpenAI 兼容客户端要求 api_key 非空。

        本地后端（Ollama / vLLM）不做鉴权，此时返回占位串，
        避免 SDK 构造期抛 "The api_key client option must be set"。
        """
        return self.llm_api_key.strip() or "not-needed"

    @property
    def langsmith_enabled(self) -> bool:
        """是否启用 LangSmith 追踪（唯一判据：有没有 API Key）。

        用「有没有 key」而不是再加一个 ENABLED 开关，是为了消灭一种自相矛盾
        的配置组合（`LANGSMITH_ENABLED=true` 但没填 key）——那种状态只能靠
        运行期日志区分，而它的表现恰好是"看起来开着、其实什么都没记录"。
        """
        return bool(self.langsmith_api_key.strip())

    @property
    def upload_extension_list(self) -> tuple[str, ...]:
        """将 UPLOAD_ALLOWED_EXTENSIONS 解析为小写扩展名元组（带点）。"""
        items = (
            item.strip().lower()
            for item in self.upload_allowed_extensions.split(",")
        )
        return tuple(item for item in items if item)

    @property
    def max_upload_bytes(self) -> int:
        """上传体积上限（字节）。"""
        return int(self.max_upload_mb) * 1024 * 1024

    @property
    def max_batch_total_bytes(self) -> int:
        """批量上传的总体积上限（字节）。"""
        return int(self.max_batch_total_mb) * 1024 * 1024

    @property
    def max_zip_uncompressed_bytes(self) -> int:
        """zip 容器解压后的总大小上限（字节）。"""
        return int(self.max_zip_uncompressed_mb) * 1024 * 1024

    # ---------- 启动期校验 ----------
    def validate_runtime(self) -> None:
        """启动期自检：必填项、路径存在性、数值区间。

        由 create_app() 在启动时调用。**有意一次性列出全部问题**而非只报
        第一个，避免改一处、重启、再撞下一处的来回折腾。

        Raises:
            ConfigurationError: 存在任一配置缺失或非法时抛出。
        """
        # 延迟导入：app.core.exceptions 会连带导入 fastapi，而本模块位于
        # 最底层，不应在模块级耦合 Web 框架（也让本文件可被独立加载校验）。
        from app.core.exceptions import ConfigurationError

        problems: list[str] = []

        # ① 必填项：类默认值为空，必须由 .env / 环境变量提供
        if not self.llm_base_url.strip():
            problems.append(
                "缺少 LLM_BASE_URL — OpenAI 兼容端点，如 http://127.0.0.1:11434/v1"
            )
        if not self.llm_model.strip():
            problems.append("缺少 LLM_MODEL — 模型名，如 qwen2:7b")
        if not self.embedding_model_path.strip():
            problems.append(
                "缺少 EMBEDDING_MODEL_PATH — BGE 权重目录（FlagEmbedding 直接加载）"
            )
        if self.rerank_enabled and not self.rerank_model_path.strip():
            problems.append("RERANK_ENABLED=true 但缺少 RERANK_MODEL_PATH")

        # ② 路径必须真实存在且是目录
        candidates = [("EMBEDDING_MODEL_PATH", self.embedding_model_path)]
        if self.rerank_enabled:
            candidates.append(("RERANK_MODEL_PATH", self.rerank_model_path))
        for field, value in candidates:
            if value.strip() and not Path(value).is_dir():
                problems.append(f"{field} 不是有效目录 — {value}")

        # ③ 数值区间
        if self.chunk_size <= 0:
            problems.append(f"CHUNK_SIZE 必须为正整数 — 当前 {self.chunk_size}")
        if self.chunk_overlap < 0:
            problems.append(f"CHUNK_OVERLAP 不能为负 — 当前 {self.chunk_overlap}")
        elif self.chunk_size > 0 and self.chunk_overlap >= self.chunk_size:
            problems.append(
                f"CHUNK_OVERLAP({self.chunk_overlap}) 必须小于 CHUNK_SIZE({self.chunk_size})"
            )
        if self.milvus_dimension <= 0:
            problems.append(f"MILVUS_DIMENSION 必须为正整数 — 当前 {self.milvus_dimension}")
        if self.milvus_upsert_batch_size <= 0:
            problems.append(
                f"MILVUS_UPSERT_BATCH_SIZE 必须为正整数 — 当前 {self.milvus_upsert_batch_size}"
            )
        if self.retrieve_top_k <= 0:
            problems.append(f"RETRIEVE_TOP_K 必须为正整数 — 当前 {self.retrieve_top_k}")
        if self.retrieve_k_min < 1:
            problems.append(f"RETRIEVE_K_MIN 必须 >= 1 — 当前 {self.retrieve_k_min}")
        if self.retrieve_k_max < self.retrieve_k_min:
            # 写反时 adaptive 会把每次召回都截成 0 条，表现为"突然什么都搜不到"，
            # 且不会报错——这类配置只能在这里拦。
            problems.append(
                f"RETRIEVE_K_MAX({self.retrieve_k_max}) 不能小于 "
                f"RETRIEVE_K_MIN({self.retrieve_k_min})"
            )
        if not 0.0 <= self.retrieve_k_drop_ratio <= 1.0:
            problems.append(
                f"RETRIEVE_K_DROP_RATIO 应落在 [0,1] — 当前 {self.retrieve_k_drop_ratio}"
                "（0 = 任何落差都算断层；1 = 几乎不切）"
            )
        if self.rerank_enabled and self.rerank_top_n <= 0:
            problems.append(f"RERANK_TOP_N 必须为正整数 — 当前 {self.rerank_top_n}")
        if self.rerank_min_top_score < 0:
            problems.append(
                f"RERANK_MIN_TOP_SCORE 不能为负 — 当前 {self.rerank_min_top_score}"
                "（0 表示关闭该闸门；负阈值等于永远不触发，不如写 0 更诚实）"
            )
        if not 0.0 <= self.min_similarity <= 1.0:
            problems.append(f"MIN_SIMILARITY 应落在 [0,1] — 当前 {self.min_similarity}")
        if self.grounding_action not in {"off", "flag", "block"}:
            # 拼错成 "blcok" 时若不拦，`ask()` 里既不核验也不报错——
            # 表现是"这个功能明明开了却什么都没发生"，最难查的一类静默失效。
            problems.append(
                "GROUNDING_ACTION 只能是 off / flag / block — "
                f"当前 {self.grounding_action!r}"
            )
        if not 0.0 <= self.hybrid_dense_min_similarity <= 1.0:
            problems.append(
                "HYBRID_DENSE_MIN_SIMILARITY 应落在 [0,1] — "
                f"当前 {self.hybrid_dense_min_similarity}"
            )
        if not 0.0 <= self.llm_temperature <= 2.0:
            problems.append(f"LLM_TEMPERATURE 应落在 [0,2] — 当前 {self.llm_temperature}")
        # 上下文预算：这里拦的是"参数组合自相矛盾"，
        # 它们不会报错、只会让每次请求都拒答或都被截断。
        if self.llm_max_tokens <= 0:
            problems.append(
                f"LLM_MAX_TOKENS 必须为正整数 — 当前 {self.llm_max_tokens}"
                "（它是输出预留，也是下发给模型的 max_tokens）"
            )
        if self.llm_context_window <= 0:
            problems.append(
                f"LLM_CONTEXT_WINDOW 必须为正整数 — 当前 {self.llm_context_window}"
                "（填**实际服务窗口**，如 Ollama 的 num_ctx）"
            )
        elif self.llm_max_tokens >= self.llm_context_window:
            problems.append(
                f"LLM_MAX_TOKENS({self.llm_max_tokens}) 必须小于 "
                f"LLM_CONTEXT_WINDOW({self.llm_context_window})，"
                "否则输出预留一扣就没有任何空间放证据"
            )
        if not 0.0 < self.context_evidence_share < 1.0:
            problems.append(
                f"CONTEXT_EVIDENCE_SHARE 应落在 (0,1) — 当前 {self.context_evidence_share}"
            )
        if not 0.0 <= self.context_margin_ratio < 1.0:
            problems.append(
                f"CONTEXT_MARGIN_RATIO 应落在 [0,1) — 当前 {self.context_margin_ratio}"
            )
        if self.context_meta_tokens < 0:
            problems.append(
                f"CONTEXT_META_TOKENS 不能为负 — 当前 {self.context_meta_tokens}"
            )
        if self.context_snippet_limit <= 0:
            problems.append(
                f"CONTEXT_SNIPPET_LIMIT 必须为正整数 — 当前 {self.context_snippet_limit}"
            )
        if self.context_max_evidence < 1:
            problems.append(
                f"CONTEXT_MAX_EVIDENCE 必须 >= 1 — 当前 {self.context_max_evidence}"
            )
        if self.chars_per_token <= 0:
            problems.append(
                f"CHARS_PER_TOKEN 必须为正数 — 当前 {self.chars_per_token}"
            )
        if self.tokenizer_path.strip() and not Path(self.tokenizer_path).is_dir():
            problems.append(
                f"TOKENIZER_PATH 不是有效目录 — {self.tokenizer_path}"
            )
        for field, value in (
            ("HYBRID_VECTOR_WEIGHT", self.hybrid_vector_weight),
            ("HYBRID_BM25_WEIGHT", self.hybrid_bm25_weight),
        ):
            if not 0.0 <= value <= 1.0:
                problems.append(f"{field} 应落在 [0,1] — 当前 {value}")
        if self.hybrid_rrf_k < 1:
            problems.append(f"HYBRID_RRF_K 必须 >= 1 — 当前 {self.hybrid_rrf_k}")
        if self.hybrid_bm25_min_score < 0:
            problems.append(
                f"HYBRID_BM25_MIN_SCORE 不能为负 — 当前 {self.hybrid_bm25_min_score}"
                "（BM25 分非负，负阈值等于不设限）"
            )
        if not 0.0 <= self.hybrid_near_dupe_ratio <= 1.0:
            problems.append(
                f"HYBRID_NEAR_DUPE_RATIO 应落在 [0,1] — 当前 {self.hybrid_near_dupe_ratio}"
                "（0 = 关闭近重复合并）"
            )
        if self.hybrid_near_dupe_min_chars < 0:
            problems.append(
                f"HYBRID_NEAR_DUPE_MIN_CHARS 不能为负 — 当前 {self.hybrid_near_dupe_min_chars}"
            )
        if self.hybrid_enabled and self.hybrid_vector_weight + self.hybrid_bm25_weight <= 0:
            problems.append(
                "HYBRID_ENABLED=true 但 HYBRID_VECTOR_WEIGHT 与 HYBRID_BM25_WEIGHT 同时为 0 "
                "— 融合分将恒为 0，排序失去意义"
            )
        if self.max_upload_mb <= 0:
            problems.append(f"MAX_UPLOAD_MB 必须为正数 — 当前 {self.max_upload_mb}")
        if self.max_batch_files <= 0:
            problems.append(f"MAX_BATCH_FILES 必须为正整数 — 当前 {self.max_batch_files}")
        if self.ingest_max_concurrency < 1:
            problems.append(
                f"INGEST_MAX_CONCURRENCY 至少为 1 — 当前 {self.ingest_max_concurrency}"
            )
        if self.max_batch_total_mb < self.max_upload_mb:
            problems.append(
                "MAX_BATCH_TOTAL_MB 不能小于 MAX_UPLOAD_MB — "
                f"当前 {self.max_batch_total_mb} < {self.max_upload_mb}，"
                "否则单个文件合法却永远过不了批量上传"
            )
        if self.max_pdf_pages <= 0:
            problems.append(f"MAX_PDF_PAGES 必须为正整数 — 当前 {self.max_pdf_pages}")
        if self.max_zip_entries <= 0:
            problems.append(f"MAX_ZIP_ENTRIES 必须为正整数 — 当前 {self.max_zip_entries}")
        if self.max_zip_uncompressed_mb <= 0:
            problems.append(
                f"MAX_ZIP_UNCOMPRESSED_MB 必须为正整数 — 当前 {self.max_zip_uncompressed_mb}"
            )
        if self.max_zip_ratio < 0:
            problems.append(f"MAX_ZIP_RATIO 不能为负 — 当前 {self.max_zip_ratio}（0 表示关闭）")
        if not self.upload_extension_list:
            problems.append("UPLOAD_ALLOWED_EXTENSIONS 不能为空，否则任何文件都传不进来")
        if self.conversation_ttl_hours <= 0:
            problems.append(
                f"CONVERSATION_TTL_HOURS 必须为正整数 — 当前 {self.conversation_ttl_hours}"
            )
        if self.conversation_history_turns < 0:
            problems.append(
                f"CONVERSATION_HISTORY_TURNS 不能为负 — 当前 {self.conversation_history_turns}"
            )

        # ④ 认证
        secret = self.auth_secret_key.strip()
        if not secret:
            problems.append(
                "缺少 AUTH_SECRET_KEY — 令牌签名密钥；生成方式："
                'python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )
        elif len(secret) < 32:
            problems.append(
                f"AUTH_SECRET_KEY 太短（{len(secret)} < 32 字符）— "
                "短密钥可被暴力穷举，签出来的令牌等同于无效"
            )
        if self.auth_token_ttl_minutes <= 0:
            problems.append(
                f"AUTH_TOKEN_TTL_MINUTES 必须为正整数 — 当前 {self.auth_token_ttl_minutes}"
            )
        if self.auth_pbkdf2_iterations < 100_000:
            problems.append(
                f"AUTH_PBKDF2_ITERATIONS 至少 100000 — 当前 {self.auth_pbkdf2_iterations}"
                "（迭代次数过低会让离线暴破变便宜）"
            )
        if self.app_env == "prod" and self.auth_allow_registration:
            problems.append(
                "APP_ENV=prod 时不应开放注册（AUTH_ALLOW_REGISTRATION=true）— "
                "任何人都能自助建号；请关闭并改用 scripts/create_user.py 建号"
            )

        if problems:
            raise ConfigurationError(
                "配置校验未通过，服务拒绝启动（共 {} 项）：\n  - {}".format(
                    len(problems), "\n  - ".join(problems)
                ),
                detail={
                    "problems": problems,
                    "app_env": self.app_env,
                    "env_file": ".env",
                },
            )


@lru_cache
def get_settings() -> Settings:
    """返回配置单例。测试中可调用 get_settings.cache_clear() 重置。

    注意：本函数只负责构造，不做校验——校验由 create_app() 显式调用
    validate_runtime() 触发，便于测试里注入非法配置而不被拦截。
    """
    return Settings()
