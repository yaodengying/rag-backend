"""问答相关 DTO。

字段命名与 `schemas/retrieval.py` **刻意保持一致**：
    同一条片段在 `/search` 与 `/chat` 里必须显示同一套字段名
    （`vector_score` / `rerank_score`），否则前端要写两套解析，
    调参时也会因为"这个接口叫 score、那个叫 vector_score"而读错口径。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceOut(BaseModel):
    """检索来源（用于前端展示引用）。"""

    doc_id: str
    filename: str = ""
    page: str | None = Field(None, description="形如「第3页」或「第3-5页」；未知为 null")
    chunk_index: int | None = None
    text: str = Field("", description="片段正文，通常截断到 400 字")
    vector_score: float | None = Field(
        None, description="余弦相似度；为 null 表示这一路没召回它（与「得 0 分」不同）"
    )
    rerank_score: float | None = Field(
        None, description="交叉编码器原始 logits；未启用重排时为 null"
    )


class CitationOut(BaseModel):
    """回答中引用的来源。"""

    doc_id: str
    filename: str = ""
    page: str | None = None
    snippet: str = ""


class GroundingOut(BaseModel):
    """回答接地性核验结果（生成之后的事后核对）。

    存在的理由：**检索幻觉最常见的形态是"找到了、但找错了"**——召回到一份
    主题相近的文档，模型照着答得通顺，数字和型号却是那份文档里的。提示词
    挡不住（模型确实只依据了资料，只是资料错），拒答判据也挡不住
    （`ranked` 非空）。这一层查的是**可证伪的硬事实**：回答里的数字、金额、
    日期、型号，是否真的在参考资料里逐字出现过。

    前端怎么用：
        status == "ok"   正常展示
        status == "warn" 正常展示（单个数字对不上，多半是量词或同义改写）
        status == "low"  展示但要提示"本回答含资料未支撑的数字/型号，请核对引用"
        为 null          本轮没有可核验的回答（拒答路径）

    `warnings` 是给人读的原始告警，前端不必解析它，只用于展开"为什么"。
    """

    status: str = Field("ok", description="ok / warn / low")
    checked: int = Field(0, description="核验过的硬事实条数（去重后）")
    unsupported: list[str] = Field(
        default_factory=list, description="在参考资料里找不到的硬事实（数字/型号/日期）"
    )
    invalid_refs: list[int] = Field(
        default_factory=list, description="引用了本轮不存在的来源序号（真幻觉）"
    )
    coverage: float = Field(1.0, description="硬事实支持率；没有可核验事实时为 1.0")
    blocked: bool = Field(
        False,
        description=(
            "是否已被拦下（GROUNDING_ACTION=block 且 status=low 时把回答换成了拒答话术）。"
            "据此可与「真的没检索到」区分开"
        ),
    )
    warnings: list[str] = Field(default_factory=list, description="人类可读告警明细")


class ChatRequest(BaseModel):
    """问答请求。"""

    kb_id: str = Field(..., description="目标知识库 ID")
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    conversation_id: str | None = Field(
        None,
        max_length=64,
        description=(
            "会话 ID（多轮对话历史存在服务端 Redis）。"
            "带上了它：本轮历史由服务端从 Redis 读并送进提示词，"
            "请求体里的 history 被忽略；回答连同溯源（文档名+页码）自动落库，"
            "done 事件/响应会回传同一 ID。"
            "不传且配置了 REDIS_URL：服务端生成新 ID（首问也落库）；"
            "未配置 REDIS_URL：忽略本字段，历史退回请求体 history。"
        ),
    )
    history: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "多轮历史，元素形如 {role, content}。"
            "仅在「未带 conversation_id」或「服务端未配置会话存储」时生效"
        ),
    )
    enable_retry: bool | None = Field(
        default=None,
        description=(
            "仅 /chat/stream 生效：首轮检索无可用片段时，是否允许一次"
            "查询改写 + 再检索。**三态**：true / false 是前端用户的选择，"
            "服务端照办；省略或传 null 则用服务端默认值"
            "（GET /chat/options 可查，配置项 CHAT_STREAM_RETRY_DEFAULT）。"
            "打开后若真的重试，会先推一个 retry 事件填住 2~4 秒静默"
            "（本机 7B 一次改写约 2~4s）。非流式 /chat 恒重试一次，与本字段无关。"
        ),
    )
    top_k: int = Field(8, ge=1, le=50, description="向量召回条数")
    rerank_k: int = Field(4, ge=1, le=20, description="重排后保留条数")


class ChatResponse(BaseModel):
    """问答响应。"""

    answer: str
    sources: list[SourceOut] = Field(default_factory=list)
    citations: list[CitationOut] = Field(default_factory=list)
    refused: bool = Field(False, description="是否因资料不足而拒答")
    retried: bool = Field(
        False,
        description=(
            "检索阶段是否**真的**用改写后的查询重新检索过。"
            "注意它不等于「调过改写」：改写无产出（模型原样抄回问题）时不会重跑检索，"
            "此时 retried 仍为 false。"
        ),
    )
    query: str = Field(
        "", description="实际用于检索的查询；发生过改写时与 question 不同"
    )
    conversation_id: str | None = Field(
        None,
        description="本轮归属的会话 ID（未配置会话存储时为 null）；后续轮次带上它即可续聊",
    )
    grounding: GroundingOut | None = Field(
        None,
        description=(
            "生成之后的接地性核验（数字/型号/日期是否真在资料里）。"
            "null 表示本轮没有可核验的回答（拒答路径）。"
            "流式接口把它放在 done 事件里——token 已经发出，服务端收不回来，"
            "只能如实标记"
        ),
    )


class ChatOptionsOut(BaseModel):
    """问答接口的前端可选项。

    存在的理由：前端要渲染「是否允许重试」的开关，就得知道**服务端当前
    的默认值**才能给出正确的初始状态。让前端自己抄一份默认值，等于把
    同一个决定放在两个地方——改一处忘一处时表现为「界面上开着、后端却
    没重试」这种最难查的错。所以默认值只由服务端持有，前端来读。
    """

    stream_retry_default: bool = Field(
        ..., description="请求未显式传 enable_retry（或传 null）时，/chat/stream 采用的默认值"
    )
    stream_retry_max_attempts: int = Field(
        ..., description="改写重试上限；流式与非流式共用同一份，前端不要假定为 1"
    )
    stream_events: list[str] = Field(
        ...,
        description=(
            "SSE 可能出现的事件名。retry 只在实际重试时出现且位置在 sources 之前；"
            "done / error 二选一收尾"
        ),
    )
    non_stream_always_retries: bool = Field(
        True,
        description=(
            "非流式 /chat 是否恒重试一次（与 enable_retry 无关）："
            "调用方本来就在等完整答案，重试不额外增加用户可感的等待"
        ),
    )
    grounding_action: str = Field(
        "flag",
        description=(
            "当前生效的接地性核验动作：off（不查）/ flag（只标记）/ "
            "block（非流式下低置信回答会变成拒答）。前端据此决定是否展示"
            "「低置信」提示条——默认值同样只由服务端持有，不要在前端抄一份"
        ),
    )
