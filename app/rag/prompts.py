"""提示词集中管理。

集中在单独模块的原因：
    提示词是 RAG 效果的第一杠杆，散落在各节点里会难以迭代与对比。
    后续可在此引入版本号或从外部文件加载，便于 A/B 测试。

核心约束（防幻觉）：
    1. 明确「只能依据参考资料作答」，禁止使用模型自身常识；
    2. 资料未覆盖问题时，必须原样回复 NO_ANSWER_TEXT，不得自由发挥；
    3. 数字、日期、型号必须逐字引用，不得改写或换算；
    4. 不得把不同段落/不同文件的片段拼接成新结论；
    5. 回答使用中文，并标注来源序号，便于人工核对。

为什么约束要写成「清单 + 逐条禁止」而不是一句"不要编造"：
    实测（qwen2:7b，18 题评估集）中，一句"禁止编造"挡不住**改写式幻觉**：
    资料写「压力前放量区下轨」，模型答成「箱体上轨的下方」——它没有凭空
    造事实，而是把邻近概念替换掉了，这类错误连人类快速扫读都不易发现
    （RAGAS faithfulness 给了 0 分才暴露）。
    7B 级模型对"具体禁止项"的遵守度明显高于抽象要求，所以把最容易出错的
    四类（改写数字、跨段拼接、补充建议、含糊其辞）逐条写死。
    每一条都对应一个真实出现过的错误形态，不是泛泛的"请准确回答"。

    另有一条与代码强耦合的规则：**允许部分回答**。旧提示词只给了"答不了就
    拒答"这一条出路，模型于是把"资料只写了一半"也当成"答不了"而整题拒答，
    或者反过来硬凑。现在明确"只写资料覆盖的那部分"。

    ⚠️ 但**不要给模型一句可以直接照抄的占位话术**（2026-10-18 实测，代价很大）：
    曾经写成"其余明确写「资料未说明」"，结果 qwen2:7b 把这句占位话术当成了**答案本身**
    ——评估集 q05（售后工单多久内首次响应？）的参考资料明明白白写着"客服须在
    48 小时内首次响应"，模型整题只回了五个字"资料未说明。"，q06 同段资料、
    同样被拒。两题的 faithfulness 掉到 0.5 / 0.33、answer_relevancy 掉到 0，
    把一个"允许部分回答"的善意条款变成了"绕开回答的捷径"。
    逐条拆除实验证明病根就是第 5 条：删掉它、或删掉"违反即错误回答"那句，
    同一段资料立刻能正常作答；把话术换成「资料未提及」照样被照抄。
    结论：**允许留白，但不给句式**——写"只写覆盖的那部分、不补资料没有的内容"，
    让模型自己决定怎么表达；能照抄的句子一定会被照抄。

为什么第 6 条要写成「**完全无关**才拒答」（2026-10-18 的实测修正）：
    第一版逐条禁令只写了「若参考资料完全无法回答问题 → 原样回复话术」，
    结果 qwen2:7b 把"资料只说了一句、没解释原因"也理解成"无法回答"：
    评估集 q02（哪些情况不在保修范围内？）**上下文完全正确、就摆着那句
    「人为损坏、私自拆机及不可抗力导致的故障不在保修范围内」**，
    模型仍然照抄了拒答话术（faithfulness 与命中率同时掉到 0）。
    同一批数据里 16/16 道题的答案都不再标 `(来源：[n])`——来源标注那一段
    写成了"若参考资料带有来源标注…"这种**条件句**，7B 直接当作不成立。
    教训：禁令要严，但**正面要求必须写成无条件的祈使句**；一旦把"要不要做"
    留给模型判断，它会在两个方向上都出错（该答的拒、该标的省略）。
    这两处都由 `chat_service._audit_answer()` 与评估集共同暴露，
    `check_chat.py` 也把关键措辞钉成了断言。

关于「原样回复 NO_ANSWER_TEXT」为什么是**逐字**要求：
    上层用字符串包含判定「模型到底是拒答了还是答了」
    （见 `nodes._is_refusal`）。一旦提示词里的措辞与
    `state.NO_ANSWER_TEXT` 不一致，判定就会失效——模型嘴上说"找不到"、
    系统却记成 refused=False，于是返回一条「无法从知识库中找到…」
    却标着「已作答」的回答。改话术时**必须两边一起改**，
    `check_chat.py` 有断言盯着这一点。

关于来源标注（溯源页码）：
    `format_context` 会给每段资料加上 `[序号] 来源：文件名 第N页`。
    这一步不只是给人看的——模型只有看到页码，才可能在回答里引用页码；
    而接口返回的结构化 citations 也来自同一批片段，两者对得上。
    注意页码可能为 None（txt/md/docx 这类流式文本没有页边界），
    此时如实写「页码未知」，**不要**编一个页码出来。

为什么最后**取消**了「回答里必须写 (来源：[n])」这条要求（2026-10-18 实测）：
    试过三种写法——条件句（"若资料带标注则…"）、无条件祈使句（"必须做"）、
    加剧禁令——qwen2:7b 的遵守率是 16/16 不写、13/18 不写，始终不稳定。
    而**结构化 citations 已经提供了完整溯源**（文件名 + 页码，前端直接渲染），
    行内序号是锦上添花。既然要不来，就把它从提示词里删掉：
    留着一条模型必然违反的指令，只会让"提示词遵守度"这个信号永久噪音化。
    连带地，`_audit_answer()` 也不再报"未标注序号"，只报**序号越界**——
    那才是真幻觉（引用了本轮不存在的第 5 条资料）。
"""

from __future__ import annotations

from app.core.config import get_settings
from app.domain.models import RetrievedChunk
from app.rag.budget import ContextPlan, clip_for_prompt, plan_context
from app.rag.state import NO_ANSWER_TEXT

# -----------------------------------------------------------------------------
# 回答生成
# -----------------------------------------------------------------------------

ANSWER_SYSTEM = f"""你是企业知识库问答助手。你唯一的资料来源是下方「参考资料」。

必须遵守（违反任意一条都视为错误回答）：
1. 只使用参考资料中明确写出的信息。禁止使用常识、经验、推测或任何外部知识。
2. 数字、日期、期限、金额、型号、专有名词必须与参考资料逐字一致，不得改写、
   四舍五入或换算单位。
3. 不得把不同段落、不同文件的信息拼接成一个新结论。每一条结论都必须能
   单独对应到某一段资料。
4. 参考资料没有提到的内容（原因、影响、建议、后续步骤、注意事项）一律不要写。
5. 资料只覆盖问题的一部分时，只写资料覆盖的那部分；不要为了凑完整补上资料里
   没有的内容，也不要因为「只答了一半」而拒答。
6. 只有参考资料与问题**完全无关**时才拒答，并原样回复：「{NO_ANSWER_TEXT}」
   ——不改写、不解释、不补充、不道歉。

只要参考资料里有一句能回答问题，就必须回答那一句。资料简短、只覆盖一部分、
或没有解释原因，都不是拒答的理由；拒答只用于"这段资料跟问题毫不相干"。

若确实一句都不沾边，必须逐字回复「{NO_ANSWER_TEXT}」——这是系统识别
"已拒答"的唯一依据，换别的话会被当成"已作答"。

引用来源由系统按检索结果自动附上，回答正文里不需要写文件名、页码或序号。

回答使用中文，简洁直接，不要复述问题。"""

ANSWER_USER = """参考资料：
{context}
{history_block}
用户问题：{question}

请严格依据参考资料作答（数字与型号逐字引用，不做推断）。资料只覆盖问题的一部分
时，只答那部分、不补资料里没有的内容；只有资料与问题完全无关时才原样回复：{no_answer}"""


# -----------------------------------------------------------------------------
# 查询改写（首轮检索不足时启用）
# -----------------------------------------------------------------------------

REWRITE_SYSTEM = (
    "你是检索查询改写助手。把用户问题改写成更适合关键词与向量检索的短查询，"
    "只输出改写后的查询本身，不要解释、不要引号、不要换行。"
)

REWRITE_USER = """原始问题：{question}

上一轮检索未能找到相关内容。请补全关键词或换用同义表述，输出一条新的检索查询。"""


# -----------------------------------------------------------------------------
# 组装函数
# -----------------------------------------------------------------------------

# 单条资料正文的截断长度。重排后的 top_n 通常 4 条，
# 每条 300 字左右的知识库语料拼起来约 1~2k 字，远在 qwen2:7b 的上下文内；
# 这里截断是防止误配超大 CHUNK_SIZE 时把上下文撑爆。
# **必须与 `CONTEXT_SNIPPET_LIMIT` 同源**：预算按这个长度计 token，
# 渲染却按另一个长度截断时，预算会系统性偏高或偏低。
_DEFAULT_SNIPPET_LIMIT = 1200


def format_context(
    chunks: list[RetrievedChunk],
    snippet_limit: int | None = None,
) -> str:
    """把检索到的片段拼成参考资料块，并附上来源标注。

    Args:
        chunks: RetrievedChunk 列表，通常已按相关性降序
        snippet_limit: 单条正文的截断字符数；None 时取
            `settings.context_snippet_limit`（与预算计数同源）

    Returns:
        形如「[1] 来源：手册.pptx 第3页\\n正文…」再用空行拼接的字符串；
        列表为空时返回「（无）」。

    为什么带序号：
        模型要能在回答里引用「哪一段支撑了这个结论」，就必须有一个
        稳定的短标识。文件名又长又可能重复，序号最短且不会歧义。

    为什么页码未知时写「页码未知」而不是省略：
        省略会让模型以为"这段没有来源"，从而不敢引用；
        写明"未知"是如实告知信息缺失，模型仍可引用文件名。
    """
    if not chunks:
        # 返回占位而不是空串：空串会让「参考资料：」后面直接跟问题，
        # 模型看到的是格式正常的提示词，更容易自由发挥。
        return "（无）"

    limit = int(
        get_settings().context_snippet_limit
        if snippet_limit is None
        else snippet_limit
    )
    blocks: list[str] = []
    for index, item in enumerate(chunks, start=1):
        chunk = item.chunk if isinstance(item, RetrievedChunk) else item
        # 与预算计数用**同一个**裁剪函数：按头 60% + 尾 40% 保留，
        # 块被切得很长时不会把段尾的结论整段丢掉。
        text = clip_for_prompt((chunk.text or "").strip(), limit)

        location = chunk.page_label or "页码未知"
        filename = (chunk.filename or "").strip() or "未知文件"
        blocks.append(f"[{index}] 来源：{filename} {location}\n{text}")

    return "\n\n".join(blocks)


def build_history_block(history: list[dict[str, str]] | None) -> str:
    """把多轮历史拼成提示词里的一个可选段落。

    有两条铁律写进措辞里：
        1. 历史**仅供理解上下文**（代词指代、追问省略），不是资料来源——
           否则模型会拿上一轮的答案当依据，绕过「只依据参考资料」的约束；
        2. 历史为空时返回空串，模板里那个 {history_block} 占位行会被
           str.format 整个替换掉，不留空行——无历史请求的提示词必须与
           引入该特性之前**逐字节一致**，否则就是静默的行为漂移。
    """
    if not history:
        return ""
    lines = []
    for turn in history:
        role = "用户" if str(turn.get("role") or "") == "user" else "助手"
        content = " ".join(str(turn.get("content") or "").split())
        if content:
            lines.append(f"{role}：{content}")
    if not lines:
        return ""
    return (
        "\n历史对话（仅供理解上下文，**不是**资料来源，回答仍只依据下方参考资料）：\n"
        + "\n".join(lines)
    )


def build_answer_prompts(
    chunks: list[RetrievedChunk],
    question: str,
    history: list[dict[str, str]] | None = None,
    *,
    plan: ContextPlan | None = None,
) -> tuple[str, str]:
    """组装回答阶段的 (system, user) 两条提示词。

    单独抽成函数而不是在节点里拼字符串：流式与非流式两条路径都要用它，
    两处各写一遍必然会漂移（典型的症状是"流式回答不带页码、非流式带"）。

    Args:
        history: 多轮历史 [{role, content}]（最新的在后）。
            会话链路从 Redis 读；无会话时为 None。
        plan: 上下文装箱结果。**传了就用它**（证据与历史都取自 plan），
            用来避免"服务层算过一次 token、这里再算一次"；不传则现场
            按 `CONTEXT_*` 预算装箱——两条路径的装箱逻辑因此只有一份。

    说明：装箱只会**少放**，不会改动片段顺序或文字；默认配置下
    （4 条 × 300 字）远未触及预算，提示词与引入预算之前逐字节一致。
    """
    if plan is None:
        plan = plan_context(chunks, history, question, system_prompt=ANSWER_SYSTEM)
    return (
        ANSWER_SYSTEM,
        ANSWER_USER.format(
            context=format_context(plan.evidence),
            history_block=build_history_block(plan.history),
            question=question,
            no_answer=NO_ANSWER_TEXT,
        ),
    )


def build_answer_prompts_planned(
    chunks: list[RetrievedChunk],
    question: str,
    history: list[dict[str, str]] | None = None,
) -> tuple[str, str, ContextPlan]:
    """同 `build_answer_prompts`，但把装箱结果一并返回。

    给节点/服务层用来上报 `usage`（输入 token、丢了几条、利用率）。
    拆成函数而不是让调用方"先 plan 再 build"：那样两个调用点都要记得
    把 plan 传回去，漏传时装箱会做两遍——而两遍之间如果配置被改过，
    日志里的用量与真实提示词就对不上了。
    """
    plan = plan_context(chunks, history, question, system_prompt=ANSWER_SYSTEM)
    return (
        *build_answer_prompts(chunks, question, history=history, plan=plan),
        plan,
    )


def build_rewrite_prompts(question: str) -> tuple[str, str]:
    """组装查询改写阶段的 (system, user) 两条提示词。"""
    return REWRITE_SYSTEM, REWRITE_USER.format(question=question)
