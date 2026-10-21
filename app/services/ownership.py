"""知识库归属校验——多租户隔离的**唯一**执行点。

为什么单独抽出来：
    归属校验是**安全边界**，不是普通业务逻辑。此前 `ChatService`、
    `IngestService`、`RetrievalService` 各写了一份 `_ensure_owned`，
    三份实现几乎一样——这正是最危险的地方：某天规则要变（比如改成
    "管理员可读他人库"、或加一条"归档库禁止写入"），改了两处忘了一处，
    漏掉的那个服务就静默失去隔离，而且**不会有任何报错**。
    安全逻辑只该有一份。

为什么放在 services 层而不是 domain：
    它依赖"知识库仓储 + 当前登录账号"这两个服务层概念，
    而 domain 不该知道 HTTP 身份是怎么来的。放在这里，
    services 层内部复用，路由层仍走 `CollectionService.ensure_exists()`。

与 `CollectionService.ensure_exists()` 的关系：
    那个方法多做一件事——**返回知识库字典**，因为它的调用方（文档列表、
    检索接口）本来就需要库信息，顺手取回来省一次查询。
    这里的函数只做校验、不返回数据，给"不需要库信息"的服务用。

统一采用 404 而不是 403：
    403 等于告诉对方"这个 ID 确实存在，只是不给你"，配合遍历就能枚举出
    别人有哪些知识库。404 让"不存在"与"不是你的"在响应上逐字节相同。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.exceptions import KB_NOT_FOUND_MESSAGE, NotFoundError

if TYPE_CHECKING:
    from app.domain.ports import ChunkRepositoryPort


def ensure_kb_owned(
    repository: ChunkRepositoryPort | None,
    kb_id: str,
    owner_id: str | None,
) -> None:
    """确认知识库存在且属于该账号，否则抛 NotFoundError（404）。

    Args:
        repository: 元数据仓储。为 None 表示调用方没装配仓储——
            此时只有在 `owner_id` 也为 None 的内部路径（脚本、自检）下才放行。
        kb_id: 知识库 ID
        owner_id: 当前登录账号 ID。**None 表示调用方明确声明"走内部路径，
            不做归属校验"**；这不是"忘记传"，忘记传的调用方会在类型检查
            与代码评审里暴露（服务方法的 owner_id 是无默认值的必传参数）。

    Raises:
        NotFoundError: 情况有三种，对外**必须**是同一种响应——
            知识库不存在、不属于该账号、或服务未装配仓储却收到了 owner_id。
            最后一种是装配错误，但宁可错杀也不能放行：
            "该拦的时候没拦"远比"该通的时候不通"严重，
            后者用户会来报障，前者没人会发现。

    设计取舍——为什么 owner_id 为 None 时放行：
        内部脚本（`scripts/ask.py`）与自检没有"登录账号"这个概念，
        强制它们传一个假 ID 反而会掩盖真实行为。放行是有意的逃生舱，
        代价是所有 HTTP 入口都必须传真实 user_id，这一点由
        `check_auth.py` 的路由级断言钉住。
    """
    if owner_id is None:
        return
    if repository is None:
        raise NotFoundError(KB_NOT_FOUND_MESSAGE, detail={"reason": "no_repository"})
    if repository.get_knowledge_base(kb_id, owner_id) is None:
        raise NotFoundError(KB_NOT_FOUND_MESSAGE)
