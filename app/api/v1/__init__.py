"""v1 路由包。

版本化策略：
    /api/v1/** 为稳定契约；破坏性变更时新增 /api/v2，不原地修改 v1。
    prefix 从 settings.api_v1_prefix 读取，便于统一调整。
"""

from .router import api_router

__all__ = ["api_router"]
