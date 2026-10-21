"""认证路由：注册、登录、当前用户。

REST 设计：
    POST /auth/register   注册（成功即签发令牌）
    POST /auth/login      登录（换发访问令牌）
    GET  /auth/me         取当前令牌对应的账号（需鉴权）

本模块同时是**鉴权依赖的唯一来源**：
    `require_current_user` 定义在这里，各业务路由从这里 import。
    之所以放在本文件而不是 api/deps.py：它不是一个"服务工厂"，
    而是一段与令牌格式、错误语义强相关的逻辑，和签发令牌同居一处
    才不会出现"签发与校验对不上"的偏差（例如两边对 scheme 大小写
    的处理不一致）。

鉴权开关（逐步收紧，现在已是第 3 步）：
    第 1 步：只有 /auth/* 公开，其余接口不校验  —— 历史状态
    第 2 步：`require_current_user` 已实现但未挂载 —— 历史状态
    第 3 步（当前）：**除 /health 与 /auth/login|register 外，全部要求令牌**。
        挂载方式统一走各子路由的 `router_dependencies`，不是逐个函数加装饰器——
        逐个加的话，新写一个接口时忘了加，就是一次静默的数据泄漏，
        而这种事不会有任何报错。

为什么这两个处理函数写成**同步 def** 而不是 async def：
    PBKDF2 的 21 万次迭代是 0.1 秒量级的纯 CPU 阻塞，函数体里没有任何 await。
    写成 async def 会让它占住事件循环，期间所有请求（包括 SSE 流）一起卡住；
    写成同步 def，FastAPI 会把它丢进线程池，互不阻塞。
    判断标准很简单：**没有 await 就不要写 async**。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, status

from app.api.deps import get_auth_service
from app.core.exceptions import UnauthorizedError
from app.schemas.auth import LoginRequest, RegisterRequest, TokenOut, UserOut
from app.services.auth_service import AuthResult, AuthService
from app.services.mappers import user_to_dict

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/login",
    response_model=TokenOut,
    summary="登录（换发访问令牌）",
    responses={401: {"description": "用户名或密码不正确 / 账号已停用"}},
)
def login(
    payload: LoginRequest,
    service: AuthService = Depends(get_auth_service),
) -> TokenOut:
    """校验账号口令并签发令牌。

    失败一律 401，且**账号不存在与口令错误返回同一句话**（防账号枚举）。
    """
    return _to_token_out(service.login(payload.username, payload.password))


@router.post(
    "/register",
    response_model=TokenOut,
    status_code=status.HTTP_201_CREATED,
    summary="注册（成功即签发令牌）",
    responses={
        409: {"description": "登录名已被占用"},
        400: {"description": "注册已关闭 / 口令不合规"},
    },
)
def register(
    payload: RegisterRequest,
    service: AuthService = Depends(get_auth_service),
) -> TokenOut:
    """注册账号。

    可用 `AUTH_ALLOW_REGISTRATION=false` 关闭本接口；
    关闭后由管理员用 `python scripts/create_user.py` 建号。
    """
    result = service.register(
        username=payload.username,
        password=payload.password,
        display_name=payload.display_name,
    )
    return _to_token_out(result)


# ---------------------------------------------------------------------------
# 鉴权依赖：**已实现，但刻意没有挂到任何路由上**
# ---------------------------------------------------------------------------


def bearer_token(authorization: str | None) -> str | None:
    """从 Authorization 头里取出 Bearer 令牌；缺失或格式不符返回 None。

    容错点：scheme 大小写不敏感（`Bearer` / `bearer` 都收），
    这是 RFC 7235 的要求；只判 `Bearer` 会让部分客户端莫名其妙地按非法令牌处理。
    """
    if not authorization:
        return None
    # 先 strip 整个头再分段：这样 "  Bearer token" 这种带前导空白的头也能认，
    # 否则 partition 得到的 scheme 是空串，会当成"非 Bearer"直接拒掉
    scheme, _, credential = authorization.strip().partition(" ")
    if scheme.strip().lower() != "bearer":
        return None
    return credential.strip() or None


def current_user_from_header(service: AuthService, authorization: str | None) -> UserOut:
    """按请求头解出当前用户（纯函数，便于脱离 HTTP 直接自检）。

    Raises:
        UnauthorizedError: 缺少令牌，或令牌非法/过期/账号已停用
    """
    token = bearer_token(authorization)
    if token is None:
        raise UnauthorizedError(
            "缺少访问令牌：请在 Authorization 头里带上 `Bearer <access_token>`",
            detail={"reason": "missing_token"},
        )
    return UserOut.model_validate(user_to_dict(service.authenticate(token)))


def require_current_user(
    authorization: str | None = Header(
        default=None, description="形如：Bearer <access_token>"
    ),
    service: AuthService = Depends(get_auth_service),
) -> UserOut:
    """鉴权依赖：解出当前用户，否则 401。

    **已挂到全部业务路由上**（knowledge-bases / documents / retrieval /
    chat / conversations）。仍然保留 `Header` 而不是 `HTTPBearer` 安全类的
    理由变了：现在有接口真的在用这个方案，Swagger 上会出现锁图标，
    这是**如实反映**；而 `HTTPBearer` 会在 OpenAPI 里再挂一个
    `HTTPBearer` securityScheme，与这里手写的 Authorization 解析
    形成两套描述，前端生成代码时会拿到一个实际不生效的鉴权声明。
    """
    return current_user_from_header(service, authorization)


# 业务路由统一使用的依赖列表。
# 用法：`APIRouter(dependencies=router_dependencies)`。
#
# 定义在 `require_current_user` **之后**（Python 需要名字已存在），
# 做成常量而不是让各模块自己写 `[Depends(require_current_user)]`：
# 将来若要在鉴权之上再叠一层（强制改口令、API 版本校验……），
# 只改这里一处，全部路由同步生效。分散写必然漏几个，而"漏一个"就是
# 一次静默的数据泄漏。
router_dependencies = [Depends(require_current_user)]


@router.get("/me", response_model=UserOut, summary="取当前登录账号")
def read_current_user(
    current_user: UserOut = Depends(require_current_user),
) -> UserOut:
    """返回当前令牌对应的账号。

    存在的理由：前端把令牌存在 localStorage，**刷新页面后无法判断它是否
    还有效**（可能已过期，也可能被管理员停用）。没有这个端点，前端只能
    等到第一次业务请求返回 401 才知道——那会让用户先看到一个空列表、
    然后才被踢回登录页。有了它，启动时问一次即可。
    """
    return current_user


def _to_token_out(result: AuthResult) -> TokenOut:
    """AuthResult → 响应 DTO。"""
    return TokenOut(
        access_token=result.access_token,
        token_type="bearer",
        expires_in=result.expires_in,
        expires_at=result.expires_at,
        user=UserOut.model_validate(user_to_dict(result.user)),
    )
