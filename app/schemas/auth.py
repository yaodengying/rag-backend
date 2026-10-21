"""认证相关 DTO。

契约设计的两条硬要求：
    1. **永远不出现 password_hash**：响应模型里根本没有这个字段，
       所以"不小心把哈希返回给前端"在类型层面就不可能发生。
    2. 登录与注册返回**同一个结构**（TokenOut）：前端两条路径只有一份处理逻辑；
       注册后直接拿到可用令牌（免得注册完再登录一次，两次都在算 PBKDF2）。

校验口径（与 services/auth_service.py 对齐）：
    长度、格式这类"入参形状"问题在 Pydantic 这层挡掉 → 422；
    "用户名已存在""密码错"这类依赖当前存储状态的问题在服务层判 → 409 / 401。
    这条界线要守住：把状态判断写进 DTO 会让校验器需要访问数据库。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

# 登录名允许：字母（含中文）、数字、下划线、点、@、连字符。
# `\w` 在 Python 的 re 里默认是 Unicode 语义，所以中文登录名（如「张三」）可用。
USERNAME_PATTERN = r"^[\w.@-]+$"

# 下限取 2 而不是 3：中文字符信息密度高，「李雷」两字就是完整姓名，
# 按西方习惯定 3 会把这些名字挡在门外。下限存在只为挡住 "a" 这类无意义输入，
# 长度不代表强度——口令那边才有强度要求。
USERNAME_MIN = 2
USERNAME_MAX = 64
PASSWORD_MIN = 6
PASSWORD_MAX = 128


class LoginRequest(BaseModel):
    """登录请求。"""

    username: str = Field(
        ..., min_length=1, max_length=USERNAME_MAX, description="登录名（不区分大小写）"
    )
    # 登录时**不**校验口令长度/格式：老账号可能不满足现在的强度要求，
    # 拦在这里会让它们彻底登录不进来。长度规则只在注册（set）时生效。
    password: str = Field(..., min_length=1, max_length=PASSWORD_MAX, description="口令")

    @field_validator("username", mode="before")
    @classmethod
    def _strip_username(cls, value: object) -> object:
        """去掉登录名两端的空白。

        在 before 阶段做，是为了让 min_length 作用在去空白**之后**：
        否则 "   " 能凑够长度、却被服务层当成空名拒掉，同一件错事报两个码。
        口令刻意**不做**任何处理——它是不可变字节序列，
        擅自 trim 等于悄悄改掉用户的密钥（在注册端改了、登录端也得同样改，
        那一处漏了就永远登不进）。
        """
        return value.strip() if isinstance(value, str) else value


class RegisterRequest(BaseModel):
    """注册请求。"""

    username: str = Field(
        ...,
        min_length=USERNAME_MIN,
        max_length=USERNAME_MAX,
        pattern=USERNAME_PATTERN,
        description=f"登录名，{USERNAME_MIN}~{USERNAME_MAX} 位字母/数字/下划线/点/连字符（可用中文）",
    )
    password: str = Field(
        ..., min_length=PASSWORD_MIN, max_length=PASSWORD_MAX, description="口令，至少 6 位"
    )
    display_name: str = Field("", max_length=64, description="展示名，缺省显示登录名")

    @field_validator("username", "display_name", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        """去两端空白（同 LoginRequest，理由见上）。"""
        return value.strip() if isinstance(value, str) else value


class UserOut(BaseModel):
    """账号信息（对外）。

    这个模型里**没有** password_hash 字段，且不要加——
    接口只该给出"这是谁、还能不能用"，凭据不属于可读信息。
    """

    id: str
    username: str
    display_name: str = ""
    is_active: bool = True
    created_at: datetime | None = None


class TokenOut(BaseModel):
    """登录 / 注册的成功响应。"""

    access_token: str = Field(..., description="访问令牌")
    token_type: str = Field("bearer", description="令牌类型，用于 Authorization 头")
    expires_in: int = Field(..., description="有效期（秒）")
    expires_at: datetime = Field(..., description="过期时刻（UTC）")
    user: UserOut
