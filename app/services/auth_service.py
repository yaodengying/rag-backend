"""认证服务：注册、登录、令牌校验。

职责边界：
    本服务负责"凭据对不对"与"令牌是不是我签的"，**不负责**"这个请求该不该放行"。
    后者是路由/依赖的事（见 app/api/v1/auth.py）。这条分界线很重要——
    把"放行判断"写进服务层，会让所有接口都间接依赖它，
    而本项目的其他接口目前**刻意不做任何鉴权**。

关于口令：绝不记录、绝不返回
    日志里只出现用户名。口令是用户在其他系统很可能复用的秘密，
    一旦进了日志文件就等于泄露；而 HTTP 响应里连 password_hash 的字段都不存在
    （见 schemas/auth.py 的 UserOut）。

用户名的规范化只有一个入口
    `normalize_username()` 被注册与登录**共用**。两处各写一遍的话，
    迟早出现"注册时小写、登录时不处理"这种错位——表现为
    「刚注册完就登不进去」，而且只在登录名含大写字母时复现。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from app.core.exceptions import (
    NotFoundError,
    UnauthorizedError,
    ValidationError,
)
from app.domain.models import User
from app.domain.ports import UserRepositoryPort
from app.infrastructure import security

logger = logging.getLogger(__name__)

# 口令最小长度。Pydantic 层也设了同样的值（挡在入口），
# 这里再判一次是因为服务层还有别的调用方（如 scripts/create_user.py），
# 它们不经过 DTO 校验。
MIN_PASSWORD_LENGTH = 6

# 登录失败的**统一话术**：账号不存在与口令错误必须返回同一句话。
# 分开写等于给攻击者一个免费的账号枚举接口。
LOGIN_FAILED_MESSAGE = "用户名或密码不正确"

# 登录失败的**统一 detail**。别小看这一项：异常处理器会把 exc.detail 原样写进
# 响应体（见 core/exceptions.app_error_handler），所以只统一 message 是不够的——
# 曾经这里写的是 reason=no_such_user / bad_password，于是消息一样、detail 不同，
# 攻击者读 detail 就能把两种失败区分开，前面所有防枚举的努力全部作废。
# 更细的原因只留在**服务端日志**里（logger.warning 的文案是区分的）。
CREDENTIALS_FAILED_DETAIL = {"reason": "bad_credentials"}


@dataclass(slots=True)
class AuthResult:
    """登录 / 注册成功后的产物。

    expires_in 与 expires_at 是同一件事的两种表达，都保留：
        - expires_in（秒）给前端直接塞进定时器；
        - expires_at（时刻）给人看日志/界面。
    让前端自己去减时间会引入客户端时钟偏差，所以不省这个字段。
    """

    user: User
    access_token: str
    expires_at: datetime
    expires_in: int


class AuthService:
    """账号与令牌服务。"""

    def __init__(
        self,
        repository: UserRepositoryPort,
        *,
        secret_key: str | None = None,
        ttl_minutes: int | None = None,
        allow_registration: bool | None = None,
        pbkdf2_iterations: int | None = None,
    ) -> None:
        """注入账号仓储；其余参数缺省时取 Settings。

        这些参数做成可注入的，是为了让自检脚本能用"短有效期 + 低迭代次数"
        跑完整流程（默认 21 万次迭代每次约 0.1 秒，几十个用例就变成分钟级）。
        生产装配（container.get_auth_service）只传仓储，一律走配置。

        Args:
            repository: 账号仓储
            secret_key: 令牌签名密钥；None 时取 settings.auth_secret_key
            ttl_minutes: 令牌有效期（分钟）；None 时取 settings.auth_token_ttl_minutes
            allow_registration: 是否允许注册；None 时取 settings.auth_allow_registration
            pbkdf2_iterations: 新口令的迭代次数；None 时取 settings.auth_pbkdf2_iterations
        """
        from app.core.config import get_settings

        settings = get_settings()
        self._repository = repository
        self._secret_key = settings.auth_secret_key if secret_key is None else secret_key
        self._ttl_minutes = (
            settings.auth_token_ttl_minutes if ttl_minutes is None else int(ttl_minutes)
        )
        self._allow_registration = (
            settings.auth_allow_registration
            if allow_registration is None
            else bool(allow_registration)
        )
        # 注意语义：这是**新口令**用的次数。校验已有口令时用的是
        # 哈希串里记的次数（见 security.verify_password），改这里不会锁住老账号。
        self._iterations = (
            settings.auth_pbkdf2_iterations
            if pbkdf2_iterations is None
            else int(pbkdf2_iterations)
        )

    # ---------- 写 ----------

    def register(self, username: str, password: str, display_name: str = "") -> AuthResult:
        """注册账号并直接签发令牌（**HTTP 自助注册通道**）。

        受 `AUTH_ALLOW_REGISTRATION` 约束——这是"陌生人能不能自助建号"这一
        策略问题的唯一落点。

        Raises:
            ValidationError: 注册已关闭、口令不合规、登录名为空
            ConflictError: 登录名已被占用
        """
        if not self._allow_registration:
            raise ValidationError(
                "注册功能已关闭（AUTH_ALLOW_REGISTRATION=false）。"
                "请由管理员创建账号：python scripts/create_user.py <用户名>"
            )
        return self._issue(self.create_user(username, password, display_name))

    def create_user(self, username: str, password: str, display_name: str = "") -> User:
        """建号（**管理员通道**，不受 AUTH_ALLOW_REGISTRATION 约束）。

        与 register() 分开的理由：注册开关管的是"陌生人能不能自助建号"，
        管不到"管理员能不能建号"。若两者走同一条路径，一旦关掉注册，
        就再没有任何办法建号了——包括本来该用的那条通道。

        Raises:
            ValidationError: 口令不合规、登录名为空
            ConflictError: 登录名已被占用
        """
        normalized = self.normalize_username(username)
        self._validate_password(password)
        password_hash = security.hash_password(password, self._iterations)
        user = self._repository.create_user(normalized, password_hash, display_name.strip())
        logger.info("账号已创建：%s（id=%s）", user.username, user.id)
        return user

    # ---------- 读 ----------

    def login(self, username: str, password: str) -> AuthResult:
        """校验凭据并签发令牌。

        Raises:
            UnauthorizedError: 账号不存在、口令错误或账号已停用
        """
        normalized = self.normalize_username(username)
        user = self._repository.get_user_by_username(normalized)

        if user is None:
            # 账号不存在也要付一次哈希的代价，否则响应时间直接把
            # "这个账号存不存在"泄漏出去（存在→约 100ms，不存在→约 1ms）。
            security.dummy_verify(password, self._iterations)
            logger.warning("登录失败（账号不存在）：%s", normalized)
            raise UnauthorizedError(LOGIN_FAILED_MESSAGE, detail=CREDENTIALS_FAILED_DETAIL)

        if not security.verify_password(password, user.password_hash):
            logger.warning("登录失败（口令错误）：%s", normalized)
            raise UnauthorizedError(LOGIN_FAILED_MESSAGE, detail=CREDENTIALS_FAILED_DETAIL)

        if not user.is_active:
            # 这里可以给出明确原因：对方已经证明了自己知道口令，
            # 不存在账号枚举问题，而含糊其辞只会让用户反复重试
            logger.warning("登录被拒（账号已停用）：%s", normalized)
            raise UnauthorizedError(
                "账号已被停用，请联系管理员", detail={"reason": "inactive"}
            )

        logger.info("登录成功：%s（id=%s）", user.username, user.id)
        return self._issue(user)

    def authenticate(self, token: str) -> User:
        """校验令牌并返回其主体账号。

        供**将来**的鉴权依赖使用；当前没有任何路由调用它。

        Raises:
            UnauthorizedError: 令牌非法/过期，或账号已不存在、已停用
        """
        payload = security.decode_access_token(
            token, secret_key=self._secret_key, now=datetime.now(timezone.utc)
        )
        user = self._repository.get_user_by_id(str(payload["sub"]))
        if user is None:
            # 令牌本身有效但账号没了：删号后旧令牌必须立刻失效，
            # 否则"删掉的账号"还能继续用——这是最容易被忽略的一条。
            raise UnauthorizedError("账号不存在或已被删除", detail={"reason": "no_such_user"})
        if not user.is_active:
            raise UnauthorizedError("账号已被停用", detail={"reason": "inactive"})
        return user

    def get_user(self, user_id: str) -> User | None:
        """按 ID 取账号（不存在返回 None）。"""
        return self._repository.get_user_by_id(user_id)

    def get_user_by_username(self, username: str) -> User | None:
        """按登录名取账号（走同一套规范化，不存在返回 None）。

        Raises:
            ValidationError: 登录名为空
        """
        return self._repository.get_user_by_username(self.normalize_username(username))

    def list_users(self) -> list[User]:
        """列出全部账号（供运维脚本与自检使用）。"""
        return self._repository.list_users()

    def reset_password(self, user_id: str, new_password: str) -> User:
        """重置口令（供 scripts/create_user.py 使用）。

        刻意**不做成 HTTP 接口**：一个"用旧口令就能改新口令"的接口需要
        鉴权（本项目还没有），一个"不要旧口令就能改"的接口等于后门。
        所以改口令只走运维通道——能跑这条命令的人本来就有服务器权限。

        Raises:
            ValidationError: 新口令不合规
            NotFoundError: 账号不存在
        """
        self._validate_password(new_password)
        user = self._repository.get_user_by_id(user_id)
        if user is None:
            raise NotFoundError(f"账号不存在：{user_id}")
        self._repository.update_password_hash(
            user_id, security.hash_password(new_password, self._iterations)
        )
        logger.info("口令已重置：%s（id=%s）", user.username, user.id)
        return user

    # ---------- 内部 ----------

    def _issue(self, user: User) -> AuthResult:
        """签发令牌并组装结果。"""
        token, expires_at = security.create_access_token(
            user_id=user.id,
            username=user.username,
            secret_key=self._secret_key,
            ttl_minutes=self._ttl_minutes,
        )
        return AuthResult(
            user=user,
            access_token=token,
            expires_at=expires_at,
            expires_in=self._ttl_minutes * 60,
        )

    @staticmethod
    def normalize_username(raw: str) -> str:
        """登录名的唯一规范化点：去两端空白 + 转小写。

        Raises:
            ValidationError: 规范化后为空
        """
        normalized = (raw or "").strip().lower()
        if not normalized:
            raise ValidationError("用户名不能为空")
        return normalized

    @staticmethod
    def _validate_password(password: str) -> None:
        """口令强度校验（服务层兜底，DTO 层已挡过一次）。

        只挡"纯空白"与"太短"：口令是用户自己的秘密，不宜引入
        "必须含大小写+数字"这类规则——它们把用户推向 `Abc123!` 这种
        更好猜的弱口令，本项目不做这种自我安慰式的复杂度要求。

        Raises:
            ValidationError: 口令为空/纯空白，或长度不足
        """
        if not password or not password.strip():
            raise ValidationError("口令不能为空或纯空白")
        if len(password) < MIN_PASSWORD_LENGTH:
            raise ValidationError(f"口令长度至少 {MIN_PASSWORD_LENGTH} 位")
