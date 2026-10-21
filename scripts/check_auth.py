# -*- coding: utf-8 -*-
"""认证链路自检：验证「注册 / 登录 / 令牌签发与校验」这一段是对的，
并证明**全部业务接口都强制校验令牌**（这一步已从"不校验"升级为"强校验"）。

为什么需要这个脚本：
    认证这段代码的特点是「错了也不报错」：
      ① 口令哈希写成明文比较，功能上完全正常，只是全库口令等于公开；
      ② 哈希串里的迭代次数被忽略、改配置就把老账号全部锁在门外；
      ③ 登录失败的两条分支话术不一致 —— 白送攻击者一个账号枚举接口；
      ④ 令牌验签被跳过（或先判过期再验签），伪造的令牌照样能过；
      ⑤ 最隐蔽的一条：**漏挂**鉴权依赖。少挂一个接口不会有任何报错，
         只会在上线后表现为"这个接口谁都能调"。
         所以下面改成"逐条枚举 + 白名单反向断言"，不再抽样。
    这些都不会让任何测试"失败"，只能靠断言钉住。

判据（不只断言"能登录"）：

    | 检查项            | 期望                                                          |
    |------------------|---------------------------------------------------------------|
    | 口令哈希          | 对/错口令分别 True/False；同一口令两次哈希不同（随机盐）        |
    | 哈希串自描述      | 4 段且含算法与迭代次数；改用例次数哈希的老口令仍能校验          |
    | 哈希串损坏        | 篡改摘要/盐/算法/段数 → 一律 False，**不抛异常**                |
    | 明文不外泄        | 哈希串、user_to_dict、repr(user)、HTTP 响应里都没有明文/哈希    |
    | 用户规范化        | ` Alice ` 与 `ALICE` 是同一个账号；重复注册 → 409 且不产生新账号 |
    | 防账号枚举        | 账号不存在与口令错误返回**完全一样**的消息                     |
    | 口令强度          | 太短/纯空白/空 → ValidationError（服务层兜底，不只靠 DTO）      |
    | 令牌签发          | sub=用户 id；exp-iat=ttl；payload 无口令字段                    |
    | 令牌校验          | 换密钥/改签名 → bad_signature；缺主体/无有效期/无点 → 明确原因  |
    | 过期语义          | now>=exp 即失效；过期判定必须在验签**之后**                    |
    | 空密钥            | 签发抛 ValueError、校验抛 UnauthorizedError（不留"可伪造"通道） |
    | 停用 / 删号       | 停用后登录被拒且旧令牌立即失效；删号后旧令牌也立即失效          |
    | 关闭注册          | AUTH_ALLOW_REGISTRATION=false → 注册被拒，登录不受影响          |
    | HTTP 登录         | 200 + access_token/token_type/expires_in/expires_at/user        |
    | HTTP 注册         | 201，结构同登录（前端只写一套逻辑）                            |
    | HTTP 失败码       | 401 凭据错 / 409 重名 / 422 入参形状                            |
    | 统一错误结构      | {code,message,detail}，与 app.core.exceptions 的约定一致        |
    | 鉴权依赖          | 探针路由挂上它 → 无令牌 401、正确令牌 200、非 Bearer 401        |
    | **强制鉴权**      | 除 health / login / register 外，每条路由都挂 require_current_user |
    | 配置守门          | 空/短密钥、低迭代次数、非正 TTL、prod 开放注册 → 启动期拦下     |
    | DTO 与服务一致    | PASSWORD_MIN == MIN_PASSWORD_LENGTH（防两处漂移）               |

运行（**不需要 Milvus、BGE 权重、Ollama**）：
    .venv\\Scripts\\python.exe scripts\\check_auth.py

    账号部分用**临时 SQLite 文件**（真仓储、真唯一约束），
    停用/删号这两种状态不好用真库造，用内存假仓储补；
    HTTP 部分用 TestClient，只挂 auth 路由，不触发 lifespan（不连 Milvus）。

退出码：0 全部通过；1 存在失败项。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sys
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# settings 的 env_file 是相对路径，必须先切到项目根
os.chdir(PROJECT_ROOT)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from fastapi import Depends, FastAPI  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError as PydanticValidationError  # noqa: E402
from sqlalchemy import inspect as sa_inspect  # noqa: E402

from app.api.deps import get_auth_service  # noqa: E402
from app.api.v1 import auth as auth_api  # noqa: E402
from app.core.config import Settings, get_settings  # noqa: E402
from app.core.exceptions import (  # noqa: E402
    ConfigurationError,
    ConflictError,
    NotFoundError,
    UnauthorizedError,
    ValidationError,
    register_exception_handlers,
)
from app.db.database import Database  # noqa: E402
from app.db.repositories import SqlAlchemyUserRepository  # noqa: E402
from app.domain.models import User  # noqa: E402
from app.domain.ports import UserRepositoryPort  # noqa: E402
from app.infrastructure import security  # noqa: E402
from app.main import create_app  # noqa: E402
from app.schemas import auth as auth_schema  # noqa: E402
from app.services.auth_service import (  # noqa: E402
    LOGIN_FAILED_MESSAGE,
    MIN_PASSWORD_LENGTH,
    AuthService,
)
from app.services.mappers import user_to_dict  # noqa: E402

# 自检用密钥与迭代次数
#   迭代次数故意调低：默认 21 万次每次约 0.1 秒，几十个用例会变成分钟级；
#   而"改迭代次数不会锁住老账号"这条恰恰要用不同次数来验证。
TEST_SECRET = "check-auth-only-secret-key-0123456789abcdef"
OTHER_SECRET = "another-secret-key-0123456789abcdefghij"
FAST_ITERATIONS = 1_000
TTL_MINUTES = 30
HTTP = "/api/v1/auth"

# ---------------------------------------------------------------------------
# 极简断言器：收集全部失败项，而不是第一个就退出
# ---------------------------------------------------------------------------

_failures: list[str] = []
_checks = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    """记录一项断言结果。"""
    global _checks
    _checks += 1
    if condition:
        print(f"  [ok]   {label}")
    else:
        print(f"  [FAIL] {label}" + (f"  -> {detail}" if detail else ""))
        _failures.append(label)


def expect_error(label: str, exc_type: type[Exception], keyword: str, action) -> None:
    """断言 action 抛出指定类型的异常，且消息里含关键词。"""
    global _checks
    _checks += 1
    try:
        action()
    except exc_type as exc:
        if keyword in str(exc):
            print(f"  [ok]   {label}（{type(exc).__name__}: {str(exc)[:56]}…）")
        else:
            print(f"  [FAIL] {label}  -> 消息里没有 {keyword!r}：{exc}")
            _failures.append(label)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {label}  -> 期望 {exc_type.__name__}，实际 {type(exc).__name__}: {exc}")
        _failures.append(label)
    else:
        print(f"  [FAIL] {label}  -> 没有抛异常")
        _failures.append(label)


def section(title: str) -> None:
    """打印分组标题。"""
    print()
    print(f"=== {title} ===")


def _message_of(exc_type: type[Exception], action) -> str:
    """取 action 抛出的异常消息（没抛则返回空串）。"""
    try:
        action()
    except exc_type as exc:
        return str(exc)
    except Exception:  # noqa: BLE001
        return ""
    return ""


def _b64(raw: bytes) -> str:
    """与 security 内部一致的 base64url 编码（自检刻意自己实现一份）。"""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _sign(body: str, secret: str) -> str:
    """自己按约定复算签名。

    刻意不调 security 的内部函数：这样"签名只取决于 body 字符串"这一约定
    就变成可外部复现的事实，而不是"实现说了算"。
    """
    return _b64(hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest())


def _forge(payload: dict, secret: str = TEST_SECRET) -> str:
    """用手工 payload 造一个签名正确的令牌（用来测各种畸形 payload）。"""
    body = _b64(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    return f"{body}.{_sign(body, secret)}"


# ---------------------------------------------------------------------------
# 假仓储（只造"真库不好造"的状态）
# ---------------------------------------------------------------------------


class FakeUserRepository:
    """内存账号仓储：用来构造停用、删号这类状态。

    真 SQLite 仓储在下面第 3 节里被真实使用（含唯一约束），
    这里只补"改状态"的部分——不是为了替代真实现。
    """

    def __init__(self) -> None:
        self._users: dict[str, User] = {}

    def create_user(self, username: str, password_hash: str, display_name: str = "") -> User:
        if any(item.username == username for item in self._users.values()):
            raise ConflictError(f"登录名已被占用：{username}")
        user = User(
            id=str(uuid.uuid4()),
            username=username,
            password_hash=password_hash,
            display_name=display_name,
        )
        self._users[user.id] = user
        return user

    def get_user_by_username(self, username: str) -> User | None:
        return next((u for u in self._users.values() if u.username == username), None)

    def get_user_by_id(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    def list_users(self) -> list[User]:
        return list(self._users.values())

    def update_password_hash(self, user_id: str, password_hash: str) -> bool:
        user = self._users.get(user_id)
        if user is None:
            return False
        user.password_hash = password_hash
        return True

    def delete(self, user_id: str) -> None:
        """删号（真仓储里对应 DELETE，这里只为构造状态）。"""
        self._users.pop(user_id, None)


def _service(repository, **overrides) -> AuthService:
    """构造一个走自检参数的认证服务。"""
    params = {
        "secret_key": TEST_SECRET,
        "ttl_minutes": TTL_MINUTES,
        "pbkdf2_iterations": FAST_ITERATIONS,
        "allow_registration": True,
    }
    params.update(overrides)
    return AuthService(repository, **params)


@contextmanager
def _temp_repository() -> Iterator[tuple[SqlAlchemyUserRepository, Database]]:
    """在临时目录里建一个真 SQLite 库，产出仓储。

    退出顺序有讲究：**先 dispose 引擎、再删目录**。
    Windows 上 SQLite 文件被连接持有，先删会撞 PermissionError（WinError 32），
    而 `TemporaryDirectory` 的清理发生在 with 块结束之后——不在 finally 里
    提前释放连接，就会在一个"全部断言通过"的脚本末尾抛异常，很难看也很好骗。
    """
    with tempfile.TemporaryDirectory() as tmp:
        database = Database(str(Path(tmp) / "auth-check.db"))
        database.init_schema()
        try:
            yield SqlAlchemyUserRepository(database), database
        finally:
            database.dispose()


# ---------------------------------------------------------------------------
# 1. 口令哈希
# ---------------------------------------------------------------------------


def check_password_hashing() -> None:
    section("1. 口令哈希")

    password = "正确口令-123"
    stored = security.hash_password(password, FAST_ITERATIONS)

    check("verify 正确口令 → True", security.verify_password(password, stored) is True)
    check("verify 错误口令 → False", security.verify_password("错误口令-124", stored) is False)
    check("verify 区分尾部空格（不擅自 trim）",
          security.verify_password(password + " ", stored) is False)

    again = security.hash_password(password, FAST_ITERATIONS)
    check("同一口令两次哈希不同（随机盐）", stored != again)
    check("两条不同的哈希都能校验通过",
          security.verify_password(password, stored) and security.verify_password(password, again))

    parts = stored.split("$")
    check("哈希串是自描述的 4 段", len(parts) == 4 and parts[0] == security.ALGORITHM,
          stored[:40])
    check("哈希串里记着迭代次数", parts[1] == str(FAST_ITERATIONS), parts[1])
    check("哈希串里不含明文口令", password not in stored)

    # 这一条是"改配置不会把老账号锁在门外"的可执行证明：
    # 用 500 次哈希，再用当前配置（210000 次）的服务去校验。
    legacy = security.hash_password(password, 500)
    check("按哈希串自带的迭代次数校验（改 AUTH_PBKDF2_ITERATIONS 不会锁住老账号）",
          security.verify_password(password, legacy) is True)

    # 篡改与损坏：必须一律 False，且**不抛异常**（抛异常会让状态码泄漏信息）
    algo, rounds, salt_b64, digest_b64 = stored.split("$")
    broken_cases = {
        "摘要被改": f"{algo}${rounds}${salt_b64}${_b64(b'x' * 32)}",
        "盐被改": f"{algo}${rounds}${_b64(b'y' * 16)}${digest_b64}",
        "算法名被换": f"scrypt${rounds}${salt_b64}${digest_b64}",
        "段数不足": f"{algo}${rounds}${salt_b64}",
        "迭代次数为 0": f"{algo}$0${salt_b64}${digest_b64}",
        "迭代次数非数字": f"{algo}$abc${salt_b64}${digest_b64}",
        "非 base64": f"{algo}${rounds}$!!!$???",
        "空串": "",
        "纯文本": "not-a-hash",
    }
    failures = []
    for label, broken in broken_cases.items():
        try:
            if security.verify_password(password, broken) is not False:
                failures.append(f"{label}(返回了 True)")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label}(抛了 {type(exc).__name__})")
    check(f"损坏的哈希串一律返回 False 且不抛异常（{len(broken_cases)} 种）",
          not failures, "；".join(failures))

    # 空口令在 security 层是"能算能验"的——强度校验属于服务层，
    # 这条断言把责任边界钉住（免得有人以为 security 会拦空口令）
    check("空口令在 security 层不拦（强度校验在服务层）",
          security.verify_password("", security.hash_password("", FAST_ITERATIONS)))

    try:
        security.dummy_verify("whatever", FAST_ITERATIONS)
        dummy_ok = True
    except Exception as exc:  # noqa: BLE001
        dummy_ok = False
        detail = f"{type(exc).__name__}: {exc}"
    else:
        detail = ""
    check("dummy_verify 正常返回（用于抹平账号不存在的时间差）", dummy_ok, detail)


# ---------------------------------------------------------------------------
# 2. 访问令牌
# ---------------------------------------------------------------------------


def check_token() -> None:
    section("2. 访问令牌")

    user_id = str(uuid.uuid4())
    token, expires_at = security.create_access_token(
        user_id=user_id,
        username="alice",
        secret_key=TEST_SECRET,
        ttl_minutes=TTL_MINUTES,
    )

    check("令牌是点分两段", token.count(".") == 1, token[:40])
    check("令牌里看不到明文用户名", "alice" not in token)

    payload = security.decode_access_token(token, secret_key=TEST_SECRET)
    check("payload.sub 是用户 ID", payload.get("sub") == user_id)
    check("payload 字段恰好是 sub/username/iat/exp（没有口令类字段）",
          set(payload) == {"sub", "username", "iat", "exp"}, str(sorted(payload)))
    check("iat 与 exp 是 int（不是字符串/浮点）",
          isinstance(payload.get("iat"), int) and isinstance(payload.get("exp"), int))
    check("exp - iat = ttl 秒", payload["exp"] - payload["iat"] == TTL_MINUTES * 60,
          str(payload["exp"] - payload["iat"]))
    check("返回的过期时刻与 exp 一致",
          abs(expires_at.timestamp() - payload["exp"]) < 1)

    # 验签
    expect_error(
        "换一把密钥验签 → bad_signature",
        UnauthorizedError,
        "签名",
        lambda: security.decode_access_token(token, secret_key=OTHER_SECRET),
    )
    expect_error(
        "签名段被替换 → bad_signature",
        UnauthorizedError,
        "签名",
        lambda: security.decode_access_token(
            f"{token.split('.')[0]}.{_sign(token.split('.')[0], OTHER_SECRET)}",
            secret_key=TEST_SECRET,
        ),
    )
    expect_error(
        "格式非法 → malformed",
        UnauthorizedError,
        "格式",
        lambda: security.decode_access_token("no-dot-here", secret_key=TEST_SECRET),
    )
    expect_error(
        "空串 → malformed",
        UnauthorizedError,
        "格式",
        lambda: security.decode_access_token("", secret_key=TEST_SECRET),
    )

    # 畸形 payload（签名正确，内容不合规）——这类只能靠自造令牌覆盖
    expect_error(
        "缺 sub → missing_subject",
        UnauthorizedError,
        "主体",
        lambda: security.decode_access_token(
            _forge({"username": "x", "iat": 1, "exp": 2**31}), secret_key=TEST_SECRET
        ),
    )
    expect_error(
        "缺 exp → missing_exp",
        UnauthorizedError,
        "有效期",
        lambda: security.decode_access_token(
            _forge({"sub": user_id}), secret_key=TEST_SECRET
        ),
    )

    # 过期语义
    moment = datetime.now(timezone.utc)
    short = security.create_access_token(
        user_id=user_id,
        username="alice",
        secret_key=TEST_SECRET,
        ttl_minutes=TTL_MINUTES,
        now=moment,
    )[0]
    check("ttl 内有效", bool(security.decode_access_token(
        short, secret_key=TEST_SECRET, now=moment + timedelta(minutes=TTL_MINUTES - 1)
    )))
    expect_error(
        "正好到期即失效（now >= exp 判定）",
        UnauthorizedError,
        "过期",
        lambda: security.decode_access_token(
            short, secret_key=TEST_SECRET, now=moment + timedelta(minutes=TTL_MINUTES)
        ),
    )
    already_expired = int((moment - timedelta(minutes=1)).timestamp())
    expect_error(
        "已过期 → expired",
        UnauthorizedError,
        "过期",
        lambda: security.decode_access_token(
            _forge({"sub": user_id, "exp": already_expired}), secret_key=TEST_SECRET
        ),
    )

    # 空密钥：宁可报错，也不留一条"人人可伪造"的通道
    expect_error(
        "空密钥签发 → ValueError（不签出可伪造的令牌）",
        ValueError,
        "AUTH_SECRET_KEY",
        lambda: security.create_access_token(
            user_id=user_id, username="alice", secret_key="", ttl_minutes=1
        ),
    )
    expect_error(
        "空密钥校验 → UnauthorizedError",
        UnauthorizedError,
        "AUTH_SECRET_KEY",
        lambda: security.decode_access_token(token, secret_key=""),
    )


# ---------------------------------------------------------------------------
# 3. 仓储与服务（真 SQLite）
# ---------------------------------------------------------------------------


def check_service_with_repository() -> None:
    section("3. 仓储与服务（临时 SQLite，真唯一约束）")

    with _temp_repository() as (repository, database):
        check("SqlAlchemyUserRepository 满足 UserRepositoryPort",
              isinstance(repository, UserRepositoryPort))
        check("users 表由 init_schema 建出（老库无需迁移）",
              "users" in sa_inspect(database.engine).get_table_names())

        service = _service(repository)
        created = service.register(" Alice ", "good-pass-1", "  爱丽丝  ")

        check("注册返回令牌与账号", bool(created.access_token) and created.user.username == "alice")
        check("登录名已规范化（去空白 + 小写）", created.user.username == "alice")
        check("展示名已去空白", created.user.display_name == "爱丽丝")
        check("库里存的是哈希而非明文",
              created.user.password_hash.startswith("pbkdf2_sha256$")
              and "good-pass-1" not in created.user.password_hash)
        check("user_to_dict 不带 password_hash", "password_hash" not in user_to_dict(created.user))
        check("repr(user) 不带口令哈希（防日志泄露）",
              "pbkdf2_sha256" not in repr(created.user), repr(created.user))

        # 规范化只有一个入口：注册与登录共用
        check("用 ALICE 登录 → 同一账号",
              service.login("ALICE", "good-pass-1").user.id == created.user.id)
        check("用两端带空白的登录名 → 同一账号",
              service.login("  alice  ", "good-pass-1").user.id == created.user.id)

        expect_error(
            "重复注册（大小写不同）→ ConflictError",
            ConflictError,
            "已被占用",
            lambda: service.register("ALICE", "another-pass"),
        )
        check("重复注册没有产生第二个账号（唯一约束生效，非「先查再写」）",
              len(service.list_users()) == 1, f"实际 {len(service.list_users())} 个")

        # 口令强度（服务层兜底：scripts/create_user.py 不经过 DTO）
        expect_error("口令过短 → ValidationError", ValidationError, str(MIN_PASSWORD_LENGTH),
                     lambda: service.register(f"u{uuid.uuid4().hex[:6]}", "12345"))
        expect_error("口令纯空白 → ValidationError", ValidationError, "空白",
                     lambda: service.register(f"u{uuid.uuid4().hex[:6]}", "      "))
        expect_error("口令为空 → ValidationError", ValidationError, "口令",
                     lambda: service.register(f"u{uuid.uuid4().hex[:6]}", ""))
        expect_error("登录名为空 → ValidationError", ValidationError, "用户名",
                     lambda: service.register("   ", "good-pass-1"))

        # 防账号枚举：两条分支的消息必须**逐字相同**
        bad_password = _message_of(UnauthorizedError, lambda: service.login("alice", "wrong-pass"))
        no_such_user = _message_of(UnauthorizedError, lambda: service.login("nobody", "wrong-pass"))
        check("账号不存在与口令错误话术逐字一致（防账号枚举）",
              bad_password == no_such_user == LOGIN_FAILED_MESSAGE,
              f"{bad_password!r} vs {no_such_user!r}")

        # 令牌
        check("有效令牌可解出账号",
              service.authenticate(created.access_token).id == created.user.id)
        check("同一账号可同时持有多枚令牌（无会话表，服务端不记录）",
              service.authenticate(created.access_token).id
              == service.authenticate(service.login("alice", "good-pass-1").access_token).id)
        expect_error(
            "拿着 B 服务的密钥签的令牌 → 校验失败",
            UnauthorizedError,
            "签名",
            lambda: service.authenticate(
                _forge({"sub": created.user.id, "exp": 2**31}, secret=OTHER_SECRET)
            ),
        )
        expect_error(
            "令牌里的 sub 指向不存在的账号 → 失效",
            UnauthorizedError,
            "不存在",
            lambda: service.authenticate(
                _forge({"sub": str(uuid.uuid4()), "exp": 2**31})
            ),
        )

        expired_service = _service(repository, ttl_minutes=0)
        expired_token = expired_service.login("alice", "good-pass-1").access_token
        expect_error("ttl=0 的令牌立即过期", UnauthorizedError, "过期",
                     lambda: service.authenticate(expired_token))

        # 关闭注册
        closed = _service(repository, allow_registration=False)
        expect_error("AUTH_ALLOW_REGISTRATION=false → 注册被拒", ValidationError, "已关闭",
                     lambda: closed.register("bob", "good-pass-1"))
        check("关闭注册不影响登录",
              closed.login("alice", "good-pass-1").user.id == created.user.id)

        # 管理员通道不能跟着自助注册一起被关掉——否则关掉注册之后
        # 就再也没有任何办法建号了（脚本存在的前提就是这条）
        admin_created = closed.create_user("bob", "good-pass-1", "  鲍勃  ")
        check("关闭注册后管理员通道仍可建号（create_user 不受开关约束）",
              admin_created.username == "bob" and admin_created.display_name == "鲍勃")
        check("管理员通道的重名判定同样生效",
              _raises(ConflictError, lambda: closed.create_user("BOB", "good-pass-1")))

        # 重置口令
        expect_error("重置口令：弱口令被拒", ValidationError, "口令",
                     lambda: closed.reset_password(admin_created.id, "123"))
        expect_error("重置口令：账号不存在 → NotFoundError", NotFoundError, "不存在",
                     lambda: closed.reset_password(str(uuid.uuid4()), "good-pass-2"))
        closed.reset_password(admin_created.id, "new-pass-2")
        expect_error("重置后旧口令立即失效", UnauthorizedError, LOGIN_FAILED_MESSAGE,
                     lambda: closed.login("bob", "good-pass-1"))
        check("重置后新口令可用",
              closed.login("bob", "new-pass-2").user.id == admin_created.id)

        check("list_users 按创建时间返回全部账号", len(closed.list_users()) == 2)
        check("按登录名查账号（走同一套小写规范化）",
              closed.get_user_by_username("  BOB  ").id == admin_created.id)


# ---------------------------------------------------------------------------
# 4. 停用与删号（内存仓储）
# ---------------------------------------------------------------------------


def check_inactive_and_deleted() -> None:
    section("4. 停用与删号（内存仓储构造状态）")

    repository = FakeUserRepository()
    check("FakeUserRepository 满足 UserRepositoryPort",
          isinstance(repository, UserRepositoryPort))
    service = _service(repository)
    created = service.register("carol", "good-pass-1")

    stored = repository.get_user_by_username("carol")
    stored.is_active = False

    expect_error("账号停用后登录被拒（并给出明确原因）", UnauthorizedError, "停用",
                 lambda: service.login("carol", "good-pass-1"))
    expect_error("账号停用后旧令牌立即失效（不必等它过期）", UnauthorizedError, "停用",
                 lambda: service.authenticate(created.access_token))

    stored.is_active = True
    check("恢复启用后旧令牌又能用（停用不是吊销令牌，令牌本身未被作废）",
          service.authenticate(created.access_token).username == "carol")

    repository.delete(created.user.id)
    expect_error("账号被删后旧令牌立即失效（令牌有 sub，账号没了）", UnauthorizedError, "不存在",
                 lambda: service.authenticate(created.access_token))
    check("删号后账号查不到", service.get_user(created.user.id) is None)


# ---------------------------------------------------------------------------
# 5. HTTP 接口
# ---------------------------------------------------------------------------


def _build_app(service: AuthService) -> FastAPI:
    """只挂 auth 路由的最小应用（不跑 lifespan，故不连 Milvus / 不加载权重）。"""
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_api.router, prefix=HTTP)
    app.dependency_overrides[get_auth_service] = lambda: service
    return app


def check_http_routes() -> None:
    section("5. HTTP 接口（TestClient）")

    with _temp_repository() as (repository, _database):
        service = _service(repository)
        app = _build_app(service)

        with TestClient(app) as client:
            # --- 注册 ---
            response = client.post(
                f"{HTTP}/register",
                json={"username": "Dave", "password": "good-pass-1", "display_name": "大卫"},
            )
            check("POST /auth/register → 201", response.status_code == 201, response.text[:160])
            body = response.json()
            check("响应字段齐全（access_token/token_type/expires_in/expires_at/user）",
                  set(body) == {"access_token", "token_type", "expires_in", "expires_at", "user"},
                  str(sorted(body)))
            check("token_type=bearer", body["token_type"] == "bearer")
            check("expires_in = ttl 秒", body["expires_in"] == TTL_MINUTES * 60,
                  str(body["expires_in"]))
            check("user.username 已规范化", body["user"]["username"] == "dave", body["user"]["username"])
            check("user 结构里没有凭据字段", set(body["user"]) ==
                  {"id", "username", "display_name", "is_active", "created_at"},
                  str(sorted(body["user"])))
            check("整段响应里没有任何哈希痕迹",
                  "pbkdf2" not in response.text and "password" not in json.dumps(body["user"]))
            token = body["access_token"]
            user_id = body["user"]["id"]

            # --- 登录 ---
            response = client.post(
                f"{HTTP}/login", json={"username": "  DAVE  ", "password": "good-pass-1"}
            )
            check("POST /auth/login → 200", response.status_code == 200, response.text[:160])
            check("登录返回同一账号", response.json()["user"]["id"] == user_id)
            check("登录响应的令牌同样可用",
                  bool(security.decode_access_token(response.json()["access_token"],
                                                    secret_key=TEST_SECRET)))

            wrong = client.post(f"{HTTP}/login",
                               json={"username": "dave", "password": "wrong-pass"})
            missing = client.post(f"{HTTP}/login",
                                  json={"username": "nobody", "password": "wrong-pass"})
            check("口令错误 → 401", wrong.status_code == 401, str(wrong.status_code))
            check("账号不存在 → 401", missing.status_code == 401, str(missing.status_code))
            check("两条 401 的响应体逐字相同（含 detail：只统一 message 不够，"
                  "detail 里的 reason 同样能用来枚举账号）",
                  wrong.json() == missing.json(), f"{wrong.text} vs {missing.text}")
            check("错误响应体符合统一结构 {code,message,detail}",
                  set(wrong.json()) == {"code", "message", "detail"}, str(sorted(wrong.json())))
            check("错误码是 unauthorized", wrong.json()["code"] == "unauthorized")
            check("错误消息就是统一话术", wrong.json()["message"] == LOGIN_FAILED_MESSAGE)

            # --- 409 重名 ---
            response = client.post(f"{HTTP}/register",
                                   json={"username": "DAVE", "password": "good-pass-2"})
            check("重复注册 → 409", response.status_code == 409, str(response.status_code))
            check("409 的错误码是 conflict", response.json()["code"] == "conflict")

            # --- 422 入参形状 ---
            for label, payload in {
                "用户名过短": {"username": "a", "password": "good-pass-1"},
                "用户名含空格": {"username": "a b", "password": "good-pass-1"},
                "口令过短": {"username": "eve", "password": "12345"},
                "口令超长（挡 PBKDF2 变 CPU 炸弹）":
                    {"username": "eve", "password": "x" * (auth_schema.PASSWORD_MAX + 1)},
                "缺字段": {"username": "eve"},
            }.items():
                response = client.post(f"{HTTP}/register", json=payload)
                check(f"{label} → 422", response.status_code == 422, str(response.status_code))
                if response.status_code == 422:
                    check(f"{label} 的错误码是 validation_error",
                          response.json()["code"] == "validation_error")
                else:
                    check(f"{label} 的错误码是 validation_error", False)

            response = client.post(f"{HTTP}/login", json={"username": "   ", "password": "x"})
            check("登录名纯空白 → 422（去空白后才判长度）",
                  response.status_code == 422, str(response.status_code))

        # --- 鉴权依赖本身（挂到探针路由上才看得见） ---
        probe = FastAPI()
        register_exception_handlers(probe)

        @probe.get("/probe")
        def _probe(user: auth_schema.UserOut = Depends(auth_api.require_current_user)) -> dict:
            return {"username": user.username, "id": user.id}

        probe.dependency_overrides[get_auth_service] = lambda: service
        with TestClient(probe) as pclient:
            response = pclient.get("/probe")
            check("探针路由不带 Authorization → 401", response.status_code == 401,
                  str(response.status_code))
            check("缺令牌的 code 是 unauthorized", response.json()["code"] == "unauthorized")
            response = pclient.get("/probe", headers={"Authorization": f"Bearer {token}"})
            check("带正确令牌 → 200 且返回当前用户",
                  response.status_code == 200 and response.json()["id"] == user_id,
                  response.text[:160])
            check("scheme 大小写不敏感（RFC 7235：bearer 也认）",
                  pclient.get("/probe",
                              headers={"Authorization": f"bearer {token}"}).status_code == 200)
            check("非 Bearer scheme → 401",
                  pclient.get("/probe",
                              headers={"Authorization": f"Basic {token}"}).status_code == 401)
            check("Bearer 后为空 → 401",
                  pclient.get("/probe", headers={"Authorization": "Bearer    "}).status_code == 401)
            check("令牌被改坏 → 401",
                  pclient.get("/probe",
                              headers={"Authorization": f"Bearer {token}x"}).status_code == 401)


# ---------------------------------------------------------------------------
# 6. "登录有、鉴权没有"
# ---------------------------------------------------------------------------


def _collect_api_routes(routes, collected: list | None = None) -> list[APIRoute]:
    """摊平出真实应用里的 APIRoute。

    注意：FastAPI 0.143 的 include_router 是**惰性**的，`app.routes` 里是
    `_IncludedRouter` 包装对象而不是 APIRoute，直接数路由会误判"没注册"。
    包装对象上挂着 `original_router`，从这里递归下去才是真路由。
    """
    collected = [] if collected is None else collected
    for route in routes:
        if isinstance(route, APIRoute):
            collected.append(route)
        elif getattr(route, "original_router", None) is not None:
            _collect_api_routes(route.original_router.routes, collected)
        elif getattr(route, "routes", None):
            _collect_api_routes(route.routes, collected)
    return collected


def _dependency_calls(route: APIRoute) -> set:
    """递归取出一个路由的全部依赖函数。"""
    calls: set = set()
    stack = list(route.dependant.dependencies)
    while stack:
        dependency = stack.pop()
        calls.add(dependency.call)
        stack.extend(dependency.dependencies)
    return calls


# 免鉴权白名单：健康检查（容器探针要用）与登录/注册（拿令牌的前提），
# 外加根路径（它只是跳到 /docs 的重定向，不含任何数据）。
# **只能有这几项**，新增任何一项都等于开了一个匿名入口。
#
# 注意路径口径：`_collect_api_routes` 拿到的是**子路由相对路径**
# （子路由的 prefix 在 app.include_router 时才拼上，不会回写到 route.path），
# 所以这里是 "/login" 而不是 "/api/v1/auth/login"。
_PUBLIC_PATHS = {
    "/health",
    "/ready",
    "/login",
    "/register",
    "/",
}


def check_auth_enforcement() -> None:
    section("6. 确认「全部业务路由都挂了鉴权」")

    app = create_app()
    paths = app.openapi()["paths"]
    check("应用已注册 /api/v1/auth/login", "/api/v1/auth/login" in paths)
    check("应用已注册 /api/v1/auth/register", "/api/v1/auth/register" in paths)
    check("应用已注册 /api/v1/auth/me", "/api/v1/auth/me" in paths)

    routes = _collect_api_routes(app.routes)
    check(f"真实应用共 {len(routes)} 条路由可枚举（>10 说明包装对象被正确展开）",
          len(routes) >= 15, str(len(routes)))

    enforced = {
        route.path
        for route in routes
        if auth_api.require_current_user in _dependency_calls(route)
    }

    # 核心断言：除白名单外，**每一条**业务路由都必须挂鉴权。
    # 写成"逐条检查遗漏"而不是"抽查几条"，是因为漏挂一个就是一次数据泄漏，
    # 而抽查永远只覆盖到现在有的路由，新增的接口不会被自动保护。
    unprotected = sorted(
        {
            route.path
            for route in routes
            if auth_api.require_current_user not in _dependency_calls(route)
        }
        - _PUBLIC_PATHS
    )
    check("**除 health / login / register 外，没有任何路由能匿名访问**",
          not unprotected, "；".join(unprotected))

    # 反向断言：白名单里的路径**必须**确实没挂鉴权。
    # 挂上了会导致死锁——拿令牌前先要令牌。
    for path in ("/login", "/register", "/health"):
        matched = [route for route in routes if route.path == path]
        check(f"{path} 保持公开（拿令牌的前提 / 容器探针要用）",
              bool(matched) and not any(
                  auth_api.require_current_user in _dependency_calls(route)
                  for route in matched
              ),
              f"matched={len(matched)}")

    # 抽样确认业务接口确实挂了（防止上面那条被"路由枚举不全"架空）
    for path in ("/chat", "/chat/stream", "/knowledge-bases/{kb_id}/search",
                 "/documents/{doc_id}", "/me", "/conversations"):
        matched = [route for route in routes if route.path == path]
        check(f"业务接口 {path} 已挂鉴权依赖",
              bool(matched) and all(
                  auth_api.require_current_user in _dependency_calls(route)
                  for route in matched
              ),
              f"matched={len(matched)}")

    check("bearer_token 对 None / 非 Bearer / 空凭据都返回 None",
          auth_api.bearer_token(None) is None
          and auth_api.bearer_token("") is None
          and auth_api.bearer_token("Basic abc") is None
          and auth_api.bearer_token("Bearer    ") is None)
    check("bearer_token 正常解析且去掉空白",
          auth_api.bearer_token("  Bearer abc.def  ") == "abc.def")


# ---------------------------------------------------------------------------
# 7. 配置守门
# ---------------------------------------------------------------------------


def _problems(settings: Settings) -> list[str]:
    """跑一遍启动期校验，返回问题清单（通过则空列表）。"""
    try:
        settings.validate_runtime()
    except ConfigurationError as exc:
        return list((exc.detail or {}).get("problems", []))
    return []


def check_config_guard() -> None:
    section("7. 配置守门（缺失/弱参数必须在启动期拦下）")

    check("空 AUTH_SECRET_KEY → 报缺失",
          any("缺少 AUTH_SECRET_KEY" in item for item in _problems(Settings(auth_secret_key=""))))
    check("过短的 AUTH_SECRET_KEY → 报太短",
          any("太短" in item for item in _problems(Settings(auth_secret_key="short-key"))))
    check("迭代次数过低 → 拦下",
          any("AUTH_PBKDF2_ITERATIONS" in item
              for item in _problems(Settings(auth_pbkdf2_iterations=1000))))
    check("TTL 非正 → 拦下",
          any("AUTH_TOKEN_TTL_MINUTES" in item
              for item in _problems(Settings(auth_token_ttl_minutes=0))))
    check("APP_ENV=prod 却开放注册 → 拦下",
          any("开放注册" in item
              for item in _problems(Settings(app_env="prod", auth_allow_registration=True))))
    check("合法配置不产生认证类问题",
          not [item for item in _problems(Settings()) if "AUTH_" in item],
          "；".join(item for item in _problems(Settings()) if "AUTH_" in item))

    # 本机 .env 必须能过认证这一关（其余项是否通过取决于本机模型目录，
    # 故这里只筛认证相关的问题，避免脚本对"模型还没下好"误报）
    local_problems = [item for item in _problems(get_settings()) if "AUTH_" in item]
    check("本机 .env 的认证配置通过校验", not local_problems, "；".join(local_problems))


# ---------------------------------------------------------------------------
# 8. DTO 契约与服务常量一致
# ---------------------------------------------------------------------------


def check_dto_contract() -> None:
    section("8. DTO 契约")

    check("DTO 与服务的口令下限一致（防两处漂移）",
          auth_schema.PASSWORD_MIN == MIN_PASSWORD_LENGTH,
          f"{auth_schema.PASSWORD_MIN} vs {MIN_PASSWORD_LENGTH}")
    check("登录请求不校验口令长度（老账号也要能登进来）",
          auth_schema.LoginRequest(username="old-user", password="1") is not None)
    check("注册请求要求口令达到下限",
          _raises(PydanticValidationError,
                  lambda: auth_schema.RegisterRequest(username="new-user", password="1")))
    check(r"中文登录名可用（\w 是 Unicode 语义）",
          auth_schema.RegisterRequest(username="张三", password="good-pass-1").username == "张三")
    check("两字中文名不被长度下限挡住（下限按字符数，2 起步）",
          auth_schema.USERNAME_MIN == 2)
    check("单字符登录名仍被拒（下限只为挡无意义输入）",
          _raises(PydanticValidationError,
                  lambda: auth_schema.RegisterRequest(username="a", password="good-pass-1")))
    check("登录名两端空白被去掉（before 阶段，长度判定在去空白之后）",
          auth_schema.RegisterRequest(username="  zhao  ", password="good-pass-1")
          .username == "zhao")
    check("口令**不**被 trim（擅自 trim 等于悄悄改掉用户的密钥）",
          auth_schema.RegisterRequest(username="zhao2", password="  spaced  ")
          .password == "  spaced  ")
    check("UserOut / TokenOut 里不存在 password 相关字段",
          not [name for name in auth_schema.UserOut.model_fields if "password" in name]
          and not [name for name in auth_schema.TokenOut.model_fields if "password" in name])


def _raises(exc_type: type[Exception], action) -> bool:
    """action 是否抛出了指定异常。"""
    try:
        action()
    except exc_type:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    check_password_hashing()
    check_token()
    check_service_with_repository()
    check_inactive_and_deleted()
    check_http_routes()
    check_auth_enforcement()
    check_config_guard()
    check_dto_contract()

    print("\n" + "=" * 60)
    if _failures:
        print(f"未通过 {len(_failures)}/{_checks} 项：")
        for item in _failures:
            print(f"  - {item}")
        return 1
    print(f"全部通过（{_checks} 项）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
