"""口令哈希与访问令牌：纯标准库实现，**不引入任何第三方依赖**。

为什么不用 passlib / bcrypt / PyJWT：
    本项目的登录接口**不做鉴权**——没有任何路由校验令牌（见 api/v1/auth.py）。
    为这点功能引入 bcrypt（要编译，Windows 上尤其容易失败）或 PyJWT，
    会与项目"依赖尽量不脏"的取向冲突，而收益只是省下下面这 60 行。
    PBKDF2-HMAC-SHA256 是 `hashlib` 内置实现，参数可存取、可自校验、跨平台。

口令哈希格式（**自描述**，参数跟着哈希走）：
    pbkdf2_sha256$<迭代次数>$<盐 base64url>$<摘要 base64url>

    把迭代次数写进哈希串，是为了让"调高 AUTH_PBKDF2_ITERATIONS"只影响新口令：
    校验时用**哈希串里记的**次数而不是配置里的当前值。否则一次参数调整
    等于把已注册的用户全部锁在门外（他们的哈希是用旧次数算的，怎么比都不对）。

令牌格式：
    <payload base64url>.<signature base64url>
    payload 是 JSON：{"sub": 用户ID, "username": 登录名, "iat": 签发时间, "exp": 过期时间}
    signature = HMAC-SHA256(secret_key, payload 段字符串)

    刻意**不做成 JWT**：不需要与外部系统互通，就不必为兼容性多背 header
    与算法协商这两块。结构上与 JWT 同构（点分两段 + 声明集），
    将来真要换 JWT，改动只落在本文件。

签名对**编码后的字符串**做，不对原始 JSON 做：
    JSON 的序列化结果不唯一（键序、空白、ensure_ascii），
    若对"重新序列化后的 payload"验签，就会出现"签名时和验签时序列化不一致
    导致合法令牌验不过"的偶发故障。签字符串则完全绕开这个坑。

分层说明：
    本模块是**无状态纯函数**，所以不做成 Port + 实现类（不像 Embedder/Reranker
    那样有"可替换的外部依赖"）。它依赖的只有标准库和领域异常。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
from datetime import datetime, timedelta, timezone

from app.core.exceptions import UnauthorizedError

logger = logging.getLogger(__name__)

# 哈希算法标识与参数
ALGORITHM = "pbkdf2_sha256"
DEFAULT_ITERATIONS = 210_000
_SALT_BYTES = 16
# 摘要长度不单列常量：pbkdf2_hmac 的 dklen 缺省等于所用摘要的长度（sha256 = 32 字节），
# 写死一个 32 只会与上面那行 hash 名重复，改算法时反而容易忘了同步。


# ---------------------------------------------------------------------------
# base64url（去掉 `=` 填充，避免出现在 URL / 请求头里还要转义）
# ---------------------------------------------------------------------------


def _b64encode(raw: bytes) -> str:
    """bytes → 无填充的 base64url 字符串。"""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(text: str) -> bytes:
    """无填充 base64url 字符串 → bytes。

    Raises:
        ValueError: 不是合法的 base64url（由调用方决定如何包装）
    """
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


# ---------------------------------------------------------------------------
# 口令哈希
# ---------------------------------------------------------------------------


def hash_password(password: str, iterations: int | None = None) -> str:
    """计算口令哈希（每次调用都用新的随机盐）。

    Args:
        password: 明文口令
        iterations: PBKDF2 迭代次数；None 时用 DEFAULT_ITERATIONS

    Returns:
        自描述哈希串，可直接入库。

    注意：
        同一个口令两次调用结果**必然不同**（盐随机）——这不是 bug，
        所以任何"哈希相等"的断言都是错的，只能断言 verify 的结果。
    """
    rounds = int(iterations or DEFAULT_ITERATIONS)
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return f"{ALGORITHM}${rounds}${_b64encode(salt)}${_b64encode(digest)}"


def verify_password(password: str, stored: str) -> bool:
    """校验口令是否与存储的哈希匹配。

    Args:
        password: 待校验的明文口令
        stored: hash_password 产出（或历史）的哈希串

    Returns:
        匹配为 True。**哈希串本身非法时也返回 False**，而不是抛异常：
        调用方（登录）只需知道"对不上"，把它变成一个 500 只会让攻击者
        通过状态码区分"格式错"与"口令错"。
    """
    try:
        algorithm, rounds_text, salt_text, digest_text = stored.split("$")
        if algorithm != ALGORITHM:
            raise ValueError(f"不支持的哈希算法：{algorithm}")
        salt = _b64decode(salt_text)
        expected = _b64decode(digest_text)
        rounds = int(rounds_text)
        if rounds <= 0:
            raise ValueError(f"迭代次数非法：{rounds}")
    except (ValueError, TypeError, AttributeError) as exc:
        logger.warning("口令哈希串无法解析，一律视为不匹配：%s", exc)
        return False

    # 迭代次数取**哈希串里的值**，不是当前配置值——理由见模块文档
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    # 用 compare_digest 而不是 ==：普通字符串比较会在第一个不同字节处返回，
    # 把"逐字节试探"的时间差暴露出去
    return hmac.compare_digest(actual, expected)


def dummy_verify(password: str, iterations: int | None = None) -> None:
    """对一条固定哈希做一次等价开销的校验，结果丢弃。

    用途：登录时**账号不存在**也要走一遍哈希计算。否则"账号不存在"会在
    1ms 内返回，而"账号存在但口令错"要 100ms——攻击者据此就能批量枚举
    出库里有哪些账号（时间侧信道），我们精心统一的那条错误消息就白费了。
    """
    hash_password(password, iterations)  # 盐随机，结果必然不匹配，直接丢弃


# ---------------------------------------------------------------------------
# 访问令牌
# ---------------------------------------------------------------------------


def create_access_token(
    *,
    user_id: str,
    username: str,
    secret_key: str,
    ttl_minutes: int,
    now: datetime | None = None,
) -> tuple[str, datetime]:
    """签发访问令牌。

    Args:
        user_id: 主体（账号 ID）
        username: 登录名，仅用于排查时肉眼辨认，**不参与任何鉴权判断**
        secret_key: HMAC 密钥，来自 AUTH_SECRET_KEY
        ttl_minutes: 有效期（分钟）
        now: 签发时刻，默认当前 UTC 时间（测试可注入）

    Returns:
        (令牌字符串, 过期时刻)

    Raises:
        ValueError: secret_key 为空——空密钥签出来的令牌人人可伪造，
            与其签一个假的，不如当场失败。
    """
    if not secret_key:
        raise ValueError("AUTH_SECRET_KEY 为空，无法签发令牌：空密钥签名可被任意伪造")

    issued = now or datetime.now(timezone.utc)
    expires = issued + timedelta(minutes=int(ttl_minutes))
    payload = {
        "sub": user_id,
        "username": username,
        "iat": int(issued.timestamp()),
        "exp": int(expires.timestamp()),
    }
    body = _b64encode(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    signature = _b64encode(_sign(body, secret_key))
    return f"{body}.{signature}", expires


def decode_access_token(
    token: str,
    *,
    secret_key: str,
    now: datetime | None = None,
) -> dict:
    """校验令牌并返回声明集。

    Args:
        token: create_access_token 签发的令牌
        secret_key: HMAC 密钥
        now: 校验时刻，默认当前 UTC 时间（测试用来构造"已过期"）

    Returns:
        payload 字典（含 sub / username / iat / exp）

    Raises:
        UnauthorizedError: 格式非法、签名不符、已过期、主体缺失；
            失败原因放在 detail["reason"] 里供排查，消息本身统一（不向调用方
            泄露"是签名错还是过期"的细节差别没有意义，客户端动作都是重新登录）。
    """
    if not secret_key:
        raise UnauthorizedError("服务端未配置 AUTH_SECRET_KEY，无法校验令牌")

    body, _, signature = token.partition(".")
    if not body or not signature:
        raise UnauthorizedError("令牌格式不正确", detail={"reason": "malformed"})

    # 先验签再解析：让不可信内容连被 json 解析的机会都没有
    if not hmac.compare_digest(_sign(body, secret_key), _b64decode_quiet(signature)):
        raise UnauthorizedError("令牌签名不匹配", detail={"reason": "bad_signature"})

    try:
        payload = json.loads(_b64decode(body).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise UnauthorizedError("令牌内容无法解析", detail={"reason": "malformed"}) from exc

    if not isinstance(payload, dict) or not payload.get("sub"):
        raise UnauthorizedError("令牌缺少主体信息", detail={"reason": "missing_subject"})

    # 过期判定要在验签之后：先判断过期的话，攻击者可以拿任意伪造令牌
    # 去撞 exp 分支，把"签名对不对"这个真正的问题掩盖掉
    expires = payload.get("exp")
    if not isinstance(expires, int):
        raise UnauthorizedError("令牌缺少有效期", detail={"reason": "missing_exp"})

    current = now or datetime.now(timezone.utc)
    if current.timestamp() >= expires:
        raise UnauthorizedError(
            "登录已过期，请重新登录",
            detail={"reason": "expired", "expired_at": expires},
        )
    return payload


def _sign(body: str, secret_key: str) -> bytes:
    """对 payload 段字符串做 HMAC-SHA256。"""
    return hmac.new(
        secret_key.encode("utf-8"), body.encode("ascii"), hashlib.sha256
    ).digest()


def _b64decode_quiet(text: str) -> bytes:
    """解码失败时返回空字节串（只为让 compare_digest 拿到等长输入）。"""
    try:
        return _b64decode(text)
    except (ValueError, TypeError):
        return b""
