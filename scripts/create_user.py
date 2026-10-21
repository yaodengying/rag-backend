# -*- coding: utf-8 -*-
"""账号运维脚本：建号、改口令、列账号。

为什么需要它（而不是只用 POST /auth/register）：
    注册接口一关（AUTH_ALLOW_REGISTRATION=false），就没有任何途径建号了；
    而且"注册"这个动作本身不该长期对外开着——本项目的注册默认打开只是为了
    本地自测方便。生产路径是：管理员在服务器上跑这条命令。

用法（cwd 无所谓，脚本自己会切到项目根，因为 .env 按相对路径加载）：
    # 建号（口令交互输入，不回显）
    .venv\\Scripts\\python.exe scripts\\create_user.py 张三

    # 建号（口令写在命令行上——**会留在 shell 历史里**，仅限本地调试）
    .venv\\Scripts\\python.exe scripts\\create_user.py zhang --password 'good-pass-1' --display-name 张三

    # 改口令（同一把钥匙开两扇门，不引入第二个脚本）
    .venv\\Scripts\\python.exe scripts\\create_user.py zhang --reset-password

    # 看有哪些账号
    .venv\\Scripts\\python.exe scripts\\create_user.py --list

退出码：0 成功；1 失败（重名、口令不合规、账号不存在等）。
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from app.core.container import get_auth_service, get_database  # noqa: E402
from app.core.exceptions import AppError  # noqa: E402
from app.services.auth_service import MIN_PASSWORD_LENGTH  # noqa: E402


def _read_password(confirm: bool = True) -> str:
    """交互读取口令（不回显）。

    做二次确认：新设口令时打错一个字，后果是"这个账号再用不了、
    而且谁也不知道正确口令是什么"。改口令不需要确认（弄错了再改一次即可），
    所以 confirm 可控。
    """
    password = getpass.getpass(f"请输入口令（至少 {MIN_PASSWORD_LENGTH} 位）：")
    if confirm:
        again = getpass.getpass("请再输入一次：")
        if password != again:
            print("两次输入不一致。", file=sys.stderr)
            raise SystemExit(1)
    return password


def _print_users(users) -> None:
    """打印账号列表。"""
    if not users:
        print("（还没有任何账号）")
        return
    print(f"{'登录名':<20}{'展示名':<16}{'状态':<8}{'ID':<38}创建时间")
    for user in users:
        state = "启用" if user.is_active else "停用"
        created = user.created_at.strftime("%Y-%m-%d %H:%M") if user.created_at else "-"
        print(f"{user.username:<20}{user.display_name:<16}{state:<8}{user.id:<38}{created}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="账号运维：建号 / 改口令 / 列账号（不经 HTTP，直接操作 SQLite）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("username", nargs="?", help="登录名（建号或改口令时必填）")
    parser.add_argument("--password", help="口令；省略则交互输入（推荐，命令行会留在 shell 历史里）")
    parser.add_argument("--display-name", default="", help="展示名，缺省显示登录名")
    parser.add_argument("--reset-password", action="store_true", help="重置已有账号的口令")
    parser.add_argument("--list", action="store_true", help="列出全部账号")
    args = parser.parse_args()

    # 建表（幂等）。服务启动时会做同样的事，但脚本可能先于服务跑第一次，
    # 那时库里连 users 表都没有——不能假设"服务已经起过"
    get_database().init_schema()
    service = get_auth_service()

    try:
        if args.list:
            _print_users(service.list_users())
            return 0

        if not args.username:
            parser.print_help()
            return 2

        if args.reset_password:
            user = service.get_user_by_username(args.username)
            if user is None:
                print(f"账号不存在：{args.username}", file=sys.stderr)
                return 1
            password = args.password or _read_password(confirm=False)
            service.reset_password(user.id, password)
            print(f"已重置口令：{user.username}（id={user.id}）")
            return 0

        password = args.password or _read_password(confirm=True)
        # 走 create_user 而不是 register：注册开关（AUTH_ALLOW_REGISTRATION）
        # 管的是"陌生人能否自助建号"，管不到管理员通道——否则关掉注册之后
        # 连这条命令也建不了号，就真的没有别的路了
        user = service.create_user(
            username=args.username,
            password=password,
            display_name=args.display_name,
        )
        print(f"已创建账号：{user.username}（id={user.id}）")
        print("提示：口令哈希已入库，明文不留存——忘记口令只能用 --reset-password 重设。")
        return 0

    except AppError as exc:
        print(f"失败（{exc.code}）：{exc.message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
