"""开发用热重载启动器（Windows 上替代 `uvicorn --reload`）。

为什么不用 `uvicorn --reload`
---------------------------
uvicorn 0.54 的 `basereload.restart()` 在 Windows 上**不给子进程发信号的替代品**：

```python
if sys.platform == "win32":
    self.is_restarting = True
    os.kill(self.process.pid, signal.CTRL_C_EVENT)   # ← 按「进程组」投递
    sys.stdout.write(" "); sys.stdout.flush()        # ← 卡住前那个多余的空格
else:
    self.process.terminate()
self.process.join()                                   # ← 永久阻塞点
```

`CTRL_C_EVENT` 是**控制台事件**，`os.kill` 在 Windows 上把它交给
`GenerateConsoleCtrlEvent`，需要目标进程组**挂着真实控制台**。一旦 uvicorn 是从
PyCharm 的运行窗、Git Bash 管道、`> log` 重定向或后台任务里启动的，事件就投递不到，
子进程照常跑；reloader 卡在 `join()` 上，**新 worker 永不启动，旧 worker 继续用旧代码
对外服务**。表现极具欺骗性：

- 端口有响应、日志正常、`/health` 200 —— 看起来一切正常；
- 改代码不生效，`touch` 也不生效；
- 日志里只有一行 `WatchFiles detected changes ... Reloading...`，后面什么都没有。

本机实测踩过三次（README 里的「8000 端口是旧构建」就是它）。所以这里改用
**`terminate()`（Windows 上是 `TerminateProcess`，必杀）**，不依赖控制台事件。

代价与取舍
---------
`TerminateProcess` 不给子进程做 lifespan shutdown 的机会：正在跑的上传/问答会被
直接打断。开发期可接受——SQLite 是 WAL 模式，硬杀不会损坏；Milvus 是服务端，客户端
被杀它自己会回收。要「优雅重启」得能发信号，那还是回到控制台的老问题。

用法
----
    .\\.venv\\Scripts\\python.exe scripts\\dev.py                 # 8000 端口
    .\\.venv\\Scripts\\python.exe scripts\\dev.py --port 8010
    .\\.venv\\Scripts\\python.exe scripts\\dev.py --no-watch      # 只起服务，不热重载

Ctrl+C 停止（子进程会一起收到，走正常 shutdown）。
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

import watchfiles

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WATCH = ("app", ".env")


def _ensure_no_proxy() -> None:
    """本机回环必须绕开全局代理，否则 pymilvus 的 gRPC 会被拦成 502。

    应用里 `vectorstore` 也会补一次，但在 CLI 里补更早——uvicorn 之前的进程
    创建阶段就要用（子进程继承环境变量）。
    """
    current = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    wanted = ["127.0.0.1", "localhost", "::1"]
    parts = [item.strip() for item in current.split(",") if item.strip()]
    missing = [item for item in wanted if item not in parts]
    if missing:
        merged = ",".join(parts + missing)
        os.environ["NO_PROXY"] = merged
        os.environ["no_proxy"] = merged


def _port_free(host: str, port: int) -> bool:
    """能不能绑上这个端口。

    Windows 上「端口被占」只有两种后果：绑定报 WinError 10048，或者更糟——
    某个孤儿 worker 继承了套接字却不 accept，连接能建立、永远不回话。启动前
    探一次，把后一种情况挡在门外，别让它表现成「服务起来了但请求超时」。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _start(host: str, port: int) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        str(port),
        # 不要 --reload：热重载由本脚本负责，见模块文档。
    ]
    print(f"[dev] 启动 worker → {' '.join(cmd[1:])}", flush=True)
    return subprocess.Popen(cmd, cwd=str(ROOT), env=os.environ.copy())


def _stop(proc: subprocess.Popen, grace: float = 3.0) -> None:
    """先 terminate，超时才 kill。

    Windows 上两者都是 TerminateProcess，`grace` 只是给 POSIX 留的语义；
    仍保留等待是因为子进程可能已经自己退了，直接 kill 会抛异常。
    """
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        print("[dev] worker 未在 %.0fs 内退出，强制 kill" % grace, flush=True)
        proc.kill()
        proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="带确定性热重载的开发服务器")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--watch",
        nargs="*",
        default=list(DEFAULT_WATCH),
        help="监听路径（相对项目根），默认 app 与 .env",
    )
    parser.add_argument("--no-watch", action="store_true", help="不监听文件变化")
    args = parser.parse_args()

    _ensure_no_proxy()

    if not _port_free(args.host, args.port):
        print(
            f"[dev] {args.host}:{args.port} 已被占用。\n"
            f"      若确认没有服务在跑，多半是上次 --reload 卡死后留下的孤儿 worker\n"
            f"      （套接字被继承、不 accept，请求会一直挂住）。查并结束：\n"
            f"        netstat -ano | findstr :{args.port}\n"
            f"        taskkill /PID <pid> /F",
            file=sys.stderr,
            flush=True,
        )
        return 1

    proc = _start(args.host, args.port)

    if args.no_watch:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            _stop(proc)
            return 0

    watched = [str(ROOT / item) for item in args.watch]
    print(f"[dev] 监听 {', '.join(args.watch)}；Ctrl+C 退出", flush=True)

    # 子进程自己崩了（语法错误、配置校验失败）只报一次，不然会每 0.5s 刷屏。
    reported_exit = False
    try:
        # yield_on_timeout + 较短的 step：没有文件变化时也定期回来看看 worker 死没死。
        # 这正是 `--reload` 的缺口——它只等控制台事件，子进程死了/卡了它不知道。
        for changes in watchfiles.watch(
            *watched,
            stop_event=None,
            yield_on_timeout=True,
            step=500,
            debounce=800,
        ):
            if not changes:
                if proc.poll() is not None and not reported_exit:
                    print(
                        f"[dev] worker 已退出（退出码 {proc.returncode}）。"
                        "修好后保存即自动重启，不需要重开终端。",
                        flush=True,
                    )
                    reported_exit = True
                continue
            names = ", ".join(sorted({Path(path).name for _, path in changes}))
            print(f"[dev] 变更：{names} → 重启", flush=True)
            _stop(proc)
            proc = _start(args.host, args.port)
            reported_exit = False
    except KeyboardInterrupt:
        print("\n[dev] 收到 Ctrl+C，停止", flush=True)
    finally:
        _stop(proc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
