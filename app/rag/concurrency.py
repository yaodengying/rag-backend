"""入库链路的并发控制：按文档的互斥锁 + 全局并发闸门。

为什么需要（两条都是实测过的真实故障，不是假设）：

    **① 同一文档被并发处理**。`reprocess` 的状态检查与"置为 processing"
    是两次独立读写，两个请求可以同时通过检查；随后两份向量被写进 Milvus
    （chunk_id 不同、内容相同），检索时互相挤占 top_k。
    加锁后第二次请求看到的就是 processing → 409。

    **② 多个大文件同时入库**。向量化是纯 CPU 的批量前向，一台机器上
    并行跑 N 份不会更快（核数固定），只会让每个请求的 P99 一起变差，
    并且 embedding 模型实例是单例——并发调用在部分实现上并不线程安全。
    Semaphore 把它们排成队，而不是轮流抢占。

**作用域必须说清楚**：这两个都是**进程内**的。uvicorn 起 4 个 worker
就有 4 份锁，跨进程的互斥需要 Redis 锁（本项目没做，也不建议在 SQLite
单文件库上起多 worker——那才是真正的并发瓶颈）。
因此部署建议写死：**入库服务单 worker**，横向扩展放到检索侧（只读）。
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class DocumentLockPool:
    """按 key 分配互斥锁，用完即回收。

    直接用 `dict[key] = asyncio.Lock()` 会随文档数无限增长（一万份文档
    就是一万个常驻锁）。这里用引用计数在最后一个持有者释放时删除条目。

    实现上不需要额外的 guard 锁：CPython 单事件循环里，两行之间没有
    `await` 就不会切换任务，`dict` 的读改写因此是原子的。
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._waiters: dict[str, int] = {}

    @asynccontextmanager
    async def acquire(self, key: str) -> AsyncIterator[None]:
        """进入临界区。同一 key 串行，不同 key 互不影响。"""
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        self._waiters[key] = self._waiters.get(key, 0) + 1
        try:
            async with lock:
                yield
        finally:
            remaining = self._waiters.get(key, 1) - 1
            if remaining <= 0:
                self._waiters.pop(key, None)
                self._locks.pop(key, None)
            else:
                self._waiters[key] = remaining

    @property
    def size(self) -> int:
        """当前活跃的锁数量（自检用：断言用完会回收）。"""
        return len(self._locks)


class ConcurrencyGate:
    """全局并发闸门（懒创建 Semaphore）。

    懒创建的原因：`container` 用 `lru_cache` 装配单例，构造时机可能是
    import 期——那里没有运行中的事件循环。Python 3.10+ 的 Semaphore
    虽然不再在构造时抓 loop，但延迟到首次使用最稳（也便于自检注入）。
    """

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError(f"并发上限必须 >= 1，当前 {limit}")
        self._limit = limit
        self._semaphore: asyncio.Semaphore | None = None
        self._peak = 0
        self._active = 0

    @property
    def limit(self) -> int:
        """允许同时进行的任务数。"""
        return self._limit

    @property
    def peak(self) -> int:
        """历史最大并发数（自检用：断言闸门真的生效）。"""
        return self._peak

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """占用一个并发名额，退出时归还。"""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._limit)
        async with self._semaphore:
            self._active += 1
            self._peak = max(self._peak, self._active)
            try:
                yield
            finally:
                self._active -= 1
