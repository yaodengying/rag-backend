# -*- coding: utf-8 -*-
"""批量重建向量：把处于 `failed` 的文档逐个 reprocess。

什么时候用：
    - **备份恢复之后**（`scripts/backup.py restore` 会把 ready 置为 failed，
      因为 Milvus 里的向量不在备份里）；
    - 换了 embedding 模型、`init_milvus.py --recreate` 之后；
    - 一批文档因为 Milvus 短暂不可用而失败（`error` 里是 `retrieval_error`）。

为什么不支持"重建 ready 文档"：
    reprocess 的准入就是 `failed`。对 ready 文档重跑等于"重建索引"：
    要先清空再写，清空到写完之间检索不到它。那是另一个语义，
    需要显式删除再上传（或者将来的 `/reindex` 接口），不在本脚本里偷偷做。

用法：
    .venv\\Scripts\\python.exe scripts/rebuild_vectors.py
    .venv\\Scripts\\python.exe scripts/rebuild_vectors.py --kb <kb_id>
    .venv\\Scripts\\python.exe scripts/rebuild_vectors.py --dry-run

依赖：**必须先把后端服务起起来**（脚本走 HTTP，不自己加载 BGE 权重）。
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


def _api(base: str) -> str:
    return base.rstrip("/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="批量重建 failed 文档的向量")
    parser.add_argument(
        "--api",
        default="http://127.0.0.1:8000/api/v1",
        help="后端 API 根路径",
    )
    parser.add_argument("--kb", default="", help="只处理某个知识库（默认全部）")
    parser.add_argument("--dry-run", action="store_true", help="只列出，不执行")
    parser.add_argument("--timeout", type=float, default=600.0, help="单个文档的超时（秒）")
    args = parser.parse_args(argv)

    base = _api(args.api)
    client = httpx.Client(timeout=args.timeout)

    try:
        client.get(f"{base}/health").raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"无法连接后端 {base}：{exc}\n请先启动服务：.venv\\Scripts\\python.exe -m uvicorn app.main:app")
        return 2

    if args.kb:
        kb_ids = [args.kb]
    else:
        kb_ids = [item["id"] for item in client.get(f"{base}/knowledge-bases").json()]

    pending: list[tuple[str, str, str]] = []  # (kb_id, doc_id, filename)
    for kb_id in kb_ids:
        documents = client.get(f"{base}/knowledge-bases/{kb_id}/documents").json()
        for doc in documents:
            if doc["status"] == "failed":
                pending.append((kb_id, doc["id"], doc["filename"]))

    if not pending:
        print("没有处于 failed 的文档，无需重建")
        return 0

    print(f"待重建 {len(pending)} 份：")
    for _kb, _doc, filename in pending:
        print(f"  - {filename}")
    if args.dry_run:
        print("\n--dry-run：未执行")
        return 0

    ok = failed = 0
    started = time.time()
    for index, (_kb, doc_id, filename) in enumerate(pending, start=1):
        try:
            response = client.post(f"{base}/documents/{doc_id}/reprocess")
        except httpx.HTTPError as exc:
            failed += 1
            print(f"[{index}/{len(pending)}] {filename}  请求失败：{exc}")
            continue

        if response.status_code != 200:
            failed += 1
            detail = response.json().get("message") or response.text
            print(f"[{index}/{len(pending)}] {filename}  HTTP {response.status_code}：{detail}")
            continue

        body = response.json()
        if body.get("status") == "ready":
            ok += 1
            print(f"[{index}/{len(pending)}] {filename}  ready（{body.get('chunk_count')} 块）")
        else:
            failed += 1
            # 失败原因里带 `[code]`，可直接判断值不值得再试
            print(f"[{index}/{len(pending)}] {filename}  failed：{body.get('error')}")

    print(f"\n完成：成功 {ok} / 失败 {failed}，耗时 {time.time() - started:.1f}s")
    if failed:
        print(
            "失败项看 error 前缀：\n"
            "  [corrupted_document] / [encrypted_document] / [empty_document] → 文件本身的问题，重试无用\n"
            "  [embedding_error] / [upstream_timeout] / [retrieval_error] → 依赖问题，修好后可再跑一次"
        )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
