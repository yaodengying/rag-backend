# -*- coding: utf-8 -*-
"""数据备份 / 恢复：SQLite 元数据 + uploads 原始文件，一体打包并可校验。

**备份什么、不备什么**（这个划分比脚本本身重要）：

| 数据                | 备份 | 理由                                                     |
|--------------------|------|----------------------------------------------------------|
| `data/rag.db`      | 是   | 知识库/文档/账号，唯一的权威副本                          |
| `data/uploads/**`  | 是   | 原始文件，**向量重建的唯一原料**                          |
| Milvus 向量         | 否   | 派生数据。可以从 uploads 重新算出来，备份它等于备份缓存     |
| Redis 会话          | 否   | TTL 数据，过一周自己就过期了，恢复它没有意义               |

**为什么用 `VACUUM INTO` 而不是复制 `.db` 文件**：
    直接 `cp data/rag.db backup.db` 在 WAL 模式下会漏掉还没 checkpoint 的
    事务，而且万一服务正在写，复制出来的文件可能是撕裂的（前半是新页、
    后半是旧页），打开时报 `database disk image is malformed`。
    `VACUUM INTO` 走一次读事务，产出的是**一致快照**，顺带整理了碎片，
    并且不需要停服务。

**为什么不备份 Milvus**：
    Milvus 的 collection 里只有 chunk_id / 向量 / 少量字段，正文在
    uploads 里。恢复时把文档置为 failed 再 reprocess，就是从原始文件
    重新解析、切分、向量化——产物与备份时逐位一致（同模型、同参数）。
    反过来"备份 Milvus 数据目录"要求版本、索引参数、段文件全部对得上，
    是给自己造一个只能原样还原的脆壳。

用法：
    # 备份（默认写到 data/backups/backup-<时间戳>/，保留最近 7 份）
    .venv\\Scripts\\python.exe scripts\\backup.py backup
    .venv\\Scripts\\python.exe scripts\\backup.py backup --out D:\\rag-backups --keep 14

    # 查看 / 校验
    .venv\\Scripts\\python.exe scripts\\backup.py list
    .venv\\Scripts\\python.exe scripts\\backup.py verify data\\backups\\backup-20261017-193000

    # 恢复（破坏性：会覆盖当前的 db 与 uploads，必须加 --yes）
    .venv\\Scripts\\python.exe scripts\\backup.py restore data\\backups\\backup-20261017-193000 --yes

恢复后文档会全部处于 `failed`（error 提示"待重建向量"），
启动服务后逐个调 `POST /api/v1/documents/{id}/reprocess` 即可重建。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

MANIFEST_NAME = "manifest.json"
DB_NAME = "rag.db"
UPLOADS_NAME = "uploads.tar.gz"
_FORMAT_VERSION = 1

# 备份目录的命名前缀。`--keep` 只清理匹配它的目录——
# 绝不能按 mtime 删一批“看起来像备份”的东西。
_BACKUP_PREFIX = "backup-"
_PRE_RESTORE_PREFIX = "pre-restore-"


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while piece := handle.read(chunk):
            digest.update(piece)
    return digest.hexdigest()


def _human(num: int) -> str:
    if num >= 1024 * 1024 * 1024:
        return f"{num / 1024 / 1024 / 1024:.2f} GB"
    if num >= 1024 * 1024:
        return f"{num / 1024 / 1024:.1f} MB"
    if num >= 1024:
        return f"{num / 1024:.1f} KB"
    return f"{num} B"


def _paths() -> tuple[Path, Path, str]:
    """从项目配置读 db 路径、uploads 目录、collection 名。"""
    from app.core.config import get_settings

    settings = get_settings()
    db = Path(settings.sqlite_path).expanduser().resolve()
    uploads = Path(settings.upload_dir).expanduser().resolve()
    return db, uploads, settings.milvus_collection


def _table_counts(db_path: Path) -> dict[str, int]:
    """各表行数。只读打开（URI mode=ro），绝不顺手改库。"""
    counts: dict[str, int] = {}
    try:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        return counts
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            counts[table] = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    finally:
        conn.close()
    return counts


def _iter_documents(db_path: Path) -> list[tuple[str, str, str, str]]:
    """(kb_id, doc_id, filename, status)。打包 uploads 时按它精确取文件。"""
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        return [
            (row[0], row[1], row[2], row[3])
            for row in conn.execute("SELECT kb_id, id, filename, status FROM documents")
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 备份
# ---------------------------------------------------------------------------


def do_backup(out_dir: Path, keep: int = 7, label: str = "") -> Path:
    """生成一份备份并返回目录。

    Args:
        out_dir: 备份根目录
        keep: 保留份数（含本次），<=0 表示不清理
        label: 目录名后缀（如 `pre-restore`），为空则用时间戳
    """
    db_path, uploads_dir, collection = _paths()
    if not db_path.exists():
        raise SystemExit(f"元数据库不存在：{db_path}（还没有任何数据可备份）")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # pre-restore 有自己的前缀（不参与 --keep 计数），别再拼一次 label，
    # 否则会出现 `pre-restore-20261017-192810-pre-restore` 这种名字。
    if label == "pre-restore":
        name = f"{_PRE_RESTORE_PREFIX}{stamp}"
    else:
        name = f"{_BACKUP_PREFIX}{stamp}" + (f"-{label}" if label else "")
    target = out_dir / name
    target.mkdir(parents=True, exist_ok=False)

    started = time.time()
    print(f"备份到 {target}")

    # 1) 元数据：VACUUM INTO 产出一致快照（见模块 docstring）
    snapshot = target / DB_NAME
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("VACUUM INTO ?", (str(snapshot),))
    finally:
        conn.close()
    counts = _table_counts(snapshot)
    print(f"  元数据  {_human(snapshot.stat().st_size)}  表行数 {counts}")

    # 2) uploads：只打**元数据里登记过**的文件。
    #    整目录打包会把上次失败的半截文件、手工丢进去的杂物一起带上，
    #    恢复时它们既不属于任何文档、也没有元数据对应。
    documents = _iter_documents(snapshot)
    wanted = {
        (uploads_dir / kb_id / f"{doc_id}_{filename}").resolve()
        for kb_id, doc_id, filename, _status in documents
    }
    packed = 0
    missing: list[str] = []
    with tarfile.open(target / UPLOADS_NAME, "w:gz") as archive:
        for path in sorted(wanted):
            if not path.exists():
                missing.append(str(path))
                continue
            # arcname 用 uploads/ 作为根，恢复时直接解到 upload_dir 的父级
            archive.add(path, arcname=f"uploads/{path.parent.name}/{path.name}")
            packed += 1

    if missing:
        print(f"  [warn] {len(missing)} 个登记过的原始文件已丢失（不影响备份，但无法重建这些文档）")
        for path in missing[:5]:
            print(f"         - {path}")

    archive_bytes = (target / UPLOADS_NAME).stat().st_size
    print(f"  原始文件 {packed} 个（元数据登记 {len(documents)} 条）→ {_human(archive_bytes)}")

    # 3) 清单：校验和 + 行数 + 元信息。恢复前必看。
    manifest = {
        "format_version": _FORMAT_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "app_version": _app_version(),
        "collection": collection,
        "sqlite": {
            "file": DB_NAME,
            "bytes": snapshot.stat().st_size,
            "sha256": _sha256(snapshot),
            "tables": counts,
        },
        "uploads": {
            "file": UPLOADS_NAME,
            "bytes": archive_bytes,
            "sha256": _sha256(target / UPLOADS_NAME),
            "files": packed,
            "missing": len(missing),
        },
        "documents": {
            "total": len(documents),
            "by_status": _count_by(documents),
        },
        "elapsed_s": round(time.time() - started, 2),
        "notes": "向量不在备份内：恢复后需 reprocess 重建（见 docs/backup.md）",
    }
    (target / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  清单    {MANIFEST_NAME}（{manifest['elapsed_s']}s）")

    _prune(out_dir, keep)
    return target


def _count_by(documents: list[tuple[str, str, str, str]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for *_rest, status in documents:
        result[status] = result.get(status, 0) + 1
    return result


def _app_version() -> str:
    try:
        from app import __version__

        return __version__
    except Exception:  # noqa: BLE001
        return "unknown"


def _prune(out_dir: Path, keep: int) -> None:
    """只保留最近 keep 份**普通备份**。

    pre-restore 目录不参与计数：它是一次破坏性操作前的安全网，
    被自动清掉就失去意义了（由用户自己决定何时删）。
    """
    if keep <= 0:
        return
    candidates = sorted(
        (p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith(_BACKUP_PREFIX)),
        key=lambda p: p.name,
        reverse=True,
    )
    for stale in candidates[keep:]:
        shutil.rmtree(stale, ignore_errors=True)
        print(f"  已清理旧备份 {stale.name}")


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------


def do_verify(backup_dir: Path) -> bool:
    """校验清单与两个产物的 sha256；不通过返回 False。"""
    manifest_path = backup_dir / MANIFEST_NAME
    if not manifest_path.exists():
        print(f"[FAIL] 缺少 {MANIFEST_NAME}：{backup_dir}")
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != _FORMAT_VERSION:
        print(f"[FAIL] 格式版本不匹配：{manifest.get('format_version')} != {_FORMAT_VERSION}")
        return False

    ok = True
    for key in ("sqlite", "uploads"):
        entry = manifest.get(key, {})
        path = backup_dir / entry.get("file", "")
        if not path.exists():
            print(f"[FAIL] 缺少 {path.name}")
            ok = False
            continue
        size = path.stat().st_size
        if size != entry.get("bytes"):
            print(f"[FAIL] {path.name} 大小不符：{size} != {entry.get('bytes')}")
            ok = False
        actual = _sha256(path)
        if actual != entry.get("sha256"):
            print(f"[FAIL] {path.name} 校验和不符（文件被改动或复制损坏）")
            ok = False
        else:
            print(f"[ok]   {path.name}  {_human(size)}  sha256={actual[:16]}…")

    if ok:
        print(
            f"[ok]   备份完整：{manifest['created_at']}  "
            f"文档 {manifest['documents']['total']} 条 {manifest['documents']['by_status']}  "
            f"原始文件 {manifest['uploads']['files']} 个"
        )
    return ok


def do_list(out_dir: Path) -> None:
    if not out_dir.exists():
        print(f"备份目录不存在：{out_dir}")
        return
    rows = sorted((p for p in out_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not rows:
        print(f"{out_dir} 下没有备份")
        return
    print(f"{'目录':<44}{'大小':>10}  文档 / 文件")
    for path in rows:
        manifest_path = path / MANIFEST_NAME
        if not manifest_path.exists():
            print(f"{path.name:<44}{'-':>10}  缺少清单")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"{path.name:<44}{'-':>10}  清单损坏")
            continue
        total_bytes = sum(
            (path / entry["file"]).stat().st_size
            for entry in (manifest.get("sqlite"), manifest.get("uploads"))
            if entry and (path / entry["file"]).exists()
        )
        print(
            f"{path.name:<44}{_human(total_bytes):>10}  "
            f"{manifest['documents']['total']} / {manifest['uploads']['files']}"
        )


# ---------------------------------------------------------------------------
# 恢复
# ---------------------------------------------------------------------------


def do_restore(backup_dir: Path, assume_yes: bool, reset_status: bool) -> int:
    """把备份还原到当前配置指向的 db 与 uploads 目录。

    **这是破坏性操作**，所以做三件保护：
      ① 先校验 sha256（坏备份不该覆盖好数据）；
      ② 先把当前数据整份备份到 `pre-restore-*`（后悔药）；
      ③ 覆盖 uploads 时用"改名挪走"而不是删除（原目录还在磁盘上）。
    """
    backup_dir = backup_dir.resolve()
    if not do_verify(backup_dir):
        print("\n校验未通过，拒绝恢复（先修好备份，别拿坏数据覆盖好数据）")
        return 1

    db_path, uploads_dir, _collection = _paths()
    manifest = json.loads((backup_dir / MANIFEST_NAME).read_text(encoding="utf-8"))

    print("\n将要覆盖：")
    print(f"  元数据库  {db_path}")
    print(f"  原始文件  {uploads_dir}")
    print(f"备份来源  {manifest['created_at']}  文档 {manifest['documents']['total']} 条")

    if not assume_yes:
        print("\n这是破坏性操作。确认后重新执行并加 --yes")
        return 2

    # ② 先保一份当前的
    out_dir = backup_dir.parent
    try:
        safety = do_backup(out_dir, keep=0, label="pre-restore")
        print(f"\n已先把当前数据备份到 {safety}")
    except SystemExit as exc:
        # 当前没有 db（首次恢复）不是错误
        print(f"\n（当前数据无法备份，继续：{exc}）")

    # ③ 覆盖 db。**必须删掉 -wal/-shm**：旧 WAL 里可能还有新库没有的事务，
    #    留着它，SQLite 会把那些页当成"新库的一部分"，恢复出来的数据是混的。
    for suffix in ("-wal", "-shm"):
        stale = db_path.with_name(db_path.name + suffix)
        if stale.exists():
            stale.unlink()
    try:
        shutil.copy2(backup_dir / DB_NAME, db_path)
    except PermissionError:
        print(
            f"\n[FAIL] 无法写入 {db_path}：文件被占用。"
            "请先停掉后端服务（uvicorn）再恢复。"
        )
        return 1
    print(f"元数据已恢复（{_human((backup_dir / DB_NAME).stat().st_size)}）")

    # ③ uploads：挪走而不是删
    if uploads_dir.exists():
        moved = uploads_dir.with_name(f"{uploads_dir.name}.old-{datetime.now():%Y%m%d-%H%M%S}")
        uploads_dir.rename(moved)
        print(f"原 uploads 已挪到 {moved.name}（确认无误后可自行删除）")
    with tarfile.open(backup_dir / UPLOADS_NAME, "r:gz") as archive:
        _safe_extract(archive, uploads_dir.parent)
    print("原始文件已恢复")

    # ④ 元数据里的 ready 是"当时确实写过向量"，但现在 Milvus 里没有对应的行。
    #    置为 failed 让它们排队等 reprocess —— 这比"看着 ready、其实搜不到"诚实。
    if reset_status:
        conn = sqlite3.connect(str(db_path))
        try:
            changed = conn.execute(
                "UPDATE documents SET status='failed', chunk_count=0, "
                "error='备份恢复后待重建向量：启动服务后调用 reprocess' "
                "WHERE status='ready'"
            ).rowcount
            conn.commit()
        finally:
            conn.close()
        print(f"已把 {changed} 条 ready 文档置为 failed（等待重建向量）")

    print(
        "\n下一步（重建向量，见 docs/backup.md）：\n"
        "  1) 启动服务：.venv\\\\Scripts\\\\python.exe -m uvicorn app.main:app --port 8000\n"
        "  2) 逐条重试：POST /api/v1/documents/{doc_id}/reprocess\n"
        "     或批量：.venv\\\\Scripts\\\\python.exe scripts/rebuild_vectors.py\n"
    )
    return 0


def _safe_extract(archive: tarfile.TarFile, target: Path) -> None:
    """解包前挡住 `../` 与绝对路径（备份文件也可能被篡改）。"""
    target = target.resolve()
    for member in archive.getmembers():
        destination = (target / member.name).resolve()
        if target != destination and target not in destination.parents:
            raise SystemExit(f"备份内含越界路径，拒绝解包：{member.name}")
        if member.issym() or member.islnk():
            raise SystemExit(f"备份内含链接文件，拒绝解包：{member.name}")
    archive.extractall(target)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="RAG 后端数据备份/恢复（SQLite + uploads 原始文件）"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    default_out = PROJECT_ROOT / "data" / "backups"

    p_backup = sub.add_parser("backup", help="生成一份备份")
    p_backup.add_argument("--out", type=Path, default=default_out, help="备份根目录")
    p_backup.add_argument("--keep", type=int, default=7, help="保留最近 N 份（0 = 不清理）")

    p_list = sub.add_parser("list", help="列出已有备份")
    p_list.add_argument("--out", type=Path, default=default_out)

    p_verify = sub.add_parser("verify", help="校验一份备份的完整性")
    p_verify.add_argument("dir", type=Path)

    p_restore = sub.add_parser("restore", help="从备份恢复（破坏性）")
    p_restore.add_argument("dir", type=Path)
    p_restore.add_argument("--yes", action="store_true", help="确认覆盖当前数据")
    p_restore.add_argument(
        "--no-reset",
        action="store_true",
        help="不把 ready 文档置为 failed（默认置为 failed，等待重建向量）",
    )

    args = parser.parse_args(argv)

    if args.command == "backup":
        args.out.mkdir(parents=True, exist_ok=True)
        do_backup(args.out, keep=args.keep)
        return 0
    if args.command == "list":
        do_list(args.out)
        return 0
    if args.command == "verify":
        return 0 if do_verify(args.dir) else 1
    if args.command == "restore":
        return do_restore(args.dir, assume_yes=args.yes, reset_status=not args.no_reset)
    parser.error("未知命令")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
