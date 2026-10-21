# 数据备份与恢复

> 结论先行：**备份 `data/rag.db` + `data/uploads/` 就够了**，Milvus 与 Redis 都不备。
> 向量是派生物，能从原始文件重算；会话是 TTL 数据，过期即失效。
> 恢复后所有文档置为 `failed`，用 `scripts/rebuild_vectors.py` 重建即可。

## 1. 备什么、不备什么

| 数据 | 位置 | 备份 | 理由 |
|---|---|:---:|---|
| 知识库 / 文档 / 账号 | `data/rag.db` | ✅ | 唯一权威副本 |
| 原始文件 | `data/uploads/{kb_id}/{doc_id}_{filename}` | ✅ | **向量重建的唯一原料** |
| 向量 + 稀疏索引 | Milvus `rag_backend_chunks` | ❌ | 派生数据。同模型同参数重算，结果与备份时一致 |
| 多轮会话 | Redis `rag:conv:*` | ❌ | TTL 7 天，恢复一周前的对话没有价值 |

**为什么不用"复制 Milvus 数据目录"来备份向量**：那要求 Milvus 版本、索引类型、
段文件布局、`collection` 的 schema 全部对得上，换一台机器或升一次版本就还原不了。
而 `uploads + rag.db` 是纯文本与二进制文件，任何机器都能读。

代价要说清楚：**恢复后重建向量需要时间**（本机 CPU 编码约 3~6 秒/篇，
大文档更久），并且重建期间检索不到这些文档。这就是 RTO 的主要构成。

## 2. 一致性：为什么不是 `cp rag.db`

SQLite 开了 WAL。直接复制 `.db` 文件有两个坑：

1. 已提交但**尚未 checkpoint** 的事务还在 `-wal` 里，`cp` 出来的文件会丢掉这部分数据；
2. 复制过程中若有写入，文件可能撕裂（前半是新页、后半是旧页），打开时报
   `database disk image is malformed`——**备份看起来成功了，恢复时才发现是坏的**。

脚本用 `VACUUM INTO`：走一次读事务拿到一致快照，顺带整理碎片，**不需要停服务**。

```sql
VACUUM INTO 'backup.db'   -- 目标文件必须不存在
```

## 3. 一次完整备份

```powershell
# 默认写到 data/backups/backup-<时间戳>/，保留最近 7 份
.venv\Scripts\python.exe scripts\backup.py backup

# 写到别处、保留 14 份
.venv\Scripts\python.exe scripts\backup.py backup --out D:\rag-backups --keep 14
```

产物：

```
data/backups/backup-20261017-193720/
├── rag.db            元数据一致快照（VACUUM INTO）
├── uploads.tar.gz    只含元数据里登记过的原始文件
└── manifest.json     校验和 + 行数 + 状态分布 + 版本
```

`manifest.json` 长这样：

```json
{
  "format_version": 1,
  "created_at": "2026-10-17T19:37:20",
  "app_version": "0.1.0",
  "collection": "rag_backend_chunks",
  "sqlite":  {"file": "rag.db", "bytes": 40960, "sha256": "caf2bd…",
              "tables": {"knowledge_bases": 3, "documents": 7, "users": 1}},
  "uploads": {"file": "uploads.tar.gz", "bytes": 23245, "sha256": "9fcbb5…",
              "files": 7, "missing": 0},
  "documents": {"total": 7, "by_status": {"ready": 7}}
}
```

**只打"登记过的"文件**是有意的：整目录打包会把上次失败的半截文件、
手工丢进 uploads 的杂物一起带上，恢复时它们不属于任何文档，只能当垃圾。
反过来，登记了但磁盘上没有的文件会记进 `missing` 并打印告警——
备份不因此失败（其他文件仍然有价值），但那几篇注定无法重建。

## 4. 校验

```powershell
.venv\Scripts\python.exe scripts\backup.py list
.venv\Scripts\python.exe scripts\backup.py verify data\backups\backup-20261017-193720
```

```
[ok]   rag.db  36.0 KB  sha256=caf2bdf70d09075e…
[ok]   uploads.tar.gz  22.7 KB  sha256=9fcbb5525e6007a5…
[ok]   备份完整：2026-10-17T19:37:20  文档 7 条 {'ready': 7}  原始文件 7 个
```

**备份不校验等于没有备份**。校验和覆盖两个产物，任何字节变化都会被抓到
（硬盘静默损坏、网盘同步冲突、手工改过文件）。

## 5. 恢复

```powershell
# 破坏性操作：不加 --yes 只打印将要覆盖什么，不会动手
.venv\Scripts\python.exe scripts\backup.py restore data\backups\backup-20261017-193720 --yes
```

执行顺序（每一步都有理由）：

| 步骤 | 做什么 | 为什么 |
|---|---|---|
| ① | 校验 sha256 | 坏备份不许覆盖好数据 |
| ② | 先把当前数据备份成 `pre-restore-<时间戳>` | 后悔药。该目录**不参与** `--keep` 清理 |
| ③ | 删除 `rag.db-wal` / `rag.db-shm` 后覆盖 db | 旧 WAL 会与新库混在一起，恢复出来的数据是"半新半旧" |
| ④ | 把 `uploads/` **改名**为 `uploads.old-<时间戳>`，再解包 | 删了就找不回来；挪走只占磁盘 |
| ⑤ | 把 `ready` 文档置为 `failed` | Milvus 里没有对应向量，标 ready 就是撒谎（看着有、实际搜不到） |
| ⑥ | 打印重建命令 | 恢复没结束，向量还没回来 |

**第 ③ 步的坑最容易踩**：只删 db 不删 `-wal`，SQLite 会用 WAL 里的页
"补"到新库上，症状是恢复后文档条数对、但内容错乱，或者直接 malformed。

恢复后的状态：

```
$ .venv\Scripts\python.exe scripts\backup.py restore data\backups\backup-20261017-193720 --yes
元数据已恢复（36.0 KB）
原 uploads 已挪到 uploads.old-20261017-194012（确认无误后可自行删除）
原始文件已恢复
已把 7 条 ready 文档置为 failed（等待重建向量）
```

## 6. 重建向量

```powershell
# 先把服务起起来（脚本走 HTTP，不自己加载 BGE）
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000

# 另开一个终端
.venv\Scripts\python.exe scripts\rebuild_vectors.py            # 全部 failed 文档
.venv\Scripts\python.exe scripts\rebuild_vectors.py --dry-run  # 先看清单
.venv\Scripts\python.exe scripts\rebuild_vectors.py --kb <kb_id>
```

输出按 `[code]` 分类，直接告诉你哪些值得再试：

```
[1/7] 售后服务制度.pdf  ready（4 块）
[2/7] 短线操作模式.txt  failed：[corrupted_document] PDF 结构损坏…

完成：成功 6 / 失败 1
失败项看 error 前缀：
  [corrupted_document] / [encrypted_document] / [empty_document] → 文件本身的问题，重试无用
  [embedding_error] / [upstream_timeout] / [retrieval_error] → 依赖问题，修好后可再跑一次
```

`reprocess` 内部**先 `delete_by_doc` 再写**，所以重复跑不会产生重复向量，
Milvus 里若还残留旧向量也会被清掉。

## 7. 定时执行

Windows 计划任务（每天 02:30，保留 14 份）：

```powershell
$root   = "<项目根目录>"            # 换成你本地的绝对路径
$py     = "$root\.venv\Scripts\python.exe"
$script = "$root\scripts\backup.py"
Register-ScheduledTask -TaskName "rag-backup" `
  -Trigger (New-ScheduledTaskTrigger -Daily -At 02:30) `
  -Action  (New-ScheduledTaskAction -Execute $py `
            -Argument "$script backup --out H:\backups\rag --keep 14")
```

（`--out` 指向**另一块盘**。默认的 `data/backups` 与原数据同盘，盘坏了一起没。）

Linux cron：

```cron
30 2 * * * cd /opt/rag-backend && .venv/bin/python scripts/backup.py backup --out /backups/rag --keep 14
```

**备份必须落到另一块盘**。`data/backups/` 只是默认值——同一块盘上，
盘坏了备份和原数据一起没。遵循 3-2-1：至少 3 份、2 种介质、1 份异地。

## 8. 恢复演练

备份最危险的状态是"每天都在跑，但从没恢复过"。建议每月一次：

```powershell
# 1) 备一份
.venv\Scripts\python.exe scripts\backup.py backup

# 2) 在临时目录里恢复，不动生产数据
$env:SQLITE_PATH = "D:\rag-drill\rag.db"
$env:UPLOAD_DIR  = "D:\rag-drill\uploads"
.venv\Scripts\python.exe scripts\backup.py restore data\backups\backup-xxxx --yes

# 3) 校验行数与文件数
.venv\Scripts\python.exe -c "import sqlite3;c=sqlite3.connect(r'D:\rag-drill\rag.db');print(c.execute('select status,count(*) from documents group by status').fetchall())"
```

`SQLITE_PATH` / `UPLOAD_DIR` 是环境变量，优先级高于 `.env`——这是本项目
做隔离测试的通用手法（自检脚本也用它）。

## 9. 常见失败

| 现象 | 原因 | 处理 |
|---|---|---|
| `无法写入 rag.db：文件被占用` | 后端还在跑，Windows 独占写 | 先停 uvicorn，再恢复 |
| `校验和不符` | 备份被改动/复制损坏/硬盘静默损坏 | 换一份备份；别用这份恢复 |
| 恢复后文档数对但搜不到 | 忘了重建向量（第 6 步） | 跑 `rebuild_vectors.py` |
| 恢复后 `database disk image is malformed` | 手工 `cp` 过 db，或没删 `-wal` | 用 `VACUUM INTO` 的备份重来 |
| `missing: N 个登记过的原始文件已丢失` | uploads 被清理过 | 这些文档无法重建，删掉或重新上传 |
| `pre-restore-*` 越来越多 | 它不参与 `--keep`（有意） | 确认恢复无误后手工删 |

## 10. 相关配置

| 变量 | 默认 | 说明 |
|---|---:|---|
| `SQLITE_PATH` | `./data/rag.db` | 元数据库路径 |
| `UPLOAD_DIR` | `./data/uploads` | 原始文件目录 |
| `MILVUS_COLLECTION` | `rag_backend_chunks` | 写进 manifest，恢复时用它核对目标环境 |
| `MAX_BATCH_TOTAL_MB` | 200 | 备份不涉及，但恢复后批量重传受它约束 |
