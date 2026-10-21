# 鉴权与多租户隔离

> 相关代码：`app/api/v1/auth.py`（令牌签发/校验）、`app/services/ownership.py`（归属校验唯一执行点）、
> `app/db/models.py` + `app/db/database.py`（`owner_id` 与迁移）、
> `app/infrastructure/redis_store.py`（会话归属）、`app/api/v1/*.py`（路由级挂载）。
> 自检：`scripts/check_auth.py`（136 项）。

## 1. 一句话概括

**登录解决"你是谁"，归属解决"这数据是不是你的"，两件事缺一不可。**

只做登录不做归属，等于给每间房装了锁，但所有房间共用一把钥匙；
只做归属不做登录，则根本不知道"你"是谁。这个项目此前只做了前者的前半段
（能签发令牌、却不校验），现已补齐。

## 2. 两层结构

```
请求 → 路由级 router_dependencies → require_current_user
             ↓ 解出 current_user（401 若不通过）
        处理函数拿到 current_user.id
             ↓ 作为 owner_id 一路传下去
        服务层 ensure_kb_owned(repository, kb_id, owner_id)
             ↓ 404 若库不存在或不属于该账号
        仓储层 get_knowledge_base(kb_id, owner_id)
```

### 2.1 第一层：路由挂载（防"忘了加"）

```python
# app/api/v1/auth.py
router_dependencies = [Depends(require_current_user)]

# app/api/v1/collections.py 等
router = APIRouter(dependencies=router_dependencies)
```

**为什么不用"每个处理函数加一个 `Depends`"**：
逐个加的话，新写一个接口时忘了加就是一次静默的数据泄漏，而且不会有任何报错。
挂在 `APIRouter` 上，新接口**自动继承**保护；漏掉的可能性从"每写一个接口一次"
降到"每建一个路由模块一次"。

**白名单只有三个**：`/health`、`/auth/login`、`/auth/register`。
`scripts/check_auth.py` 会遍历所有已注册路由，断言白名单外**每一条**都挂了
`require_current_user`——包括将来新加的。

### 2.2 第二层：归属校验（防"知道 ID 就能读"）

```python
# 唯一的执行点
def ensure_kb_owned(repository, kb_id, owner_id) -> None:
    if owner_id is None:          # 内部路径（脚本/自检）的逃生舱
        return
    if repository is None:        # 装配错误：宁可错杀
        raise NotFoundError(KB_NOT_FOUND_MESSAGE, detail={"reason": "no_repository"})
    if repository.get_knowledge_base(kb_id, owner_id) is None:
        raise NotFoundError(KB_NOT_FOUND_MESSAGE)
```

**为什么归到一个函数**：此前 `ChatService` / `IngestService` / `RetrievalService`
各写了一份，三份几乎一样。安全逻辑一旦有多个副本，改规则时漏改一处就是漏洞，
且同样不会有报错。

**为什么 `owner_id` 是无默认值的必传参数**：
有默认值就意味着某天漏传时会静默退化成"谁都能读"。
强制必传让漏传在代码评审和类型检查阶段就暴露。

## 3. 为什么越权返回 404 而不是 403

403 的含义是"这个资源存在，但你不许看"——它本身就是一条信息。
配合 ID 遍历，攻击者可以：

1. 用 404/403 的差异**枚举**出别人有哪些知识库；
2. 从 ID 的创建顺序推断业务规模；
3. 对确认存在的 ID 做进一步攻击（社工、猜测命名规律）。

因此本项目把"不存在"与"不是你的"做成**逐字节相同**的响应：

```json
{"code": "not_found", "message": "知识库不存在或无权访问", "detail": null}
```

注意 `detail` 对归属类失败恒为 `null`——连"是哪种不存在"都不透露。

**唯一的例外**是 `detail: {"reason": "no_repository"}`：它表示"服务没装配仓储
却收到了 owner_id"，属于装配错误而非归属判断。它与归属类 404 的响应体不同，
理论上也是一个可观测的差异位——但触发它需要"容器漏传 repository"，那是部署事故
（所有知识库接口同时不可用），不构成针对单个 ID 的探测手段。

`KB_NOT_FOUND_MESSAGE` 定义在 `app/core/exceptions.py`，**全文只有这一个常量**，
不拼接 `kb_id`：拼了 ID，两个不同 ID 就会产生不同字符串，逐字节相同的前提就没了。

## 4. 数据层的落地细节

### 4.1 为什么是 `owner_id` 而不是另建一张权限表

当前需求是"每人只能看自己的"，这是一个**归属**问题，不是**授权**问题。
归属用一个列就够了；等真的要做"团队共享""只读成员"时再引入 ACL 表，
那时 `owner_id` 仍然有用（它是资源的创建者）。

### 4.2 SQLite 加列的约束

`ALTER TABLE ADD COLUMN` **不能加 NOT NULL 外键列**（需要整表重建），
所以 `owner_id` 用 `NOT NULL DEFAULT ''`：

```python
# app/db/database.py 的 _migrate_knowledge_bases()
ALTER TABLE knowledge_bases ADD COLUMN owner_id VARCHAR(36) NOT NULL DEFAULT ''
CREATE INDEX IF NOT EXISTS ix_knowledge_bases_owner_id ON knowledge_bases(owner_id)
```

**为什么给 `DEFAULT ''` 而不是允许 NULL**：
`owner_id = ?` 这类查询在遇到 NULL 时会因三值逻辑永不成立——连"自己的库"都查不到，
这是最典型的静默失效。空串能正常参与等值比较，只是匹配不上任何真实账号。

空字符串表示"历史遗留数据，没有归属"——它不会匹配任何真实账号 ID，
所以老数据在隔离生效后对所有人都不可见。这是**有意的**：
与其猜一个归属，不如让它消失，由运维确认后手工认领。

**为什么要显式写索引**：`ix_knowledge_bases_owner_id` 是列表查询的唯一过滤条件，
缺了它在库变大后表现为"知识库列表越来越慢"，却不会有任何报错。

**为什么要显式写迁移**：项目不引 Alembic，而 `create_all` **不会**给已存在的表
加列或补索引。只改 `models.py` 在旧库上不生效，且不会报错。

### 4.3 会话（Redis）的归属

会话存在 Redis 里，不在 SQLite。meta hash 加一个 `user_id` 字段，用 `HSETNX` 语义
（**只在空缺时写入**）：

```
append_turn(conv_id, ..., user_id="u1")   # 首轮：写入 user_id=u1
append_turn(conv_id, ..., user_id="u2")   # 续聊：user_id 仍是 u1，不被改写
```

为什么不是普通 `HSET`：续聊时若允许覆盖，任何人只要知道 `conversation_id`
就能"认领"别人的会话，归属随即失效。

列表/读/删三条路径都做过滤，且"不存在"与"不是你的"抛同一句话。

## 5. SSE 流式的特殊约束

**一旦开始推流，HTTP 状态码就改不了了。** 所以归属校验必须在建流**之前**完成，
且要完成两遍（路由依赖一次、处理函数里 `ensure_exists` 一次）。

实测验证（`/chat/stream` 越权请求）：

```
HTTP 404
content-type: application/json
{"code":"not_found","message":"知识库不存在或无权访问"}
```

返回的是 JSON 而不是 SSE 流——这正是"拦在建流前"的证据。
如果校验写进了 `_sse_generator` 内部，客户端会收到 `200 + 一条 error 事件`，
前端就得为流式和非流式写两套错误处理。

另外，`_sse_generator` 的 `owner_id` 类型是 `str | None`：`None` 表示内部调用，
与 `ensure_kb_owned` 的逃生舱语义保持一致。

## 6. 前端配合

| 机制 | 位置 | 作用 |
|---|---|---|
| 请求注入令牌 | `api/http.ts`、`api/sse.ts` | 两条链路（axios 与 fetch）都要注入，漏一条就会"普通接口正常、流式 401" |
| 401 统一处理 | `api/unauthorized.ts` | 清登录态 → 广播事件 → 无刷新跳登录页；并发 401 只跳一次 |
| 路由守卫 | `components/RequireAuth.tsx` | 无 token 直接跳；有 token 时先问 `/auth/me` |
| 启动校验 | 同上 + `api/modules/auth.ts` 的 `fetchCurrentUser` | localStorage 里的令牌可能早已过期或被停用 |
| 过期预判 | `store/auth.ts` 的 `isTokenExpired` | 本地能看出肯定过期就不必发请求；留 30 秒余量避免临界抖动 |

**启动校验的一个关键取舍**：`/auth/me` 返回 401 → 清登录态跳登录页；
但网络错误 / 5xx → **不清登录态，放行**。后端临时挂了不代表令牌无效，
此时把人踢去登录页，他连登录都登不上，只会更糟。

## 7. 已知边界（刻意不做）

| 边界 | 现状 | 说明 |
|---|---|---|
| 单点登出 | 无会话表，令牌在有效期内一直可用 | 换 `AUTH_SECRET_KEY` 可让全部令牌立即失效 |
| 团队共享 | 只有 `owner_id`，没有 ACL | 协作功能，与"隔离"是两件事 |
| 细粒度权限 | 无"只读/可写"区分 | 同上 |
| 审计日志 | 有创建账号与登录失败的日志（`auth_service`），但没有"谁读了哪个库"的访问日志 | 生产环境建议补：归属校验点天然是记录访问的好位置 |

## 8. 自检覆盖

`scripts/check_auth.py`（136 项）断言：

- 白名单外**每条**路由都挂了 `require_current_user`；
- 令牌签发/验签往返、篡改与过期被拒；
- 账号不存在与口令错误返回同一句话（防枚举）；
- 注册开关生效。

端到端隔离另有一套手工验证（两账号交叉读写），要点：

| 场景 | 期望 |
|---|---|
| bob 列知识库 | 只有自己的（alice 建 1 个 → bob 看到 0 个） |
| bob 读/改/删 alice 的库 | 全部 404，且 alice 的库仍在 |
| bob 检索 / 问答 / 流式问答 alice 的库 | 全 404；流式返回 JSON 而非 SSE |
| "不存在" vs "别人的" 响应体 | 逐字节相同 |
| alice 操作自己的库 | 200 / 204 正常 |
