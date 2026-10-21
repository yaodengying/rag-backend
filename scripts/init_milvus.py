"""Milvus 初始化脚本：建 collection、校验维度、打印统计。

用途：
    首次部署，或**更换 embedding 模型后重建索引**——Milvus 的向量维度
    无法原地变更，换模型本质上就是换一个 collection。

运行：
    .venv\\Scripts\\python.exe scripts\\init_milvus.py
    .venv\\Scripts\\python.exe scripts\\init_milvus.py --recreate --yes

重要：
    1. 维度**取自 embedding 权重**（权重目录 config.json 的 hidden_size），
       不是 MILVUS_DIMENSION 的字面量——后者只是声明值。两者不符时
       `BgeEmbedder` 在构造阶段就会抛错，本脚本如实转述，不会用它去建表。
       已存在的 collection 维度不符时**只报错、不静默重建**（那会丢数据）。
    2. `--recreate` 不可逆（连同全部向量一起删），必须显式确认：
       终端下要求手动输入 RECREATE；非交互环境下必须显式加 `--yes`，
       否则直接拒绝执行。绝不能出现"随手跑个脚本把库跑没了"。
    3. Milvus 走 gRPC。本机若有全局 http_proxy 需 `NO_PROXY=localhost,127.0.0.1`；
       向量库适配器在连接前已自动补上，脚本无需额外设置。

退出码：0 成功；1 失败（参数非法、连接失败、维度不符、确认被拒）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# settings 的 env_file 是相对路径，必须先切到项目根
os.chdir(PROJECT_ROOT)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from app.core.config import get_settings  # noqa: E402
from app.core.container import get_embedder, get_vector_store  # noqa: E402
from app.core.exceptions import ConfigurationError  # noqa: E402

# 交互确认要输入的字（手打一遍，避免回车键误触）
_CONFIRM_WORD = "RECREATE"
_KNOWN_FLAGS = ("--recreate", "--yes")


def _parse_args(argv: list[str]) -> tuple[bool, bool]:
    """解析参数。

    Returns:
        (是否重建, 是否跳过确认)

    Raises:
        ValueError: 出现未知参数
    """
    unknown = [item for item in argv if item not in _KNOWN_FLAGS]
    if unknown:
        raise ValueError(
            f"未知参数：{' '.join(unknown)}；可用参数：{' '.join(_KNOWN_FLAGS)}"
        )
    return "--recreate" in argv, "--yes" in argv


def _confirm_recreate(collection: str, assume_yes: bool) -> bool:
    """为不可逆的删除操作取一次显式同意。"""
    if assume_yes:
        print("  （已通过 --yes 显式确认重建）")
        return True

    if not sys.stdin.isatty():
        print(f"\n拒绝执行：--recreate 会删除 collection `{collection}` 及其全部向量，不可逆。")
        print("非交互环境下请显式表达意图，例如：init_milvus.py --recreate --yes")
        return False

    print(f"\n⚠️  即将删除 collection `{collection}` 及其全部向量，此操作**不可逆**。")
    try:
        answer = input(f"输入 {_CONFIRM_WORD} 以确认：").strip()
    except EOFError:  # 管道提前关闭
        return False
    return answer == _CONFIRM_WORD


def _safe(func, *args):  # noqa: ANN001, ANN201
    """读取失败就返回 None：统计信息读不到不该影响脚本的结论与退出码。"""
    try:
        return func(*args)
    except Exception:  # noqa: BLE001
        return None


def _print_schema(store, collection: str, dimension: int) -> None:  # noqa: ANN001
    """打印最终 collection 的字段、索引与行数。"""
    client = store.client
    description = _safe(client.describe_collection, collection)
    if not description:
        print("  （读取 schema 失败，跳过统计）")
        return

    fields = description.get("fields", [])
    names = [field.get("name", "?") for field in fields]
    vector_field = next((f for f in fields if f.get("name") == "embedding"), {})
    actual_dim = (vector_field.get("params") or {}).get("dim")

    print("\n  —— schema ——")
    print(f"    字段      : {', '.join(names)}")
    print(f"    向量维度  : {actual_dim}")
    if actual_dim is not None and int(actual_dim) != int(dimension):
        print("    ⚠️ 与权重维度不一致，这不该发生——请检查上方的维度提示")

    for index_name in _safe(client.list_indexes, collection) or []:
        info = _safe(client.describe_index, collection, index_name) or {}
        print(
            f"    索引 {index_name}   : {info.get('index_type')} "
            f"/ metric={info.get('metric_type')}"
        )

    stats = _safe(client.get_collection_stats, collection) or {}
    print(f"    向量行数  : {stats.get('row_count', '?')}")


def main(argv: list[str]) -> int:
    """脚本入口。

    Args:
        argv: 命令行参数（不含脚本名）

    Returns:
        进程退出码，0 表示成功。
    """
    try:
        recreate, assume_yes = _parse_args(argv)
    except ValueError as exc:
        print(f"参数错误：{exc}")
        return 1

    settings = get_settings()
    collection = settings.milvus_collection

    print("=" * 64)
    print("Milvus 初始化（rag-backend）")
    print(f"  服务地址    : {settings.milvus_uri}")
    print(f"  collection  : {collection}")
    print(f"  权重目录    : {settings.embedding_model_path}")

    # 维度与启动路径走同一处逻辑：不符会直接抛错，不会拿错维度去建表
    try:
        dimension = get_embedder().dimension
    except ConfigurationError as exc:
        print(f"\n无法确定向量维度：{exc}")
        return 1
    print(f"  向量维度    : {dimension}（取自权重，非配置字面量）")

    try:
        store = get_vector_store()
    except Exception as exc:  # noqa: BLE001
        print(f"\n连接 Milvus 失败：{exc}")
        return 1

    existed = bool(store.client.has_collection(collection))
    print(f"  Milvus 版本 : {_safe(store.client.get_server_version) or '?'}")
    print(f"  当前状态    : {'已存在' if existed else '不存在'}")
    print("=" * 64)

    if existed and recreate:
        if not _confirm_recreate(collection, assume_yes):
            print("已取消，未做任何改动。")
            return 1
        store.drop_collection()
        existed = False

    # ensure_collection 已内含"维度不符只报错不重建"的护栏，直接用
    try:
        store.ensure_collection(dimension)
    except ConfigurationError as exc:
        print(f"\n{exc}")
        return 1

    print(f"  collection  : {'沿用现有' if existed else '已新建'}")
    _print_schema(store, collection, dimension)
    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
