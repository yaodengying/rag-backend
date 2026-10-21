"""健康检查与就绪探针。

两个端点语义不同，不要混用：
    /health   存活探针（liveness）：进程还在就返回 200，不查依赖。
              用于容器重启判断——依赖挂了不应导致进程被反复重启。
    /ready    就绪探针（readiness）：检查 Milvus / 模型等关键依赖，
              未就绪返回 503。用于流量接入判断。
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app import __version__
from app.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", summary="存活检查")
async def health() -> dict:
    """返回进程存活状态，不检查外部依赖。"""
    return {"status": "ok", "version": __version__}


@router.get("/ready", summary="就绪检查")
async def ready():
    """检查 Milvus 与模型是否就绪；未就绪时返回 503。

    关于向量模型：这里**不加载**模型（首次加载要十几秒，探针会被打爆），
    只校验权重目录存在且维度配置与 Milvus 一致。真正的加载检查放在启动期。
    """
    settings = get_settings()
    checks: dict[str, str] = {}
    ready_ok = True

    # ① Milvus 连通性
    try:
        from app.core import container

        store = container.get_vector_store()
        if getattr(store, "ping", None) and store.ping():
            checks["milvus"] = "ok"
        else:
            checks["milvus"] = "unreachable"
            ready_ok = False
    except Exception as exc:  # noqa: BLE001
        checks["milvus"] = f"error: {exc}"
        ready_ok = False

    # ② 向量权重目录
    model_path = Path(settings.embedding_model_path) if settings.embedding_model_path else None
    if model_path and model_path.is_dir():
        checks["embedding_weights"] = "ok"
    else:
        checks["embedding_weights"] = f"missing: {settings.embedding_model_path or '(未配置)'}"
        ready_ok = False

    payload = {"status": "ready" if ready_ok else "not_ready", "checks": checks}
    if not ready_ok:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload)
    return payload
