"""领域异常体系与 FastAPI 异常处理器。

约定：
    - 领域层/服务层只抛本模块定义的异常，不抛 HTTPException。
    - 由 register_exception_handlers() 统一转换为 HTTP 响应，
      保证接口层不掺业务判断，响应结构统一。

响应结构：{"code": <str>, "message": <str>, "detail": <any>}
    与 app.schemas.common.ErrorResponse 一致，前端只需写一次错误处理。
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """所有业务异常的基类。

    Attributes:
        message: 面向用户的错误描述
        detail: 附加调试信息，生产环境可不下发
    """

    code: str = "app_error"
    http_status: int = 500

    def __init__(self, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class ConfigurationError(AppError):
    """配置缺失或非法（如模型目录不存在、维度不匹配）。"""

    code = "configuration_error"
    http_status = 500


class ValidationError(AppError):
    """入参校验失败（如知识库不存在、文件类型不支持）。"""

    code = "validation_error"
    http_status = 400


class NotFoundError(AppError):
    """资源不存在（如知识库 ID、文档 ID 未命中）。"""

    code = "not_found"
    http_status = 404


# 知识库找不到时的**唯一文案**。
#
# 为什么单独抽成常量、并且**不回显 kb_id**：
#   多租户隔离下，"这个库不存在"与"这个库不是你的"必须逐字节相同，
#   否则用随机 ID 试几次就能区分出哪种情况是"存在但不给我"——
#   那就是一个免费的知识库枚举接口。
#   回显 ID 看起来无害（它本来就是请求方自己传的），但它会让两种情况的
#   响应体出现差异，而"差异"正是枚举的依据。所以这里连 ID 也不带。
#
# 放在 exceptions.py 而不是各服务里各写一份：五处文案若有一处不同，
# 上面那条"逐字节相同"就破了，而这种不一致不会有任何报错。
KB_NOT_FOUND_MESSAGE = "知识库不存在或无权访问"


class UnsafeFileError(ValidationError):
    """上传内容未通过静态安全检查（魔数不符、压缩炸弹、伪装二进制）。

    单独成类而不是复用 `ValidationError`：两者的处置动作不同——
    400 的普通校验失败是"参数填错了，改一改重传"，
    而这里通常是"这个文件本身有问题，别传了"。
    前端/CLI 可以只按 `code` 分支，也可以按 400 统一处理，都不冲突。
    """

    code = "unsafe_file"
    http_status = 400


class ConflictError(AppError):
    """资源冲突（如登录名已被占用）。

    与 ValidationError 分开：入参本身合法，冲突来自"当前存储状态"。
    前端对这两者的处理不同——400 提示改输入，409 提示"换个名字/去登录"。
    """

    code = "conflict"
    http_status = 409


class UnauthorizedError(AppError):
    """未通过身份校验（登录失败、令牌缺失/过期/被篡改）。

    全部收敛到 401 而不是细分状态码：客户端动作只有一个——重新登录。
    具体原因放进 detail 供排查（例如 reason=expired 与 reason=bad_signature
    对运维的意义完全不同，但都不该体现在 HTTP 状态码上）。
    """

    code = "unauthorized"
    http_status = 401


class DocumentError(AppError):
    """上传/解析阶段的文档问题（客户端可修复）。

    **与 AppError 的区别是状态码**：文档坏了是 400（客户端换文件即可），
    而 AppError 默认 500（服务内部错误）。此前统一抛 AppError，
    结果同一份坏 PDF 在 `/documents` 上传里是 `status=failed`（看着对），
    但一旦走到别的入口（脚本、批处理、将来的同步接口）就变成 500，
    运维会去翻服务日志，而真正的动作是「让用户重新导出这个文件」。

    子类按可操作的动作再细分（`code` 是前端/CLI 可以分支的稳定标识）：
        corrupted_document —— 文件结构损坏，只能重新导出
        encrypted_document —— 有密码，解除保护即可
        empty_document     —— 没抽到文本，多半是扫描件需 OCR
    三种的处置方式完全不同，所以不能只用一个 `document_error` 糊过去。
    """

    code = "document_error"
    http_status = 400


class CorruptedDocumentError(DocumentError):
    """文件结构损坏（zip/pdf 流截断、交叉引用表缺失等）。"""

    code = "corrupted_document"


class EncryptedDocumentError(DocumentError):
    """文件加密，需要先解除密码保护。"""

    code = "encrypted_document"


class EmptyDocumentError(DocumentError):
    """文件可读但没有任何可提取文本（扫描件、纯图片页）。"""

    code = "empty_document"


class RetrievalError(AppError):
    """检索链路失败（向量库不可用、embedding 失败等）。"""

    code = "retrieval_error"
    http_status = 502


class EmbeddingError(RetrievalError):
    """向量化本身失败（模型前向异常、显存不足、输出形状非法）。

    继承 `RetrievalError` 而不是 `AppError`：语义上它确实属于「检索链路
    的上游坏了」（502），但**单独一个 code** 才能让运维区分
    「Milvus 连不上」与「BGE 前向炸了」——前者查容器，后者查显存与批大小。
    此前这类异常没有包装，会一路冒到兜底处理器变成 500「服务内部错误」，
    日志里只剩一段 torch 栈。
    """

    code = "embedding_error"


class RerankError(RetrievalError):
    """重排（交叉编码器）失败：显存不足、权重损坏、分数条数不符或含 NaN。

    与 `EmbeddingError` 对称：同属「检索链路的上游坏了」(502)，但单独一个
    code 才能让运维分辨该查哪一段——Embedding 炸了查向量化模型，
    Rerank 炸了查重排模型与 `RERANK_TOP_N`（后者直接决定这一批要算多少对）。
    此前 `compute_score` 的异常没有包装，会冒到兜底处理器变成 500，
    与已经修好的向量化路径行为不一致（同一个类别的故障、两种响应）。
    """

    code = "rerank_error"


class UpstreamError(AppError):
    """上游依赖失败（LLM 服务不可用、超时等）。"""

    code = "upstream_error"
    http_status = 502


class ContextWindowExceeded(AppError):
    """上下文预算连"最小可用提示词"都装不下。

    单独一个类而不是复用 UpstreamError：这不是上游坏了，而是**窗口/预留
    这组参数与本模型不匹配**（典型是把 `LLM_CONTEXT_WINDOW` 按界面声明值
    填了 262144，而实际服务的 `num_ctx` 只有 16384）。混进 502 会让运维
    去重启 Ollama，而真正该改的是 `.env`。

    为什么是 500 而不是 400：请求本身完全合法，是服务端配置/容量问题；
    客户端重试同一个请求也不会变好。
    """

    code = "context_window_exceeded"
    http_status = 500


class UpstreamTimeoutError(UpstreamError):
    """上游超时（LLM 单次请求超过 LLM_TIMEOUT）。

    用 504 而不是 502，是因为**两者的处置动作不同**：
        502 —— 上游没起来 / 崩了，动作是去启动或重启它；
        504 —— 上游在跑但太慢，动作是调大 LLM_TIMEOUT、换更小的模型，
               或者把问题拆短。前端也可以据此给出「重试」而不是「等服务恢复」。
    客户端对 504 重试是安全的（请求大概率没被处理完），对 502 则未必。
    """

    code = "upstream_timeout"
    http_status = 504


class ServiceUnavailableError(AppError):
    """依赖的**可选**能力未配置或暂不可用（如会话存储 REDIS_URL 未配置）。

    与 UpstreamError 分开的原因：502 说的是「依赖坏了」，运维动作是去修
    依赖；503 说的是「这个能力没开」，运维动作是去补配置。两者给运维的
    信号完全不同，混用会让「该改 .env」被误判成「该重启 Redis」。
    """

    code = "service_unavailable"
    http_status = 503


# =============================================================================
# 异常处理器
# =============================================================================


async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    """将 AppError 转为统一 JSON 错误响应。"""
    logger.warning("业务异常 %s：%s", exc.code, exc.message)
    return JSONResponse(
        status_code=exc.http_status,
        content={
            "code": exc.code,
            "message": exc.message,
            "detail": exc.detail,
        },
    )


async def request_validation_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    """把 FastAPI 的 422 也收敛成统一结构。

    默认响应体是 {"detail": [...]}，与业务错误结构不同，
    前端就得写两套解析逻辑，这里统一掉。
    """
    return JSONResponse(
        status_code=422,
        content={
            "code": "validation_error",
            "message": "请求参数校验失败",
            "detail": [
                {"loc": list(error.get("loc", [])), "msg": error.get("msg", "")}
                for error in exc.errors()
            ],
        },
    )


async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """兜底：未预期异常也返回统一结构。

    注意：星标（Starlette）的 ServerErrorMiddleware 仍会把 traceback 打进日志，
    所以这里收敛响应体不会损失可观测性。
    """
    logger.exception("未处理异常：%s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "code": "internal_error",
            "message": "服务内部错误",
            "detail": None,
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """在 FastAPI 应用上注册全部异常处理器。"""
    # AppError 一个处理器覆盖全部业务异常（子类都继承它）
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, request_validation_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
