"""rag-backend 应用包。

分层（依赖方向严格由上向下，禁止反向导入）：

    api  →  services  →  rag  →  infrastructure  →  domain

- api             HTTP 接口层：路由、参数校验、SSE 输出
- services        应用层：用例编排（入库 / 检索 / 问答 / 知识库管理）
- rag             编排层：LangGraph 状态图、节点、Prompt、入库链
- infrastructure  适配层：解析 / 切分 / 向量化 / 向量库 / 重排 / LLM
- domain          领域层：领域模型与端口（抽象接口），不依赖任何实现
- core            横切关注点：配置、日志、异常、依赖容器
- db              持久化层：SQLite 元数据（SQLAlchemy）
- schemas         对外 DTO（Pydantic）
"""

__version__ = "0.1.0"
