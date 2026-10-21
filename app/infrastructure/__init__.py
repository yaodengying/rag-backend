"""适配层：实现 app/domain/ports.py 中定义的能力端口。

每个模块对应一类外部能力，且是**唯一**允许直接依赖第三方驱动的地方：

    loading.py      文档加载      LangChain Loader（社区版 PDF/DOCX/Text + 自研 PPTX）
    parsing.py      文档解析      langchain-core 的 Document → 领域 Page（无第三方依赖）
    splitting.py    文本切分      langchain-text-splitters
    embedding.py    向量化        FlagEmbedding（本地 BGE 权重）
    vectorstore.py  向量库        pymilvus（MilvusClient）
    rerank.py       重排          FlagEmbedding 交叉编码器
    llm.py          大模型        langchain-openai（OpenAI 兼容协议）

替换任一实现（如把 Milvus 换成 pgvector）只需新增一个文件并在
app/core/container.py 中改一行装配代码，业务层零改动。
"""

from .embedding import BgeEmbedder
from .llm import OpenAICompatibleLLM
from .loading import LangChainDocumentLoader, PptxLoader
from .parsing import CompositeDocumentParser
from .rerank import BgeReranker
from .splitting import LangChainSplitter
from .vectorstore import MilvusVectorStore

__all__ = [
    "BgeEmbedder",
    "OpenAICompatibleLLM",
    "LangChainDocumentLoader",
    "PptxLoader",
    "CompositeDocumentParser",
    "BgeReranker",
    "LangChainSplitter",
    "MilvusVectorStore",
]
