# =============================================================================
# 后端服务镜像
#
# 用途：把 rag-backend 打包成容器运行。
#
# 注意：本项目采用「Milvus 在容器、后端在本机」的开发方式时，
#       后端容器访问宿主机的 Milvus 与 Ollama 需用 host.docker.internal。
#       见 deploy/docker-compose.yml 中的环境变量。
# =============================================================================
FROM python:3.12-slim

# 国内网络下可解除下面一行注释以加速 apt
# RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    TZ=Asia/Shanghai

WORKDIR /app

# 先装依赖，利用镜像层缓存：只要 requirements.txt 不变就不重复安装
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 再拷代码
COPY app ./app
COPY configs ./configs
COPY scripts ./scripts

# 运行时数据目录（建议在 compose 中挂载出来，避免容器重建丢数据）
RUN mkdir -p /app/data/uploads

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
