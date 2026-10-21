# -*- coding: utf-8 -*-
"""离线评估（RAGAS）子包。

只被 `scripts/evaluate_rag.py` 使用，不进 `app/`：

    compat.py    让 `import ragas` 能在本机依赖栈上成立
    adapters.py  把项目的 LLM / 向量化能力接到 RAGAS 期望的 langchain 接口上
    dataset.py   评估数据集（JSONL）的加载与校验
    pipeline.py  进程内跑真实问答链路，收集预测与"喂给模型的上下文"
    scoring.py   RAGAS 指标装配 + 不依赖 LLM 的确定性检索指标
    report.py    结果落盘（summary.json / samples.jsonl / scores.csv / report.md）

为什么评估代码不放进 `app/`：
    `app/` 是被 uvicorn 服务的那棵树，每加一个模块都会进入生产进程的导入面。
    评估是开发期工具，指标库又是个重依赖（ragas + instructor），
    不该让线上服务的启动依赖它。
"""

from __future__ import annotations

__all__ = ["adapters", "compat", "dataset", "pipeline", "report", "scoring"]
