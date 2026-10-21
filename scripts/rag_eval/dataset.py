# -*- coding: utf-8 -*-
"""评估数据集的加载与校验。

数据集是**一行一条**的 JSONL，字段含义：

    id                 唯一标识（写进结果文件，便于回溯到具体题目）
    kb_id              该题所属知识库。不同题可以打不同库，
                       因为"评估"评的是库 + 检索 + 生成的整条链路
    question           用户问题（原样送进问答链路，不做改写）
    reference          标准答案。RAGAS 的 context_recall / context_precision
                       以它为判据，因此必须是与语料一致的事实句
    reference_contexts 标准答案所依据的原文片段（可选，供定性的 NonLLM 指标）
    expected_sources   期望命中的「文件名 + 页码」，用于计算**确定性**召回率。
                       这是本项目"溯源到页码"能力的直接验收口
    answerable         该题在语料里是否**有**答案。false 表示应当拒答，
                       只参与拒答正确率统计，不参与 RAGAS 聚合——
                       RAGAS 没有"应该拒答"的概念，硬算会污染均值

为什么不用 `datasets` 直接读：字段校验要在跑模型之前完成。一条坏数据
（少 kb_id、answerable=true 却没 reference）若等到 RAGAS 里才炸，
前面的 20 分钟 LLM 调用就白花了。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class DatasetError(ValueError):
    """数据集格式错误（内容问题，不是环境问题）。"""


@dataclass(frozen=True, slots=True)
class ExpectedSource:
    """期望命中的来源。

    Attributes:
        filename: 文件名，与 `Document.filename` 完全一致
        page: 1 起的页码；None 表示只校验文件名（不分页的格式）
    """

    filename: str
    page: int | None = None

    def describe(self) -> str:
        return f"{self.filename} 第{self.page}页" if self.page else self.filename


@dataclass(frozen=True, slots=True)
class EvalSample:
    """一条评估样本。"""

    id: str
    kb_id: str
    question: str
    reference: str = ""
    reference_contexts: tuple[str, ...] = field(default_factory=tuple)
    expected_sources: tuple[ExpectedSource, ...] = field(default_factory=tuple)
    answerable: bool = True

    @property
    def is_refusal_case(self) -> bool:
        """是否为"应当拒答"的样本。"""
        return not self.answerable


_REQUIRED = ("id", "kb_id", "question")


def _parse_sources(raw: object, where: str) -> tuple[ExpectedSource, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise DatasetError(f"{where}: expected_sources 必须是数组")

    sources: list[ExpectedSource] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise DatasetError(f"{where}: expected_sources[{index}] 必须是对象")
        filename = item.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            raise DatasetError(f"{where}: expected_sources[{index}].filename 缺失")
        page = item.get("page")
        if page is not None and not isinstance(page, int):
            raise DatasetError(f"{where}: expected_sources[{index}].page 必须是整数或 null")
        sources.append(ExpectedSource(filename=filename.strip(), page=page))
    return tuple(sources)


def _to_sample(raw: dict, where: str) -> EvalSample:
    for key in _REQUIRED:
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise DatasetError(f"{where}: 缺少必填字段 {key}")

    answerable = raw.get("answerable", True)
    if not isinstance(answerable, bool):
        raise DatasetError(f"{where}: answerable 必须是布尔值")

    reference = raw.get("reference") or ""
    if not isinstance(reference, str):
        raise DatasetError(f"{where}: reference 必须是字符串")
    if answerable and not reference.strip():
        raise DatasetError(
            f"{where}: answerable=true 却没有 reference；"
            "context_recall / context_precision 需要标准答案作判据"
        )

    contexts = raw.get("reference_contexts") or []
    if not isinstance(contexts, list) or any(not isinstance(c, str) for c in contexts):
        raise DatasetError(f"{where}: reference_contexts 必须是字符串数组")

    expected = _parse_sources(raw.get("expected_sources"), where)
    if answerable and not expected:
        # 没有期望来源也能算 RAGAS 指标，但召回率无从计算。不拦，
        # 只在这里说明：调用方最好补齐，否则确定性召回指标会缺这一题。
        pass

    return EvalSample(
        id=raw["id"].strip(),
        kb_id=raw["kb_id"].strip(),
        question=raw["question"].strip(),
        reference=reference.strip(),
        reference_contexts=tuple(c.strip() for c in contexts if c.strip()),
        expected_sources=expected,
        answerable=answerable,
    )


def load_dataset(path: Path | str) -> list[EvalSample]:
    """读取并校验数据集。

    Args:
        path: JSONL 文件路径

    Returns:
        样本列表（保持文件中的顺序）

    Raises:
        DatasetError: 文件不存在、某行不是 JSON、字段缺失或取值非法、id 重复
        FileNotFoundError: 路径不存在（与内容错误区分开）
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"数据集不存在：{file_path}")

    samples: list[EvalSample] = []
    seen: dict[str, int] = {}
    with file_path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("//"):
                continue
            where = f"{file_path.name}:{lineno}"
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{where}: 不是合法 JSON（{exc.msg}）") from exc
            if not isinstance(raw, dict):
                raise DatasetError(f"{where}: 每行必须是一个 JSON 对象")

            sample = _to_sample(raw, where)
            if sample.id in seen:
                raise DatasetError(
                    f"{where}: id={sample.id} 与第 {seen[sample.id]} 行重复"
                )
            seen[sample.id] = lineno
            samples.append(sample)

    if not samples:
        raise DatasetError(f"{file_path} 里没有任何样本")
    return samples


def select(
    samples: list[EvalSample],
    ids: list[str] | None = None,
    limit: int | None = None,
) -> list[EvalSample]:
    """按 id 白名单 / 条数上限裁剪样本（调试与小样本试跑用）。"""
    selected = samples
    if ids:
        wanted = {item.strip() for item in ids if item.strip()}
        unknown = wanted - {s.id for s in samples}
        if unknown:
            raise DatasetError(f"数据集里没有这些 id：{', '.join(sorted(unknown))}")
        selected = [s for s in selected if s.id in wanted]
    if limit is not None:
        if limit <= 0:
            raise DatasetError("limit 必须为正整数")
        selected = selected[:limit]
    if not selected:
        raise DatasetError("筛选后没有剩余样本")
    return selected
