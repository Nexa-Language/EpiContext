"""Baseline Agents (用于对比实验)。"""

from __future__ import annotations

from pathlib import Path

from .agent import EpiContextAgent


class FullContextBaselineAgent(EpiContextAgent):
    """全量上下文基线 Agent。"""

    @staticmethod
    def name() -> str:
        return "full-context-baseline"

    def __init__(self, logs_dir: Path, model_name: str | None = None, **kwargs):
        kwargs["strategy"] = "full"
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)

    def version(self) -> str:
        return "2.0.0"


class SlidingWindowBaselineAgent(EpiContextAgent):
    """滑动窗口基线 Agent。"""

    @staticmethod
    def name() -> str:
        return "sliding-window-baseline"

    def __init__(self, logs_dir: Path, model_name: str | None = None, **kwargs):
        kwargs["strategy"] = "sliding"
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)

    def version(self) -> str:
        return "2.0.0"


class MethylationOnlyAgent(EpiContextAgent):
    """仅甲基化消融 Agent。"""

    @staticmethod
    def name() -> str:
        return "methylation-only"

    def __init__(self, logs_dir: Path, model_name: str | None = None, **kwargs):
        kwargs["strategy"] = "epicontext"
        kwargs["alpha"] = 1.0
        kwargs["beta"] = 0.0  # 不考虑 token 成本
        kwargs["gamma"] = 0.0  # 不考虑信息密度
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)

    def version(self) -> str:
        return "2.0.0"


class AcetylationOnlyAgent(EpiContextAgent):
    """仅乙酰化消融 Agent。"""

    @staticmethod
    def name() -> str:
        return "acetylation-only"

    def __init__(self, logs_dir: Path, model_name: str | None = None, **kwargs):
        kwargs["strategy"] = "epicontext"
        kwargs["alpha"] = 0.0
        kwargs["beta"] = 2.0
        kwargs["gamma"] = 0.3
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)

    def version(self) -> str:
        return "2.0.0"


class AdaptiveEpiContextAgent(EpiContextAgent):
    """自适应 EpiContext Agent - 前 N 轮用 SlidingWindow，之后用 EpiContext。"""

    @staticmethod
    def name() -> str:
        return "adaptive-epicontext"

    def __init__(self, logs_dir: Path, model_name: str | None = None, **kwargs):
        kwargs["strategy"] = "adaptive"
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)

    def version(self) -> str:
        return "2.0.0"
