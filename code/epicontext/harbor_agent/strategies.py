"""上下文选择策略: Full / SlidingWindow / EpiContext / Adaptive。"""

from __future__ import annotations

from .context_graph import ContextGraph
from .operators import EpigeneticOperators


class ContextStrategy:
    """上下文选择策略基类。"""

    def select_context(self, graph: ContextGraph, current_turn: int,
                       max_context_tokens: int = 4000) -> str:
        raise NotImplementedError


class FullContextStrategy(ContextStrategy):
    """全量上下文策略 - 包含所有历史。"""

    def select_context(self, graph: ContextGraph, current_turn: int,
                       max_context_tokens: int = 4000) -> str:
        parts = []
        token_budget = max_context_tokens
        for node in sorted(graph.nodes.values(), key=lambda n: n.node_id):
            text = f"[Turn {node.turn}] {node.role}: {node.content}"
            est_tokens = len(text) // 4
            if token_budget - est_tokens < 0:
                break
            parts.append(text)
            token_budget -= est_tokens
        return "\n\n".join(parts)


class SlidingWindowStrategy(ContextStrategy):
    """滑动窗口策略 - 仅保留最近 N 轮。"""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def select_context(self, graph: ContextGraph, current_turn: int,
                       max_context_tokens: int = 4000) -> str:
        min_turn = max(0, current_turn - self.window_size)
        parts = []
        for node in sorted(graph.nodes.values(), key=lambda n: n.node_id):
            if node.turn >= min_turn:
                parts.append(f"[Turn {node.turn}] {node.role}: {node.content}")
        return "\n\n".join(parts)


class EpiContextStrategy(ContextStrategy):
    """EpiContext 策略 - 甲基化 + 乙酰化 + 适应度反馈 (改进版)。"""

    def __init__(self, alpha: float = 1.0, beta: float = 2.0, gamma: float = 0.3,
                 silence_threshold: float = 1e-3, relevance_threshold: float = 0.5):
        self.alpha = alpha
        self.beta = beta  # 增大 token 效率权重
        self.gamma = gamma
        self.silence_threshold = silence_threshold  # 更激进的沉默阈值
        self.relevance_threshold = relevance_threshold  # 更高的相关性要求

    def select_context(self, graph: ContextGraph, current_turn: int,
                       max_context_tokens: int = 4000) -> str:
        ops = EpigeneticOperators(graph)
        ops.methylate(self.silence_threshold)
        ops.acetylate(self.relevance_threshold)
        ops.apply_fitness_feedback(self.alpha, self.beta, self.gamma)

        # 使用更高的激活阈值 (0.5)
        active = sorted(
            [n for n in graph.nodes.values() if n.activation > 0.5],
            key=lambda n: (n.activation, n.node_id),
            reverse=True,
        )

        parts = []
        token_budget = max_context_tokens
        for node in active:
            # 移除元数据标签，减少 token 开销
            text = f"[Turn {node.turn}] {node.role}: {node.content}"
            est_tokens = len(text) // 4
            if token_budget - est_tokens < 0:
                break
            parts.append(text)
            token_budget -= est_tokens

        return "\n\n".join(parts)


class AdaptiveEpiContextStrategy(ContextStrategy):
    """自适应 EpiContext 策略 - 前 N 轮用滑动窗口，之后用 EpiContext。"""

    def __init__(self, switch_turn: int = 10, window_size: int = 5, **kwargs):
        self.switch_turn = switch_turn
        self.window_strategy = SlidingWindowStrategy(window_size)
        self.epicontext_strategy = EpiContextStrategy(**kwargs)

    def select_context(self, graph: ContextGraph, current_turn: int,
                       max_context_tokens: int = 4000) -> str:
        if current_turn < self.switch_turn:
            return self.window_strategy.select_context(graph, current_turn, max_context_tokens)
        else:
            return self.epicontext_strategy.select_context(graph, current_turn, max_context_tokens)
