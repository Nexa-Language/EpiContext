"""表观遗传算子: 甲基化、乙酰化、适应度反馈。"""

from __future__ import annotations

import numpy as np

from .context_graph import ContextGraph


class EpigeneticOperators:
    """表观遗传算子: 甲基化、乙酰化、交叉重组。"""

    def __init__(self, graph: ContextGraph):
        self.graph = graph

    def methylate(self, silence_threshold: float = 1e-4) -> int:
        """甲基化: 沉默 loss 变化小的历史节点。返回沉默的节点数。"""
        silenced = 0
        for node in self.graph.nodes.values():
            if node.role == "observation" and abs(node.loss_delta) < silence_threshold:
                if node.activation > 0.1:
                    node.activation *= 0.5
                    silenced += 1
        return silenced

    def acetylate(self, relevance_threshold: float = 0.3) -> int:
        """乙酰化: 激活梯度方向一致的历史。返回激活的节点数。"""
        activated = 0
        recent_grads = [
            n.grad_norm for n in self.graph.nodes.values()
            if n.role == "observation" and n.grad_norm > 0
        ]
        if not recent_grads:
            return 0
        avg_grad = np.mean(recent_grads)

        for node in self.graph.nodes.values():
            if node.role == "observation" and node.grad_norm > 0:
                similarity = 1.0 - abs(node.grad_norm - avg_grad) / max(avg_grad, 1e-8)
                if similarity > relevance_threshold:
                    node.activation = min(1.0, node.activation * 1.5)
                    activated += 1
        return activated

    def apply_fitness_feedback(self, alpha: float = 1.0, beta: float = 0.5,
                                gamma: float = 0.3) -> float:
        """应用适应度反馈调整激活权重。返回平均适应度。"""
        active_nodes = self.graph.get_active_nodes()
        if not active_nodes:
            return 0.0

        total_tokens = sum(n.token_count for n in active_nodes)
        info_density = len(active_nodes) / max(total_tokens, 1)

        fitness_values = []
        for node in active_nodes:
            R_task = node.loss_delta if node.loss_delta > 0 else 0
            C_token = node.token_count / max(total_tokens, 1)
            I_density = info_density
            fitness = alpha * R_task - beta * C_token + gamma * I_density
            fitness_values.append(fitness)
            # 根据适应度调整激活
            node.activation = max(0.0, min(1.0, node.activation + 0.1 * fitness))

        return float(np.mean(fitness_values)) if fitness_values else 0.0
