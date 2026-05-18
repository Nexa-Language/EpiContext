"""上下文图谱: EpiContext 的核心数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class ContextNode:
    """上下文图谱中的节点。"""
    node_id: int
    turn: int
    role: str  # "thought", "action", "observation"
    content: str
    activation: float = 1.0  # 表观遗传激活权重 [0, 1]
    loss_delta: float = 0.0  # 该步带来的进展变化
    grad_norm: float = 0.0   # 梯度范数 (用于乙酰化)
    token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "turn": self.turn,
            "role": self.role,
            "content": self.content[:200],
            "activation": self.activation,
            "loss_delta": self.loss_delta,
            "grad_norm": self.grad_norm,
            "token_count": self.token_count,
        }


class ContextGraph:
    """上下文图谱 - EpiContext 的核心数据结构。"""

    def __init__(self, max_nodes: int = 10000):
        self.nodes: Dict[int, ContextNode] = {}
        self.edges: List[Tuple[int, int, float]] = []  # (from_id, to_id, weight)
        self._next_id: int = 0
        self.max_nodes: int = max_nodes

    def add_node(self, turn: int, role: str, content: str,
                 loss_delta: float = 0.0, grad_norm: float = 0.0,
                 token_count: int = 0) -> int:
        node = ContextNode(
            node_id=self._next_id, turn=turn, role=role,
            content=content, loss_delta=loss_delta,
            grad_norm=grad_norm, token_count=token_count,
        )
        self.nodes[self._next_id] = node
        if self._next_id > 0:
            self.edges.append((self._next_id - 1, self._next_id, 1.0))
        self._next_id += 1
        self._auto_prune()
        return node.node_id

    def get_active_nodes(self, threshold: float = 0.1) -> List[ContextNode]:
        return [n for n in self.nodes.values() if n.activation > threshold]

    def get_recent_nodes(self, n: int = 10) -> List[ContextNode]:
        sorted_nodes = sorted(self.nodes.values(), key=lambda x: x.node_id, reverse=True)
        return sorted_nodes[:n]

    def set_activation(self, node_id: int, value: float) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].activation = max(0.0, min(1.0, value))

    def _auto_prune(self) -> None:
        if len(self.nodes) > self.max_nodes:
            # 移除激活值最低的节点
            sorted_ids = sorted(
                self.nodes.keys(),
                key=lambda nid: self.nodes[nid].activation
            )
            to_remove = sorted_ids[:len(self.nodes) - self.max_nodes]
            for nid in to_remove:
                del self.nodes[nid]
            self.edges = [(f, t, w) for f, t, w in self.edges
                          if f in self.nodes and t in self.nodes]

    def to_summary(self) -> Dict[str, Any]:
        active = self.get_active_nodes()
        return {
            "total_nodes": len(self.nodes),
            "active_nodes": len(active),
            "avg_activation": np.mean([n.activation for n in self.nodes.values()]) if self.nodes else 0,
            "total_tokens": sum(n.token_count for n in self.nodes.values()),
        }
