"""
EpiContext: Epigenetic Context Evolution for Efficient Long-Horizon Agent Reasoning.

基于表观遗传学的Agent上下文动态演化框架。
"""

from epicontext.core import (
    ContextNode,
    ContextEdge,
    ContextGraph,
    EpigeneticOperators,
    FitnessFunction,
    EpiContextRouter,
)

__version__ = "0.1.0"
__all__ = [
    "ContextNode",
    "ContextEdge",
    "ContextGraph",
    "EpigeneticOperators",
    "FitnessFunction",
    "EpiContextRouter",
]