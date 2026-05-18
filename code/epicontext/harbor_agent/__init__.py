"""EpiContext Harbor Agent v2 - 基于表观遗传学的 Agent 上下文动态演化框架。

改进版:
- 紧凑激活编码 (移除文本标签，消除 15-22% 元数据开销)
- 自适应激活阈值 (前 N 轮用滑动窗口，之后用 EpiContext)
- 激进过滤参数 (threshold=0.5, β=2.0)
- 集成 tiktoken 精确 token 计数

用法:
    cd harbor-framework
    uv run harbor run -p examples/tasks/hello-world \\
        --agent-import-path epicontext.harbor_agent:EpiContextAgent \\
        -n 1 -q

模块结构:
    llm_client     -> LLMClient
    context_graph  -> ContextNode, ContextGraph
    operators      -> EpigeneticOperators (甲基化 / 乙酰化 / 适应度反馈)
    strategies     -> Full / SlidingWindow / EpiContext / Adaptive 策略
    agent          -> TurnRecord, EpiContextAgent (主 Agent)
    baselines      -> 对比实验用的 4 个基线 + 自适应 Agent

为保持向后兼容 (`epicontext.harbor_agent:XxxAgent`)，本 __init__ 重新导出所有原符号。
"""

from .llm_client import LLMClient
from .context_graph import ContextNode, ContextGraph
from .operators import EpigeneticOperators
from .strategies import (
    AdaptiveEpiContextStrategy,
    ContextStrategy,
    EpiContextStrategy,
    FullContextStrategy,
    SlidingWindowStrategy,
)
from .agent import EpiContextAgent, TurnRecord
from .baselines import (
    AcetylationOnlyAgent,
    AdaptiveEpiContextAgent,
    FullContextBaselineAgent,
    MethylationOnlyAgent,
    SlidingWindowBaselineAgent,
)

__all__ = [
    # llm
    "LLMClient",
    # graph
    "ContextNode",
    "ContextGraph",
    # operators
    "EpigeneticOperators",
    # strategies
    "ContextStrategy",
    "FullContextStrategy",
    "SlidingWindowStrategy",
    "EpiContextStrategy",
    "AdaptiveEpiContextStrategy",
    # agents
    "TurnRecord",
    "EpiContextAgent",
    "FullContextBaselineAgent",
    "SlidingWindowBaselineAgent",
    "MethylationOnlyAgent",
    "AcetylationOnlyAgent",
    "AdaptiveEpiContextAgent",
]
