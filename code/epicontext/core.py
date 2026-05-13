"""
EpiContext Core Implementation

基于表观遗传学的Agent上下文动态演化框架核心实现。

核心概念:
- 基因组 (Genome): Agent的完整历史记录、工具Schema、环境状态构成的全量知识库
- 表观遗传表达 (Epigenetic Expression): 每次发送给LLM的Request Payload
- 甲基化 (Methylation): 沉默噪声记忆，降低不相关历史记录的活性权重
- 乙酰化 (Acetylation): 激活关键工具，提升相关工具的活性权重
- 交叉重组 (Crossover): 遇到死锁时并行生成多个变异Request，选取最优路径
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ContextNode:
    """上下文图谱中的节点，代表单个事件。

    Attributes:
        node_id: 唯一标识符
        node_type: 节点类型 ('system', 'thought', 'action', 'observation', 'error', 'summary')
        content: 节点内容
        timestamp: 创建时间戳
        epigenetic_tag: 表观遗传标签 - 表达活性权重 w ∈ [0,1]
            1.0 = 完全激活 (高表达)
            0.0 = 完全沉默 (甲基化)
        metadata: 附加元数据
    """
    node_id: int
    node_type: str
    content: str
    timestamp: float
    epigenetic_tag: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_active(self, threshold: float = 0.1) -> bool:
        """检查节点是否处于活跃状态。"""
        return self.epigenetic_tag > threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'content': self.content,
            'timestamp': self.timestamp,
            'epigenetic_tag': self.epigenetic_tag,
            'metadata': self.metadata,
        }


@dataclass
class ContextEdge:
    """上下文图谱中的边，代表节点间的关系。

    Attributes:
        source_id: 源节点ID
        target_id: 目标节点ID
        edge_type: 边类型 ('temporal', 'causal', 'hierarchical', 'summarizes')
        weight: 边权重
    """
    source_id: int
    target_id: int
    edge_type: str = 'temporal'
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'edge_type': self.edge_type,
            'weight': self.weight,
        }


@dataclass
class Chromosome:
    """染色体 - 单次Request Payload的抽象表示。

    四条染色体:
    - C_sys: 系统规则 (不变基因)
    - C_env: 环境状态 (如目录树、终端状态)
    - C_tool: 工具声明 (JSON Schema)
    - C_mem: 历史记忆序列
    """
    sys_nodes: List[int] = field(default_factory=list)
    env_nodes: List[int] = field(default_factory=list)
    tool_nodes: List[int] = field(default_factory=list)
    mem_nodes: List[int] = field(default_factory=list)

    def total_nodes(self) -> int:
        return (len(self.sys_nodes) + len(self.env_nodes) +
                len(self.tool_nodes) + len(self.mem_nodes))


# ============================================================================
# Context Graph
# ============================================================================

class ContextGraph:
    """上下文图谱 (Context Graph)

    将Agent的整个生命周期维护为一个有向无环图（DAG）。
    不再使用平面的 messages 数组，而是结构化的图表示。

    使用示例:
        graph = ContextGraph()
        nid = graph.add_node('thought', 'I need to search for...')
        graph.add_edge(prev_nid, nid)
    """

    def __init__(self, max_nodes: int = 10000):
        self.nodes: Dict[int, ContextNode] = {}
        self.edges: List[ContextEdge] = []
        self._next_id: int = 0
        self.max_nodes: int = max_nodes
        # 邻接表用于快速查询
        self._adjacency: Dict[int, List[int]] = defaultdict(list)
        self._reverse_adjacency: Dict[int, List[int]] = defaultdict(list)

    def add_node(
        self,
        node_type: str,
        content: str,
        epigenetic_tag: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """添加节点到图谱中。

        Args:
            node_type: 节点类型
            content: 节点内容
            epigenetic_tag: 初始表观遗传标签
            metadata: 附加元数据

        Returns:
            新节点的ID
        """
        node_id = self._next_id
        self._next_id += 1

        node = ContextNode(
            node_id=node_id,
            node_type=node_type,
            content=content,
            timestamp=time.time(),
            epigenetic_tag=epigenetic_tag,
            metadata=metadata or {},
        )
        self.nodes[node_id] = node

        # 如果超过最大节点数，触发自动甲基化
        if len(self.nodes) > self.max_nodes:
            self._auto_prune()

        return node_id

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        edge_type: str = 'temporal',
        weight: float = 1.0,
    ) -> None:
        """添加边到图谱中。

        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            edge_type: 边类型
            weight: 边权重
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError(f"Invalid node IDs: {source_id}, {target_id}")

        edge = ContextEdge(source_id, target_id, edge_type, weight)
        self.edges.append(edge)
        self._adjacency[source_id].append(target_id)
        self._reverse_adjacency[target_id].append(source_id)

    def get_node(self, node_id: int) -> Optional[ContextNode]:
        """获取指定ID的节点。"""
        return self.nodes.get(node_id)

    def get_active_nodes(self, threshold: float = 0.1) -> List[ContextNode]:
        """获取所有活性权重高于阈值的节点。

        Args:
            threshold: 活性阈值

        Returns:
            活跃节点列表，按时间戳排序
        """
        active = [n for n in self.nodes.values() if n.is_active(threshold)]
        active.sort(key=lambda n: n.timestamp)
        return active

    def get_nodes_by_type(self, node_type: str) -> List[ContextNode]:
        """按类型获取节点。"""
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def get_recent_nodes(self, n: int = 10) -> List[ContextNode]:
        """获取最近的n个节点。"""
        sorted_nodes = sorted(self.nodes.values(), key=lambda x: x.timestamp, reverse=True)
        return sorted_nodes[:n]

    def get_neighbors(self, node_id: int) -> List[int]:
        """获取节点的邻居。"""
        return self._adjacency.get(node_id, [])

    def get_predecessors(self, node_id: int) -> List[int]:
        """获取节点的前驱。"""
        return self._reverse_adjacency.get(node_id, [])

    def set_epigenetic_tag(self, node_id: int, tag: float) -> None:
        """设置节点的表观遗传标签。

        Args:
            node_id: 节点ID
            tag: 新的标签值 (0.0 到 1.0)
        """
        if node_id in self.nodes:
            self.nodes[node_id].epigenetic_tag = max(0.0, min(1.0, tag))

    def set_epigenetic_tags_batch(self, tags: Dict[int, float]) -> None:
        """批量设置表观遗传标签。"""
        for node_id, tag in tags.items():
            self.set_epigenetic_tag(node_id, tag)

    def build_chromosome(self) -> Chromosome:
        """从当前图谱构建染色体表示。

        Returns:
            包含四条染色体的Chromosome对象
        """
        active = self.get_active_nodes()
        chromosome = Chromosome()

        for node in active:
            if node.node_type in ('system',):
                chromosome.sys_nodes.append(node.node_id)
            elif node.node_type in ('environment', 'file_snapshot', 'terminal'):
                chromosome.env_nodes.append(node.node_id)
            elif node.node_type in ('tool_schema', 'tool_call', 'tool_result'):
                chromosome.tool_nodes.append(node.node_id)
            else:
                chromosome.mem_nodes.append(node.node_id)

        return chromosome

    def to_dict(self) -> Dict[str, Any]:
        """将图谱序列化为字典。"""
        return {
            'nodes': [n.to_dict() for n in self.nodes.values()],
            'edges': [e.to_dict() for e in self.edges],
            'next_id': self._next_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextGraph':
        """从字典反序列化图谱。"""
        graph = cls()
        graph._next_id = data.get('next_id', 0)

        for node_data in data.get('nodes', []):
            node = ContextNode(**node_data)
            graph.nodes[node.node_id] = node

        for edge_data in data.get('edges', []):
            edge = ContextEdge(**edge_data)
            graph.edges.append(edge)
            graph._adjacency[edge.source_id].append(edge.target_id)
            graph._reverse_adjacency[edge.target_id].append(edge.source_id)

        return graph

    def _auto_prune(self) -> None:
        """自动修剪：甲基化最老的节点。"""
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.timestamp)
        prune_count = len(self.nodes) - self.max_nodes + 100  # 保留一些余量

        for node in sorted_nodes[:prune_count]:
            if node.node_type not in ('system',):  # 不修剪系统节点
                node.epigenetic_tag = 0.0

    def __len__(self) -> int:
        return len(self.nodes)


# ============================================================================
# Epigenetic Operators
# ============================================================================

class EpigeneticOperators:
    """表观遗传算子

    实现三种核心表观遗传操作:
    1. 甲基化 (Methylation): 沉默噪声记忆
    2. 乙酰化 (Acetylation): 激活关键工具
    3. 交叉重组 (Crossover): 并行探索变异上下文
    """

    def __init__(self, graph: ContextGraph):
        self.graph = graph
        self.methylation_history: List[Dict[str, Any]] = []
        self.acetylation_history: List[Dict[str, Any]] = []
        self.crossover_history: List[Dict[str, Any]] = []

    # ---- Methylation ----

    def methylate(
        self,
        node_ids: List[int],
        summary_content: Optional[str] = None,
        strategy: str = 'full_silence',
    ) -> Optional[int]:
        """记忆甲基化: 沉默指定节点并生成摘要。

        当Agent在某条路径上疯狂试错并最终解决时，
        详细Log会成为后续任务的噪声。甲基化将这些节点的
        活性权重降至0，并生成高维度的摘要节点。

        Args:
            node_ids: 要甲基化的节点ID列表
            summary_content: 可选的摘要内容 (None则自动生成)
            strategy: 甲基化策略
                - 'full_silence': 完全沉默 (w=0)
                - 'partial': 部分沉默 (w=0.1)
                - 'gradual': 渐进沉默 (w递减)

        Returns:
            摘要节点的ID，如果无需生成摘要则返回None
        """
        if not node_ids:
            return None

        # 根据策略设置权重
        weight_map = {
            'full_silence': 0.0,
            'partial': 0.1,
            'gradual': 0.3,  # 首次甲基化
        }
        target_weight = weight_map.get(strategy, 0.0)

        for nid in node_ids:
            if nid in self.graph.nodes:
                node = self.graph.nodes[nid]
                if strategy == 'gradual':
                    # 渐进式: 每次减半
                    node.epigenetic_tag *= 0.5
                else:
                    node.epigenetic_tag = target_weight

        # 生成摘要节点
        if summary_content is None:
            summary_content = self._auto_summarize(node_ids)

        summary_id = self.graph.add_node(
            node_type='summary',
            content=summary_content,
            epigenetic_tag=1.0,
            metadata={
                'methylated_nodes': node_ids,
                'strategy': strategy,
                'methylation_time': time.time(),
            },
        )

        # 添加摘要边
        for nid in node_ids:
            self.graph.add_edge(summary_id, nid, 'summarizes')

        # 记录历史
        self.methylation_history.append({
            'node_ids': node_ids,
            'summary_id': summary_id,
            'strategy': strategy,
            'timestamp': time.time(),
        })

        return summary_id

    def methylate_by_type(
        self,
        node_type: str,
        keep_recent: int = 5,
    ) -> Optional[int]:
        """按类型甲基化: 沉默指定类型的老节点。

        Args:
            node_type: 要甲基化的节点类型
            keep_recent: 保留最近的N个节点

        Returns:
            摘要节点ID
        """
        typed_nodes = self.graph.get_nodes_by_type(node_type)
        typed_nodes.sort(key=lambda n: n.timestamp)

        if len(typed_nodes) <= keep_recent:
            return None

        to_methylate = [n.node_id for n in typed_nodes[:-keep_recent]]
        return self.methylate(to_methylate, strategy='full_silence')

    def methylate_error_loops(
        self,
        max_consecutive_errors: int = 3,
    ) -> Optional[int]:
        """甲基化错误循环: 检测并沉默连续报错的节点。

        Args:
            max_consecutive_errors: 触发甲基化的连续错误数

        Returns:
            摘要节点ID
        """
        error_nodes = self.graph.get_nodes_by_type('error')
        if len(error_nodes) < max_consecutive_errors:
            return None

        # 找到最近的连续错误块
        error_nodes.sort(key=lambda n: n.timestamp)
        consecutive_block = []
        for node in reversed(error_nodes):
            consecutive_block.append(node.node_id)
            if len(consecutive_block) >= max_consecutive_errors:
                break

        return self.methylate(
            consecutive_block,
            summary_content=f"Resolved after {len(consecutive_block)} consecutive errors.",
            strategy='full_silence',
        )

    # ---- Acetylation ----

    def acetylate_tools(
        self,
        tool_schemas: List[Dict[str, Any]],
        current_task: str,
        relevance_threshold: float = 0.3,
        max_tools: int = 10,
    ) -> List[Dict[str, Any]]:
        """工具乙酰化: 基于当前任务筛选相关工具。

        Agent通常携带20+个工具，但某个具体时刻只需要2-3个。
        乙酰化评估各工具Schema的相关性，暂时移除无关工具，
        极大减少System Prompt区域的Token浪费。

        Args:
            tool_schemas: 所有可用工具的Schema列表
            current_task: 当前任务描述
            relevance_threshold: 相关性阈值 (0-1)
            max_tools: 最大保留工具数

        Returns:
            筛选后的工具Schema列表，按相关性降序排列
        """
        if not tool_schemas:
            return []

        # 计算每个工具的相关性
        scored_tools = []
        for tool in tool_schemas:
            relevance = self._compute_tool_relevance(tool, current_task)
            scored_tools.append((relevance, tool))

        # 按相关性降序排列
        scored_tools.sort(key=lambda x: x[0], reverse=True)

        # 筛选
        result = []
        for relevance, tool in scored_tools:
            if relevance >= relevance_threshold and len(result) < max_tools:
                tool_copy = dict(tool)
                tool_copy['_epigenetic_relevance'] = relevance
                result.append(tool_copy)

        # 记录历史
        self.acetylation_history.append({
            'total_tools': len(tool_schemas),
            'selected_tools': len(result),
            'task': current_task[:100],
            'timestamp': time.time(),
        })

        return result

    def acetylate_memories(
        self,
        current_task: str,
        max_memories: int = 20,
    ) -> List[int]:
        """记忆乙酰化: 激活与当前任务最相关的记忆节点。

        Args:
            current_task: 当前任务描述
            max_memories: 最大激活记忆数

        Returns:
            激活的记忆节点ID列表
        """
        mem_nodes = self.graph.get_nodes_by_type('observation')
        mem_nodes += self.graph.get_nodes_by_type('thought')

        if not mem_nodes:
            return []

        # 计算相关性并排序
        scored = []
        for node in mem_nodes:
            relevance = self._compute_text_relevance(node.content, current_task)
            scored.append((relevance, node))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 激活top-k
        activated = []
        for relevance, node in scored[:max_memories]:
            node.epigenetic_tag = min(1.0, node.epigenetic_tag + 0.3)
            activated.append(node.node_id)

        return activated

    # ---- Crossover ----

    def crossover_explore(
        self,
        base_context: List[ContextNode],
        num_variants: int = 3,
        mutation_rate: float = 0.3,
    ) -> List[List[ContextNode]]:
        """交叉重组: 生成多个上下文变体进行并行探索。

        当Agent遇到死锁（连续报错），触发并行交叉:
        生成多个变异的上下文，并行发送给LLM试探，
        选取未报错的路径遗传给下一代。

        Args:
            base_context: 基础上下文节点列表
            num_variants: 生成的变体数量
            mutation_rate: 变异率

        Returns:
            上下文变体列表
        """
        variants = []

        for i in range(num_variants):
            variant = self._mutate_context(base_context, mutation_rate, i)
            variants.append(variant)

        # 记录历史
        self.crossover_history.append({
            'num_variants': num_variants,
            'mutation_rate': mutation_rate,
            'timestamp': time.time(),
        })

        return variants

    def crossover_select(
        self,
        variants: List[List[ContextNode]],
        fitness_scores: List[float],
    ) -> Tuple[List[ContextNode], int]:
        """交叉选择: 从多个变体中选择最优的。

        Args:
            variants: 上下文变体列表
            fitness_scores: 对应的适应度分数

        Returns:
            (最优上下文, 最优变体索引)
        """
        if not variants or not fitness_scores:
            return [], -1

        best_idx = int(np.argmax(fitness_scores))

        # 将最优变体的配置遗传给下一代
        # (提高相关节点的活性权重)
        best_variant = variants[best_idx]
        for node in best_variant:
            if node.node_id in self.graph.nodes:
                self.graph.nodes[node.node_id].epigenetic_tag = min(
                    1.0,
                    self.graph.nodes[node.node_id].epigenetic_tag + 0.2,
                )

        return best_variant, best_idx

    # ---- Private Helpers ----

    def _auto_summarize(self, node_ids: List[int]) -> str:
        """自动生成节点摘要。"""
        nodes = [self.graph.nodes[nid] for nid in node_ids if nid in self.graph.nodes]
        if not nodes:
            return "No content."

        types = defaultdict(int)
        total_chars = 0
        for node in nodes:
            types[node.node_type] += 1
            total_chars += len(node.content)

        type_summary = ', '.join(f'{k}: {v}' for k, v in types.items())
        return (
            f"[Methylated Block] {len(nodes)} nodes ({type_summary}), "
            f"total {total_chars} chars compressed."
        )

    def _compute_tool_relevance(
        self, tool: Dict[str, Any], task: str
    ) -> float:
        """计算工具与任务的相关性分数。

        使用多层次匹配策略:
        1. 精确名称匹配
        2. 描述关键词匹配
        3. 参数名匹配
        """
        task_lower = task.lower()
        score = 0.0

        # 名称匹配
        tool_name = tool.get('name', '').lower()
        if tool_name and tool_name in task_lower:
            score += 0.5

        # 描述匹配
        description = tool.get('description', '').lower()
        if description:
            task_words = set(task_lower.split())
            desc_words = set(description.split())
            overlap = task_words & desc_words
            if overlap:
                score += 0.3 * min(1.0, len(overlap) / max(len(task_words), 1))

        # 参数匹配
        params = tool.get('parameters', {}).get('properties', {})
        if params:
            param_names = ' '.join(params.keys()).lower()
            param_words = set(param_names.split())
            task_words = set(task_lower.split())
            overlap = task_words & param_words
            if overlap:
                score += 0.2 * min(1.0, len(overlap) / max(len(task_words), 1))

        return min(1.0, score)

    def _compute_text_relevance(self, text: str, query: str) -> float:
        """计算文本与查询的相关性 (基于词汇重叠)。"""
        if not text or not query:
            return 0.0

        text_words = set(text.lower().split())
        query_words = set(query.lower().split())

        if not query_words:
            return 0.0

        overlap = text_words & query_words
        return len(overlap) / len(query_words)

    def _mutate_context(
        self,
        base: List[ContextNode],
        rate: float,
        seed: int,
    ) -> List[ContextNode]:
        """对上下文进行变异操作。

        变异策略:
        - 随机丢弃一些节点 (模拟甲基化)
        - 随机提升一些节点的权重 (模拟乙酰化)
        - 随机打乱部分节点顺序
        """
        rng = np.random.RandomState(seed)
        mutated = []

        for node in base:
            if rng.random() > rate:
                mutated.append(node)
            # 以rate概率丢弃节点

        # 随机激活一些被甲基化的节点
        inactive = [n for n in self.graph.nodes.values() if n.epigenetic_tag < 0.1]
        if inactive and rng.random() < rate:
            resurrect = rng.choice(inactive, size=min(3, len(inactive)), replace=False)
            for node in resurrect:
                mutated.append(node)

        return mutated


# ============================================================================
# Fitness Function
# ============================================================================

class FitnessFunction:
    """适应度函数

    评估上下文表达的质量，指导表观遗传调控。

    F(P) = α·R_task(P) - β·C_token(P) + γ·I_density(P)

    其中:
    - R_task: 任务推进奖励 (Task Reward)
    - C_token: Token消耗惩罚 (Cost Penalty)
    - I_density: 信息密度 (Information Density)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        token_baseline: int = 10000,
    ):
        """
        Args:
            alpha: 任务奖励权重
            beta: Token惩罚权重
            gamma: 信息密度权重
            token_baseline: Token归一化基准值
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.token_baseline = token_baseline
        self.evaluation_history: List[Dict[str, Any]] = []

    def evaluate(
        self,
        task_success: bool,
        token_count: int,
        effective_decisions: int,
        tool_call_success: bool = True,
        error_count: int = 0,
    ) -> float:
        """评估上下文表达的适应度。

        Args:
            task_success: 任务是否成功完成
            token_count: 消耗的Token数量
            effective_decisions: 有效决策数量
            tool_call_success: 工具调用是否成功
            error_count: 错误数量

        Returns:
            适应度分数 (越高越好)
        """
        # 任务推进奖励: 综合考虑任务成功和工具调用成功
        r_task = 0.0
        if task_success:
            r_task += 1.0
        if tool_call_success:
            r_task += 0.3
        r_task -= 0.1 * error_count  # 错误惩罚
        r_task = max(0.0, r_task)

        # Token消耗惩罚 (归一化)
        c_token = token_count / self.token_baseline

        # 信息密度: 有效决策占总Token的比例
        if token_count > 0:
            i_density = effective_decisions / max(token_count, 1) * 1000
        else:
            i_density = 0.0

        fitness = (
            self.alpha * r_task
            - self.beta * c_token
            + self.gamma * i_density
        )

        # 记录历史
        self.evaluation_history.append({
            'fitness': fitness,
            'r_task': r_task,
            'c_token': c_token,
            'i_density': i_density,
            'token_count': token_count,
            'task_success': task_success,
        })

        return fitness

    def evaluate_batch(
        self,
        results: List[Dict[str, Any]],
    ) -> List[float]:
        """批量评估多个结果。"""
        scores = []
        for r in results:
            score = self.evaluate(
                task_success=r.get('task_success', False),
                token_count=r.get('token_count', 0),
                effective_decisions=r.get('effective_decisions', 0),
                tool_call_success=r.get('tool_call_success', True),
                error_count=r.get('error_count', 0),
            )
            scores.append(score)
        return scores

    def get_stats(self) -> Dict[str, Any]:
        """获取适应度统计信息。"""
        if not self.evaluation_history:
            return {'mean': 0.0, 'std': 0.0, 'count': 0}

        scores = [h['fitness'] for h in self.evaluation_history]
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'count': len(scores),
        }


# ============================================================================
# EpiContext Router
# ============================================================================

class EpiContextRouter:
    """EpiContext Router - 表观遗传上下文路由器

    核心协调器，位于Agent客户端和LLM网关之间。
    负责在每个轮次中执行完整的表观遗传调控流水线:

    1. State Capture: 捕获新状态并加入Context Graph
    2. Epigenetic Masking: 生成表观掩码 (决定激活/沉默哪些节点)
    3. Payload Assembly: 根据掩码组装优化的Request Payload
    4. LLM Inference: 发送优化后的Request给LLM
    5. Fitness Evaluation: 评估结果并更新进化策略

    使用示例:
        router = EpiContextRouter()
        router.initialize_system("You are a helpful assistant.", tools)
        payload = router.process_turn(thought, action, observation, task)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        methylation_threshold: int = 20,
        max_active_nodes: int = 50,
        error_threshold: int = 3,
    ):
        """
        Args:
            alpha: 适应度函数 - 任务奖励权重
            beta: 适应度函数 - Token惩罚权重
            gamma: 适应度函数 - 信息密度权重
            methylation_threshold: 触发甲基化的活跃节点数阈值
            max_active_nodes: 最大活跃节点数
            error_threshold: 触发交叉重组的连续错误数
        """
        self.graph = ContextGraph()
        self.operators = EpigeneticOperators(self.graph)
        self.fitness = FitnessFunction(alpha, beta, gamma)

        self.methylation_threshold = methylation_threshold
        self.max_active_nodes = max_active_nodes
        self.error_threshold = error_threshold

        # 状态追踪
        self.current_turn: int = 0
        self.consecutive_errors: int = 0
        self.total_tokens_saved: int = 0
        self.turn_history: List[Dict[str, Any]] = []

        # 工具注册表
        self._tool_registry: List[Dict[str, Any]] = []

    # ---- Initialization ----

    def initialize_system(
        self,
        system_prompt: str,
        tools: List[Dict[str, Any]],
    ) -> None:
        """初始化系统配置。

        Args:
            system_prompt: 系统提示词
            tools: 可用工具列表
        """
        # 注册系统节点
        sys_id = self.graph.add_node(
            node_type='system',
            content=system_prompt,
            epigenetic_tag=1.0,
            metadata={'immutable': True},
        )

        # 注册工具节点
        for tool in tools:
            self.graph.add_node(
                node_type='tool_schema',
                content=json.dumps(tool),
                epigenetic_tag=1.0,
                metadata={'tool_name': tool.get('name', 'unknown')},
            )

        self._tool_registry = tools

    # ---- Main Pipeline ----

    def process_turn(
        self,
        thought: str,
        action: str,
        observation: str,
        current_task: str,
        action_success: bool = True,
    ) -> Dict[str, Any]:
        """处理单个Agent轮次的完整流水线。

        Args:
            thought: Agent的思考内容
            action: Agent执行的动作
            observation: 环境返回的观察
            current_task: 当前任务描述
            action_success: 动作是否成功执行

        Returns:
            包含优化后Payload和元数据的字典
        """
        self.current_turn += 1

        # === Step 1: State Capture ===
        thought_id = self.graph.add_node('thought', thought)
        action_id = self.graph.add_node('action', action)

        if action_success:
            obs_id = self.graph.add_node('observation', observation)
            self.consecutive_errors = 0
        else:
            obs_id = self.graph.add_node('error', observation)
            self.consecutive_errors += 1

        # 建立时间边
        self.graph.add_edge(thought_id, action_id, 'temporal')
        self.graph.add_edge(action_id, obs_id, 'temporal')

        # === Step 2: Epigenetic Masking ===
        mask = self._generate_epigenetic_mask(current_task)

        # === Step 3: Payload Assembly ===
        payload = self._assemble_payload(mask, current_task)

        # === Step 4: Fitness Evaluation ===
        token_count = self._estimate_tokens(payload)
        fitness_score = self.fitness.evaluate(
            task_success=action_success,
            token_count=token_count,
            effective_decisions=1 if action_success else 0,
            tool_call_success=action_success,
            error_count=self.consecutive_errors,
        )

        # === Step 5: Evolutionary Update ===
        self._evolutionary_update(fitness_score, mask)

        # 记录轮次
        turn_record = {
            'turn': self.current_turn,
            'thought': thought[:200],
            'action': action[:200],
            'observation': observation[:200],
            'action_success': action_success,
            'fitness': fitness_score,
            'token_count': token_count,
            'active_nodes': len(self.graph.get_active_nodes()),
            'mask': mask,
            'timestamp': time.time(),
        }
        self.turn_history.append(turn_record)

        return {
            'payload': payload,
            'fitness': fitness_score,
            'token_count': token_count,
            'mask': mask,
            'turn': self.current_turn,
        }

    # ---- Private: Pipeline Steps ----

    def _generate_epigenetic_mask(
        self, current_task: str
    ) -> Dict[str, Any]:
        """生成表观遗传掩码。

        决定哪些记忆和工具应该在当前轮次中被激活。

        Returns:
            掩码字典，包含:
            - active_tools: 激活的工具名列表
            - silenced_memory_ranges: 沉默的记忆范围
            - active_memory_ids: 激活的记忆节点ID
        """
        # 工具乙酰化
        active_tools = self.operators.acetylate_tools(
            self._tool_registry, current_task
        )
        active_tool_names = [t.get('name', '') for t in active_tools]

        # 记忆乙酰化
        active_memory_ids = self.operators.acetylate_memories(current_task)

        # 检查是否需要甲基化
        active_nodes = self.graph.get_active_nodes()
        silenced_ranges = []
        if len(active_nodes) > self.methylation_threshold:
            # 甲基化最老的节点
            old_nodes = [n.node_id for n in active_nodes[:-self.max_active_nodes]]
            if old_nodes:
                self.operators.methylate(old_nodes, strategy='full_silence')
                silenced_ranges.append({
                    'start': old_nodes[0],
                    'end': old_nodes[-1],
                    'count': len(old_nodes),
                })

        # 检查是否需要交叉重组
        if self.consecutive_errors >= self.error_threshold:
            active_list = self.graph.get_active_nodes()
            variants = self.operators.crossover_explore(active_list, num_variants=3)
            # 在实际实现中，这里会并行发送variants给LLM
            # 此处简化为记录

        return {
            'active_tools': active_tool_names,
            'silenced_memory_ranges': silenced_ranges,
            'active_memory_ids': active_memory_ids,
            'crossover_triggered': self.consecutive_errors >= self.error_threshold,
        }

    def _assemble_payload(
        self, mask: Dict[str, Any], current_task: str
    ) -> Dict[str, Any]:
        """根据掩码组装优化的Request Payload。

        Args:
            mask: 表观遗传掩码
            current_task: 当前任务

        Returns:
            组装好的Payload字典
        """
        # 获取活跃节点
        active_nodes = self.graph.get_active_nodes()

        # 按类型分组
        system_content = []
        memory_content = []
        tool_content = []

        for node in active_nodes:
            if node.node_type == 'system':
                system_content.append(node.content)
            elif node.node_type in ('tool_schema',):
                tool_name = node.metadata.get('tool_name', '')
                if tool_name in mask.get('active_tools', []):
                    tool_content.append(node.content)
            elif node.node_id in mask.get('active_memory_ids', []):
                memory_content.append({
                    'type': node.node_type,
                    'content': node.content,
                    'tag': node.epigenetic_tag,
                })

        # 组装
        payload = {
            'system': system_content,
            'task': current_task,
            'tools': mask.get('active_tools', []),
            'memory': memory_content[-20:],  # 最多保留20条记忆
            'metadata': {
                'turn': self.current_turn,
                'active_node_count': len(active_nodes),
                'mask_summary': {
                    'tools_active': len(mask.get('active_tools', [])),
                    'memories_active': len(mask.get('active_memory_ids', [])),
                    'silenced_ranges': len(mask.get('silenced_memory_ranges', [])),
                },
            },
        }

        return payload

    def _estimate_tokens(self, payload: Dict[str, Any]) -> int:
        """估算Payload的Token数量。

        使用简单的字符数/4估算 (适用于英文)。
        """
        text = json.dumps(payload, ensure_ascii=False)
        return len(text) // 4

    def _evolutionary_update(
        self, fitness_score: float, mask: Dict[str, Any]
    ) -> None:
        """进化更新: 根据适应度调整进化策略。

        高适应度 → 强化当前掩码策略
        低适应度 → 扩大激活范围 (降低甲基化程度)
        """
        if fitness_score > 0.5:
            # 高适应度: 强化当前策略
            # 提高活跃工具的权重
            for tool_name in mask.get('active_tools', []):
                for node in self.graph.get_nodes_by_type('tool_schema'):
                    if node.metadata.get('tool_name') == tool_name:
                        node.epigenetic_tag = min(1.0, node.epigenetic_tag + 0.1)
        else:
            # 低适应度: 扩大激活范围
            # 激活更多记忆节点
            inactive_mem = [
                n for n in self.graph.nodes.values()
                if n.node_type in ('observation', 'thought')
                and n.epigenetic_tag < 0.3
            ]
            for node in inactive_mem[:5]:
                node.epigenetic_tag = min(1.0, node.epigenetic_tag + 0.3)

    # ---- Statistics & Serialization ----

    def get_stats(self) -> Dict[str, Any]:
        """获取路由器统计信息。"""
        active = self.graph.get_active_nodes()
        return {
            'total_nodes': len(self.graph.nodes),
            'active_nodes': len(active),
            'methylated_nodes': len(self.graph.nodes) - len(active),
            'total_edges': len(self.graph.edges),
            'total_turns': self.current_turn,
            'consecutive_errors': self.consecutive_errors,
            'fitness_stats': self.fitness.get_stats(),
            'methylation_count': len(self.operators.methylation_history),
            'acetylation_count': len(self.operators.acetylation_history),
            'crossover_count': len(self.operators.crossover_history),
        }

    def save_state(self, filepath: str) -> None:
        """保存路由器状态到文件。"""
        state = {
            'graph': self.graph.to_dict(),
            'current_turn': self.current_turn,
            'consecutive_errors': self.consecutive_errors,
            'turn_history': self.turn_history,
            'fitness_history': self.fitness.evaluation_history,
            'methylation_history': self.operators.methylation_history,
            'acetylation_history': self.operators.acetylation_history,
            'crossover_history': self.operators.crossover_history,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def load_state(cls, filepath: str) -> 'EpiContextRouter':
        """从文件加载路由器状态。"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)

        router = cls()
        router.graph = ContextGraph.from_dict(state['graph'])
        router.current_turn = state.get('current_turn', 0)
        router.consecutive_errors = state.get('consecutive_errors', 0)
        router.turn_history = state.get('turn_history', [])
        router.fitness.evaluation_history = state.get('fitness_history', [])
        router.operators.methylation_history = state.get('methylation_history', [])
        router.operators.acetylation_history = state.get('acetylation_history', [])
        router.operators.crossover_history = state.get('crossover_history', [])

        return router


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("EpiContext Core - Self Test")
    print("=" * 60)

    # 创建路由器
    router = EpiContextRouter(
        alpha=1.0, beta=0.5, gamma=0.3,
        methylation_threshold=15,
        max_active_nodes=10,
    )

    # 初始化系统
    router.initialize_system(
        system_prompt="You are a helpful coding assistant.",
        tools=[
            {'name': 'read_file', 'description': 'Read a file from disk'},
            {'name': 'write_file', 'description': 'Write content to a file'},
            {'name': 'search_code', 'description': 'Search codebase for patterns'},
            {'name': 'execute_command', 'description': 'Run a shell command'},
            {'name': 'web_search', 'description': 'Search the web'},
        ],
    )

    # 模拟多轮交互
    tasks = [
        "Fix the bug in utils.py",
        "Add a new feature to the API",
        "Refactor the database module",
    ]

    for task_idx, task in enumerate(tasks):
        print(f"\n--- Task {task_idx + 1}: {task} ---")

        for step in range(8):
            success = step < 6 or (task_idx == 0 and step == 7)

            result = router.process_turn(
                thought=f"Step {step}: Analyzing {task}",
                action=f"Running tool for step {step}",
                observation=f"Result of step {step}: {'OK' if success else 'ERROR'}",
                current_task=task,
                action_success=success,
            )

            if step % 3 == 0:
                print(f"  Turn {result['turn']}: "
                      f"fitness={result['fitness']:.3f}, "
                      f"tokens={result['token_count']}, "
                      f"active_tools={len(result['mask']['active_tools'])}")

    # 打印统计
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)
    stats = router.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✅ EpiContext Core self-test completed successfully!")