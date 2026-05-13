"""
EpiContext Agent Implementation

基于EpiContext框架的Agent实现，支持:
- ReAct风格的Thought-Action-Observation循环
- 表观遗传上下文动态演化
- 多轮交互的适应度追踪
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from epicontext.core import EpiContextRouter


@dataclass
class AgentConfig:
    """Agent配置。"""
    max_turns: int = 50
    methylation_threshold: int = 20
    max_active_nodes: int = 50
    error_threshold: int = 3
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.3
    verbose: bool = False


@dataclass
class TurnResult:
    """单轮交互结果。"""
    turn: int
    thought: str
    action: str
    observation: str
    success: bool
    fitness: float
    token_count: int
    elapsed: float


@dataclass
class TaskResult:
    """任务执行结果。"""
    task_id: str
    task_description: str
    success: bool
    total_turns: int
    total_tokens: int
    total_time: float
    average_fitness: float
    turn_results: List[TurnResult] = field(default_factory=list)
    final_stats: Dict[str, Any] = field(default_factory=dict)


class EpiContextAgent:
    """基于EpiContext的Agent。

    封装了EpiContextRouter，提供高层Agent接口。
    支持ReAct风格的交互循环。

    使用示例:
        agent = EpiContextAgent(config)
        agent.initialize(system_prompt, tools)
        result = agent.run_task(task_id, task_description, environment)
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.router = EpiContextRouter(
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
            methylation_threshold=self.config.methylation_threshold,
            max_active_nodes=self.config.max_active_nodes,
            error_threshold=self.config.error_threshold,
        )
        self._initialized = False

    def initialize(
        self,
        system_prompt: str,
        tools: List[Dict[str, Any]],
    ) -> None:
        """初始化Agent。

        Args:
            system_prompt: 系统提示词
            tools: 可用工具列表
        """
        self.router.initialize_system(system_prompt, tools)
        self._initialized = True

    def run_task(
        self,
        task_id: str,
        task_description: str,
        environment: Union[Callable[[str], Tuple[str, bool]], Any],
        thought_generator: Optional[Callable[[Dict[str, Any]], str]] = None,
        action_generator: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> TaskResult:
        """运行一个完整的任务。

        Args:
            task_id: 任务标识符
            task_description: 任务描述
            environment: 环境交互函数(接收action返回(observation, success))
                         或具有.step(action)方法的BenchmarkEnvironment对象
            thought_generator: 可选的思考生成器 (默认使用简单模板)
            action_generator: 可选的动作生成器 (默认使用简单模板)

        Returns:
            TaskResult包含完整的任务执行结果
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        turn_results: List[TurnResult] = []
        task_start = time.time()

        # 判断environment类型
        if hasattr(environment, 'step') and callable(environment.step):
            env_step = environment.step
        elif callable(environment):
            env_step = environment
        else:
            raise TypeError(
                f"environment must be callable or have .step() method, "
                f"got {type(environment)}"
            )

        for turn in range(1, self.config.max_turns + 1):
            turn_start = time.time()

            # 生成思考 (在实际实现中由LLM生成)
            if thought_generator:
                thought = thought_generator({
                    'task': task_description,
                    'turn': turn,
                    'history': [r.observation for r in turn_results[-5:]],
                })
            else:
                thought = self._default_thought(task_description, turn)

            # 生成动作 (在实际实现中由LLM生成)
            if action_generator:
                action = action_generator({
                    'task': task_description,
                    'turn': turn,
                    'thought': thought,
                })
            else:
                action = self._default_action(task_description, turn)

            # 与环境交互
            observation, success = env_step(action)

            # 通过EpiContext Router处理
            result = self.router.process_turn(
                thought=thought,
                action=action,
                observation=observation,
                current_task=task_description,
                action_success=success,
            )

            elapsed = time.time() - turn_start

            turn_result = TurnResult(
                turn=turn,
                thought=thought,
                action=action,
                observation=observation,
                success=success,
                fitness=result['fitness'],
                token_count=result['token_count'],
                elapsed=elapsed,
            )
            turn_results.append(turn_result)

            if self.config.verbose:
                status = "✅" if success else "❌"
                print(f"  [{status}] Turn {turn}: "
                      f"fitness={result['fitness']:.3f}, "
                      f"tokens={result['token_count']}")

            # 检查终止条件
            if self._check_termination(task_description, turn_results):
                break

        total_time = time.time() - task_start
        total_tokens = sum(r.token_count for r in turn_results)
        avg_fitness = (
            sum(r.fitness for r in turn_results) / len(turn_results)
            if turn_results else 0.0
        )

        return TaskResult(
            task_id=task_id,
            task_description=task_description,
            success=turn_results[-1].success if turn_results else False,
            total_turns=len(turn_results),
            total_tokens=total_tokens,
            total_time=total_time,
            average_fitness=avg_fitness,
            turn_results=turn_results,
            final_stats=self.router.get_stats(),
        )

    def run_benchmark(
        self,
        tasks: List[Dict[str, Any]],
        environment_factory: Callable[[str], Callable[[str], Tuple[str, bool]]],
    ) -> List[TaskResult]:
        """运行一组基准任务。

        Args:
            tasks: 任务列表，每个包含 'id' 和 'description'
            environment_factory: 环境工厂函数

        Returns:
            任务结果列表
        """
        results = []
        for i, task in enumerate(tasks):
            if self.config.verbose:
                print(f"\n{'='*50}")
                print(f"Task {i+1}/{len(tasks)}: {task['id']}")
                print(f"{'='*50}")

            env = environment_factory(task['id'])
            result = self.run_task(
                task_id=task['id'],
                task_description=task['description'],
                environment=env,
            )
            results.append(result)

            if self.config.verbose:
                status = "✅ SUCCESS" if result.success else "❌ FAILED"
                print(f"  {status} | Turns: {result.total_turns} | "
                      f"Tokens: {result.total_tokens} | "
                      f"Time: {result.total_time:.1f}s")

        return results

    def _default_thought(self, task: str, turn: int) -> str:
        """默认思考生成器。"""
        return (
            f"[Turn {turn}] Analyzing task: {task}. "
            f"Considering available tools and previous observations."
        )

    def _default_action(self, task: str, turn: int) -> str:
        """默认动作生成器。"""
        return f"execute_step_{turn}('{task[:30]}...')"

    def _check_termination(
        self,
        task: str,
        turn_results: List[TurnResult],
    ) -> bool:
        """检查任务是否应该终止。

        终止条件:
        1. 最近3轮全部成功 → 任务完成
        2. 连续5轮失败 → 任务失败
        """
        if len(turn_results) < 3:
            return False

        recent = turn_results[-3:]
        if all(r.success for r in recent):
            return True

        if len(turn_results) >= 5:
            recent5 = turn_results[-5:]
            if not any(r.success for r in recent5):
                return True

        return False

    def get_summary(self) -> Dict[str, Any]:
        """获取Agent运行摘要。"""
        return {
            'config': {
                'max_turns': self.config.max_turns,
                'methylation_threshold': self.config.methylation_threshold,
                'alpha': self.config.alpha,
                'beta': self.config.beta,
                'gamma': self.config.gamma,
            },
            'stats': self.router.get_stats(),
        }

    def reset(self) -> None:
        """重置Agent状态。"""
        self.router = EpiContextRouter(
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
            methylation_threshold=self.config.methylation_threshold,
            max_active_nodes=self.config.max_active_nodes,
            error_threshold=self.config.error_threshold,
        )
        self._initialized = False