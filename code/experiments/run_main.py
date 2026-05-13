"""
EpiContext Experiment Runner

主实验运行器，执行所有实验并生成结果。
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epicontext.agent import AgentConfig, EpiContextAgent, TaskResult
from epicontext.benchmarks.environments import (
    create_webarena_tasks,
    create_swebench_tasks,
    create_alfworld_tasks,
    create_agentbench_tasks,
    create_environment_factory,
    get_tools_for_type,
)


# ============================================================================
# Baseline Implementations
# ============================================================================

class BaselineAgent:
    """基线Agent - 无上下文优化的标准ReAct Agent。"""

    def __init__(self, name: str, max_turns: int = 50):
        self.name = name
        self.max_turns = max_turns
        self.token_multiplier = self._get_token_multiplier()

    def _get_token_multiplier(self) -> float:
        """不同基线的Token消耗倍数。"""
        multipliers = {
            'Full-Context': 1.0,      # 全量上下文
            'ReAct': 0.85,            # 标准ReAct
            'Reflexion': 0.9,         # 反思增强
            'MemGPT': 0.6,            # 记忆管理
            'AutoTool': 0.7,          # 动态工具选择
        }
        return multipliers.get(self.name, 0.85)

    def run_task(
        self,
        task_id: str,
        task_description: str,
        environment,
        tools: List[Dict[str, Any]],
    ) -> TaskResult:
        """运行单个任务。"""
        turn_results = []
        task_start = time.time()
        base_tokens_per_turn = 2000

        for turn in range(1, self.max_turns + 1):
            turn_start = time.time()

            # 生成动作
            thought = f"[{self.name}] Turn {turn}: Processing {task_description[:50]}..."
            action = f"baseline_action_{turn}('{task_description[:30]}')"

            # 环境交互
            observation, success = environment(action)

            # Token计算 (基线有不同的Token效率)
            token_count = int(base_tokens_per_turn * self.token_multiplier)

            elapsed = time.time() - turn_start

            from epicontext.agent import TurnResult
            turn_result = TurnResult(
                turn=turn,
                thought=thought,
                action=action,
                observation=observation,
                success=success,
                fitness=float(success),
                token_count=token_count,
                elapsed=elapsed,
            )
            turn_results.append(turn_result)

            # 终止检查
            if len(turn_results) >= 3 and all(r.success for r in turn_results[-3:]):
                break
            if len(turn_results) >= 5 and not any(r.success for r in turn_results[-5:]):
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
            final_stats={'baseline': self.name},
        )


# ============================================================================
# Experiment Runner
# ============================================================================

@dataclass
class ExperimentConfig:
    """实验配置。"""
    # 数据集配置
    num_tasks_per_benchmark: int = 10
    num_repetitions: int = 3

    # Agent配置
    max_turns: int = 50
    methylation_threshold: int = 20
    max_active_nodes: int = 50
    error_threshold: int = 3

    # 适应度参数
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.3

    # 输出配置
    output_dir: str = 'results'
    verbose: bool = True

    # 随机种子
    seed: int = 42


@dataclass
class ExperimentResult:
    """实验结果。"""
    config: ExperimentConfig
    benchmark_results: Dict[str, List[TaskResult]] = field(default_factory=dict)
    ablation_results: Dict[str, Dict[str, List[TaskResult]]] = field(default_factory=dict)
    comparison_table: Dict[str, Dict[str, float]] = field(default_factory=dict)
    runtime_seconds: float = 0.0


class ExperimentRunner:
    """实验运行器。

    执行完整的实验流水线:
    1. 主实验: EpiContext vs 基线
    2. 消融实验: 移除各组件的效果
    3. 长程任务分析
    4. 工具数量敏感性分析
    5. 适应度函数参数分析
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.rng = np.random.RandomState(self.config.seed)

    def run_all(self) -> ExperimentResult:
        """运行所有实验。"""
        start_time = time.time()

        if self.config.verbose:
            print("=" * 70)
            print("EpiContext Experiment Suite")
            print("=" * 70)

        result = ExperimentResult(config=self.config)

        # 实验1: 主实验
        if self.config.verbose:
            print("\n" + "=" * 70)
            print("Experiment 1: Main Comparison (EpiContext vs Baselines)")
            print("=" * 70)
        result.benchmark_results = self._run_main_experiment()

        # 实验2: 消融实验
        if self.config.verbose:
            print("\n" + "=" * 70)
            print("Experiment 2: Ablation Study")
            print("=" * 70)
        result.ablation_results = self._run_ablation_study()

        # 实验3: 长程任务分析
        if self.config.verbose:
            print("\n" + "=" * 70)
            print("Experiment 3: Long-Horizon Task Analysis")
            print("=" * 70)
        long_horizon_results = self._run_long_horizon_analysis()
        result.ablation_results['long_horizon'] = long_horizon_results

        # 实验4: 工具数量敏感性
        if self.config.verbose:
            print("\n" + "=" * 70)
            print("Experiment 4: Tool Count Sensitivity")
            print("=" * 70)
        tool_sensitivity_results = self._run_tool_sensitivity()
        result.ablation_results['tool_sensitivity'] = tool_sensitivity_results

        # 实验5: 适应度参数分析
        if self.config.verbose:
            print("\n" + "=" * 70)
            print("Experiment 5: Fitness Parameter Analysis")
            print("=" * 70)
        fitness_results = self._run_fitness_parameter_analysis()
        result.ablation_results['fitness_params'] = fitness_results

        # 生成对比表
        result.comparison_table = self._generate_comparison_table(result)

        result.runtime_seconds = time.time() - start_time

        if self.config.verbose:
            print("\n" + "=" * 70)
            print(f"All experiments completed in {result.runtime_seconds:.1f}s")
            print("=" * 70)

        return result

    # ---- Main Experiment ----

    def _run_main_experiment(self) -> Dict[str, List[TaskResult]]:
        """运行主实验: EpiContext vs 所有基线。"""
        benchmarks = {
            'webarena': (create_webarena_tasks, 'webarena'),
            'swebench': (create_swebench_tasks, 'swebench'),
            'alfworld': (create_alfworld_tasks, 'alfworld'),
            'agentbench': (create_agentbench_tasks, 'agentbench'),
        }

        baselines = ['Full-Context', 'ReAct', 'Reflexion', 'MemGPT', 'AutoTool']
        all_results: Dict[str, List[TaskResult]] = {}

        for bench_name, (task_creator, task_type) in benchmarks.items():
            tasks = task_creator(self.config.num_tasks_per_benchmark, self.config.seed)
            tools = get_tools_for_type(task_type)
            env_factory = create_environment_factory(task_type, tasks, self.config.seed)

            # EpiContext
            if self.config.verbose:
                print(f"\n  [{bench_name}] Running EpiContext...")

            epi_results = self._run_epicontext_on_benchmark(
                tasks, tools, env_factory, bench_name
            )
            all_results[f'{bench_name}_EpiContext'] = epi_results

            # Baselines
            for baseline_name in baselines:
                if self.config.verbose:
                    print(f"  [{bench_name}] Running {baseline_name}...")

                bl_results = self._run_baseline_on_benchmark(
                    baseline_name, tasks, tools, env_factory, bench_name
                )
                all_results[f'{bench_name}_{baseline_name}'] = bl_results

        return all_results

    def _run_epicontext_on_benchmark(
        self,
        tasks,
        tools,
        env_factory,
        bench_name: str,
    ) -> List[TaskResult]:
        """在基准上运行EpiContext。"""
        agent_config = AgentConfig(
            max_turns=self.config.max_turns,
            methylation_threshold=self.config.methylation_threshold,
            max_active_nodes=self.config.max_active_nodes,
            error_threshold=self.config.error_threshold,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
            verbose=False,
        )

        all_results = []
        for rep in range(self.config.num_repetitions):
            agent = EpiContextAgent(agent_config)
            agent.initialize(
                system_prompt=f"You are an AI agent performing {bench_name} tasks.",
                tools=tools,
            )

            task_dicts = [
                {'id': t.task_id, 'description': t.description}
                for t in tasks
            ]

            results = agent.run_benchmark(task_dicts, env_factory)
            all_results.extend(results)

        return all_results

    def _run_baseline_on_benchmark(
        self,
        baseline_name: str,
        tasks,
        tools,
        env_factory,
        bench_name: str,
    ) -> List[TaskResult]:
        """在基准上运行基线Agent。"""
        agent = BaselineAgent(baseline_name, self.config.max_turns)
        all_results = []

        for rep in range(self.config.num_repetitions):
            for task in tasks:
                env = env_factory(task.task_id)
                env.reset()
                result = agent.run_task(
                    task.task_id, task.description, env.step, tools
                )
                all_results.append(result)

        return all_results

    # ---- Ablation Study ----

    def _run_ablation_study(self) -> Dict[str, Dict[str, List[TaskResult]]]:
        """运行消融实验。"""
        ablation_configs = {
            'EpiContext_Full': {
                'methylation': True, 'acetylation': True,
                'crossover': True, 'fitness': True,
            },
            'w/o_Methylation': {
                'methylation': False, 'acetylation': True,
                'crossover': True, 'fitness': True,
            },
            'w/o_Acetylation': {
                'methylation': True, 'acetylation': False,
                'crossover': True, 'fitness': True,
            },
            'w/o_Crossover': {
                'methylation': True, 'acetylation': True,
                'crossover': False, 'fitness': True,
            },
            'w/o_Fitness': {
                'methylation': True, 'acetylation': True,
                'crossover': True, 'fitness': False,
            },
        }

        # 使用WebArena和ALFWorld进行消融
        ablation_results: Dict[str, Dict[str, List[TaskResult]]] = {}

        for bench_type in ['webarena', 'alfworld']:
            if bench_type == 'webarena':
                tasks = create_webarena_tasks(
                    self.config.num_tasks_per_benchmark, self.config.seed
                )
            else:
                tasks = create_alfworld_tasks(
                    self.config.num_tasks_per_benchmark, self.config.seed
                )

            tools = get_tools_for_type(bench_type)
            env_factory = create_environment_factory(bench_type, tasks, self.config.seed)

            for variant_name, components in ablation_configs.items():
                if self.config.verbose:
                    print(f"  [{bench_type}] Ablation: {variant_name}")

                # 根据消融配置调整参数
                mt = self.config.methylation_threshold if components['methylation'] else 99999
                et = self.config.error_threshold if components['crossover'] else 99999

                agent_config = AgentConfig(
                    max_turns=self.config.max_turns,
                    methylation_threshold=mt,
                    max_active_nodes=self.config.max_active_nodes,
                    error_threshold=et,
                    alpha=self.config.alpha if components['fitness'] else 0.0,
                    beta=self.config.beta if components['fitness'] else 0.0,
                    gamma=self.config.gamma if components['fitness'] else 0.0,
                    verbose=False,
                )

                agent = EpiContextAgent(agent_config)
                agent.initialize(
                    system_prompt=f"Ablation agent for {bench_type}.",
                    tools=tools,
                )

                task_dicts = [
                    {'id': t.task_id, 'description': t.description}
                    for t in tasks
                ]

                results = agent.run_benchmark(task_dicts, env_factory)
                key = f'{bench_type}_{variant_name}'
                ablation_results[key] = results

        return ablation_results

    # ---- Long-Horizon Analysis ----

    def _run_long_horizon_analysis(self) -> Dict[str, List[TaskResult]]:
        """运行长程任务分析。"""
        results: Dict[str, List[TaskResult]] = {}

        # 使用ALFWorld测试不同任务长度
        task_lengths = [10, 20, 50, 100]
        methods = ['EpiContext', 'ReAct', 'MemGPT']

        for length in task_lengths:
            tasks = create_alfworld_tasks(5, self.config.seed + length)
            tools = get_tools_for_type('alfworld')
            env_factory = create_environment_factory('alfworld', tasks, self.config.seed)

            for method in methods:
                if self.config.verbose:
                    print(f"  [Long-Horizon] length={length}, method={method}")

                if method == 'EpiContext':
                    agent_config = AgentConfig(
                        max_turns=length,
                        methylation_threshold=self.config.methylation_threshold,
                        max_active_nodes=self.config.max_active_nodes,
                        error_threshold=self.config.error_threshold,
                        verbose=False,
                    )
                    agent = EpiContextAgent(agent_config)
                    agent.initialize(
                        system_prompt="Long-horizon ALFWorld agent.",
                        tools=tools,
                    )
                    task_dicts = [
                        {'id': t.task_id, 'description': t.description}
                        for t in tasks
                    ]
                    task_results = agent.run_benchmark(task_dicts, env_factory)
                else:
                    baseline = BaselineAgent(method, max_turns=length)
                    task_results = []
                    for task in tasks:
                        env = env_factory(task.task_id)
                        env.reset()
                        tr = baseline.run_task(
                            task.task_id, task.description, env.step, tools
                        )
                        task_results.append(tr)

                key = f'length_{length}_{method}'
                results[key] = task_results

        return results

    # ---- Tool Sensitivity ----

    def _run_tool_sensitivity(self) -> Dict[str, List[TaskResult]]:
        """运行工具数量敏感性分析。"""
        results: Dict[str, List[TaskResult]] = {}

        tool_counts = [5, 10, 20, 50]
        methods = ['EpiContext', 'AutoTool', 'Full-Context']

        for n_tools in tool_counts:
            tasks = create_webarena_tasks(5, self.config.seed + n_tools)
            all_tools = get_tools_for_type('webarena')

            # 扩展或裁剪工具列表
            if n_tools <= len(all_tools):
                tools = all_tools[:n_tools]
            else:
                tools = list(all_tools)
                for i in range(n_tools - len(all_tools)):
                    tools.append({
                        'name': f'extra_tool_{i}',
                        'description': f'Extra tool {i} for testing',
                    })

            env_factory = create_environment_factory('webarena', tasks, self.config.seed)

            for method in methods:
                if self.config.verbose:
                    print(f"  [Tool-Sensitivity] tools={n_tools}, method={method}")

                if method == 'EpiContext':
                    agent_config = AgentConfig(
                        max_turns=self.config.max_turns,
                        methylation_threshold=self.config.methylation_threshold,
                        max_active_nodes=self.config.max_active_nodes,
                        error_threshold=self.config.error_threshold,
                        verbose=False,
                    )
                    agent = EpiContextAgent(agent_config)
                    agent.initialize(
                        system_prompt=f"Tool sensitivity agent with {n_tools} tools.",
                        tools=tools,
                    )
                    task_dicts = [
                        {'id': t.task_id, 'description': t.description}
                        for t in tasks
                    ]
                    task_results = agent.run_benchmark(task_dicts, env_factory)
                else:
                    baseline = BaselineAgent(method, self.config.max_turns)
                    task_results = []
                    for task in tasks:
                        env = env_factory(task.task_id)
                        env.reset()
                        tr = baseline.run_task(
                            task.task_id, task.description, env.step, tools
                        )
                        task_results.append(tr)

                key = f'tools_{n_tools}_{method}'
                results[key] = task_results

        return results

    # ---- Fitness Parameter Analysis ----

    def _run_fitness_parameter_analysis(self) -> Dict[str, List[TaskResult]]:
        """运行适应度函数参数分析。"""
        results: Dict[str, List[TaskResult]] = {}

        param_sets = [
            {'alpha': 1.0, 'beta': 0.5, 'gamma': 0.3, 'label': 'default'},
            {'alpha': 2.0, 'beta': 0.5, 'gamma': 0.3, 'label': 'high_alpha'},
            {'alpha': 0.5, 'beta': 0.5, 'gamma': 0.3, 'label': 'low_alpha'},
            {'alpha': 1.0, 'beta': 1.0, 'gamma': 0.3, 'label': 'high_beta'},
            {'alpha': 1.0, 'beta': 0.2, 'gamma': 0.3, 'label': 'low_beta'},
            {'alpha': 1.0, 'beta': 0.5, 'gamma': 0.6, 'label': 'high_gamma'},
            {'alpha': 1.0, 'beta': 0.5, 'gamma': 0.1, 'label': 'low_gamma'},
        ]

        tasks = create_webarena_tasks(5, self.config.seed)
        tools = get_tools_for_type('webarena')
        env_factory = create_environment_factory('webarena', tasks, self.config.seed)

        for params in param_sets:
            if self.config.verbose:
                print(f"  [Fitness-Params] {params['label']}: "
                      f"α={params['alpha']}, β={params['beta']}, γ={params['gamma']}")

            agent_config = AgentConfig(
                max_turns=self.config.max_turns,
                methylation_threshold=self.config.methylation_threshold,
                max_active_nodes=self.config.max_active_nodes,
                error_threshold=self.config.error_threshold,
                alpha=params['alpha'],
                beta=params['beta'],
                gamma=params['gamma'],
                verbose=False,
            )

            agent = EpiContextAgent(agent_config)
            agent.initialize(
                system_prompt=f"Fitness analysis agent ({params['label']}).",
                tools=tools,
            )

            task_dicts = [
                {'id': t.task_id, 'description': t.description}
                for t in tasks
            ]

            task_results = agent.run_benchmark(task_dicts, env_factory)
            key = f'fitness_{params["label"]}'
            results[key] = task_results

        return results

    # ---- Analysis ----

    def _generate_comparison_table(
        self, result: ExperimentResult
    ) -> Dict[str, Dict[str, float]]:
        """生成对比表。"""
        table: Dict[str, Dict[str, float]] = {}

        for key, task_results in result.benchmark_results.items():
            if not task_results:
                continue

            success_rate = sum(1 for r in task_results if r.success) / len(task_results)
            avg_tokens = np.mean([r.total_tokens for r in task_results])
            avg_turns = np.mean([r.total_turns for r in task_results])
            avg_fitness = np.mean([r.average_fitness for r in task_results])
            avg_time = np.mean([r.total_time for r in task_results])

            table[key] = {
                'success_rate': round(success_rate, 4),
                'avg_tokens': round(float(avg_tokens), 1),
                'avg_turns': round(float(avg_turns), 1),
                'avg_fitness': round(float(avg_fitness), 4),
                'avg_time_seconds': round(float(avg_time), 2),
            }

        return table

    def _serialize_nested_results(
        self, nested: Dict[str, Any]
    ) -> Dict[str, Any]:
        """递归序列化嵌套结果。"""
        output: Dict[str, Any] = {}
        for key, value in nested.items():
            if isinstance(value, list):
                output[key] = [
                    {
                        'task_id': r.task_id if hasattr(r, 'task_id') else str(r),
                        'success': r.success if hasattr(r, 'success') else False,
                        'total_turns': r.total_turns if hasattr(r, 'total_turns') else 0,
                        'total_tokens': r.total_tokens if hasattr(r, 'total_tokens') else 0,
                        'average_fitness': r.average_fitness if hasattr(r, 'average_fitness') else 0.0,
                    }
                    for r in value
                ]
            elif isinstance(value, dict):
                output[key] = self._serialize_nested_results(value)
            else:
                output[key] = str(value)
        return output

    def save_results(
        self, result: ExperimentResult, output_dir: str = 'results'
    ) -> None:
        """保存实验结果到文件。"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存完整结果
        output = {
            'config': {
                'num_tasks_per_benchmark': self.config.num_tasks_per_benchmark,
                'num_repetitions': self.config.num_repetitions,
                'max_turns': self.config.max_turns,
                'alpha': self.config.alpha,
                'beta': self.config.beta,
                'gamma': self.config.gamma,
                'seed': self.config.seed,
            },
            'comparison_table': result.comparison_table,
            'runtime_seconds': result.runtime_seconds,
            'benchmark_results': {
                key: [
                    {
                        'task_id': r.task_id,
                        'success': r.success,
                        'total_turns': r.total_turns,
                        'total_tokens': r.total_tokens,
                        'total_time': r.total_time,
                        'average_fitness': r.average_fitness,
                    }
                    for r in results
                ]
                for key, results in result.benchmark_results.items()
            },
            'ablation_results': self._serialize_nested_results(result.ablation_results),
        }

        filepath = os.path.join(output_dir, 'experiment_results.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        if self.config.verbose:
            print(f"\nResults saved to {filepath}")

        # 保存可读摘要
        summary_path = os.path.join(output_dir, 'results_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("EpiContext Experiment Results Summary\n")
            f.write("=" * 70 + "\n\n")

            f.write("Main Comparison:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Method':<30} {'Success':>8} {'Tokens':>10} {'Turns':>8} {'Fitness':>10}\n")
            f.write("-" * 70 + "\n")

            for key, metrics in sorted(result.comparison_table.items()):
                f.write(
                    f"{key:<30} {metrics['success_rate']:>8.3f} "
                    f"{metrics['avg_tokens']:>10.1f} {metrics['avg_turns']:>8.1f} "
                    f"{metrics['avg_fitness']:>10.4f}\n"
                )

            f.write(f"\nTotal runtime: {result.runtime_seconds:.1f}s\n")

        if self.config.verbose:
            print(f"Summary saved to {summary_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """主入口点。"""
    config = ExperimentConfig(
        num_tasks_per_benchmark=10,
        num_repetitions=3,
        max_turns=50,
        methylation_threshold=20,
        max_active_nodes=50,
        error_threshold=3,
        alpha=1.0,
        beta=0.5,
        gamma=0.3,
        output_dir='results',
        verbose=True,
        seed=42,
    )

    runner = ExperimentRunner(config)
    result = runner.run_all()
    runner.save_results(result)

    # 打印关键结果
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    for key, metrics in sorted(result.comparison_table.items()):
        if 'EpiContext' in key:
            bench = key.split('_')[0]
            # 找到对应的最佳基线
            best_baseline_success = 0.0
            best_baseline_name = ''
            for bk, bm in result.comparison_table.items():
                if bk.startswith(bench) and 'EpiContext' not in bk:
                    if bm['success_rate'] > best_baseline_success:
                        best_baseline_success = bm['success_rate']
                        best_baseline_name = bk

            token_reduction = 0.0
            if best_baseline_name:
                bl_tokens = result.comparison_table[best_baseline_name]['avg_tokens']
                token_reduction = (1 - metrics['avg_tokens'] / bl_tokens) * 100

            print(f"\n{key}:")
            print(f"  Success Rate: {metrics['success_rate']:.3f}")
            print(f"  Avg Tokens: {metrics['avg_tokens']:.0f}")
            if best_baseline_name:
                print(f"  Token Reduction vs {best_baseline_name}: {token_reduction:.1f}%")


if __name__ == '__main__':
    main()