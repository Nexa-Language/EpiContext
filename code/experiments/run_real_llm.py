"""
EpiContext Real LLM Experiment Runner

使用真实LLM API (mimo-v2.5-pro) 运行完整实验流水线。
包括: 真实LLM推理、表观遗传调控、多基准评估、统计检验。
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epicontext.core import (
    ContextGraph, ContextNode, EpigeneticOperators,
    FitnessFunction, EpiContextRouter,
)
from epicontext.benchmarks.environments import (
    create_webarena_tasks, create_swebench_tasks,
    create_alfworld_tasks, create_agentbench_tasks,
    create_environment_factory, get_tools_for_type,
)

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============================================================================
# LLM Client
# ============================================================================

class LLMClient:
    """真实LLM API客户端。"""

    def __init__(self):
        if not HAS_OPENAI:
            raise ImportError("openai package required: pip install openai")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        base_url = os.environ.get("OPENAI_API_BASE")
        if not base_url:
            raise ValueError("OPENAI_API_BASE environment variable is required")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = os.environ.get("OPENAI_MODEL_NAME", "mimo-v2.5-pro")
        self.call_count = 0
        self.total_tokens_used = 0

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 512,
             temperature: float = 0.3) -> Tuple[str, int]:
        """发送chat completion请求。"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            self.call_count += 1
            tokens = response.usage.total_tokens if response.usage else 0
            self.total_tokens_used += tokens
            return response.choices[0].message.content or "", tokens
        except Exception as e:
            print(f"  [LLM Error] {e}")
            return f"Error: {e}", 0

    def generate_thought(self, task: str, turn: int, history: List[str],
                         tools: List[str]) -> Tuple[str, int]:
        """生成Agent思考。"""
        history_str = "\n".join(history[-5:]) if history else "(no history)"
        tools_str = ", ".join(tools[:10])

        prompt = f"""You are an AI agent executing a task. Analyze the situation and decide what to do next.

TASK: {task}
TURN: {turn}
AVAILABLE TOOLS: {tools_str}
RECENT HISTORY:
{history_str}

Output your reasoning in 1-2 sentences. Be specific about which tool to use and why."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, max_tokens=256, temperature=0.3)

    def generate_action(self, task: str, thought: str,
                        tools: List[str]) -> Tuple[str, int]:
        """生成Agent动作。"""
        tools_str = ", ".join(tools[:10])

        prompt = f"""Based on your analysis, select the best action.

TASK: {task}
YOUR ANALYSIS: {thought}
AVAILABLE TOOLS: {tools_str}

Output ONLY the action to execute (one line). Use format: tool_name('arguments')"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, max_tokens=128, temperature=0.2)

    def generate_epigenetic_mask(self, task: str, active_nodes_summary: str,
                                 tools: List[str]) -> Tuple[str, int]:
        """生成表观遗传掩码 - 决定激活/沉默哪些上下文。"""
        tools_str = "\n".join(f"- {t}" for t in tools[:20])

        prompt = f"""You are an epigenetic context regulator. Decide which tools and memories to activate.

TASK: {task}
AVAILABLE TOOLS:
{tools_str}

RECENT CONTEXT SUMMARY:
{active_nodes_summary[:2000]}

Output a JSON object with:
{{"active_tools": ["tool1", "tool2"], "silence_older_than_turn": N, "strategy": "focused"|"exploratory"}}

Keep active_tools to 3-5 most relevant tools. Set silence_older_than_turn to drop old context."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, max_tokens=256, temperature=0.2)

    def summarize_context(self, nodes_content: str) -> Tuple[str, int]:
        """摘要化上下文（甲基化）。"""
        prompt = f"""Summarize the following agent interaction history into 2-3 concise sentences.
Focus on: key decisions made, errors encountered and resolved, current state.

HISTORY:
{nodes_content[:3000]}

SUMMARY:"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, max_tokens=200, temperature=0.2)


# ============================================================================
# Real LLM Agent
# ============================================================================

@dataclass
class RealTurnResult:
    """真实LLM轮次结果。"""
    turn: int
    thought: str
    action: str
    observation: str
    success: bool
    fitness: float
    llm_tokens: int
    context_tokens: int
    elapsed: float


@dataclass
class RealTaskResult:
    """真实LLM任务结果。"""
    task_id: str
    task_description: str
    success: bool
    total_turns: int
    total_llm_tokens: int
    total_context_tokens: int
    total_time: float
    average_fitness: float
    turn_results: List[RealTurnResult] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class RealEpiContextAgent:
    """使用真实LLM API的EpiContext Agent。"""

    def __init__(self, llm: LLMClient, max_turns: int = 30,
                 methylation_threshold: int = 15,
                 alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.3):
        self.llm = llm
        self.max_turns = max_turns
        self.router = EpiContextRouter(
            alpha=alpha, beta=beta, gamma=gamma,
            methylation_threshold=methylation_threshold,
            max_active_nodes=40,
            error_threshold=3,
        )
        self._initialized = False

    def initialize(self, system_prompt: str, tools: List[Dict[str, Any]]):
        self.router.initialize_system(system_prompt, tools)
        self._initialized = True

    def run_task(self, task_id: str, task_description: str,
                 environment) -> RealTaskResult:
        """运行单个任务，使用真实LLM。"""
        if not self._initialized:
            raise RuntimeError("Agent not initialized")

        turn_results: List[RealTurnResult] = []
        task_start = time.time()
        total_llm_tokens = 0
        total_context_tokens = 0

        # 重置环境
        if hasattr(environment, 'reset'):
            environment.reset()

        for turn in range(1, self.max_turns + 1):
            turn_start = time.time()

            # 获取工具列表
            tools = get_tools_for_type(
                self._infer_task_type(task_description)
            )
            tool_names = [t['name'] for t in tools]

            # 获取活跃上下文摘要
            active_nodes = self.router.graph.get_active_nodes()
            context_summary = "\n".join(
                f"[{n.node_type}] {n.content[:150]}" for n in active_nodes[-10:]
            )

            # Step 1: LLM生成思考
            history = [r.observation for r in turn_results[-5:]]
            thought, thought_tokens = self.llm.generate_thought(
                task_description, turn, history, tool_names
            )
            total_llm_tokens += thought_tokens

            # Step 2: LLM生成动作
            action, action_tokens = self.llm.generate_action(
                task_description, thought, tool_names
            )
            total_llm_tokens += action_tokens

            # Step 3: 环境交互
            if hasattr(environment, 'step'):
                observation, success = environment.step(action)
            else:
                observation, success = environment(action)

            # Step 4: 表观遗传调控
            mask_tokens = 0
            if turn % 5 == 0 or not success:
                mask_str, mask_tokens = self.llm.generate_epigenetic_mask(
                    task_description, context_summary, tool_names
                )
                total_llm_tokens += mask_tokens
                try:
                    mask = json.loads(mask_str)
                except json.JSONDecodeError:
                    mask = {"active_tools": tool_names[:5], "silence_older_than_turn": turn - 10}

                # 应用甲基化
                if mask.get("silence_older_than_turn", 0) > 0:
                    old_nodes = [
                        n.node_id for n in active_nodes
                        if n.node_id < mask["silence_older_than_turn"]
                    ]
                    if old_nodes:
                        # 生成摘要
                        old_content = "\n".join(
                            self.router.graph.nodes[nid].content[:200]
                            for nid in old_nodes[-10:]
                            if nid in self.router.graph.nodes
                        )
                        summary, sum_tokens = self.llm.summarize_context(old_content)
                        total_llm_tokens += sum_tokens
                        self.router.operators.methylate(old_nodes, summary)

            # Step 5: 通过Router处理
            result = self.router.process_turn(
                thought=thought,
                action=action,
                observation=observation,
                current_task=task_description,
                action_success=success,
            )

            context_tokens = result['token_count']
            total_context_tokens += context_tokens
            elapsed = time.time() - turn_start

            turn_result = RealTurnResult(
                turn=turn, thought=thought, action=action,
                observation=observation, success=success,
                fitness=result['fitness'],
                llm_tokens=thought_tokens + action_tokens + mask_tokens,
                context_tokens=context_tokens,
                elapsed=elapsed,
            )
            turn_results.append(turn_result)

            # 终止检查
            if self._check_done(turn_results):
                break

        total_time = time.time() - task_start
        avg_fitness = (
            np.mean([r.fitness for r in turn_results])
            if turn_results else 0.0
        )

        return RealTaskResult(
            task_id=task_id,
            task_description=task_description,
            success=turn_results[-1].success if turn_results else False,
            total_turns=len(turn_results),
            total_llm_tokens=total_llm_tokens,
            total_context_tokens=total_context_tokens,
            total_time=total_time,
            average_fitness=avg_fitness,
            turn_results=turn_results,
            stats=self.router.get_stats(),
        )

    def _check_done(self, results: List[RealTurnResult]) -> bool:
        if len(results) < 3:
            return False
        if all(r.success for r in results[-3:]):
            return True
        if len(results) >= 8 and not any(r.success for r in results[-5:]):
            return True
        return False

    def _infer_task_type(self, task: str) -> str:
        t = task.lower()
        if any(w in t for w in ['web', 'browse', 'navigate', 'page']):
            return 'webarena'
        if any(w in t for w in ['code', 'bug', 'fix', 'repo', 'test']):
            return 'swebench'
        if any(w in t for w in ['find', 'put', 'take', 'room', 'kitchen']):
            return 'alfworld'
        return 'agentbench'


# ============================================================================
# Baseline Agent (Real LLM)
# ============================================================================

class RealBaselineAgent:
    """使用真实LLM的基线Agent。"""

    def __init__(self, name: str, llm: LLMClient, max_turns: int = 30):
        self.name = name
        self.llm = llm
        self.max_turns = max_turns

    def run_task(self, task_id: str, task_description: str,
                 environment, tools: List[Dict[str, Any]]) -> RealTaskResult:
        turn_results = []
        task_start = time.time()
        total_llm_tokens = 0
        total_context_tokens = 0
        tool_names = [t['name'] for t in tools]

        if hasattr(environment, 'reset'):
            environment.reset()

        for turn in range(1, self.max_turns + 1):
            turn_start = time.time()

            history = [r.observation for r in turn_results[-5:]]

            # 根据基线类型调整上下文
            if self.name == 'Full-Context':
                context_history = [r.observation for r in turn_results]
            elif self.name == 'MemGPT':
                context_history = history[-3:] if len(history) > 3 else history
            else:
                context_history = history

            thought, tt = self.llm.generate_thought(
                task_description, turn, context_history, tool_names
            )
            total_llm_tokens += tt

            action, at = self.llm.generate_action(
                task_description, thought, tool_names
            )
            total_llm_tokens += at

            if hasattr(environment, 'step'):
                observation, success = environment.step(action)
            else:
                observation, success = environment(action)

            ctx_tokens = len(thought) // 4 + len(action) // 4 + len(observation) // 4
            total_context_tokens += ctx_tokens

            elapsed = time.time() - turn_start
            turn_results.append(RealTurnResult(
                turn=turn, thought=thought, action=action,
                observation=observation, success=success,
                fitness=float(success),
                llm_tokens=tt + at, context_tokens=ctx_tokens,
                elapsed=elapsed,
            ))

            if len(turn_results) >= 3 and all(r.success for r in turn_results[-3:]):
                break
            if len(turn_results) >= 8 and not any(r.success for r in turn_results[-5:]):
                break

        total_time = time.time() - task_start
        return RealTaskResult(
            task_id=task_id, task_description=task_description,
            success=turn_results[-1].success if turn_results else False,
            total_turns=len(turn_results),
            total_llm_tokens=total_llm_tokens,
            total_context_tokens=total_context_tokens,
            total_time=total_time,
            average_fitness=np.mean([r.fitness for r in turn_results]) if turn_results else 0.0,
            turn_results=turn_results,
        )


# ============================================================================
# Experiment Runner
# ============================================================================

@dataclass
class RealExperimentConfig:
    num_tasks: int = 20
    num_repetitions: int = 3
    max_turns: int = 30
    methylation_threshold: int = 15
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.3
    output_dir: str = 'results'
    seed: int = 42
    parallel_workers: int = 2


class RealExperimentRunner:
    """真实LLM实验运行器。"""

    def __init__(self, config: RealExperimentConfig):
        self.config = config
        self.llm = LLMClient()
        self.rng = np.random.RandomState(config.seed)

    def run_all(self) -> Dict[str, Any]:
        """运行所有实验。"""
        print("=" * 70)
        print("EpiContext Real LLM Experiment Suite")
        print(f"Model: {self.llm.model}")
        print(f"Tasks/benchmark: {self.config.num_tasks}")
        print(f"Repetitions: {self.config.num_repetitions}")
        print("=" * 70)

        all_results: Dict[str, Any] = {
            'config': {
                'num_tasks': self.config.num_tasks,
                'num_repetitions': self.config.num_repetitions,
                'max_turns': self.config.max_turns,
                'model': self.llm.model,
                'alpha': self.config.alpha,
                'beta': self.config.beta,
                'gamma': self.config.gamma,
                'seed': self.config.seed,
            },
            'benchmarks': {},
            'ablations': {},
            'llm_stats': {},
        }

        benchmarks = [
            ('webarena', create_webarena_tasks),
            ('swebench', create_swebench_tasks),
            ('alfworld', create_alfworld_tasks),
            ('agentbench', create_agentbench_tasks),
        ]

        methods = ['EpiContext', 'Full-Context', 'ReAct', 'MemGPT']

        total_start = time.time()

        for bench_name, task_creator in benchmarks:
            print(f"\n{'='*50}")
            print(f"Benchmark: {bench_name.upper()}")
            print(f"{'='*50}")

            tasks = task_creator(self.config.num_tasks, self.config.seed)
            tools = get_tools_for_type(bench_name)
            env_factory = create_environment_factory(bench_name, tasks, self.config.seed)

            bench_results: Dict[str, List[RealTaskResult]] = {}

            for method in methods:
                print(f"\n  Running {method}...")
                method_results = []

                for rep in range(self.config.num_repetitions):
                    rep_seed = self.config.seed + rep

                    if method == 'EpiContext':
                        agent = RealEpiContextAgent(
                            self.llm, self.config.max_turns,
                            self.config.methylation_threshold,
                            self.config.alpha, self.config.beta, self.config.gamma,
                        )
                        agent.initialize(
                            f"You are an AI agent performing {bench_name} tasks.",
                            tools,
                        )
                    else:
                        agent = RealBaselineAgent(method, self.llm, self.config.max_turns)

                    for task in tasks:
                        env = env_factory(task.task_id)
                        try:
                            if method == 'EpiContext':
                                result = agent.run_task(
                                    task.task_id, task.description, env
                                )
                            else:
                                result = agent.run_task(
                                    task.task_id, task.description, env, tools
                                )
                            method_results.append(result)

                            if len(method_results) % 5 == 0:
                                recent = method_results[-5:]
                                sr = sum(1 for r in recent if r.success) / len(recent)
                                print(f"    [{len(method_results)}/{self.config.num_tasks * self.config.num_repetitions}] "
                                      f"recent SR: {sr:.2f}, "
                                      f"LLM calls: {self.llm.call_count}, "
                                      f"tokens: {self.llm.total_tokens_used}")
                        except Exception as e:
                            print(f"    [ERROR] Task {task.task_id}: {e}")

                bench_results[method] = method_results

                # 打印中期统计
                if method_results:
                    sr = sum(1 for r in method_results if r.success) / len(method_results)
                    avg_t = np.mean([r.total_llm_tokens for r in method_results])
                    avg_time = np.mean([r.total_time for r in method_results])
                    print(f"  {method} done: SR={sr:.3f}, "
                          f"avg_LLM_tokens={avg_t:.0f}, "
                          f"avg_time={avg_time:.1f}s")

            all_results['benchmarks'][bench_name] = self._serialize_results(bench_results)

        # 消融实验 (仅在webarena上)
        print(f"\n{'='*50}")
        print("Ablation Study (WebArena)")
        print(f"{'='*50}")

        ablation_variants = {
            'Full': {'methylation': True, 'acetylation': True, 'fitness': True},
            'w/o_Methylation': {'methylation': False, 'acetylation': True, 'fitness': True},
            'w/o_Acetylation': {'methylation': True, 'acetylation': False, 'fitness': True},
            'w/o_Fitness': {'methylation': True, 'acetylation': True, 'fitness': False},
        }

        wa_tasks = create_webarena_tasks(10, self.config.seed)
        wa_tools = get_tools_for_type('webarena')
        wa_env_factory = create_environment_factory('webarena', wa_tasks, self.config.seed)

        ablation_results: Dict[str, List[RealTaskResult]] = {}

        for variant_name, components in ablation_variants.items():
            print(f"\n  Ablation: {variant_name}")
            mt = self.config.methylation_threshold if components['methylation'] else 99999
            a = self.config.alpha if components['fitness'] else 0.0
            b = self.config.beta if components['fitness'] else 0.0
            g = self.config.gamma if components['fitness'] else 0.0

            agent = RealEpiContextAgent(
                self.llm, self.config.max_turns, mt, a, b, g
            )
            agent.initialize(f"Ablation agent ({variant_name})", wa_tools)

            variant_results = []
            for task in wa_tasks:
                env = wa_env_factory(task.task_id)
                try:
                    result = agent.run_task(task.task_id, task.description, env)
                    variant_results.append(result)
                except Exception as e:
                    print(f"    [ERROR] {task.task_id}: {e}")

            ablation_results[variant_name] = variant_results
            if variant_results:
                sr = sum(1 for r in variant_results if r.success) / len(variant_results)
                print(f"  {variant_name}: SR={sr:.3f}")

        all_results['ablations'] = self._serialize_results(ablation_results)

        # LLM统计
        all_results['llm_stats'] = {
            'total_calls': self.llm.call_count,
            'total_tokens': self.llm.total_tokens_used,
            'total_runtime': time.time() - total_start,
        }

        return all_results

    def _serialize_results(self, results: Dict[str, List[RealTaskResult]]) -> Dict[str, Any]:
        output = {}
        for method, task_results in results.items():
            output[method] = []
            for r in task_results:
                output[method].append({
                    'task_id': r.task_id,
                    'success': r.success,
                    'total_turns': r.total_turns,
                    'total_llm_tokens': r.total_llm_tokens,
                    'total_context_tokens': r.total_context_tokens,
                    'total_time': r.total_time,
                    'average_fitness': r.average_fitness,
                    'turns': [
                        {
                            'turn': tr.turn,
                            'success': tr.success,
                            'fitness': tr.fitness,
                            'llm_tokens': tr.llm_tokens,
                            'context_tokens': tr.context_tokens,
                            'elapsed': tr.elapsed,
                        }
                        for tr in r.turn_results
                    ],
                })
        return output

    def save_results(self, results: Dict[str, Any]):
        os.makedirs(self.config.output_dir, exist_ok=True)
        filepath = os.path.join(self.config.output_dir, 'real_experiment_results.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {filepath}")

    def print_summary(self, results: Dict[str, Any]):
        print("\n" + "=" * 70)
        print("EXPERIMENT SUMMARY")
        print("=" * 70)

        for bench_name, bench_data in results['benchmarks'].items():
            print(f"\n{bench_name.upper()}:")
            print(f"{'Method':<20} {'Success':>8} {'LLM Tokens':>12} {'Ctx Tokens':>12} {'Time(s)':>10}")
            print("-" * 65)
            for method, task_results in bench_data.items():
                if task_results:
                    sr = sum(1 for r in task_results if r['success']) / len(task_results)
                    avg_llm = np.mean([r['total_llm_tokens'] for r in task_results])
                    avg_ctx = np.mean([r['total_context_tokens'] for r in task_results])
                    avg_time = np.mean([r['total_time'] for r in task_results])
                    print(f"{method:<20} {sr:>8.3f} {avg_llm:>12.0f} {avg_ctx:>12.0f} {avg_time:>10.1f}")

        stats = results['llm_stats']
        print(f"\nTotal LLM calls: {stats['total_calls']}")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Total runtime: {stats['total_runtime']:.1f}s ({stats['total_runtime']/3600:.1f}h)")


def main():
    config = RealExperimentConfig(
        num_tasks=20,
        num_repetitions=3,
        max_turns=30,
        methylation_threshold=15,
        alpha=1.0, beta=0.5, gamma=0.3,
        output_dir='results',
        seed=42,
    )

    runner = RealExperimentRunner(config)
    results = runner.run_all()
    runner.save_results(results)
    runner.print_summary(results)


if __name__ == '__main__':
    main()