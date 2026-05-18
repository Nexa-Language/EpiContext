"""EpiContext Harbor Agent 主类与单轮记录。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .context_graph import ContextGraph
from .llm_client import LLMClient
from .strategies import (
    AdaptiveEpiContextStrategy,
    ContextStrategy,
    EpiContextStrategy,
    FullContextStrategy,
    SlidingWindowStrategy,
)


@dataclass
class TurnRecord:
    """单轮交互记录。"""
    turn: int
    thought: str
    action: str
    observation: str
    reward: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class EpiContextAgent(BaseAgent):
    """基于 EpiContext 的 Harbor Agent。

    使用表观遗传算子动态调控上下文，在真实 Harbor 任务上运行。
    """

    # 可配置参数 (通过 kwargs 传入)
    DEFAULT_MAX_TURNS: int = 20
    DEFAULT_STRATEGY: str = "epicontext"  # "full", "sliding", "epicontext", "adaptive"
    DEFAULT_WINDOW_SIZE: int = 5
    DEFAULT_SWITCH_TURN: int = 10  # 自适应策略切换点
    DEFAULT_ALPHA: float = 1.0
    DEFAULT_BETA: float = 2.0  # 增大 token 效率权重
    DEFAULT_GAMMA: float = 0.3
    DEFAULT_SILENCE_THRESHOLD: float = 1e-3  # 更激进的沉默
    DEFAULT_RELEVANCE_THRESHOLD: float = 0.5  # 更高的相关性要求

    @staticmethod
    def name() -> str:
        return "epicontext-agent"

    def __init__(self, logs_dir: Path, model_name: str | None = None, **kwargs):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._max_turns = int(kwargs.get("max_turns", self.DEFAULT_MAX_TURNS))
        self._strategy_name = kwargs.get("strategy", self.DEFAULT_STRATEGY)
        self._window_size = int(kwargs.get("window_size", self.DEFAULT_WINDOW_SIZE))
        self._switch_turn = int(kwargs.get("switch_turn", self.DEFAULT_SWITCH_TURN))
        self._alpha = float(kwargs.get("alpha", self.DEFAULT_ALPHA))
        self._beta = float(kwargs.get("beta", self.DEFAULT_BETA))
        self._gamma = float(kwargs.get("gamma", self.DEFAULT_GAMMA))
        self._silence_threshold = float(kwargs.get("silence_threshold", self.DEFAULT_SILENCE_THRESHOLD))
        self._relevance_threshold = float(kwargs.get("relevance_threshold", self.DEFAULT_RELEVANCE_THRESHOLD))

        self._llm: Optional[LLMClient] = None
        self._graph: Optional[ContextGraph] = None
        self._strategy: Optional[ContextStrategy] = None
        self._turns: List[TurnRecord] = []
        self._tokenizer = None

    def version(self) -> str:
        return "2.0.0"

    def _estimate_tokens(self, text: str) -> int:
        """使用 tiktoken 精确估算 token 数。"""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._tokenizer = "fallback"
        if self._tokenizer == "fallback":
            return len(text) // 4
        return len(self._tokenizer.encode(text))

    async def setup(self, environment: BaseEnvironment) -> None:
        """安装 agent 所需工具。"""
        self._llm = LLMClient()
        self._graph = ContextGraph()

        if self._strategy_name == "full":
            self._strategy = FullContextStrategy()
        elif self._strategy_name == "sliding":
            self._strategy = SlidingWindowStrategy(self._window_size)
        elif self._strategy_name == "adaptive":
            self._strategy = AdaptiveEpiContextStrategy(
                switch_turn=self._switch_turn, window_size=self._window_size,
                alpha=self._alpha, beta=self._beta, gamma=self._gamma,
                silence_threshold=self._silence_threshold,
                relevance_threshold=self._relevance_threshold,
            )
        else:
            self._strategy = EpiContextStrategy(
                alpha=self._alpha, beta=self._beta, gamma=self._gamma,
                silence_threshold=self._silence_threshold,
                relevance_threshold=self._relevance_threshold,
            )

        # 记录 agent 配置
        config_path = self.logs_dir / "epicontext_config.json"
        config_path.write_text(json.dumps({
            "agent": self.name(),
            "version": self.version(),
            "strategy": self._strategy_name,
            "max_turns": self._max_turns,
            "window_size": self._window_size,
            "switch_turn": self._switch_turn,
            "alpha": self._alpha,
            "beta": self._beta,
            "gamma": self._gamma,
            "silence_threshold": self._silence_threshold,
            "relevance_threshold": self._relevance_threshold,
            "model": self._llm.model if self._llm else "unknown",
            "tokenizer": "tiktoken" if self._tokenizer != "fallback" else "char_estimation",
        }, indent=2))

    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        """运行 agent 主循环。"""
        if self._llm is None or self._graph is None or self._strategy is None:
            raise RuntimeError("Agent not set up. Call setup() first.")

        start_time = time.time()

        for turn in range(self._max_turns):
            # 1. 选择上下文
            selected_context = self._strategy.select_context(
                self._graph, turn, max_context_tokens=4000
            )

            # 2. 生成思考+动作 (合并为单次 LLM 调用)
            thought, action, llm_in, llm_out = self._generate_turn(
                instruction, turn, selected_context
            )

            # 3. 执行动作
            observation, reward = await self._execute_action(action, environment)

            # 4. 更新上下文图谱
            loss_delta = reward
            grad_norm = 1.0 if reward > 0 else 0.1

            self._graph.add_node(turn, "thought", thought,
                                 loss_delta=loss_delta, grad_norm=grad_norm,
                                 token_count=llm_in + llm_out)
            self._graph.add_node(turn, "action", action,
                                 loss_delta=0, grad_norm=0,
                                 token_count=0)
            self._graph.add_node(turn, "observation", observation,
                                 loss_delta=loss_delta, grad_norm=grad_norm,
                                 token_count=self._estimate_tokens(observation))

            # 5. 记录
            record = TurnRecord(
                turn=turn, thought=thought, action=action,
                observation=observation, reward=reward,
                input_tokens=llm_in,
                output_tokens=llm_out,
            )
            self._turns.append(record)

            # 6. 检查终止
            if self._check_done():
                break

        elapsed = time.time() - start_time

        # 写入结果
        self._write_results(instruction, elapsed, context)

    def _generate_turn(self, task: str, turn: int,
                       context_str: str) -> Tuple[str, str, int, int]:
        """生成单轮思考+动作 (合并为一次 LLM 调用)。"""
        prompt = f"""You are an AI agent executing a task in a Linux terminal environment.

TASK: {task}
CURRENT TURN: {turn}

RECENT CONTEXT:
{context_str if context_str else "(no prior context)"}

Output your response in exactly this format:
THOUGHT: <1-2 sentences analyzing what to do next>
ACTION: <the exact shell command to execute>

Examples:
THOUGHT: I need to create a file with specific content. I'll use echo with redirection.
ACTION: echo "Hello, world!" > /app/hello.txt

THOUGHT: I should verify the file was created correctly by checking its contents.
ACTION: cat /app/hello.txt"""
        messages = [{"role": "user", "content": prompt}]
        content, in_tok, out_tok = self._llm.chat(messages, max_tokens=256, temperature=0.3)

        # 解析 THOUGHT 和 ACTION
        thought = ""
        action = ""
        for line in content.split("\n"):
            if line.startswith("THOUGHT:"):
                thought = line.replace("THOUGHT:", "").strip()
            elif line.startswith("ACTION:"):
                action = line.replace("ACTION:", "").strip()

        if not action:
            # fallback: 尝试从内容中提取命令
            action = content.strip()

        return thought or content[:100], action, in_tok, out_tok

    async def _execute_action(self, action: str,
                              environment: BaseEnvironment) -> Tuple[str, float]:
        """执行动作并返回观察和奖励。"""
        # 清理动作 (移除 markdown 代码块标记)
        clean_action = action.strip()
        if clean_action.startswith("```"):
            lines = clean_action.split("\n")
            clean_action = "\n".join(lines[1:-1]) if len(lines) > 2 else clean_action
        clean_action = clean_action.replace("`", "").strip()

        try:
            result = await environment.exec(clean_action)
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            return_code = result.return_code

            if return_code == 0 and stdout:
                observation = stdout[:1000]
                reward = 0.5  # 成功执行
            elif return_code == 0:
                observation = "(command executed successfully, no output)"
                reward = 0.3
            else:
                observation = f"Error (return code {return_code}): {stderr[:500]}"
                reward = -0.1  # 失败
        except Exception as e:
            observation = f"Execution error: {str(e)[:500]}"
            reward = -0.2

        return observation, reward

    def _check_done(self) -> bool:
        """检查任务是否完成。"""
        if len(self._turns) < 2:
            return False
        # 连续 3 轮无进展 (reward <= 0)
        recent_rewards = [t.reward for t in self._turns[-3:]]
        if len(recent_rewards) >= 3 and all(r <= 0 for r in recent_rewards):
            return True
        # Agent 自己声明任务完成
        last_thought = self._turns[-1].thought.lower()
        completion_phrases = [
            "task completed", "task already completed", "task has been completed",
            "no further action", "nothing more to do", "done", "finished",
            "successfully completed", "already exists"
        ]
        if any(phrase in last_thought for phrase in completion_phrases):
            return True
        # 连续 2 轮执行相同命令 (验证循环)
        if len(self._turns) >= 2:
            last_actions = [t.action.strip() for t in self._turns[-2:]]
            if last_actions[0] == last_actions[1]:
                return True
        return False

    def _write_results(self, instruction: str, elapsed: float,
                       context: AgentContext) -> None:
        """写入实验结果。"""
        graph_summary = self._graph.to_summary() if self._graph else {}

        results = {
            "agent": self.name(),
            "version": self.version(),
            "strategy": self._strategy_name,
            "model": self._llm.model if self._llm else "unknown",
            "instruction": instruction[:200],
            "total_turns": len(self._turns),
            "elapsed_sec": round(elapsed, 2),
            "total_input_tokens": self._llm.total_input_tokens if self._llm else 0,
            "total_output_tokens": self._llm.total_output_tokens if self._llm else 0,
            "total_llm_calls": self._llm.call_count if self._llm else 0,
            "graph_summary": graph_summary,
            "turns": [
                {
                    "turn": t.turn,
                    "thought": t.thought[:200],
                    "action": t.action[:200],
                    "observation": t.observation[:200],
                    "reward": t.reward,
                    "input_tokens": t.input_tokens,
                    "output_tokens": t.output_tokens,
                }
                for t in self._turns
            ],
        }

        result_path = self.logs_dir / "epicontext_result.json"
        result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

        # 同时写入 Harbor 期望的 reward 格式
        final_reward = sum(t.reward for t in self._turns) / max(len(self._turns), 1)
        reward_path = self.logs_dir / "reward.txt"
        reward_path.write_text(str(round(final_reward, 4)))
