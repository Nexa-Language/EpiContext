"""Harbor agents backed by the core EpiContextRouter evolution modes."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from epicontext.core import EpiContextRouter

from .llm_client import LLMClient


@dataclass
class RouterTurnRecord:
    turn: int
    thought: str
    action: str
    observation: str
    command_reward: float
    fitness: float
    router_tokens: int
    input_tokens: int
    output_tokens: int
    elapsed_sec: float


class RouterBackedEpiContextAgent(BaseAgent):
    """Use the canonical EpiContextRouter inside a Harbor agent loop."""

    EVOLUTION_MODE = "logistic"
    DEFAULT_MAX_TURNS = 12

    @staticmethod
    def name() -> str:
        return "router-backed-epicontext"

    def version(self) -> str:
        return "3.0.0"

    def __init__(self, logs_dir: Path, model_name: str | None = None, **kwargs):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._max_turns = int(os.environ.get("EPICONTEXT_MAX_TURNS", self.DEFAULT_MAX_TURNS))
        self._llm: Optional[LLMClient] = None
        self._router: Optional[EpiContextRouter] = None
        self._turns: List[RouterTurnRecord] = []
        self._last_payload: Optional[Dict[str, Any]] = None

    async def setup(self, environment: BaseEnvironment) -> None:
        self._llm = LLMClient()
        self._router = EpiContextRouter(
            alpha=float(os.environ.get("EPICONTEXT_ALPHA", "1.0")),
            beta=float(os.environ.get("EPICONTEXT_BETA", "0.5")),
            gamma=float(os.environ.get("EPICONTEXT_GAMMA", "0.3")),
            methylation_threshold=int(os.environ.get("EPICONTEXT_METHYLATION_THRESHOLD", "15")),
            max_active_nodes=int(os.environ.get("EPICONTEXT_MAX_ACTIVE_NODES", "40")),
            error_threshold=int(os.environ.get("EPICONTEXT_ERROR_THRESHOLD", "3")),
            evolution_mode=self.EVOLUTION_MODE,
            logistic_r=float(os.environ.get("EPICONTEXT_LOGISTIC_R", "0.25")),
            fitness_midpoint=float(os.environ.get("EPICONTEXT_FITNESS_MIDPOINT", "0.5")),
            fitness_slope=float(os.environ.get("EPICONTEXT_FITNESS_SLOPE", "4.0")),
        )
        self._router.initialize_system(
            system_prompt="You are a terminal agent. Use shell commands to complete tasks.",
            tools=[
                {
                    "name": "shell",
                    "description": "Execute one Linux shell command in the task environment.",
                }
            ],
        )
        (self.logs_dir / "epicontext_config.json").write_text(
            json.dumps(
                {
                    "agent": self.name(),
                    "version": self.version(),
                    "evolution_mode": self.EVOLUTION_MODE,
                    "max_turns": self._max_turns,
                    "model": self._llm.model if self._llm else "unknown",
                },
                indent=2,
            )
        )

    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        if self._llm is None or self._router is None:
            raise RuntimeError("Agent not set up. Call setup() first.")

        start_time = time.time()
        for turn in range(1, self._max_turns + 1):
            turn_start = time.time()
            context_text = self._payload_to_context(self._last_payload)
            thought, action, input_tokens, output_tokens = self._generate_turn(
                instruction, turn, context_text
            )
            observation, command_reward, command_success = await self._execute_action(
                action, environment
            )
            router_result = self._router.process_turn(
                thought=thought,
                action=action,
                observation=observation,
                current_task=instruction,
                action_success=command_success,
            )
            self._last_payload = router_result.get("payload")

            self._turns.append(
                RouterTurnRecord(
                    turn=turn,
                    thought=thought,
                    action=action,
                    observation=observation,
                    command_reward=command_reward,
                    fitness=float(router_result.get("fitness", 0.0)),
                    router_tokens=int(router_result.get("token_count", 0)),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    elapsed_sec=time.time() - turn_start,
                )
            )
            if self._check_done():
                break

        self._write_results(instruction, time.time() - start_time, context)

    def _payload_to_context(self, payload: Optional[Dict[str, Any]]) -> str:
        if not payload:
            return "(no prior context)"
        memories = payload.get("memory", [])
        lines = []
        for item in memories[-12:]:
            role = item.get("type", "memory")
            content = str(item.get("content", ""))[:500]
            lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "(no selected memory)"

    def _generate_turn(self, task: str, turn: int,
                       context_text: str) -> Tuple[str, str, int, int]:
        prompt = f"""You are an AI agent executing a task in a Linux terminal environment.

TASK: {task}
CURRENT TURN: {turn}

SELECTED CONTEXT:
{context_text}

Return exactly:
THOUGHT: <1-2 sentences>
ACTION: <one shell command>

Use one command only. Do not wrap the command in markdown."""
        content, input_tokens, output_tokens = self._llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.2,
        )
        thought = ""
        action = ""
        for line in content.splitlines():
            if line.startswith("THOUGHT:"):
                thought = line.replace("THOUGHT:", "", 1).strip()
            elif line.startswith("ACTION:"):
                action = line.replace("ACTION:", "", 1).strip()
        if not action:
            action = content.strip()
        return thought or content[:120], action, input_tokens, output_tokens

    async def _execute_action(self, action: str,
                              environment: BaseEnvironment) -> Tuple[str, float, bool]:
        clean_action = action.strip()
        if clean_action.startswith("```"):
            lines = clean_action.splitlines()
            clean_action = "\n".join(lines[1:-1]) if len(lines) > 2 else clean_action
        clean_action = clean_action.replace("`", "").strip()
        try:
            result = await environment.exec(clean_action)
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            if result.return_code == 0:
                observation = stdout[:1200] if stdout else "(command executed successfully, no output)"
                return observation, 1.0, True
            observation = f"Error (return code {result.return_code}): {stderr[:800]}"
            return observation, -0.25, False
        except Exception as exc:
            return f"Execution error: {str(exc)[:800]}", -0.5, False

    def _check_done(self) -> bool:
        if os.environ.get("EPICONTEXT_DISABLE_EARLY_STOP", "").lower() in {"1", "true", "yes"}:
            return False
        if len(self._turns) < 3:
            return False
        recent_rewards = [t.command_reward for t in self._turns[-3:]]
        if all(r > 0 for r in recent_rewards):
            return True
        if len(self._turns) >= 5 and all(r <= 0 for r in recent_rewards):
            return True
        last = self._turns[-1].thought.lower()
        done_markers = (
            "task completed",
            "successfully completed",
            "nothing more to do",
            "no further action",
            "done",
            "finished",
        )
        return any(marker in last for marker in done_markers)

    def _write_results(self, instruction: str, elapsed: float,
                       context: AgentContext) -> None:
        total_input = sum(t.input_tokens for t in self._turns)
        total_output = sum(t.output_tokens for t in self._turns)
        total_router_tokens = sum(t.router_tokens for t in self._turns)
        avg_fitness = (
            sum(t.fitness for t in self._turns) / len(self._turns)
            if self._turns else 0.0
        )
        avg_command_reward = (
            sum(t.command_reward for t in self._turns) / len(self._turns)
            if self._turns else 0.0
        )
        results = {
            "agent": self.name(),
            "version": self.version(),
            "strategy": self.EVOLUTION_MODE,
            "evolution_mode": self.EVOLUTION_MODE,
            "model": self._llm.model if self._llm else "unknown",
            "instruction": instruction[:300],
            "total_turns": len(self._turns),
            "elapsed_sec": round(elapsed, 2),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_llm_calls": self._llm.call_count if self._llm else 0,
            "total_router_tokens": total_router_tokens,
            "average_fitness": avg_fitness,
            "average_command_reward": avg_command_reward,
            "router_stats": self._router.get_stats() if self._router else {},
            "turns": [
                {
                    "turn": t.turn,
                    "thought": t.thought[:240],
                    "action": t.action[:240],
                    "observation": t.observation[:240],
                    "command_reward": t.command_reward,
                    "fitness": t.fitness,
                    "router_tokens": t.router_tokens,
                    "input_tokens": t.input_tokens,
                    "output_tokens": t.output_tokens,
                    "elapsed_sec": round(t.elapsed_sec, 2),
                }
                for t in self._turns
            ],
        }
        (self.logs_dir / "epicontext_result.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False)
        )
        (self.logs_dir / "reward.txt").write_text(str(round(avg_fitness, 6)))


class RouterLogisticHarborAgent(RouterBackedEpiContextAgent):
    EVOLUTION_MODE = "logistic"

    @staticmethod
    def name() -> str:
        return "router-logistic-epicontext"


class RouterLegacyHarborAgent(RouterBackedEpiContextAgent):
    EVOLUTION_MODE = "legacy"

    @staticmethod
    def name() -> str:
        return "router-legacy-epicontext"
