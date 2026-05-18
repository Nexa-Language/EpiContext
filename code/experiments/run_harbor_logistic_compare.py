"""Compare Router logistic vs legacy Harbor agents without Docker.

This is a local fallback runner for machines where Harbor CLI/Docker are not
available. It still instantiates the Harbor BaseAgent subclasses and calls the
real OpenAI-compatible LLM endpoint, but uses a tiny local environment shim for
`environment.exec`.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from epicontext.harbor_agent import (
    RouterLegacyHarborAgent,
    RouterLogisticHarborAgent,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(os.environ.get(
    "EPICONTEXT_COMPARE_OUT_DIR",
    ROOT / "code" / "results" / "harbor_logistic_compare",
))
DEFAULT_TASKS = [
    ("hello-world", "Create a file called hello.txt with \"Hello, world!\" as the content."),
    (
        "hello-workdir",
        "Run `pwd` and write its output to a file called `workdir.txt` in the current directory.",
    ),
    (
        "hello-user",
        "Run `whoami` and write the output to `/app/whoami.txt`. Then create a file called "
        "`/app/hello.txt` with \"Hello, world!\" as the content.",
    ),
]
REMAINING_FEASIBLE_TASKS = [
    (
        "hello-world-bat",
        "Write a batch script at `C:\\app\\greet.bat` that prints exactly `Hello World` to stdout.",
    ),
    ("hello-alpine", "Create a file called hello.txt with \"Hello, world!\" as the content."),
    ("llm-judge-example", "Write a funny poem in `/app/poem.txt`."),
    (
        "reward-kit-example",
        """Create a text statistics module and analysis script.

Implement `/app/textstats.py` with `word_count(text: str) -> int` and
`most_common(text: str) -> str`. Implement `/app/analyze.py` so it reads
`/app/sample.txt`, computes those statistics, and writes `/app/results.json`.
""",
    ),
    (
        "hello-cuda",
        "Write a simple CUDA program `/app/hello.cu` that prints "
        "\"Hello from GPU thread X\" where X is the thread ID.",
    ),
]


def select_tasks() -> List[tuple[str, str]]:
    suite = os.environ.get("EPICONTEXT_COMPARE_SUITE", "default").lower()
    if suite == "remaining-feasible":
        return REMAINING_FEASIBLE_TASKS
    if suite == "all-feasible":
        return DEFAULT_TASKS + REMAINING_FEASIBLE_TASKS
    return DEFAULT_TASKS


TASKS = select_tasks()
REPETITIONS = int(os.environ.get("EPICONTEXT_COMPARE_REPETITIONS", "2"))


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    return_code: int


class LocalShellEnvironment:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.app_dir = workdir / "app"
        self.workspace_dir = workdir / "workspace"
        self.app_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        (self.app_dir / "sample.txt").write_text(
            "Harbor Harbor context evolution test\n",
            encoding="utf-8",
        )

    async def exec(self, command: str) -> ExecResult:
        command = command.replace("/app", str(self.app_dir))
        command = command.replace("/workspace", str(self.workspace_dir))
        command = command.replace("C:\\app", str(self.app_dir))
        command = command.replace("C:/app", str(self.app_dir))
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            cwd=self.workdir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return ExecResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            return_code=completed.returncode,
        )


async def run_one(agent_cls: Any, task_name: str, instruction: str, rep: int) -> Dict[str, Any]:
    run_dir = OUT_DIR / "runs" / f"{agent_cls.name()}__{task_name}__rep{rep}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"epictx_{task_name}_") as tmp:
        env = LocalShellEnvironment(Path(tmp))
        agent = agent_cls(logs_dir=run_dir)
        started = time.time()
        await agent.setup(env)
        await agent.run(instruction, env, context=None)
        elapsed = time.time() - started

    data = json.loads((run_dir / "epicontext_result.json").read_text(encoding="utf-8"))
    return {
        "agent": agent_cls.name(),
        "strategy": data.get("strategy"),
        "task": task_name,
        "rep": rep,
        "turns": data.get("total_turns", 0),
        "elapsed_sec": data.get("elapsed_sec", elapsed),
        "wall_sec": round(elapsed, 2),
        "input_tokens": data.get("total_input_tokens", 0),
        "output_tokens": data.get("total_output_tokens", 0),
        "router_tokens": data.get("total_router_tokens", 0),
        "llm_calls": data.get("total_llm_calls", 0),
        "average_fitness": data.get("average_fitness", 0.0),
        "average_command_reward": data.get("average_command_reward", 0.0),
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_agent: Dict[str, Dict[str, Any]] = {}
    for agent in sorted({r["agent"] for r in results}):
        rows = [r for r in results if r["agent"] == agent]
        n = len(rows)
        by_agent[agent] = {
            "n": n,
            "avg_turns": sum(r["turns"] for r in rows) / n,
            "avg_elapsed_sec": sum(r["elapsed_sec"] for r in rows) / n,
            "avg_input_tokens": sum(r["input_tokens"] for r in rows) / n,
            "avg_output_tokens": sum(r["output_tokens"] for r in rows) / n,
            "avg_router_tokens": sum(r["router_tokens"] for r in rows) / n,
            "avg_llm_calls": sum(r["llm_calls"] for r in rows) / n,
            "avg_fitness": sum(r["average_fitness"] for r in rows) / n,
            "avg_command_reward": sum(r["average_command_reward"] for r in rows) / n,
        }
    return {"by_agent": by_agent}


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("EPICONTEXT_MAX_TURNS", "6")

    results: List[Dict[str, Any]] = []
    for agent_cls in (RouterLegacyHarborAgent, RouterLogisticHarborAgent):
        for task_name, instruction in TASKS:
            for rep in range(REPETITIONS):
                print(f"Running {agent_cls.name()} / {task_name} / rep {rep}", flush=True)
                result = await run_one(agent_cls, task_name, instruction, rep)
                results.append(result)
                print(
                    f"  turns={result['turns']} in={result['input_tokens']} "
                    f"out={result['output_tokens']} router={result['router_tokens']} "
                    f"fitness={result['average_fitness']:.4f} time={result['elapsed_sec']:.2f}s",
                    flush=True,
                )
                (OUT_DIR / "partial_results.json").write_text(
                    json.dumps(results, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

    final = {
        "config": {
            "tasks": [name for name, _ in TASKS],
            "repetitions": REPETITIONS,
            "max_turns": int(os.environ.get("EPICONTEXT_MAX_TURNS", "6")),
            "model": os.environ.get("OPENAI_MODEL_NAME", "mimo-v2.5-pro"),
        },
        "results": results,
        "summary": summarize(results),
    }
    (OUT_DIR / "final_results.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(final["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
