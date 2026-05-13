#!/usr/bin/env python3
"""
EpiContext Harbor 批量实验运行器。

运行多个 agent 变体在多个 Harbor 任务上的对比实验，
收集结果并生成统计报告。

用法:
    cd harbor-framework
    PYTHONPATH="/root/proj/papers/EXPERIMENT/EpiContext/code:$PYTHONPATH" \
        uv run python /root/proj/papers/EXPERIMENT/EpiContext/code/experiments/run_harbor_experiments.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# ============================================================================
# Configuration
# ============================================================================

HARBOR_DIR = Path("/root/proj/papers/EXPERIMENT/EpiContext/harbor-framework")
CODE_DIR = Path("/root/proj/papers/EXPERIMENT/EpiContext/code")
RESULTS_DIR = Path("/root/proj/papers/EXPERIMENT/EpiContext/code/results/harbor_experiments")

AGENTS = {
    "EpiContext": "epicontext.harbor_agent:EpiContextAgent",
    "AdaptiveEpiContext": "epicontext.harbor_agent:AdaptiveEpiContextAgent",
    "FullContext": "epicontext.harbor_agent:FullContextBaselineAgent",
    "SlidingWindow": "epicontext.harbor_agent:SlidingWindowBaselineAgent",
    "MethylationOnly": "epicontext.harbor_agent:MethylationOnlyAgent",
    "AcetylationOnly": "epicontext.harbor_agent:AcetylationOnlyAgent",
}

# Harbor tasks: 简单 + 复杂混合
TASKS = [
    # 简单单步任务 (校准基线)
    "examples/tasks/hello-world",
    "examples/tasks/hello-user",
    "examples/tasks/hello-workdir",
    # 复杂多步任务 (真正测试上下文策略)
    "examples/tasks/hello-multi-step-advanced",
    "examples/tasks/hello-healthcheck",
    "examples/tasks/describe-image",
    "examples/tasks/llm-judge-example",
    "examples/tasks/reward-kit-example",
]

N_REPETITIONS = 3  # 每个配置重复次数
TIMEOUT_MULTIPLIER = 8.0  # 复杂任务需要更多时间
N_CONCURRENT = 1  # 串行运行避免 API 限流
SUBPROCESS_TIMEOUT = 900  # 单次实验 subprocess 超时 (15 min)


@dataclass
class RunResult:
    """单次运行结果。"""
    agent_name: str
    task_name: str
    repetition: int
    success: bool
    total_turns: int
    elapsed_sec: float
    input_tokens: int
    output_tokens: int
    llm_calls: int
    strategy: str
    error: Optional[str] = None
    raw_result: Optional[Dict[str, Any]] = None

    @classmethod
    def from_job_dir(cls, agent_name: str, task_name: str, rep: int,
                     job_dir: Path) -> "RunResult":
        """从 Harbor job 目录解析结果。"""
        # 查找 epicontext_result.json
        result_files = list(job_dir.glob("*/agent/epicontext_result.json"))
        if not result_files:
            # 检查是否有异常
            exception_files = list(job_dir.glob("*/exception.txt"))
            if exception_files:
                error_text = exception_files[0].read_text()[:500]
                return cls(
                    agent_name=agent_name, task_name=task_name,
                    repetition=rep, success=False,
                    total_turns=0, elapsed_sec=0,
                    input_tokens=0, output_tokens=0, llm_calls=0,
                    strategy="unknown", error=error_text,
                )
            return cls(
                agent_name=agent_name, task_name=task_name,
                repetition=rep, success=False,
                total_turns=0, elapsed_sec=0,
                input_tokens=0, output_tokens=0, llm_calls=0,
                strategy="unknown", error="No result file found",
            )

        try:
            data = json.loads(result_files[0].read_text())
            return cls(
                agent_name=agent_name, task_name=task_name,
                repetition=rep, success=True,
                total_turns=data.get("total_turns", 0),
                elapsed_sec=data.get("elapsed_sec", 0),
                input_tokens=data.get("total_input_tokens", 0),
                output_tokens=data.get("total_output_tokens", 0),
                llm_calls=data.get("total_llm_calls", 0),
                strategy=data.get("strategy", "unknown"),
                raw_result=data,
            )
        except Exception as e:
            return cls(
                agent_name=agent_name, task_name=task_name,
                repetition=rep, success=False,
                total_turns=0, elapsed_sec=0,
                input_tokens=0, output_tokens=0, llm_calls=0,
                strategy="unknown", error=str(e),
            )


def run_single_experiment(agent_name: str, agent_path: str, task_path: str,
                          rep: int) -> Path:
    """运行单次实验，返回 job 目录路径。"""
    job_name = f"epictx_{agent_name}_{Path(task_path).name}_{rep}"
    
    cmd = [
        "uv", "run", "harbor", "run",
        "-c", "examples/configs/job.yaml",
        "-p", task_path,
        "--agent-import-path", agent_path,
        "--agent-timeout-multiplier", str(TIMEOUT_MULTIPLIER),
        "--job-name", job_name,
        "-n", str(N_CONCURRENT),
        "-q",
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{CODE_DIR}:{env.get('PYTHONPATH', '')}"
    
    try:
        result = subprocess.run(
            cmd, cwd=HARBOR_DIR, env=env,
            capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT,
        )
        # 查找最新 job 目录
        jobs_dir = HARBOR_DIR / "jobs"
        job_dirs = sorted(jobs_dir.glob(f"*{job_name}*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if job_dirs:
            return job_dirs[0]
        
        # fallback: 查找最近创建的目录
        all_jobs = sorted(jobs_dir.glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True)
        if all_jobs:
            return all_jobs[0]
            
        print(f"  [WARN] No job directory found for {agent_name}/{task_path}/{rep}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {agent_name}/{task_path}/{rep}")
        return None
    except Exception as e:
        print(f"  [ERROR] {agent_name}/{task_path}/{rep}: {e}")
        return None


def run_all_experiments() -> List[RunResult]:
    """运行所有实验。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results: List[RunResult] = []
    total_runs = len(AGENTS) * len(TASKS) * N_REPETITIONS
    run_count = 0
    
    start_time = time.time()
    
    for agent_name, agent_path in AGENTS.items():
        for task_path in TASKS:
            task_short = Path(task_path).name
            for rep in range(N_REPETITIONS):
                run_count += 1
                elapsed = time.time() - start_time
                eta = (elapsed / max(run_count - 1, 1)) * (total_runs - run_count + 1)
                
                print(f"\n[{run_count}/{total_runs}] {agent_name} on {task_short} (rep {rep+1}/{N_REPETITIONS})")
                print(f"  Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min")
                
                job_dir = run_single_experiment(agent_name, agent_path, task_path, rep)
                
                if job_dir:
                    result = RunResult.from_job_dir(agent_name, task_short, rep, job_dir)
                else:
                    result = RunResult(
                        agent_name=agent_name, task_name=task_short,
                        repetition=rep, success=False,
                        total_turns=0, elapsed_sec=0,
                        input_tokens=0, output_tokens=0, llm_calls=0,
                        strategy="unknown", error="Job directory not found",
                    )
                
                all_results.append(result)
                status = "OK" if result.success else f"FAIL: {result.error[:80] if result.error else 'unknown'}"
                print(f"  -> {status}")
                
                # 中间保存
                if run_count % 5 == 0:
                    save_intermediate_results(all_results)
    
    return all_results


def save_intermediate_results(results: List[RunResult]) -> None:
    """保存中间结果。"""
    path = RESULTS_DIR / "intermediate_results.json"
    data = [
        {
            "agent": r.agent_name,
            "task": r.task_name,
            "rep": r.repetition,
            "success": r.success,
            "turns": r.total_turns,
            "time_s": r.elapsed_sec,
            "input_tok": r.input_tokens,
            "output_tok": r.output_tokens,
            "calls": r.llm_calls,
            "strategy": r.strategy,
            "error": r.error,
        }
        for r in results
    ]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"  [Saved] {len(results)} results to {path}")


def analyze_results(results: List[RunResult]) -> Dict[str, Any]:
    """分析实验结果。"""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    analysis = {
        "total_runs": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "by_agent": {},
        "by_task": {},
        "statistical_tests": {},
    }
    
    # 按 agent 汇总
    for agent_name in AGENTS:
        agent_results = [r for r in successful if r.agent_name == agent_name]
        if not agent_results:
            continue
        
        analysis["by_agent"][agent_name] = {
            "n": len(agent_results),
            "avg_turns": float(np.mean([r.total_turns for r in agent_results])),
            "std_turns": float(np.std([r.total_turns for r in agent_results])),
            "avg_time_s": float(np.mean([r.elapsed_sec for r in agent_results])),
            "avg_input_tok": float(np.mean([r.input_tokens for r in agent_results])),
            "avg_output_tok": float(np.mean([r.output_tokens for r in agent_results])),
            "avg_calls": float(np.mean([r.llm_calls for r in agent_results])),
            "total_input_tok": int(sum(r.input_tokens for r in agent_results)),
            "total_output_tok": int(sum(r.output_tokens for r in agent_results)),
        }
    
    # 按 task 汇总
    for task_path in TASKS:
        task_short = Path(task_path).name
        task_results = [r for r in successful if r.task_name == task_short]
        if not task_results:
            continue
        
        analysis["by_task"][task_short] = {
            "n": len(task_results),
            "avg_turns": float(np.mean([r.total_turns for r in task_results])),
            "avg_time_s": float(np.mean([r.elapsed_sec for r in task_results])),
        }
    
    # 统计检验: EpiContext vs FullContext
    epi_results = [r for r in successful if r.agent_name == "EpiContext"]
    full_results = [r for r in successful if r.agent_name == "FullContext"]
    
    if epi_results and full_results:
        # 配对比较 (按 task+rep 匹配)
        epi_turns = [r.total_turns for r in epi_results]
        full_turns = [r.total_turns for r in full_results]
        
        if len(epi_turns) == len(full_turns) and len(epi_turns) > 1:
            t_stat, p_val = stats.ttest_rel(epi_turns, full_turns)
            analysis["statistical_tests"]["epi_vs_full_turns"] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05),
            }
        
        epi_tok = [r.input_tokens for r in epi_results]
        full_tok = [r.input_tokens for r in full_results]
        if len(epi_tok) == len(full_tok) and len(epi_tok) > 1:
            t_stat, p_val = stats.ttest_rel(epi_tok, full_tok)
            analysis["statistical_tests"]["epi_vs_full_input_tokens"] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05),
            }
    
    return analysis


def generate_report(results: List[RunResult], analysis: Dict[str, Any]) -> str:
    """生成实验报告。"""
    lines = []
    lines.append("=" * 70)
    lines.append("EpiContext Harbor Experiment Report")
    lines.append("=" * 70)
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total runs: {analysis['total_runs']}")
    lines.append(f"Successful: {analysis['successful']}")
    lines.append(f"Failed: {analysis['failed']}")
    lines.append("")
    
    # Agent 对比表
    lines.append("-" * 70)
    lines.append("Agent Comparison")
    lines.append("-" * 70)
    lines.append(f"{'Agent':<20} {'N':>4} {'AvgTurns':>10} {'AvgTime(s)':>12} {'AvgInTok':>10} {'AvgOutTok':>10} {'AvgCalls':>10}")
    lines.append("-" * 70)
    
    for agent_name in AGENTS:
        if agent_name in analysis["by_agent"]:
            a = analysis["by_agent"][agent_name]
            lines.append(
                f"{agent_name:<20} {a['n']:>4} {a['avg_turns']:>10.1f} "
                f"{a['avg_time_s']:>12.1f} {a['avg_input_tok']:>10.1f} "
                f"{a['avg_output_tok']:>10.1f} {a['avg_calls']:>10.1f}"
            )
    
    lines.append("")
    
    # 统计检验
    if analysis["statistical_tests"]:
        lines.append("-" * 70)
        lines.append("Statistical Tests")
        lines.append("-" * 70)
        for test_name, test_result in analysis["statistical_tests"].items():
            sig = "SIGNIFICANT" if test_result["significant"] else "not significant"
            lines.append(
                f"  {test_name}: t={test_result['t_statistic']:.3f}, "
                f"p={test_result['p_value']:.4f} ({sig})"
            )
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    """主函数。"""
    print("=" * 70)
    print("EpiContext Harbor Batch Experiment Runner")
    print("=" * 70)
    print(f"Agents: {list(AGENTS.keys())}")
    print(f"Tasks: {[Path(t).name for t in TASKS]}")
    print(f"Repetitions: {N_REPETITIONS}")
    print(f"Total runs: {len(AGENTS) * len(TASKS) * N_REPETITIONS}")
    print(f"Estimated time: ~{len(AGENTS) * len(TASKS) * N_REPETITIONS * 1.5 / 60:.1f} hours")
    print("=" * 70)
    
    # 运行实验
    results = run_all_experiments()
    
    # 分析
    analysis = analyze_results(results)
    
    # 保存最终结果
    final_path = RESULTS_DIR / "final_results.json"
    final_data = {
        "config": {
            "agents": list(AGENTS.keys()),
            "tasks": [Path(t).name for t in TASKS],
            "n_repetitions": N_REPETITIONS,
        },
        "results": [
            {
                "agent": r.agent_name,
                "task": r.task_name,
                "rep": r.repetition,
                "success": r.success,
                "turns": r.total_turns,
                "time_s": r.elapsed_sec,
                "input_tok": r.input_tokens,
                "output_tok": r.output_tokens,
                "calls": r.llm_calls,
                "strategy": r.strategy,
                "error": r.error,
            }
            for r in results
        ],
        "analysis": analysis,
    }
    final_path.write_text(json.dumps(final_data, indent=2, ensure_ascii=False))
    
    # 生成报告
    report = generate_report(results, analysis)
    report_path = RESULTS_DIR / "report.txt"
    report_path.write_text(report)
    
    print("\n" + report)
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()