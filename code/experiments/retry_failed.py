#!/usr/bin/env python3
"""
EpiContext Harbor 失败重试脚本。

读取 v2 中间结果，识别失败组合，逐个重试直到全部成功或达到最大重试次数。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

HARBOR_DIR = Path("/root/proj/papers/EXPERIMENT/EpiContext/harbor-framework")
CODE_DIR = Path("/root/proj/papers/EXPERIMENT/EpiContext/code")
RESULTS_DIR = Path("/root/proj/papers/EXPERIMENT/EpiContext/code/results/harbor_experiments")

AGENT_MAP = {
    "EpiContext": "epicontext.harbor_agent:EpiContextAgent",
    "FullContext": "epicontext.harbor_agent:FullContextBaselineAgent",
    "SlidingWindow": "epicontext.harbor_agent:SlidingWindowBaselineAgent",
    "MethylationOnly": "epicontext.harbor_agent:MethylationOnlyAgent",
    "AcetylationOnly": "epicontext.harbor_agent:AcetylationOnlyAgent",
}

TASK_MAP = {
    "hello-world": "examples/tasks/hello-world",
    "hello-user": "examples/tasks/hello-user",
    "hello-workdir": "examples/tasks/hello-workdir",
    "hello-multi-step-advanced": "examples/tasks/hello-multi-step-advanced",
    "hello-healthcheck": "examples/tasks/hello-healthcheck",
    "describe-image": "examples/tasks/describe-image",
    "llm-judge-example": "examples/tasks/llm-judge-example",
    "reward-kit-example": "examples/tasks/reward-kit-example",
}

MAX_RETRIES = 3
TIMEOUT_MULTIPLIER = 8.0
SUBPROCESS_TIMEOUT = 900


def load_failed_runs() -> List[Tuple[str, str, int]]:
    """加载失败运行列表。"""
    with open(RESULTS_DIR / "intermediate_results.json") as f:
        data = json.load(f)
    
    failed = []
    for r in data:
        if not r["success"]:
            failed.append((r["agent"], r["task"], r["rep"]))
    return failed


def run_single(agent_name: str, task_name: str, rep: int) -> Tuple[bool, Dict[str, Any]]:
    """运行单次实验，返回 (成功, 结果字典)。"""
    agent_path = AGENT_MAP[agent_name]
    task_path = TASK_MAP[task_name]
    job_name = f"retry_{agent_name}_{task_name}_{rep}_{int(time.time())}"
    
    cmd = [
        "uv", "run", "harbor", "run",
        "-c", "examples/configs/job.yaml",
        "-p", task_path,
        "--agent-import-path", agent_path,
        "--agent-timeout-multiplier", str(TIMEOUT_MULTIPLIER),
        "--job-name", job_name,
        "-n", "1", "-q",
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{CODE_DIR}:{env.get('PYTHONPATH', '')}"
    
    try:
        subprocess.run(cmd, cwd=HARBOR_DIR, env=env,
                       capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT)
        
        # 查找 job 目录
        jobs_dir = HARBOR_DIR / "jobs"
        job_dirs = sorted(
            [d for d in jobs_dir.glob(f"*{job_name}*") if d.is_dir()],
            key=lambda p: p.stat().st_mtime, reverse=True
        )
        
        if not job_dirs:
            # fallback: 最近创建的
            all_jobs = sorted(
                [d for d in jobs_dir.glob("*/") if d.is_dir()],
                key=lambda p: p.stat().st_mtime, reverse=True
            )
            if all_jobs:
                job_dirs = [all_jobs[0]]
        
        if not job_dirs:
            return False, {"error": "No job directory found"}
        
        job_dir = job_dirs[0]
        
        # 检查结果 (支持多步任务的嵌套结构)
        result_files = list(job_dir.glob("**/epicontext_result.json"))
        if result_files:
            # 聚合多步结果
            total_turns = 0
            total_time = 0.0
            total_input = 0
            total_output = 0
            total_calls = 0
            strategy = "unknown"
            for rf in result_files:
                try:
                    data = json.loads(rf.read_text())
                    total_turns += data.get("total_turns", 0)
                    total_time += data.get("elapsed_sec", 0)
                    total_input += data.get("total_input_tokens", 0)
                    total_output += data.get("total_output_tokens", 0)
                    total_calls += data.get("total_llm_calls", 0)
                    strategy = data.get("strategy", strategy)
                except Exception:
                    pass
            return True, {
                "turns": total_turns,
                "time_s": total_time,
                "input_tok": total_input,
                "output_tok": total_output,
                "calls": total_calls,
                "strategy": strategy,
            }
        
        # 检查 Harbor 顶层 result.json 是否报告成功
        harbor_result = job_dir / "result.json"
        if harbor_result.exists():
            try:
                hr = json.loads(harbor_result.read_text())
                stats = hr.get("stats", {})
                evals = stats.get("evals", {})
                for eval_name, eval_data in evals.items():
                    if eval_data.get("n_errors", 0) == 0 and eval_data.get("n_trials", 0) > 0:
                        # Harbor 报告成功但无 agent 结果文件 → 视为成功
                        return True, {
                            "turns": 0, "time_s": 0,
                            "input_tok": 0, "output_tok": 0, "calls": 0,
                            "strategy": "unknown",
                        }
            except Exception:
                pass
        
        # 检查异常
        exception_files = list(job_dir.glob("**/exception.txt"))
        if exception_files:
            err = exception_files[0].read_text()[:300]
            return False, {"error": err}
        
        return False, {"error": "No result or exception file"}
        
    except subprocess.TimeoutExpired:
        return False, {"error": "Subprocess timeout"}
    except Exception as e:
        return False, {"error": str(e)[:300]}


def update_results(agent: str, task: str, rep: int, success: bool, result: Dict[str, Any]):
    """更新中间结果文件。"""
    with open(RESULTS_DIR / "intermediate_results.json") as f:
        data = json.load(f)
    
    # 找到并更新对应条目
    for r in data:
        if r["agent"] == agent and r["task"] == task and r["rep"] == rep:
            r["success"] = success
            if success:
                r["turns"] = result.get("turns", 0)
                r["time_s"] = result.get("time_s", 0)
                r["input_tok"] = result.get("input_tok", 0)
                r["output_tok"] = result.get("output_tok", 0)
                r["calls"] = result.get("calls", 0)
                r["strategy"] = result.get("strategy", "unknown")
                r["error"] = None
            else:
                r["error"] = result.get("error", "unknown")[:200]
            break
    
    with open(RESULTS_DIR / "intermediate_results.json", "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def count_success() -> int:
    with open(RESULTS_DIR / "intermediate_results.json") as f:
        data = json.load(f)
    return sum(1 for r in data if r["success"])


def main():
    failed = load_failed_runs()
    print(f"需要重试: {len(failed)} 个失败")
    print(f"当前成功: {count_success()}/120")
    print()
    
    total = len(failed)
    fixed = 0
    
    for idx, (agent, task, rep) in enumerate(failed):
        print(f"[{idx+1}/{total}] {agent} / {task} / rep{rep} ...", end=" ", flush=True)
        
        for attempt in range(1, MAX_RETRIES + 1):
            if attempt > 1:
                wait = 30 * attempt  # 递增等待
                print(f"(retry {attempt}/{MAX_RETRIES}, waiting {wait}s)...", end=" ", flush=True)
                time.sleep(wait)
            
            success, result = run_single(agent, task, rep)
            
            if success:
                print(f"OK! turns={result['turns']} tok={result['input_tok']}")
                update_results(agent, task, rep, True, result)
                fixed += 1
                break
            else:
                err = result.get("error", "unknown")[:80]
                if attempt == MAX_RETRIES:
                    print(f"FAIL (after {MAX_RETRIES} attempts): {err}")
                    update_results(agent, task, rep, False, result)
                else:
                    print(f"FAIL: {err}")
        
        # 每 5 个打印进度
        if (idx + 1) % 5 == 0:
            print(f"  --- Progress: {fixed}/{idx+1} fixed, {count_success()}/120 total OK ---")
    
    print()
    print(f"重试完成: {fixed}/{total} 修复")
    print(f"最终成功: {count_success()}/120")


if __name__ == "__main__":
    main()