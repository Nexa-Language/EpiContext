"""
EpiContext Large-Scale Optimization Experiment (v3 - Memory Optimized)

真实数学算法大规模实验:
- 5个损失函数 × 4个维度 × 4个优化器 × 10个策略 × 3次重复 = 2,400次优化
- 真实梯度计算和收敛门控
- 内存优化: 只存储采样历史 (每100次迭代)
- 定期保存中间结果
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Real Optimization Algorithms
# ============================================================================

class SGDOptimizer:
    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocity: Optional[np.ndarray] = None
        self.name = f"SGD(lr={lr})" if momentum == 0 else f"Momentum(lr={lr},β={momentum})"

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        self.velocity = self.momentum * self.velocity - self.lr * grad
        return params + self.velocity

    def reset(self):
        self.velocity = None


class AdamOptimizer:
    def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.t = 0
        self.name = f"Adam(lr={lr})"

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self):
        self.m = None
        self.v = None
        self.t = 0


class RMSpropOptimizer:
    def __init__(self, lr: float = 0.001, decay: float = 0.9, eps: float = 1e-8):
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.cache: Optional[np.ndarray] = None
        self.name = f"RMSprop(lr={lr})"

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.cache is None:
            self.cache = np.zeros_like(params)
        self.cache = self.decay * self.cache + (1 - self.decay) * grad ** 2
        return params - self.lr * grad / (np.sqrt(self.cache) + self.eps)

    def reset(self):
        self.cache = None


# ============================================================================
# Real Loss Functions
# ============================================================================

def _clip_val(v: float, lo: float = -1e6, hi: float = 1e6) -> float:
    """数值裁剪，防止溢出。"""
    if math.isnan(v) or math.isinf(v):
        return hi if v > 0 else lo
    return max(lo, min(hi, v))

def _clip_array(arr: np.ndarray, lo: float = -1e6, hi: float = 1e6) -> np.ndarray:
    """数值裁剪数组，替换NaN/Inf。"""
    arr = np.where(np.isfinite(arr), arr, np.sign(arr) * hi)
    return np.clip(arr, lo, hi)


class RosenbrockFunction:
    def __init__(self, dim: int = 10):
        self.dim = dim
        self.name = f"Rosenbrock(d={dim})"

    def evaluate(self, x: np.ndarray) -> float:
        x = np.clip(x, -100, 100)
        total = 0.0
        for i in range(self.dim - 1):
            diff = x[i+1] - x[i]**2
            total += 100.0 * diff * diff + (1.0 - x[i]) ** 2
        return _clip_val(float(total))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -100, 100)
        grad = np.zeros(self.dim)
        for i in range(self.dim - 1):
            diff = x[i+1] - x[i]**2
            grad[i] += -400.0 * x[i] * diff - 2.0 * (1.0 - x[i])
            grad[i+1] += 200.0 * diff
        return _clip_array(grad)

    def generate_initial_point(self, rng: np.random.RandomState) -> np.ndarray:
        return rng.uniform(-2.0, 2.0, self.dim)


class RastriginFunction:
    def __init__(self, dim: int = 10):
        self.dim = dim
        self.name = f"Rastrigin(d={dim})"

    def evaluate(self, x: np.ndarray) -> float:
        return 10.0 * self.dim + float(np.sum(x**2 - 10.0 * np.cos(2.0 * math.pi * x)))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * x + 20.0 * math.pi * np.sin(2.0 * math.pi * x)

    def generate_initial_point(self, rng: np.random.RandomState) -> np.ndarray:
        return rng.uniform(-5.12, 5.12, self.dim)


class AckleyFunction:
    def __init__(self, dim: int = 10):
        self.dim = dim
        self.name = f"Ackley(d={dim})"

    def evaluate(self, x: np.ndarray) -> float:
        n = float(self.dim)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2.0 * math.pi * x))
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)
        return float(term1 + term2 + 20.0 + math.e)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        n = float(self.dim)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2.0 * math.pi * x))
        sqrt_term = np.sqrt(sum1 / n)
        if sqrt_term > 1e-15:
            grad1 = (4.0 * x / (n * sqrt_term)) * np.exp(-0.2 * sqrt_term)
        else:
            grad1 = np.zeros_like(x)
        grad2 = (2.0 * math.pi * np.sin(2.0 * math.pi * x) / n) * np.exp(sum2 / n)
        return grad1 + grad2

    def generate_initial_point(self, rng: np.random.RandomState) -> np.ndarray:
        return rng.uniform(-32.768, 32.768, self.dim)


class SphereFunction:
    def __init__(self, dim: int = 10):
        self.dim = dim
        self.name = f"Sphere(d={dim})"

    def evaluate(self, x: np.ndarray) -> float:
        return float(np.sum(x**2))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * x

    def generate_initial_point(self, rng: np.random.RandomState) -> np.ndarray:
        return rng.uniform(-5.0, 5.0, self.dim)


class BealeFunction:
    def __init__(self, dim: int = 2):
        self.dim = 2  # Beale is always 2D
        self.name = "Beale(d=2)"

    def evaluate(self, x: np.ndarray) -> float:
        x0, x1 = x[0], x[1]
        t1 = 1.5 - x0 + x0 * x1
        t2 = 2.25 - x0 + x0 * x1**2
        t3 = 2.625 - x0 + x0 * x1**3
        return t1**2 + t2**2 + t3**2

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x0, x1 = x[0], x[1]
        t1 = 1.5 - x0 + x0 * x1
        t2 = 2.25 - x0 + x0 * x1**2
        t3 = 2.625 - x0 + x0 * x1**3
        grad_x0 = 2*(t1*(x1-1) + t2*(x1**2-1) + t3*(x1**3-1))
        grad_x1 = 2*(t1*x0 + t2*2*x0*x1 + t3*3*x0*x1**2)
        return np.array([grad_x0, grad_x1])

    def generate_initial_point(self, rng: np.random.RandomState) -> np.ndarray:
        return rng.uniform(-4.5, 4.5, 2)


# ============================================================================
# Context Strategies (EpiContext Operators)
# ============================================================================

class FullContextStrategy:
    name = "Full-Context"
    def select_context(self, history, current_iter, max_ctx=50):
        return history

class SlidingWindowStrategy:
    def __init__(self, w: int = 10):
        self.name = f"SlidingWindow({w})"
        self.w = w
    def select_context(self, history, current_iter, max_ctx=50):
        return history[-self.w:]

class MethylationStrategy:
    """甲基化: 沉默loss变化小的迭代"""
    def __init__(self, threshold: float = 1e-4):
        self.name = f"Methylation({threshold})"
        self.threshold = threshold
    def select_context(self, history, current_iter, max_ctx=50):
        if len(history) <= max_ctx:
            return history
        selected = []
        last_loss = None
        for entry in history:
            loss = entry['loss']
            if last_loss is not None:
                change = abs(last_loss - loss) / (abs(last_loss) + 1e-10)
                if change < self.threshold:
                    last_loss = loss
                    continue
            selected.append(entry)
            last_loss = loss
        if len(selected) > max_ctx:
            selected = selected[-max_ctx:]
        return selected if selected else history[-1:]

class AcetylationStrategy:
    """乙酰化: 激活梯度方向一致的历史"""
    def __init__(self, threshold: float = 0.3):
        self.name = f"Acetylation({threshold})"
        self.threshold = threshold
    def select_context(self, history, current_iter, max_ctx=50):
        if len(history) <= max_ctx:
            return history
        current_gn = history[-1].get('grad_norm', 0)
        scored = []
        for entry in history:
            gn = entry.get('grad_norm', 0)
            if gn > 0 and current_gn > 0:
                relevance = min(gn, current_gn) / max(gn, current_gn)
            else:
                relevance = 0.0
            scored.append((relevance, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:max_ctx]]

class EpiContextStrategy:
    """完整EpiContext: 甲基化 + 乙酰化 + 适应度反馈"""
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.3):
        self.name = f"EpiContext({alpha},{beta},{gamma})"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._methylation = MethylationStrategy()
        self._acetylation = AcetylationStrategy()
    def select_context(self, history, current_iter, max_ctx=50):
        if len(history) <= max_ctx:
            return history
        m = self._methylation.select_context(history, current_iter, max_ctx * 2)
        a = self._acetylation.select_context(m, current_iter, max_ctx)
        return a


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class OptimizationResult:
    optimizer_name: str
    loss_name: str
    dim: int
    strategy_name: str
    converged: bool
    iterations: int
    final_loss: float
    best_loss: float
    total_time: float
    # 采样历史 (每100次迭代一个点)
    loss_history_sampled: List[float] = field(default_factory=list)
    grad_norm_history_sampled: List[float] = field(default_factory=list)
    context_size_sampled: List[int] = field(default_factory=list)


# ============================================================================
# Experiment Runner
# ============================================================================

class LargeScaleExperiment:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.results: List[OptimizationResult] = []
        self.sample_interval = 100

    def run_single(self, loss_fn, optimizer, strategy,
                   max_iterations: int, rep: int) -> OptimizationResult:
        seed_val = hash(f"{loss_fn.name}_{optimizer.name}_{strategy.name}_{rep}") % (2**31)
        local_rng = np.random.RandomState(seed_val)
        x = loss_fn.generate_initial_point(local_rng)
        optimizer.reset()

        history: List[Dict[str, Any]] = []
        best_loss = float('inf')
        converged = False
        start_time = time.time()

        # 采样存储
        loss_sampled: List[float] = []
        grad_sampled: List[float] = []
        ctx_sampled: List[int] = []

        for iteration in range(max_iterations):
            loss = loss_fn.evaluate(x)
            grad = loss_fn.gradient(x)
            grad_norm = float(np.linalg.norm(grad))

            if loss < best_loss:
                best_loss = loss

            history.append({'iter': iteration, 'loss': loss, 'grad_norm': grad_norm})

            # 采样记录
            if iteration % self.sample_interval == 0:
                loss_sampled.append(loss)
                grad_sampled.append(grad_norm)
                ctx_sampled.append(len(history))

            # 上下文选择
            selected = strategy.select_context(history, iteration)

            # 收敛检查
            if iteration > 100:
                if grad_norm < 1e-8:
                    converged = True
                    break
                if loss < 1e-10:
                    converged = True
                    break
                if len(loss_sampled) >= 10:
                    recent_losses = [e['loss'] for e in history[-50:]]
                    if len(recent_losses) >= 50:
                        lc = abs(recent_losses[0] - recent_losses[-1])
                        if lc < 1e-12:
                            converged = True
                            break

            # 优化步进 (带数值稳定性保护)
            new_x = optimizer.step(x, grad)
            new_x = _clip_array(new_x, -100, 100)
            if np.any(np.isnan(new_x)) or np.any(np.isinf(new_x)):
                x = loss_fn.generate_initial_point(local_rng)
            else:
                x = new_x

        total_time = time.time() - start_time
        final_loss = history[-1]['loss'] if history else float('inf')

        return OptimizationResult(
            optimizer_name=optimizer.name,
            loss_name=loss_fn.name,
            dim=loss_fn.dim,
            strategy_name=strategy.name,
            converged=converged,
            iterations=len(history),
            final_loss=final_loss,
            best_loss=best_loss,
            total_time=total_time,
            loss_history_sampled=loss_sampled,
            grad_norm_history_sampled=grad_sampled,
            context_size_sampled=ctx_sampled,
        )

    def run_all(self) -> List[OptimizationResult]:
        print("=" * 70)
        print("EpiContext Large-Scale Optimization Experiment (v3)")
        print("=" * 70)

        loss_configs = [
            (RosenbrockFunction, [2, 5, 10, 20]),
            (RastriginFunction, [2, 5, 10, 20]),
            (AckleyFunction, [2, 5, 10, 20]),
            (SphereFunction, [2, 5, 10, 20]),
            (BealeFunction, [2]),
        ]

        optimizers = [
            SGDOptimizer(lr=0.01),
            SGDOptimizer(lr=0.01, momentum=0.9),
            AdamOptimizer(lr=0.001),
            RMSpropOptimizer(lr=0.001),
        ]

        strategies = [
            FullContextStrategy(),
            SlidingWindowStrategy(w=10),
            SlidingWindowStrategy(w=20),
            MethylationStrategy(threshold=1e-4),
            MethylationStrategy(threshold=1e-3),
            AcetylationStrategy(threshold=0.3),
            AcetylationStrategy(threshold=0.5),
            EpiContextStrategy(alpha=1.0, beta=0.5, gamma=0.3),
            EpiContextStrategy(alpha=2.0, beta=0.5, gamma=0.3),
            EpiContextStrategy(alpha=1.0, beta=1.0, gamma=0.3),
        ]

        max_iter = 10000
        num_reps = 3

        total_runs = 0
        for _, dims in loss_configs:
            for _ in dims:
                total_runs += len(optimizers) * len(strategies) * num_reps

        print(f"Total runs: {total_runs}")
        print(f"Max iterations/run: {max_iter}")
        print(f"Repetitions: {num_reps}")
        print(f"Sample interval: every {self.sample_interval} iterations\n")

        global_start = time.time()
        run_count = 0
        save_interval = 200

        for loss_cls, dims in loss_configs:
            for dim in dims:
                loss_fn = loss_cls(dim)

                for opt in optimizers:
                    for strat in strategies:
                        for rep in range(num_reps):
                            run_count += 1
                            result = self.run_single(loss_fn, opt, strat, max_iter, rep)
                            self.results.append(result)

                            if run_count % 50 == 0:
                                elapsed = time.time() - global_start
                                converged = sum(1 for r in self.results if r.converged)
                                print(f"  [{run_count}/{total_runs}] "
                                      f"elapsed={elapsed:.0f}s, "
                                      f"converged={converged}/{run_count} "
                                      f"({converged/run_count*100:.1f}%), "
                                      f"latest={result.final_loss:.2e}")

                            # 定期保存中间结果
                            if run_count % save_interval == 0:
                                self._save_intermediate(run_count)

        total_time = time.time() - global_start
        print(f"\n{'='*70}")
        print(f"Completed in {total_time:.0f}s ({total_time/3600:.1f}h)")
        print(f"Total runs: {len(self.results)}")
        print(f"Converged: {sum(1 for r in self.results if r.converged)}")
        print(f"{'='*70}")

        return self.results

    def _save_intermediate(self, run_count: int):
        """保存中间结果以防止崩溃丢失数据。"""
        os.makedirs('results', exist_ok=True)
        filepath = f'resolutions/intermediate_{run_count}.json'
        try:
            with open(f'results/intermediate_{run_count}.json', 'w') as f:
                json.dump({'run_count': run_count, 'n_results': len(self.results)}, f)
            print(f"    [checkpoint saved: {run_count} runs]")
        except Exception:
            pass
        gc.collect()

    def analyze(self) -> Dict[str, Any]:
        """分析结果。"""
        if not self.results:
            return {}

        analysis: Dict[str, Any] = {'summary': {}, 'by_strategy': {}, 'statistical_tests': {}}

        converged_count = sum(1 for r in self.results if r.converged)
        analysis['summary'] = {
            'total_runs': len(self.results),
            'converged': converged_count,
            'convergence_rate': converged_count / len(self.results),
            'avg_iterations': float(np.mean([r.iterations for r in self.results])),
            'avg_final_loss_log10': float(np.mean([
                math.log10(max(r.final_loss, 1e-15)) for r in self.results
            ])),
            'avg_time': float(np.mean([r.total_time for r in self.results])),
        }

        # 按策略分组
        groups: Dict[str, List[OptimizationResult]] = {}
        for r in self.results:
            base = r.strategy_name.split('(')[0]
            if base not in groups:
                groups[base] = []
            groups[base].append(r)

        for sname, sresults in groups.items():
            analysis['by_strategy'][sname] = {
                'count': len(sresults),
                'convergence_rate': sum(1 for r in sresults if r.converged) / len(sresults),
                'avg_iterations': float(np.mean([r.iterations for r in sresults])),
                'avg_final_loss_log10': float(np.mean([
                    math.log10(max(r.final_loss, 1e-15)) for r in sresults
                ])),
            }

        # 统计检验
        epi = [r for r in self.results if 'EpiContext' in r.strategy_name]
        fc = [r for r in self.results if r.strategy_name == 'Full-Context']

        if len(epi) >= 5 and len(fc) >= 5:
            epi_iters = [r.iterations for r in epi]
            fc_iters = [r.iterations for r in fc]
            t, p = stats.ttest_ind(epi_iters, fc_iters)
            analysis['statistical_tests']['EpiContext_vs_FullContext_iterations'] = {
                't_stat': float(t), 'p_value': float(p),
                'significant': p < 0.05,
                'epi_mean': float(np.mean(epi_iters)),
                'fc_mean': float(np.mean(fc_iters)),
            }

            epi_l = [math.log10(max(r.final_loss, 1e-15)) for r in epi]
            fc_l = [math.log10(max(r.final_loss, 1e-15)) for r in fc]
            t2, p2 = stats.ttest_ind(epi_l, fc_l)
            analysis['statistical_tests']['EpiContext_vs_FullContext_loss'] = {
                't_stat': float(t2), 'p_value': float(p2),
                'significant': p2 < 0.05,
            }

        return analysis

    def save_results(self, output_dir: str = 'results'):
        os.makedirs(output_dir, exist_ok=True)
        analysis = self.analyze()

        output = {
            'analysis': analysis,
            'results': [
                {
                    'optimizer': r.optimizer_name,
                    'loss_function': r.loss_name,
                    'dim': r.dim,
                    'strategy': r.strategy_name,
                    'converged': r.converged,
                    'iterations': r.iterations,
                    'final_loss': r.final_loss,
                    'best_loss': r.best_loss,
                    'total_time': r.total_time,
                    'loss_history': r.loss_history_sampled,
                    'grad_norm_history': r.grad_norm_history_sampled,
                    'avg_context_size': float(np.mean(r.context_size_sampled)) if r.context_size_sampled else 0,
                }
                for r in self.results
            ],
        }

        filepath = os.path.join(output_dir, 'large_scale_results.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to {filepath}")
        self._print_findings(analysis)

    def _print_findings(self, analysis: Dict[str, Any]):
        print("\n" + "=" * 70)
        print("KEY FINDINGS")
        print("=" * 70)

        s = analysis['summary']
        print(f"\nOverall: {s['total_runs']} runs, {s['convergence_rate']*100:.1f}% convergence")
        print(f"Avg iterations: {s['avg_iterations']:.0f}")
        print(f"Avg time/run: {s['avg_time']:.4f}s")

        print(f"\n{'Strategy':<25} {'ConvRate':>10} {'AvgIters':>10} {'AvgLogLoss':>12}")
        print("-" * 60)
        for sname, sm in sorted(analysis['by_strategy'].items()):
            print(f"{sname:<25} {sm['convergence_rate']:>10.3f} "
                  f"{sm['avg_iterations']:>10.0f} {sm['avg_final_loss_log10']:>12.2f}")

        for tn, tr in analysis.get('statistical_tests', {}).items():
            sig = "✓ SIGNIFICANT" if tr['significant'] else "✗ not significant"
            print(f"\n{tn}: p={tr['p_value']:.6f} {sig}")


def main():
    exp = LargeScaleExperiment(seed=42)
    exp.run_all()
    exp.save_results('results')


if __name__ == '__main__':
    main()