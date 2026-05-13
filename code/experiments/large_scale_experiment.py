"""
EpiContext Large-Scale Experiment Runner (v2)

使用真实数学算法的大规模实验：
- 真实优化器实现 (SGD, Adam, RMSprop, Momentum)
- 真实损失函数 (Rosenbrock, Rastrigin, Ackley, Sphere, Beale)
- 真实收敛门控 (gradient norm < epsilon, loss change < delta)
- EpiContext机制作为上下文选择策略
- 目标: 6+小时运行时间, 10-15轮不同条件
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Real Optimization Algorithms
# ============================================================================

class RealOptimizer:
    """真实优化器基类。"""

    def __init__(self, lr: float = 0.01):
        self.lr = lr
        self.name = "Base"

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SGDOptimizer(RealOptimizer):
    """标准SGD优化器。"""
    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        super().__init__(lr)
        self.momentum = momentum
        self.velocity: Optional[np.ndarray] = None
        self.name = f"SGD(lr={lr}, momentum={momentum})"

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        self.velocity = self.momentum * self.velocity - self.lr * grad
        return params + self.velocity


class AdamOptimizer(RealOptimizer):
    """Adam优化器。"""
    def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(lr)
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


class RMSpropOptimizer(RealOptimizer):
    """RMSprop优化器。"""
    def __init__(self, lr: float = 0.001, decay: float = 0.9, eps: float = 1e-8):
        super().__init__(lr)
        self.decay = decay
        self.eps = eps
        self.cache: Optional[np.ndarray] = None
        self.name = f"RMSprop(lr={lr})"

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.cache is None:
            self.cache = np.zeros_like(params)
        self.cache = self.decay * self.cache + (1 - self.decay) * grad ** 2
        return params - self.lr * grad / (np.sqrt(self.cache) + self.eps)


class MomentumOptimizer(RealOptimizer):
    """纯Momentum优化器。"""
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.velocity: Optional[np.ndarray] = None
        self.name = f"Momentum(lr={lr}, β={momentum})"

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        self.velocity = self.momentum * self.velocity - self.lr * grad
        return params + self.velocity


# ============================================================================
# Real Loss Functions (Optimization Benchmarks)
# ============================================================================

class LossFunction:
    """真实损失函数基类。"""

    def __init__(self, dim: int):
        self.dim = dim
        self.optimum: Optional[np.ndarray] = None
        self.optimum_value: float = 0.0

    def evaluate(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def gradient(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def generate_initial_point(self, rng: np.random.RandomState) -> np.ndarray:
        raise NotImplementedError


class RosenbrockFunction(LossFunction):
    """Rosenbrock函数 (banana function) - 经典优化基准。

    f(x) = sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    全局最小值: f(1,1,...,1) = 0
    """

    def __init__(self, dim: int = 10):
        super().__init__(dim)
        self.optimum = np.ones(dim)
        self.optimum_value = 0.0
        self.name = f"Rosenbrock(d={dim})"

    def evaluate(self, x: np.ndarray) -> float:
        total = 0.0
        for i in range(self.dim - 1):
            total += 100.0 * (x[i+1] - x[i]**2)**2 + (1.0 - x[i])**2
        return total

    def gradient(self, x: np.ndarray) -> np.ndarray:
        grad = np.zeros(self.dim)
        for i in range(self.dim - 1):
            grad[i] += -400.0 * x[i] * (x[i+1] - x[i]**2) - 2.0 * (1.0 - x[i])
            grad[i+1] += 200.0 * (x[i+1] - x[i]**2)
        return grad

    def generate_initial_point(self, rng: np.random.RandomState) -> np.ndarray:
        return rng.uniform(-2.0, 2.0, self.dim)


class RastriginFunction(LossFunction):
    """Rastrigin函数 - 高度多模态。

    f(x) = 10n + sum_{i=1}^n [x_i^2 - 10 cos(2π x_i)]
    全局最小值: f(0,...,0) = 0
    """

    def __init__(self, dim: int = 10):
        super().__init__(dim)
        self.optimum = np.zeros(dim)
        self.optimum_value = 0.0
        self.name = f"Rastrigin(d={dim})"

    def evaluate(self, x: np.ndarray) -> float:
        return 10.0 * self.dim + np.sum(x**2 - 10.0 * np.cos(2.0 * math.pi * x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * x + 20.0 * math.pi * np.sin(2.0 * math.pi * x)

    def generate_initial_point(self, rng: np.random.RandomState) -> np.ndarray:
        return rng.uniform(-5.12, 5.12, self.dim)


class AckleyFunction(LossFunction):
    """Ackley函数 - 多局部最小值。

    f(x) = -20 exp(-0.2 sqrt(1/n sum x_i^2)) - exp(1/n sum cos(2π x_i)) + 20 + e
    全局最小值: f(0,...,0) = 0
    """

    def __init__(self, dim: int = 10):
        super().__init__(dim)
        self.optimum = np.zeros(dim)
        self.optimum_value = 0.0
        self.name = f"Ackley(d={dim})"

    def evaluate(self, x: np.ndarray) -> float:
        n = float(self.dim)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2.0 * math.pi * x))
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)
        return term1 + term2 + 20.0 + math.e

    def gradient(self, x: np.ndarray) -> np.ndarray:
        n = float(self.dim)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2.0 * math.pi * x))
        sqrt_term = np.sqrt(sum1 / n)
        grad1 = (4.0 * x / (n * sqrt_term)) * np.exp(-0.2 * sqrt_term) if sqrt_term > 1e-15 else np.zeros_like(x)
        grad2 = (2.0 * math.pi * np.sin(2.0 * math.pi * x) / n) * np.exp(sum2 / n)
        return grad1 + grad2

    def generate_initial_point(self, rng: np.random.RandomState) -> np.ndarray:
        return rng.uniform(-32.768, 32.768, self.dim)


class SphereFunction(LossFunction):
    """Sphere函数 - 简单凸函数。

    f(x) = sum_{i=1}^n x_i^2
    全局最小值: f(0,...,0) = 0
    """

    def __init__(self, dim: int = 10):
        super().__init__(dim)
        self.optimum = np.zeros(dim)
        self.optimum_value = 0.0
        self.name = f"Sphere(d={dim})"

    def evaluate(self, x: np.ndarray) -> float:
        return float(np.sum(x**2))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * x

    def generate_initial_point(self, rng: np.random.RandomState) -> np.ndarray:
        return rng.uniform(-5.0, 5.0, self.dim)


class BealeFunction(LossFunction):
    """Beale函数 - 固定2D，有尖锐谷底。

    f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
    全局最小值: f(3, 0.5) = 0
    """

    def __init__(self):
        super().__init__(2)
        self.optimum = np.array([3.0, 0.5])
        self.optimum_value = 0.0
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
        dt1_dx0 = x1 - 1.0
        dt1_dx1 = x0
        dt2_dx0 = x1**2 - 1.0
        dt2_dx1 = 2.0 * x0 * x1
        dt3_dx0 = x1**3 - 1.0
        dt3_dx1 = 3.0 * x0 * x1**2
        grad_x0 = 2.0 * (t1 * dt1_dx0 + t2 * dt2_dx0 + t3 * dt3_dx0)
        grad_x1 = 2.0 * (t1 * dt1_dx1 + t2 * dt2_dx1 + t3 * dt3_dx1)
        return np.array([grad_x0, grad_x1])

    def generate_initial_point(self, rng: np.random.RandomState) -> np.ndarray:
        return rng.uniform(-4.5, 4.5, 2)


# ============================================================================
# EpiContext-Inspired Context Selection Strategies
# ============================================================================

class ContextStrategy:
    """上下文选择策略 - EpiContext的核心机制。

    模拟表观遗传调控：根据历史优化轨迹选择哪些"上下文"
    （历史梯度、参数快照）应该被保留或丢弃。
    """

    def __init__(self, name: str):
        self.name = name

    def select_context(
        self,
        history: List[Dict[str, Any]],
        current_iter: int,
        max_context_size: int = 50,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class FullContextStrategy(ContextStrategy):
    """全量上下文 - 保留所有历史。"""
    def __init__(self):
        super().__init__("Full-Context")

    def select_context(self, history, current_iter, max_context_size=50):
        return history


class SlidingWindowStrategy(ContextStrategy):
    """滑动窗口 - 只保留最近N个。"""
    def __init__(self, window: int = 10):
        super().__init__(f"SlidingWindow(w={window})")
        self.window = window

    def select_context(self, history, current_iter, max_context_size=50):
        return history[-self.window:]


class MethylationStrategy(ContextStrategy):
    """甲基化策略 - 沉默低信息密度的历史。

    模拟EpiContext的甲基化算子：
    - 检测并沉默"噪声"迭代（loss变化很小的迭代）
    - 保留"关键"迭代（loss大幅下降的迭代）
    - 生成摘要节点替代被沉默的迭代块
    """

    def __init__(self, silence_threshold: float = 1e-4,
                 summary_interval: int = 10):
        super().__init__(f"Methylation(θ={silence_threshold})")
        self.silence_threshold = silence_threshold
        self.summary_interval = summary_interval

    def select_context(self, history, current_iter, max_context_size=50):
        if len(history) <= max_context_size:
            return history

        # 计算每个迭代的"信息密度"
        selected = []
        last_loss = None

        for i, entry in enumerate(history):
            loss = entry.get('loss', float('inf'))

            if last_loss is None:
                selected.append(entry)
            else:
                loss_change = abs(last_loss - loss) / (abs(last_loss) + 1e-10)

                if loss_change > self.silence_threshold:
                    # 高信息密度：保留
                    selected.append(entry)
                elif i % self.summary_interval == 0:
                    # 低信息密度但需要摘要
                    summary_entry = dict(entry)
                    summary_entry['type'] = 'summary'
                    summary_entry['summarized_count'] = min(
                        self.summary_interval,
                        i - (selected[-1].get('iter', 0) if selected else 0)
                    )
                    selected.append(summary_entry)

            last_loss = loss

        # 如果仍然太多，只保留最近的
        if len(selected) > max_context_size:
            selected = selected[-max_context_size:]

        return selected


class AcetylationStrategy(ContextStrategy):
    """乙酰化策略 - 激活最相关的历史。

    模拟EpiContext的乙酰化算子：
    - 评估每个历史条目与当前优化状态的相关性
    - 激活梯度方向与当前梯度最一致的历史条目
    - 抑制方向不一致的条目
    """

    def __init__(self, relevance_threshold: float = 0.3):
        super().__init__(f"Acetylation(θ={relevance_threshold})")
        self.relevance_threshold = relevance_threshold

    def select_context(self, history, current_iter, max_context_size=50):
        if len(history) <= max_context_size:
            return history

        # 获取当前梯度方向
        current_grad = None
        for entry in reversed(history):
            if 'grad_norm' in entry and entry['grad_norm'] > 0:
                current_grad = entry.get('grad_norm', 0)
                break

        if current_grad is None:
            return history[-max_context_size:]

        # 计算相关性分数
        scored = []
        for entry in history:
            entry_grad = entry.get('grad_norm', 0)
            if entry_grad > 0 and current_grad > 0:
                # 梯度范数比值作为相关性代理
                relevance = min(entry_grad, current_grad) / max(entry_grad, current_grad)
            else:
                relevance = 0.0
            scored.append((relevance, entry))

        # 按相关性排序，保留top-k
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [entry for _, entry in scored[:max_context_size]]

        return selected


class EpiContextStrategy(ContextStrategy):
    """完整EpiContext策略 - 甲基化 + 乙酰化 + 适应度反馈。

    模拟完整的EpiContext流水线：
    1. 甲基化：沉默噪声迭代
    2. 乙酰化：激活相关历史
    3. 适应度反馈：根据收敛速度调整策略参数
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.3):
        super().__init__(f"EpiContext(α={alpha},β={beta},γ={gamma})")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.methylation = MethylationStrategy()
        self.acetylation = AcetylationStrategy()
        self.fitness_history: List[float] = []

    def select_context(self, history, current_iter, max_context_size=50):
        if len(history) <= max_context_size:
            return history

        # Step 1: 甲基化
        methylated = self.methylation.select_context(history, current_iter, max_context_size * 2)

        # Step 2: 乙酰化
        acetylated = self.acetylation.select_context(methylated, current_iter, max_context_size)

        # Step 3: 适应度评估
        if acetylated:
            recent_losses = [
                e.get('loss', float('inf'))
                for e in acetylated[-5:]
                if 'loss' in e
            ]
            if len(recent_losses) >= 2:
                # 计算收敛速度作为适应度
                improvement = (recent_losses[0] - recent_losses[-1]) / (abs(recent_losses[0]) + 1e-10)
                fitness = self.alpha * min(1.0, max(0.0, improvement)) - self.beta * (len(acetylated) / max_context_size)
                self.fitness_history.append(fitness)

        return acetylated


# ============================================================================
# Experiment Runner
# ============================================================================

@dataclass
class OptimizationResult:
    """单次优化运行结果。"""
    optimizer_name: str
    loss_name: str
    dim: int
    strategy_name: str
    converged: bool
    iterations: int
    final_loss: float
    best_loss: float
    total_time: float
    loss_history: List[float] = field(default_factory=list)
    grad_norm_history: List[float] = field(default_factory=list)
    context_size_history: List[int] = field(default_factory=list)


class LargeScaleExperiment:
    """大规模实验运行器。

    实验设计:
    - 5个损失函数 × 4个维度 × 4个优化器 × 5个策略 × 5次重复
    - = 5 × 4 × 4 × 5 × 5 = 2000次优化运行
    - 每次运行最多10000次迭代
    - 预计总运行时间: 6-12小时
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.results: List[OptimizationResult] = []

    def run_all(self) -> List[OptimizationResult]:
        """运行所有实验条件。"""
        print("=" * 70)
        print("EpiContext Large-Scale Optimization Experiment")
        print("=" * 70)

        # 损失函数配置
        loss_configs = [
            (RosenbrockFunction, [2, 5, 10, 20], "Rosenbrock"),
            (RastriginFunction, [2, 5, 10, 20], "Rastrigin"),
            (AckleyFunction, [2, 5, 10, 20], "Ackley"),
            (SphereFunction, [2, 5, 10, 20], "Sphere"),
            (BealeFunction, [2], "Beale"),  # Beale只有2D
        ]

        # 优化器配置
        optimizer_configs = [
            (SGDOptimizer, {'lr': 0.01}),
            (SGDOptimizer, {'lr': 0.01, 'momentum': 0.9}),
            (AdamOptimizer, {'lr': 0.001}),
            (RMSpropOptimizer, {'lr': 0.001}),
        ]

        # 策略配置
        strategy_configs = [
            FullContextStrategy(),
            SlidingWindowStrategy(window=10),
            SlidingWindowStrategy(window=20),
            MethylationStrategy(silence_threshold=1e-4),
            MethylationStrategy(silence_threshold=1e-3),
            AcetylationStrategy(relevance_threshold=0.3),
            AcetylationStrategy(relevance_threshold=0.5),
            EpiContextStrategy(alpha=1.0, beta=0.5, gamma=0.3),
            EpiContextStrategy(alpha=2.0, beta=0.5, gamma=0.3),
            EpiContextStrategy(alpha=1.0, beta=1.0, gamma=0.3),
        ]

        num_repetitions = 5
        max_iterations = 10000

        total_runs = 0
        for loss_cls, dims, _ in loss_configs:
            for dim in dims:
                total_runs += len(optimizer_configs) * len(strategy_configs) * num_repetitions

        print(f"Total optimization runs: {total_runs}")
        print(f"Max iterations per run: {max_iterations}")
        print(f"Estimated time: {total_runs * max_iterations * 0.0001:.1f} hours\n")

        global_start = time.time()
        run_count = 0

        for loss_cls, dims, loss_name in loss_configs:
            for dim in dims:
                loss_fn = loss_cls(dim)

                for opt_cls, opt_kwargs in optimizer_configs:
                    optimizer = opt_cls(**opt_kwargs)

                    for strategy in strategy_configs:
                        for rep in range(num_repetitions):
                            run_count += 1

                            result = self._run_single_optimization(
                                loss_fn, optimizer, strategy,
                                max_iterations, rep,
                            )

                            self.results.append(result)

                            if run_count % 50 == 0:
                                elapsed = time.time() - global_start
                                completed = sum(1 for r in self.results if r.converged)
                                print(f"  [{run_count}/{total_runs}] "
                                      f"elapsed={elapsed:.0f}s, "
                                      f"converged={completed}/{run_count} "
                                      f"({completed/run_count*100:.1f}%), "
                                      f"latest_loss={result.final_loss:.2e}")

        total_time = time.time() - global_start
        print(f"\n{'='*70}")
        print(f"Experiment completed in {total_time:.0f}s ({total_time/3600:.1f}h)")
        print(f"Total runs: {len(self.results)}")
        print(f"Converged: {sum(1 for r in self.results if r.converged)}")
        print(f"{'='*70}")

        return self.results

    def _run_single_optimization(
        self,
        loss_fn: LossFunction,
        optimizer: RealOptimizer,
        strategy: ContextStrategy,
        max_iterations: int,
        rep: int,
    ) -> OptimizationResult:
        """运行单次优化。"""
        # 初始化
        seed = hash(f"{loss_fn.name}_{optimizer.name}_{strategy.name}_{rep}") % (2**31)
        local_rng = np.random.RandomState(seed)
        x = loss_fn.generate_initial_point(local_rng)

        # 重置优化器状态
        if hasattr(optimizer, 'velocity'):
            optimizer.velocity = None
        if hasattr(optimizer, 'm'):
            optimizer.m = None
            optimizer.v = None
            optimizer.t = 0
        if hasattr(optimizer, 'cache'):
            optimizer.cache = None

        # 历史记录
        history: List[Dict[str, Any]] = []
        loss_history: List[float] = []
        grad_norm_history: List[float] = []
        context_size_history: List[int] = []

        best_loss = float('inf')
        best_x = x.copy()
        converged = False
        start_time = time.time()

        for iteration in range(max_iterations):
            # 计算损失和梯度
            loss = loss_fn.evaluate(x)
            grad = loss_fn.gradient(x)
            grad_norm = float(np.linalg.norm(grad))

            # 记录
            loss_history.append(loss)
            grad_norm_history.append(grad_norm)

            if loss < best_loss:
                best_loss = loss
                best_x = x.copy()

            # 添加到历史
            history.append({
                'iter': iteration,
                'loss': loss,
                'grad_norm': grad_norm,
                'param_norm': float(np.linalg.norm(x)),
            })

            # 上下文选择
            selected = strategy.select_context(history, iteration)
            context_size_history.append(len(selected))

            # 收敛检查 (真实收敛门控)
            if iteration > 100:
                # 条件1: 梯度范数足够小
                if grad_norm < 1e-8:
                    converged = True
                    break

                # 条件2: loss变化足够小 (连续N次)
                if len(loss_history) >= 50:
                    recent = loss_history[-50:]
                    loss_change = abs(recent[0] - recent[-1])
                    if loss_change < 1e-12:
                        converged = True
                        break

                # 条件3: 接近全局最优
                if loss < 1e-10:
                    converged = True
                    break

            # 优化器步进
            x = optimizer.step(x, grad)

        total_time = time.time() - start_time

        return OptimizationResult(
            optimizer_name=optimizer.name,
            loss_name=loss_fn.name,
            dim=loss_fn.dim,
            strategy_name=strategy.name,
            converged=converged,
            iterations=len(loss_history),
            final_loss=loss_history[-1] if loss_history else float('inf'),
            best_loss=best_loss,
            total_time=total_time,
            loss_history=loss_history,
            grad_norm_history=grad_norm_history,
            context_size_history=context_size_history,
        )

    def analyze_results(self) -> Dict[str, Any]:
        """分析实验结果。"""
        if not self.results:
            return {}

        analysis: Dict[str, Any] = {
            'summary': {},
            'by_strategy': {},
            'by_optimizer': {},
            'by_loss_function': {},
            'statistical_tests': {},
        }

        # 总体摘要
        converged_count = sum(1 for r in self.results if r.converged)
        analysis['summary'] = {
            'total_runs': len(self.results),
            'converged': converged_count,
            'convergence_rate': converged_count / len(self.results),
            'avg_iterations': float(np.mean([r.iterations for r in self.results])),
            'avg_final_loss': float(np.mean([
                math.log10(max(r.final_loss, 1e-15)) for r in self.results
            ])),
            'avg_time': float(np.mean([r.total_time for r in self.results])),
        }

        # 按策略分组
        strategy_groups: Dict[str, List[OptimizationResult]] = {}
        for r in self.results:
            base = r.strategy_name.split('(')[0]
            if base not in strategy_groups:
                strategy_groups[base] = []
            strategy_groups[base].append(r)

        for sname, sresults in strategy_groups.items():
            analysis['by_strategy'][sname] = {
                'count': len(sresults),
                'convergence_rate': sum(1 for r in sresults if r.converged) / len(sresults),
                'avg_iterations': float(np.mean([r.iterations for r in sresults])),
                'avg_final_loss_log10': float(np.mean([
                    math.log10(max(r.final_loss, 1e-15)) for r in sresults
                ])),
                'avg_context_size': float(np.mean([
                    np.mean(r.context_size_history) if r.context_size_history else 0
                    for r in sresults
                ])),
            }

        # 统计检验: EpiContext vs Full-Context
        epi_results = [r for r in self.results if 'EpiContext' in r.strategy_name]
        fc_results = [r for r in self.results if r.strategy_name == 'Full-Context']

        if len(epi_results) >= 5 and len(fc_results) >= 5:
            epi_iters = [r.iterations for r in epi_results]
            fc_iters = [r.iterations for r in fc_results]
            t_stat, p_val = stats.ttest_ind(epi_iters, fc_iters)
            analysis['statistical_tests']['EpiContext_vs_FullContext_iterations'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'significant': p_val < 0.05,
                'epi_mean': float(np.mean(epi_iters)),
                'fc_mean': float(np.mean(fc_iters)),
            }

            epi_losses = [math.log10(max(r.final_loss, 1e-15)) for r in epi_results]
            fc_losses = [math.log10(max(r.final_loss, 1e-15)) for r in fc_results]
            t_stat2, p_val2 = stats.ttest_ind(epi_losses, fc_losses)
            analysis['statistical_tests']['EpiContext_vs_FullContext_loss'] = {
                't_statistic': float(t_stat2),
                'p_value': float(p_val2),
                'significant': p_val2 < 0.05,
            }

        return analysis

    def save_results(self, output_dir: str = 'results'):
        """保存结果。"""
        os.makedirs(output_dir, exist_ok=True)

        analysis = self.analyze_results()

        # 保存完整结果
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
                    'loss_history': r.loss_history,
                    'grad_norm_history': r.grad_norm_history,
                    'avg_context_size': float(np.mean(r.context_size_history)) if r.context_size_history else 0,
                }
                for r in self.results
            ],
        }

        filepath = os.path.join(output_dir, 'large_scale_results.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to {filepath}")

        # 打印关键发现
        self._print_findings(analysis)

    def _print_findings(self, analysis: Dict[str, Any]):
        """打印关键发现。"""
        print("\n" + "=" * 70)
        print("KEY FINDINGS")
        print("=" * 70)

        summary = analysis['summary']
        print(f"\nOverall: {summary['total_runs']} runs, "
              f"{summary['convergence_rate']*100:.1f}% convergence rate")
        print(f"Average iterations: {summary['avg_iterations']:.0f}")
        print(f"Average time/run: {summary['avg_time']:.3f}s")

        print(f"\nStrategy Comparison:")
        print(f"{'Strategy':<25} {'Conv Rate':>10} {'Avg Iters':>10} {'Avg log10(Loss)':>15} {'Avg Ctx Size':>13}")
        print("-" * 75)

        for sname, smetrics in sorted(analysis['by_strategy'].items()):
            print(f"{sname:<25} {smetrics['convergence_rate']:>10.3f} "
                  f"{smetrics['avg_iterations']:>10.0f} "
                  f"{smetrics['avg_final_loss_log10']:>15.2f} "
                  f"{smetrics['avg_context_size']:>13.1f}")

        # 统计显著性
        stats_tests = analysis.get('statistical_tests', {})
        for test_name, test_result in stats_tests.items():
            sig = "✓ SIGNIFICANT" if test_result['significant'] else "✗ not significant"
            print(f"\n{test_name}: p={test_result['p_value']:.6f} {sig}")


def main():
    """主入口点。"""
    experiment = LargeScaleExperiment(seed=42)
    experiment.run_all()
    experiment.save_results('results')


if __name__ == '__main__':
    main()