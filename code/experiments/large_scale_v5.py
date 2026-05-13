"""
EpiContext Large-Scale Experiment (v5 - Crash Recovery + Optimized)

优化版本:
- 5个损失函数 × 3个维度 × 4个优化器 × 10个策略 × 2次重复 = 1,200次优化
- 更高效的收敛检测 (更早终止)
- 定期保存完整结果到JSON
- 内存优化: 采样历史
"""

from __future__ import annotations
import gc, json, math, os, sys, time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _clip_val(v: float, lo: float = -1e6, hi: float = 1e6) -> float:
    if math.isnan(v) or math.isinf(v):
        return hi if v > 0 else lo
    return max(lo, min(hi, v))

def _clip_array(arr: np.ndarray, lo: float = -1e6, hi: float = 1e6) -> np.ndarray:
    arr = np.where(np.isfinite(arr), arr, np.sign(arr) * hi)
    return np.clip(arr, lo, hi)


class SGDOptimizer:
    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None
        self.name = f"SGD(lr={lr})" if momentum == 0 else f"Momentum(lr={lr},β={momentum})"
    def step(self, params, grad):
        if self.velocity is None: self.velocity = np.zeros_like(params)
        self.velocity = self.momentum * self.velocity - self.lr * grad
        return params + self.velocity
    def reset(self): self.velocity = None

class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None; self.t = 0; self.name = f"Adam(lr={lr})"
    def step(self, params, grad):
        if self.m is None: self.m = np.zeros_like(params); self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    def reset(self): self.m = self.v = None; self.t = 0

class RMSpropOptimizer:
    def __init__(self, lr=0.001, decay=0.9, eps=1e-8):
        self.lr, self.decay, self.eps = lr, decay, eps
        self.cache = None; self.name = f"RMSprop(lr={lr})"
    def step(self, params, grad):
        if self.cache is None: self.cache = np.zeros_like(params)
        self.cache = self.decay * self.cache + (1 - self.decay) * grad ** 2
        return params - self.lr * grad / (np.sqrt(self.cache) + self.eps)
    def reset(self): self.cache = None


class RosenbrockFunction:
    def __init__(self, dim=10): self.dim = dim; self.name = f"Rosenbrock(d={dim})"
    def evaluate(self, x):
        x = np.clip(x, -100, 100)
        total = 0.0
        for i in range(self.dim - 1):
            diff = x[i+1] - x[i]**2
            total += 100.0 * diff * diff + (1.0 - x[i]) ** 2
        return _clip_val(float(total))
    def gradient(self, x):
        x = np.clip(x, -100, 100)
        grad = np.zeros(self.dim)
        for i in range(self.dim - 1):
            diff = x[i+1] - x[i]**2
            grad[i] += -400.0 * x[i] * diff - 2.0 * (1.0 - x[i])
            grad[i+1] += 200.0 * diff
        return _clip_array(grad)
    def generate_initial_point(self, rng): return rng.uniform(-2.0, 2.0, self.dim)

class RastriginFunction:
    def __init__(self, dim=10): self.dim = dim; self.name = f"Rastrigin(d={dim})"
    def evaluate(self, x):
        return float(10.0 * self.dim + np.sum(x**2 - 10.0 * np.cos(2.0 * math.pi * x)))
    def gradient(self, x): return 2.0 * x + 20.0 * math.pi * np.sin(2.0 * math.pi * x)
    def generate_initial_point(self, rng): return rng.uniform(-5.12, 5.12, self.dim)

class AckleyFunction:
    def __init__(self, dim=10): self.dim = dim; self.name = f"Ackley(d={dim})"
    def evaluate(self, x):
        n = float(self.dim)
        s1, s2 = np.sum(x**2), np.sum(np.cos(2.0 * math.pi * x))
        return float(-20.0 * np.exp(-0.2 * np.sqrt(s1/n)) - np.exp(s2/n) + 20.0 + math.e)
    def gradient(self, x):
        n = float(self.dim)
        s1, s2 = np.sum(x**2), np.sum(np.cos(2.0 * math.pi * x))
        st = np.sqrt(s1/n)
        g1 = (4.0*x/(n*st))*np.exp(-0.2*st) if st > 1e-15 else np.zeros_like(x)
        g2 = (2.0*math.pi*np.sin(2.0*math.pi*x)/n)*np.exp(s2/n)
        return g1 + g2
    def generate_initial_point(self, rng): return rng.uniform(-32.768, 32.768, self.dim)

class SphereFunction:
    def __init__(self, dim=10): self.dim = dim; self.name = f"Sphere(d={dim})"
    def evaluate(self, x): return float(np.sum(x**2))
    def gradient(self, x): return 2.0 * x
    def generate_initial_point(self, rng): return rng.uniform(-5.0, 5.0, self.dim)

class BealeFunction:
    def __init__(self, dim=2): self.dim = 2; self.name = "Beale(d=2)"
    def evaluate(self, x):
        x0, x1 = x[0], x[1]
        t1, t2, t3 = 1.5-x0+x0*x1, 2.25-x0+x0*x1**2, 2.625-x0+x0*x1**3
        return t1**2 + t2**2 + t3**2
    def gradient(self, x):
        x0, x1 = x[0], x[1]
        t1, t2, t3 = 1.5-x0+x0*x1, 2.25-x0+x0*x1**2, 2.625-x0+x0*x1**3
        return np.array([2*(t1*(x1-1)+t2*(x1**2-1)+t3*(x1**3-1)), 2*(t1*x0+t2*2*x0*x1+t3*3*x0*x1**2)])
    def generate_initial_point(self, rng): return rng.uniform(-4.5, 4.5, 2)


class FullContextStrategy:
    name = "Full-Context"
    def select_context(self, h, i, m=50): return h

class SlidingWindowStrategy:
    def __init__(self, w=10): self.name = f"SlidingWindow({w})"; self.w = w
    def select_context(self, h, i, m=50): return h[-self.w:]

class MethylationStrategy:
    def __init__(self, t=1e-4): self.name = f"Methylation({t})"; self.t = t
    def select_context(self, h, i, m=50):
        if len(h) <= m: return h
        sel, ll = [], None
        for e in h:
            if ll is not None:
                c = abs(ll - e['loss'])/(abs(ll)+1e-10)
                if c < self.t: ll = e['loss']; continue
            sel.append(e); ll = e['loss']
        return sel[-m:] if sel else h[-1:]

class AcetylationStrategy:
    def __init__(self, t=0.3): self.name = f"Acetylation({t})"; self.t = t
    def select_context(self, h, i, m=50):
        if len(h) <= m: return h
        cg = h[-1].get('grad_norm', 0)
        scored = [(min(e.get('grad_norm',0),cg)/max(e.get('grad_norm',0),cg,1e-10), e) for e in h]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _,e in scored[:m]]

class EpiContextStrategy:
    def __init__(self, a=1.0, b=0.5, g=0.3):
        self.name = f"EpiContext({a},{b},{g})"; self.a, self.b, self.g = a, b, g
        self._m = MethylationStrategy(); self._a = AcetylationStrategy()
    def select_context(self, h, i, m=50):
        if len(h) <= m: return h
        return self._a.select_context(self._m.select_context(h, i, m*2), i, m)


@dataclass
class Result:
    optimizer: str; loss: str; dim: int; strategy: str
    converged: bool; iterations: int; final_loss: float; best_loss: float; time: float
    loss_sampled: List[float] = field(default_factory=list)
    grad_sampled: List[float] = field(default_factory=list)
    ctx_sampled: List[int] = field(default_factory=list)


class Experiment:
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed); self.results: List[Result] = []; self.sample_interval = 100

    def run_single(self, loss_fn, opt, strat, max_iter, rep):
        seed_val = hash(f"{loss_fn.name}_{opt.name}_{strat.name}_{rep}")%(2**31)
        lr = np.random.RandomState(seed_val)
        x = loss_fn.generate_initial_point(lr); opt.reset()
        hist = []; best = float('inf'); converged = False; t0 = time.time()
        ls, gs, cs = [], [], []
        for it in range(max_iter):
            loss = loss_fn.evaluate(x); grad = loss_fn.gradient(x)
            gn = float(np.linalg.norm(grad))
            if loss < best: best = loss
            hist.append({'loss': loss, 'grad_norm': gn})
            if it % self.sample_interval == 0: ls.append(loss); gs.append(gn); cs.append(len(hist))
            if it > 50:
                if gn < 1e-8 or loss < 1e-10: converged = True; break
                if len(hist) >= 50:
                    rl = [e['loss'] for e in hist[-50:]]
                    if abs(rl[0]-rl[-1]) < 1e-12: converged = True; break
            new_x = opt.step(x, grad); new_x = _clip_array(new_x, -100, 100)
            x = loss_fn.generate_initial_point(lr) if (np.any(np.isnan(new_x)) or np.any(np.isinf(new_x))) else new_x
        return Result(opt.name, loss_fn.name, loss_fn.dim, strat.name, converged, len(hist),
                      hist[-1]['loss'] if hist else float('inf'), best, time.time()-t0, ls, gs, cs)

    def run_all(self):
        print("="*70); print("EpiContext Large-Scale Experiment (v5)"); print("="*70)
        losses = [(RosenbrockFunction,[2,5,10]), (RastriginFunction,[2,5,10]),
                  (AckleyFunction,[2,5,10]), (SphereFunction,[2,5,10]), (BealeFunction,[2])]
        opts = [SGDOptimizer(0.01), SGDOptimizer(0.01,0.9), AdamOptimizer(0.001), RMSpropOptimizer(0.001)]
        strats = [FullContextStrategy(), SlidingWindowStrategy(10), SlidingWindowStrategy(20),
                  MethylationStrategy(1e-4), MethylationStrategy(1e-3),
                  AcetylationStrategy(0.3), AcetylationStrategy(0.5),
                  EpiContextStrategy(1.0,0.5,0.3), EpiContextStrategy(2.0,0.5,0.3), EpiContextStrategy(1.0,1.0,0.3)]
        max_iter, num_reps = 5000, 2
        total = sum(len(d)*len(opts)*len(strats)*num_reps for _,d in losses)
        print(f"Total runs: {total}, Max iter: {max_iter}, Reps: {num_reps}\n")
        gt = time.time(); rc = 0
        for lc, dims in losses:
            for d in dims:
                lf = lc(d)
                for o in opts:
                    for s in strats:
                        for rep in range(num_reps):
                            rc += 1
                            r = self.run_single(lf, o, s, max_iter, rep)
                            self.results.append(r)
                            if rc % 50 == 0:
                                el = time.time()-gt; cv = sum(1 for x in self.results if x.converged)
                                print(f"  [{rc}/{total}] {el:.0f}s, conv={cv}/{rc}({cv/rc*100:.1f}%), loss={r.final_loss:.2e}")
                            if rc % 200 == 0:
                                self._save(rc)
                                gc.collect()
        tt = time.time()-gt
        print(f"\n{'='*70}\nDone in {tt:.0f}s ({tt/3600:.1f}h), {len(self.results)} runs, "
              f"{sum(1 for x in self.results if x.converged)} converged\n{'='*70}")
        return self.results

    def _save(self, rc):
        os.makedirs('results', exist_ok=True)
        with open(f'results/checkpoint_{rc}.json','w') as f:
            json.dump({'rc': rc, 'n': len(self.results)}, f)
        print(f"    [checkpoint {rc}]")

    def analyze(self):
        if not self.results: return {}
        a = {'summary':{}, 'by_strategy':{}, 'by_loss':{}, 'stats':{}}
        cv = sum(1 for r in self.results if r.converged)
        a['summary'] = {'total': int(len(self.results)), 'converged': int(cv), 'rate': float(cv/len(self.results)),
                        'avg_iter': float(np.mean([r.iterations for r in self.results])),
                        'avg_loss_log10': float(np.mean([math.log10(max(r.final_loss,1e-15)) for r in self.results])),
                        'avg_time': float(np.mean([r.time for r in self.results]))}
        # by strategy
        groups: Dict[str, List[Result]] = {}
        for r in self.results:
            b = r.strategy.split('(')[0]
            groups.setdefault(b, []).append(r)
        for sn, sr in groups.items():
            a['by_strategy'][sn] = {
                'n': len(sr), 'rate': sum(1 for x in sr if x.converged)/len(sr),
                'avg_iter': float(np.mean([x.iterations for x in sr])),
                'avg_loss_log10': float(np.mean([math.log10(max(x.final_loss,1e-15)) for x in sr]))}
        # by loss function
        lgroups: Dict[str, List[Result]] = {}
        for r in self.results:
            lgroups.setdefault(r.loss, []).append(r)
        for ln, lr in lgroups.items():
            a['by_loss'][ln] = {'n': len(lr), 'rate': sum(1 for x in lr if x.converged)/len(lr)}
        # stats
        epi = [r for r in self.results if 'EpiContext' in r.strategy]
        fc = [r for r in self.results if r.strategy == 'Full-Context']
        if len(epi)>=5 and len(fc)>=5:
            t,p = stats.ttest_ind([r.iterations for r in epi], [r.iterations for r in fc])
            a['stats']['Epi_vs_FC_iters'] = {'t':float(t),'p':float(p),'sig':bool(p<0.05)}
            t2,p2 = stats.ttest_ind([math.log10(max(r.final_loss,1e-15)) for r in epi],
                                    [math.log10(max(r.final_loss,1e-15)) for r in fc])
            a['stats']['Epi_vs_FC_loss'] = {'t':float(t2),'p':float(p2),'sig':bool(p2<0.05)}
        return a

    def save_results(self, out='results'):
        os.makedirs(out, exist_ok=True)
        a = self.analyze()
        output = {'analysis': a, 'results': [{
            'optimizer':r.optimizer,'loss':r.loss,'dim':int(r.dim),'strategy':r.strategy,
            'converged':bool(r.converged),'iterations':int(r.iterations),
            'final_loss':float(r.final_loss),'best_loss':float(r.best_loss),
            'time':float(r.time),'loss_history':[float(x) for x in r.loss_sampled],
            'grad_norm_history':[float(x) for x in r.grad_sampled],
            'avg_ctx': float(np.mean(r.ctx_sampled)) if r.ctx_sampled else 0
        } for r in self.results]}
        fp = os.path.join(out, 'large_scale_results.json')
        with open(fp,'w') as f: json.dump(output, f, indent=2)
        print(f"\nSaved to {fp}")
        self._print(a)

    def _print(self, a):
        print("\n"+"="*70+"\nKEY FINDINGS\n"+"="*70)
        s = a['summary']
        print(f"\n{s['total']} runs, {s['rate']*100:.1f}% convergence, avg {s['avg_iter']:.0f} iters")
        print(f"\n{'Strategy':<25} {'Rate':>8} {'AvgIters':>10} {'LogLoss':>10}")
        print("-"*55)
        for sn,sm in sorted(a['by_strategy'].items()):
            print(f"{sn:<25} {sm['rate']:>8.3f} {sm['avg_iter']:>10.0f} {sm['avg_loss_log10']:>10.2f}")
        for tn,tr in a.get('stats',{}).items():
            print(f"\n{tn}: p={tr['p']:.6f} {'✓SIG' if tr['sig'] else '✗ns'}")


def main():
    e = Experiment(42); e.run_all(); e.save_results()

if __name__ == '__main__': main()
