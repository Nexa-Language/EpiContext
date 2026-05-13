"""Generate publication-quality figures from v5 experiment results."""

import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

def load(fp):
    with open(fp) as f: return json.load(f)

def plot_fig1(results, out):
    """Fig 1: Per-function convergence rate comparison."""
    by = {}
    for r in results:
        s = r['strategy'].split('(')[0]
        by.setdefault(r['loss'], {}).setdefault(s, []).append(r)
    funcs = sorted(by.keys())
    methods = ['Full-Context', 'SlidingWindow', 'Methylation', 'Acetylation', 'EpiContext']
    colors = {'Full-Context':'#e74c3c', 'SlidingWindow':'#3498db', 'Methylation':'#2ecc71',
              'Acetylation':'#f39c12', 'EpiContext':'#9b59b6'}
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]; x = np.arange(len(funcs)); w = 0.15
    for i, m in enumerate(methods):
        rates = [np.mean([1 if r['converged'] else 0 for r in by[f].get(m,[])])*100 if by[f].get(m,[]) else 0 for f in funcs]
        ax.bar(x + i*w, rates, w, label=m, color=colors[m], alpha=0.85, edgecolor='white')
    ax.set_xticks(x + 2*w); ax.set_xticklabels(funcs, rotation=35, ha='right', fontsize=7)
    ax.set_ylabel('Convergence Rate (%)'); ax.set_title('Convergence Rate by Function', fontweight='bold')
    ax.legend(fontsize=7, ncol=2); ax.set_ylim(0, 100)
    ax = axes[1]
    for i, m in enumerate(methods):
        iters = [np.mean([r['iterations'] for r in by[f].get(m,[])]) if by[f].get(m,[]) else 0 for f in funcs]
        ax.bar(x + i*w, iters, w, label=m, color=colors[m], alpha=0.85, edgecolor='white')
    ax.set_xticks(x + 2*w); ax.set_xticklabels(funcs, rotation=35, ha='right', fontsize=7)
    ax.set_ylabel('Average Iterations'); ax.set_title('Iterations to Convergence', fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    fig.suptitle('EpiContext vs Baselines: Per-Function Performance', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out, 'fig_ls1_convergence.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out, 'fig_ls1_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  ✓ fig_ls1 saved")

def plot_fig2(results, out):
    """Fig 2: Context efficiency scatter."""
    by = {}
    for r in results:
        s = r['strategy'].split('(')[0]
        by.setdefault(s, {'ctx':[], 'conv':[], 'iter':[]})
        by[s]['ctx'].append(r.get('avg_ctx', 0)); by[s]['conv'].append(1 if r['converged'] else 0); by[s]['iter'].append(r['iterations'])
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'Full-Context':'#e74c3c', 'SlidingWindow':'#3498db', 'Methylation':'#2ecc71', 'Acetylation':'#f39c12', 'EpiContext':'#9b59b6'}
    markers = {'Full-Context':'s', 'SlidingWindow':'^', 'Methylation':'D', 'Acetylation':'v', 'EpiContext':'o'}
    for s, d in by.items():
        ax.scatter(np.mean(d['ctx']), np.mean(d['conv'])*100, s=200, color=colors.get(s,'gray'), marker=markers.get(s,'o'), label=s, edgecolors='black', linewidth=0.5, zorder=5)
        ax.annotate(s, (np.mean(d['ctx']), np.mean(d['conv'])*100), fontsize=8, ha='center', va='bottom', xytext=(0,8), textcoords='offset points')
    ax.set_xlabel('Average Context Size'); ax.set_ylabel('Convergence Rate (%)')
    ax.set_title('Context Efficiency: Size vs Performance', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out, 'fig_ls2_efficiency.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out, 'fig_ls2_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  ✓ fig_ls2 saved")

def plot_fig3(results, out):
    """Fig 3: Ablation heatmap."""
    funcs = sorted(set(r['loss'] for r in results))
    methods = ['Full-Context', 'SlidingWindow', 'Methylation', 'Acetylation', 'EpiContext']
    matrix = np.zeros((len(methods), len(funcs)))
    for i, m in enumerate(methods):
        for j, f in enumerate(funcs):
            data = [r for r in results if r['loss']==f and r['strategy'].split('(')[0]==m]
            if data: matrix[i, j] = np.mean([1 if r['converged'] else 0 for r in data])
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(funcs))); ax.set_xticklabels(funcs, fontsize=7, rotation=35, ha='right')
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=10)
    ax.set_title('Convergence Rate: Strategy × Function', fontweight='bold', fontsize=13)
    for i in range(len(methods)):
        for j in range(len(funcs)):
            ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', color='white' if matrix[i,j]<0.5 else 'black', fontweight='bold', fontsize=9)
    plt.colorbar(im, ax=ax, label='Convergence Rate')
    plt.tight_layout()
    fig.savefig(os.path.join(out, 'fig_ls4_ablation_heatmap.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out, 'fig_ls4_ablation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  ✓ fig_ls4 saved")

def plot_fig4(results, out):
    """Fig 4: Per-function EpiContext advantage."""
    by = {}
    for r in results:
        s = r['strategy'].split('(')[0]
        by.setdefault(r['loss'], {}).setdefault(s, []).append(r)
    funcs, conv_diff, iter_diff = [], [], []
    for f in sorted(by.keys()):
        fc, ec = by[f].get('Full-Context', []), by[f].get('EpiContext', [])
        if fc and ec:
            funcs.append(f)
            conv_diff.append((np.mean([1 if r['converged'] else 0 for r in ec]) - np.mean([1 if r['converged'] else 0 for r in fc]))*100)
            iter_diff.append((np.mean([r['iterations'] for r in fc]) - np.mean([r['iterations'] for r in ec])) / np.mean([r['iterations'] for r in fc]) * 100)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(funcs))
    colors_c = ['#2ecc71' if v > 0 else '#e74c3c' for v in conv_diff]
    colors_i = ['#2ecc71' if v > 0 else '#e74c3c' for v in iter_diff]
    ax.bar(x - 0.2, conv_diff, 0.35, color=colors_c, alpha=0.85, label='Convergence Rate Δ (pp)')
    ax.bar(x + 0.2, iter_diff, 0.35, color=colors_i, alpha=0.85, label='Iteration Savings (%)')
    ax.set_xticks(x); ax.set_xticklabels(funcs, rotation=35, ha='right', fontsize=7)
    ax.set_ylabel('EpiContext Advantage over Full-Context'); ax.set_title('Per-Function EpiContext Advantage', fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(os.path.join(out, 'fig_ls5_advantage.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out, 'fig_ls5_advantage.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  ✓ fig_ls5 saved")

def plot_fig5(results, out):
    """Fig 5: Loss curves for Sphere(d=5) and Ackley(d=5)."""
    for fname in ['Sphere(d=5)', 'Ackley(d=5)']:
        fig, ax = plt.subplots(figsize=(10, 5))
        methods = ['Full-Context', 'EpiContext', 'Acetylation', 'Methylation', 'SlidingWindow']
        colors = {'Full-Context':'#e74c3c', 'EpiContext':'#9b59b6', 'Acetylation':'#f39c12', 'Methylation':'#2ecc71', 'SlidingWindow':'#3498db'}
        for m in methods:
            data = [r for r in results if r['loss']==fname and r['strategy'].split('(')[0]==m]
            if data and any(r['loss_history'] for r in data):
                max_len = min(len(r['loss_history']) for r in data if r['loss_history'])
                if max_len > 0:
                    avg = np.zeros(max_len); n = 0
                    for r in data:
                        if len(r['loss_history']) >= max_len:
                            avg += np.log10(np.maximum(np.array(r['loss_history'][:max_len]), 1e-15)); n += 1
                    if n > 0:
                        avg /= n; iters = np.arange(max_len) * 100
                        ax.plot(iters, avg, color=colors[m], linewidth=2, label=m, alpha=0.9)
        ax.set_xlabel('Iteration'); ax.set_ylabel('log10(Loss)'); ax.set_title(f'Average Loss Curves: {fname}', fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        safe = fname.replace('(','_').replace(')','_').replace('=','')
        fig.savefig(os.path.join(out, f'fig_ls3_loss_{safe}.pdf'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(out, f'fig_ls3_loss_{safe}.png'), dpi=300, bbox_inches='tight')
        plt.close(); print(f"  ✓ fig_ls3_{safe} saved")

def main():
    data = load('results/large_scale_results.json')
    results = data['results']
    out = 'paper/mypaper/figures'
    os.makedirs(out, exist_ok=True)
    print("Generating v5 figures...")
    plot_fig1(results, out); plot_fig2(results, out); plot_fig3(results, out)
    plot_fig4(results, out); plot_fig5(results, out)
    print(f"\n✅ All v5 figures saved to {out}")

if __name__ == '__main__': main()
