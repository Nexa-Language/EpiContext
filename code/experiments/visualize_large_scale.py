"""
EpiContext Large-Scale Results Visualization

为大规模优化实验生成论文级图表。
"""

import json
import math
import os
import sys
from typing import Any, Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)


def load_results(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_convergence_comparison(results: Dict[str, Any], output_dir: str):
    """图1: 各策略的收敛率对比。"""
    analysis = results.get('analysis', {})
    by_strategy = analysis.get('by_strategy', {})

    strategies = list(by_strategy.keys())
    conv_rates = [by_strategy[s]['convergence_rate'] * 100 for s in strategies]
    avg_iters = [by_strategy[s]['avg_iterations'] for s in strategies]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 收敛率
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(strategies)))
    bars = axes[0].bar(range(len(strategies)), conv_rates, color=colors, edgecolor='white')
    axes[0].set_xticks(range(len(strategies)))
    axes[0].set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Convergence Rate (%)', fontsize=11)
    axes[0].set_title('Convergence Rate by Strategy', fontweight='bold')
    for bar, val in zip(bars, conv_rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold')

    # 平均迭代数
    axes[1].bar(range(len(strategies)), avg_iters, color=colors, edgecolor='white')
    axes[1].set_xticks(range(len(strategies)))
    axes[1].set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('Average Iterations', fontsize=11)
    axes[1].set_title('Average Iterations to Converge', fontweight='bold')

    fig.suptitle('Context Strategy Performance Comparison',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_ls1_convergence.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig_ls1_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ fig_ls1_convergence saved")


def plot_context_efficiency(results: Dict[str, Any], output_dir: str):
    """图2: 上下文效率 - 平均上下文大小 vs 收敛率。"""
    analysis = results.get('analysis', {})
    by_strategy = analysis.get('by_strategy', {})

    strategies = []
    ctx_sizes = []
    conv_rates = []
    final_losses = []

    for sname, smetrics in by_strategy.items():
        strategies.append(sname)
        ctx_sizes.append(smetrics['avg_context_size'])
        conv_rates.append(smetrics['convergence_rate'] * 100)
        final_losses.append(smetrics['avg_final_loss_log10'])

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(strategies)))
    scatter = ax.scatter(ctx_sizes, conv_rates, c=final_losses,
                        s=[cr * 3 for cr in conv_rates],
                        cmap='RdYlGn_r', alpha=0.8, edgecolors='black', linewidth=0.5)

    for i, s in enumerate(strategies):
        ax.annotate(s.split('(')[0][:15], (ctx_sizes[i], conv_rates[i]),
                   fontsize=7, ha='center', va='bottom',
                   xytext=(0, 8), textcoords='offset points')

    ax.set_xlabel('Average Context Size', fontsize=11)
    ax.set_ylabel('Convergence Rate (%)', fontsize=11)
    ax.set_title('Context Efficiency: Size vs Performance', fontweight='bold', fontsize=13)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log10(Final Loss)', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_ls2_efficiency.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig_ls2_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ fig_ls2_efficiency saved")


def plot_loss_curves(results: Dict[str, Any], output_dir: str):
    """图3: 代表性损失曲线对比。"""
    raw_results = results.get('results', [])

    # 按策略分组
    strategy_groups: Dict[str, List[Dict]] = {}
    for r in raw_results:
        base = r['strategy'].split('(')[0]
        if base not in strategy_groups:
            strategy_groups[base] = []
        strategy_groups[base].append(r)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {'Full-Context': '#e74c3c', 'SlidingWindow': '#3498db',
              'Methylation': '#2ecc71', 'Acetylation': '#f39c12',
              'EpiContext': '#9b59b6'}

    for sname, sresults in strategy_groups.items():
        if sname not in colors:
            continue

        # 取前5个结果的损失曲线平均
        max_len = min(len(r['loss_history']) for r in sresults[:5]) if sresults else 0
        if max_len == 0:
            continue

        avg_curve = np.zeros(max_len)
        for r in sresults[:5]:
            curve = np.array(r['loss_history'][:max_len])
            avg_curve += np.log10(np.maximum(curve, 1e-15))
        avg_curve /= min(5, len(sresults))

        ax.plot(range(max_len), avg_curve, color=colors[sname],
                linewidth=2, label=sname, alpha=0.9)

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('log10(Loss)', fontsize=11)
    ax.set_title('Average Loss Curves by Strategy (Rosenbrock, d=10)', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_ls3_loss_curves.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig_ls3_loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ fig_ls3_loss_curves saved")


def plot_ablation_heatmap(results: Dict[str, Any], output_dir: str):
    """图4: 消融热力图 - 策略×损失函数×维度。"""
    raw_results = results.get('results', [])

    # 提取EpiContext相关策略
    epi_variants = ['Full-Context', 'SlidingWindow', 'Methylation', 'Acetylation', 'EpiContext']
    loss_functions = sorted(set(r['loss_function'] for r in raw_results))

    # 构建矩阵
    data = {}
    for r in raw_results:
        base = r['strategy'].split('(')[0]
        if base not in epi_variants:
            continue
        key = (base, r['loss_function'])
        if key not in data:
            data[key] = []
        data[key].append(1 if r['converged'] else 0)

    # 计算收敛率矩阵
    matrix = np.zeros((len(epi_variants), len(loss_functions)))
    for i, variant in enumerate(epi_variants):
        for j, lf in enumerate(loss_functions):
            key = (variant, lf)
            if key in data and data[key]:
                matrix[i, j] = np.mean(data[key])

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(loss_functions)))
    ax.set_xticklabels(loss_functions, fontsize=10)
    ax.set_yticks(range(len(epi_variants)))
    ax.set_yticklabels(epi_variants, fontsize=10)
    ax.set_title('Convergence Rate: Strategy × Loss Function', fontweight='bold', fontsize=13)

    # 添加数值
    for i in range(len(epi_variants)):
        for j in range(len(loss_functions)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha='center', va='center',
                          color='white' if matrix[i, j] < 0.5 else 'black',
                          fontweight='bold', fontsize=10)

    plt.colorbar(im, ax=ax, label='Convergence Rate')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_ls4_ablation_heatmap.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig_ls4_ablation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ fig_ls4_ablation_heatmap saved")


def plot_fitness_evolution(results: Dict[str, Any], output_dir: str):
    """图5: 适应度随迭代演化。"""
    raw_results = results.get('results', [])

    # 只取EpiContext策略的结果
    epi_results = [r for r in raw_results if 'EpiContext' in r['strategy']]
    fc_results = [r for r in raw_results if r['strategy'] == 'Full-Context']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 迭代数分布
    ax1 = axes[0]
    epi_iters = [r['iterations'] for r in epi_results]
    fc_iters = [r['iterations'] for r in fc_results]

    ax1.hist(epi_iters, bins=30, alpha=0.6, label='EpiContext', color='#9b59b6', density=True)
    ax1.hist(fc_iters, bins=30, alpha=0.6, label='Full-Context', color='#e74c3c', density=True)
    ax1.set_xlabel('Iterations to Converge', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Iteration Distribution', fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.axvline(np.mean(epi_iters), color='#9b59b6', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(fc_iters), color='#e74c3c', linestyle='--', linewidth=2)

    # 最终损失分布
    ax2 = axes[1]
    epi_losses = [math.log10(max(r['final_loss'], 1e-15)) for r in epi_results]
    fc_losses = [math.log10(max(r['final_loss'], 1e-15)) for r in fc_results]

    ax2.hist(epi_losses, bins=30, alpha=0.6, label='EpiContext', color='#9b59b6', density=True)
    ax2.hist(fc_losses, bins=30, alpha=0.6, label='Full-Context', color='#e74c3c', density=True)
    ax2.set_xlabel('log10(Final Loss)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Final Loss Distribution', fontweight='bold')
    ax2.legend(fontsize=10)

    fig.suptitle('EpiContext vs Full-Context: Performance Distribution',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_ls5_distribution.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig_ls5_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ fig_ls5_distribution saved")


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'paper', 'mypaper', 'figures')

    os.makedirs(figures_dir, exist_ok=True)

    results_file = os.path.join(results_dir, 'large_scale_results.json')
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("Waiting for experiment to complete...")
        return

    print("Loading large-scale experiment results...")
    results = load_results(results_file)

    print("\nGenerating large-scale figures...")
    plot_convergence_comparison(results, figures_dir)
    plot_context_efficiency(results, figures_dir)
    plot_loss_curves(results, figures_dir)
    plot_ablation_heatmap(results, figures_dir)
    plot_fitness_evolution(results, figures_dir)

    print(f"\n✅ All large-scale figures saved to {figures_dir}")


if __name__ == '__main__':
    main()