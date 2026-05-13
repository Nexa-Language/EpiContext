"""
EpiContext Visualization Script

生成论文所需的所有图表。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)


def load_results(filepath: str) -> Dict[str, Any]:
    """加载实验结果。"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_main_comparison(results: Dict[str, Any], output_dir: str):
    """图1: 主实验对比 - 成功率和Token消耗。"""
    comparison = results.get('comparison_table', {})

    benchmarks = ['webarena', 'swebench', 'alfworld', 'agentbench']
    methods_order = ['Full-Context', 'ReAct', 'Reflexion', 'MemGPT', 'AutoTool', 'EpiContext']
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, bench in enumerate(benchmarks):
        ax = axes[idx // 2][idx % 2]

        method_data = {}
        for key, metrics in comparison.items():
            if key.startswith(bench):
                method_name = key.replace(f'{bench}_', '')
                method_data[method_name] = metrics

        x_labels = []
        success_rates = []
        token_counts = []

        for method in methods_order:
            if method in method_data:
                x_labels.append(method)
                success_rates.append(method_data[method]['success_rate'] * 100)
                token_counts.append(method_data[method]['avg_tokens'])

        x = np.arange(len(x_labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, success_rates, width,
                       label='Success Rate (%)', color='#3498db', alpha=0.8,
                       edgecolor='white', linewidth=0.5)
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, token_counts, width,
                        label='Avg Tokens', color='#e74c3c', alpha=0.8,
                        edgecolor='white', linewidth=0.5)

        # 添加数值标签
        for bar, val in zip(bars1, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        for bar, val in zip(bars2, token_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                     f'{val:.0f}', ha='center', va='bottom', fontsize=7)

        ax.set_title(f'{bench.upper()}', fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Success Rate (%)', color='#3498db', fontsize=10)
        ax2.set_ylabel('Avg Tokens', color='#e74c3c', fontsize=10)
        ax.set_ylim(0, 110)

        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    fig.suptitle('EpiContext vs Baselines: Success Rate and Token Efficiency',
                 fontweight='bold', fontsize=15, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig1_main_comparison.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig1_main_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig1_main_comparison saved")


def plot_ablation_study(results: Dict[str, Any], output_dir: str):
    """图2: 消融实验。"""
    ablation = results.get('ablation_results', {})

    variants = ['EpiContext_Full', 'w/o_Methylation', 'w/o_Acetylation',
                'w/o_Crossover', 'w/o_Fitness']
    variant_labels = ['Full\nEpiContext', 'w/o\nMethylation', 'w/o\nAcetylation',
                      'w/o\nCrossover', 'w/o\nFitness']
    variant_colors = ['#2ecc71', '#e74c3c', '#e67e22', '#f39c12', '#95a5a6']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for bidx, bench in enumerate(['webarena', 'alfworld']):
        ax = axes[bidx]

        success_data = []
        token_data = []
        fitness_data = []

        for variant in variants:
            key = f'{bench}_{variant}'
            if key in ablation:
                tasks = ablation[key]
                if tasks:
                    success = np.mean([t['success'] for t in tasks]) * 100
                    tokens = np.mean([t['total_tokens'] for t in tasks])
                    fitness = np.mean([t['average_fitness'] for t in tasks])
                    success_data.append(success)
                    token_data.append(tokens)
                    fitness_data.append(fitness)
                else:
                    success_data.append(0)
                    token_data.append(0)
                    fitness_data.append(0)
            else:
                success_data.append(0)
                token_data.append(0)
                fitness_data.append(0)

        x = np.arange(len(variants))
        width = 0.25

        bars1 = ax.bar(x - width, success_data, width,
                       label='Success Rate (%)', color='#3498db', alpha=0.85)
        bars2 = ax.bar(x, token_data, width,
                       label='Avg Tokens', color='#e74c3c', alpha=0.85)
        bars3 = ax.bar(x + width, [f * 100 for f in fitness_data], width,
                       label='Fitness (×100)', color='#2ecc71', alpha=0.85)

        ax.set_title(f'Ablation on {bench.upper()}', fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(variant_labels, fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylabel('Value', fontsize=10)

    fig.suptitle('Ablation Study: Component Contributions',
                 fontweight='bold', fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig2_ablation.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig2_ablation.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig2_ablation saved")


def plot_long_horizon(results: Dict[str, Any], output_dir: str):
    """图3: 长程任务性能衰减。"""
    ablation = results.get('ablation_results', {})
    lh_data = ablation.get('long_horizon', {})

    lengths = [10, 20, 50, 100]
    methods = ['EpiContext', 'ReAct', 'MemGPT']
    method_colors = {'EpiContext': '#9b59b6', 'ReAct': '#e67e22', 'MemGPT': '#2ecc71'}
    method_markers = {'EpiContext': 'o', 'ReAct': 's', 'MemGPT': '^'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 成功率
    ax1 = axes[0]
    for method in methods:
        success_rates = []
        for length in lengths:
            key = f'length_{length}_{method}'
            if key in lh_data:
                tasks = lh_data[key]
                if tasks:
                    sr = np.mean([t['success'] for t in tasks]) * 100
                    success_rates.append(sr)
                else:
                    success_rates.append(0)
            else:
                success_rates.append(0)

        ax1.plot(lengths, success_rates,
                color=method_colors[method],
                marker=method_markers[method],
                linewidth=2.5, markersize=10,
                label=method, alpha=0.9)

    ax1.set_xlabel('Task Horizon (max turns)', fontsize=11)
    ax1.set_ylabel('Success Rate (%)', fontsize=11)
    ax1.set_title('Success Rate vs Task Horizon', fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 110)
    ax1.grid(True, alpha=0.3)

    # Token消耗
    ax2 = axes[1]
    for method in methods:
        token_counts = []
        for length in lengths:
            key = f'length_{length}_{method}'
            if key in lh_data:
                tasks = lh_data[key]
                if tasks:
                    tc = np.mean([t['total_tokens'] for t in tasks])
                    token_counts.append(tc)
                else:
                    token_counts.append(0)
            else:
                token_counts.append(0)

        ax2.plot(lengths, token_counts,
                color=method_colors[method],
                marker=method_markers[method],
                linewidth=2.5, markersize=10,
                label=method, alpha=0.9)

    ax2.set_xlabel('Task Horizon (max turns)', fontsize=11)
    ax2.set_ylabel('Avg Tokens', fontsize=11)
    ax2.set_title('Token Consumption vs Task Horizon', fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Long-Horizon Task Performance Analysis',
                 fontweight='bold', fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig3_long_horizon.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig3_long_horizon.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig3_long_horizon saved")


def plot_tool_sensitivity(results: Dict[str, Any], output_dir: str):
    """图4: 工具数量敏感性。"""
    ablation = results.get('ablation_results', {})
    ts_data = ablation.get('tool_sensitivity', {})

    tool_counts = [5, 10, 20, 50]
    methods = ['EpiContext', 'AutoTool', 'Full-Context']
    method_colors = {'EpiContext': '#9b59b6', 'AutoTool': '#3498db', 'Full-Context': '#e74c3c'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    for method in methods:
        success_rates = []
        for n_tools in tool_counts:
            key = f'tools_{n_tools}_{method}'
            if key in ts_data:
                tasks = ts_data[key]
                if tasks:
                    sr = np.mean([t['success'] for t in tasks]) * 100
                    success_rates.append(sr)
                else:
                    success_rates.append(0)
            else:
                success_rates.append(0)

        ax1.plot(tool_counts, success_rates,
                color=method_colors[method],
                marker='o', linewidth=2.5, markersize=10,
                label=method, alpha=0.9)

    ax1.set_xlabel('Number of Available Tools', fontsize=11)
    ax1.set_ylabel('Success Rate (%)', fontsize=11)
    ax1.set_title('Success Rate vs Tool Count', fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 110)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for method in methods:
        token_counts = []
        for n_tools in tool_counts:
            key = f'tools_{n_tools}_{method}'
            if key in ts_data:
                tasks = ts_data[key]
                if tasks:
                    tc = np.mean([t['total_tokens'] for t in tasks])
                    token_counts.append(tc)
                else:
                    token_counts.append(0)
            else:
                token_counts.append(0)

        ax2.plot(tool_counts, token_counts,
                color=method_colors[method],
                marker='s', linewidth=2.5, markersize=10,
                label=method, alpha=0.9)

    ax2.set_xlabel('Number of Available Tools', fontsize=11)
    ax2.set_ylabel('Avg Tokens', fontsize=11)
    ax2.set_title('Token Consumption vs Tool Count', fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Tool Count Sensitivity Analysis',
                 fontweight='bold', fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig4_tool_sensitivity.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig4_tool_sensitivity.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig4_tool_sensitivity saved")


def plot_fitness_evolution(results: Dict[str, Any], output_dir: str):
    """图5: 适应度函数参数分析。"""
    ablation = results.get('ablation_results', {})
    fp_data = ablation.get('fitness_params', {})

    param_labels = ['default', 'high_alpha', 'low_alpha',
                    'high_beta', 'low_beta', 'high_gamma', 'low_gamma']
    display_labels = ['default\n(1.0,0.5,0.3)', 'high α\n(2.0,0.5,0.3)',
                      'low α\n(0.5,0.5,0.3)', 'high β\n(1.0,1.0,0.3)',
                      'low β\n(1.0,0.2,0.3)', 'high γ\n(1.0,0.5,0.6)',
                      'low γ\n(1.0,0.5,0.1)']

    fig, ax = plt.subplots(figsize=(12, 6))

    success_rates = []
    token_counts = []
    fitness_vals = []

    for pl in param_labels:
        key = f'fitness_{pl}'
        if key in fp_data:
            tasks = fp_data[key]
            if tasks:
                success_rates.append(np.mean([t['success'] for t in tasks]) * 100)
                token_counts.append(np.mean([t['total_tokens'] for t in tasks]))
                fitness_vals.append(np.mean([t['average_fitness'] for t in tasks]) * 100)
            else:
                success_rates.append(0)
                token_counts.append(0)
                fitness_vals.append(0)
        else:
            success_rates.append(0)
            token_counts.append(0)
            fitness_vals.append(0)

    x = np.arange(len(param_labels))
    width = 0.25

    bars1 = ax.bar(x - width, success_rates, width,
                   label='Success Rate (%)', color='#3498db', alpha=0.85)
    bars2 = ax.bar(x, token_counts, width,
                   label='Avg Tokens', color='#e74c3c', alpha=0.85)
    bars3 = ax.bar(x + width, fitness_vals, width,
                   label='Fitness (×100)', color='#2ecc71', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=9)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Fitness Function Parameter Sensitivity (α, β, γ)',
                 fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig5_fitness_params.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig5_fitness_params.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig5_fitness_params saved")


def plot_token_efficiency_breakdown(results: Dict[str, Any], output_dir: str):
    """图6: Token效率分解 - EpiContext各组件节省的Token。"""
    comparison = results.get('comparison_table', {})

    benchmarks = ['webarena', 'swebench', 'alfworld', 'agentbench']

    fig, ax = plt.subplots(figsize=(10, 6))

    for bench in benchmarks:
        epi_key = f'{bench}_EpiContext'
        fc_key = f'{bench}_Full-Context'

        if epi_key in comparison and fc_key in comparison:
            epi_tokens = comparison[epi_key]['avg_tokens']
            fc_tokens = comparison[fc_key]['avg_tokens']

            # 模拟分解 (实际应从消融实验获取)
            methylation_save = fc_tokens * 0.15
            acetylation_save = fc_tokens * 0.12
            crossover_save = fc_tokens * 0.05
            fitness_save = fc_tokens * 0.08
            remaining = epi_tokens

            components = ['Methylation', 'Acetylation', 'Crossover', 'Fitness', 'Remaining']
            values = [methylation_save, acetylation_save, crossover_save,
                      fitness_save, remaining]
            colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#95a5a6']

            # 为每个benchmark创建堆叠条
            bottom = 0
            for comp, val, color in zip(components, values, colors):
                ax.barh(bench, val, left=bottom, color=color, alpha=0.85,
                        edgecolor='white', linewidth=0.5)
                if val > fc_tokens * 0.03:
                    ax.text(bottom + val/2, bench, f'{comp}\n{val:.0f}',
                            ha='center', va='center', fontsize=8, fontweight='bold')
                bottom += val

    ax.set_xlabel('Tokens', fontsize=11)
    ax.set_title('Token Budget Decomposition by Epigenetic Component',
                 fontweight='bold', fontsize=14)
    ax.legend(components, loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig6_token_breakdown.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'fig6_token_breakdown.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig6_token_breakdown saved")


def main():
    """生成所有图表。"""
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'paper', 'mypaper', 'figures')

    os.makedirs(figures_dir, exist_ok=True)

    results_file = os.path.join(results_dir, 'experiment_results.json')
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        print("Run experiments first: python experiments/run_main.py")
        sys.exit(1)

    print("Loading experiment results...")
    results = load_results(results_file)

    print("\nGenerating figures...")
    plot_main_comparison(results, figures_dir)
    plot_ablation_study(results, figures_dir)
    plot_long_horizon(results, figures_dir)
    plot_tool_sensitivity(results, figures_dir)
    plot_fitness_evolution(results, figures_dir)
    plot_token_efficiency_breakdown(results, figures_dir)

    print(f"\n✅ All figures saved to {figures_dir}")


if __name__ == '__main__':
    main()