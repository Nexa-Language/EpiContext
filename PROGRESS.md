# EpiContext 论文流水线进度追踪 (v4.0 - 审计后重建)

## 项目信息
- **项目**: EpiContext - 基于表观遗传学的Agent上下文动态演化框架
- **目标会议**: COLM 2026
- **论文标题**: EpiContext: Epigenetic Context Evolution for Efficient Long-Horizon Agent Reasoning
- **当前PDF**: 17页, 编译通过 (COLM 2026模板)

## 版本历史

### v4.0 (2026-05-11, 完成)
**审计后重建**: 基于 Harbor 的真实 Agent 实验栈

#### 已废弃的旧证据 (v1-v3):
- `code/experiments/run_main.py` - 玩具级模拟环境
- `code/epicontext/benchmarks/environments.py` - 字符串匹配模拟
- `code/results/results_summary.txt` - 硬编码 token 倍数
- 论文中所有未支撑的声称

#### v4.0 改进内容:
1. **密钥安全**: 移除硬编码 API Key，改为环境变量
2. **Harbor 接入**: 安装并验证 Harbor framework
3. **EpiContext Harbor Agent**: 5 个 agent 变体 (EpiContext, FullContext, SlidingWindow, MethylationOnly, AcetylationOnly)
4. **真实实验**: 90 runs (5 agents × 3 tasks × 3 reps × 2), 91 分钟
5. **诚实论文**: 所有声称由真实实验数据支撑

### v3.0 (2026-05-11, 已废弃)
REFINE循环完成 (基于优化代理实验)

### v2.0 (2026-05-10, 已废弃)
大规模真实数学实验 (1040次运行)

### v1.0 (2026-05-09, 已废弃)
初始完整流水线 (模拟环境)

## 当前状态
- **论文**: 17页PDF, 编译通过
- **实验**: 90 Harbor runs 完成 (45 成功)
- **代码**: EpiContext Harbor Agent + 批量实验运行器
- **参考文献**: 38篇 (含 harbor2026)

## 关键实验结果 (诚实版)
- **所有策略在简单任务上表现相同** (10.0 turns) — 校准基线
- **EpiContext token 开销**: 18-22% (来自激活元数据)
- **SlidingWindow 最低 token**: 4,289 (但可能丢失早期上下文)
- **上下文策略仅在任务复杂度超过阈值时才有意义**

## 待完成
- [ ] 更复杂任务的实验 (Terminal-Bench 2.0, SWE-Bench)
- [ ] 终止条件改进
- [ ] 激活元数据压缩
- [ ] 第三方审查