# Stage D-F: 实验规划、设计、执行计划

## 阶段D: 实验规划

### 实验目标
1. 验证EpiContext在任务成功率上的提升
2. 验证EpiContext在Token效率上的优化
3. 验证各组件的必要性（消融实验）
4. 验证长程任务中的优势

### 实验设计

#### 实验1: 主实验
- **数据集**: WebArena, SWE-bench Lite, AgentBench, ALFWorld
- **基线**: ReAct, Reflexion, MemGPT, AutoTool, Full-Context
- **指标**: 成功率, Token消耗, 平均步数, 信息密度
- **重复**: 每个任务3次

#### 实验2: 消融实验
- **变体**:
  - EpiContext (完整)
  - w/o Methylation
  - w/o Acetylation
  - w/o Crossover
  - w/o Fitness
- **数据集**: WebArena + ALFWorld
- **指标**: 同上

#### 实验3: 长程任务分析
- **数据集**: ALFWorld
- **任务长度**: 10/20/50/100步
- **比较**: EpiContext vs ReAct vs MemGPT

#### 实验4: 工具数量敏感性
- **数据集**: WebArena
- **工具数量**: 5/10/20/50个
- **比较**: EpiContext vs AutoTool vs Full-Context

#### 实验5: 适应度函数分析
- **参数**: α, β, γ的不同组合
- **可视化**: 适应度随任务推进的变化

## 阶段E: 实验设计

### 环境配置
- Python 3.10+
- OpenAI API / Anthropic API
- 依赖: numpy, pandas, matplotlib, requests, tiktoken

### 沙箱环境
- 隔离的Python虚拟环境
- 依赖冻结: requirements.txt
- 版本控制: Git

### 硬件适配
- GPU: NVIDIA RTX 4060 (可选，主要用于可视化)
- CPU: Intel i9-13900H
- RAM: 32GB

## 阶段F: 实验执行

### 代码生成
- 使用Claude Code生成核心代码
- 模块化设计: context_graph.py, epigenetic_operators.py, fitness.py, agent.py
- 可复现性: 固定随机种子

### 迭代优化
- 检测NaN/Inf
- 修复运行时Bug
- 优化性能

### 数据记录
- 所有实验结果保存为JSON
- 生成可视化图表
- 记录运行时间和资源消耗
