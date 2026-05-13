# Stage A2: RADICAL_IDEA_GEN 计划

## 目标
基于FRONTIER搜索结果和现有EpiContext构想，形成足够激进、有破坏性创新的Idea方向。

## 核心洞察
从搜索结果中提炼出的关键洞察：
1. 上下文管理正从"工程细节"上升为"核心研究问题"
2. 现有方法多为静态/半静态，缺乏动态调控
3. 工具选择和记忆管理被分开处理，缺乏联合优化
4. 表观遗传学概念在AI Agent领域完全空白

## EpiContext激进Idea

### 核心创新：表观遗传上下文路由器 (Epigenetic Context Router)

**激进点1: 生物隐喻的工程化实现**
- 将Agent的完整历史视为"基因组"（不变的全量知识库）
- 每次Request的上下文视为"表观遗传表达"（动态调控后的RNA）
- 引入"甲基化"（沉默噪声记忆）和"乙酰化"（激活关键工具）机制

**激进点2: 适应度驱动的动态进化**
- 定义严格的适应度函数 F(P) = α·R_task(P) - β·C_token(P) + γ·I_density(P)
- 基于任务成功率实时反馈调控上下文表达
- 实现"用进废退"的上下文进化

**激进点3: 交叉重组与并行探索**
- 遇到死锁时并行生成多个变异Request
- 选取最优路径遗传给下一代
- 实现Agent"性格"的演化（细胞分化隐喻）

## 论文档案 (Paper Card)

- **论文标题**: EpiContext: Epigenetic Context Evolution for Efficient Long-Horizon Agent Reasoning
- **目标会议**: COLM 2026 (Conference on Language Modeling)
- **核心贡献**:
  1. 首个基于表观遗传学隐喻的Agent上下文动态演化框架
  2. 适应度驱动的上下文选择算法，实现任务成功率与Token效率的联合优化
  3. 在多个长程任务基准上的实证验证，展示显著的性能提升和Token效率改进
